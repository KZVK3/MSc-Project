import torch, numpy as np
nn = torch.nn
# import collections, numbers, functools, tree
import pytorch_lightning as pl
import os, json
from modules import EmbeddingsAndEvoformer, makeLinear, DistogramHead, StructureModule, gdt_ts, apx_lddt, TMLowerBound

import matplotlib.pylab as plt
from IPython import display

'''
this contains two distogram loss functions.
'''

def dyplot(fig, axs, ims, hdisplay, data):
  # for t,ax,im in zip(['Pred Evo','True Evo','Pred Final','True Final'],axs,ims):
  #   im.set_data(data[t])
  #   ax.set_title('%s, min: %.0f, max: %.0f'%(t,data[t].min(), data[t].max()))
  for t,ax,im in zip(['Pred Evo','True Evo'],axs,ims):
    im.set_data(data[t])
    ax.set_title('%s, min: %.0f, max: %.0f'%(t,data[t].min(), data[t].max()))

  for t,ax,im in zip(['Pred Final','True Final'],axs[2:],ims[2:]):
    ax.cla()
    ax.imshow(data[t])
    ax.axis('off')
    ax.set_title('%s, min: %.0f, max: %.0f'%(t,data[t].min(), data[t].max()))
  
  axs[-1].cla()
  axs[-1].set_title('lr: %s, RMSD: %.3f'%(str(data['lr']), data['RMSE']))
  axs[-1].plot(*data['true coords'].T, '-o', lw=0.3, label='Target')
  axs[-1].plot(*data['pred coords'].T, '-^', lw=0.3, label='Predicted')
  axs[-1].legend(loc='lower right')
  axs[-1].axis('off')
  
  hdisplay.update(fig)

def init_plot(vmin, vmax):
  get_im = lambda :np.random.rand(50,50) + vmin

  fig = plt.figure()
  fig.set_figheight(18)
  fig.set_figwidth(24)

  gr_sh = (3,4)
  axs = [plt.subplot2grid(gr_sh,s) for s in [(0,0),(1,0),(0,3),(1,3)]]
  ax = plt.subplot2grid(gr_sh, (0, 1), rowspan=2, colspan=2, projection='3d')
  ax.plot(*np.random.randn(3,100), '-o', lw=0.3, label='Target')
  ax.plot(*np.random.randn(3,100), '-^', lw=0.3, label='Prediction')
  ax.legend()

  kw = {'vmin':vmin, 'vmax':vmax, 'norm':None}

  hdisplay = display.display("", display_id=True)
  ims = [a.imshow(get_im(), **(kw if i<2 else {})) for i,a in enumerate(axs)]
  axs = axs + [ax]
  _ = [a.axis('off') for a in axs]

  return fig, axs+[ax], ims, hdisplay

def alignCoordsRMSE(X1, X2):
  ''' applys a translation to align centroids of X1 and X2, then find rotation '''
  X1, X2 = X1 - X1.mean(0)[None,:], X2 - X2.mean(0)[None,:]
  H = X1.T @ X2
  U, _, Vh = torch.linalg.svd(H, full_matrices=True)

  R = (U @ Vh).T
  X2 = X2 @ R
  rmse = ((X1-X2)**2).sum(1).mean()**0.5
  return X1, X2, rmse, R


class DistogramPostsHead(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations):
    dt = representations['pair'].dtype
    self.min_diff = torch.tensor([0.01], device=self.global_config.device)
    init = 'zeros' if self.global_config.zero_init else 'linear'

    self.post_logits = makeLinear(representations['pair'].size(-1), 
      2, dt, self.global_config.device, initializer=init)
    self.sigmoid = torch.nn.Sigmoid() 

    self.forward = self.go
    return self(representations)

  def go(self, representations):
    half_logits = self.sigmoid(self.post_logits(representations['pair']))
    logits = half_logits + torch.swapaxes(half_logits, -2, -3)
    return logits

  def loss(self, logits, batch, reweight=True):
    assert len(logits.shape) == 3
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']
    square_mask = mask[...,None,:] * mask[...,None]
    assert positions.shape[-1] == 3

    p = torch.exp(logits)
    p = p / p.sum(-1)[...,None]
    # linearly interpolate the min and max dist
    d_est = p[...,0] * self.config.first_break + p[...,1] * self.config.last_break

    d = (1e-6 + ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1))**0.5

    # most are above the last break, weight the lower ones more
    d_less = d < self.config.last_break

    if reweight:
      n_below = d_less.sum()
      n_total = torch.numel(d)
      w_min = n_below/n_total
      w_max = 1 - w_min
      diff = torch.max(w_max, self.min_diff)
      d_weight = w_min + diff * d_less
    else:
      d_weight = 1

    d = torch.clamp(d, self.config.first_break, self.config.last_break)
    
    mse = (((d-d_est)**2) * square_mask * d_weight).mean()
    return dict(loss=mse, pred=d_est, target=d)
  


class BaseTrunk(pl.LightningModule):
  ''' 
  alphafold:
  alphafolditeration:
    embeddings_and_evoformer:
      relpos
      seq_emb
      msa_emb
      (prev_emb)
      evoformer_iteration (48): {
        msa_row_attn_p_b
        col_glob_attnt
        msa_transition
        outer_prod_mean
        tri_mult_out
        tri_mult_in
        tri_attn_str
        tri_attn_end
        pair_transition
      }
    distogram_head
    (msa_head)
  '''
  def __init__(self, config, loss_weights):
    super().__init__()
    self.config = config['model']
    self.global_config = config['model'].global_config
    # self.data_config = config['data']
    self.c_msa = self.config.embeddings_and_evoformer.msa_channel
    self.c_pair = self.config.embeddings_and_evoformer.pair_channel

    self.trunk = EmbeddingsAndEvoformer(self.config.embeddings_and_evoformer, self.global_config)
    self.distogram_head = DistogramPostsHead(self.config.heads.distogram, self.global_config)
    # self.original_distogram_head = DistogramHead(self.config.heads.distogram, self.global_config)
    self.structure_module = StructureModule(self.config.heads.structure_module, self.global_config, compute_loss=True)
    self.w1 = loss_weights['delta_disto']
    # self.w2 = loss_weights['original_disto']
    self.w3 = loss_weights['structure_module']
  
  def forward(self, total_batch):
    # note! the recycle embedding layers do not exist yet, 
    # will need to use augment init
    # for b in total_batch:
    #   print((b, total_batch[b].shape))
    # print()

    n = total_batch['residue_index'].shape[-1]
    R = total_batch['num_iter_recycling'][0].item()

    ret = {'final_frame_pos':torch.zeros(n, 3, device=self.global_config.device)}
    representations = {
      'msa_first_row':torch.zeros(n, self.c_msa, device=self.global_config.device),
      'pair':torch.zeros(n, n, self.c_pair, device=self.global_config.device)
    }

    def get(i, ret, representations):
      batch = {k:t[i] for k,t in total_batch.items()}
      batch['prev_pos'] = ret['final_frame_pos'].detach()
      batch['prev_msa_first_row'] = representations['msa_first_row'].detach()
      batch['prev_pair'] = representations['pair'].detach()
      return batch

    with torch.no_grad():
      for r in range(R-1):
        batch = get(r, ret, representations)
        representations = self.trunk(batch, True)
        ret = self.structure_module( representations, batch, True )

    # compute gradients and losses
    batch = get(R-1, ret, representations)
    representations = self.trunk(batch, True)
    ret = self.structure_module( representations, batch, True )

    out3 = self.structure_module.loss(ret, batch)

    logits1 = self.distogram_head( representations )
    out1 = self.distogram_head.loss(logits1, batch)

    # logits2 = self.original_distogram_head( representations, batch, True )
    # out2 = self.original_distogram_head.loss(logits2, batch)

    return out1, None, ret, out3

  def track_codes(self, track, wait_time=100, plot_every=50):
    self.wait_time = wait_time
    self.plot_every = plot_every
    self.v_track = {c:[] for c in track['val_coords_to_track']}
    self.t_track = {c:[] for c in track['train_coords_to_track']}
    self.dircts = {'val_disto': 0, 'train_disto': 0, 'train_loss': 0, 
      'val_loss': 0, 'train_coords':0, 'val_coords':0, 'test_loss': 0, 
      'train_store':0}
    self.epo = 0
    self.time = 0
    self.train_list = []
    dc = self.config.heads.distogram
    self.plot_args = init_plot(dc.first_break, dc.last_break)
    self.train_store = []
  
  def save_data(self, key, data, js=True, code=''):
    if key is None or data is None or self.trainer.log_dir is None: return
    dr = self.trainer.log_dir + '/' + key
    if not os.path.isdir(dr): os.mkdir(dr)

    curr_ix = self.dircts[key]
    path, path2 = '%s/%s/'%(self.trainer.log_dir, key), 'epo-%d-%d'%(self.epo, curr_ix)
    if js:
      f = open(path+path2+'.json', 'w')
      f.write(json.dumps(data))
      f.close()
    else:
      if 'coords' in key:
        nam = path+code
      else:
        nam = path+path2+'-'+code
      np.save(nam, data)
    self.dircts[key] = curr_ix + 1
  
  def organise(self, train_batch, outs):
    mask = train_batch['backbone_affine_mask'].data[0]
    pr_crds = outs[-1]['final_affines'].data[mask==1, -3:]
    tr_crds = train_batch['backbone_affine_tensor'].data[0,mask==1, -3:]# the first dim is a repeat dim
    
    tr_crds, pr_crds, rmse, _ = alignCoordsRMSE(tr_crds, pr_crds)
    pr_d = (((pr_crds[:,None,:] - pr_crds[None,:,:])**2).sum(-1) + 1e-6)**0.5
    tr_d = (((tr_crds[:,None,:] - tr_crds[None,:,:])**2).sum(-1) + 1e-6)**0.5
    
    # (torch.Size([1, 80]), torch.Size([1, 80]), torch.Size([80, 7]), torch.Size([80, 7]))
    # print((pr_d.shape, tr_d.shape, pr_crds.shape, tr_crds.shape))

    return {
      'Pred Evo': outs[0]['pred'].data.cpu().numpy(), 
      'True Evo': outs[0]['target'].data.cpu().numpy(), 
      'Pred Final': pr_d.cpu().numpy(), 
      'True Final': tr_d.cpu().numpy(),
      'true coords':tr_crds.cpu().numpy(),
      'pred coords':pr_crds.cpu().numpy(),
      'pred affine':outs[-1]['final_affines'].data[mask==1].cpu().numpy(),
      'true affine':train_batch['backbone_affine_tensor'].data[0,mask==1].cpu().numpy(),
      'RMSE': rmse,
      'lr':self.lr_schedulers().get_last_lr(),
      'l':[o['loss'].item() for o in outs[:-1]]
    }

  def training_step(self, train_batch, batch_idx):
    # pass in the chain-code
    code = train_batch['code']
    del train_batch['code']
    out1, _, ret, out3 = self.forward(train_batch)
    loss = self.w1 * out1['loss'] + self.w3 * out3['loss']

    if batch_idx%self.plot_every==0:
      dyplot(*self.plot_args, self.organise(train_batch, [out1, out3, ret]))

    if code in self.t_track:
      tm, rmsd, gdt, lddt = self.get_metrics(train_batch, ret)
      self.train_store.append(
        (code, self.epo, batch_idx, loss.item(), tm, rmsd, gdt, lddt, out1['loss'].item(), out3['loss'].item())
      )

      if self.time > self.wait_time:
        self.save_data('train_disto', out1['pred'].data.cpu().numpy(), False, code)

        data = self.organise(train_batch, [out1, out3, ret])
        # coords = np.stack([data['pred coords'], data['true coords']])
        coords = np.stack([data['pred affine'], data['true affine']])
        self.save_data('train_coords', coords, False, code)

        self.time = 0
    self.time += 1

    if self.recall is not None:
      # batch_idx resets on each epoch
      if batch_idx//self.recall==0:
        # call scheduler
        prop = float(batch_idx)/self.num_data_per_epoch
        t = self.epo + prop
        self.lr_schedulers().step(t)

    self.log('train_loss', loss)
    self.train_list.append((code, loss.item()))
    return loss

  def training_epoch_end(self, _):
    self.save_data('train_loss', self.train_list)
    self.save_data('train_store', self.train_store)
    self.train_store = []
    self.train_list = []
    self.epo += 1

  def validation_step(self, val_batch, batch_idx):
    code = val_batch['code']
    del val_batch['code']
    out1, _, ret, out3 = self.forward(val_batch)
    l1, l3 = out1['loss'], out3['loss']
    loss = self.w1 * l1 + self.w3 * l3

    if code in self.v_track:
      self.save_data('val_disto', out1['pred'].data.cpu().numpy(), False, code)

      data = self.organise(val_batch, [out1, out3, ret])
      # coords = np.stack([data['pred coords'], data['true coords']])
      coords = np.stack([data['pred affine'], data['true affine']])
      self.save_data('val_coords', coords, False, code)

    tm, rmsd, gdt, lddt = self.get_metrics(val_batch, ret)

    self.log('val_loss', loss)
    return code, loss.item(), tm, rmsd, gdt, lddt, l1.item(), l3.item()
  
  def validation_epoch_end(self, val_outs):
    self.save_data('val_loss', val_outs)
  
  def get_metrics(self, batch, struct_out):
    return calculate_metrics(
      batch['backbone_affine_tensor'].data[0], 
      struct_out['final_affines'].data, 
      batch['backbone_affine_mask'].data[0], 
      self.global_config.device
    )
  
  def test_step(self, val_batch, batch_idx):
    code = val_batch['code']
    del val_batch['code']
    out1, _, ret, out3 = self.forward(val_batch)
    l1, l3 = out1['loss'], out3['loss']
    loss = self.w1 * l1 + self.w3 * l3
    tm, rmsd, gdt, lddt = self.get_metrics(val_batch, ret)
    self.log('test_loss', loss)
    return code, loss.item(), tm, rmsd, gdt, lddt, l1.item(), l3.item()
  
  def test_epoch_end(self, test_outs):
    self.save_data('test_loss', test_outs)
  
  def set_optim_config(self, cfg, n_epoch=None, num_data_per_epoch=None):
    opt_gr = cfg['optim_groups']
    self.optim_type = cfg['optim_type']
    self.groups = {k:[] for k in opt_gr}
    for n,p in self.named_parameters():
      added = False
      for keyword in opt_gr:
        if keyword in n:
          self.groups[keyword].append(p)
          added = True
      if not added:
        self.groups['default'].append(p)
    self.optim_groups = [{'params':self.groups[k], **v} for k,v in opt_gr.items() if len(self.groups[k])]

    self.num_data_per_epoch = num_data_per_epoch

    self.recall = None
    if 'scheduler' in cfg:
      print('scheduler found')
      if 'num_call_per_epoch' in cfg['scheduler']:
        n = cfg['scheduler']['num_call_per_epoch']
        self.recall = int( float(num_data_per_epoch) / float(n) )
        print('scheduler calling at '+str(self.recall))
      else:
        print('scheduler calling every epoch')

      sch = eval(cfg['scheduler']['class'])
      lmb_kw = eval(cfg['scheduler']['kwargs'])
      self.scheduler = (sch, lmb_kw)
    else:
      print('no scheduler found')
      self.scheduler = None

  def configure_optimizers(self, n_epoch=None):
    opt = eval('torch.optim.%s'%self.optim_type)
    optimizer = opt(self.optim_groups)

    if self.scheduler is None:
      return optimizer

    sch, lmb_kw = self.scheduler
    return [optimizer], [sch(optimizer, **lmb_kw)]

def calculate_metrics(true_affine, pred_affine, mask, device):
  true_coords = true_affine[:,-3:]
  pred_coords = pred_affine[:,-3:]
  
  tm = TMLowerBound(pred_affine, true_affine, mask, device)
  _, _, rmsd, _ = alignCoordsRMSE(true_coords[mask==1], pred_coords[mask==1])
  gdt = gdt_ts(pred_coords[None,...], true_coords[None,...], mask[None,...,None], device=device)
  lddt = apx_lddt(pred_coords[None,...], true_coords[None,...], mask[None,...,None], device=device)
  return tm.item(), rmsd.item(), gdt.item(), lddt.item()