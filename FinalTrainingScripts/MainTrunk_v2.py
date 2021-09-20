import torch, numpy as np
nn = torch.nn
# import collections, numbers, functools, tree
import pytorch_lightning as pl
import os, json
from modules import EmbeddingsAndEvoformer, makeLinear, DistogramHead#, TriangleMultiplication, TriangleAttention, Transition

import matplotlib.pylab as plt
from IPython import display

'''
this contains two distogram loss functions.
'''

def dyplot(fig, axs, ims, hdisplay, pred, true, lr, l1, l2):
  ims[0].set_data(pred.cpu())
  axs[0].set_title('lr: %s, min: %.2f, max: %.2f'%(str(lr), pred.min(), pred.max()))

  ims[1].set_data(true.cpu())
  axs[1].set_title('l1: %.5f, l2: %.5f, min: %.2f, max: %.2f'%(l1, l2, true.min(), true.max()))

  hdisplay.update(fig)

def init_plot(vmin, vmax):
  fig, axs = plt.subplots(1, 2, figsize=(10, 10*2))
  hdisplay = display.display("", display_id=True)
  ims = [axs[0].imshow(np.random.rand(50,50)+vmin, vmin=vmin, vmax=vmax, norm=None),
         axs[1].imshow(np.random.rand(50,50)+vmin, vmin=vmin, vmax=vmax, norm=None)]
  axs[0].axis('off')
  axs[1].axis('off')
  return fig, axs, ims, hdisplay

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
    self.trunk = EmbeddingsAndEvoformer(self.config.embeddings_and_evoformer, self.global_config)
    self.distogram_head = DistogramPostsHead(self.config.heads.distogram, self.global_config)
    self.original_distogram_head = DistogramHead(self.config.heads.distogram, self.global_config)
    self.w1 = loss_weights['delta_disto']
    self.w2 = loss_weights['original_disto']
  
  def forward(self, batch):
    representations = self.trunk(batch, True)

    logits1 = self.distogram_head( representations )
    out1 = self.distogram_head.loss(logits1, batch)

    logits2 = self.original_distogram_head( representations, batch, True )
    out2 = self.original_distogram_head.loss(logits2, batch)
    return out1, out2

  def track_codes(self, track, wait_time=100, plot_every=50):
    self.wait_time = wait_time
    self.plot_every = plot_every
    self.v_track = {c:[] for c in track['val_coords_to_track']}
    self.t_track = {c:[] for c in track['train_coords_to_track']}
    self.dircts = {'val_disto': 0, 'train_disto': 0, 'train_loss': 0, 'val_loss': 0}
    self.epo = 0
    self.time = 0
    self.train_list = []
    dc = self.config.heads.distogram
    self.plot_args = init_plot(dc.first_break, dc.last_break)
  
  def save_data(self, key, data, js=True, code=''):
    dr = self.trainer.log_dir + '/' + key
    if not os.path.isdir(dr): os.mkdir(dr)

    curr_ix = self.dircts[key]
    path = '%s/%s/epo-%d-%d'%(self.trainer.log_dir, key, self.epo, curr_ix)
    if js:
      f = open(path+'.json', 'w')
      f.write(json.dumps(data))
      f.close()
    else:
      np.save(path+'-'+code, data)
    self.dircts[key] = curr_ix + 1

  def training_step(self, train_batch, batch_idx):
    # pass in the chain-code
    code = train_batch['code']
    out1, out2 = self.forward(train_batch)
    l1, l2 = out1['loss'], out2['loss']
    loss = self.w1 * l1 + self.w2 * l2

    if batch_idx%self.plot_every==0:
      dargs = (out1['pred'].data, out1['target'].data, 
        self.lr_schedulers().get_last_lr(), l1.item(), l2.item())
      dyplot(*self.plot_args, *dargs)

    if self.time > self.wait_time and code in self.t_track:
      self.save_data('train_disto', out1['pred'].data.cpu().numpy(), False, code)
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
    self.train_list = []
    self.epo += 1

  def validation_step(self, val_batch, batch_idx):
    code = val_batch['code']
    out1, out2 = self.forward(val_batch)
    l1, l2 = out1['loss'], out2['loss']
    loss = self.w1 * l1 + self.w2 * l2

    if code in self.v_track:
      self.save_data('val_disto', out1['pred'].data.cpu().numpy(), False, code)

    self.log('val_loss', loss)
    return code, loss.item()
  
  def validation_epoch_end(self, val_outs):
    self.save_data('val_loss', val_outs)
  
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
