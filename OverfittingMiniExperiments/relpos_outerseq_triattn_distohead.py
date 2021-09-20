import numpy as np
import matplotlib.pyplot as plt
import torch
from modules import makeLinear, TriangleMultiplication, TriangleAttention, Transition
import ml_collections
from tqdm import tqdm


# def _distogram_log_loss(logits, bin_edges, bin_centres, batch, num_bins):
#   """
#   DeepMind AlphaFold code https://github.com/deepmind/alphafold
#   ported to pytorch by Louis Robinson (21 Aug 2021).
  
#   Log loss of a distogram."""
#   assert len(logits.shape) == 3
#   positions = batch['pseudo_beta']
#   mask = batch['pseudo_beta_mask']
#   square_mask = mask[...,None,:] * mask[...,None]

#   assert positions.shape[-1] == 3

#   probs = torch.exp(logits)
#   probs = probs/probs.sum(-1)[...,None]
#   # mean_entropy = (probs * (probs < probs.max(-1)[0][...,None])).sum(-1).mean()
#   d_est = torch.einsum('ijk,k->ij', probs, bin_centres)

#   d2 = ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1)
#   d = (1e-6 + d2)**0.5

#   # clip!!!!!

#   mse = (torch.abs(d-d_est) * square_mask).mean()

#   true_bins = (d2[...,None] > bin_edges**2).sum(-1)

#   sh = true_bins.shape
#   smxe = torch.nn.CrossEntropyLoss(reduction='none')


#   bin_err = torch.abs(torch.arange(-num_bins, num_bins+1))
#   be = lambda i: bin_err[num_bins-i:-i-1]
#   all_bin_err = torch.stack([be(i) for i in range(num_bins)])

#   flat_logits = logits.reshape(-1, logits.size(-1))
#   flat_probs = probs.reshape(-1, probs.size(-1))
#   flat_true_bins = true_bins.reshape(-1).long()

#   manhtn_bin_dist = all_bin_err[flat_true_bins]
#   mean_entropy = (flat_probs * manhtn_bin_dist[None,...]).mean()
#   # mm = torch.functional.F.one_hot(logits.max(-1)[1], 64)

#   errors = smxe(flat_logits, flat_true_bins).reshape(sh)

#   avg_error = (
#       torch.sum(errors * square_mask, dim=(-2, -1)) /
#       (1e-6 + torch.sum(square_mask, dim=(-2, -1))))
#   return dict(loss=avg_error, mse=mse, true_dist=d, pred=d_est, 
#     entropy=mean_entropy, 
#     probs=probs, 
#     true_bins=true_bins)

def _distogram_log_loss(logits, bin_edges, bin_centres, batch, num_bins):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Log loss of a distogram."""
  assert len(logits.shape) == 3
  positions = batch['pseudo_beta']
  mask = batch['pseudo_beta_mask']

  assert positions.shape[-1] == 3

  # multiply logits by bins and compute mse with true distogram
  # diff = bins[1]-bins[0]
  # b0 = torch.tensor([bins[0] - diff], device=CONFIG.global_config.device)
  # bins = torch.cat((b0, torch.flip(bins, (0,))), dim=0) - diff*0.5
  probs = torch.exp(logits)
  probs = probs/probs.sum(-1)[...,None]
  mean_entropy = -((probs * torch.log(probs)).sum(-1)).mean()
  # print(probs.max(dim=-1))
  # mean_entropy = torch.exp(-probs.max(dim=-1)[0]).mean()
  # mean_entropy = (probs * (probs < probs.max(-1)[0][...,None])).sum(-1).mean()
  d_est = torch.einsum('ijk,k->ij', probs, bin_centres)
  d2 = ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1)
  d = (1e-6 + d2)**0.5

  dclamp = torch.clamp(d, min=bin_edges[0],max=bin_edges[-1])
  # the axis of logits corresponds to true_bins
  mse = (torch.abs(dclamp-d_est)).mean()

  # bin_edges = [2.3125, 2.6250,..., 21.3750, 21.6875]
  # sq_breaks = [5.3477, 6.8906,..., 456.8906, 470.3477]]
  # sq_breaks = linspace(first_break, last_break, num_bins - 1)^2
  true_bins = (d2[...,None] > bin_edges**2).sum(-1)
  # true bins is a [n, n] matrix of ints, 
  #   i = 0 is the largest bin (last_break < d) --> use length 30
  #   i = 62 is the second smallest bin (first_break < d) --> use length av(first_break, second_break)
  #   i = 63 is the smallest bin (d <= first_break) --> use length 1
  # the right-most dim of logits is over the true_bins

  sh = true_bins.shape
  smxe = torch.nn.CrossEntropyLoss(reduction='none')
  errors = smxe(logits.reshape(-1, logits.size(-1)), true_bins.reshape(-1).long()).reshape(sh)
  square_mask = mask[...,None,:] * mask[...,None]

  avg_error = (
      torch.sum(errors * square_mask, dim=(-2, -1)) /
      (1e-6 + torch.sum(square_mask, dim=(-2, -1))))
  return dict(loss=avg_error, mse=mse, true_dist=d, pred=d_est, 
    entropy=mean_entropy, 
    probs=probs,#torch.functional.F.one_hot(logits.max(-1)[1], 64), 
    true_bins=true_bins,
    target=dclamp,)


class DistogramHead(torch.nn.Module):
  """  
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['pair'].dtype
    init = 'zeros' if self.global_config.zero_init else 'linear'
    self.half_logits = makeLinear(representations['pair'].size(-1), 
      self.config.num_bins, dt, self.global_config.device, initializer=init)
    self.breaks = torch.linspace(self.config.first_break, self.config.last_break,
                      self.config.num_bins - 1, device=self.global_config.device)
    ends = torch.tensor([[self.config.last_break + 1], [1]], device=self.global_config.device)
    centre = 0.5 * (self.breaks[1] - self.breaks[0])
    b = torch.linspace(self.config.last_break, self.config.first_break,
                        self.config.num_bins - 1, device=self.global_config.device)
    self.bin_centres = torch.cat((ends[0], b[:-1]+centre, ends[1]), dim=0)

    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    """
    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = self.half_logits(representations['pair'])

    logits = half_logits + torch.swapaxes(half_logits, -2, -3)

    return dict(logits=logits, bin_edges=self.breaks)

  def loss(self, value, batch):
    dll = _distogram_log_loss(value['logits'], value['bin_edges'], self.bin_centres,
                               batch, self.config.num_bins)
    return dll

class DistogramHeadSingle(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['pair'].dtype
    init = 'zeros' if self.global_config.zero_init else 'linear'
    # self.half_dist = makeLinear(representations['pair'].size(-1), 
    #     1, dt, self.global_config.device, initializer=init)
    h = 64
    self.half_dist = torch.nn.Sequential(
      makeLinear(representations['pair'].size(-1), 
        h, dt, self.global_config.device, initializer=init),
      torch.nn.ReLU(),
      makeLinear(h, 1, dt, self.global_config.device, initializer=init),
    )
    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    half_d = self.half_dist(representations['pair'])
    # half_d = torch.max(0, half_d) + torch.exp(torch.min(0, half_d))
    pred_dist = half_d + torch.swapaxes(half_d, -2, -3)
    return pred_dist

  def loss(self, value, batch):
    # dll = _distogram_log_loss(value['logits'], value['bin_edges'], self.bin_centres,
    #                            batch, self.config.num_bins)
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']
    assert positions.shape[-1] == 3

    d2 = ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1)
    d = (1e-6 + d2)**0.5
    mask2d = mask[...,None,:] * mask[...,None]
    mse = (((d-value)**2) * mask2d).mean()
    
    return dict(loss=mse, true_dist=d, pred=value)



class Evo(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config

    self.tri_mult_out = TriangleMultiplication(config.tri_mult_out, global_config)
    self.tri_mult_in = TriangleMultiplication(config.tri_mult_in, global_config)
    self.tri_attn_str = TriangleAttention(config.tri_attn_str, global_config)
    self.tri_attn_end = TriangleAttention(config.tri_attn_end, global_config)
    self.pair_transition = Transition(config.pair_transition, global_config)

  def forward(self, x):
    z, mask2d = x
    z += self.tri_mult_out(z.clone(), mask2d)
    z += self.tri_mult_in(z.clone(), mask2d)
    z += self.tri_attn_str(z.clone(), mask2d)
    z += self.tri_attn_end(z.clone(), mask2d)
    z += self.pair_transition(z.clone(), mask2d)
    return (z, mask2d)

class Model(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.dh = DistogramHead(config.heads.distogram, global_config)

    cf = config.embeddings_and_evoformer

    args = (cf.pair_channel, torch.float, global_config.device)
    self.relpos = makeLinear(2 * cf.max_relative_feature + 1, *args)

    self.left_single = makeLinear(self.global_config.n_base, *args)
    self.right_single = makeLinear(self.global_config.n_base, *args)

    self.evo = torch.nn.Sequential(
      *[Evo(cf.evoformer, global_config) for _ in range(1)]
    )

  def forward(self, seq, mask, y_feat):
    n = len(seq)
    
    x = torch.functional.F.one_hot(seq.long(), num_classes=self.global_config.n_base).float()
    left, right = self.left_single(x), self.right_single(x)
    pair_activations = left[:, None] + right[None]

    pos = torch.arange(0, n, device=x.device)
    offset = pos[:, None] - pos[None, :]

    mrf = self.config.embeddings_and_evoformer.max_relative_feature
    rel_pos = torch.functional.F.one_hot(
      torch.clip(offset + mrf, min=0, max=2 * mrf).long(), 2 * mrf + 1
    ).to(left.dtype)

    pair_activations += self.relpos(rel_pos)

    mask2d = mask[:,None] * mask[None,:]
    # attn
    pair_activations,_ = self.evo((pair_activations, mask2d))

    value = self.dh(representations=dict(pair=pair_activations), batch=y_feat, is_training=True)
    return value
  
  def loss(self, value, y_feat):
    return self.dh.loss(value, y_feat)

CONFIG = ml_collections.ConfigDict({
  'embeddings_and_evoformer': {
    'evoformer': {
      'tri_attn_str': {
        'dropout_rate': .0,#0.25,
        'gating': True,
        'num_head': 4,
        'orientation': 'per_row',
        'shared_dropout': True
      },
      'tri_attn_end': {
        'dropout_rate': .0,#0.25,
        'gating': True,
        'num_head': 4,
        'orientation': 'per_column',
        'shared_dropout': True
      },
      'tri_mult_out': {
        'dropout_rate': .0,#0.25,
        'equation': 'ikc,jkc->ijc',
        'num_intermediate_channel': 128,
        'orientation': 'per_row',
        'shared_dropout': True
      },
      'tri_mult_in': {
        'dropout_rate': .0,#0.25,
        'equation': 'kjc,kic->ijc',
        'num_intermediate_channel': 128,
        'orientation': 'per_row',
        'shared_dropout': True
      },
      'pair_transition': {
        'dropout_rate': 0.0,
        'num_intermediate_factor': 4,
        'orientation': 'per_row',
        'shared_dropout': True
      }
    },
    'max_relative_feature': 32,
    'pair_channel': 128,
  },
  'global_config': {
    'zero_init': False,
    'device':'cpu',
    'n_base':4
  },
  'heads': {
    'distogram': {
      'first_break': 2.3125,
      'last_break': 21.6875,
      'num_bins': 64,
      'weight': 0.3
    },
  }
})

# config = Config(64, 2.0, 30.0, 32, 32,
#   )
# global_config = GConfig(False, 'cpu', 4)

# def construct_dist_from_logits(logits, bins):
#   diff = bins[1]-bins[0]
#   b0 = torch.tensor([bins[0] - diff], device=CONFIG.global_config.device)
#   bins = torch.cat((b0, torch.flip(bins, (0,))), dim=0) - diff*0.5
#   return torch.einsum('ijk,k->ij', logits, bins)

def plot(c):
  dist = ((c[...,None,:] - c[...,None,:,:])**2).sum(-1)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(*c.T, marker='o')
  plt.show()
  plt.imshow(dist)
  plt.show()
  config = CONFIG.heads.distogram
  bins = torch.linspace(config.first_break, config.last_break, CONFIG.global_config.n_bins-1)
  bixs = (dist[...,None] > bins**2).sum(-1)

  bb = torch.linspace(0, 10, 2*len(bins)+1)-5

  pp = torch.exp(-(bb**2)/0.2)
  pp /= pp.sum()

  logits = torch.zeros((len(s),len(s),len(bins)+1,))
  h = (len(bins)+1)//2
  for i in range(logits.shape[0]):
    for j in range(logits.shape[1]):
      centre = bixs[i,j] + h
      logits[i,j,:] = pp[centre-h:centre+h]
  logits /= logits.sum(-1)[...,None]

  d = construct_dist_from_logits(logits, bins)
  plt.imshow(d)
  plt.show()


def generate_example(n=100, a=1,b=6,c=3):
  n_base = 4
  s = torch.randint(0, n_base, size=(n,))
  cr = torch.zeros((n,3))
  t = torch.linspace(0, 1, len(cr))
  cr[:,0] = torch.sin(a*t) + 4
  cr[:,1] = 0.6*torch.cos(b*t)**2
  cr[:,2] = 0.1*torch.sin(c*t)
  cr *= 25
  y_feat = dict(
    pseudo_beta=cr, 
    pseudo_beta_mask=torch.ones(n)
  )
  return s, torch.ones(n), y_feat

s, m, y_feat = generate_example(n=37)
model = Model(CONFIG, CONFIG.global_config)
_ = model(s, m, y_feat)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_step(opt, model, x, y, 
  main_loss_weight=0.0, mse_weight=0.1, entropy_weight=0.02):
  yh = model(*x, y)
  l = model.loss(yh, y)
  info = (l['loss'].item()*main_loss_weight, 
    l['mse'].item() * mse_weight, entropy_weight*l['entropy'].item())
  L = main_loss_weight*l['loss'] + l['mse'] * mse_weight + l['entropy'] * entropy_weight
  opt.zero_grad()
  L.backward()
  opt.step()
  return L.item(), l['pred'].data, l['target'].data, l['probs'].data, l['true_bins'].data, info, #yh['logits'].data, yh['bin_edges'].data, 

# def train_step(opt, model, x, y):
#   yh = model(*x, y)
#   l = model.loss(yh, y)
#   opt.zero_grad()
#   l['loss'].backward()
#   opt.step()
#   return l['loss'].item(), l['pred'].data, l['true_dist'].data, None, (None,)#yh['logits'].data, yh['bin_edges'].data, 

import json
from constants import RNA_const
b = RNA_const.basis[0]
data = json.loads(open('minidata.json','r').read())
base2i = {base:i for i, base in enumerate('ACGUX')}
toseq = lambda ss: torch.tensor([base2i[s] for s in ss])
xs = [(toseq(v['sequence']), torch.tensor(v['atom_mask'])[:,b], {'pseudo_beta':torch.tensor(v['atom_positions'])[:,b], 'pseudo_beta_mask':torch.tensor(v['atom_mask'])[:,b]}) for k,v in data.items()]
xs = [(s[:40],m[:40],{k:v[:40] for k,v in y.items()}) for s,m,y in xs]
# xs = [generate_example(n,a,b,c) for n,a,b,c in [(37,0.5,2,4), (19,0.1,0.4,5),(43,5,0.3,1.9)]]
s,m,y_feat = xs[2]

# opt.param_groups[0]['lr'] = 5e-3
ds = {}
pbar = tqdm(range(1, 4000+1))
for i in pbar:
  # s,m,y_feat = xs[np.random.randint(len(xs))]
  l, pred, true, pr, tb, info = train_step(opt, model, (s, m), y_feat)
  deets = (l, opt.param_groups[0]['lr'])+info
  pbar.set_description('L: %.3f, lr: %s, dm: %.3f, mse: %.5f, entr: %.5f, '%deets)
  # if i%5==0: print(deets)
  if i%50==0:
    ds[i] = (l, pred, true, pr, tb, info)
    # d = construct_dist_from_logits(lg, be)
    w = 2 + 2*int(pr is not None)
    fig, axes = plt.subplots(1, w, figsize=(10, 10*w))
    for i,da in enumerate([pred, true]):
      axes[i].imshow(da)
      axes[i].set_title('%.3f, %.3f'%(da.min(), da.max()))
    if pr is not None:
      idx = 9
      axes[-2].imshow(torch.flip(pr[idx], (-1, )), norm=None, vmin=0, vmax=1)
      # axes[-2].imshow(pr[idx], norm=None, vmin=0, vmax=1)
      axes[-2].set_xlabel('bins')
      axes[-2].set_ylabel('j')
      axes[-2].set_title('$\mathbb{P}(b|i=%d, j=j)$'%(idx+1, ))
      axes[-1].imshow(torch.functional.F.one_hot(tb[idx], 64))
      axes[-1].set_xlabel('bins')
      axes[-1].set_ylabel('j')
      axes[-1].set_title('$\mathbb{I}(b|i=%d, j=j)$'%(idx+1, ))
    plt.tight_layout()
    plt.show()
  opt.param_groups[0]['lr'] = 4e-5 * (np.exp(-i/80)+0.01)#4e-5 * np.cos(i/20)**2 + 4e-6

