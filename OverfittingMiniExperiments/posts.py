import numpy as np
import matplotlib.pyplot as plt
import torch
from modules import makeLinear, TriangleMultiplication, TriangleAttention, Transition
import ml_collections
from tqdm import tqdm
import json
from constants import RNA_const

class DistogramPostsHead(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['pair'].dtype
    self.min_diff = torch.tensor([0.01])
    init = 'zeros' if self.global_config.zero_init else 'linear'

    self.post_logits = makeLinear(representations['pair'].size(-1), 
      2, dt, self.global_config.device, initializer=init)
    self.sigmoid = torch.nn.Sigmoid() 

    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    half_logits = self.sigmoid(self.post_logits(representations['pair']))
    logits = half_logits + torch.swapaxes(half_logits, -2, -3)
    return logits

  def loss(self, value, batch, reweight=True):
    assert len(value.shape) == 3
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']
    square_mask = mask[...,None,:] * mask[...,None]
    assert positions.shape[-1] == 3

    p = torch.exp(value)
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

# class DistogramHead(torch.nn.Module):
#   def __init__(self, config, global_config):
#     super().__init__()
#     self.config, self.global_config = config, global_config
#     self.forward = self.init_parameters

#   def init_parameters(self, representations, batch, is_training):
#     dt = representations['pair'].dtype
#     self.min_diff = torch.tensor([0.01])
#     init = 'zeros' if self.global_config.zero_init else 'linear'

#     self.post_logits = makeLinear(representations['pair'].size(-1), 
#       1, dt, self.global_config.device, initializer=init)
#     self.sigmoid = torch.nn.Sigmoid() 

#     self.forward = self.go
#     return self(representations, batch, is_training)

#   def go(self, representations, batch, is_training):
#     half_logits = self.sigmoid(self.post_logits(representations['pair']))
#     logits = half_logits + torch.swapaxes(half_logits, -2, -3)
#     return logits * 0.5

#   def loss(self, value, batch, reweight=True):
#     assert len(value.shape) == 3
#     positions = batch['pseudo_beta']
#     mask = batch['pseudo_beta_mask']
#     square_mask = mask[...,None,:] * mask[...,None]
#     assert positions.shape[-1] == 3

#     p = value
#     # linearly interpolate the min and max dist
#     d_est = p * self.config.first_break + (1 - p) * self.config.last_break
#     d = (1e-6 + ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1))**0.5

#     if reweight:
#       # most are above the last break, weight the lower ones more
#       d_less = d < self.config.last_break
#       n_below = d_less.sum()
#       n_total = torch.numel(d)
#       # print((n_below, n_total, n_below/n_total))
#       w_min = n_below/n_total
#       w_max = 1 - w_min
#       diff = torch.max(w_max, self.min_diff)
#       # print((w_min.item(), w_max.item()))
#       d_weight = w_min + diff * d_less
#       # d_weight = 0.1 + d_less
#     else:
#       d_weight = 1

#     d = torch.clamp(d, self.config.first_break, self.config.last_break)
    
#     p_target = (self.config.last_break - d) / (self.config.last_break-self.config.first_break)
#     mse = (square_mask * d_weight * torch.abs(p_target - p)).mean()

#     # mse = (((d-d_est)**2) * square_mask * d_weight).mean()

#     return dict(loss=mse, pred=d_est, target=d)


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
    z = z + self.tri_mult_out(z, mask2d)
    z = z + self.tri_mult_in(z, mask2d)
    z = z + self.tri_attn_str(z, mask2d)
    z = z + self.tri_attn_end(z, mask2d)
    z = z + self.pair_transition(z, mask2d)
    return (z, mask2d)

class Model(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.dh = DistogramPostsHead(config.heads.distogram, global_config)

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
  
  def loss(self, value, y_feat, reweight):
    return self.dh.loss(value, y_feat, reweight)

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

def train_step(opt, model, x, y, reweight=True):
  yh = model(*x, y)
  l = model.loss(yh, y, reweight)
  L = l['loss']
  opt.zero_grad()
  L.backward()
  opt.step()
  return L.item(), l['pred'].data, l['target'].data




b = RNA_const.basis[0]
data = json.loads(open('minidata.json','r').read())
base2i = {base:i for i, base in enumerate('ACGUX')}
toseq = lambda ss: torch.tensor([base2i[s] for s in ss])
xs = [(toseq(v['sequence']), torch.tensor(v['atom_mask'])[:,b], 
  {'pseudo_beta':torch.tensor(v['atom_positions'])[:,b], 
  'pseudo_beta_mask':torch.tensor(v['atom_mask'])[:,b]}) for k,v in data.items()]
# cp = 110
# xs = [(s[:cp],m[:cp],{k:v[:cp] for k,v in y.items()}) for s,m,y in xs]
# xs = [generate_example(n,a,b,c) for n,a,b,c in [(37,0.5,2,4), (19,0.1,0.4,5),(43,5,0.3,1.9)]]

s, m, y_feat = xs[2]
model = Model(CONFIG, CONFIG.global_config)
_ = model(s, m, y_feat)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

ds = {}
reweight = False
pbar = tqdm(range(2001, 4000+1))
for i in pbar:
  reweight = i < 3600
  # s,m,y_feat = xs[np.random.randint(len(xs))]
  l, pred, true = train_step(opt, model, (s, m), y_feat, reweight)
  deets = (l, opt.param_groups[0]['lr'])
  pbar.set_description('L: %.3f, lr: %s'%deets)
  # if i%5==0: print(deets)
  if i%100==0:
    ds[i] = pred
    # d = construct_dist_from_logits(lg, be)
    w = 2
    fig, axes = plt.subplots(1, w, figsize=(10, 10*w))
    for i,da in enumerate([pred, true]):
      axes[i].imshow(da)
      axes[i].set_title('%.3f, %.3f'%(da.min(), da.max()))
    plt.show()
  # opt.param_groups[0]['lr'] = 9e-5 * (np.exp(-i/800)+0.01)#4e-5 * np.cos(i/20)**2 + 4e-6
ds['true'] = true

f = open('posts.json', 'w')
f.write(json.dumps({k:v.tolist() for k,v in ds.items()}))
f.close()
# torch.save(model.state_dict(), 'overfit')
dd = json.loads(open('posts.json', 'r').read())


import seaborn as sns

# compute the MSE in the target and the prediction
tr = torch.clamp(torch.tensor(dd['true']),
  CONFIG.heads.distogram.first_break, 
  CONFIG.heads.distogram.last_break)

dmse = [(k,((torch.tensor(v)-tr)**2).mean()) for k,v in dd.items() if k!='true' and k!='nans']

x_d_, y_d_ = list(zip(*dmse))
x_ = [x_d_[i-1] for i in range(1, len(x_d_)) if i%10==0]

# plt.plot(list(map(int,x_d_)), y_d_)
# plt.plot(list(map(int,x_d)), y_d)
# plt.xticks(list(map(int,x_)), list(map(str, x_)))
# plt.ylim(0,175)
# plt.savefig('relpos_outerseq_disto-mse.pdf')
# plt.show()

cm = sns.color_palette("rocket", as_cmap=True)

times = [100, 1000]
fig = plt.figure(figsize=((len(times)+2)*5, 1*5))
for i, t in enumerate(times):
  ax = fig.add_subplot(1, len(times)+2, i+1)
  ax.matshow(ds[t], cmap=cm)
  # ax.axis('off')
  ax.set_title('Iteration: %d'%t)


ax = fig.add_subplot(1, len(times)+2, 3)
ax.matshow(torch.clamp(tr, 
  CONFIG.heads.distogram.first_break, 
  CONFIG.heads.distogram.last_break), cmap=cm)
# ax.axis('off')
ax.set_title('Target')

ax = fig.add_subplot(1, len(times)+2, 4)
ax.plot(list(map(int,x_d_)), y_d_)
# ax.plot(list(map(int,x_d)), y_d)
# ax.set_xticks(list(map(int,x_)))
# ax.set_ylim(0,10)
ax.set_title('MSE')
# set_size(2, 2, ax)
# plt.savefig('relpos_outerseq_disto-mse.pdf')
# plt.show()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('posts.pdf')
plt.show()
