import numpy as np
import matplotlib.pyplot as plt
import torch
from modules import makeLinear, TriangleMultiplication, TriangleAttention, Transition, DistogramHead
import ml_collections
from tqdm import tqdm
import seaborn as sns
import json


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

def construct_dist_from_logits(logits, bins):
  diff = bins[1]-bins[0]
  b0 = torch.tensor([bins[0] - diff], device='cpu')
  bins = torch.cat((b0, bins), dim=0) - diff*0.5
  p = torch.exp(logits)
  p /= p.sum(-1)[...,None]
  return torch.einsum('ijk,k->ij', p, bins)


def train_step(opt, model, x, y):
  yh = model(*x, y)
  l = model.loss(yh, y)
  opt.zero_grad()
  l['loss'].backward()
  opt.step()
  return l['loss'].item(), yh['logits'].data, yh['bin_edges'].data, l['true_dist'].data 

import json
from constants import RNA_const
b = RNA_const.basis[0]
data = json.loads(open('minidata.json','r').read())
base2i = {base:i for i, base in enumerate('ACGUX')}
toseq = lambda ss: torch.tensor([base2i[s] for s in ss])
xs = [(toseq(v['sequence']), torch.tensor(v['atom_mask'])[:,b], {'pseudo_beta':torch.tensor(v['atom_positions'])[:,b], 'pseudo_beta_mask':torch.tensor(v['atom_mask'])[:,b]}) for k,v in data.items()]
# xs = [(s[:40],m[:40],{k:v[:40] for k,v in y.items()}) for s,m,y in xs][:2]
xs = [(s,m,{k:v for k,v in y.items()}) for s,m,y in xs]
# xs = [generate_example(n,a,b,c) for n,a,b,c in [(37,0.5,2,4), (19,0.1,0.4,5),(43,5,0.3,1.9)]]
s,m,y_feat = xs[2]

run = False
cm = sns.color_palette("rocket", as_cmap=True)
if run:
  ds = {}
  # opt.param_groups[0]['lr'] = 5e-3
  pbar = tqdm(range(1, 10000+1))
  for i in pbar:
    # s,m,y_feat = xs[np.random.randint(len(xs))]
    l, lg, bn, tr = train_step(opt, model, (s, m), y_feat)
    deets = (l, opt.param_groups[0]['lr'])
    pbar.set_description('L: %.3f, lr: %s'%deets)
    # if i%5==0: print(deets)
    if i%100==0:
      pred = construct_dist_from_logits(lg, bn)
      ds[i] = pred
    # w = 2 
    # fig, axes = plt.subplots(1, w, figsize=(10, 10*w))
    # for i,da in enumerate([pred, tr]):
    #   axes[i].imshow(da)
    #   axes[i].set_title('%.3f, %.3f'%(da.min(), da.max()))
    # if pr is not None:
    #   # axes[-2].imshow(torch.flip(pr[0], (-1, )))
    #   axes[-2].imshow(pr[0], norm=None, vmin=0, vmax=1)
    #   axes[-2].set_xlabel('bins')
    #   axes[-2].set_title('probs for\nbase 1->i')
    #   axes[-1].imshow(torch.functional.F.one_hot(tb[0], 64))
    #   axes[-1].set_xlabel('bins')
    #   axes[-1].set_title('true bins\nbase 1->i')
    # plt.show()
  # opt.param_groups[0]['lr'] = 4e-5 * (np.exp(-i/80)+0.01)#4e-5 * np.cos(i/20)**2 + 4e-6


# ds = {}
# for i in range(1,10000+1):
#   l, lg, be, tr = train_step(opt, model, s, y_feat)
#   if i%100==0:
#     print(l)
#     d = construct_dist_from_logits(lg, be)
#     ds[i] = d
#     print((d.min(), d.max()))
#     plt.imshow(d)
#     plt.show()

  times = [8000, 9000, 10000]
  fig = plt.figure(figsize=((len(times)+1)*5, 1*5))
  for i, t in enumerate(times):
    ax = fig.add_subplot(1, len(times)+1, i+1)
    ax.matshow(ds[t], cmap=cm)
    # ax.axis('off')
    ax.set_title('Iteration: %d'%t)

  ax = fig.add_subplot(1, len(times)+1, 4)
  ax.matshow(torch.clamp(tr, 
    CONFIG.heads.distogram.first_break, 
    CONFIG.heads.distogram.last_break), cmap=cm)
  # ax.axis('off')
  ax.set_title('Target')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.savefig('relpos_outerseq_evo_disto.pdf')
  plt.show()



  nans = [(k, (torch.isnan(d).sum()/torch.numel(d)).item()) for k,d in ds.items()]
  a = {k:i.tolist() for k,i in ds.items()}
  a['true'] = tr.tolist()
  a['nans'] = nans
  f = open('data-evo-true-disto-loss.json', 'w')
  f.write(json.dumps(a))
  f.close()


dd = json.loads(open('data-evo-true-disto-loss.json', 'r').read())


times = [100, 2000, 10000]
fig = plt.figure(figsize=((len(times)+3)*5, 1*5))
for i, t in enumerate(times):
  ax = fig.add_subplot(1, 5, i+1)
  ax.matshow(dd[str(t)], cmap=cm)
  ax.set_title('Iteration: %d'%t)

ax = fig.add_subplot(1, 5, 4)
ax.matshow(torch.clamp(torch.tensor(dd['true']),
  CONFIG.heads.distogram.first_break, 
  CONFIG.heads.distogram.last_break), cmap=cm)
ax.set_title('Target')

# compute the MSE in the target and the prediction
tr = torch.clamp(torch.tensor(dd['true']),
  CONFIG.heads.distogram.first_break, 
  CONFIG.heads.distogram.last_break)

dmse = [(k,((torch.tensor(v)-tr)**2).mean()) for k,v in dd.items() if k!='true' and k!='nans']

x_d, y_d = list(zip(*dd['nans']))
x_d_, y_d_ = list(zip(*dmse))
x_ = [x_d_[i-1] for i in range(1, len(x_d_)) if i%10==0]

ax = fig.add_subplot(1, 5, 5)
ax.plot(list(map(int,x_d_)), y_d_, label='MSE')
ax.plot(list(map(int,x_d)), y_d, label='Prop. of float overflow')
ax.set_xticks(list(map(int,x_)))
ax.set_ylim(0,10)
ax.legend()
# ax.set_title('')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('relpos_outerseq_evo_disto.pdf')
plt.show()


