import numpy as np
import matplotlib.pyplot as plt
import torch
from modules import DistogramHead, makeLinear
import dataclasses
import seaborn as sns

def generate_example(n=100):
  n_base = 4
  s = torch.randint(0, n_base, size=(n,))
  c = torch.zeros((n,3))
  t = torch.linspace(0, 1, len(c))
  c[:,0] = torch.sin(t) + 4
  c[:,1] = 0.6*torch.cos(6*t)**2
  c[:,2] = 0.1*torch.sin(3*t)
  c *= 10
  return c, s

@dataclasses.dataclass(frozen=True)
class GConfig:
  zero_init: bool
  device: str
  n_base: int

@dataclasses.dataclass(frozen=True)
class Config:
  num_bins: int
  first_break: int
  last_break: int
  max_relative_feature: int
  pair_channel: int

class Model(torch.nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.dh = DistogramHead(config, global_config)

    args = (config.pair_channel, torch.float, global_config.device)
    self.relpos = makeLinear(2 * config.max_relative_feature + 1, *args)

    self.left_single = makeLinear(self.global_config.n_base, *args)
    self.right_single = makeLinear(self.global_config.n_base, *args)

  def forward(self, seq, y_feat):
    n = len(seq)
    
    x = torch.functional.F.one_hot(seq.long(), num_classes=self.global_config.n_base).float()
    left, right = self.left_single(x), self.right_single(x)
    pair_activations = left[:, None] + right[None]

    pos = torch.arange(0, n, device=x.device)
    offset = pos[:, None] - pos[None, :]
    rel_pos = torch.functional.F.one_hot(
      torch.clip(
        offset + config.max_relative_feature,
        min=0, 
        max=2 * config.max_relative_feature
      ).long(),
      2 * config.max_relative_feature + 1
    ).to(left.dtype)

    pair_activations += self.relpos(rel_pos)
    value = self.dh(representations=dict(pair=pair_activations), batch=y_feat, is_training=True)
    return value
  
  def loss(self, value, y_feat):
    return self.dh.loss(value, y_feat)

config = Config(64, 2.0, 22.0, 32, 32)
global_config = GConfig(False, 'cpu', 4)

# def construct_dist_from_logits(logits, bins):
#   diff = bins[1]-bins[0]
#   b0 = torch.tensor([bins[0] - diff], device=global_config.device)
#   bins = torch.cat((b0, torch.flip(bins, (0,))), dim=0) - diff*0.5
#   return torch.einsum('ijk,k->ij', logits, bins)

def construct_dist_from_logits(logits, bins):
  diff = bins[1]-bins[0]
  b0 = torch.tensor([bins[0] - diff], device='cpu')
  bins = torch.cat((b0, bins), dim=0) - diff*0.5
  p = torch.exp(logits)
  p /= p.sum(-1)[...,None]
  return torch.einsum('ijk,k->ij', p, bins)

def plot(c):
  dist = ((c[...,None,:] - c[...,None,:,:])**2).sum(-1)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(*c.T, marker='o')
  plt.show()
  plt.imshow(dist)
  plt.show()

  bins = torch.linspace(config.first_break, config.last_break, global_config.n_bins-1)
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


import json
from constants import RNA_const
b = RNA_const.basis[0]
data = json.loads(open('minidata.json','r').read())
base2i = {base:i for i, base in enumerate('ACGUX')}
toseq = lambda ss: torch.tensor([base2i[s] for s in ss])
xs = [(toseq(v['sequence']), torch.tensor(v['atom_mask'])[:,b], {'pseudo_beta':torch.tensor(v['atom_positions'])[:,b], 'pseudo_beta_mask':torch.tensor(v['atom_mask'])[:,b]}) for k,v in data.items()]
xs = [(s,m,{k:v for k,v in y.items()}) for s,m,y in xs]

s,m,y_feat = xs[2]

model = Model(config, global_config)
_ = model(s, y_feat)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_step(opt, model, x, y):
  yh = model(x, y)
  l = model.loss(yh, y)
  opt.zero_grad()
  l['loss'].backward()
  opt.step()
  return l['loss'].item(), yh['logits'].data, yh['bin_edges'].data, l['true_dist']

c = y_feat['pseudo_beta']
dist = ((c[...,None,:] - c[...,None,:,:])**2).sum(-1)**0.5
plt.imshow(dist)
plt.show()

ds = {}
for i in range(1,10000+1):
  l, lg, be, tr = train_step(opt, model, s, y_feat)
  if i%100==0:
    print(l)
    d = construct_dist_from_logits(lg, be)
    ds[i] = d
    print((d.min(), d.max()))
    plt.imshow(d)
    plt.show()


nans = [(k, (torch.isnan(d).sum()/torch.numel(d)).item()) for k,d in ds.items()]
a = {k:i.tolist() for k,i in ds.items()}
a['true'] = tr.tolist()
a['nans'] = nans
f = open('data-disto-loss.json', 'w')
f.write(json.dumps(a))
f.close()


dd = json.loads(open('data-disto-loss.json', 'r').read())


x_d, y_d = list(zip(*dd['nans']))

# compute the MSE in the target and the prediction
tr = torch.clamp(torch.tensor(dd['true']),
  config.first_break, 
  config.last_break)

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

times = [100, 10000]
fig = plt.figure(figsize=((len(times)+2)*5, 1*5))
for i, t in enumerate(times):
  ax = fig.add_subplot(1, len(times)+2, i+1)
  ax.matshow(ds[t], cmap=cm)
  # ax.axis('off')
  ax.set_title('Iteration: %d'%t)


ax = fig.add_subplot(1, len(times)+2, 3)
ax.matshow(torch.clamp(tr, config.first_break, config.last_break), cmap=cm)
# ax.axis('off')
ax.set_title('Target')

ax = fig.add_subplot(1, len(times)+2, 4)
ax.plot(list(map(int,x_d_)), y_d_)
# ax.plot(list(map(int,x_d)), y_d)
ax.set_xticks(list(map(int,x_)))
ax.set_ylim(0,100)
ax.set_title('MSE')
# set_size(2, 2, ax)
# plt.savefig('relpos_outerseq_disto-mse.pdf')
# plt.show()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('relpos_outerseq_disto.pdf')
plt.show()
