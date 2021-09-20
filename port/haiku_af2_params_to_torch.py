import sys
# work in the parent directory
sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-1]))
from config import model_config
from modules import AlphaFold
import os, shutil, glob
import torch
import numpy as np
from data.generate_fake_example import generate_fake_example
'''
# download the parameters if they arent already there
# load in the params 
# move jax params to torch 
'''

def move_jax_params_to_torch(torch_module, params, 
  skip={}, replace_exc={}, replace=[], to_double=False):

  def execute(torch_module, full_path, p, mod):
    full_path = full_path.replace('scale', 'weight').replace('offset', 'bias')
    param = torch.tensor(p)
    if to_double:
      param = param.double()
    if mod=='Linear' and 'weights' in full_path:
      full_path = full_path.replace('weights', 'weight')
      param = param.T# torch linear layers seem to store Linear.weight as it's transpose
    exec('%s.data = param'%full_path)
    return torch_module
  
  differences = {
    'evoformer_iteration':'evoformer_iteration',
    '__layer_stack_no_state':'blocks',
    'extra_msa_stack':'extra_msa_stack'
  }
  skipped = []
  for path, jparams in params.items():
    # if 'template' in path:print(path)
    if any(s in path for s in skip): 
      skipped.append((path, jparams.keys()))
      continue
    for a,b in replace_exc.items():
      path = path.replace(a, b)

    for a,b in replace: 
      if a in path:
        path = path.replace(a, b)
        break
    module_stack = path.split('/')
    module_stack[0] = 'torch_module'
    # only checks for 1 type of stacking atm..
    l = [k for k in differences if k in module_stack]
    if len(l):
      [ss] = l
      trch_strg = differences[ss]
      # find each occurence, left to right
      ixs = [i for i,m in enumerate(module_stack) if m==ss]
      pn, p = next(iter(jparams.items()))
      # go through len(ixs) axes of ixs
      block_sizes = p.shape[:len(ixs)]
      all_ixs = np.array(np.meshgrid(*[np.arange(n) 
            for n in block_sizes])).reshape(len(block_sizes),-1).T

      # flatten 
      s = '.'.join(module_stack[:ixs[0]])
      m = ['.'.join(module_stack[i+1:j]) for i,j in zip(ixs[:-1], ixs[1:])]
      e = '.'.join(module_stack[ixs[-1]+1:])
      rest = m + [e]

      paths = [(s + ''.join(['.%s[%d].%s'%(trch_strg, i, r) 
            for i, r in zip(row, rest)]), row) for row in all_ixs]
      
      for pn, p in jparams.items():
        for path_, ixs in paths:
          mod = str(type(eval(path_))).split('.')[-1][:-2]
          full_path = path_ + '.' + pn
          torch_module = execute(torch_module, full_path, np.array(p[tuple(ixs[...,None])]).squeeze(), mod)
    else:
      pth = path[path.index('/')+1:]+'.' if '/' in path else ''
      pth = pth.replace('/','.')
      mod = str(type(eval(('torch_module.'+pth)[:-1]))).split('.')[-1][:-2]
      for pn, p in jparams.items():
        full_path = 'torch_module.'+pth+pn
        torch_module = execute(torch_module, full_path, np.array(p), mod)
  return torch_module, skipped

def download_params():
  ndir = not os.path.isdir('params/')
  if ndir or len(glob.glob("params/*.npz"))==0: 
    if ndir: os.mkdir('params/')
    print('downloading all parameters')
    os.system('curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar \
      | tar x -C params')

def get_haiku(model_name, random_batch):
  if not os.path.isdir('alphafold/'): 
    print('getting haiku alphafold')
    os.system('git clone https://github.com/deepmind/alphafold.git --quiet')
    os.system('(cd alphafold; git checkout 0bab1bf84d9d887aba5cfb6d09af1e8c3ecbc408 --quiet)')
    os.system('mv alphafold/alphafold af')
    shutil.rmtree('alphafold/')
    os.system('mv af alphafold')

  print('loading haiku params')
  from alphafold.model import data, config as af_config, model
  params = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir="")

  # note: models 1,2 have diff number of params compared to models 3,4,5
  if any(str(m) in model_name for m in [1,2]): 
    model_config = af_config.model_config("model_1_ptm")
    model_config.data.eval.num_ensemble = 1
    model_runner = model.RunModel(model_config, params)
  if any(str(m) in model_name for m in [3,4,5]): 
    model_config = af_config.model_config("model_3_ptm")
    model_config.data.eval.num_ensemble = 1
    model_runner = model.RunModel(model_config, params)

  model_runner.params = params
  model_runner.init_params({k:v.numpy() for k,v in random_batch.items()})
  return model_runner

def clean():
  # we dont need haiku alphafold anymore
  shutil.rmtree('alphafold/')

  # remove any haiku params
  for fn in glob.glob("params/*.npz"): os.remove(fn)

def main():
  download_params()
  # open the config and create a fake batch
  model_name = 'model_5'
  config = model_config(model_name)
  num_recycle = 3#torch.randint(1,config.data.common.num_recycle, (1,)).item()

  random_batch = generate_fake_example(num_recycle)

  # pass through the model to initialise all params
  # need to have is_training and compute_loss true, as it wont init head params otherwise!
  print('initialising model...',end='')
  config.model.global_config.device = 'cpu'
  af = AlphaFold(config, True, True)
  out, loss, info = af(random_batch)
  del loss
  del out
  print('done')

  model_runner = get_haiku(model_name, random_batch)

  jax_n_params = 0
  for _, jparams in model_runner.params.items():
    jax_n_params += sum([np.prod(p.shape) for _, p in jparams.items()])
  torch_n_params = sum([np.prod(p.shape) for _, p in af.named_parameters()])
  print('should be roughly the same (just to check weight sharing):')
  print(torch_n_params)
  print(jax_n_params)


  print('transferring params...')
  # this function doesn't generalise well, it just looks for diffences between the
  # specific implementation in jax and torch
  loaded_af, skipped_parameters = move_jax_params_to_torch(
      af, 
      model_runner.params, 
      skip={
        'template', 
        'experimentally_resolved_head',
        'rigid_sidechain',
        'predicted_lddt_head',
        'predicted_aligned_error_head'
      },
      replace_exc={
        'distogram_head':'heads/distogram', 
        'structure_module':'heads/structure_module/generate_affines',
        'masked_msa_head':'heads/masked_msa'
      },
      replace=[
        ('transition_layer_norm', 'transition_layer_norm'),
        ('fold_iteration/transition_1','fold_iteration/transition[2]'),
        ('fold_iteration/transition_2','fold_iteration/transition[4]'),
        ('fold_iteration/transition','fold_iteration/transition[0]')
      ]
    )
  print('saving torch params')
  torch.save(loaded_af.state_dict(), 'params/torch_'+model_name)
  print('didn\'t load the following:\n\n'+str('\n'.join(m+': ['+', '.join([a for a in t])+']' for m,t in skipped_parameters)))

  clean()

if __name__ == '__main__':
  main()
