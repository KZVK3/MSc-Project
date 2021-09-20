import sys
# work in the parent directory
sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-1]))
from config import model_config
from modules import AlphaFold
from data.generate_fake_example import generate_fake_example
import torch
import json
'''
# load state dict from previous saved torch model of alphafold trained params
# load in data example
# instantiate new model
# pass example through it-- with the larger embedding dimensions
# for each named param go through and copy, slice last dim if it is an embedding param
'''

def instantiateNewAF(config, num_recycle):
  random_batch = generate_fake_example(num_recycle)
  # pass through the model to initialise all params
  # need to have is_training and compute_loss true, as it wont init head params otherwise!
  # config.model.global_config.msa_n_token = n_token
  # config.model.heads.masked_msa.num_output = n_token#== num logits in MaskedMSAHead
  config.model.global_config.device = 'cpu'
  af = AlphaFold(config, True, True)
  out, loss = af(random_batch)
  del loss
  del out
  return af
  
def main():
  model_name = 'model_5'
  config = model_config(model_name)

  num_recycle = torch.randint(1,config.data.common.num_recycle+1, (1,)).item()

  # prevmsa_emb = 49
  # prevtrg_emb = 22
  # prevn_token = 23 # Equal to true MSA: 20 res types, X, gap, bert mask
  prevaf = instantiateNewAF(config, num_recycle)

  for name, p in prevaf.named_parameters():
    if 'rna' in name:
      print(name)
  
  #  test if I can pass in only RNA keys
  random_batch = generate_fake_example(num_recycle)
  del random_batch['msa_feat']
  del random_batch['target_feat']
  out, loss = prevaf(random_batch)
  del loss
  del out
  print('succcess!')


  # # load up the trained parameters
  # prevaf.load_state_dict(torch.load('params/torch_'+model_name))
  # prevaf.eval()
  
  # d = '/data'
  # p = sys.path[0]
  # sys.path.insert(0, p+d)# bit hacky, should really refactor
  # print(sys.path[0])
  # from data.loader import loadRNA, RNA
  # from data.data_transforms import transform

  # num_recycle = 3
  # code, chain_id = '1MMS','C'
  # ten = lambda d: {k:torch.tensor(v) if k!='msa' and type(v)!=int else v for k,v in d.items()}
  # rna = RNA(**ten(loadRNA(code, chain_id, 'data/').__dict__))

  # batch = transform(rna, config, num_recycle)
  # af = instantiateNewAF(config, num_recycle)

  # # copy the parameters and copy old emb into a slice of the new emb
  # # record the slices to use later as param groups for different optimisers
  # param_groups = {}
  # print('\n\nFinding the embedding parameters...')
  # for (_, pp), (pn, p) in zip(prevaf.named_parameters(), af.named_parameters()):
  #   if p.shape!=pp.shape:
  #     print(pn+', '+str(p.shape)+', '+str(pp.shape))

  #     # find the dim of the mismatch
  #     old_slc_ = []
  #     old_slc = []
  #     new_slc = []
  #     for s, ns in zip(pp.shape, p.shape):
  #       if ns==s:
  #         old_slc_.append(slice(None))
  #         old_slc.append((None, None))
  #         new_slc.append((None, None))
  #       else:
  #         old_slc_.append(slice(None, s))
  #         old_slc.append((None, s))
  #         new_slc.append((s, None))

  #     param_groups[pn] = (old_slc, new_slc)

  #     p.data[old_slc_] = pp.data
  #   else:
  #     p.data = pp.data

  # print('saving augmented torch params')
  # torch.save(af.state_dict(), '../params/aug_torch_'+model_name)

  # name = '../params/aug_param_group.json'
  # f = open(name, 'w')
  # f.write(json.dumps(param_groups))
  # f.close()


if __name__ == '__main__':
  main()