import torch

# def generate_fake_example(config, num_recycle=3, n_res=9, n_msa=12, n_xtr=17, 
#     msa_emb=49, targ_emb=22, n_token=21):
#   required_feat = {'aatype','backbone_affine_tensor','backbone_affine_mask',
#       'bert_mask','extra_msa','extra_has_deletion','extra_deletion_value',
#       'extra_msa_mask','msa_feat','msa_mask','pseudo_beta','pseudo_beta_mask',
#       'residue_index','seq_length','seq_mask','target_feat','true_msa'}

#   dims = {'msa placeholder':n_msa, 'extra msa placeholder':n_xtr, 
#           'num residues placeholder':n_res}
#   embs = {'msa_feat':msa_emb, 'backbone_affine_tensor':7,
#           'target_feat':targ_emb, 'pseudo_beta':3}
#   redun = {'seq_length':dims['num residues placeholder'],}
#   _1hot = {'target_feat'}
#   categ = {'extra_msa', 'aatype', 'true_msa'}



#   random_batch = {}
#   for k in required_feat:
#     shape = config.data.eval.feat[k]
#     emb = embs[k] if k in embs else None
#     shape = [num_recycle+1]+[dims[p] if p is not None else emb for p in shape]
#     if k in redun:
#       random_batch[k] = torch.tensor([[redun[k]]*n_res]*(num_recycle + 1))
#     elif k in _1hot:
#       cat = torch.randint(0,shape[-1], shape[:-1])
#       random_batch[k] = torch.nn.functional.one_hot(cat, shape[-1]).float()
#     elif 'mask' in k:
#       random_batch[k] = (torch.rand(*shape) < 0.8).float()
#     elif k in categ:
#       random_batch[k] = torch.randint(0,n_token, shape)
#     else:
#       random_batch[k] = torch.randn(*shape)

#   random_batch['num_iter_recycling'] = torch.tensor([[num_recycle]]*(num_recycle+1))
#   random_batch['rna'] = True
#   # if print_:
#   #   for k,v in random_batch.items():
#   #     print(k+': '+str(v.shape)+', '+str(type(v)))
#   return random_batch

def generate_fake_example(num_recycle=3, n_res=9, n_msa=12, n_xtr=17):
  B = num_recycle + 1
  nprot, nbase, x, gap, bert = 20, 4, 1, 1, 1
  prot_msa_emb, prot_targ = 49, nprot + x + gap
  rna_msa_emb, rna_targ = 17, nbase + x + gap
  ex = {}
  ex['aatype'] = torch.randint(0, nbase+x, (B, n_res))
  ex['seq_mask'] = torch.randint(0,2, (B, n_res))
  ex['true_msa'] = torch.randint(0,nbase+x+gap, (B, n_msa, n_res))
  ex['bert_mask'] = torch.randint(0,2, (B, n_msa, n_res))
  ex['msa_mask'] = torch.randint(0,2, (B, n_msa, n_res))
  ex['extra_msa'] = torch.randint(0,nbase+x+gap, (B, n_xtr, n_res))
  ex['extra_msa_mask'] = torch.randint(0,1, (B, n_xtr, n_res))
  ex['extra_has_deletion'] = torch.randint(0,2, (B, n_xtr, n_res))
  ex['extra_deletion_value'] = torch.randint(0,2, (B, n_xtr, n_res))
  ex['residue_index'] = torch.arange(n_res, dtype=int)[None,:].repeat(B, 1)
  ex['backbone_affine_tensor'] = torch.randn(B, n_res, 7)
  ex['backbone_affine_mask'] = torch.randint(0,2, (B, n_res))
  ex['pseudo_beta_mask'] = ex['backbone_affine_mask']
  ex['pseudo_beta'] = torch.randn(B, n_res, 3)
  ex['seq_length'] = torch.tensor([[n_res]*n_res]*B)

  # protein shape
  ex['msa_feat'] = torch.randn(B, n_msa, n_res, prot_msa_emb)
  ex['target_feat'] = torch.randint(0,2, (B, n_res, prot_targ))
  # rna shape
  ex['rna_msa_feat'] = torch.randn(B, n_msa, n_res, rna_msa_emb)
  ex['rna_target_feat'] = torch.randint(0,2, (B, n_res, rna_targ))


  ex = {k:v.float() for k,v in ex.items()}
  ex['num_iter_recycling'] = torch.tensor([[num_recycle]]*B, dtype=int)
  ex['rna'] = torch.tensor([[1]]*B)
  
  return ex