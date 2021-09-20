import sys
# work in the parent directory
sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-1]))
from data.constants import RNA_const
from modules import make_transform_from_reference, QuatAffine, rot_to_quat
import torch
import string, copy
from numpy import pi
one_hot = torch.nn.functional.one_hot

_MSA_FEATURE_NAMES = [
    'msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask',
    'true_msa'
]
NUM_RES = 'num residues placeholder'

def non_ensemble_fn(batch, device):
  batch['seq_mask'] = torch.ones(*batch['aatype'].shape, dtype=float, device=device)
  batch['msa_mask'] = torch.ones(*batch['msa'].shape, dtype=float, device=device)
  # batch['msa_row_mask'] = torch.ones(batch['msa'].shape[0], dtype=float)
  
  """Compute the HHblits MSA profile if not already present."""
  if 'hhblits_profile' in batch:
    return batch

  # Compute the profile for every residue (over all MSA sequences).
  if 'rna' in batch:
    n_base, x, gap = 4, 1, 1
    n_class = n_base+x+gap
  else:
    n_prot, x, gap = 20, 1, 1
    n_class = n_prot+x+gap
  batch['hhblits_profile'] = one_hot(batch['msa'], n_class).float().mean(dim=0)

  # print('hhblits_profile: '+str(batch['hhblits_profile'].shape))

  return batch

def sample_msa(chain, max_seq, keep_extra):
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.

  Args:
    chain: batch to sample msa from.
    max_seq: number of sequences to sample.
    keep_extra: When True sequences not sampled are put into fields starting
      with 'extra_*'.

  Returns:
    chain with sampled msa.
  """
  num_seq = chain['msa'].shape[0]
  if num_seq > 1:
    shuffled = torch.multinomial(torch.ones(num_seq-1), num_samples=num_seq-1, replacement=False)
    index_order = torch.cat([torch.zeros(1, dtype=int), shuffled+1], axis=0)
    num_sel = min(max_seq, num_seq)

    sel_seq, not_sel_seq = index_order[:num_sel], index_order[num_sel:]
  else:
    sel_seq, not_sel_seq = slice(None), slice(None)

  for k in _MSA_FEATURE_NAMES:
    if k in chain:
      if keep_extra:
        chain['extra_' + k] = chain[k][not_sel_seq]
      chain[k] = chain[k][sel_seq]
  return chain

def make_masked_msa(batch, config, replace_fraction, device, test=False):
  """Create data for BERT on raw MSA."""
  
  if 'rna' in batch:
    n_base, x, gap = 4, 1, 1
    random_aa = torch.tensor([0.25] * n_base + [0.] * (x  + gap), device=device)
  else:
    n_prot, x, gap = 20, 1, 1
    random_aa = torch.tensor([0.] * n_prot + [0.] * (x  + gap), device=device)

  categorical_probs = (
      config.uniform_prob * random_aa[None,None,:] +
      config.profile_prob * batch['hhblits_profile'][None,...] +
      config.same_prob * one_hot(batch['msa'], len(random_aa)))

  # Put all remaining probability on [MASK] which is a new column
  mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob

  assert mask_prob >= 0.
  _1 = torch.ones(*categorical_probs.shape[:-1], 1, device=device)
  categorical_probs = torch.cat((categorical_probs, _1 * mask_prob), dim=-1)

  if test: replace_fraction = 0
  mask_position = torch.rand(*batch['msa'].shape, device=device) < replace_fraction

  sh = categorical_probs.shape
  probs_2d = categorical_probs.reshape(-1, sh[-1])

  samples_2d = torch.multinomial(probs_2d, 1, True).T
  bert_msa = samples_2d.reshape(sh[:-1])

  # Mix real and masked MSA
  batch['bert_mask'] = mask_position.float()
  batch['true_msa'] = batch['msa']
  batch['msa'] = torch.where(mask_position, bert_msa, batch['msa'])
  # batch['extra_msa'] = batch['extra_msa']
  return batch

def make_msa_feat(batch):
  """Create and concatenate MSA features."""
  x, gap, bert = 1, 1, 1
  n_prot = 20
  n_base_ = 4
  n_base = n_base_ if 'rna' in batch else n_prot
  n_res = batch['aatype'].shape[0]
  n_msa = batch['msa'].shape[0]
  n_tok_msa = n_base + x + gap + bert
  rna_msa_emb = n_base_ + x + gap + bert
  prt_msa_emb = n_prot + x + gap + bert
  
  msa_1hot = one_hot(batch['msa'], n_tok_msa)
  device = msa_1hot.device

  has_deletion = torch.clamp(batch['deletion_matrix'], min=0., max=1.)
  deletion_value = torch.atan(batch['deletion_matrix'] / 3.) * (2. / pi)

  msa_feat = [msa_1hot, has_deletion[...,None], deletion_value[...,None]]
  
  hasdel_delval = 2
  rna_msa_emb, prt_msa_emb = rna_msa_emb + hasdel_delval, prt_msa_emb + hasdel_delval

  if 'cluster_profile' in batch:
    deletion_mean_value = (torch.atan(batch['cluster_deletion_mean'] / 3.) * (2. / pi))
    msa_feat.extend([batch['cluster_profile'], deletion_mean_value[...,None],])

    del_mean = 1
    cmn = x + gap + bert + del_mean
    rna_msa_emb, prt_msa_emb = rna_msa_emb + n_base + cmn, prt_msa_emb + n_prot + cmn

  if 'extra_deletion_matrix' in batch:
    batch['extra_has_deletion'] = torch.clamp(batch['extra_deletion_matrix'], 0., 1.)
    batch['extra_deletion_value'] = torch.atan(batch['extra_deletion_matrix'] / 3.) * (2. / pi)

  batch['msa_feat'] = torch.cat(msa_feat, dim=-1)

  # the plus one is because the first dim is reserved for domain break indicators 
  # for chains this is always 0
  domain_break = 1
  batch['target_feat'] = one_hot(batch['aatype']+domain_break, domain_break + n_base + x)

  if 'rna' in batch:# we must include both 'target_feat' and 'rna_target_feat'
    batch['rna_target_feat'] = batch['target_feat']
    batch['rna_msa_feat'] =  batch['msa_feat']
    batch['msa_feat'] = torch.zeros(n_msa, n_res, prt_msa_emb, dtype=int, device=device)
    batch['target_feat'] = torch.zeros(n_res, n_prot+domain_break+x, dtype=int, device=device)
  else:
    batch['rna_target_feat'] = torch.zeros(n_res, n_base_+domain_break+x, dtype=int, device=device)
    batch['rna_msa_feat'] =  torch.zeros(n_msa, n_res, rna_msa_emb, dtype=int, device=device)
  return batch

def ensemble_fn(cfg, data, device, test=False):
  """Function to be mapped over the ensemble dimension."""
  batch = data.copy()

  ###################
  """Input pipeline functions that can be ensembled and averaged."""
  common_cfg = cfg.common
  train_cfg = cfg.training.constant

  # map_fns = []

  pad_msa_clusters = train_cfg.max_msa_clusters
  max_msa_clusters = pad_msa_clusters
  max_extra_msa = common_cfg.max_extra_msa
    
    #     print(k + ', ' + str(batch[k].shape))
    #     seen.append(k)
    #   else:
    #     print(k+' not in batch!! ')
    # print('in batch not seen')
    # for k, v in batch.items():
    #   if k not in seen:
    #     print(k+', '+str(v.shape))
    # n = train_cfg.crop_size

  batch = sample_msa(batch, max_msa_clusters, keep_extra=True)
  if 'masked_msa' in common_cfg:
    # Masked MSA should come *before* MSA clustering so that
    # the clustering and full MSA profile do not leak information about
    # the masked locations and secret corrupted locations.
    batch = make_masked_msa(batch, common_cfg.masked_msa, 
      train_cfg.masked_msa_replace_fraction, device, test=test)

  if common_cfg.msa_cluster_features:
    gap_agreement_weight=0.
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask

    x = 1
    n_base = 4 if 'rna' in batch else 20

    weights = torch.cat([
      torch.zeros(n_base + x, device=device), 
      gap_agreement_weight * torch.ones(1, device=device), 
      torch.zeros(1, device=device)
    ], dim=0)

    ''' note the +bert below, this is because the msa_ids dict doesn't have the bert token'''
    # Make agreement score as weighted Hamming distance
    sample_one_hot = (batch['msa_mask'][:, :, None] * one_hot(batch['msa'], len(weights)))
    extra_one_hot = (batch['extra_msa_mask'][:, :, None] * one_hot(batch['extra_msa'], len(weights)))

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(extra_one_hot.reshape(extra_num_seq, num_res * len(weights)),
               (sample_one_hot * weights).reshape(num_seq, num_res * len(weights)).T)

    # Assign each sequence in the extra sequences to the closest MSA sample
    batch['extra_cluster_assignment'] = torch.argmax(agreement, dim=1)

    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = batch['msa'].shape[0]

    def unsorted_segment_sum(data, indices, n_segments):
      z = torch.zeros(n_segments, *data.shape[1:], dtype=data.dtype, device=device)
      for i, j in enumerate(indices): z[j] += data[i]
      return z
      
    def csum(x): 
      return unsorted_segment_sum(x, batch['extra_cluster_assignment'], num_seq)

    mask = batch['extra_msa_mask']
    mask_counts = 1e-6 + batch['msa_mask'] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * one_hot(batch['extra_msa'], len(weights)))
    msa_sum += one_hot(batch['msa'], len(weights))  # Original sequence
    batch['cluster_profile'] = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = csum(mask * batch['extra_deletion_matrix'])
    del_sum += batch['deletion_matrix']  # Original sequence
    batch['cluster_deletion_mean'] = del_sum / mask_counts
    del del_sum


  # Crop after creating the cluster profiles.
  if max_extra_msa:
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    num_seq = batch['extra_msa'].shape[0]
    num_sel = min(max_extra_msa, num_seq)

    if num_sel:
      select_indices = torch.multinomial(torch.ones(num_seq, device=device), num_samples=num_sel, replacement=False)
      for k in _MSA_FEATURE_NAMES:
        if 'extra_' + k in batch:
          batch['extra_' + k] = batch['extra_' + k][select_indices]
  else:
    for k in _MSA_FEATURE_NAMES:
      if 'extra_' + k in batch:
        del batch['extra_' + k]

  batch = make_msa_feat(batch)
  # crop_feats = dict(train_cfg.feat)

  return batch

def get_y_feat(chain, basis, device, basis_shifts=None):
  # y code, deterministic
  y_feat = {}

  def shift(arr, s):
    # s = -1 or 0 or 1
    if s==0: return arr
    if s==-1:# use the previous values
      a = torch.empty_like(arr)
      a[:-1] = arr[1:]
      a[-1] = 2*arr[-1] - arr[-2]# create a fake point at the end
    elif s==1:
      a = torch.empty_like(arr)
      a[1:] = arr[:-1]
      a[0] = 2*arr[0] - arr[1]# create a fake point at the end
    else:
      return None
    return a

  atm_pos = torch.tensor(chain.atom_positions, device=device)
  atom_mask = torch.tensor(chain.atom_mask, device=device)
  # basis is a tuple of 3 ints indicating the origin atom index, x axis atom, xy-plane atom
  origin_ix, xax_ix, xypl_ix = basis

  y_feat['pseudo_beta'] = atm_pos[:,origin_ix]
  y_feat['pseudo_beta_mask'] = atom_mask[:,origin_ix]

  if basis_shifts is None:
    xy_pos = atm_pos[:, xypl_ix]
    o_pos = atm_pos[:, origin_ix]
    x_pos = atm_pos[:, xax_ix]
  else:
    # this is made for the case origin_ix=C4, x_axis=nextP, xy_plane=P
    # basis_shifts = (0,1,0)
    so, sx, sxy = basis_shifts
    xy_pos = shift(atm_pos, sxy)[:, xypl_ix]
    o_pos = shift(atm_pos, so)[:, origin_ix]
    x_pos = shift(atm_pos, sx)[:, xax_ix]

  rot, trans = make_transform_from_reference(
                                  n_xyz=xy_pos, 
                                  ca_xyz=o_pos, 
                                  c_xyz=x_pos)
  
  affines = QuatAffine(quaternion=rot_to_quat(rot, unstack_inputs=True),
          translation=trans, device=device, rotation=rot, unstack_inputs=True)
  
  y_feat['backbone_affine_tensor'] = affines.to_tensor()
  y_feat['backbone_affine_mask'] = y_feat['pseudo_beta_mask']

  # if slc[0] is None:
  #   y_feat = {k:v[None,...].repeat(num_ensemble, *([1]*len(v.shape)))
  #                                       for k,v in y_feat.items()}
  # else:
  #   # slice residue dim for each item in the batch
  #   y_feat = {k:torch.stack([v[s] for s in slc]) for k,v in y_feat.items()}
  return y_feat

def get_deletion_mat(sequences):
  """Parses sequences and deletion matrix from a3m format alignment.

  Args:
    The first sequence should be the query sequence.

  Returns:
    A tuple of:
      * A list of sequences that have been aligned to the query. These
        might contain duplicates.
      * The deletion matrix for the alignment as a list of lists. The element
        at `deletion_matrix[i][j]` is the number of residues deleted from
        the aligned sequence i at residue position j.
  """
  deletion_matrix = []
  for msa_sequence in sequences:
    deletion_vec = []
    deletion_count = 0
    for j in msa_sequence:
      if j.islower():
        deletion_count += 1
      else:
        deletion_vec.append(deletion_count)
        deletion_count = 0
    deletion_matrix.append(deletion_vec)

  # Make the MSA matrix out of aligned (deletion-free) sequences.
  deletion_table = str.maketrans('', '', string.ascii_lowercase)
  aligned_sequences = [s.translate(deletion_table) for s in sequences]
  return aligned_sequences, deletion_matrix

def prepare_features(
    chain, # contains seq, atom coords, atom mask, msa strings
    device,
    rna=True,
    basis=None,
    basis_shifts=None
  ):# -> batch # dict with tensors as input to the model
  
  num_res=chain.num_res

  if rna:
    base_ids = RNA_const.base_ids
    msa_ids = RNA_const.msa_ids
    feature_dict = {'rna':torch.tensor([1], device=device)}
    if basis is None:
      basis = RNA_const.basis 
      basis_shifts = None
    else:
      basis = tuple(RNA_const.backbone2ix[l] for l in basis)
  else:
    # TODO
    base_ids = None
    msa_ids = None
  
  feature_dict['aatype'] = torch.tensor([base_ids[l] for l in chain.sequence], dtype=int, device=device)
  feature_dict['residue_index'] = torch.arange(num_res, dtype=int, device=device)
  feature_dict['seq_length'] = torch.tensor([num_res] * num_res, dtype=int, device=device)# may only need to be length 1

  msa, deletion_matrix_ = get_deletion_mat(chain.msa)

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for sequence_index, sequence in enumerate(msa):
    if sequence in seen_sequences:
      continue
    seen_sequences.add(sequence)
    int_msa.append([msa_ids[base] if base in msa_ids else msa_ids['X'] for base in sequence])
    deletion_matrix.append(deletion_matrix_[sequence_index])

  feature_dict['deletion_matrix_int'] = torch.tensor(deletion_matrix, dtype=int, device=device)
  feature_dict['msa'] = torch.tensor(int_msa, dtype=int, device=device)

  feature_dict = dict(feature_dict)
  num_res = int(feature_dict['seq_length'][0])

  # feature_names = cfg.common.unsupervised_features

  feature_dict['deletion_matrix'] = (feature_dict.pop('deletion_matrix_int').float())

  """Apply filters and maps to an existing dataset, based on the config."""

  y_feat = get_y_feat(chain, basis, device, basis_shifts=basis_shifts)

  feature_dict = non_ensemble_fn(feature_dict, device)
  return feature_dict, y_feat

REQUIRED_FEAT = {
  'aatype',
  'bert_mask',
  'extra_msa',
  'extra_has_deletion',
  'extra_deletion_value',
  'extra_msa_mask',
  'msa_feat',
  'msa_mask',
  'residue_index',
  'seq_length',
  'seq_mask',
  'target_feat',
  'true_msa',
  'backbone_affine_tensor',
  'backbone_affine_mask',
  'pseudo_beta',
  'pseudo_beta_mask',
  'rna_msa_feat',
  'rna_target_feat',
}

def transform(feature_dict, y_feat, config, 
    num_ensemble, num_recycle, rna=True, test=False):# -> batch # dict with tensors as input to the model
  B = num_recycle + 1
  # Separate batch per ensembling & recycling step.
  num_ensemble *= num_recycle + 1
  device = feature_dict['aatype'].device
  n = feature_dict['aatype'].shape[0]

  feat = config.training.constant.feat
  crop_size = config.training.constant.crop_size
  slc = None
  # crop if it is too long
  if crop_size < n:# CROP SEQUENCE DIM
    l = (crop_size+1)//2
    mid = torch.randint(l, n-l+1, (1,), device=device).item()
    slc = slice(mid-l, mid+l)
    # print('slicing residue dimension')
    # seen = []
    for k, dim in feat.items():# dim = list of placeholders
      if k in feature_dict:
        feature_dict[k] = feature_dict[k][[slc if NUM_RES==u else slice(None) for u in dim]]
      # IT IS IMPORTANT TO CROP Y HERE, BEFORE RECYCLE ENSEMBLING
      if k in y_feat:
        y_feat[k] = y_feat[k][[slc if NUM_RES==u else slice(None) for u in dim]]
  
  
  # print(feat.keys())
  # print(feature_dict.keys())
  # print()

  # THIS ENSEMBLING IS OVER RECYCLING ITERATIONS, ONE CROP IS ASSOCIATED WITH ALL
  tenss = [ensemble_fn(config, feature_dict, device, test=test) for _ in range(num_ensemble)]

  required_feat = REQUIRED_FEAT.copy()
  if rna: required_feat.add('rna')

  b = {k:torch.stack([t[k] for t in tenss]).float() for k in tenss[0].keys() if k in required_feat}
  b['num_iter_recycling'] = torch.tensor([[num_recycle]]*B, device=device)

  # combine y features and x features
  for k, v in y_feat.items(): 
    b[k] = v[None,...].repeat(B, *([1]*len(v.shape)))

  return b









if __name__ == '__main__':
  from config import model_config
  from loader import RNA
  

  import gzip, json
  f = gzip.open('dataset.gz','rb')
  file_content = gzip.decompress(f.read()).decode("utf-8") 
  loaded_rnas = json.loads(file_content)
  data = {}
  for dname, dat in loaded_rnas.items():
    data[dname] = {}
    for code,dictt in dat.items():
      data[dname][code] = RNA(**dictt)

  model_name = 'model_5'
  cfg = model_config(model_name)
  num_recycle = 3#torch.randint(1,cfg.data.common.num_recycle+1, (1,)).item()

  for _ in range(20):
    i = torch.randint(0, len(data['train']), (1,)).item()
    code = list(data['train'].keys())[i]
    rna = data['train'][code]

    batch = prepare_features(rna,'cpu',rna=True)
    batch = transform(*batch, cfg.data, cfg.data.training.constant.num_ensemble, num_recycle, rna=True)
    print('\n'+'-'*100+'\n')
    if batch['seq_length'][0,0].item() < 40:break


  from data.generate_fake_example import generate_fake_example

  n_res=rna.num_res
  n_msa=batch['true_msa'].shape[1]
  n_xtr=batch['extra_msa'].shape[1]

  random_batch = generate_fake_example(num_recycle, n_res=n_res, n_msa=n_msa, n_xtr=n_xtr)
  
  print('difference: ')
  for k,v in random_batch.items():
    check = batch[k].shape == v.shape
    if check:
      print(k+', successful: \t\t'+str(batch[k].shape))
      pass
    else:
      print('>>>>>>>>>> '+k+', fail: '+str(batch[k].shape)+', correct (for proteins):'+str(v.shape))
  

  # init AF using random_batch, then try and pass through batch
  from modules import AlphaFold

  cfg.model.global_config.device = 'cpu'
  af = AlphaFold(cfg, True, True)
  out, loss, info = af(random_batch)
  del loss
  del out

  af.load_state_dict(torch.load('../params/torch_'+model_name))
  af.eval()

  # for name, p in prevaf.named_parameters():
  #   if 'rna' in name:
  #     print(name)
  print('now passing in real data')
  out, loss, info = af(batch)
  print('succcess!')
  af.track_codes({}, 5)
  batch['code'] = code
  # check if the metrics work...
  af.validation_step(batch, 0)
