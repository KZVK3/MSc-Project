import numpy as np
import glob
from tqdm import tqdm
from loader import loadRNA, RNA
import gzip, json, torch

'''
load all PDB into RNA obj (linked with MSA), check the lengths match
place into a json format, then dump, compress, save
'''

# def save_data(jdata, name):
#   print('compressing...', end=' ')
#   comp = gzip.compress(bytes(json.dumps(jdata), encoding='utf-8'))
#   print('done.\nwriting to file...', end=' ')
#   f = gzip.open('%s.gz'%name, 'wb')
#   f.write(comp)
#   f.close()
#   print('done.')

def open_data(name='compressed_RNA_data'):
  f = gzip.open('%s.gz'%name,'rb')
  file_content = gzip.decompress(f.read()).decode("utf-8") 
  loaded_rnas = json.loads(file_content)
  data = {}
  for code,dictt in loaded_rnas.items():
    rna = {k:torch.tensor(v) if k!='msa' and type(v)!=int else v for k,v in dictt.items()}
    data[code] = RNA(**rna)
  return data

# def construct_dataset(codes, name):
#   rnas = {ch:loadRNA(*ch.split('-'), msa=msa).__dict__ for ch, msa in tqdm(codes)}
#   save_data(rnas, name)



def check_msa(codech, msa):
  seq = open('data/fasta/%s.fasta'%codech, 'r').read().split('\n')[1]
  n = len(seq)
  # print(msa)
  nl = [len([l for l in s if not l.islower()]) for s in msa]
  return all(l==n for l in nl)
  # assert all(l==n for l in nl), '%s len %d, msa is inconsistent:\n%s\n%s'%(codech,n,seq,'\n'.join(msa))

def split_by_category(proportions, category_to_example_map):
  ''' 
  category_to_example_map: key = any hashable, value = list
  proportions: array like, sum to 1
  partition all of the examples (flattened values of category_to_example_map)
  into the apx proportions given, adding categories one-by-one
  '''
  assert sum(proportions)==1

  keys, length = list(zip(*[(k,len(v)) for k,v in category_to_example_map.items()]))
  assert all(l>0 for l in length), 'cannot have 0 counts'

  counts = np.array(length)
  total = counts.sum()

  n = len(counts)
  mask = np.ones(n)

  group = 0
  group_sizes = np.zeros(len(proportions))
  group_ixs = [set() for _ in proportions]

  for _ in range(n):
    p = counts * mask
    i = np.random.choice(range(n), p=p/p.sum())

    group_ixs[group].add(i)
    group_sizes[group] += counts[i]

    mask[i] = 0

    if group_sizes[group] > total * proportions[group]:
      group += 1
  assert len(set(range(n)) - {i for s in group_ixs for i in s})==0

  size_desired = [(s, total * p) for s,p in zip(group_sizes, proportions)]
  print('\n'.join(['Desired %d, got %d'%(d,s) for s,d in size_desired]))
  print('Final proportions: %.2f, %.2f, %.2f'%tuple((100*group_sizes/group_sizes.sum()).tolist()))
  print('Desired proportions: %.2f, %.2f, %.2f'%tuple(proportions))
  # recover the values
  sets = []
  for ixs in group_ixs:
    sets.append([e for i in ixs for e in category_to_example_map[keys[i]]])
  return sets

  # # place whole families in the training and hold out set
  # hold_codes_msa = []
  # train_codes_msa = []
  # for fam_ix in range(n):
  #   fm = families[fam_ix]
  #   chain_codes = fam2chains[fm]
  #   msas = [msa_files[c] for c in chain_codes]
  #   code_msa = list(zip(chain_codes, msas))

  #   if fam_ix in hold_ixs:
  #     hold_codes_msa.extend(code_msa)
  #   else:
  #     train_codes_msa.extend(code_msa)

  # nh, nt = len(hold_codes_msa), len(train_codes_msa)
  # print('MSA examples prop %.3f in hold out'%(nh/(nh+nt)))

  # return 

def collect_dataset(proportion_names, fprop=None):
  '''
  read in family map, compute counts for each family c<-{fam:count}
  while hold out is less than (1-split)*tot_seq_in_fams
    add a random family to hold out
  place remainder of examples in the training set
  compress datasets
  '''
  names, proportions = list(zip(*proportion_names))
  assert sum(proportions)==1

  msa_files = {fn.split('/')[-1].split('.')[0]:open(fn, 'r').read().split('\n') for fn in glob.glob('data/msa/*.a3m')}
  msa_files = {k:[s for s in msa if len(s)] for k,msa in msa_files.items()}
  
  msa_chains = list(msa_files.keys())
  # msa_files = [msa_files[k] for k in msa_chains]

  ##################################################
  ## the MSA families
  lines = open('data/RNA-family-evalue.csv','r').read().split('\n')
  family_map = [a.split(',') for a in lines if len(a)]
  chain2fam = {ch:fm for (ch,fm,ev) in family_map}
  fam2chains = {fm:[] for (ch,fm,ev) in family_map}

  for ch in msa_chains: 
    fam2chains[chain2fam[ch]].append(ch)

  fam2chains = {k:v for k,v in fam2chains.items() if len(v)>0}
  ##################################################

  ##################################################
  ## the single 'families'
  single_fams = json.loads(open('data/single_apx_families.json','r').read())
  single_fams = {str(i):s for i,s in enumerate(single_fams) if len(s)>0}
  ##################################################
  fp = proportions if fprop is None else fprop
  assert sum(fp)==1
  msa_groups = split_by_category(fp, fam2chains)
  single_groups = split_by_category(proportions, single_fams)
  dataset_codes = [a + b for a,b in zip(msa_groups, single_groups)]

  ##################################################
  ## check that all codes have been added, add MSA
  all_code_chains = [fn.split('/')[-1].split('.')[0] for fn in glob.glob("data/fasta/*.fasta")]
  ##################################################

  data = {}
  msacodes = set()

  for name, codes in zip(names, dataset_codes):
    codes_msa = [(code, msa_files[code] if code in msa_files else None) for code in codes]
    msacodes = msacodes.union({c for c, m in codes_msa if m is not None})

    rnas = {ch:loadRNA(*ch.split('-'), msa=msa).__dict__ for ch, msa in tqdm(codes_msa)}
    data[name] = rnas
  assert len(set(msa_files.keys()) - msacodes)==0, 'not all MSAs have been used'
    # construct_dataset(codes_msa, name)

  print('compressing...', end=' ')
  comp = gzip.compress(bytes(json.dumps(data), encoding='utf-8'))
  print('done.\nwriting to file...', end=' ')
  f = gzip.open('dataset.gz', 'wb')
  f.write(comp)
  f.close()
  print('done.')


  # families = list(fam2chains.keys())
  # # add fams to hold out set until the set is (1-split)*total num of examples
  # family_counts = np.array([len(fam2chains[fm]) for fm in families])
  # total = family_counts.sum()
  # hold_ixs = set()
  # hold_size = 0
  # n = len(family_counts)
  # mask = np.ones(n)
  # for _ in range(n):
  #   p = np.array(family_counts * mask)
  #   i = np.random.choice(range(n), p=p/p.sum())
  #   hold_ixs.add(i)
  #   hold_size += family_counts[i]
  #   mask[i] = 0
  #   if hold_size > total * (1 - split):
  #     break

  # # place whole families in the training and hold out set
  # hold_codes_msa = []
  # train_codes_msa = []
  # for fam_ix in range(n):
  #   fm = families[fam_ix]
  #   chain_codes = fam2chains[fm]
  #   msas = [msa_files[c] for c in chain_codes]
  #   code_msa = list(zip(chain_codes, msas))
  #   if fam_ix in hold_ixs:
  #     hold_codes_msa.extend(code_msa)
  #   else:
  #     train_codes_msa.extend(code_msa)
  # nh, nt = len(hold_codes_msa), len(train_codes_msa)
  # print('MSA examples prop %.3f in hold out'%(nh/(nh+nt)))

  # add all other codes to the training set

  # # family_map = {ch:(fm,ev) for (ch,fm,ev) in family_map}
  # hold_codes = {c for c, m in hold_codes_msa}
  # train_codes = {c for c, m in train_codes_msa}
  # so_far = hold_codes.union(train_codes)
  # all_code_chains = [fn.split('/')[-1].split('.')[0] for fn in glob.glob("data/fasta/*.fasta")]

  # for c in all_code_chains:
  #   if c not in so_far:
  #     train_codes_msa.append((c, None))

  # print('Final prop %.3f in hold out'%(nh/(nh+len(train_codes_msa))))

  # construct_dataset(train_codes_msa, 'training_set')
  # construct_dataset(hold_codes_msa, 'holdout_set')

if __name__=='__main__':
  # the last prop will collect less
  proportion_names = [('hold-out', 0.2), ('validation', 0.16), ('train', 0.64)]
  collect_dataset(proportion_names, [0.1, 0.14, 0.76])

  # category_to_example_map = {'fam%d'%i:['abc%d'%j for j in range(np.random.randint(10, 40))] for i in range(50)}
  # s = split_by_category([0.2, 0.2, 0.6], category_to_example_map)
  # print(len(s))
  # print(s)


  '''
  # usage:
  from config import model_config
  from data_transforms import transform

  data = open_data()
  config = model_config('model_5')
  data_gen = RNAData(data, config, transform)
  '''
  
  # code_chains = [fn.split('/')[-1][:-4] for fn in glob.glob("data/pdb/*.pdb")]
  # rnas = {code_chain:loadRNA(*code_chain.split('-')).__dict__ for code_chain in tqdm(code_chains)}

  # subset = {}
  # match, allmsa = 0, 0
  # for code_chain, rna in rnas.items():
  #   l = len(rna['msa'][0])
  #   if rna['num_msa']>1:
  #     msa_consistent = all(len(s)==l for s in rna['msa'])
  #     print((code_chain, l, rna['num_res'], l == rna['num_res'], msa_consistent))
  #     correct = (l == rna['num_res']) and msa_consistent
  #     match += correct
  #     allmsa += 1
  #     if correct: subset[code_chain] = rna
  #   else:
  #     if rna['num_res']==l:subset[code_chain] = rna
  # # rnas = subset

  # print('\n\n%.2f%% of the %d MSA\'s are the right length\n'%(100*match/allmsa, allmsa))


  # save_data(rnas, 'compressed_RNA_data')
  # save_data(subset, 'subset_RNA_data')

  # print('compressing...', end=' ')
  # d = gzip.compress(bytes(json.dumps(d), encoding='utf-8'))
  # print('done.\nwriting to file...', end=' ')
  # f = gzip.open('compressed_RNA_data.gz', 'wb')
  # f.write(d)
  # f.close()
  # print('done.')



  # i=0
  # for codename,rna in loaded_rnas.items():
  #   print(rna)
  #   i+=1
  #   if i > 3:break

  # for k,v in dd.items():
  #   if v['msa'] is not None:
  #     print(k)
  #     print('seq::')
  #     print(v['sequence'][:14])
  #     print('strc::')
  #     print(v['structure'][:14])
  #     print('msa::')
  #     print(v['msa'][:140])
  #     break
      


