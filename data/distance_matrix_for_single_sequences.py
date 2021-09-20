from Bio import pairwise2
from tqdm import tqdm
import glob, os, json
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def find_duplicates():
  ''' 
  Some fasta is repeated
  Enumerate the set of (unique) fasta, mapping fasta->index
  Go through all codes and map to unique fasta id
  '''
  data = []
  msa_fasta = []
  for fn in tqdm(glob.glob("data/fasta/*.fasta")):
    code =  fn.split('/')[-1].split('.')[0]
    mfn = 'data/msa/%s.a3m'%code
    d = (code, open(fn, 'r').read().split('\n')[1])
    if os.path.isfile(mfn):
      msa_fasta.append(d+(open(mfn, 'r').read(),))
    else:
      data.append(d)


  codes, records = list(zip(*data))

  ##
  fasta2id = {fasta:i for i, fasta in enumerate(set(records))}
  code2fasta = {code:fasta for code, fasta in zip(codes, records)}
  code2id = {code:fasta2id[code2fasta[code]] for code in codes}
  return code2id
  ##

  # rec_set = set()
  # record, code = [], []
  # a = []
  # for c, r in zip(codes,records):
  #   if r not in rec_set:
  #     record.append(r)
  #     code.append(c)
  #     rec_set.add(r)
  #   else:
  #     cc = code[record.index(r)]
  #     a.append((c, cc))
  # print((len(records), len(record)))
  # records, codes = record, code

  # f = open('duplicate_fasta.txt', 'w')
  # f.write('\n'.join(['(%s, %s)'%(b, c) for b,c in a]))
  # f.close()
  

if not os.path.isfile('data/single_distances.json'):
  chars = {'A', 'G', 'U', 'C', 'X'}
  matrix = {(c1, c2):-int(c1!=c2) for c1 in chars for c2 in chars}
  gap_open = -10
  gap_extend = -0.5
  num_clusters = 200

  data = []
  msa_fasta = []
  for fn in tqdm(glob.glob("data/fasta/*.fasta")):
    code =  fn.split('/')[-1].split('.')[0]
    mfn = 'data/msa/%s.a3m'%code
    d = (code, open(fn, 'r').read().split('\n')[1])
    if os.path.isfile(mfn):
      msa_fasta.append(d+(open(mfn, 'r').read(),))
    else:
      data.append(d)

  codes, records = list(zip(*data))
  rec_set = set()
  record, code = [], []
  for c, r in zip(codes,records):
    if r not in rec_set:
      record.append(r)
      code.append(c)
      rec_set.add(r)
  print((len(records), len(record)))
  records, codes = record, code

  a=[all(c in chars for c in r) for r in records]
  print(all(a))

  num_seqs = len(records)


  distances = np.zeros((num_seqs, num_seqs))
  for i in tqdm(range(0,num_seqs)):
    for j in range(i+1,num_seqs):
      a = pairwise2.align.globalds(records[i],records[j],matrix,gap_open,gap_extend)
      (s1,s2,score,start,end) = a[0]
      apx_norm = min(len(records[i]), len(records[j]))
      distances[i, j] = (0.01 - score) / (1 + apx_norm)# score is negative
  distances += distances.T

  f = open('data/single_distances.json', 'w')
  f.write(json.dumps({'dist_mat':distances.tolist(), 'codes':codes}))
  f.close()

print('loading distance matrix')
d = json.loads(open('data/single_distances.json', 'r').read())
codes, distances = d['codes'], np.array(d['dist_mat'])

print('finding duplicates')
code2id = find_duplicates()

# ##
# [code2id[c] for c in codes]
# ##

# code_ix = [code2id[code] for code in codes]

id2codes = {id_:set() for _, id_ in code2id.items()}
for code,id_ in code2id.items(): id2codes[id_].add(code)

print('clustering')
clustering = AgglomerativeClustering(
  n_clusters=50, 
  affinity='precomputed',
  linkage='average'
).fit(distances)

clusters = np.array(clustering.labels_)

cluster_codes = {c:set() for c in clusters}
for i,cluster in enumerate(clusters):
  # we only recorded one of the codes that has this fasta
  # we must find all of the other codes that also had that fasta
  all_codes = id2codes[code2id[codes[i]]]
  cluster_codes[cluster] = all_codes.union(cluster_codes[cluster])
cluster_codes = [list(v) for _,v in cluster_codes.items()]

print('should be roughly equal... hopefully')
print([len(c) for c in cluster_codes])

f = open('data/single_apx_families.json', 'w')
f.write(json.dumps(cluster_codes))
f.close()


