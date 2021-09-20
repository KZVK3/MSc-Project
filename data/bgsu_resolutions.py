from urllib.request import urlopen
import json

op_ascii = lambda path:urlopen(path).read().decode("ascii", errors='ignore')

def pull_BGSU_non_redundant_structures(min_resolution, version='3.183'):
  res = {'1.5A','2.0A','2.5A','3.0A','3.5A','4.0A','20.0A','all'}
  assert min_resolution in res, 'Invalid resolution, use one of'+str(res)

  def split(x):
    x=x.split(',')
    return (x[0], x[1], x[2:])

  file_loc = 'http://rna.bgsu.edu/rna3dhub/nrlist/download/%s/%s/csv'%(version, min_resolution)
  f = op_ascii(file_loc).replace('"','').split('\n')[:-1]# last char is ''
  bgsu_data = list(map(split, f))
  bgsu_code, pdb_code, _ = list(zip(*bgsu_data))

  pdb_codes = list(map(lambda x:x.split('+'), pdb_code))
  def sp(codechain):
    code, _, chain = codechain.split('|')
    return code+'-'+chain

  return {sp(c) for codes in pdb_codes for c in codes}

increasing_res = ['1.5A','2.0A','2.5A','3.0A','3.5A','4.0A','20.0A','all']
all_sets = {}
for i in range(len(increasing_res)):
  res = increasing_res[i]
  new_set = pull_BGSU_non_redundant_structures(res)

  if i>0:
    prev_res = increasing_res[i-1]
    all_sets[res] = set(new_set.union(all_sets[prev_res]))
  else:
    all_sets[res] = new_set

total = set(all_sets['all'])
assert all(all(c in total for c in all_sets[r]) for r in increasing_res[:-1]), 'not all codes contained in "all".'


partition = {}
for i in list(range(len(increasing_res)))[::-1]:
  res = increasing_res[i]
  prev_res = set([]) if i==0 else all_sets[increasing_res[i-1]]
  partition[res] = set(all_sets[res] - prev_res)
  assert all(p in all_sets[res] for p in prev_res), 'invalid sets'

partition['above 20'] = partition['all']
partition['all'] = total
partition = {res:list(s) for res,s in partition.items()}
f = open('data/resolution_partition.json', 'w')
f.write(json.dumps(partition))
f.close()

