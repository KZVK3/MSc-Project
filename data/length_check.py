# compress all pdb rna objects
# read all fasta file
# map fasta to ints 
# compare lengths and values
from tqdm import tqdm
import numpy as np
import glob
d = {}
for fn in tqdm(glob.glob("data/fasta/*.fasta")):
  d[fn.split('/')[-1][:-6]] = open(fn, 'r').read().split('\n')[1]

dl =  {k:len(v) for k,v in d.items()}


# print(d)
from build_dataset import open_data
from constants import RNA_const
data = open_data()

extra_data = [k for k in data if k not in dl]
extra_fasta = [k for k in dl if k not in data]
equal_keys = len(extra_data)==0 and len(extra_fasta)==0
print('equal_keys: '+str(equal_keys))
if not equal_keys: 
  print(extra_data)
  print(extra_fasta)

err = 0
for k in data:
  aa = data[k].base_type
  fa = [RNA_const.base_ids[l] for l in d[k]]
  dif = np.array(aa) - np.array(fa)
  err += np.abs(dif).sum()
print(err)