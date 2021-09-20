import glob, numpy as np
from tqdm import tqdm

def clean(code, fasta, fasta_len, msa, msa_len):
  msa1 = [a for a in msa[0] if not a.islower()]
  assert msa_len < fasta_len, 'doesnt support this case'
  diff = fasta_len-msa_len
  score = [sum(a==b for a,b in zip(fasta[i:i+msa_len], msa1)) for i in range(diff+1)]
  shift = np.argmax(score)
  # print((shift, score))
  prefix, suffix = fasta[:shift], (fasta[shift-diff:] if shift-diff < 0 else '')
  # msa = [prefix + m + suffix for m in msa if len(m)]
  new_msa = []
  for m in msa:
    if len(m):
      p = '-'*len(prefix) if m[0]=='-' else prefix
      e = '-'*len(suffix) if m[-1]=='-' else suffix
      new_msa.append(p + m + e)
  return '\n'.join(new_msa)
  # print('cleaned:')
  # print(fasta)
  # print(msa[0])
  # pass
def check_msa(msa_string):
  line = [len([a for a in l if not a.islower()]) for l in msa_string.split('\n') if len(l)]
  n = line[0]
  assert all(n==l for l in line), 'inconsistent'
  return n

s = set()
check = []

to_write = []
# i = 0
for fn in tqdm(glob.glob('data/a3m/*.a3m')):
  # i += 1
  # if i>40:break
  c = fn.split('/')[-1].split('.')[0]
  msa = open(fn).read()
  s = s.union(set(msa))
  
  f = open('data/fasta/%s.fasta'%c, 'r').read().split('\n')[1]
  N = len(f)
  n = check_msa(msa)

  check.append(n==N)
  if not check[-1]:
    # print(c+',')
    # print((n, N, c))
    # print(f)
    # print(msa[:msa.index('\n')])
    msa = clean(c, f, N, msa.split('\n'), n)
  full_msa = f + '\n' + msa
  assert check_msa(full_msa)==N, 'correction failed'
  to_write.append( (c, full_msa) )

print(s)
print(sum(check)/len(check))

for c, msa in to_write:
  f = open('data/msa/%s.a3m'%c, 'w')
  f.write(msa)
  f.close()



