# print(open('data/fasta/7D7W-A.fasta', 'r').read().split('\n')[1])

import glob

s = set()
check = []
wlowercase = {}
for fn in glob.glob('data/msa/*.a3m'):
  c = fn.split('/')[-1].split('.')[0]
  msa = open(fn).read()
  s = s.union(set(msa))
  line = [len([a for a in l if not a.islower()]) for l in msa.split('\n') if len(l)]
  n = line[0]
  assert all(n==l for l in line), 'inconsistent'
  wlowercase[c] = n
  f = open('data/fasta/%s.fasta'%c, 'r').read().split('\n')[1]
  N = len(f)
  check.append(n==N)
  if not check[-1]:
    print((n, N, c))
    print(f)
    print(msa[:msa.index('\n')])
print(s)
print(sum(check)/len(check))


# prevv = {}
# for fn in glob.glob('data/msa/*.a3m'):
#   prevv[fn.split('/')[-1].split('.')[0]] = open(fn).read()

# wolower = {}
# for fn in glob.glob('final_msa/*.aln'):
#   wolower[fn.split('/')[-1].split('.')[0]] = open(fn).read()


# w = set(wlowercase.keys())
# p = set(prevv.keys())
# print((set(w-p), set(p-w)))
# wo = set(wolower.keys())
# print(wo)
# d1, d2 = w-wo, wo-w
# print((d1, d2))
# print(len(w))

# lower = {}
# for k,v in wlowercase.items():
#   if any(l.islower() for l in v):
#     lower[k] = v

# print(len(lower))

# wolc = {}
# # check = []
# import matplotlib.pyplot as plt
# lensx = []
# lens = []
# for k,v in wolower.items():
#   line = [len(l) for l in wolower[k].split('\n') if len(l)]
#   lensx.append(line[0])
#   lens.append(len(line))
  # [print(l) for l in line]
  # a = [line[i] == line[0] for i in range(1, len(line))]
  # # print(all(a))
  # check.append(all(a))
# plt.plot(lensx, lens)
# # plt.hist(lens, bins=100)
# plt.show()
# # print(all(check))