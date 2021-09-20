import requests, time, glob, os
from tqdm import tqdm

def read_folder(dir_, ext=''):
  files = glob.glob('%s*%s'%(dir_, ext))
  return list(map(lambda x:x.split('/')[-1].split('.')[0], files))

def seq_search(seq, total_waits=10):
  headers = {'Accept': 'application/json'}
  files = {'seq': (None, seq)}

  response = requests.post('https://rfam.org/search/sequence', headers=headers, files=files)
  d = response.json()
  url = d['resultURL']
  t = d['estimatedTime']

  for i in range(total_waits):
    pbar.set_description(desc + ' waiting... %d/%d'%(i, total_waits))
    time.sleep(3*int(t))
    response = requests.get(url, headers=headers)
    d = response.json()
    if 'hits' in d: break
  return d['hits']

def get_fam_and_e_val(seq):
  try:
    hits = seq_search(seq)
  except:
    return None, float('inf')
  min_e = float('inf')
  best_fam = None
  for name, [val] in hits.items():
    # score = val['score']
    e = float(val['E'])
    fam = val['acc']

    if e < min_e:
      best_fam = fam
      min_e = e

  pbar.set_description('found family: %s, with an E-value of %s'%(best_fam, str(min_e)))
  return best_fam, str(min_e)

def save(done):
  s='\n'.join([p+','+fam+','+min_e for p,(fam,min_e) in done.items()])
  f = open(fn, 'w')
  f.write(s)
  f.close()

fn = 'data/familymap.csv'
fnf = 'data/failed_codes.csv'
done = {}
if os.path.isfile(fn):
  # continue from checkpoint 
  done = open(fn,'r').read().split('\n')
  done = [a.split(',') for a in done]
  done = {a:(b,c) for (a,b,c) in done}
  print(len(done))

if os.path.isfile(fnf):
  cant_do = set(open(fnf,'r').read().split(','))

print('cant_do: '+str(len(cant_do)))
# print(cant_do)

print('done: '+str(len(done)))
# print(done)
# print()

sequence_dir = 'data/fasta/'
sq = set(map(lambda x:x.upper(), read_folder(sequence_dir, ext='.fasta')))
print('%d total'%len(sq))
sq = {s for s in sq if (s not in cant_do) and (s not in done)}
print('%d remaining'%len(sq))

failed = [cd for cd in cant_do]
pbar = tqdm(list(sq))
desc = ''
save_freq = 5
i = 0
for p in pbar:
  i += 1
  desc = 'running %s'%p
  pbar.set_description(desc)
  seq = open(sequence_dir+p+'.fasta', 'r').read().split('\n')[-1]

  if i%save_freq==0: print('WARNING: Do not quit')
  try:
    fam, min_e = get_fam_and_e_val(seq)
  except :
    # very rare- unpacking error
    fam = None

  if fam is not None: 
    done[p] = (fam, min_e)
  else:
    failed.append(p)

  if i%save_freq==0:
    save(done)
    f = open(fnf, 'w')
    f.write(','.join(failed))
    f.close()
    print('Save done, free to quit.')
save(done)
f = open(fnf, 'w')
f.write(','.join(failed))
f.close()
print('Save done, free to quit.')
