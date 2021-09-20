import glob, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable

dat = {}
for fn in tqdm(glob.glob("data/fasta/*.fasta")):
  code =  fn.split('/')[-1].split('.')[0]
  l = len(open(fn, 'r').read().split('\n')[1])
  mfn = 'data/msa/%s.a3m'%code
  if os.path.isfile(mfn):
    m = len(open(mfn, 'r').read().split('\n'))
    dat[code] = (l, m)
  else:
    dat[code] = (l, None)
  
clip = 80
count_lim = 10
short = [l for k,(l,m) in dat.items() if l<clip]
short_no_msa = [l for k,(l,m) in dat.items() if l<clip and m is None]
total = len(dat)
short_ = len(short)
short_no_msa_ = len(short_no_msa)
print((total, len([None for _,(_,v) in dat.items() if v is not None])))


def plotpie(total, within, wwithin):
  # Pie chart, where the slices will be ordered and plotted counter-clockwise:
  labels = '$\{\ell<80\}\cap\{$No MSA$\}$', '$\ell\geq 80$', '$\{\ell<80\}\cap\{$MSA$\}$'
  sizes = [wwithin, total-within, within-wwithin]
  explode = (0.03, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

  fig1, ax1 = plt.subplots()
  c = sns.color_palette("husl", 8)
  c2 = sns.color_palette("Set2")
  ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
          shadow=True, startangle=180, colors=[c[6],c2[2],c2[6]])
  ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.savefig('data_MSA_proportion_pie.pdf')
  plt.show()

plotpie(total, short_, short_no_msa_)


short_no_msa = sorted(short_no_msa)
plt.plot(range(len(short_no_msa)), short_no_msa)
plt.show()

prev = short_no_msa[0]
count = 1
counts = []
for i, l in enumerate(short_no_msa[1:]):
  if l==prev:
    count += 1
  else:
    counts.append((prev, count))
    prev = l
    count = 1
print(counts)
lns, con = list(zip(*counts))
con = np.array(con)
xx = (con/max(con))#**0.5

cmp = sns.color_palette("mako", as_cmap=True)
color = [cmp(i) for i in xx]

plt.rcParams["figure.figsize"] = (10,4)
plt.bar(lns, con, color=color)

sm = ScalarMappable(cmap=cmp, norm=plt.Normalize(0,200))
sm.set_array([])

cbar = plt.colorbar(sm)
cbar.set_label('Count', rotation=270,labelpad=10)

plt.plot([0, clip], [count_lim]*2, 'r', lw=0.5, label='Count = %d'%count_lim)
plt.xlabel('Sequence Length')
plt.ylabel('Length Count')
plt.legend()
plt.savefig('short_length_counts.pdf')
plt.show()
    



plot_len = False
if plot_len:
  d = {}
  for fn in tqdm(glob.glob("data/fasta/*.fasta")):
    if len(fn.split('/')[-1][:-6]):
      d[fn.split('/')[-1][:-6]] = {'msa':None}
  for fn in tqdm(glob.glob("data/fasta/*.fasta")):
    d[fn.split('/')[-1][:-6]]['sequence'] = open(fn, 'r').read().split('\n')[1]
  # for fn in tqdm(glob.glob("data/msa/*.a3m")):
  #   msa = open(fn, 'r').read().split('>')
  #   msa = [s[s.index('\n')+1:].replace('\n','') for s in msa if '\n' in s]
  #   # ss = '\n'.join(s for s in msa if len(s) and s[0]!='>')
  #   d[fn.split('/')[-1][:-4-8]]['msa'] = msa


  flen = {k:len(v['sequence']) for k,v in d.items()}
  # msalen = {k:len(v['msa'][0]) for k,v in d.items() if v['msa'] is not None}

  # corr = [msalen[k]==flen[k] for k in msalen]
  # print('%.2f%% correct lengths'%(100*sum(corr)/len(corr)))

  # diff = np.array([msalen[k]-flen[k] for k in msalen])
  # print((np.mean([msalen[k] for k in msalen]), np.std(diff)))




  def plot_prop_len(ax, xlim=False):
    ax.scatter(np.array(lengths)+(np.arange(len(lengths))%2)*0.1 - 0.05, 
      np.linspace(0, 1, len(lengths)), 
      c=colours, s=0.4, alpha=0.5)
    # hacky (extra fake point to add labels)
    ax.scatter([-1], [0], c='r', s=1.7, alpha=0.5, label='No hit')
    ax.scatter([-1], [0], c='g', s=1.7, alpha=0.5, label='Hit')
    ax.set_ylim(-0.05, 1.05)
    if xlim: 
      ax.set_xlim(-1, 61)
      ax.set_xlabel('Sequence length')
      ax.set_ylabel(' '*45+'Proportion of sequences with length $<$ $x$')
    else:
      ax.legend(loc='lower right')
    return ax

  def plot_ev(ax, xlim=False):
    ax.scatter(len_hits, logev, c='g', s=0.7, alpha=0.5)
    ax.plot([len_hits[0], len_hits[-1]],[np.log(1e-100)]*2, alpha=0.4, label='underflow')
    # plt.title('Sequences With Hits')
    if xlim: 
      ax.set_xlim(0, 500)
      ax.set_xlabel('Sequence length')
      ax.set_ylabel(' '*45+'$log($e-value$)$')
    else:
      ax.legend()
    return ax

  s = open('data/RNA-family-evalue.csv', 'r').read().split('\n')[1:]
  hits = {t.split(',')[0]:float(t.split(',')[-1]) for t in s if len(t.split(','))>1}
  print(hits)

  len_ev = [(v, hits[k] if k in hits else None) for k,v in flen.items()]
  len_ev.sort(key=lambda x:x[0])
  lengths, evalues = list(zip(*len_ev))

  colours = ['r' if ev is None else 'g' for ev in evalues]
  len_hits = [lengths[i] for i, ev in enumerate(evalues) if ev is not None]
  logev = [np.log(ev+1e-100) for ev in evalues if ev is not None]

  plt.rcParams["figure.figsize"] = (6,4)
  fig, axs = plt.subplots(2)
  ax1 = plot_prop_len(axs[0])
  ax2 = plot_prop_len(axs[1], xlim=True)
  plt.tight_layout()
  plt.savefig('length_prop.pdf')
  # plt.savefig('length_prop_zoom.pdf')
  plt.show()


  fig, axs = plt.subplots(2)
  ax1 = plot_ev(axs[0], xlim=False)
  ax2 = plot_ev(axs[1], xlim=True)
  plt.tight_layout()
  plt.savefig('length_log_eval.pdf')
  plt.show()