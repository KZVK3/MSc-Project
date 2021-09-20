from Bio import PDB
import numpy as np, os
from urllib.request import urlopen
from tqdm import tqdm
import json

'''

* load in BGSU list of RNA structures
* pull raw data not already pulled
* load each pdb file, write new file corresponding to just one chain
* write the sequences in fasta format to run msa

(in the case of multiple models it only takes the first)

'''

op_ascii = lambda path:urlopen(path).read().decode("ascii", errors='ignore')

def save(text, code, extension, dir_='data/raw/'):
  f = open(dir_+'%s.%s'%(code, extension), 'w')
  f.write(text)
  f.close()

def get_pdb(pdb_code_chain):
  pbar.set_description('Running %s...'%pdb_code_chain)
  bases = {'A','C','G','U'}
  pdb_id, chain_id = pdb_code_chain.split('-')

  raw_path_pdb = 'data/raw/%s.pdb'%pdb_id
  raw_path_cif = 'data/raw/%s.cif'%pdb_id
  final_path = 'data/pdb/%s-%s.pdb'%(pdb_id, chain_id)
  parser, fio = PDB.PDBParser, PDB.PDBIO

  structure_exists = (os.path.isfile(final_path) or os.path.isfile(final_path.replace('.pdb','.cif')))
  fasta_exists = os.path.isfile('data/fasta/%s-%s.fasta'%(pdb_id, chain_id))
  if structure_exists and fasta_exists: 
    pbar.set_description('%s-%s already exists.'%(pdb_id, chain_id))
    return

  if os.path.isfile(raw_path_pdb):
    raw_path = raw_path_pdb
  elif os.path.isfile(raw_path_cif):
    raw_path = raw_path_cif
    final_path = final_path.replace('.pdb','.cif')
    parser, fio = PDB.MMCIFParser, PDB.MMCIFIO
  else:
    pbar.set_description('Pulling data as the raw file for %s didn\'t exist'%pdb_code_chain)
    pdb_src = 'https://files.rcsb.org/download/%s.pdb'%pdb_id
    try:# download the pdb file
      save(op_ascii(pdb_src), pdb_id, 'pdb')
      raw_path = raw_path_pdb
    except :# download the cif file
      try:
        save(op_ascii(pdb_src.replace('pdb','cif')), pdb_id, 'cif')
        raw_path = raw_path_cif
        final_path = final_path.replace('.pdb','.cif')
        parser, fio = PDB.MMCIFParser, PDB.MMCIFIO
      except:
        print('failed on %s'%pdb_id)

  ## Read the PDB file and extract the chain from structure[0]
  model = parser(QUIET=1).get_structure(pdb_id, raw_path)[0]

  io = fio()
  io.set_structure(model[chain_id])
  io.save(final_path)
  pbar.set_description('Saved %s in %s'%(pdb_code_chain, final_path))
              
  structure = parser(QUIET=1).get_structure("X", final_path)
  # if len(structure): print('%d models, only using first'%len(structure))
  model,*_ = structure.get_list()
  [chain_obj] = [ch for ch in model if ch.get_id()==chain_id]
  res = [r.get_resname().replace(' ', '') for r in chain_obj.get_residues()]
  # res = [r if r in bases else '_' for r in res]
  # seq = ''.join(res)
  seq = ''.join([r for r in res if r in bases]).replace('_','')
  save('>%s\n%s'%(pdb_code_chain,seq), 
    pdb_code_chain, 'fasta', dir_='data/fasta/')

pdb_codes = json.loads(open('data/resolution_partition.json', 'r').read())['all']

paths = ['data/','data/raw/','data/pdb/','data/fasta/']
[os.mkdir(path) for path in paths if not os.path.isdir(path)]

pbar = tqdm(pdb_codes)
for codech in pbar:
  get_pdb(codech)

print(pdb_codes)
# pdb_code_chain = '1YFG-A'

# pdb_string, query_sequence = get_pdb(pdb_code_chain)
# residue_index = np.array([i for i, l in enumerate(query_sequence) if l!='_'])
# query_sequence = query_sequence.replace('_', '')

# print(query_sequence)
# print(residue_index)



