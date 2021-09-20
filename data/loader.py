import numpy as np
import dataclasses
from Bio import PDB
import os
from constants import RNA_const
'''
this code is inspired by the protein.py file in the pipeline of AlphaFold 
https://github.com/deepmind/alphafold/blob/main/alphafold/common/protein.py
'''

@dataclasses.dataclass(frozen=True)
class RNA:
  code:str
  num_res: int
  num_msa: int
  # atom coords (angstroms). The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: list  # [num_res, num_atom_type, 3]

  # RNA base type for each residue represented as an integer token
  sequence: str  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: list  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  base_index: list  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: list  # [num_res, num_atom_type]

  msa: list  # [num_msa, num_res]


def loadRNA(code, chain_id, path='', msa=None):
  # base_ids = RNA_const.base_ids
  base_atom_order = RNA_const.base_atom_order
  atom_type_num = RNA_const.atom_type_num

  # msa_path = path+'data/msa/%s-%s.a3m'%(code, chain_id)
  pdb_path = path+'data/pdb/%s-%s.pdb'%(code, chain_id)
  if msa is None:
    msa = [open(path+'data/fasta/%s-%s.fasta'%(code, chain_id), 'r').read().split('\n')[1]]

  parser = PDB.PDBParser
  if not os.path.isfile(pdb_path):
    pdb_path = pdb_path.replace('.pdb', '.cif')
    parser = PDB.MMCIFParser

  model = parser().get_structure('none', pdb_path)[0]
  # model = PDB.PDBParser(QUIET=1).get_structure("X", pdb_path).get_list()[0]
  # print((code, chain_id, pdb_path))
  # print(next(iter(model)))
  chain = model[chain_id] if chain_id in model else next(iter(model))

  atom_positions = []
  base_type = []
  atom_mask = []
  base_index = []
  b_factors = []
  for res in chain:
    rname = res.resname
    if rname not in base_atom_order:
      continue
    else:
      atom_order = base_atom_order[rname]

      assert len(rname)==1, 'invalid base'

      # basetype = base_ids[rname] if rname in base_ids else len(base_ids)-1

      pos = np.zeros((atom_type_num, 3))
      mask = np.zeros((atom_type_num,), dtype=int)
      res_b_factors = np.zeros((atom_type_num,))

      for atom in res:
        if atom.name not in atom_order:
          continue
        i = atom_order[atom.name]
        pos[i] = atom.coord
        mask[i] = 1.
        res_b_factors[i] = atom.bfactor

      if np.sum(mask) < 0.5:
        # If no known atom positions are reported for the residue then skip it.
        continue

      base_type.append(rname)
      atom_positions.append(pos.tolist())
      atom_mask.append(mask.tolist())
      base_index.append(res.id[1])
      b_factors.append(res_b_factors.tolist())

  n = len(base_type)
  # print(msa)
  nl = [len([l for l in s if (not l.islower()) or l=='_']) for s in msa]
  assert all(l==n for l in nl), '%s-%s len %d, msa is inconsistent:\n%s'%(code, chain_id, n, '\n'.join(msa))

  return RNA(
    code='%s-%s'%(code, chain_id),
    num_res=n,
    num_msa=len(msa),
    atom_positions=atom_positions,
    atom_mask=atom_mask,
    sequence=''.join(base_type),
    base_index=base_index,
    b_factors=b_factors,
    msa=msa
  )

def toPDB(rna, chain_id='A'):
  '''
  This is based on the toPDB() function in alphafold
  '''
  base_atom_order = RNA_const.base_atom_order
  atom_type_num = RNA_const.atom_type_num
  base_ids = RNA_const.base_ids

  # base_ids_inv = {v:k for k,v in base_ids.items()}
  base_atom_order_inv = {k:{vv:kk for kk,vv in v.items()} for k,v in base_atom_order.items()}
  base_atom_order_inv = {k:[v[i] for i in range(len(v))] for k,v in base_atom_order_inv.items()}

  pdb_lines = []

  atom_mask = np.array(rna.atom_mask)
  aatype = rna.sequence
  atom_positions = np.array(rna.atom_positions)
  base_index = np.array(rna.base_index).astype(np.int32)
  b_factors = np.array(rna.b_factors)

  # if np.any(aatype > len(base_ids)):
  #   raise ValueError('Invalid aatypes.')

  pdb_lines.append('MODEL     1')
  atom_index = 1
  # Add all atom sites.
  for i in range(len(aatype)):
    # res_name_3 = res_1to3(aatype[i])
    base_name = aatype[i]
    atom_types = base_atom_order_inv[base_name]
    for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{base_name:>3} {chain_id:>1}'
                   f'{base_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the chain.
  chain_end = 'TER'
  chain_termination_line = (
      f'{chain_end:<6}{atom_index:>5}      {aatype[-1]:>3} '
      f'{chain_id:>1}{base_index[-1]:>4}')
  pdb_lines.append(chain_termination_line)
  pdb_lines.append('ENDMDL')

  pdb_lines.append('END')
  pdb_lines.append('')
  pdb_string = '\n'.join(pdb_lines)

  return pdb_string


if __name__ == '__main__':
  code, chain_id = '1ZCI', 'A'#'3SKL', 'B'#'3ADD', 'C'

  rna = loadRNA(code, chain_id)

