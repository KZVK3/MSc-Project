import dataclasses

@dataclasses.dataclass(frozen=True)
class Const:
  # map from RNA base one letter code --> integer token
  # includes X for unknown RNA token, and '-' for MSA
  base_ids: dict
  msa_ids: dict
  # map from RNA base one letter code --> dict
  #    each value maps from atom --> integer index
  base_atom_order: dict
  # map for the atoms common to all bases to an integer 
  # token, the backbone atoms
  backbone2ix: dict
  # number of atoms in the base with the most atoms
  atom_type_num: int
  basis: tuple

def get_constants():
  # restypes = [
  #     'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
  #     'S', 'T', 'W', 'Y', 'V'
  # ]
  # # restype_order = {restype: i for i, restype in enumerate(restypes)}
  # # restype_num = len(restypes)  # := 20.
  # # unk_restype_index = restype_num  # Catch-all index for unknown restypes.

  # restypes_with_x = restypes + ['X']
  # restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}



  # start_ix = len(restype_order_with_x)

  base_enum = 'ACGUX'
  msa_enum = base_enum + '_'
  base_ids = {l:i for i,l in enumerate(base_enum)}
  msa_ids = {l:i for i,l in enumerate(msa_enum)}
  msa_ids['-'] = msa_ids['_']

  backbone_order = ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"]
  base_order = {
    'G':["N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4"],
    'A':["N9","C8","N7","C5","C6","N6","N1","C2","N3","C4"],
    'C':["N1","C2","O2","N3","C4","N4","C5","C6"],
    'U':["N1","C2","O2","N3","C4","O4","C5","C6"],
  }

  base_atom_order = {k:list(backbone_order+v) for k,v in base_order.items()}
  base_atom_order = {k:dict(zip(v, range(len(v)))) for k,v in base_atom_order.items()}
  atom_type_num = max(len(v) for k,v in base_atom_order.items())

  backbone2ix = {l:i for i,l in enumerate(backbone_order)}


  origin_atom = "C4'"
  xaxis_atom = "O3'"
  xy_plane_atom = "O5'"
  basis = (origin_atom, xaxis_atom, xy_plane_atom)
  ############################################################################
  # Deepmind use ca as origin, c as x-axis, which has direction in the N->C terminus
  # C1' origin, C2' as x-axis, O4' in the xy-plane
  # should be a good basis, as it is common to all bases, 
  # C1' is closest to the bases, C2' and O4' are adjacent
  # and they are going in the same direction for all chains
  # additionally, RNA is released from it's 5' end first, 
  '''
  Since the backbone of RNA is larger and has many moving parts,
  it would be better to use atoms closer to the adjacent frames.
  Set:
  - C1' origin
  - previous P1 as the x-axis
  - C5' / O5' / P1 as in the xy-plane
  '''
  # basis is contained in backbone, get the indices
  basis = tuple(backbone2ix[l] for l in basis)

  return Const(
    base_ids=base_ids,
    msa_ids=msa_ids,
    base_atom_order=base_atom_order,
    backbone2ix=backbone2ix,
    atom_type_num=atom_type_num,
    basis=basis,
  )

RNA_const = get_constants()