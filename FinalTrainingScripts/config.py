# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model config."""

import copy
import ml_collections

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'


def model_config(name: str) -> ml_collections.ConfigDict:
  """Get the ConfigDict of a CASP14 model."""

  if name not in CONFIG_DIFFS:
    raise ValueError(f'Invalid model name {name}.')
  cfg = copy.deepcopy(CONFIG)
  cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
  return cfg

def optim_config(name):
  if name not in OPTIMISER_CONFIG:
    raise ValueError(f'Invalid model name {name}.')
  return OPTIMISER_CONFIG[name]


CONFIG_DIFFS = {
    'model_1': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.1.1
        'data.common.max_extra_msa': 5120,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True
    },
    'model_2': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.1.2
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True
    },
    'model_3': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.1
        'data.common.max_extra_msa': 5120,
    },
    'model_4': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.2
        'data.common.max_extra_msa': 5120,
    },
    'model_5': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.3
    },

    # The following models are fine-tuned from the corresponding models above
    # with an additional predicted_aligned_error head that can produce
    # predicted TM-score (pTM) and predicted aligned errors.
    'model_1_ptm': {
        'data.common.max_extra_msa': 5120,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        # 'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_2_ptm': {
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_3_ptm': {
        'data.common.max_extra_msa': 5120,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_4_ptm': {
        'data.common.max_extra_msa': 5120,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_5_ptm': {
        'model.heads.predicted_aligned_error.weight': 0.1
    }
}


OPTIM_SETTING_1 = {
    'optim_type' : 'SGD',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'rna':{
            'lr':1e-3,
        },
        'structure_module':{# conditions for placing params in groups in order
            # checks if the param is within the submodule structure_module
            'lr':2e-4,
            'momentum':0,
            'dampening':0,
            'nesterov':False,
            # 'warm_up':128,
        }, 
        # 'evoformer_iteration':{
        #     'lr':2e-4,
        #     # 'warm_up':128,
        # },
        'default':{
            'lr':1e-4,
            'weight_decay':0,
        }
    },
}

OPTIM_SETTING_2 = {
    'optim_type' : 'SGD',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'rna':{
            'lr':1e-3,
        },
        'structure_module':{# conditions for placing params in groups in order
            # checks if the param is within the submodule structure_module
            'lr':4e-4,
            'momentum':0,
            'dampening':0,
            'nesterov':False,
            # 'warm_up':128,
        }, 
        # 'evoformer_iteration':{
        #     'lr':2e-4,
        #     # 'warm_up':128,
        # },
        'default':{
            'lr':3e-4,
            'weight_decay':0,
        }
    },
}

OPTIM_SETTING_3 = {
    'optim_type' : 'AdamW',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'default':{
            'lr':3e-4,
            'weight_decay':0,
        }
    },
}

OPTIM_SETTING_4 = {
    'optim_type' : 'AdamW',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'default':{
            'lr':1e-3,
            'weight_decay':0,
        }
    },
    'scheduler':{
        'class':'torch.optim.lr_scheduler.LambdaLR',
        'kwargs':"{'lr_lambda':[lambda epoch: 1 - 0.9 * epoch / n_epoch]}"
    }
}

OPTIM_SETTING_5 = {
    'optim_type' : 'AdamW',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'default':{
            'lr':1e-3,
            'weight_decay':0,
        }
    },
    'scheduler':{
        'num_call_per_epoch':300,
        'class':'torch.optim.lr_scheduler.LambdaLR',
        'kwargs':"{'lr_lambda':[lambda t: 0.55 + 0.45*np.cos(40*t)]}"
    }
}

OPTIM_SETTING_6 = {
    'optim_type' : 'AdamW',# torch.optim module name e.g. AdamW
    'optim_groups':{
        'default':{
            'lr':1e-3,
            'weight_decay':0,
        }
    },
}

OPTIMISER_CONFIG = {
    'far_lr':OPTIM_SETTING_1,
    'close_lr':OPTIM_SETTING_2,
    'default':OPTIM_SETTING_3,
    'decay':OPTIM_SETTING_4,
    'cosine':OPTIM_SETTING_5,
    'original':OPTIM_SETTING_6,
}


TRAINING_FEAT = {
    'feat': {
        'hhblits_profile': [NUM_RES, None],
        'rna_msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
        'rna_target_feat': [NUM_RES, None],
        'msa': [NUM_MSA_SEQ, NUM_RES],
        'deletion_matrix': [NUM_MSA_SEQ, NUM_RES],
        ###
        'aatype': [NUM_RES],
        # 'all_atom_mask': [NUM_RES, None],
        # 'all_atom_positions': [NUM_RES, None, None],
        # 'alt_chi_angles': [NUM_RES, None],
        # 'atom14_alt_gt_exists': [NUM_RES, None],
        # 'atom14_alt_gt_positions': [NUM_RES, None, None],
        # 'atom14_atom_exists': [NUM_RES, None],
        # 'atom14_atom_is_ambiguous': [NUM_RES, None],
        # 'atom14_gt_exists': [NUM_RES, None],
        # 'atom14_gt_positions': [NUM_RES, None, None],
        # 'atom37_atom_exists': [NUM_RES, None],
        'backbone_affine_mask': [NUM_RES],
        'backbone_affine_tensor': [NUM_RES, None],
        'bert_mask': [NUM_MSA_SEQ, NUM_RES],
        # 'chi_angles': [NUM_RES, None],
        # 'chi_mask': [NUM_RES, None],
        'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
        'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
        # 'extra_msa_row_mask': [NUM_EXTRA_SEQ],
        # 'is_distillation': [],
        'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
        'msa_mask': [NUM_MSA_SEQ, NUM_RES],
        'msa_row_mask': [NUM_MSA_SEQ],
        'pseudo_beta': [NUM_RES, None],
        'pseudo_beta_mask': [NUM_RES],
        # 'random_crop_to_size_seed': [None],
        'residue_index': [NUM_RES],
        # 'residx_atom14_to_atom37': [NUM_RES, None],
        # 'residx_atom37_to_atom14': [NUM_RES, None],
        # 'resolution': [],
        # 'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
        # 'rigidgroups_group_exists': [NUM_RES, None],
        # 'rigidgroups_group_is_ambiguous': [NUM_RES, None],
        # 'rigidgroups_gt_exists': [NUM_RES, None],
        # 'rigidgroups_gt_frames': [NUM_RES, None, None],
        'seq_length': [],
        'seq_mask': [NUM_RES],
        'target_feat': [NUM_RES, None],
        # 'template_aatype': [NUM_TEMPLATES, NUM_RES],
        # 'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
        # 'template_all_atom_positions': [
        #     NUM_TEMPLATES, NUM_RES, None, None],
        # 'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
        # 'template_backbone_affine_tensor': [
        #     NUM_TEMPLATES, NUM_RES, None],
        # 'template_mask': [NUM_TEMPLATES],
        # 'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
        # 'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
        # 'template_sum_probs': [NUM_TEMPLATES, None],
        'true_msa': [NUM_MSA_SEQ, NUM_RES]
    },
    'fixed_size': True,
    'subsample_templates': False,  # We want top templates.
    'masked_msa_replace_fraction': 0.15,
    'max_msa_clusters': 64,
    # 'max_templates': 4,
    'num_ensemble': 1,
    'crop_size':160,
}

VAL_TRACK_COORDS = [# all shorter than 160 -- the crop size
    # short (<30) very high res (<1.5A)
    '1L2X-A', '1F27-A', '3SJ2-A', '5AY2-B', '5L00-B', '2Q1O-C', '4U3L-A',
    # long (>100) and high res (<2.5A)
    '4V9F-9', '4YBB-DB', '4Y4O-1B', '7O7Y-B7', '6AZ3-8', '6YSI-5', '6S0Z-B',
    # short (< 100) and high res (<2.5A)
    '2V3C-N', '5AOX-C', '4JNG-L', '1MJI-D', '5NS3-C', '3K1V-A', '6V9B-B', 
    '6V9B-D', '4KQ0-E', '3SZX-A', '1FEU-B', '4PY5-B', '2P7E-D', '4ILL-R', 
    '2AB4-B', '7EAG-D', '7EAG-C', '2QKB-C', '3SZX-B', '7EAG-E', '4KQ0-B', 
    # long, very low res (4A < res < 20A)
    '2J37-A', '4V6U-B3',
    # short (<20), very low res (4A < res < 20A)
    '4CXG-c', '3J0E-e', '6A5R-P', '3AOI-S',
]

# TRAIN_TRACK_COORDS = [
#     '2XLK-C', '6IV6-G', '5H3U-C', '1NEM-A', '1R3E-E', '1VQN-4', '2R1S-A', 
#     '3NNC-B', '3BSB-C', '6F4H-D', '2KF0-A', '1NJP-5', '6H9H-D', '3CJZ-A', 
#     '5WQ1-A', '6RA4-L', '5DV7-B', '2M22-A', '5WNT-a', '4NH5-B', '1N38-B', 
#     '6RT4-D', '4ERD-D', '6T0V-R', '3J0O-d', '5NDH-A', '1AL5-B', '4Z7K-C', 
#     '1YYK-D', '1BVJ-A', '2G4B-B', '1ZX7-D', '3HTX-C', '5JC7-X', '6XBU-E', 
#     '4IQS-E', '3K62-B', '2D19-A', '5VCI-B', '2L41-B', '6ID1-H', '4M6D-H', 
#     '5Y85-D', '4PRF-B', '6HAG-A', '6S8B-V', '6NY2-B', '4PR6-B', '4WQ1-3K', 
#     '3OXE-A', '3P22-A', '7JJU-B', '3EGZ-B', '6UZ7-8', '5LZD-X', '4L81-A', 
#     '1MZP-B', '2FEY-A', '6SGC-23', '4V5Z-BY', '4ADX-8', '2WW9-D', '1YMO-A', 
#     '6SY6-D', '3PU1-R', '2GTT-W', '5KQE-A', '7LJY-B', '2QWY-C', '7C7L-C', 
#     '7L49-E', '4Y1M-B', '6ZJ3-LN', '2KU0-A', '2LU0-A', '6NUE-H', '4V7H-A7', 
#     '1P5M-A', '2NBX-A', '2N6S-A', '4WF9-X', '6QX9-1', '4V8M-AA', '7NHK-A', 
#     '6Q97-4'
# ]

TRAIN_TRACK_COORDS = [
    '2XLK-C','5H3U-C','1R3E-E','2R1S-A','3BSB-C','2KF0-A','6H9H-D','5WQ1-A',
    '5DV7-B','5WNT-a','1N38-B','4ERD-D','3J0O-d','1AL5-B','1YYK-D','2G4B-B',
    '3HTX-C','6XBU-E','3K62-B','5VCI-B','6ID1-H','5Y85-D','6HAG-A','6NY2-B',
    '4WQ1-3K','3P22-A','3EGZ-B','5LZD-X','1MZP-B','6SGC-23','4ADX-8','1YMO-A',
    '3PU1-R','5KQE-A','2QWY-C','7L49-E','6ZJ3-LN','2LU0-A','4V7H-A7','2NBX-A',
    '4WF9-X','4V8M-AA','6Q97-4','3SN2-B','2QUX-F','6YML-A','6YL5-I','1JID-B',
    '5DEA-C','4NLF-A','3B5S-B','6UVN-M','2YHM-K','2HW8-B','4QYZ-L','1KOG-O',
    '3WBM-Y','6NE0-M','6F4G-C','7CYQ-I','2ZY6-A','5H9F-L','6H0R-A','5XLO-K',
    '6S8B-U','6D12-C','5DEA-A','6S8B-V','7KRP-T','1HMH-A','6DTD-C','2BH2-D',
    '3HJY-D','6SY6-D','6LNB-M','359D-B','2OZB-C','5NFV-B','7CXM-I','1OOA-C',
    '7KRN-T','5DE8-A','3CIY-C','6UGI-A','1I6U-C','5KK5-B','5U0A-K','6C66-J',
    '3CIY-D','6VQV-L','6DU4-B','7KRN-P','6TQN-R','6VFF-C','3R9W-B','4BW0-A',
    '3P22-A','7DMQ-B','7KHA-J','6B44-M','1C9S-W','6PIJ-1','3VJR-B','3GS5-A',
    '5UNE-B','3P22-A', '2QWY-C', '5Y85-D', '3PU1-R', '1MZP-B'
]

# Randomly drawn subset of 500 training codes for tracking metrics, 
# fixed so that it can compare fairly
TRACK_TRAIN_METRICS = {
    '2I7Z-B', '5E81-2K', '2WW9-G', '3J0E-B', '3J16-K', '7MSF-R', '4R3I-B', 
    '4OAV-A', '2LUB-A', '5T3K-A', '3BT7-D', '4ADX-0', '3PDR-X', '6XLL-R', 
    '4AM3-E', '5LQO-A', '5MS0-R', '6I1W-B', '6NOA-A', '1JU1-A', '1N32-Y', 
    '2QK9-B', '6N6E-D', '3HSB-X', '4DB2-E', '6LQP-3A', '5JPQ-3', '2A0P-B', 
    '2GV4-A', '1TJZ-A', '7O81-AJ', '7KRP-P', '3WBM-X', '4QIK-C', '2ET8-B', 
    '6EEN-G', '5VH8-B', '2JLW-D', '1T0K-D', '2GVO-A', '2MGZ-C', '4A4R-A', 
    '5V16-A', '5JU8-AY', '1YSH-B', '3TS2-U', '4GV6-C', '3BX2-D', '2OIU-P', 
    '2XB2-R', '4AL5-B', '1JZV-A', '4EYA-h', '6EXN-2', '280D-B', '1QZC-A', 
    '1JO7-A', '2Y9H-N', '6PZQ-J', '7LJY-B', '2L2K-A', '6UU3-333', '2VAL-C', 
    '5LM7-G', '7ABG-2', '7B0Y-A', '4JF2-A', '5Y88-x', '4G0A-F', '4H8K-C', 
    '4RUM-A', '5JBJ-Y', '6AZ3-4', '6ND4-2', '3QJJ-Q', '2DB3-E', '4FRN-B', 
    '6V5B-D', '6D90-4', '3ERC-F', '3OL9-B', '2JYM-A', '1ZC5-A', '3J7P-S2', 
    '3HAX-E', '6ICZ-H', '1IKD-A', '2KF0-A', '2JLV-C', '6W5C-B', '6G19-X', 
    '2BH2-D', '5HBY-A', '5MGA-B', '6D92-C', '438D-A', '1YNE-A', '4P5J-A', 
    '255D-A', '6ND4-0', '3P22-A', '4PR6-B', '6ZQA-D4', '4TYW-B', '1RPU-D', 
    '6Z1P-Bb', '3J79-C', '4G9Z-F', '2P7E-A', '1IL2-C', '3JB9-C', '6LQS-3A', 
    '1UVJ-F', '4C8Z-C', '7NSI-AV', '2KY2-A', '5K8H-A', '4NFQ-A', '1CVJ-Q', 
    '1DQF-A', '1G2J-A', '1EVV-A', '5VSU-I', '6WB2-D', '4UE5-A', '6TZ2-M', 
    '2GTT-X', '2AZ0-D', '6PIJ-1', '1B36-A', '3TD0-B', '1OW9-A', '2O32-A', 
    '283D-A', '6JQ6-U', '2MF0-G', '5T5H-C', '1G1X-I', '7O7Y-B8', '3KTW-C', 
    '6GKH-X', '4G0A-G', '6E0O-C', '4WTM-P', '4EYA-g', '1F85-A', '3HJF-Y', 
    '7O7Y-AT', '6YYT-Q', '6JQ5-B', '485D-B', '3DEG-G', '5JUP-EC', '2I2Y-B', 
    '3K5Z-B', '6ZMI-S2', '435D-C', '6DTD-C', '1JTJ-A', '6ZQD-D4', '1ESY-A', 
    '6Z1P-BA', '6SZV-R', '3OG8-C', '4V8M-BB', '3QGC-B', '2WWB-D', '5KMZ-A', 
    '2IRN-B', '1K2G-A', '2HGH-B', '1OQ0-A', '3AVT-T', '2WW9-D', '4PEI-X', 
    '6XA1-Bv', '7OF0-B', '7DMQ-B', '2V7R-A', '6YYT-P', '5WNU-a', '1LMV-A', 
    '1H4S-T', '1FOQ-B', '6G1X-X', '2DD3-B', '2L41-B', '5DEA-A', '7D6Z-3', 
    '4M59-D', '5XWP-C', '6RJA-D', '5U3G-B', '5LYS-B', '4P95-A', '6CFJ-1x', 
    '2G1W-A', '2LHP-A', '1TFW-E', '4TS2-X', '1Z30-A', '4V99-C2', '5N8L-C', 
    '6SPC-a', '217D-A', '2W89-A', '4QJH-F', '4BWM-G', '6ZDU-C', '6ZM6-B', 
    '2JLT-A', '1M5L-A', '5B2T-A', '5AWH-C', '6RA4-L', '1ZN1-C', '6FZ0-A', 
    '7KFN-B', '1TBK-A', '4GL2-C', '7D6Z-2', '6PMI-3', '464D-A', '5DV7-A', 
    '3J7A-A', '5K77-X', '4XNR-X', '4CXG-1', '2NZ4-Q', '1UVI-D', '3ADD-C', 
    '4RCJ-B', '3J0O-4', '7A01-E1', '6Q98-4', '3SKI-A', '2GIO-A', '6S8B-V', 
    '4V8Y-CN', '1K6H-A', '1FEU-C', '3F2X-X', '4OJI-A', '4P3U-C', '4NLF-A', 
    '4U35-B', '4V5Z-BN', '4QK9-A', '2F4V-Z', '6DN2-X', '4QQB-P', '6KUR-V', 
    '2N0R-A', '3J45-5', '4PJO-1', '3IYR-A', '2LKR-A', '6W62-B', '4V6W-A5', 
    '7BV2-T', '7NHN-a', '2IXY-A', '6D8A-C', '6YML-A', '6YRQ-E', '6U6Y-E', 
    '4GV6-B', '422D-A', '2L94-A', '6MCB-B', '2L8W-A', '1YVP-C', '1N66-A', 
    '2E9R-A', '4WAN-B', '5GAN-W', '5X70-E', '5DDO-B', '3J0O-2', '1ZC8-H', 
    '1YZD-B', '5E3H-C', '5H0R-H', '421D-B', '6RT5-A', '4WRT-V', '3NKB-B', 
    '3J45-2', '2N6W-A', '3J45-3', '6TW1-V', '5MS0-H', '5EV1-B', '5H1L-B', 
    '4V7H-B5', '1SAQ-A', '6E7L-A', '2OM7-F', '3NDB-M', '6K32-p', '1NTA-A', 
    '1ML5-a', '1ZIG-A', '2MXK-A', '2HEM-A', '1RMN-A', '4KR9-M', '7JQC-I', 
    '1MVR-A', '2MQV-B', '5H3U-C', '2MQT-A', '2LC8-A', '2XS2-B', '5EW4-B', 
    '4PMI-A', '6L5N-C', '2A43-A', '3IGI-A', '4WTL-T', '4KZY-i', '1KOG-O', 
    '4K31-C', '6SJD-D', '3KOA-C', '1N32-Z', '4QPX-T', '4DR7-b', '6BM4-R', 
    '6N6C-D', '2MI0-A', '6EXN-6', '4QVD-H', '4G0A-E', '397D-A', '6GC5-G', 
    '7AQC-A', '4M4O-B', '5EN1-B', '2F8S-C', '1C9S-W', '6Q8V-A', '4GZY-R', 
    '2KPD-A', '1B7F-Q', '1Q8N-A', '2KRV-A', '2L3J-B', '3OL9-C', '6G90-1', 
    '1N38-B', '4U34-B', '6ZJ3-LF', '5GAP-V', '5Y36-B', '6OW3-I', '1AJL-A', 
    '2AGN-D', '6S8B-U', '6SGC-23', '1OOA-C', '6UFM-B', '1T0E-C', '4V8Z-CX', 
    '6RXX-C2', '2IRN-A', '5IT9-I', '1K8W-B', '4PQU-T', '5V3F-A', '5A8L-P', 
    '3IZZ-D', '1MMS-C', '4V8M-AA', '3SIV-F', '4EYA-f', '2F8T-C', '4V4B-B3', 
    '7LYF-A', '2N6X-A', '3AVW-T', '4H5P-F', '1Y26-X', '1BAU-A', '1S9S-A', 
    '5NDH-A', '6S0X-a', '6MCI-A', '2D18-A', '3NVI-F', '1NC0-A', '3J0O-d', 
    '6QW6-4', '3SSF-A', '3MOJ-A', '5ZQ1-B', '7EU9-B', '2O3V-B', '3CZW-X', 
    '419D-D', '4GKJ-W', '6QX9-2', '5MPG-B', '7JJU-A', '4GZZ-R', '2GJW-L', 
    '1QZW-F', '7D7V-A', '5KH8-A', '5XTM-B', '2AO5-D', '1ROQ-A', '1D6K-B', 
    '1WWE-B', '4PWD-T', '7ABG-6', '1U6B-B', '2F87-A', '1S76-R', '2K7E-A', 
    '6RXU-C1', '4Z31-D', '6RT7-E', '5U0Q-C', '2OE5-B', '5XXB-4', '4WTM-T', 
    '6ZYM-6', '5U9B-B', '4W90-C', '4V5Z-AC', '6TH6-Aa', '4NKU-D', '7MKY-A', 
    '2G91-B', '3JB9-N', '1H1K-K', '6CYT-N', '3PKM-R', '6ZJ3-LB', '7JL3-X', 
    '4Z4D-D', '2ERR-B', '6L0Y-A', '3VNV-G', '3NVK-K', '6ZQG-D5', '6IZP-A', 
    '3NMA-C', '1S03-A', '5DCV-D', '6W64-B', '4JZU-C', '3GTM-M', '2BX2-R', 
    '4TYY-B', '3FTE-D', '6O1O-M', '4IQS-E', '6WB0-C', '6N8I-B', '1RNA-A', 
    '6AZ3-2', '2UWM-D', '5VW1-C', '2M24-A', '7A5P-2', '7A3Y-B', '1VQN-4', 
    '5JC7-Y', '3V71-B', '2LV0-A', '6JOO-B', '6BHJ-E', '4PEH-Z', '6DTA-D', 
    '5XPG-G', '4JVY-D', '7L0Z-G', '1ZIH-A', '4CSF-u', '4ERD-D', '7DCO-B', 
    '5DV7-B', '1H1K-J', '2YHM-K', '4E58-E', '2IHX-B', '4JRT-A', '6E8U-B', 
    '6MXQ-A', '1Z2J-A', '6U6Y-G', '3NMU-K', '5LSN-A', '4XW0-A', '2XEB-B', 
    '2KRY-A', '1KOD-B', '2KD8-A'
}

TRACK_CODES = {
    'train_metrics':TRACK_TRAIN_METRICS,
    'train_coords_to_track':TRAIN_TRACK_COORDS,
    'val_coords_to_track':VAL_TRACK_COORDS
}

CONFIG = ml_collections.ConfigDict({
    'data': {
        'training': {
            'constant':TRAINING_FEAT,
        },
        'common': {
            'masked_msa': {
                'profile_prob': 0.1,
                'same_prob': 0.1,
                'uniform_prob': 0.1
            },
            'max_extra_msa': 128,#1024
            'msa_cluster_features': True,
            'num_recycle': 3,
            'reduce_msa_clusters_by_max_templates': False,
            'resample_msa_in_recycling': True,
            'template_features': [
                'template_all_atom_positions', 'template_sum_probs',
                'template_aatype', 'template_all_atom_masks',
                'template_domain_names'
            ],
            'unsupervised_features': [
                'aatype', 'residue_index', 'sequence', 'msa', 'domain_name',
                'num_alignments', 'seq_length', 'between_segment_residues',
                'deletion_matrix'
            ],
            'use_templates': False,
        },
        'eval': {
            'feat': {
                'aatype': [NUM_RES],
                'all_atom_mask': [NUM_RES, None],
                'all_atom_positions': [NUM_RES, None, None],
                'alt_chi_angles': [NUM_RES, None],
                'atom14_alt_gt_exists': [NUM_RES, None],
                'atom14_alt_gt_positions': [NUM_RES, None, None],
                'atom14_atom_exists': [NUM_RES, None],
                'atom14_atom_is_ambiguous': [NUM_RES, None],
                'atom14_gt_exists': [NUM_RES, None],
                'atom14_gt_positions': [NUM_RES, None, None],
                'atom37_atom_exists': [NUM_RES, None],
                'backbone_affine_mask': [NUM_RES],
                'backbone_affine_tensor': [NUM_RES, None],
                'bert_mask': [NUM_MSA_SEQ, NUM_RES],
                'chi_angles': [NUM_RES, None],
                'chi_mask': [NUM_RES, None],
                'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa_row_mask': [NUM_EXTRA_SEQ],
                'is_distillation': [],
                'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
                'msa_mask': [NUM_MSA_SEQ, NUM_RES],
                'msa_row_mask': [NUM_MSA_SEQ],
                'pseudo_beta': [NUM_RES, None],
                'pseudo_beta_mask': [NUM_RES],
                'random_crop_to_size_seed': [None],
                'residue_index': [NUM_RES],
                'residx_atom14_to_atom37': [NUM_RES, None],
                'residx_atom37_to_atom14': [NUM_RES, None],
                'resolution': [],
                'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
                'rigidgroups_group_exists': [NUM_RES, None],
                'rigidgroups_group_is_ambiguous': [NUM_RES, None],
                'rigidgroups_gt_exists': [NUM_RES, None],
                'rigidgroups_gt_frames': [NUM_RES, None, None],
                'seq_length': [],
                'seq_mask': [NUM_RES],
                'target_feat': [NUM_RES, None],
                'template_aatype': [NUM_TEMPLATES, NUM_RES],
                'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
                'template_all_atom_positions': [
                    NUM_TEMPLATES, NUM_RES, None, None],
                'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
                'template_backbone_affine_tensor': [
                    NUM_TEMPLATES, NUM_RES, None],
                'template_mask': [NUM_TEMPLATES],
                'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
                'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
                'template_sum_probs': [NUM_TEMPLATES, None],
                'true_msa': [NUM_MSA_SEQ, NUM_RES]
            },
            'fixed_size': True,
            'subsample_templates': False,  # We want top templates.
            'masked_msa_replace_fraction': 0.15,
            'max_msa_clusters': 512,
            'max_templates': 4,
            'num_ensemble': 1,
        },
    },
    'model': {
        'embeddings_and_evoformer': {
            'evoformer_num_block': 48,
            'evoformer': {
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': .0,#0.15,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'outer_product_mean': {
                    'chunk_size': 128,
                    'dropout_rate': 0.0,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_attention_starting_node': {
                    'dropout_rate': .0,#0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': .0,#0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': .0,#0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': .0,#0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                }
            },
            'extra_msa_channel': 64,
            'extra_msa_stack_num_block': 4,
            'max_relative_feature': 32,
            'msa_channel': 256,
            'pair_channel': 128,
            'prev_pos': {
                'min_bin': 3.0,#3.25,
                'max_bin': 35.0,#20.75,
                'num_bins': 15
            },
            'recycle_features': True,
            'recycle_pos': True,
            'seq_channel': 384,
            'template': {
                'attention': {
                    'gating': False,
                    'key_dim': 64,
                    'num_head': 4,
                    'value_dim': 64
                },
                'dgram_features': {
                    'min_bin': 3.25,
                    'max_bin': 50.75,
                    'num_bins': 39
                },
                'embed_torsion_angles': False,
                'enabled': False,
                'template_pair_stack': {
                    'num_block': 2,
                    'triangle_attention_starting_node': {
                        'dropout_rate': .0,#0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'value_dim': 64
                    },
                    'triangle_attention_ending_node': {
                        'dropout_rate': .0,#0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_column',
                        'shared_dropout': True,
                        'value_dim': 64
                    },
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': .0,#0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'triangle_multiplication_incoming': {
                        'dropout_rate': .0,#0.25,
                        'equation': 'kjc,kic->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    }
                },
                'max_templates': 4,
                'subbatch_size': 128,
                'use_template_unit_vector': False,
            }
        },
        'global_config': {
            'deterministic': False,
            'subbatch_size': 4,
            'use_remat': False,
            'zero_init': False,
            'msa_n_token':23,
            'rna_msa_n_token':7,
        },
        'heads': {
            'distogram': {
                'first_break': 3.0,#2.3125,
                'last_break': 35.0,#21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            # 'predicted_aligned_error': {
            #     # `num_bins - 1` bins uniformly space the
            #     # [0, max_error_bin A] range.
            #     # The final bin covers [max_error_bin A, +infty]
            #     # 31A gives bins with 0.5A width.
            #     'max_error_bin': 31.,
            #     'num_bins': 64,
            #     'num_channels': 128,
            #     'filter_by_resolution': True,
            #     'min_resolution': 0.1,
            #     'max_resolution': 3.0,
            #     'weight': 0.0,
            # },
            # 'experimentally_resolved': {
            #     'filter_by_resolution': True,
            #     'max_resolution': 3.0,
            #     'min_resolution': 0.1,
            #     'weight': 0.01
            # },
            'structure_module': {
                'num_layer': 8,
                'fape': {
                    'clamp_distance': 10.0,
                    'clamp_type': 'relu',
                    'loss_unit_distance': 10.0
                },
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'compute_in_graph_metrics': True,
                'dropout': 0.0,#0.1,
                'num_channel': 384,
                'num_head': 12,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 10.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5,
                    'length_scale': 10.,
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 1.0
            },
            # 'predicted_lddt': {
            #     'filter_by_resolution': True,
            #     'max_resolution': 3.0,
            #     'min_resolution': 0.1,
            #     'num_bins': 50,
            #     'num_channels': 128,
            #     'weight': 0.01
            # },
            'masked_msa': {
                'num_output': 23,
                'rna_num_output':7,
                'weight': 2.0
            },
        },
        'num_recycle': 3,
        'resample_msa_in_recycling': True
    },
})
