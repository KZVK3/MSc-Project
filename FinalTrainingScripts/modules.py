import torch, numpy as np
nn = torch.nn
import collections, numbers, functools, tree
import pytorch_lightning as pl
import os, gzip, json, glob
from Bio.SVDSuperimposer import SVDSuperimposer

from torch.utils.checkpoint import checkpoint_sequential, checkpoint
# if torch.cuda.is_available():
#   import deepspeed
#   checkpoint = deepspeed.checkpointing.checkpoint
# else:
#   checkpoint = lambda x:x

class VarianceScaling:
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  """
  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    op = {'fan_in':lambda x:x[0], 'fan_out':lambda x:x[1], 'fan_avg':lambda x:0.5*sum(x)}
    self.fn = op[mode]
    if scale < 0.0: raise ValueError('`scale` must be a positive float.')
    d = {
      'normal': lambda c, sh, dt, dv: torch.randn(*sh, dtype=dt, device=dv)*((scale/c)**0.5), 
      'truncated_normal': lambda c, sh, dt, dv: (torch.sqrt(-2*torch.log(
        torch.rand(*sh, dtype=dt, device=dv)*(1-np.exp(-2)) + np.exp(-2))) * torch.cos(
          2*np.pi*torch.rand(*sh, dtype=dt, device=dv)))*scale/c,# box-muller
      'uniform': lambda c, sh, dt, dv: (torch.rand(*sh, dtype=dt, device=dv)-0.5)*2*((3.*scale/c)**0.5)
    }
    self.dist = d[distribution.lower()]
    self.shape2fan = [lambda sh:(1,1), lambda sh:(sh[0],sh[0]), lambda sh:(sh[0],sh[1])]

  def __call__(self, shape, dtype, device):
    if len(shape) < len(self.shape2fan):
      fan_in, fan_out = self.shape2fan[len(shape)](shape)
    else:
      fan_in, fan_out = shape[-2] * np.prod(shape[:-2]), shape[-1] * np.prod(shape[:-2])
    return self.dist(max(1.0, self.fn([fan_in, fan_out])), shape, dtype, device)

def glorot_uniform(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')(shape, dtype, device)

def glorot_normal(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')(shape, dtype, device)

def lecun_uniform(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')(shape, dtype, device)

def lecun_normal(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')(shape, dtype, device)

def he_uniform(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform')(shape, dtype, device)

def he_normal(*shape, dtype=torch.float, device=None):
  return VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')(shape, dtype, device)

def makeLinear(num_input: int,
               num_output: int,
               dtype: torch.dtype,
               device: torch.device,
               initializer: str = 'linear',
               use_bias: bool = True,
               bias_init: float = 0.,
  ):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  """
  init = {'linear': lambda *s,dtype=dtype: lecun_normal(*s, dtype=dtype, device=device), 
          'relu': lambda *s,dtype=dtype: he_normal(*s, dtype=dtype, device=device), 
          'zeros': lambda *s,dtype=dtype: torch.zeros(*s, dtype=dtype, device=device)}
  assert initializer in init
  
  lin = nn.Linear(num_input, num_output, bias=use_bias, device=device)
  lin.weight.data = init[initializer](num_output, num_input, dtype=dtype)
  
  if use_bias: lin.bias.data.fill_(bias_init).to(dtype)
  return lin


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Masked mean.
  """
  if drop_mask_channel:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(value_shape)

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(axis, collections.Iterable), (
      'axis needs to be either an iterable, integer or "None"')

  broadcast_factor = 1.
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size

  return (mask * value).sum(axis) / (mask.sum(axis) * broadcast_factor + eps)


class Attention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Multihead attention.
  """
  def __init__(self, config, global_config, output_dim):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.output_dim = output_dim
    self.forward = self.init_parameters
  
  def init_parameters(self, q_data, m_data, bias, nonbatched_bias=None):
    dt = q_data.dtype

    key_dim = self.config.get('key_dim', int(q_data.size(-1)))
    val_dim = self.config.get('value_dim', int(m_data.size(-1)))
    N_head = self.config.num_head
    
    assert key_dim % N_head == 0
    assert val_dim % N_head == 0
    self.key_dim = key_dim // N_head
    val_dim = val_dim // N_head

    p = lambda d_dim, prj_dim:nn.Parameter(glorot_uniform(d_dim, N_head, prj_dim, dtype=dt, device=self.global_config.device))
    self.query_w = p(q_data.size(-1), self.key_dim)
    self.key_w = p(m_data.size(-1), self.key_dim)
    self.value_w = p(m_data.size(-1), val_dim)

    if self.config.gating:
      self.gating_w = nn.Parameter(torch.zeros(q_data.size(-1), N_head, val_dim, dtype=dt, device=self.global_config.device))
      self.gating_b = nn.Parameter(torch.ones(N_head, val_dim, dtype=dt, device=self.global_config.device))
    
    init = torch.zeros if self.global_config.zero_init else glorot_uniform
    self.output_w = nn.Parameter(init(N_head, val_dim, self.output_dim, dtype=dt, device=self.global_config.device))
    self.output_b = nn.Parameter(torch.zeros(self.output_dim, dtype=dt, device=self.global_config.device))

    self.forward = self.go
    return self(q_data, m_data, bias, nonbatched_bias)

  def go(self, q_data, m_data, bias, nonbatched_bias=None):
    """
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].

    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_dim**(-0.5)
    k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w)
    v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w)

    logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias
    if nonbatched_bias is not None: logits += nonbatched_bias[None,...]

    weights = torch.functional.F.softmax(logits, dim=-1)
    weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.config.gating:
      gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w)
      gate_values = torch.sigmoid(gate_values + self.gating_b)
      weighted_avg *= gate_values

    output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    return output


class GlobalAttention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
  """
  def __init__(self, config, global_config, output_dim):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.output_dim = output_dim
    self.forward = self.init_parameters
  
  def init_parameters(self, q_data, m_data, q_mask, bias):
    dt = q_data.dtype

    key_dim = self.config.get('key_dim', int(q_data.size(-1)))
    val_dim = self.config.get('value_dim', int(m_data.size(-1)))
    N_head = self.config.num_head
    
    assert key_dim % N_head == 0
    assert val_dim % N_head == 0
    self.key_dim = key_dim // N_head
    val_dim = val_dim // N_head

    self.query_w = nn.Parameter(glorot_uniform(q_data.size(-1), N_head, self.key_dim, dtype=dt, device=self.global_config.device))
    self.key_w = nn.Parameter(glorot_uniform(m_data.size(-1), self.key_dim, dtype=dt, device=self.global_config.device))
    self.value_w = nn.Parameter(glorot_uniform(m_data.size(-1), val_dim, dtype=dt, device=self.global_config.device))

    init = torch.zeros if self.global_config.zero_init else glorot_uniform
    self.output_w = nn.Parameter(init(N_head, val_dim, self.output_dim, dtype=dt, device=self.global_config.device))
    self.output_b = nn.Parameter(torch.zeros(self.output_dim, dtype=dt, device=self.global_config.device))

    if self.config.gating:
      self.gating_w = nn.Parameter(torch.zeros(q_data.size(-1), N_head, val_dim, dtype=dt, device=self.global_config.device))
      self.gating_b = nn.Parameter(torch.ones(N_head, val_dim, dtype=dt, device=self.global_config.device))

    self.forward = self.go
    return self(q_data, m_data, q_mask, bias)

  def go(self, q_data, m_data, q_mask, bias):
    """
    Arguments:
      q_data: A tensor of queries with size [batch_size, N_queries,
        q_channels]
      m_data: A tensor of memories from which the keys and values
        projected. Size [batch_size, N_keys, m_channels]
      q_mask: A binary mask for q_data with zeros in the padded sequence
        elements and ones otherwise. Size [batch_size, N_queries, q_channels]
        (or broadcastable to this shape).
      bias: A bias for the attention.

    Returns:
      A float32 tensor of size [batch_size, N_queries, output_dim].
    """
    
    v = torch.einsum('bka,ac->bkc', m_data, self.value_w)

    q_avg = mask_mean(q_mask, q_data, axis=1)

    q = torch.einsum('ba,ahc->bhc', q_avg, self.query_w) * self.key_dim**(-0.5)
    k = torch.einsum('bka,ac->bkc', m_data, self.key_w)
    bias = (1e9 * (q_mask[:, None, :, 0] - 1.))
    logits = torch.einsum('bhc,bkc->bhk', q, k) + bias
    weights = torch.functional.F.softmax(logits, dim=-1)
    weighted_avg = torch.einsum('bhk,bkc->bhc', weights, v)

    if self.config.gating:
      gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w)
      gate_values = torch.sigmoid(gate_values + self.gating_b)
      weighted_avg = weighted_avg[:, None] * gate_values
      output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    else:
      output = torch.einsum('bhc,hco->bo', weighted_avg, self.output_w) + self.output_b
      output = output[:, None]
    return output


class MSARowAttentionWithPairBias(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  MSA per-row attention biased by the pair representation.

  Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, msa_act, msa_mask, pair_act, is_training=False):
    dt = msa_act.dtype

    self.query_norm = nn.LayerNorm(msa_act.size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)
    self.feat_2d_norm = nn.LayerNorm(pair_act.size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)

    self.feat_2d_weights = nn.Parameter(torch.randn(pair_act.shape[-1], self.config.num_head, 
                                dtype=dt, device=self.global_config.device) / (pair_act.size(-1)**0.5))

    self.attention = Attention(self.config, self.global_config, msa_act.size(-1))

    self.forward = self.go
    return self(msa_act, msa_mask, pair_act, is_training)

  def go(self, msa_act, msa_mask, pair_act, is_training=False):
    """
    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      pair_act: [N_res, N_res, c_z] pair representation.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert self.config.orientation == 'per_row'

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = self.query_norm(msa_act)
    pair_act = self.feat_2d_norm(pair_act)
    nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
    
    msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)
    return msa_act


class TriangleAttention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Triangle Attention.

  Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
  Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters
  
  def init_parameters(self, pair_act, pair_mask, is_training=False):
    dt = pair_act.dtype
    self.query_norm = nn.LayerNorm(pair_act.size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)
    self.feat_2d_weights = nn.Parameter(torch.randn(pair_act.shape[-1], self.config.num_head, 
                                dtype=dt, device=self.global_config.device) / (pair_act.size(-1)**0.5))
    self.attention = Attention(self.config, self.global_config, pair_act.shape[-1])

    self.forward = self.go
    return self(pair_act, pair_mask, is_training)

  def go(self, pair_act, pair_mask, is_training=False):
    """
    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    assert len(pair_act.shape) == 3
    assert len(pair_mask.shape) == 2
    assert self.config.orientation in ['per_row', 'per_column']

    if self.config.orientation == 'per_column':
      pair_act = torch.swapaxes(pair_act, -2, -3)
      pair_mask = torch.swapaxes(pair_mask, -1, -2)

    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    pair_act = self.query_norm(pair_act)

    nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
    pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)

    if self.config.orientation == 'per_column':
      pair_act = torch.swapaxes(pair_act, -2, -3)
    return pair_act


class TriangleMultiplication(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Triangle multiplication layer ("outgoing" or "incoming").

  Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
  Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, act, mask, is_training=True):
    dt = act.dtype
    
    channel = act.size(-1)
    intermed_c = self.config.num_intermediate_channel
    
    self.layer_norm_input = nn.LayerNorm(channel, elementwise_affine=True, device=self.global_config.device).to(dt)

    self.left_projection = makeLinear(channel, intermed_c, dt, self.global_config.device)
    self.right_projection = makeLinear(channel, intermed_c, dt, self.global_config.device)

    init = 'zeros' if self.global_config.zero_init else 'linear'
    
    self.left_gate = makeLinear(channel, intermed_c, dt, self.global_config.device, initializer=init, bias_init=1.)
    self.right_gate = makeLinear(channel, intermed_c, dt, self.global_config.device, initializer=init, bias_init=1.)
    
    self.center_layer_norm = nn.LayerNorm(intermed_c, elementwise_affine=True, device=self.global_config.device).to(dt)

    self.output_projection = makeLinear(intermed_c, channel, dt, self.global_config.device, initializer=init)
    self.gating_linear = makeLinear(channel, channel, dt, self.global_config.device, initializer=init, bias_init=1.)

    self.forward = self.go
    return self(act, mask, is_training)

  def go(self, act, mask, is_training=True):
    """
    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Outputs, same shape/type as act.
    """
    mask = mask[..., None]

    act = self.layer_norm_input(act)
    input_act = act.clone()# not sure about this in the origiinal code

    left_proj_act = mask * self.left_projection(act)
    right_proj_act = mask * self.right_projection(act)

    left_gate_values = torch.sigmoid(self.left_gate(act))
    right_gate_values = torch.sigmoid(self.right_gate(act))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    act = torch.einsum(self.config.equation, left_proj_act, right_proj_act)
    act = self.center_layer_norm(act)
    act = self.output_projection(act)

    gate_values = torch.sigmoid(self.gating_linear(input_act))
    act *= gate_values
    return act


class Transition(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Transition layer.

  Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
  Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, act, mask, is_training=True):
    dt = act.dtype
    _, _, nc = act.shape

    num_intermediate = int(nc * self.config.num_intermediate_factor)
    mask = mask[...,None]

    self.input_layer_norm = nn.LayerNorm(nc, elementwise_affine=True, device=self.global_config.device).to(dt)

    init = 'zeros' if self.global_config.zero_init else 'linear'
    self.transition1 = makeLinear(nc, num_intermediate, dt, self.global_config.device, 'relu')
    self.transition2 = makeLinear(num_intermediate, nc, dt, self.global_config.device, init)
    self.transition = nn.Sequential(self.transition1, nn.ReLU(inplace=False), self.transition2)

    self.forward = self.go
    return self(act, mask, is_training)

  def go(self, act, mask, is_training=True):
    """
    Arguments:
      act: A tensor of queries of size [batch_size, N_res, N_channel].
      mask: A tensor denoting the mask of size [batch_size, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      A float32 tensor of size [batch_size, N_res, N_channel].
    """
    mask = mask[...,None]
    act = self.input_layer_norm(act)
    act = self.transition(act)
    return act


class OuterProductMean(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Computes mean outer product.

  Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
  """
  def __init__(self, config, global_config, num_output_channel):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.num_output_channel = num_output_channel
    self.forward = self.init_parameters
  
  def init_parameters(self, act, mask, is_training=True):
    dt = act.dtype
    c = self.config
    channel = act.size(-1)

    self.layer_norm_input = nn.LayerNorm(channel, elementwise_affine=True, device=self.global_config.device).to(dt)

    self.left_projection = makeLinear(channel, c.num_outer_channel, dt, self.global_config.device)
    self.right_projection = makeLinear(channel, c.num_outer_channel, dt, self.global_config.device)

    init = torch.zeros if self.global_config.zero_init else he_normal
    self.output_w = nn.Parameter(init(c.num_outer_channel, 
          c.num_outer_channel, self.num_output_channel, dtype=dt, device=self.global_config.device))
    self.output_b = nn.Parameter(torch.zeros(self.num_output_channel, dtype=dt, device=self.global_config.device))

    self.forward = self.go
    return self(act, mask, is_training)

  def go(self, act, mask, is_training=True):
    """
    Arguments:
      act: MSA representation, shape [N_seq, N_res, c_m].
      mask: MSA mask, shape [N_seq, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair representation, shape [N_res, N_res, c_z].
    """
    mask = mask[..., None]
    act = self.layer_norm_input(act)
    
    left_act = mask * self.left_projection(act)
    right_act = mask * self.right_projection(act)

    act = torch.einsum('abc,ade,cef->bdf', left_act, right_act, self.output_w) + self.output_b

    epsilon = 1e-3
    norm = torch.einsum('abc,adc->bdc', mask, mask)
    act /= epsilon + norm
    return act


class MSAColumnAttention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  MSA per-column attention.

  Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, msa_act, msa_mask, is_training=False):
    dt = msa_act.dtype

    self.query_norm = nn.LayerNorm(msa_act.size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)

    self.attention = Attention(self.config, self.global_config, msa_act.size(-1))


    self.forward = self.go
    return self(msa_act, msa_mask, is_training)

  def go(self, msa_act, msa_mask, is_training=False):
    """
    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m]
    """
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert self.config.orientation == 'per_column'

    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = self.query_norm(msa_act)
    msa_act = self.attention(msa_act, msa_act, bias)
    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


class MSAColumnGlobalAttention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  MSA per-column global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, msa_act, msa_mask, is_training=False):
    dt = msa_act.dtype

    self.query_norm = nn.LayerNorm(msa_act.size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)
    self.attention = GlobalAttention(self.config, self.global_config, msa_act.size(-1))


    self.forward = self.go
    return self(msa_act, msa_mask, is_training)

  def go(self, msa_act, msa_mask, is_training=False):
    """
    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert self.config.orientation == 'per_column'

    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = self.query_norm(msa_act)

    
    # [N_seq, N_res, 1]
    msa_mask = msa_mask[...,None]
    msa_act = self.attention(msa_act, msa_act, msa_mask, bias)
    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


class ResidualDropOut(nn.Module):
  def __init__(self, config, global_config):
    super().__init__()
    self.dim = 0 if config.orientation=='per_row' else 1
    self.dropout_rate = 0.0 if global_config.deterministic else config.dropout_rate
    self.bern = lambda sh: self.dropout_rate < torch.rand(*sh, device=global_config.device)
    self.keep_rate = 1 - self.dropout_rate

  def forward(self, x, residual, training):
    if training:
      shape = list(residual.size())
      shape[self.dim] = 1
      residual = residual * self.bern(shape) / self.keep_rate
    return x + residual


def create_extra_msa_feature(batch, n_cat):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Arguments:
    batch: a dictionary with the following keys:
     * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
       centre. Note, that this is not one-hot encoded.
     * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
       the left of each position in the extra MSA.
     * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
       the left of each position in the extra MSA.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 7 = 4 bases + 'X' for unknown + gap + bert mask
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  dt = batch['extra_msa'].dtype
  msa_1hot = torch.functional.F.one_hot(batch['extra_msa'].long(), n_cat).to(dt)
  hd = batch['extra_has_deletion']
  dv = batch['extra_deletion_value']
  return torch.cat([msa_1hot,hd[...,None],dv[...,None]], axis=-1)


# def create_extra_rna_msa_feature(batch):
#   # 7 = 4 bases + 'X' for unknown + gap + bert mask
#   dt = batch['extra_msa'].dtype
#   msa_1hot = torch.functional.F.one_hot(batch['extra_msa'].long(), 7).to(dt)
#   hd = batch['extra_has_deletion']
#   dv = batch['extra_deletion_value']
#   return torch.cat([msa_1hot,hd[...,None],dv[...,None]], axis=-1)


class EvoformerIteration(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Single iteration (block) of Evoformer stack.

  Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
  """
  def __init__(self, config, global_config, is_extra_msa):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.is_extra_msa = is_extra_msa
    self.forward = self.init_parameters

  def init_parameters(self, x):
    activations, masks, is_training = x
    c = self.config
    gc = self.global_config

    pair_act = activations['pair']
    self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
            c.msa_row_attention_with_pair_bias, gc)
    self.dropout_msa_r_a_w_p_b = ResidualDropOut(c.msa_transition, gc)

    if not self.is_extra_msa:
      self.msa_column_attention = MSAColumnAttention(
            c.msa_column_attention, gc)
      self.attn_mod = self.msa_column_attention
    else:
      self.msa_column_global_attention = MSAColumnGlobalAttention(
          c.msa_column_attention, gc)
      self.attn_mod = self.msa_column_global_attention

    self.dropout_msa_attn = ResidualDropOut(c.msa_column_attention, gc)


    self.msa_transition = Transition(c.msa_transition, gc)
    self.dropout_msa_transition = ResidualDropOut(c.msa_transition, gc)# not sure about the config

    self.outer_product_mean = OuterProductMean(c.outer_product_mean, gc, 
                    num_output_channel=int(pair_act.size(-1)))
    self.dropout_out_prd_mn = ResidualDropOut(c.outer_product_mean, gc)

    self.triangle_multiplication_outgoing = TriangleMultiplication(
      c.triangle_multiplication_outgoing, gc)
    self.dropout_tri_mult_out = ResidualDropOut(
      c.triangle_multiplication_outgoing, gc)
    
    con_tmi = c.triangle_multiplication_incoming
    self.triangle_multiplication_incoming = TriangleMultiplication(con_tmi, gc)
    self.dropout_tri_mult_inc = ResidualDropOut(con_tmi, gc)

    con_tas = c.triangle_attention_starting_node
    self.triangle_attention_starting_node = TriangleAttention(con_tas, gc)
    self.dropout_tri_atn_srt = ResidualDropOut(con_tas, gc)

    con_tae = c.triangle_attention_ending_node
    self.triangle_attention_ending_node = TriangleAttention(con_tae, gc)
    self.dropout_tri_atn_end = ResidualDropOut(con_tae, gc)

    con_prt = c.pair_transition
    self.pair_transition = Transition(con_prt, gc)
    self.dropout_pair_trn = ResidualDropOut(con_prt, gc)
    
    self.forward = self.go
    return self(x)

  def go(self, x):
    """
    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    activations, masks, is_training = x

    msa_act, pair_act = activations['msa'], activations['pair']
    msa_mask, pair_mask = masks['msa'], masks['pair']
    
    msa_act = self.dropout_msa_r_a_w_p_b(msa_act, 
          self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act), training=is_training)

    msa_act = self.dropout_msa_attn(msa_act, self.attn_mod(msa_act, msa_mask), training=is_training)

    msa_act = self.dropout_msa_transition(msa_act, 
          self.msa_transition(msa_act, msa_mask), training=is_training)

    pair_act = self.dropout_out_prd_mn(pair_act, 
        self.outer_product_mean(msa_act, msa_mask), training=is_training)

    pair_act = self.dropout_tri_mult_out(pair_act, 
          self.triangle_multiplication_outgoing(pair_act, pair_mask), training=is_training)
    
    pair_act = self.dropout_tri_mult_inc(pair_act, 
          self.triangle_multiplication_incoming(pair_act, pair_mask), training=is_training)

    pair_act = self.dropout_tri_atn_srt(pair_act, 
          self.triangle_attention_starting_node(pair_act, pair_mask), training=is_training)

    pair_act = self.dropout_tri_atn_end(pair_act, 
          self.triangle_attention_ending_node(pair_act, pair_mask), training=is_training)
          
    pair_act = self.dropout_pair_trn(pair_act, 
          self.pair_transition(pair_act, pair_mask), training=is_training)

    return [{'msa': msa_act, 'pair': pair_act}, masks, is_training]


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Compute distogram from amino acid positions.

  Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.

  Returns:
    Distogram with the specified number of bins.
  """
  lower_breaks = torch.linspace(min_bin, max_bin, num_bins, device=positions.device)**2
  upper_breaks = torch.cat([lower_breaks[1:],
                torch.tensor([1e8], dtype=torch.float32, device=positions.device)], dim=-1)
  dist2 = (positions[:,None,:] - positions[None,:,:]).sum(-1)[...,None]

  dgram = ((dist2>lower_breaks).float() * (dist2<upper_breaks).float())

  return dgram

class EmbeddingsAndEvoformer(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters
  
  def init_parameters(self, batch, is_training):
    dt = batch['target_feat'].dtype
    self.dt = dt
    c = self.config
    gc = self.global_config

    self.preprocess_1d = makeLinear(batch['target_feat'].size(-1), c.msa_channel, dt, gc.device)
    self.preprocess_msa = makeLinear(batch['msa_feat'].size(-1), c.msa_channel, dt, gc.device)
    self.left_single = makeLinear(batch['target_feat'].size(-1), c.pair_channel, dt, gc.device)
    self.right_single = makeLinear(batch['target_feat'].size(-1), c.pair_channel, dt, gc.device)

    self.rna_preprocess_1d = makeLinear(batch['rna_target_feat'].size(-1), c.msa_channel, dt, gc.device)
    self.rna_preprocess_msa = makeLinear(batch['rna_msa_feat'].size(-1), c.msa_channel, dt, gc.device)
    self.rna_left_single = makeLinear(batch['rna_target_feat'].size(-1), c.pair_channel, dt, gc.device)
    self.rna_right_single = makeLinear(batch['rna_target_feat'].size(-1), c.pair_channel, dt, gc.device)

    if c.recycle_features:
      if 'prev_msa_first_row' in batch:
        self.prev_pos_linear = makeLinear(self.config.prev_pos['num_bins'], c.pair_channel, dt, gc.device)
        self.prev_msa_first_row_norm = nn.LayerNorm(batch['prev_msa_first_row'].size(-1), elementwise_affine=True, device=gc.device).to(dt)
        self.prev_pair_norm = nn.LayerNorm(batch['prev_pair'].size(-1), elementwise_affine=True, device=gc.device).to(dt)

    if c.max_relative_feature:
      self.pair_activiations = makeLinear(2 * c.max_relative_feature + 1, c.pair_channel, dt, gc.device)

    extra_msa_feat = create_extra_msa_feature(batch, gc.msa_n_token)
    self.extra_msa_activations = makeLinear(extra_msa_feat.size(-1), c.extra_msa_channel, dt, gc.device)

    extra_rna_msa_feat = create_extra_msa_feature(batch, gc.rna_msa_n_token)
    self.rna_extra_msa_activations = makeLinear(extra_rna_msa_feat.size(-1), c.extra_msa_channel, dt, gc.device)

    self.extra_msa_stack = nn.Sequential(*[EvoformerIteration(c.evoformer,
        gc, is_extra_msa=True) for _ in range(c.extra_msa_stack_num_block)])
    
    self.evoformer_iteration = nn.Sequential(*[EvoformerIteration(c.evoformer,
        gc, is_extra_msa=False) for _ in range(c.evoformer_num_block)])

    self.single_activations = makeLinear(c.msa_channel, c.seq_channel, dt, gc.device)

    self.forward = self.go
    return self(batch, is_training)

  def go(self, batch, is_training):
    c = self.config
    gc = self.global_config

    if 'rna' in batch:
      preprocess_1d = self.rna_preprocess_1d(batch['rna_target_feat'])
      preprocess_msa = self.rna_preprocess_msa(batch['rna_msa_feat'])
      left_single = self.rna_left_single(batch['rna_target_feat'])
      right_single = self.rna_right_single(batch['rna_target_feat'])
      extra_emb = self.rna_extra_msa_activations
      n_cat = gc.rna_msa_n_token
    else:
      preprocess_1d = self.preprocess_1d(batch['target_feat'])
      preprocess_msa = self.preprocess_msa(batch['msa_feat'])
      left_single = self.left_single(batch['target_feat'])
      right_single = self.right_single(batch['target_feat'])
      extra_emb = self.extra_msa_activations
      n_cat = gc.msa_n_token
    
    msa_activations = preprocess_1d[None,...] + preprocess_msa
    pair_activations = left_single[:, None] + right_single[None]
    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

    # Inject previous outputs for recycling.
    # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
    # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
    if c.recycle_pos and 'prev_pos' in batch:
      prev_pseudo_beta = batch['prev_pos']
      dgram = dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
      pair_activations += self.prev_pos_linear(dgram)

    if c.recycle_features:
      if 'prev_msa_first_row' in batch:
        prev_msa_first_row = self.prev_msa_first_row_norm(batch['prev_msa_first_row'])
        msa_activations[0] += prev_msa_first_row

      if 'prev_pair' in batch:
        pair_activations += self.prev_pair_norm(batch['prev_pair'])

    # Relative position encoding.
    # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
    # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
    if c.max_relative_feature:
      # Add one-hot-encoded clipped residue distances to the pair activations.
      pos = batch['residue_index']
      offset = pos[:, None] - pos[None, :]
      rel_pos = torch.functional.F.one_hot(
          torch.clip(offset + c.max_relative_feature,min=0, max=2 * c.max_relative_feature).long(),
          2 * c.max_relative_feature + 1).to(self.dt)
      pair_activations += self.pair_activiations(rel_pos)
    
    # Embed extra MSA features.
    # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
    extra_msa_feat = create_extra_msa_feature(batch, n_cat)
    extra_msa_activations = extra_emb(extra_msa_feat)

    # Extra MSA Stack.
    # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
    extra_msa_stack_input = {
        'msa': extra_msa_activations,
        'pair': pair_activations,
    }

    x = self.extra_msa_stack([extra_msa_stack_input, {'msa': batch['extra_msa_mask'], 
                'pair': mask_2d}, is_training])
    extra_msa_output = x[0]
    
    pair_activations = extra_msa_output['pair']

    evoformer_input = {
        'msa': msa_activations,
        'pair': pair_activations,
    }

    evoformer_masks = {'msa': batch['msa_mask'], 'pair': mask_2d}

    x = checkpoint_sequential(
      self.evoformer_iteration,
      c.evoformer_num_block,
      [evoformer_input, evoformer_masks, is_training]
    )
    evoformer_output = x[0]

    msa_activations = evoformer_output['msa']
    pair_activations = evoformer_output['pair']

    single_activations = self.single_activations(msa_activations[0])

    # num_sequences = batch['msa_feat'].size(0)
    # msa_activations =  msa_activations[:num_sequences, :, :]
    output = {
        'single': single_activations,
        'pair': pair_activations,
        # Crop away template rows such that they are not used in MaskedMsaHead.
        'msa': msa_activations,
        'msa_first_row': msa_activations[0],
    }

    return output


class InvariantPointAttention(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Invariant Point attention module.

  The high-level idea is that this attention module works over a set of points
  and associated orientations in 3D space (e.g. protein residues).

  Each residue outputs a set of queries and keys as points in their local
  reference frame.  The attention is then defined as the euclidean distance
  between the queries and keys in the global frame.

  Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
  """

  def __init__(self, config, global_config, dist_epsilon=1e-8):
    """Initialize.

    Args:
      config: Structure Module Config
      global_config: Global Config of Model.
      dist_epsilon: Small value to avoid NaN in distance calculation.
      name: Haiku Module name.
    """
    super().__init__()
    self.config, self.global_config = config, global_config
    self._dist_epsilon = dist_epsilon
    self._zero_initialize_last = global_config.zero_init
    self.forward = self.init_parameters

  def init_parameters(self, inputs_1d, inputs_2d, mask, affine):
    dt = inputs_1d.dtype
    c = self.config

    # Improve readability by removing a large number of 'self's.
    assert c.num_scalar_qk > 0
    assert c.num_point_qk > 0
    assert c.num_point_v > 0

    self.q_scalar = makeLinear(inputs_1d.size(-1), c.num_head * c.num_scalar_qk, dt, self.global_config.device)
    self.kv_scalar = makeLinear(inputs_1d.size(-1), c.num_head * (c.num_scalar_v + c.num_scalar_qk), dt, self.global_config.device)
    self.q_point_local = makeLinear(inputs_1d.size(-1), c.num_head * 3 * c.num_point_qk, dt, self.global_config.device)
    self.kv_point_local = makeLinear(inputs_1d.size(-1), c.num_head * 3 * (c.num_point_qk + c.num_point_v), dt, self.global_config.device)

    self.trainable_point_weights = nn.Parameter(torch.ones(c.num_head, dtype=dt, device=self.global_config.device)*np.log(np.exp(1.) - 1.))
    self.softplus = nn.Softplus()

    self.attention_2d = makeLinear(inputs_2d.size(-1), c.num_head, dt, self.global_config.device)
    final_init = 'zeros' if self._zero_initialize_last else 'linear'

    out_dim = c.num_scalar_v + c.num_point_v * 4 + inputs_2d.size(-1)
    self.output_projection = makeLinear(c.num_head * out_dim, c.num_channel, dt, self.global_config.device, initializer=final_init)

    self.forward = self.go
    return self(inputs_1d, inputs_2d, mask, affine)

  def go(self, inputs_1d, inputs_2d, mask, affine):
    """Compute geometry-aware attention.

    Given a set of query residues (defined by affines and associated scalar
    features), this function computes geometry-aware attention between the
    query residues and target residues.

    The residues produce points in their local reference frame, which
    are converted into the global frame in order to compute attention via
    euclidean distance.

    Equivalently, the target residues produce points in their local frame to be
    used as attention values, which are converted into the query residues'
    local frames.

    Args:
      inputs_1d: (N, C) 1D input embedding that is the basis for the
        scalar queries.
      inputs_2d: (N, M, C') 2D input embedding, used for biases and values.
      mask: (N, 1) mask to indicate which elements of inputs_1d participate
        in the attention.
      affine: QuatAffine object describing the position and orientation of
        every element in inputs_1d.

    Returns:
      Transformation of the input embedding.
    """
    c = self.config
    num_residues = inputs_1d.size(0)

    # Construct scalar queries of shape:
    # [num_query_residues, num_head, num_points]
    q_scalar = self.q_scalar(inputs_1d)
    q_scalar = q_scalar.reshape(num_residues, c.num_head, c.num_scalar_qk)

    # Construct scalar keys/values of shape:
    # [num_target_residues, num_head, num_points]
    kv_scalar = self.kv_scalar(inputs_1d)
    kv_scalar = kv_scalar.reshape(num_residues, c.num_head, c.num_scalar_v + c.num_scalar_qk)
    k_scalar, v_scalar = torch.split(kv_scalar, c.num_scalar_qk, dim=-1)

    # Construct query points of shape:
    # [num_residues, num_head, num_point_qk]

    # First construct query points in local frame.
    q_point_local = self.q_point_local(inputs_1d)
    q_point_local = torch.split(q_point_local, q_point_local.size(-1)//3, dim=-1)

    # Project query points into global frame.
    q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
    # Reshape query point for later use.
    q_point = [x.reshape(num_residues, c.num_head, c.num_point_qk) for x in q_point_global]

    # Construct key and value points.
    # Key points have shape [num_residues, num_head, num_point_qk]
    # Value points have shape [num_residues, num_head, num_point_v]

    # Construct key and value points in local frame.
    kv_point_local = self.kv_point_local(inputs_1d)
    kv_point_local = torch.split(kv_point_local, kv_point_local.size(-1)//3, dim=-1)

    # Project key and value points into global frame.
    kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
    kv_point_global = [x.reshape(num_residues, c.num_head, (c.num_point_qk + c.num_point_v))
        for x in kv_point_global]
    # Split key and value points.
    k_point, v_point = list(
        zip(*[torch.split(x, [c.num_point_qk, x.size(-1)-c.num_point_qk], dim=-1) for x in kv_point_global]))

    # We assume that all queries and keys come iid from N(0, 1) distribution
    # and compute the variances of the attention logits.
    # Each scalar pair (q, k) contributes Var q*k = 1
    scalar_variance = max(c.num_scalar_qk, 1) * 1.
    # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
    point_variance = max(c.num_point_qk, 1) * 9. / 2

    # Allocate equal variance to scalar, point and attention 2d parts so that
    # the sum is 1.

    num_logit_terms = 3

    scalar_weights = (1.0 / (num_logit_terms * scalar_variance))**0.5
    point_weights = (1.0 / (num_logit_terms * point_variance))**0.5
    attention_2d_weights = (1.0 / (num_logit_terms))**0.5

    # Trainable per-head weights for points.
    trainable_point_weights = self.softplus(self.trainable_point_weights)
    point_weights *= trainable_point_weights[:,None,...]

    v_point = [torch.swapaxes(x, -2, -3) for x in v_point]
    q_point = [torch.swapaxes(x, -2, -3) for x in q_point]
    k_point = [torch.swapaxes(x, -2, -3) for x in k_point]
    dist2 = [(qx[:, :, None, :] - kx[:, None, :, :])**2
        for qx, kx in zip(q_point, k_point)]
    dist2 = sum(dist2)
    attn_qk_point = -0.5 * (point_weights[:, None, None, :] * dist2).sum(dim=-1)

    v = torch.swapaxes(v_scalar, -2, -3)
    q = torch.swapaxes(scalar_weights * q_scalar, -2, -3)
    k = torch.swapaxes(k_scalar, -2, -3)
    attn_qk_scalar = torch.matmul(q, torch.swapaxes(k, -2, -1))
    attn_logits = attn_qk_scalar + attn_qk_point

    attention_2d = self.attention_2d(inputs_2d)

    attention_2d = attention_2d.permute(2, 0, 1)
    attention_2d = attention_2d_weights * attention_2d
    attn_logits += attention_2d

    mask_2d = mask * torch.swapaxes(mask, -1, -2)
    attn_logits -= 1e5 * (1. - mask_2d)

    # [num_head, num_query_residues, num_target_residues]
    self.softmax = nn.Softmax(dim=-1)
    attn = self.softmax(attn_logits)

    # [num_head, num_query_residues, num_head * num_scalar_v]
    result_scalar = torch.matmul(attn, v)

    # For point result, implement matmul manually so that it will be a float32
    # on TPU.  This is equivalent to
    # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
    #                        for vx in v_point]
    # but on the TPU, doing the multiply and reduce_sum ensures the
    # computation happens in float32 instead of bfloat16.
    result_point_global = [(attn[:, :, :, None] * vx[:, None, :, :]).sum(dim=-2) for vx in v_point]

    # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
    result_scalar = torch.swapaxes(result_scalar, -2, -3)
    result_point_global = [
        torch.swapaxes(x, -2, -3) for x in result_point_global]

    # Features used in the linear output projection. Should have the size
    # [num_query_residues, ?]
    output_features = []

    result_scalar = result_scalar.reshape(num_residues, c.num_head * c.num_scalar_v)
    output_features.append(result_scalar)

    result_point_global = [
        r.reshape(num_residues, c.num_head * c.num_point_v) for r in result_point_global]
    result_point_local = affine.invert_point(result_point_global, extra_dims=1)
    output_features.extend(result_point_local)

    output_features.append((self._dist_epsilon + (result_point_local[0]**2) +
                          (result_point_local[1]**2) + (result_point_local[2]**2))**0.5)

    # Dimensions: h = heads, i and j = residues,
    # c = inputs_2d channels
    # Contraction happens over the second residue dimension, similarly to how
    # the usual attention is performed.
    result_attention_over_2d = torch.einsum('hij, ijc->ihc', attn, inputs_2d)
    num_out = c.num_head * result_attention_over_2d.size(-1)
    output_features.append(result_attention_over_2d.reshape(num_residues, num_out))

    final_act = torch.cat(output_features, dim=-1)
    return self.output_projection(final_act)



# pylint: disable=bad-whitespace
class Quat:
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  """
  def __init__(self, device):
    zeros = lambda s: torch.zeros(*s, device=device)
    toTensor = lambda x: torch.tensor(x, device=device)

    rr = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]
    ii = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]
    jj = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]
    kk = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]

    ij = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]
    ik = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]
    jk = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]

    ir = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]
    jr = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]
    kr = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]

    [rr,ii,jj,kk,ij,ik,jk,ir,jr,kr] = list(map(toTensor, [rr,ii,jj,kk,ij,ik,jk,ir,jr,kr]))
    QUAT_TO_ROT = zeros((4, 4, 3, 3))
    QUAT_TO_ROT[0, 0] = rr
    QUAT_TO_ROT[1, 1] = ii
    QUAT_TO_ROT[2, 2] = jj
    QUAT_TO_ROT[3, 3] = kk
    QUAT_TO_ROT[1, 2] = ij
    QUAT_TO_ROT[1, 3] = ik
    QUAT_TO_ROT[2, 3] = jk
    for i, t in zip([1,2,3], [ir,jr,kr]):
      QUAT_TO_ROT[0, i] = t

    qml = ([[ 1, 0, 0, 0],
            [ 0,-1, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 0, 0,-1]],

           [[ 0, 1, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 1],
            [ 0, 0,-1, 0]],

           [[ 0, 0, 1, 0],
            [ 0, 0, 0,-1],
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0]],

           [[ 0, 0, 0, 1],
            [ 0, 0, 1, 0],
            [ 0,-1, 0, 0],
            [ 1, 0, 0, 0]])

    QMs = list(map(toTensor, qml))
    QUAT_MULTIPLY = zeros((4, 4, 4))
    for i in range(4):
      QUAT_MULTIPLY[:,:,i] = QMs[i]
    self.QUAT_TO_ROT = QUAT_TO_ROT
    self.QUAT_MULTIPLY = QUAT_MULTIPLY
    self.QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]
# pylint: enable=bad-whitespace


def rot_to_quat(rot, unstack_inputs=False):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Convert rotation matrix to quaternion.

  Note that this function calls self_adjoint_eig which is extremely expensive on
  the GPU. If at all possible, this function should run on the CPU.

  Args:
     rot: rotation matrix (see below for format).
     unstack_inputs:  If true, rotation matrix should be shape (..., 3, 3)
       otherwise the rotation matrix should be a list of lists of tensors.

  Returns:
    Quaternion as (..., 4) tensor.
  """
  if unstack_inputs:
    rot = [torch.moveaxis(x, -1, 0) for x in torch.moveaxis(rot, -2, 0)]

  [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

  # pylint: disable=bad-whitespace
  k = [[ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
       [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
       [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
       [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]]
  # pylint: enable=bad-whitespace

  k = (1./3.) * torch.stack([torch.stack(x, dim=-1) for x in k],
                          dim=-2)

  # Get eigenvalues in non-decreasing order and associated.
  # d = k.device
  # _, qs = torch.linalg.eigh(k.to('cpu')).to(d)
  _, qs = torch.linalg.eigh(k)
  return qs[..., -1]


def rot_list_to_tensor(rot_list):
  """Convert list of lists to rotation tensor."""
  return torch.stack(
      [torch.stack(rot_list[0], dim=-1),
       torch.stack(rot_list[1], dim=-1),
       torch.stack(rot_list[2], dim=-1)],
      dim=-2)


def vec_list_to_tensor(vec_list):
  """Convert list to vector tensor."""
  return torch.stack(vec_list, dim=-1)


def quat_to_rot(QUAT_TO_ROT, normalized_quat):
  """Convert a normalized quaternion to a rotation matrix."""
  rot_tensor = torch.sum(
      torch.reshape(QUAT_TO_ROT, (4, 4, 9)) *
      normalized_quat[..., :, None, None] *
      normalized_quat[..., None, :, None],
      dim=(-3, -2))
  rot = torch.moveaxis(rot_tensor, -1, 0)  # Unstack.
  return [[rot[0], rot[1], rot[2]],
          [rot[3], rot[4], rot[5]],
          [rot[6], rot[7], rot[8]]]


def quat_multiply_by_vec(QUAT_MULTIPLY_BY_VEC, quat, vec):
  """Multiply a quaternion by a pure-vector quaternion."""
  return torch.sum(
      QUAT_MULTIPLY_BY_VEC *
      quat[..., :, None, None] *
      vec[..., None, :, None],
      dim=(-3, -2))


def quat_multiply(QUAT_MULTIPLY, quat1, quat2):
  """Multiply a quaternion by another quaternion."""
  return torch.sum(
      QUAT_MULTIPLY *
      quat1[..., :, None, None] *
      quat2[..., None, :, None],
      dim=(-3, -2))

def apply_rot_to_vec(rot, vec, unstack=False):
  """Multiply rotation matrix by a vector."""
  if unstack:
    x, y, z = [vec[:, i] for i in range(3)]
  else:
    x, y, z = vec
  return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
          rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
          rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]

def apply_inverse_rot_to_vec(rot, vec):
  """Multiply the inverse of a rotation matrix by a vector."""
  # Inverse rotation is just transpose
  return [rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
          rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
          rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2]]


class QuatAffine(object):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Affine transformation represented by quaternion and vector."""

  def __init__(self, quaternion, translation, device, rotation=None, normalize=True,
               unstack_inputs=False):
    """Initialize from quaternion and translation.

    Args:
      quaternion: Rotation represented by a quaternion, to be applied
        before translation.  Must be a unit quaternion unless normalize==True.
      translation: Translation represented as a vector.
      rotation: Same rotation as the quaternion, represented as a (..., 3, 3)
        tensor.  If None, rotation will be calculated from the quaternion.
      normalize: If True, l2 normalize the quaternion on input.
      unstack_inputs: If True, translation is a vector with last component 3
    """
    quat = Quat(device)
    self.QUAT_TO_ROT = quat.QUAT_TO_ROT
    self.QUAT_MULTIPLY = quat.QUAT_MULTIPLY
    self.QUAT_MULTIPLY_BY_VEC = quat.QUAT_MULTIPLY_BY_VEC

    self.device = device


    if quaternion is not None:
      assert quaternion.shape[-1] == 4

    if unstack_inputs:
      if rotation is not None:
        rotation = [torch.moveaxis(x, -1, 0)   # Unstack.
                    for x in torch.moveaxis(rotation, -2, 0)]  # Unstack.
      translation = torch.moveaxis(translation, -1, 0)  # Unstack.

    if normalize and quaternion is not None:
      quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1,
                                                keepdims=True)

    if rotation is None:
      rotation = quat_to_rot(self.QUAT_TO_ROT, quaternion)

    self.quaternion = quaternion
    self.rotation = [list(row) for row in rotation]
    self.translation = list(translation)

    assert all(len(row) == 3 for row in self.rotation)
    assert len(self.translation) == 3

  def to_tensor(self):
    return torch.cat(
        [self.quaternion] + [x[...,None] for x in self.translation], dim=-1)

  def clone(self):
    return QuatAffine(self.quaternion.clone(),
        [x.clone() for x in self.translation],
        self.device,
        rotation=[[x.clone() for x in row] for row in self.rotation],
        normalize=False)

  def detach(self):
    """Return a new QuatAffine with tensor_fn applied (e.g. stop_gradient)."""
    return QuatAffine(self.quaternion.detach(),
        [x.detach() for x in self.translation],
        self.device,
        rotation=[[x.detach() for x in row] for row in self.rotation],
        normalize=False)

  def detach_rot(self):
    return QuatAffine(self.quaternion.detach(),
        [x for x in self.translation],
        self.device,
        rotation=[[x.detach() for x in row] for row in self.rotation],
        normalize=False)

  def scale_translation(self, position_scale):
    """Return a new quat affine with a different scale for translation."""

    return QuatAffine(
        self.quaternion,
        [x * position_scale for x in self.translation],
        self.device,
        rotation=[[x for x in row] for row in self.rotation],
        normalize=False)

  @classmethod
  def from_tensor(cls, tensor, device, normalize=False):
    quaternion, tx, ty, tz = torch.split(tensor, [4, 1, 1, 1], dim=-1)
    return cls(quaternion, [tx[..., 0], ty[..., 0], tz[..., 0]], device, normalize=normalize)

  def pre_compose(self, update):
    """Return a new QuatAffine which applies the transformation update first.

    Args:
      update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
        update is (1, x, y, z) and zero for the 3-vector is the identity
        quaternion. 3-vector for translation concatenated.

    Returns:
      New QuatAffine object.
    """
    vector_quaternion_update, x, y, z = torch.split(update, [3, 1, 1, 1], dim=-1)
    trans_update = [torch.squeeze(x, dim=-1),
                    torch.squeeze(y, dim=-1),
                    torch.squeeze(z, dim=-1)]

    new_quaternion = (self.quaternion +
                      quat_multiply_by_vec(self.QUAT_MULTIPLY_BY_VEC,
                                           self.quaternion,
                                           vector_quaternion_update))

    trans_update = apply_rot_to_vec(self.rotation, trans_update)
    new_translation = [
        self.translation[0] + trans_update[0],
        self.translation[1] + trans_update[1],
        self.translation[2] + trans_update[2]]

    return QuatAffine(new_quaternion, new_translation, self.device)

  def apply_to_point(self, point, extra_dims=0):
    """Apply affine to a point.

    Args:
      point: List of 3 tensors to apply affine.
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      rotation = [[r[...,None] for r in row] for row in self.rotation]
      translation = [t[...,None] for t in self.translation]

    rot_point = apply_rot_to_vec(rotation, point)
    return [
        rot_point[0] + translation[0],
        rot_point[1] + translation[1],
        rot_point[2] + translation[2]]

  def invert_point(self, transformed_point, extra_dims=0):
    """Apply inverse of transformation to a point.

    Args:
      transformed_point: List of 3 tensors to apply affine
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      rotation = [[r[...,None] for r in row] for row in self.rotation]
      translation = [t[...,None] for t in self.translation]

    rot_point = [
        transformed_point[0] - translation[0],
        transformed_point[1] - translation[1],
        transformed_point[2] - translation[2]]

    return apply_inverse_rot_to_vec(rotation, rot_point)

  def __repr__(self):
    return 'QuatAffine(%r, %r)' % (self.quaternion, self.translation)


def _multiply(a, b):
  return torch.stack([
      torch.stack([a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
                 a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
                 a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]]),

      torch.stack([a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
                 a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
                 a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]]),

      torch.stack([a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
                 a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
                 a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]])])


def make_canonical_transform(n_xyz, ca_xyz, c_xyz):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Returns translation and rotation matrices to canonicalize residue atoms.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (translation, rotation) where:
      translation is an array of shape [batch, 3] defining the translation.
      rotation is an array of shape [batch, 3, 3] defining the rotation.
    After applying the translation and rotation to all atoms in a residue:
      * All atoms will be shifted so that CA is at the origin,
      * All atoms will be rotated so that C is at the x-axis,
      * All atoms will be shifted so that N is in the xy plane.
  """
  assert len(n_xyz.shape) == 2, n_xyz.shape
  assert n_xyz.shape[-1] == 3, n_xyz.shape
  assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
      n_xyz.shape, ca_xyz.shape, c_xyz.shape)

  device = c_xyz.device
  # Place CA at the origin.
  translation = -ca_xyz
  n_xyz = n_xyz + translation
  c_xyz = c_xyz + translation

  # Place C on the x-axis.
  c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
  # Rotate by angle c1 in the x-y plane (around the z-axis).
  sin_c1 = -c_y / (1e-20 + c_x**2 + c_y**2)**0.5
  cos_c1 = c_x / (1e-20 + c_x**2 + c_y**2)**0.5
  zeros = torch.zeros_like(sin_c1, device=device)
  ones = torch.ones_like(sin_c1, device=device)

  # pylint: disable=bad-whitespace
  c1_rot_matrix = torch.stack([torch.stack([cos_c1, -sin_c1, zeros]),
                             torch.stack([sin_c1,  cos_c1, zeros]),
                             torch.stack([zeros,    zeros,  ones])])

  # Rotate by angle c2 in the x-z plane (around the y-axis).
  sin_c2 = c_z / (1e-20 + c_x**2 + c_y**2 + c_z**2)**0.5
  cos_c2 = ((c_x**2 + c_y**2)**0.5) / (
      1e-20 + c_x**2 + c_y**2 + c_z**2)**0.5
  c2_rot_matrix = torch.stack([torch.stack([cos_c2,  zeros, sin_c2]),
                             torch.stack([zeros,    ones,  zeros]),
                             torch.stack([-sin_c2, zeros, cos_c2])])

  c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
  n_xyz = torch.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).T

  # Place N in the x-y plane.
  _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
  # Rotate by angle alpha in the y-z plane (around the x-axis).
  sin_n = -n_z / (1e-20 + n_y**2 + n_z**2)**0.5
  cos_n = n_y / (1e-20 + n_y**2 + n_z**2)**0.5
  n_rot_matrix = torch.stack([torch.stack([ones,  zeros,  zeros]),
                            torch.stack([zeros, cos_n, -sin_n]),
                            torch.stack([zeros, sin_n,  cos_n])])
  # pylint: enable=bad-whitespace

  return (translation,
          torch.permute(_multiply(n_rot_matrix, c_rot_matrix), [2, 0, 1]))


def make_transform_from_reference(n_xyz, ca_xyz, c_xyz):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Returns rotation and translation matrices to convert from reference.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (rotation, translation) where:
      rotation is an array of shape [batch, 3, 3] defining the rotation.
      translation is an array of shape [batch, 3] defining the translation.
    After applying the translation and rotation to the reference backbone,
    the coordinates will approximately equal to the input coordinates.

    The order of translation and rotation differs from make_canonical_transform
    because the rotation from this function should be applied before the
    translation, unlike make_canonical_transform.
  """
  translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
  return torch.permute(rotation, (0, 2, 1)), -translation


def generate_new_affine(sequence_mask, device):
  num_residues = sequence_mask.size(0)
  quaternion = torch.tile(
      torch.tensor([1., 0., 0., 0.], device=device).reshape(1, 4),
      [num_residues, 1])

  translation = torch.zeros(num_residues, 3, device=device)
  return QuatAffine(quaternion, translation, device, unstack_inputs=True)


class FoldIteration(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  A single iteration of the main structure module loop.

  Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

  First, each residue attends to all residues using InvariantPointAttention.
  Then, we apply transition layers to update the hidden representations.
  Finally, we use the hidden representations to produce an update to the
  affine of each residue.
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, activations,
               sequence_mask,
               update_affine,
               is_training,
               initial_act,
               static_feat_2d=None,
               aatype=None):
    dt = activations['act'].dtype
    c = self.config
    self.dropout1 = nn.Dropout()#p=c.dropout * (1-self.global_config.deterministic))
    self.dropout2 = nn.Dropout()

    n = activations['act'].size(-1)
    self.invariant_point_attention = InvariantPointAttention(self.config, self.global_config)
    self.attention_layer_norm = nn.LayerNorm(n, elementwise_affine=True, device=self.global_config.device).to(dt)

    final_init = 'zeros' if self.global_config.zero_init else 'linear'
    ch = n
    transitions = []
    for i in range(c.num_layer_in_transition):
      init = 'relu' if i < c.num_layer_in_transition - 1 else final_init
      transitions.append(makeLinear(ch, c.num_channel, dt, self.global_config.device, initializer=init))
      if i < c.num_layer_in_transition - 1: transitions.append(nn.ReLU(inplace=False))
      ch = c.num_channel
    self.transition = nn.Sequential(*transitions)
    self.transition_layer_norm = nn.LayerNorm(c.num_channel, elementwise_affine=True, device=self.global_config.device).to(dt)

    if update_affine:
      # This block corresponds to
      # Jumper et al. (2021) Alg. 23 "Backbone update"
      affine_update_size = 6
      self.affine_update = makeLinear(c.num_channel, affine_update_size, dt, self.global_config.device, initializer=final_init)

    self.forward = self.go
    return self(activations,sequence_mask,update_affine,
               is_training,initial_act,static_feat_2d,aatype)

  def go(self, activations, sequence_mask, update_affine, is_training,
               initial_act, static_feat_2d=None, aatype=None):
    self.dropout1.p = int(not self.global_config.deterministic) * self.config.dropout * int(is_training)
    self.dropout2.p = self.dropout1.p
    affine = QuatAffine.from_tensor(activations['affine'], self.global_config.device)

    act = activations['act']

    # Attention
    residual = self.invariant_point_attention(act.clone(), static_feat_2d, sequence_mask, affine)
    act += residual
    dropped = self.dropout1(act)
    act = self.attention_layer_norm(dropped)

    # Transition
    residual = self.transition(act.clone())
    act += residual
    dropped = self.dropout2(act)
    act = self.transition_layer_norm(dropped)

    if update_affine:
      # This block corresponds to
      # Jumper et al. (2021) Alg. 23 "Backbone update"
      # Affine update
      affine_update = self.affine_update(act.clone())
      affine = affine.pre_compose(affine_update)
      
    outputs = {'affine': affine.to_tensor()}
    affine = affine.detach_rot()

    return {'act': act, 'affine': affine.to_tensor()}, outputs


class GenerateAffines(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Generate predicted affines for a single chain.

  Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

  This is the main part of the structure module - it iteratively applies
  folding to produce a set of predicted residue positions.
  """
  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['single'].dtype
    s = representations['single'].size(-1)
    self.initial_projection = makeLinear(s, self.config.num_channel, dt, self.global_config.device)
    self.fold_iteration = FoldIteration(self.config, self.global_config)

    self.single_layer_norm = nn.LayerNorm(s, elementwise_affine=True, device=self.global_config.device).to(dt)
    self.pair_layer_norm = nn.LayerNorm(representations['pair'].size(-1), elementwise_affine=True, device=self.global_config.device).to(dt)

    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    """
    Args:
      representations: Representations dictionary.
      batch: Batch dictionary.
      config: Config for the structure module.
      global_config: Global config.
      is_training: Whether the model is being trained.
      safe_key: A prng.SafeKey object that wraps a PRNG key.

    Returns:
      A dictionary containing residue affines and sidechain positions.
    """
    c = self.config
    sequence_mask = batch['seq_mask'][:, None]

    act = self.single_layer_norm(representations['single'])
    initial_act = act

    act = self.initial_projection(act)
    affine = generate_new_affine(sequence_mask, self.global_config.device)

    assert len(batch['seq_mask'].shape) == 1

    activations = {'act': act, 'affine': affine.to_tensor()}
    act_2d = self.pair_layer_norm(representations['pair'])

    outputs = []
    for _ in range(c.num_layer):
      # activations have detached grads for rot, and output 
      activations, output = self.fold_iteration(
          activations,
          initial_act=initial_act,
          static_feat_2d=act_2d,
          sequence_mask=sequence_mask,
          update_affine=True,
          is_training=is_training,
          aatype=batch['aatype'])
      outputs.append(output)
      # outputs.append({k:torch.stack(v) for k,v in output.items()})
    output = {k:torch.stack([o[k] for o in outputs]) for k in output}
    # output = jax.tree_map(lambda *x: jnp.stack(x), *outputs)
    # Include the activations in the output dict for use by the LDDT-Head.
    output['act'] = activations['act']
    return output


# Array of 3-component vectors, stored as individual array for
# each component.
Vecs = collections.namedtuple('Vecs', ['x', 'y', 'z'])

# Array of 3x3 rotation matrices, stored as individual array for
# each component.
Rots = collections.namedtuple('Rots', ['xx', 'xy', 'xz',
                                       'yx', 'yy', 'yz',
                                       'zx', 'zy', 'zz'])
# Array of rigid 3D transformations, stored as array of rotations and
# array of translations.
Rigids = collections.namedtuple('Rigids', ['rot', 'trans'])

def rigids_from_quataffine(a: QuatAffine) -> Rigids:
  """Converts QuatAffine object to the corresponding Rigids object."""
  return Rigids(Rots(*tree.flatten(a.rotation)),
                Vecs(*a.translation))

def vecs_squared_distance(v1: Vecs, v2: Vecs):
  """Computes squared euclidean difference between 'v1' and 'v2'."""
  return (v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2

def vecs_add(v1: Vecs, v2: Vecs) -> Vecs:
  """Add two vectors 'v1' and 'v2'."""
  return Vecs(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

def rigids_mul_vecs(r: Rigids, v: Vecs) -> Vecs:
  """Apply rigid transforms 'r' to points 'v'."""
  return vecs_add(rots_mul_vecs(r.rot, v), r.trans)

def rots_mul_vecs(m: Rots, v: Vecs) -> Vecs:
  """Apply rotations 'm' to vectors 'v'."""
  return Vecs(m.xx * v.x + m.xy * v.y + m.xz * v.z,
              m.yx * v.x + m.yy * v.y + m.yz * v.z,
              m.zx * v.x + m.zy * v.y + m.zz * v.z)


def invert_rots(m: Rots) -> Rots:
  """Computes inverse of rotations 'm'."""
  return Rots(m.xx, m.yx, m.zx,
              m.xy, m.yy, m.zy,
              m.xz, m.yz, m.zz)


def invert_rigids(r: Rigids) -> Rigids:
  """Computes group inverse of rigid transformations 'r'."""
  inv_rots = invert_rots(r.rot)
  t = rots_mul_vecs(inv_rots, r.trans)
  inv_trans = Vecs(-t.x, -t.y, -t.z)
  return Rigids(inv_rots, inv_trans)


def frame_aligned_point_error(
    pred_frames,  # shape (num_frames)
    target_frames,  # shape (num_frames)
    frames_mask,  # shape (num_frames)
    pred_positions,  # shape (num_positions)
    target_positions,  # shape (num_positions)
    positions_mask,  # shape (num_positions)
    length_scale,
    l1_clamp_distance,
    epsilon=1e-4):  # shape ()
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Measure point error under different alignments.

  Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

  Computes error between two structures with B points under A alignments derived
  from the given pairs of frames.
  Args:
    pred_frames: num_frames reference frames for 'pred_positions'.
    target_frames: num_frames reference frames for 'target_positions'.
    frames_mask: Mask for frame pairs to use.
    pred_positions: num_positions predicted positions of the structure.
    target_positions: num_positions target positions of the structure.
    positions_mask: Mask on which positions to score.
    length_scale: length scale to divide loss by.
    l1_clamp_distance: Distance cutoff on error beyond which gradients will
      be zero.
    epsilon: small value used to regularize denominator for masked average.
  Returns:
    Masked Frame Aligned Point Error.
  """

  # pred_frames and pred_positions arrays may have a batch dim,
  # if so will need to repeat along other arrays
  assert pred_frames.rot.xx.ndim in {1,2}
  assert target_frames.rot.xx.ndim == 1
  assert frames_mask.ndim == 1, frames_mask.ndim
  assert pred_positions.x.ndim in {1,2}
  assert target_positions.x.ndim == 1
  assert positions_mask.ndim == 1

  s = pred_frames.rot.xx.shape[:-1]
  mapper = lambda C,o,f: C(**{k:f(v) for k,v in o._asdict().items()})

  # Compute array of predicted positions in the predicted frames.
  ir = invert_rigids(pred_frames)
  local_pred_pos = rigids_mul_vecs(
    Rigids(mapper(Rots, ir.rot, lambda a:a[..., None]), 
           mapper(Vecs, ir.trans, lambda a:a[..., None])),
    mapper(Vecs, pred_positions, lambda a:a[..., None, :])
  )

  # Compute array of target positions in the target frames.
  irt = invert_rigids(target_frames)
  local_target_pos = rigids_mul_vecs(
    Rigids(mapper(Rots, irt.rot, lambda a:a[..., None].repeat(*s, 1, 1)),
           mapper(Vecs, irt.trans, lambda a:a[..., None].repeat(*s, 1, 1))),
    mapper(Vecs, target_positions, lambda a:a[..., None, :].repeat(*s, 1, 1))
  )

  # Compute errors between the structures.
  error_dist = (vecs_squared_distance(local_pred_pos, local_target_pos) + epsilon)**0.5

  if l1_clamp_distance:
    error_dist = torch.clip(error_dist, 0, l1_clamp_distance)

  normed_error = error_dist / length_scale

  extra_dims = [None]*(len(normed_error.shape)-2)
  normed_error *= frames_mask[extra_dims+[slice(None),None]]
  normed_error *= positions_mask[extra_dims+[None,slice(None)]]

  normalization_factor = (frames_mask.sum(-1) * positions_mask.sum(-1))
  return normed_error.sum(dim=(-2, -1)) / (epsilon + normalization_factor)


def backbone_loss(ret, batch, value, config, device):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Backbone FAPE Loss.

  Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17

  Args:
    ret: Dictionary to write outputs into, needs to contain 'loss'.
    batch: Batch, needs to contain 'backbone_affine_tensor',
      'backbone_affine_mask'.
    value: Dictionary containing structure module output, needs to contain
      'traj', a trajectory of rigids.
    config: Configuration of loss, should contain 'fape.clamp_distance' and
      'fape.loss_unit_distance'.
  """
  affine_trajectory = QuatAffine.from_tensor(value['traj'], device)
  rigid_trajectory = rigids_from_quataffine(affine_trajectory)

  gt_affine = QuatAffine.from_tensor(
      batch['backbone_affine_tensor'], device)
  gt_rigid = rigids_from_quataffine(gt_affine)
  backbone_mask = batch['backbone_affine_mask']
  
  args = (rigid_trajectory, gt_rigid, backbone_mask, rigid_trajectory.trans, gt_rigid.trans, backbone_mask)
  
  fape_loss = frame_aligned_point_error(
    *args,
    l1_clamp_distance=config.fape.clamp_distance,
    length_scale=config.fape.loss_unit_distance
  )

  if 'use_clamped_fape' in batch:
    # Jumper et al. (2021) Suppl. Sec. 1.11.5 "Loss clamping details"
    use_clamped_fape = batch['use_clamped_fape'].float()
    fape_loss_unclamped = frame_aligned_point_error(
      *args,
      l1_clamp_distance=None,
      length_scale=config.fape.loss_unit_distance
    )
    fape_loss = (fape_loss * use_clamped_fape +
                 fape_loss_unclamped * (1 - use_clamped_fape))

  ret['fape'] = fape_loss[-1]
  ret['loss'] += fape_loss.mean()

class StructureModule(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  StructureModule as a network head.

  Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
  """

  def __init__(self, config, global_config, compute_loss=True):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.compute_loss = compute_loss
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    self.generate_affines = GenerateAffines(self.config, self.global_config)
    self.scale = torch.tensor([1.] * 4 + [self.config.position_scale] * 3, 
                            device=self.global_config.device)
    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    ret = {}
    
    # print('repres:')
    # print({k+', '+str(v.shape) for k,v in representations.items()})
    # print('batch:')
    # print({k+', '+str(v.shape) for k,v in batch.items()})

    output = self.generate_affines(representations, batch, is_training)
    representations['structure_module'] = output['act']
    ret['traj'] = output['affine'] * self.scale
    ret['final_affines'] = ret['traj'][-1]

    ret['final_frame_pos'] = ret['traj'][-1][...,-3:]

    if self.compute_loss:
      return ret
    else:
      no_loss_features = ['final_frame_pos']#['final_atom_positions', 'final_atom_mask']
      no_loss_ret = {k: ret[k] for k in no_loss_features}
      return no_loss_ret

  def loss(self, value, batch):
    ret = {'loss': 0.}
    backbone_loss(ret, batch, value, self.config, self.global_config.device)
    return ret


class MaskedMsaHead(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Head to predict MSA at the masked locations.

  The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
  version of the full MSA, based on a linear projection of
  the MSA representation.
  Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['msa'].dtype
    gc = self.global_config
    init = 'zeros' if gc.zero_init else 'linear'
    h = representations['msa'].size(-1)
    self.logits = makeLinear(h, gc.msa_n_token, dt, gc.device, initializer=init)
    self.rnalogits = makeLinear(h, gc.rna_msa_n_token, dt, gc.device, initializer=init)

    if is_training:
      self.smxe = nn.CrossEntropyLoss(reduction='none')

    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    """
    Arguments:
      representations: Dictionary of representations, must contain:
        * 'msa': MSA representation, shape [N_seq, N_res, c_m].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_seq, N_res, N_aatype] with
            (unnormalized) log probabilies of predicted aatype at position.
    """
    layer = self.rnalogits if 'rna' in batch else self.logits
    del batch
    logits = layer(representations['msa'])
    return dict(logits=logits)

  def loss(self, value, batch):
    s = batch['true_msa'].shape
    l = value['logits'].reshape(-1,value['logits'].size(-1))
    tr = batch['true_msa'].reshape(-1).long()
    errors = self.smxe(l, tr).reshape(s)

    loss = (torch.sum(errors * batch['bert_mask'], dim=(-2, -1)) /
            (1e-8 + torch.sum(batch['bert_mask'], dim=(-2, -1))))
    return {'loss': loss}


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Log loss of a distogram."""

  assert len(logits.shape) == 3
  positions = batch['pseudo_beta']
  mask = batch['pseudo_beta_mask']

  assert positions.shape[-1] == 3

  sq_breaks = bin_edges**2
  dist2 = ((positions[...,None,:] - positions[...,None,:,:])**2).sum(-1)[...,None]
  true_bins = (dist2 > sq_breaks).sum(-1)

  s = true_bins.shape
  smxe = nn.CrossEntropyLoss(reduction='none')
  errors = smxe(logits.reshape(-1, logits.size(-1)), true_bins.reshape(-1).long()).reshape(s)
  square_mask = mask[...,None,:] * mask[...,None]

  avg_error = (
      torch.sum(errors * square_mask, dim=(-2, -1)) /
      (1e-6 + torch.sum(square_mask, dim=(-2, -1))))
  dist2 = dist2[..., 0]
  return dict(loss=avg_error, true_dist=(1e-6 + dist2)**0.5)


class DistogramHead(nn.Module):
  """  
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters

  def init_parameters(self, representations, batch, is_training):
    dt = representations['pair'].dtype
    init = 'zeros' if self.global_config.zero_init else 'linear'
    self.half_logits = makeLinear(representations['pair'].size(-1), 
      self.config.num_bins, dt, self.global_config.device, initializer=init)
    self.breaks = torch.linspace(self.config.first_break, self.config.last_break,
                      self.config.num_bins - 1, device=self.global_config.device)

    self.forward = self.go
    return self(representations, batch, is_training)

  def go(self, representations, batch, is_training):
    """
    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = self.half_logits(representations['pair'])

    logits = half_logits + torch.swapaxes(half_logits, -2, -3)

    return dict(logits=logits, bin_edges=self.breaks)

  def loss(self, value, batch):
    dll = _distogram_log_loss(value['logits'], value['bin_edges'],
                               batch, self.config.num_bins)
    return dll


class AlphaFoldIteration(nn.Module):
  """
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).
  
  A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file. Each head also returns a
  loss which is combined as a weighted sum to produce the total loss.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config, self.global_config = config, global_config
    self.forward = self.init_parameters
  
  def init_parameters(self, ensembled_batch, non_ensembled_batch, is_training,
      compute_loss=False, ensemble_representations=False):
    
    self.evoformer = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)

    self.head_cfg = {}
    self.heads = nn.ModuleDict()
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_namespace = {
        'masked_msa': MaskedMsaHead,
        'distogram': DistogramHead,
        'structure_module': functools.partial(StructureModule, compute_loss=True),
      }
      self.head_cfg[head_name] = head_config
      self.heads[head_name] = head_namespace[head_name](head_config, self.global_config)

    self.forward = self.go
    return self(ensembled_batch, non_ensembled_batch, is_training,
      compute_loss=compute_loss, ensemble_representations=ensemble_representations)

  def go(self, ensembled_batch, non_ensembled_batch, is_training,
      compute_loss=False, ensemble_representations=False):
    '''
    if not ensemble_representations:
      ensembled_batch : is unbatched when it passes through the embeddings and evoformer
      non_ensembled_batch : similarly the representations are unbatched when passing through heads
    '''

    num_ensemble = ensembled_batch['seq_length'].shape[0]

    if not ensemble_representations:
      # print(ensembled_batch['seq_length'].shape)
      assert ensembled_batch['seq_length'].shape[0] == 1

    def slice_batch(i):
      b = {k: v[i] for k, v in ensembled_batch.items()}
      b.update(non_ensembled_batch)
      return b

    # Compute representations for each batch element and average.
    # print('ensembled_batch:')
    # print({k+', '+str(v.shape) for k,v in ensembled_batch.items()})
    
    batch0 = slice_batch(0)
    representations = self.evoformer(batch0, is_training)

    # MSA representations are not ensembled so
    # we don't pass tensor into the loop.
    msa_representation = representations['msa']
    del representations['msa']

    # Average the representations (except MSA) over the batch dimension.
    if ensemble_representations:

      for i in range(1, num_ensemble):
        representations_update = self.evoformer(slice_batch(i), is_training)
        for k in representations:
          representations[k] += representations_update[k]
      
      for k in representations:
        if k != 'msa':
          representations[k] /= num_ensemble

    representations['msa'] = msa_representation
    batch = batch0  # We are not ensembled from here on.

    total_loss = 0.
    ret = {}
    ret['representations'] = representations

    def loss(module, head_config, ret, name, filter_ret=True):
      loss_output = module.loss(ret[name] if filter_ret else ret, batch)
      ret[name].update(loss_output)
      return head_config.weight * ret[name]['loss']

    loss_rec = {}
    for name, module in self.heads.items():
      # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
      # StructureModule is executed.
      if name not in ('predicted_lddt', 'predicted_aligned_error'):
        ret[name] = module(representations, batch, is_training)
      if compute_loss:
        l = loss(module, self.head_cfg[name], ret, name)
        total_loss += l
        loss_rec[name] = l.data.item()

    if compute_loss: 
      return ret, total_loss, loss_rec
    return ret

def tree_map(fn, tree_):
  return {k:tree_map(fn, b) if type(b)==dict else fn(b) for k,b in tree_.items()}

class AlphaFold(pl.LightningModule):
  """ 
  DeepMind AlphaFold code https://github.com/deepmind/alphafold
  ported to pytorch by Louis Robinson (21 Aug 2021).

  AlphaFold model with recycling.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference"
  """

  def __init__(self, config, is_training, compute_loss=False,
      ensemble_representations=False, return_representations=False,):
    super().__init__()
    self.config = config['model']
    self.global_config = config['model'].global_config
    self.data_config = config['data']
    self.alphafold_iteration = AlphaFoldIteration(self.config, self.global_config)
    self.is_training = is_training
    self.compute_loss = compute_loss
    self.ensemble_representations = ensemble_representations
    self.return_representations = return_representations
  
  def forward(self, batch):
    """
    Arguments:
      batch: Dictionary with inputs to the AlphaFold model.
      is_training: Whether the system is in training or inference mode.
      compute_loss: Whether to compute losses (requires extra features
        to be present in the batch and knowing the true structure).
      ensemble_representations: Whether to use ensembling of representations.
      return_representations: Whether to also return the intermediate
        representations.

    Returns:
      When compute_loss is True:
        a tuple of loss and output of AlphaFoldIteration.
      When compute_loss is False:
        just output of AlphaFoldIteration.

      The output of AlphaFoldIteration is a nested dictionary containing
      predictions from the various heads.
    """
    batch_size, num_residues = batch['aatype'].shape

    def get_prev(ret):
      return {
        # 'prev_pos': ret['structure_module']['final_atom_positions'].detach(),
        'prev_pos': ret['structure_module']['final_frame_pos'].detach(),
        'prev_msa_first_row': ret['representations']['msa_first_row'].detach(),
        'prev_pair': ret['representations']['pair'].detach(),
      }

    def call_af_iter(prev, recycle_idx, comp_loss, num_iter):
      if self.config.resample_msa_in_recycling:
        num_ensemble = batch_size // (num_iter + 1)
        
        def slice_recycle_idx(x):
          # take the next 'num_ensemble' length slice from the first dim of each array in dict
          return x[recycle_idx * num_ensemble:(1+recycle_idx) * num_ensemble]
          
        ensembled_batch = tree_map(slice_recycle_idx, batch)
      else:
        # num_ensemble = batch_size
        ensembled_batch = batch
      args = [ensembled_batch, prev, self.is_training]
      kwargs = {'compute_loss':comp_loss, 'ensemble_representations':self.ensemble_representations}
      if not comp_loss:
        with torch.no_grad():
          outp = self.alphafold_iteration(*args, **kwargs)
        return outp
      return self.alphafold_iteration(*args, **kwargs)

    if self.config.num_recycle:
      emb_config = self.config.embeddings_and_evoformer
      prev = {
        # 'prev_pos': torch.zeros(num_residues, residue_constants.atom_type_num, 3),
        'prev_pos': torch.zeros(num_residues, 3, device=self.global_config.device),
        'prev_msa_first_row': torch.zeros(num_residues, emb_config.msa_channel, device=self.global_config.device),
        'prev_pair': torch.zeros(num_residues, num_residues, emb_config.pair_channel, device=self.global_config.device),
      }

      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # The value for each ensemble batch is the same, so arbitrarily taking
        # 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that we will not run more
        # recyclings than the model is configured to run.
        # num_iter = min(num_iter, self.config.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = self.config.num_recycle
      
      for i in range(num_iter-1):
        prev = get_prev(call_af_iter(prev, i, False, num_iter))

    else:
      prev = {}
      num_iter = 0

    ret = call_af_iter(prev, num_iter, self.compute_loss, num_iter)
    if self.compute_loss:
      ret = ret[0], [ret[1]], ret[2]

    if not self.return_representations:
      del (ret[0] if self.compute_loss else ret)['representations']  # pytype: disable=unsupported-operands
    return ret

  def track_codes(self, track, freq, rec_t_coords_every=5, train_coord_buffer_size=50):
    self.n_fails = 0
    self.train_set_metric_codes = set(track['train_metrics'])
    self.train_metrics = []
    self.v_coords = {c:[] for c in track['val_coords_to_track']}
    self.t_coords = {c:[] for c in track['train_coords_to_track']}
    self.track_all = []
    self.dircts = {
      'val_coords': ('/val_coords', 'coords', 0),
      'train_coords': ('/train_coords', 'coords', 0),
      'train': ('/train', 'losses', 0),
      'train_metrics': ('/train_metrics', 'metrics', 0),
      'validation': ('/validation', 'metrics', 0),
    }
    self.log_freq = freq
    self.rec_t_coords_every = rec_t_coords_every
    self.train_coord_buffer_size = train_coord_buffer_size
    self.sup = SVDSuperimposer()

    self.epo = 0
  
  def save_data(self, key, data):
    for _,(d,_,_) in self.dircts.items():
      if not os.path.isdir(self.trainer.log_dir + d): 
        os.mkdir(self.trainer.log_dir + d)

    _dir, name, curr_ix = self.dircts[key]
    f = open('%s/%s%d.json'%(self.trainer.log_dir + _dir, name, curr_ix), 'w')
    f.write(json.dumps(data))
    f.close()
    self.dircts[key] = (_dir, name, curr_ix+1)
  
  def tidy_data(self):
    ''' go through data dirs and open, collect, jsonify, compress, save, delete old files '''
    # left overs..
    coords = {code:[(i,c.tolist()) for i,c in tenlist] for code, tenlist in self.v_coords.items()}
    self.save_data('val_coords', coords)
    convert = lambda lt: [(e,i,c.tolist(),evd.tolist()) for e,i,c,evd in lt]
    coords = {code:convert(tenlist) for code, tenlist in self.t_coords.items()}
    self.save_data('train_coords', coords)

    print('Tidying logs...', end=' ')
    all_filenames = []

    def collect_coords(scrds, key):
      coord_dir, name, curr_ix = self.dircts[key]
      crds = {c:[] for c in scrds}
      filenames = ['%s/%s%d.json'%(self.trainer.log_dir + coord_dir, name, i) for i in range(curr_ix)]
      for fn in filenames:
        saved = json.loads(open(fn, 'r').read())
        for c in scrds:
          crds[c].extend(saved[c])
      all_filenames.extend(filenames)
      return crds
    
    def collect_dicts(key, extend=True):
      train_dir, name, curr_ix = self.dircts[key]
      lst = []
      filenames = ['%s/%s%d.json'%(self.trainer.log_dir + train_dir, name, i) for i in range(curr_ix)]
      for fn in filenames:
        if extend:
          lst.extend( json.loads(open(fn, 'r').read()) )
        else:
          lst.append( json.loads(open(fn, 'r').read()) )
      all_filenames.extend(filenames)
      return lst

    all_records = {
      'train' : collect_dicts('train'),
      'train_metrics' : collect_dicts('train_metrics', extend=False),
      'valid_epochs' : collect_dicts('validation', extend=False),
      'val_coords' : collect_coords(self.v_coords, 'val_coords'),
      'train_coords' : collect_coords(self.t_coords, 'train_coords')
    }

    comp = gzip.compress(bytes(json.dumps(all_records), encoding='utf-8'))
    f = gzip.open(self.trainer.log_dir + '/records.gz', 'wb')
    f.write(comp)
    f.close()
    for fn in all_filenames: os.remove(fn)
    print('Done.')

  def training_step(self, train_batch, batch_idx):
    # pass in the chain-code
    code = train_batch['code']
    # del train_batch['code']

    out, [loss], losses = self.forward(train_batch)

    # add in other losses, e.g. FAPE and any metrics.
    self.track_all.append((batch_idx, code, loss.item(),
                           losses['masked_msa'], losses['distogram'], 
                           losses['structure_module']))

    if self.epo%self.rec_t_coords_every==0 and code in self.t_coords:
      coords = out['structure_module']['final_frame_pos'].data#.detach()
      logits = out['distogram']['logits'].data
      bins = out['distogram']['bin_edges'].data
      # reconstruct dist matrix from logits and bins
      p = torch.exp(logits)
      p /= p.sum(-1)[...,None]
      diff = bins[1]-bins[0]
      b0 = torch.tensor([bins[0] - diff], device=self.global_config.device)
      bins = torch.cat((b0, bins), dim=0) - diff*0.5
      # in the distogram head- the one-hot enc is based on (dist > breaks).sum(-1)
      # so we must reverse the breaks (bins) so that it multiplies the appropriate indices
      evo_disto = torch.einsum('ijk,k->ij', logits, torch.flip(bins, (0,))).data

      self.t_coords[code].append((self.epo, batch_idx, coords, evo_disto))

      if sum(len(v) for _,v in self.t_coords.items()) >= self.train_coord_buffer_size:
        convert = lambda lt: [(e,i,c.tolist(),evd.tolist()) for e,i,c,evd in lt]
        coords = {code:convert(tenlist) for code, tenlist in self.t_coords.items()}
        # save data
        self.save_data('train_coords', coords)
        # reset the dict
        self.t_coords = {c:[] for c in self.t_coords}

    if len(self.track_all)>=self.log_freq:
      # save the losses etc.
      self.save_data('train', self.track_all)
      # reset the list
      self.track_all = []

    if code in self.train_set_metric_codes:
      try:
        tm, rmsd, gdt, lddt = get_metrics(train_batch, out, self.sup, self.global_config.device)
      except:
        tm, rmsd, gdt, lddt = [None]*4
        self.n_fails += 1
      self.train_metrics.append((batch_idx, code, tm, rmsd, gdt, lddt))

    if self.recall is not None:
      # batch_idx resets on each epoch
      if batch_idx//self.recall==0:
        # call scheduler
        # self.epo += int(batch_idx < self.pbatch_idx)
        self.pbatch_idx = batch_idx
        prop = float(batch_idx)/self.num_data_per_epoch
        t = self.epo + prop
        self.lr_schedulers().step(t)

    self.log('train_loss', loss)
    return loss

  def training_epoch_end(self, train_out):
    print('Failed to record %d codes metrics in training step, lr = %s'%(
      self.n_fails, str(self.lr_schedulers().get_last_lr())))
    self.n_fails = 0
    self.save_data('train_metrics', self.train_metrics)
    self.train_metrics = []
    self.epo += 1

  def validation_step(self, val_batch, batch_idx):
    code = val_batch['code']

    out, [loss], losses = self.forward(val_batch)
    self.log('val_loss', loss)

    coords = out['structure_module']['final_frame_pos'].data#.detach()

    if code in self.v_coords:
      # this is the backbone coords and everythin needed to plot
      self.v_coords[code].append((batch_idx, coords))

      # this event happens once near the end of a deterministic validation epoch
      if all(len(v)>0 for _,v in self.v_coords.items()):
        coords = {code:[(i,c.tolist()) for i,c in tenlist] for code, tenlist in self.v_coords.items()}
        # save data
        self.save_data('val_coords', coords)
        # reset the dict
        self.v_coords = {c:[] for c in self.v_coords}
    
    try:
      tm, rmsd, gdt, lddt = get_metrics(val_batch, out, self.sup, self.global_config.device)
    except:
      tm, rmsd, gdt, lddt = [None]*4
      print('%s failed to record metrics in validation step'%code)

    # self.val_metrics.append((code, tm, rmsd, gdt, lddt, loss.item(),
    #       losses['masked_msa'], losses['distogram'], losses['structure_module']))
    return (code, tm, rmsd, gdt, lddt, loss.item(), losses['masked_msa'], 
      losses['distogram'], losses['structure_module'])
  
  def validation_epoch_end(self, val_outs):
    # save the validation metrics
    self.save_data('validation', val_outs)
    # self.val_metrics = []
  
  def test_step(self, test_batch, batch_idx):
    code = test_batch['code']

    out, [loss], losses = self.forward(test_batch)
    self.log('test_loss', loss)

    try:
      tm, rmsd, gdt, lddt = get_metrics(test_batch, out, self.sup, self.global_config.device)
    except:
      tm, rmsd, gdt, lddt = [None]*4
      print('%s failed to record metrics in test step'%code)

    return (code, tm, rmsd, gdt, lddt, loss.item(), losses['masked_msa'], 
      losses['distogram'], losses['structure_module'])

  def test_epoch_end(self, test_out):
    comp = gzip.compress(bytes(json.dumps(test_out), encoding='utf-8'))
    f = gzip.open(self.trainer.log_dir + '/holdout_records.gz', 'wb')
    f.write(comp)
    f.close()

  def set_optim_config(self, cfg, n_epoch=None, num_data_per_epoch=None):
    opt_gr = cfg['optim_groups']
    self.optim_type = cfg['optim_type']
    groups = {k:[] for k in opt_gr}
    for n,p in self.named_parameters():
      added = False
      for keyword in opt_gr:
        if keyword in n:
          groups[keyword].append(p)
          added = True
      if not added:
        groups['default'].append(p)
    self.optim_groups = [{'params':groups[k], **v} for k,v in opt_gr.items() if len(groups[k])]

    self.num_data_per_epoch = num_data_per_epoch
    # self.epo = 0

    self.recall = None
    if 'scheduler' in cfg:
      print('scheduler found')
      if 'num_call_per_epoch' in cfg['scheduler']:
        n = cfg['scheduler']['num_call_per_epoch']
        self.recall = int( float(num_data_per_epoch) / float(n) )
        self.pbatch_idx = -1
        print('scheduler calling at '+str(self.recall))
      else:
        print('scheduler calling every epoch')

      sch = eval(cfg['scheduler']['class'])
      lmb_kw = eval(cfg['scheduler']['kwargs'])
      self.scheduler = (sch, lmb_kw)
    else:
      print('no scheduler found')
      self.scheduler = None

  def configure_optimizers(self, n_epoch=None):
    opt = eval('torch.optim.%s'%self.optim_type)
    optimizer = opt(self.optim_groups)

    if self.scheduler is None:
      return optimizer

    sch, lmb_kw = self.scheduler
    return [optimizer], [sch(optimizer, **lmb_kw)]


def TMLowerBound(pred, true, mask, device):
  ''' the bound of the TM score described in the alphafold paper '''
  predicted_affine = QuatAffine.from_tensor(pred, device)
  # Shape (num_res, 7)
  true_affine = QuatAffine.from_tensor(true, device)


  # print((len(predicted_affine.quaternion),
  #   predicted_affine.quaternion[0].shape, 
  #   predicted_affine.translation[0].shape, 
  #   len(true_affine.quaternion),
  #   true_affine.quaternion[0].shape, 
  #   true_affine.translation[0].shape))
  # (predicted_affine, true_affine)
  # (69, torch.Size([4]), torch.Size([69]), 2, torch.Size([69, 4]), torch.Size([2, 69]))
  
  # Shape (num_res, num_res)
  square_mask = mask[:, None] * mask[None, :]



  # (mask.shape, square_mask.shape)
  # (torch.Size([2, 69]), torch.Size([2, 2, 69]))

  # print('(mask.shape, square_mask.shape)')  
  # print((mask.shape, square_mask.shape))  

  # num_bins = self.config.num_bins
  # # (1, num_bins - 1)
  # breaks = value['predicted_aligned_error']['breaks']
  # # (1, num_bins)
  # logits = value['predicted_aligned_error']['logits']

  # Compute the squared error for each alignment.
  def lfp(affine):
    ''' local frame points'''
    return affine.invert_point([x[...,None,:] for x in affine.translation], extra_dims=1)

  error_dist2 = 0
  for a, b in zip(lfp(predicted_affine), lfp(true_affine)):
    error_dist2 += (a - b)**2# summing over R3, pythag
  
  # error_dist2
  # torch.Size([2, 69, 69])
  
  # Shape (num_res, num_res)
  # First num_res are alignment frames, second num_res are the residues.

  # e = square_mask * error_dist2



  num_res = error_dist2.shape[0]
  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
  # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  # Yang & Skolnick "Scoring function for automated
  # assessment of protein structure template quality" 2004
  d0 = 1.24 * (max(num_res, 19) - 15) ** (1./3) - 1.8

  f = lambda d2:1/(1 + d2/(d0**2))

  # f(error_dist2).shape,square_mask.shape )
  # (torch.Size([2, 69, 69]), torch.Size([2, 2, 69]))
  return torch.max((f(error_dist2) * square_mask).sum(-1) / num_res)

def apx_lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False, 
         device='cpu'):
  """Measure (approximate) lDDT for a batch of coordinates.

  lDDT reference:
  Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
  superposition-free score for comparing protein structures and models using
  distance difference tests. Bioinformatics 29, 2722–2728 (2013).

  lDDT is a measure of the difference between the true distance matrix and the
  distance matrix of the predicted points.  The difference is computed only on
  points closer than cutoff *in the true structure*.

  This function does not compute the exact lDDT value that the original paper
  describes because it does not include terms for physical feasibility
  (e.g. bond length violations). Therefore this is only an approximate
  lDDT score.

  Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.

  Returns:
    An (approximate, see above) lDDT score in the range 0-1.
  """

  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.
  dmat_true = (1e-10 + ((true_points[:, :, None] - true_points[:, None, :])**2).sum(-1))**0.5

  dmat_predicted = (1e-10 + (
      (predicted_points[:, :, None] - predicted_points[:, None, :])**2).sum(-1))**0.5

  dists_to_score = (
      (dmat_true < cutoff).float() * true_points_mask *
      true_points_mask.permute(0,2,1) * (1. - torch.eye(dmat_true.shape[1], device=device))  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = torch.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0.25 * ((dist_l1 < 0.5).float() +
                  (dist_l1 < 1.0).float() +
                  (dist_l1 < 2.0).float() +
                  (dist_l1 < 4.0).float())

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + dists_to_score.sum(reduce_axes))
  score = norm * (1e-10 + (dists_to_score * score).sum(reduce_axes))

  return score

def gdt_ts(predicted_points,
           true_points,
           true_points_mask,
           cutoffs=(1,2,4,8),
           per_residue=False,
           device='cpu'):
  """Measure GDT-TS for a batch of coordinates.

  https://predictioncenter.org/casp14/doc/help.html#GDT_TS

  Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.

  Returns:
    An (approximate, see above) lDDT score in the range 0-1.
  """

  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.
  dmat_true = (1e-10 + ((true_points[:, :, None] - true_points[:, None, :])**2).sum(-1))**0.5

  dmat_predicted = (1e-10 + (
      (predicted_points[:, :, None] - predicted_points[:, None, :])**2).sum(-1))**0.5

  dists_to_score = (
      true_points_mask *
      true_points_mask.permute(0,2,1) * (1. - torch.eye(dmat_true.shape[1], device=device))  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = torch.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0
  for c in cutoffs:
    score += (dist_l1 < c).float()
  score /= len(cutoffs)

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + dists_to_score.sum(reduce_axes))
  score = norm * (1e-10 + (dists_to_score * score).sum(reduce_axes))

  return score

def get_metrics(batch, out, sup, device):
  # the first dim is a repeat dim
  true = batch['backbone_affine_tensor'].data[0]
  mask = batch['backbone_affine_mask'].data[0]
  # coords of the origin atom
  true_coords = batch['pseudo_beta'].data[0]
  mask = batch['pseudo_beta_mask'].data[0]
  
  # TM-score (apx)
  pred = out['structure_module']['final_affines'].data
  coords = out['structure_module']['final_frame_pos'].data

  # (pred.shape, true.shape, mask.shape)
  # (torch.Size([69, 7]), torch.Size([2, 69, 7]), torch.Size([2, 69]))

  tm = TMLowerBound(pred, true, mask, device)
  
  # print('(true_coords.shape, coords.shape)')
  # print((true_coords.shape, coords.shape))

  # RMSD
  sup.set(true_coords[mask==1].clone().cpu().numpy(), coords[mask==1].clone().cpu().numpy())
  sup.run()
  rmsd = sup.get_rms()

  # GDT
  gdt = gdt_ts(coords[None,...], true_coords[None,...], mask[None,...,None], device=device)

  # LDDT
  lddt = apx_lddt(coords[None,...], true_coords[None,...], mask[None,...,None], device=device)
  return tm.item(), rmsd, gdt.item(), lddt.item()