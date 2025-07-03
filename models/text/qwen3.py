"""
Qwen3-0.6B
Architecture:

Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)

"""

import torch
import torch.nn as nn
from typing import Optional

from models.text.rope import apply_rotary_pos_emb
from configs.configs import Qwen3Config


class Qwen3MLP(nn.Module):
  """
  Gated MLP 
  Helps in controlling the flow of information
  """
  def __init__(self, config: Qwen3Config):
    super().__init__()
    self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
    self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    self.act_fn = nn.SiLU()
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))


class Qwen3RMSNorm(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.variance_eps = 1e-6

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    if isinstance(x, tuple):
      ip_dtype = x[0].dtype
      x = x[0]
    else:
      ip_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True) 
    x = x * torch.rsqrt(variance + self.variance_eps)
    return self.weight * x.to(ip_dtype)
  

class Qwen3Attention(nn.Module):
  """
  Multi-head attention with rotary embeddings and GQA
  """
  def __init__(self, config: Qwen3Config, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.num_heads = config.num_attention_heads
    self.num_kv_groups = config.num_attention_heads // self.num_kv_heads
    self.scaling = self.head_dim ** -0.5
    self.attention_dropout = config.attention_dropout

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    self.q_norm = Qwen3RMSNorm(self.head_dim)
    self.k_norm = Qwen3RMSNorm(self.head_dim)



  def _repeat_kv(
    self,
    hidden_states: torch.Tensor,
    num_repeats: int,
  ):
    """
    Repeat hidden states for each group of key-value heads
    """
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if num_repeats == 1:
      return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, num_repeats, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * num_repeats, seq_len, head_dim)


  def forward(self, x: torch.Tensor, 
              position_embeddings: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    
    ip_shape = x.shape[:-1]
    hidden_shape = (*ip_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(x).view(hidden_shape).transpose(1, 2))
    key_states = self.k_norm(self.k_proj(x).view(hidden_shape).transpose(1, 2))
    value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Self-attention 
    key_states = self._repeat_kv(key_states, self.num_kv_groups)
    value_states = self._repeat_kv(value_states, self.num_kv_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
      causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
      attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).reshape(*ip_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
    

class Qwen3DecoderLayer(nn.Module):
   def __init__(self, config: Qwen3Config, layer_idx: int):
      super().__init__()
      self.layer_idx = layer_idx
      self.self_attn = Qwen3Attention(config, layer_idx)
      self.mlp = Qwen3MLP(config)
      self.input_layernorm = Qwen3RMSNorm(config.hidden_size)
      self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size)

   def forward(self, 
              x: torch.Tensor,
              position_embeddings: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              output_attentions: bool = False) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
      
      residual = x
      x = self.input_layernorm(x)
      x, attn_weights = self.self_attn(x, position_embeddings, attention_mask)
      x = residual + x
      residual = x
      x = self.post_attention_layernorm(x)
      x = self.mlp(x)
      x = residual + x

      return x
  

class Qwen3RotaryEmbedding(nn.Module):
   def __init__(self, config: Qwen3Config):
      super().__init__()
      self.head_dim = config.head_dim
      self.max_position_embeddings = config.max_position_embeddings
      self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)), persistent=False)
   
   @torch.no_grad()
   def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      expanded_inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1).to(x.device)
      position_ids_expanded = position_ids[:, None, :].float()

      with torch.autocast(device_type=x.device.type, enabled=False):
         freqs = (expanded_inv_freq.float() @ position_ids_expanded.float()).transpose(1, 2)
         emb = torch.cat((freqs, freqs), dim=-1)
         cos = emb.cos() 
         sin = emb.sin()

      return cos.to(x.dtype), sin.to(x.dtype)


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):

        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(x, position_ids)

        causal_mask = build_causal_attention_mask(seq_len, bsz, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, pos_emb, causal_mask)
        x = self.norm(x)
        return x
    
class Qwen3ForSigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.model(input_ids, attention_mask)
        return x
    

def build_causal_attention_mask(seq_len: int,
                                batch_size: int,
                                device,
                                dtype=torch.float32) -> torch.Tensor:
    #  shape: (batch, 1, seq_len, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)          # hide future tokens
    mask = mask.unsqueeze(0).unsqueeze(0)        # (1,1,L,L)
    mask = mask.expand(batch_size, -1, -1, -1)   # (B,1,L,L)
    return mask
   

if __name__ == "__main__":
   # Load the state dict from huggingface
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_name = "Qwen/Qwen3-0.6B"
   config = Qwen3Config()
   model = Qwen3ForSigLIP(config)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
   state_dict = pretrained_model.state_dict()
   state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head.")}

   model.load_state_dict(state_dict, strict=False)
   print(model)
   print("Model parameters:")
   print(sum(p.numel() for p in model.parameters()))

   # Test the model
   input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")
   attention_mask = torch.ones_like(input_ids)
   output = model(input_ids, attention_mask)
   print(output.shape)
   
    