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

  def __init__(self, config: Qwen3Config):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(config.hidden_size))
    self.variance_eps = 1e-6

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    
    ip_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True) 
    x = x * torch.rsqrt(variance + self.variance_eps)
    return self.weight * x.to(ip_dtype)
  

class Qwen3Attention(nn.Module):
  """
  Multi-head attention with rotary embeddings and GQA
  """
  def __init__(self, config: Qwen3Config):
    super().__init__()
    pass

  def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    pass
  

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):

        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, attention_mask)
        output = self.lm_head(output)
        return output
    

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):

        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.rotary_emb(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return x
    
    