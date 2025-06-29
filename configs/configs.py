from dataclasses import dataclass


@dataclass
class SigLIPConfig:
    num_hidden_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_channels: int = 3
    max_position_embeddings: int = 512
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-12


@dataclass
class VisionConfig:
    num_hidden_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_channels: int = 3
    max_position_embeddings: int = 512
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 196
    layer_norm_eps: float = 1e-12
    use_dropout: bool = True
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_bias: bool = False
    use_cls_token: bool = True
    use_position_embedding: bool = True


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    head_dim: int = 128
    num_hidden_layers: int = 28
    num_attention_heads: int = 16 
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    num_kv_heads: int = 8
    torch_dtype: str = "bfloat16"
    bos_token_id: int = 151643
    eos_token_id: int = 151645