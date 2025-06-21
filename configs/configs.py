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