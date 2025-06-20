import torch
import torch.nn as nn

from configs.configs import VisionConfig
from models.blocks import MLP


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.hidden_size))
        if config.use_dropout:
            self.dropout = nn.Dropout(config.dropout_rate)
        else:
            self.dropout = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        # pixel_values: [B,C,H,W]
        embeddings = self.patch_embeddings(pixel_values) # [B, hidden_size, H/patch_size, W/patch_size]
        embeddings = embeddings.flatten(2) # [B, hidden_size, H/patch_size * W/patch_size]
        embeddings = embeddings.transpose(1, 2) # [B, H/patch_size * W/patch_size, hidden_size]
        cls = self.cls_token.expand(pixel_values.size(0), 1, self.config.hidden_size)
        embeddings = torch.cat([cls, embeddings], dim=1)
        embeddings = embeddings + self.position_embeddings # [B, Num_patches, hidden_size]
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return embeddings
    
class SiglipVisionEncoderLayer(nn.Module):

    def __init__(self, config: VisionConfig):

        super().__init__()
        self.config = config
        self.prenorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout_rate,
            batch_first=True,
            bias=False,
        )
        self.postnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_norm = self.prenorm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        residual = x + attn_output
        x = self.postnorm(residual)
        x = self.mlp(x)
        output = x + residual

        return output


class SiglipVisionEncoder(nn.Module):

    def __init__(self, config: VisionConfig):

        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder_layers = nn.ModuleList([SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        embeddings = self.embeddings(pixel_values)
        for encoder_layer in self.encoder_layers:
            embeddings = encoder_layer(embeddings)
        output = self.layernorm(embeddings)

        return output
    

def test_siglip_vision_encoder():

    config = VisionConfig()
    encoder = SiglipVisionEncoder(config).to("cuda")

    x = torch.randn(1, 3, 224, 224).to("cuda")
    print(encoder(x).shape)

    # print number of parameters
    print(sum(p.numel() for p in encoder.parameters()))


test_siglip_vision_encoder()