import torch
import torch.nn as nn
import math

from models.vision.vision_transformer import ViTModel
from models.text.qwen3 import Qwen3ForSigLIP
from configs.config import VisionConfig, Qwen3Config

class SigLIP(nn.Module):
    """A simplified SigLIP model wrapping vision & text encoders with projection heads."""

    def __init__(self,
                 vision_cfg: VisionConfig,
                 text_cfg: Qwen3Config,
                 embed_dim: int = 512):
        super().__init__()
        self.vision_encoder = ViTModel(vision_cfg)
        self.text_encoder = Qwen3ForSigLIP(text_cfg)
        self.vision_proj = nn.Linear(vision_cfg.hidden_size, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_cfg.hidden_size, embed_dim, bias=False)
        

    def forward(self, pixel_values, input_ids, attention_mask):
        """Return L2-normalised image & text embeddings."""
        # Vision branch (CLS token from last layer)
        seq_img, _ = self.vision_encoder(pixel_values)
        img_feat = seq_img[:, 0]  # [B, H]
        img_emb = nn.functional.normalize(self.vision_proj(img_feat), dim=-1)

        # Text branch (mean-pool token embeddings)
        txt_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_feat.mean(dim=1)  # [B, H]
        txt_emb = nn.functional.normalize(self.text_proj(txt_feat), dim=-1)
        
        return img_emb, txt_emb