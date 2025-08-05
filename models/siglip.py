import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models.vision.vision_transformer import ViTModel
from models.text.qwen3 import Qwen3ForSigLIP
from configs.configs import VisionConfig, Qwen3Config
from safetensors.torch import load_file as load_safetensors

class SigLIP(nn.Module):
    """A simplified SigLIP model wrapping vision & text encoders with projection heads."""

    def __init__(self,
                 vision_cfg: VisionConfig,
                 text_cfg: Qwen3Config,
                 embed_dim: int = 512):
        super().__init__()
        self.vision_encoder = ViTModel(vision_cfg)
        self.text_encoder = Qwen3ForSigLIP(text_cfg)
        vision_cfg.pretrained_weights = hf_hub_download(repo_id="google/vit-base-patch16-224", filename="pytorch_model.bin")
        state_dict = torch.load(vision_cfg.pretrained_weights)

        # model surgery to match the custom implementation
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vit."):
                new_key = k[len("vit."):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        self.vision_encoder.load_state_dict(new_state_dict, strict=False)

        text_cfg.pretrained_weights = hf_hub_download(repo_id="Qwen/Qwen3-0.6B", filename="model.safetensors")
        print(f"Loading pretrained text weights from {text_cfg.pretrained_weights}")
        state_dict = load_safetensors(text_cfg.pretrained_weights)
        state_dict.pop("lm_head.weight", None)  

        self.text_encoder.load_state_dict(state_dict, strict=True)
        print("Both Image and Text Encoder loaded successfully!")

        # Projection heads
        self.vision_proj = nn.Linear(vision_cfg.hidden_size, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_cfg.hidden_size, embed_dim, bias=False)
        

    def forward(self, pixel_values, input_ids, attention_mask):
        """Return L2-normalised image & text embeddings."""
        # Vision branch (CLS token from last layer)
        seq_img = self.vision_encoder(pixel_values)
        img_feat = seq_img[:, 0]  # [B, H]
        img_emb = nn.functional.normalize(self.vision_proj(img_feat), dim=-1)

        # Text branch (mean-pool token embeddings)
        txt_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_feat.mean(dim=1)  # [B, H]
        txt_emb = nn.functional.normalize(self.text_proj(txt_feat), dim=-1)
        
        return img_emb, txt_emb
    

if __name__ == "__main__":

    vision_cfg = VisionConfig()
    text_cfg = Qwen3Config()
    model = SigLIP(vision_cfg, text_cfg, embed_dim=768)
    pixel_values = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 77))
    attention_mask = torch.ones_like(input_ids)
    img_emb, txt_emb = model(pixel_values, input_ids, attention_mask)
    print("Image Embeddings:", img_emb.shape)  # (2, embed_dim)
    print("Text Embeddings:", txt_emb.shape)    # (2, embed_dim)