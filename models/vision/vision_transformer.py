"""
Vision Transformer keys from huggingface

ViTModel(
  (embeddings): ViTEmbeddings(
    (patch_embeddings): ViTPatchEmbeddings(
      (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (encoder): ViTEncoder(
    (layer): ModuleList(
      (0-11): 12 x ViTLayer(
        (attention): ViTAttention(
          (attention): ViTSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
          )
          (output): ViTSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (intermediate): ViTIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): ViTOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
    )
  )
  (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (pooler): ViTPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)

"""



import torch
import torch.nn as nn

from configs.configs import VisionConfig


class ViTPatchEmbeddings(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values)


class ViTEmbeddings(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = ViTPatchEmbeddings(config)
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


class ViTSelfAttention(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.use_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.use_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.use_bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        return q, k, v


class ViTSelfOutput(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.use_bias)
        if config.use_dropout:
            self.dropout = nn.Dropout(config.attention_dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTAttention(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q, k, v = self.attention(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        attn_output = self.output(attn_output)
        return attn_output


class ViTIntermediate(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=not config.use_bias)
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=not config.use_bias)
        if config.use_dropout:
            self.dropout = nn.Dropout(config.dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTLayer(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ViTAttention(config)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.layernorm_before(x)
        attn_output = self.attention(x_norm)
        residual = x + attn_output
        
        # Pre-norm feedforward
        x_norm = self.layernorm_after(residual)
        intermediate_output = self.intermediate(x_norm)
        output = self.output(intermediate_output)
        output = output + residual

        return output


class ViTPooler(nn.Module):
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.use_bias)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take the [CLS] token representation
        
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTEncoder(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layer:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class ViTModel(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embeddings)
        sequence_output = self.layernorm(encoder_outputs)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output
    

def test_vit_model():
    config = VisionConfig()
    model = ViTModel(config).to("cuda")

    x = torch.randn(1, 3, 224, 224).to("cuda")
    sequence_output, pooled_output = model(x)
    print(f"Sequence output shape: {sequence_output.shape}")
    print(f"Pooled output shape: {pooled_output.shape}")

    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    load_huggingface_vision_encoder(model)
    sequence_output, pooled_output = model(x)
    print(f"Sequence output shape after loading huggingface: {sequence_output.shape}")
    print(f"Pooled output shape after loading huggingface: {pooled_output.shape}")


def load_huggingface_vision_encoder(model: nn.Module):
    
    from transformers import ViTModel
    hf_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    print(hf_model)
    print(sum(p.numel() for p in hf_model.parameters()))

    # Load the state dict from huggingface
    state_dict = hf_model.state_dict()
    print(state_dict.keys())

    model.load_state_dict(state_dict)
    return model
    


if __name__ == "__main__":
    test_vit_model()