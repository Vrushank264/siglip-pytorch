### SigLIP-style ViT + Qwen3 (PyTorch)

I reimplemented Vision Transformer (ViT) and Qwen3 from scratch and trained a SigLIP-style image–text contrastive model on CC3M. ViT is used as the image encoder and Qwen3 as the text encoder; both are followed by projection heads and trained with a sigmoid contrastive loss.

- **From scratch**: custom ViT and Qwen3 implementations
- **Architecture**: ViT-B/16 image encoder + Qwen3-0.6B text encoder, mean-pooled, then projected and L2-normalized
- **Training**: multi-GPU DDP, grad accumulation, bfloat16 autocast, cosine LR schedule
- **Data**: CC3M (loaded via hugginface's  `datasets`, pre-sharded WebDataset-style)
- **Result**: 71.4% R@1 on CC3M validation after 10 epochs on 4× L4 GPUs

### Install

Use any recent Python + CUDA PyTorch. Minimal deps:

```bash
pip install torch torchvision transformers datasets wandb safetensors huggingface_hub tqdm
```

### Data

No manual download needed. The loader pulls CC3M shards from `pixparse/cc3m-wds` and handles images/text:
- Images are resized to 224×224 and normalized to [-1, 1]
- Text is tokenized with `Qwen/Qwen3-0.6B` (pad/truncate to 77 tokens)

If you want to control cache locations, set `HF_HOME` / `HF_DATASETS_CACHE`.

### Train

I trained for 10 epochs on 4× L4 GPUs. Launch with `torchrun` (DDP):

```bash
torchrun --nproc_per_node=4 -m tools.train_siglip_wandb \
  --epochs 10 \
  --batch-size 16 \
  --accumulation-steps 8 \
  --embed-dim 768 \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --output-dir checkpoints \
  --wandb-project "SigLIP on CC3M"
```

Notes:
- Weights are initialized from `google/vit-base-patch16-224` and `Qwen/Qwen3-0.6B` and then trained end‑to‑end.
- Validation runs every epoch; the script logs loss and R@1. Best checkpoint is saved to `checkpoints/best.pt`.
- Effective LR is scaled by world size × batch size × accumulation steps (target 256 global batch baseline).

### Repo layout

- `models/vision/vision_transformer.py`: ViT encoder (patch embed, MHSA, MLP, pre-norm)
- `models/text/qwen3.py`: Qwen3 decoder-only LM with rotary embeddings and GQA
- `models/siglip.py`: wrapper that pools/normalizes and projects image/text features
- `data/cc3m.py`: CC3M dataset (images + captions)
- `tools/train_siglip_wandb.py`: DDP trainer with validation and WandB logging

### Result

- 10 epochs on 4× L4 → **71.4% R@1** on CC3M validation

### Acknowledgements

- ViT weights: `google/vit-base-patch16-224`
- Qwen3 weights/tokenizer: `Qwen/Qwen3-0.6B`
- Data: `pixparse/cc3m-wds`
