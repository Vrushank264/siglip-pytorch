import os
import math
import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from configs.configs import VisionConfig, Qwen3Config
from models.siglip import SigLIP
# new HF dataset loader
from data.cc3m import CC3MDataset


def init_distributed_mode() -> int:
    """Initialise torch distributed and return the local rank."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but distributed training was requested")
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process() -> bool:
    """True for rank-0 process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0



class SigmoidLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        logits = image_features @ text_features.t()
        logits = logits * self.logit_scale.exp() + self.bias
        labels = torch.eye(logits.size(0), dtype=logits.dtype, device=logits.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss


def gather_features(image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return image_features, text_features

    img_list = [torch.zeros_like(image_features) for _ in range(world_size)]
    txt_list = [torch.zeros_like(text_features) for _ in range(world_size)]

    dist.all_gather(img_list, image_features.detach())
    dist.all_gather(txt_list, text_features.detach())

    img_all = torch.cat(img_list, dim=0)
    txt_all = torch.cat(txt_list, dim=0)
    return img_all, txt_all


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Run one validation epoch and return metrics dict."""
    model.eval()
    total_loss = 0.0
    correct_r1 = 0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not is_main_process(), desc="Val", dynamic_ncols=True)
        for images, input_ids, attention_mask in pbar:
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                img_emb, txt_emb = model(images, input_ids, attention_mask)
                img_all, txt_all = gather_features(img_emb, txt_emb)
                loss = criterion(img_all, txt_all)

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            # Retrieval R@1 on-the-fly (across gathered batch)
            logits = img_all @ txt_all.t()
            preds = logits.argmax(dim=1)
            target = torch.arange(logits.size(0), device=logits.device)
            correct_r1 += (preds == target).sum().item()

        avg_loss = total_loss / max(num_samples, 1)
        r1 = correct_r1 / max(num_samples, 1)

    # Ensure all ranks have the same metrics
    if dist.is_initialized():
        tensor_metrics = torch.tensor([avg_loss, r1], device=device)
        dist.all_reduce(tensor_metrics, op=dist.ReduceOp.SUM)
        tensor_metrics /= dist.get_world_size()
        avg_loss, r1 = tensor_metrics.tolist()

    return {"val_loss": avg_loss, "val_r1": r1}


def train(args):
    local_rank = init_distributed_mode()
    device = torch.device("cuda", local_rank)

    # Logger (only rank-0 prints to stdout)
    logging.basicConfig(level=logging.INFO if is_main_process() else logging.WARNING,
                        format="%(asctime)s — %(levelname)s — %(message)s",
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger("siglip_train")

    # Seed
    torch.manual_seed(42 + local_rank)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    train_set = CC3MDataset(tokenizer=tokenizer, split='train')
    val_set = CC3MDataset(tokenizer=tokenizer, split='validation')

    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    if val_set:
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(val_set,
                                batch_size=args.val_batch_size or args.batch_size,
                                sampler=val_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=False)
    else:
        val_loader = None

    vision_cfg = VisionConfig()
    text_cfg = Qwen3Config()
    model = SigLIP(vision_cfg, text_cfg, embed_dim=args.embed_dim).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    criterion = SigmoidLoss().to(device)
    lr = args.lr * args.batch_size * dist.get_world_size() / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    if is_main_process() and args.wandb_project:
        wandb.init(project=args.wandb_project,
                   config=vars(args))

    accumulation_steps = args.accumulation_steps
    global_step = 0
    best_eval_r1 = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, disable=not is_main_process(), dynamic_ncols=True)
        for step, (images, input_ids, attention_mask) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            # Use no_sync for all accumulation steps except the last one
            if (step + 1) % accumulation_steps != 0:
                with model.no_sync():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        img_emb, txt_emb = model(images, input_ids, attention_mask)
                        img_all, txt_all = gather_features(img_emb, txt_emb)
                        loss = criterion(img_all, txt_all) / accumulation_steps

                    loss.backward()
                    running_loss += loss.item()
            else:
                # Last accumulation step - allow gradient sync
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    img_emb, txt_emb = model(images, input_ids, attention_mask)
                    img_all, txt_all = gather_features(img_emb, txt_emb)
                    loss = criterion(img_all, txt_all) / accumulation_steps

                loss.backward()
                running_loss += loss.item()

                # Update parameters after accumulation is complete
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                if is_main_process() and args.wandb_project:
                    wandb.log({"train_loss": running_loss * accumulation_steps}, step=global_step)
                running_loss = 0.0
                global_step += 1

            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}")

        if val_loader and ((epoch + 1) % args.val_every == 0):
            metrics = validate(model, val_loader, criterion, device)
            if is_main_process():
                logger.info(f"Val — loss: {metrics['val_loss']:.4f} | R@1: {metrics['val_r1']*100:.2f}%")
                if args.wandb_project:
                    wandb.log(metrics, step=global_step)

                if metrics['val_r1'] > best_eval_r1:
                    best_eval_r1 = metrics['val_r1']
                    ckpt_dir = Path(args.output_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"best.pt"
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }, ckpt_path)
                    logger.info(f"New best R@1: {best_eval_r1*100:.2f}%. Saved checkpoint to {ckpt_path}")

        if is_main_process() and (epoch + 1) % args.save_every == 0:
            ckpt_dir = Path(args.output_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"latest.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    if is_main_process() and args.wandb_project:
        wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description="Train SigLIP on CC3M with WandB logging and validation")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--val-batch-size", type=int, default=8, help="Validation batch size (defaults to train BS)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--wandb-project", type=str, default="SigLIP on CC3M", help="Weights&Biases project name (if None, wandb is disabled)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args) 