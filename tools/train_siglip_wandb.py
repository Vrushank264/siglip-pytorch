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
from models.siglip import SigLIP, load_pretrained_weights

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

    # Placeholders for gathering
    img_list = [torch.zeros_like(image_features) for _ in range(world_size)]
    txt_list = [torch.zeros_like(text_features) for _ in range(world_size)]

    # Gather detached features from all GPUs
    dist.all_gather(img_list, image_features.detach())
    dist.all_gather(txt_list, text_features.detach())

    # Restore the original tensor for the current rank to keep the gradient
    rank = dist.get_rank()
    img_list[rank] = image_features
    txt_list[rank] = text_features

    # Concatenate all features
    img_all = torch.cat(img_list, dim=0)
    txt_all = torch.cat(txt_list, dim=0)
    return img_all, txt_all

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Run one validation epoch and return metrics dict."""
    model.eval()
    total_loss = 0.0
    correct_r1 = 0.0
    num_samples = 0.0
    main_proc = is_main_process()

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

            # Compute metrics once using gathered (global) batch size
            logits = img_all @ txt_all.t()
            batch_global = float(logits.size(0))
            if main_proc:
                # Accumulate loss weighted by global batch to later average per-sample
                total_loss += loss.item() * batch_global
                num_samples += batch_global
                # Retrieval R@1 on-the-fly (across gathered batch)
                preds = logits.argmax(dim=1)
                target = torch.arange(logits.size(0), device=logits.device)
                correct_r1 += float((preds == target).sum().item())

    if dist.is_initialized():
        # Only main process accumulated metrics; sum to share across ranks
        metrics_tensor = torch.tensor([total_loss, correct_r1, num_samples], device=device, dtype=torch.float32)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, correct_r1, num_samples = metrics_tensor.tolist()

    # Compute final ratios from aggregated counts
    avg_loss = total_loss / max(num_samples, 1)
    r1 = correct_r1 / max(num_samples, 1)

    return {"val_loss": avg_loss, "val_r1": r1}


def train(args):
    local_rank = init_distributed_mode()
    device = torch.device("cuda", local_rank)

    logging.basicConfig(level=logging.INFO if is_main_process() else logging.WARNING,
                        format="%(asctime)s — %(levelname)s — %(message)s",
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger("siglip_train")

    torch.manual_seed(264 + local_rank)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Ensure PAD token exists for proper attention masking
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    model = SigLIP(vision_cfg, text_cfg, embed_dim=args.embed_dim)
    
    if is_main_process():
        load_pretrained_weights(model, vision_cfg, text_cfg)
    
    model = model.to(device)
    if dist.is_initialized():
        # Ensure all processes have the same weights
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    criterion = SigmoidLoss().to(device)
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    effective_batch_size = args.batch_size * world_size * args.accumulation_steps
    lr_scale = effective_batch_size / 256  
    lr = args.lr * min(lr_scale, 4.0)  
    
    # Also optimize loss temperature and bias
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": 1e-2},
        {"params": criterion.parameters(), "weight_decay": 0.0},
    ], lr=lr)
    
    warmup_steps = args.warmup_steps
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if is_main_process():
        logger.info(f"Learning rate: {lr:.6f} (scaled from {args.lr:.6f}, effective batch size: {effective_batch_size})")
        logger.info(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

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

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                if is_main_process() and args.wandb_project:
                    log_dict = {
                        "train_loss": running_loss * accumulation_steps,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }
                    wandb.log(log_dict, step=global_step)
                
                if is_main_process() and global_step % 100 == 0:
                    logger.info(f"Step {global_step} | Loss: {running_loss * accumulation_steps:.4f} | Grad norm: {grad_norm:.4f}")
                
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
    parser.add_argument("--val-batch-size", type=int, default=16, help="Validation batch size (defaults to train BS)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for learning rate")
    parser.add_argument("--wandb-project", type=str, default="SigLIP on CC3M", help="Weights&Biases project name (if None, wandb is disabled)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args) 