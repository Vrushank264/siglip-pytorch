"""
Train the SigLIP model on CC3M dataset
1. Distributed training on multiple GPUs
2. The Qwen3 will be the text transformer and the normal Vision Transformer will be the image transformer
3. Signoid loss will be used to train the model.

"""


import torch
import torch.nn as nn


class Trainer:

    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: Optimizer, scheduler: Scheduler):
        