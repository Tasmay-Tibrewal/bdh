import os
from contextlib import nullcontext

import numpy as np
import torch

import bdh

MODEL_SIZE_PRESETS = {
    "25m": {
        "n_layer": 6,
        "n_embd": 256,
        "n_head": 4,
        "mlp_internal_dim_multiplier": 128,
    },
    "100m": {
        "n_layer": 8,
        "n_embd": 512,
        "n_head": 8,
        "mlp_internal_dim_multiplier": 128,
    },
}


def build_config(
    model_size=None,
    n_layer=None,
    n_embd=None,
    n_head=None,
    mlp_mult=None,
    dropout=None,
    vocab_size=None,
):
    config = bdh.BDHConfig()
    if model_size:
        if model_size not in MODEL_SIZE_PRESETS:
            raise ValueError(f"Unknown model_size: {model_size}")
        for key, value in MODEL_SIZE_PRESETS[model_size].items():
            setattr(config, key, value)
    if n_layer is not None:
        config.n_layer = n_layer
    if n_embd is not None:
        config.n_embd = n_embd
    if n_head is not None:
        config.n_head = n_head
    if mlp_mult is not None:
        config.mlp_internal_dim_multiplier = mlp_mult
    if dropout is not None:
        config.dropout = dropout
    if vocab_size is not None:
        config.vocab_size = vocab_size
    return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def setup_device(dtype_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype_override:
        dtype = dtype_override
    else:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        elif device.type == "cuda":
            dtype = "float16"
        else:
            dtype = "float32"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(dtype == "float16" and device.type == "cuda")
    )
    return device, dtype, ctx, scaler


class MemmapDataset:
    def __init__(self, path, block_size):
        self.path = path
        self.block_size = block_size
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def get_batch(self, batch_size, device):
        max_idx = len(self.data) - self.block_size - 1
        if max_idx <= 0:
            raise ValueError(
                f"Not enough data in {self.path} for block_size={self.block_size}"
            )
        ix = torch.randint(max_idx, (batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i : i + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters, ctx, device, batch_size):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size=batch_size, device=device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def maybe_compile(model, enable_compile):
    if enable_compile and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
