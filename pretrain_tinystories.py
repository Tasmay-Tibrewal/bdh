import argparse
import dataclasses
import math
import os
import time

import numpy as np
import torch
import wandb

import bdh
from data_utils import build_tinystories_bins
from train_utils import (
    MemmapDataset,
    build_config,
    count_parameters,
    estimate_loss,
    maybe_compile,
    setup_device,
    steps_per_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain BDH on TinyStories v2.")
    parser.add_argument(
        "--dataset",
        default="noanabeshima/TinyStoriesV2",
        help="Hugging Face dataset path.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset config/subset name (optional).",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--data-dir", default="data/tinystories_v2_gpt4")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)

    parser.add_argument("--model-size", choices=["25m", "100m"], default="25m")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--mlp-mult", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--max-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--out-dir", default="checkpoints/pretrain_tinystories")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--resume-from", default=None, help="Path to checkpoint .pt to resume training."
    )

    parser.add_argument("--wandb-project", default="bdh-pretrain")
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity/team (defaults to your logged-in user).",
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (set offline if you want to sync later).",
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grad_acc_steps < 1:
        raise ValueError("--grad-acc-steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset_config = args.dataset_config
    if dataset_config is not None:
        stripped = dataset_config.strip().lower()
        if stripped in ("", "none", "null", "default"):
            dataset_config = None

    train_bin, val_bin, _meta = build_tinystories_bins(
        out_dir=args.data_dir,
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        train_split=args.train_split,
        val_split=args.val_split,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        force=args.rebuild_data,
    )

    device, dtype, ctx, scaler = setup_device(args.dtype)
    config = build_config(
        model_size=args.model_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
    )
    ckpt = None
    start_step = 0
    best_val_loss = None
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        ckpt_config = ckpt.get("config")
        if isinstance(ckpt_config, dict):
            config = bdh.BDHConfig(**ckpt_config)
        elif dataclasses.is_dataclass(ckpt_config):
            config = ckpt_config
        start_step = int(ckpt.get("step", 0))
        best_val_loss = ckpt.get("best_val_loss", None)
        print(f"resuming from {args.resume_from} at step {start_step}")

    model = bdh.BDH(config).to(device)
    if ckpt and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    model = maybe_compile(model, args.compile)

    print("Initialising optimiser with learning rate:", args.learning_rate)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if ckpt and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    param_count = count_parameters(model)
    print(f"Using device: {device} dtype={dtype} params={param_count:,}")

    effective_batch_size = args.batch_size * args.grad_acc_steps
    steps_per_epoch_value = steps_per_epoch(
        train_bin, args.block_size, effective_batch_size
    )
    if args.max_epochs is not None:
        target_steps = int(math.ceil(args.max_epochs * steps_per_epoch_value))
    else:
        target_steps = args.max_steps
    if start_step:
        max_steps = start_step + target_steps
    else:
        max_steps = target_steps

    use_wandb = not args.no_wandb and args.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "computed_max_steps": max_steps,
                "effective_batch_size": effective_batch_size,
                "resume_step": start_step,
                "steps_per_epoch": steps_per_epoch_value,
                "model_config": dataclasses.asdict(config),
                "param_count": param_count,
                "device": str(device),
            },
        )

    train_data = MemmapDataset(train_bin, args.block_size)
    val_data = MemmapDataset(val_bin, args.block_size)

    loss_acc = 0.0
    loss_steps = 0
    start_time = time.time()
    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_acc_steps):
            x, y = train_data.get_batch(args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_acc_steps
            scaler.scale(loss).backward()
            loss_acc += loss.item() * args.grad_acc_steps
            loss_steps += 1
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            avg_loss = loss_acc / max(1, loss_steps)
            elapsed = time.time() - start_time
            print(f"step {step} train_loss={avg_loss:.4f} time={elapsed:.1f}s")
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/step_time_sec": elapsed / args.log_interval,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
            loss_acc = 0.0
            loss_steps = 0
            start_time = time.time()

        if step % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_data,
                eval_iters=args.eval_iters,
                ctx=ctx,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"step {step} val_loss={val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.out_dir, exist_ok=True)
                best_path = os.path.join(args.out_dir, "ckpt_best.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": dataclasses.asdict(config),
                        "step": step,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
                print(f"saved best checkpoint to {best_path}")

        if args.save_every and step % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{step}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": dataclasses.asdict(config),
                    "step": step,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "ckpt_last.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": dataclasses.asdict(config),
            "step": max_steps,
            "best_val_loss": best_val_loss,
        },
        final_path,
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
