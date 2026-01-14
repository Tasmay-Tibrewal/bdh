import argparse
import dataclasses
import math
import os
import time

import numpy as np
import torch
import wandb

import bdh
from data_utils import build_instruction_bins
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
    parser = argparse.ArgumentParser(
        description="Small-scale supervised finetuning on Alpaca + Dolly."
    )
    parser.add_argument("--alpaca-dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dolly-dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--data-dir", default="data/instruction/alpaca_dolly_small")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-examples", type=int, default=2000)

    parser.add_argument("--model-size", choices=["25m", "100m"], default="25m")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--mlp-mult", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=25)
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--out-dir", default="checkpoints/sft_small")
    parser.add_argument("--save-every", type=int, default=250)

    parser.add_argument("--wandb-project", default="bdh-sft-small")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_bin, val_bin, _meta = build_instruction_bins(
        out_dir=args.data_dir,
        alpaca_dataset=args.alpaca_dataset,
        dolly_dataset=args.dolly_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_examples=args.max_examples,
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

    model = bdh.BDH(config).to(device)
    model = maybe_compile(model, args.compile)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    param_count = count_parameters(model)
    print(f"Using device: {device} dtype={dtype} params={param_count:,}")

    steps_per_epoch_value = steps_per_epoch(
        train_bin, args.block_size, args.batch_size
    )
    if args.max_epochs is not None:
        max_steps = int(math.ceil(args.max_epochs * steps_per_epoch_value))
    else:
        max_steps = args.max_steps

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
    best_val_loss = None
    for step in range(1, max_steps + 1):
        x, y = train_data.get_batch(args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_acc += loss.item()
        loss_steps += 1

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
