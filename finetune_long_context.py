import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb

import bdh
from train_utils import (
    build_config,
    count_parameters,
    maybe_compile,
    setup_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune BDH with long context and masked loss."
    )
    parser.add_argument(
        "--train-jsonl",
        default=None,
        help="Path to train JSONL (context + prompt + response).",
    )
    parser.add_argument(
        "--context-jsonl",
        default=None,
        help="Path to contexts JSONL (context_id + context).",
    )
    parser.add_argument(
        "--pairs-jsonl",
        default=None,
        help="Path to prompt/response JSONL (context_id + prompt + response).",
    )
    parser.add_argument("--val-jsonl", default=None, help="Path to val JSONL.")
    parser.add_argument("--val-context-jsonl", default=None)
    parser.add_argument("--val-pairs-jsonl", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--context-key", default="context")
    parser.add_argument("--context-id-key", default="context_id")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--context-sep", default="")
    parser.add_argument("--prompt-sep", default="")
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--context-block-size", type=int, default=8192)
    parser.add_argument("--latent-tokens-per-block", type=int, default=64)
    parser.add_argument("--latent-token-id", type=int, default=256)
    parser.add_argument(
        "--backprop-context-tokens",
        type=int,
        default=8192,
        help="Number of context tokens to backpropagate through.",
    )
    parser.add_argument(
        "--pairs-per-step",
        type=int,
        default=1,
        help="Number of prompt/response pairs per optimizer step.",
    )
    parser.add_argument(
        "--context-loss-weight",
        type=float,
        default=1.0,
        help="Weight for context loss when using cached contexts.",
    )

    parser.add_argument("--model-size", choices=["25m", "100m"], default="25m")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--mlp-mult", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attn-window", type=int, default=8192)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--max-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--out-dir", default="checkpoints/finetune_long_context")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--resume-from", default=None, help="Path to checkpoint .pt to resume training."
    )

    parser.add_argument("--wandb-project", default="bdh-long-context")
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


def _encode(text):
    if not text:
        return bytearray()
    return bytearray(text, "utf-8")


def _truncate(tokens, max_tokens, keep_tail=False):
    if max_tokens is None or len(tokens) <= max_tokens:
        return tokens
    if keep_tail:
        return tokens[-max_tokens:]
    return tokens[:max_tokens]


TYPE_CONTEXT = 0
TYPE_LATENT = 1
TYPE_PROMPT = 2


class JsonlContextDataset:
    def __init__(
        self,
        path,
        context_key,
        prompt_key,
        response_key,
        context_sep,
        prompt_sep,
        max_context_tokens=None,
        max_prompt_tokens=None,
        max_response_tokens=None,
        context_block_size=8192,
        latent_tokens_per_block=64,
        latent_token_id=256,
        backprop_context_tokens=8192,
        max_examples=None,
    ):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
                if max_examples and len(self.examples) >= max_examples:
                    break
        if not self.examples:
            raise ValueError(f"No examples found in {path}")
        self.context_key = context_key
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.context_sep = context_sep
        self.prompt_sep = prompt_sep
        self.max_context_tokens = max_context_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.context_block_size = context_block_size
        self.latent_tokens_per_block = latent_tokens_per_block
        self.latent_token_id = latent_token_id
        self.backprop_context_tokens = backprop_context_tokens

    def __len__(self):
        return len(self.examples)

    def _get_text(self, example, key, fallbacks=()):
        if key in example and example[key] is not None:
            return str(example[key])
        for fallback in fallbacks:
            if fallback in example and example[fallback] is not None:
                return str(example[fallback])
        return ""

    def _build_tokens(self, example):
        context = self._get_text(example, self.context_key)
        prompt = self._get_text(
            example, self.prompt_key, fallbacks=("instruction", "input", "question")
        )
        response = self._get_text(
            example, self.response_key, fallbacks=("output", "completion")
        )

        if context and self.context_sep:
            context = f"{context}{self.context_sep}"
        if prompt and self.prompt_sep:
            prompt = f"{prompt}{self.prompt_sep}"

        context_tokens = _truncate(
            _encode(context), self.max_context_tokens, keep_tail=True
        )
        prompt_tokens = _truncate(_encode(prompt), self.max_prompt_tokens)
        response_tokens = _truncate(_encode(response), self.max_response_tokens)

        tokens = []
        token_types = []
        segment_ids = []

        context_len = len(context_tokens)
        segment_id = 0
        for start in range(0, context_len, self.context_block_size):
            block = context_tokens[start : start + self.context_block_size]
            for token in block:
                tokens.append(token)
                token_types.append(TYPE_CONTEXT)
                segment_ids.append(segment_id)
            for _ in range(self.latent_tokens_per_block):
                tokens.append(self.latent_token_id)
                token_types.append(TYPE_LATENT)
                segment_ids.append(segment_id)
            segment_id += 1

        if segment_id == 0:
            segment_id = 1
        last_segment_id = segment_id - 1
        for token in prompt_tokens + response_tokens:
            tokens.append(token)
            token_types.append(TYPE_PROMPT)
            segment_ids.append(last_segment_id)

        if len(tokens) < 2:
            raise ValueError("Example is too short after encoding.")
        return tokens, token_types, segment_ids

    def get_example(self, idx, device):
        tokens, token_types, segment_ids = self._build_tokens(self.examples[idx])
        token_types = np.asarray(token_types, dtype=np.int64)
        segment_ids = np.asarray(segment_ids, dtype=np.int64)

        loss_mask = np.zeros(len(tokens), dtype=np.float32)
        grad_mask = np.zeros(len(tokens), dtype=np.float32)

        context_positions = np.where(token_types == TYPE_CONTEXT)[0]
        if context_positions.size:
            keep = context_positions[-self.backprop_context_tokens :]
            loss_mask[keep] = 1.0
            grad_mask[keep] = 1.0

        prompt_positions = np.where(token_types == TYPE_PROMPT)[0]
        if prompt_positions.size:
            loss_mask[prompt_positions] = 1.0
            grad_mask[prompt_positions] = 1.0

        latent_positions = np.where(token_types == TYPE_LATENT)[0]
        if latent_positions.size:
            grad_mask[latent_positions] = 1.0

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
        loss_mask = (
            torch.tensor(loss_mask[1:], dtype=torch.float32, device=device).unsqueeze(0)
        )
        grad_mask = (
            torch.tensor(grad_mask[:-1], dtype=torch.float32, device=device).unsqueeze(0)
        )
        token_types = torch.tensor(
            token_types[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)
        segment_ids = torch.tensor(
            segment_ids[:-1], dtype=torch.long, device=device
        ).unsqueeze(0)
        return x, y, loss_mask, grad_mask, token_types, segment_ids


class ContextStore:
    def __init__(
        self,
        contexts,
        context_id_key,
        context_key,
        context_sep,
        context_block_size,
        latent_tokens_per_block,
        latent_token_id,
        max_context_tokens=None,
        backprop_context_tokens=8192,
    ):
        self.contexts = {}
        for entry in contexts:
            if context_id_key not in entry:
                raise ValueError(f"Missing {context_id_key} in context entry.")
            context_id = str(entry[context_id_key])
            context_text = entry.get(context_key) or ""
            if context_text and context_sep:
                context_text = f"{context_text}{context_sep}"
            context_tokens = _truncate(
                _encode(context_text), max_context_tokens, keep_tail=True
            )
            (
                tokens,
                token_types,
                segment_ids,
                context_positions,
                latent_positions,
                last_segment_id,
                last_segment_context_positions,
            ) = build_context_tokens(
                context_tokens,
                context_block_size,
                latent_tokens_per_block,
                latent_token_id,
            )
            if context_positions.size:
                keep = context_positions[-backprop_context_tokens:]
            else:
                keep = np.array([], dtype=np.int64)
            self.contexts[context_id] = {
                "tokens": tokens,
                "token_types": token_types,
                "segment_ids": segment_ids,
                "context_len": len(tokens),
                "context_keep": keep,
                "latent_positions": latent_positions,
                "last_segment_id": last_segment_id,
                "cache_indices": np.concatenate(
                    [last_segment_context_positions, latent_positions]
                )
                if latent_positions.size or last_segment_context_positions.size
                else np.array([], dtype=np.int64),
            }
        if not self.contexts:
            raise ValueError("No contexts loaded.")

    def get(self, context_id):
        return self.contexts[context_id]


def build_context_tokens(
    context_tokens,
    context_block_size,
    latent_tokens_per_block,
    latent_token_id,
):
    tokens = []
    token_types = []
    segment_ids = []
    context_positions = []
    latent_positions = []
    segment_id = 0
    for start in range(0, len(context_tokens), context_block_size):
        block = context_tokens[start : start + context_block_size]
        for token in block:
            context_positions.append(len(tokens))
            tokens.append(token)
            token_types.append(TYPE_CONTEXT)
            segment_ids.append(segment_id)
        for _ in range(latent_tokens_per_block):
            latent_positions.append(len(tokens))
            tokens.append(latent_token_id)
            token_types.append(TYPE_LATENT)
            segment_ids.append(segment_id)
        segment_id += 1
    if segment_id == 0:
        segment_id = 1
    return (
        np.asarray(tokens, dtype=np.int64),
        np.asarray(token_types, dtype=np.int64),
        np.asarray(segment_ids, dtype=np.int64),
        np.asarray(context_positions, dtype=np.int64),
        np.asarray(latent_positions, dtype=np.int64),
        segment_id - 1,
        np.asarray(
            [pos for pos in context_positions if segment_ids[pos] == segment_id - 1],
            dtype=np.int64,
        ),
    )


class ContextPairsDataset:
    def __init__(
        self,
        context_store,
        pairs,
        context_id_key,
        prompt_key,
        response_key,
        prompt_sep,
        max_prompt_tokens=None,
        max_response_tokens=None,
        max_examples=None,
    ):
        self.context_store = context_store
        self.context_id_key = context_id_key
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_sep = prompt_sep
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.examples = []
        for entry in pairs:
            self.examples.append(entry)
            if max_examples and len(self.examples) >= max_examples:
                break
        if not self.examples:
            raise ValueError("No prompt/response pairs loaded.")

    def __len__(self):
        return len(self.examples)

    def get_context_id(self, idx):
        example = self.examples[idx]
        if self.context_id_key not in example:
            raise ValueError(f"Missing {self.context_id_key} in pair example.")
        return str(example[self.context_id_key])

    def _get_text(self, example, key, fallbacks=()):
        if key in example and example[key] is not None:
            return str(example[key])
        for fallback in fallbacks:
            if fallback in example and example[fallback] is not None:
                return str(example[fallback])
        return ""

    def get_prompt_response_tokens(self, idx):
        example = self.examples[idx]
        prompt = self._get_text(
            example, self.prompt_key, fallbacks=("instruction", "input", "question")
        )
        response = self._get_text(
            example, self.response_key, fallbacks=("output", "completion")
        )
        if prompt and self.prompt_sep:
            prompt = f"{prompt}{self.prompt_sep}"

        prompt_tokens = _truncate(_encode(prompt), self.max_prompt_tokens)
        response_tokens = _truncate(_encode(response), self.max_response_tokens)
        return np.asarray(list(prompt_tokens + response_tokens), dtype=np.int64)


def masked_loss(logits, targets, mask):
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab), targets.view(-1), reduction="none"
    )
    mask_flat = mask.view(-1)
    denom = mask_flat.sum().clamp_min(1.0)
    return (loss * mask_flat).sum() / denom


def prepare_context_tensors(context, device):
    ctx_tokens = context["tokens"]
    ctx_token_types = context["token_types"]
    ctx_segment_ids = context["segment_ids"]
    ctx_len = context["context_len"]

    ctx_loss_mask = np.zeros(ctx_len, dtype=np.float32)
    ctx_grad_mask = np.zeros(ctx_len, dtype=np.float32)
    if context["context_keep"].size:
        ctx_loss_mask[context["context_keep"]] = 1.0
        ctx_grad_mask[context["context_keep"]] = 1.0
    if context["latent_positions"].size:
        ctx_grad_mask[context["latent_positions"]] = 1.0

    x_ctx = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
    y_ctx = torch.tensor(ctx_tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
    loss_mask_ctx = torch.tensor(
        ctx_loss_mask[1:], dtype=torch.float32, device=device
    ).unsqueeze(0)
    grad_mask_ctx = torch.tensor(
        ctx_grad_mask, dtype=torch.float32, device=device
    ).unsqueeze(0)
    token_types_ctx = torch.tensor(
        ctx_token_types, dtype=torch.long, device=device
    ).unsqueeze(0)
    segment_ids_ctx = torch.tensor(
        ctx_segment_ids, dtype=torch.long, device=device
    ).unsqueeze(0)

    cache_indices = context["cache_indices"]
    cache_indices = cache_indices[cache_indices < ctx_len]
    cache_indices = torch.tensor(cache_indices, dtype=torch.long, device=device)

    return (
        x_ctx,
        y_ctx,
        loss_mask_ctx,
        grad_mask_ctx,
        token_types_ctx,
        segment_ids_ctx,
        cache_indices,
        ctx_len,
    )


@torch.no_grad()
def estimate_masked_loss(model, dataset, eval_iters, ctx, device):
    model.eval()
    losses = []
    total_iters = min(eval_iters, len(dataset))
    for i in range(total_iters):
        x, y, loss_mask, grad_mask, token_types, segment_ids = dataset.get_example(
            i, device
        )
        with ctx:
            logits, _ = model(
                x,
                token_types=token_types,
                segment_ids=segment_ids,
                grad_mask=grad_mask,
            )
            loss = masked_loss(logits, y, loss_mask)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_cached_loss(
    model,
    context_store,
    pairs_dataset,
    eval_iters,
    ctx,
    device,
    context_loss_weight,
):
    model.eval()
    losses = []
    total_iters = min(eval_iters, len(pairs_dataset))
    for i in range(total_iters):
        context_id = pairs_dataset.get_context_id(i)
        context = context_store.get(context_id)
        (
            x_ctx,
            y_ctx,
            loss_mask_ctx,
            grad_mask_ctx,
            token_types_ctx,
            segment_ids_ctx,
            cache_indices,
            ctx_len,
        ) = prepare_context_tensors(context, device)

        with ctx:
            logits_ctx, _, caches = model(
                x_ctx,
                token_types=token_types_ctx,
                segment_ids=segment_ids_ctx,
                grad_mask=grad_mask_ctx,
                cache_indices=cache_indices,
                return_cache=True,
            )
            context_loss = masked_loss(
                logits_ctx[:, :-1, :], y_ctx, loss_mask_ctx
            )

        prompt_tokens = pairs_dataset.get_prompt_response_tokens(i)
        if prompt_tokens.size < 2:
            continue
        x_prompt = torch.tensor(prompt_tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_prompt = torch.tensor(prompt_tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
        loss_mask_prompt = torch.ones_like(y_prompt, dtype=torch.float32, device=device)
        grad_mask_prompt = torch.ones_like(x_prompt, dtype=torch.float32, device=device)

        with ctx:
            logits_prompt = model.forward_with_cache(
                x_prompt,
                caches,
                grad_mask=grad_mask_prompt,
                position_offset=ctx_len,
            )
            prompt_loss = masked_loss(logits_prompt, y_prompt, loss_mask_prompt)

        loss = prompt_loss + context_loss_weight * context_loss
        losses.append(loss.item())

    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def expand_vocab_state(state_dict, new_vocab_size):
    state = dict(state_dict)
    embed_key = "embed.weight"
    head_key = "lm_head"
    if embed_key in state:
        old = state[embed_key]
        if old.shape[0] < new_vocab_size:
            new = old.new_empty((new_vocab_size, old.shape[1]))
            new[: old.shape[0]] = old
            torch.nn.init.normal_(new[old.shape[0] :], std=0.02)
            state[embed_key] = new
    if head_key in state:
        old = state[head_key]
        if old.shape[1] < new_vocab_size:
            new = old.new_empty((old.shape[0], new_vocab_size))
            new[:, : old.shape[1]] = old
            torch.nn.init.normal_(new[:, old.shape[1] :], std=0.02)
            state[head_key] = new
    return state


def main():
    args = parse_args()
    if args.grad_acc_steps < 1:
        raise ValueError("--grad-acc-steps must be >= 1")
    if args.batch_size != 1:
        raise ValueError("Only --batch-size 1 is supported for long contexts.")
    if not (args.train_jsonl or args.pairs_jsonl):
        raise ValueError("Provide --train-jsonl or --pairs-jsonl.")
    if args.pairs_jsonl and not args.context_jsonl:
        raise ValueError("Provide --context-jsonl when using --pairs-jsonl.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device, dtype, ctx, scaler = setup_device(args.dtype)
    config = build_config(
        model_size=args.model_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        attn_window=args.attn_window,
    )
    if args.latent_token_id is not None:
        config.vocab_size = max(config.vocab_size, args.latent_token_id + 1)
        config.latent_token_id = args.latent_token_id

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
        if args.latent_token_id is not None:
            config.vocab_size = max(config.vocab_size, args.latent_token_id + 1)
            config.latent_token_id = args.latent_token_id
        if args.attn_window is not None:
            config.attn_window = args.attn_window
        start_step = int(ckpt.get("step", 0))
        best_val_loss = ckpt.get("best_val_loss", None)
        print(f"resuming from {args.resume_from} at step {start_step}")

    model = bdh.BDH(config).to(device)
    if ckpt and "model_state" in ckpt:
        state = ckpt["model_state"]
        state = expand_vocab_state(state, config.vocab_size)
        model.load_state_dict(state)
    model = maybe_compile(model, args.compile)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if ckpt and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    param_count = count_parameters(model)
    print(f"Using device: {device} dtype={dtype} params={param_count:,}")

    if args.pairs_jsonl:
        if args.grad_acc_steps != 1:
            raise ValueError("grad_acc_steps is not supported with --pairs-jsonl.")
        with open(args.context_jsonl, "r", encoding="utf-8") as f:
            contexts = [json.loads(line) for line in f if line.strip()]
        context_store = ContextStore(
            contexts,
            context_id_key=args.context_id_key,
            context_key=args.context_key,
            context_sep=args.context_sep,
            context_block_size=args.context_block_size,
            latent_tokens_per_block=args.latent_tokens_per_block,
            latent_token_id=args.latent_token_id,
            max_context_tokens=args.max_context_tokens,
            backprop_context_tokens=args.backprop_context_tokens,
        )
        with open(args.pairs_jsonl, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        train_data = ContextPairsDataset(
            context_store,
            pairs,
            context_id_key=args.context_id_key,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            prompt_sep=args.prompt_sep,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            max_examples=args.max_examples,
        )
    else:
        train_data = JsonlContextDataset(
            args.train_jsonl,
            context_key=args.context_key,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            context_sep=args.context_sep,
            prompt_sep=args.prompt_sep,
            max_context_tokens=args.max_context_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            context_block_size=args.context_block_size,
            latent_tokens_per_block=args.latent_tokens_per_block,
            latent_token_id=args.latent_token_id,
            backprop_context_tokens=args.backprop_context_tokens,
            max_examples=args.max_examples,
        )
    val_data = None
    val_context_store = None
    if args.val_context_jsonl or args.val_pairs_jsonl:
        val_context_store = context_store if args.pairs_jsonl else None
        if args.val_context_jsonl:
            with open(args.val_context_jsonl, "r", encoding="utf-8") as f:
                val_contexts = [json.loads(line) for line in f if line.strip()]
            val_context_store = ContextStore(
                val_contexts,
                context_id_key=args.context_id_key,
                context_key=args.context_key,
                context_sep=args.context_sep,
                context_block_size=args.context_block_size,
                latent_tokens_per_block=args.latent_tokens_per_block,
                latent_token_id=args.latent_token_id,
                max_context_tokens=args.max_context_tokens,
                backprop_context_tokens=args.backprop_context_tokens,
            )
        if args.val_pairs_jsonl:
            if val_context_store is None:
                raise ValueError("Provide --val-context-jsonl with --val-pairs-jsonl.")
            with open(args.val_pairs_jsonl, "r", encoding="utf-8") as f:
                val_pairs = [json.loads(line) for line in f if line.strip()]
            val_data = ContextPairsDataset(
                val_context_store,
                val_pairs,
                context_id_key=args.context_id_key,
                prompt_key=args.prompt_key,
                response_key=args.response_key,
                prompt_sep=args.prompt_sep,
                max_prompt_tokens=args.max_prompt_tokens,
                max_response_tokens=args.max_response_tokens,
                max_examples=args.max_val_examples,
            )
    elif args.val_jsonl:
        val_data = JsonlContextDataset(
            args.val_jsonl,
            context_key=args.context_key,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            context_sep=args.context_sep,
            prompt_sep=args.prompt_sep,
            max_context_tokens=args.max_context_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            context_block_size=args.context_block_size,
            latent_tokens_per_block=args.latent_tokens_per_block,
            latent_token_id=args.latent_token_id,
            backprop_context_tokens=args.backprop_context_tokens,
            max_examples=args.max_val_examples,
        )

    steps_per_epoch_value = max(1, len(train_data) // args.batch_size)
    if args.max_epochs is not None:
        target_steps = int(math.ceil(args.max_epochs * steps_per_epoch_value))
    else:
        target_steps = args.max_steps
    max_steps = start_step + target_steps if start_step else target_steps

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
                "param_count": param_count,
                "device": str(device),
            },
        )

    loss_acc = 0.0
    loss_steps = 0
    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        if args.pairs_jsonl:
            pair_indices = rng.integers(len(train_data), size=args.pairs_per_step)
            groups = {}
            for idx in pair_indices:
                context_id = train_data.get_context_id(int(idx))
                groups.setdefault(context_id, []).append(int(idx))

            total_prompt_loss = 0.0
            total_prompt_count = 0
            total_context_loss = 0.0

            for context_id, idxs in groups.items():
                context = context_store.get(context_id)
                (
                    x_ctx,
                    y_ctx,
                    loss_mask_ctx,
                    grad_mask_ctx,
                    token_types_ctx,
                    segment_ids_ctx,
                    cache_indices,
                    ctx_len,
                ) = prepare_context_tensors(context, device)

                with ctx:
                    logits_ctx, _, caches = model(
                        x_ctx,
                        token_types=token_types_ctx,
                        segment_ids=segment_ids_ctx,
                        grad_mask=grad_mask_ctx,
                        cache_indices=cache_indices,
                        return_cache=True,
                    )
                    context_loss = masked_loss(logits_ctx[:, :-1, :], y_ctx, loss_mask_ctx)
                total_context_loss += context_loss

                for idx in idxs:
                    prompt_tokens = train_data.get_prompt_response_tokens(idx)
                    if prompt_tokens.size < 2:
                        continue
                    x_prompt = torch.tensor(
                        prompt_tokens[:-1], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    y_prompt = torch.tensor(
                        prompt_tokens[1:], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    loss_mask_prompt = torch.ones_like(
                        y_prompt, dtype=torch.float32, device=device
                    )
                    grad_mask_prompt = torch.ones_like(
                        x_prompt, dtype=torch.float32, device=device
                    )
                    with ctx:
                        logits_prompt = model.forward_with_cache(
                            x_prompt,
                            caches,
                            grad_mask=grad_mask_prompt,
                            position_offset=ctx_len,
                        )
                        prompt_loss = masked_loss(logits_prompt, y_prompt, loss_mask_prompt)
                    total_prompt_loss += prompt_loss
                    total_prompt_count += 1

            prompt_loss = total_prompt_loss / max(1, total_prompt_count)
            context_loss = total_context_loss / max(1, len(groups))
            loss = prompt_loss + args.context_loss_weight * context_loss
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            loss_steps += 1
        else:
            for _ in range(args.grad_acc_steps):
                idx = int(rng.integers(len(train_data)))
                x, y, loss_mask, grad_mask, token_types, segment_ids = train_data.get_example(
                    idx, device
                )
                with ctx:
                    logits, _ = model(
                        x,
                        token_types=token_types,
                        segment_ids=segment_ids,
                        grad_mask=grad_mask,
                    )
                    loss = masked_loss(logits, y, loss_mask)
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

        if val_data and step % args.eval_interval == 0:
            if args.pairs_jsonl:
                val_loss = estimate_cached_loss(
                    model,
                    val_context_store or context_store,
                    val_data,
                    eval_iters=args.eval_iters,
                    ctx=ctx,
                    device=device,
                    context_loss_weight=args.context_loss_weight,
                )
            else:
                val_loss = estimate_masked_loss(
                    model,
                    val_data,
                    eval_iters=args.eval_iters,
                    ctx=ctx,
                    device=device,
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
