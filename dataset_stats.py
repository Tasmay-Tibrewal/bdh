import argparse
import importlib
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_utils import (
    format_alpaca_prompt,
    iter_dataset_texts,
    normalize_alpaca,
    normalize_dolly,
)
from train_utils import build_config


def ensure_package(import_name, pip_name=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(import_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute BDH byte counts and HF token counts for datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["pretrain", "ift"],
        default="pretrain",
        help="Dataset type: pretrain (TinyStories) or ift (instruction).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split(s) to compute.",
    )
    parser.add_argument("--bdh-model-size", choices=["25m", "100m"], default="25m")
    parser.add_argument(
        "--hf-model",
        default="gpt2",
        help="Hugging Face model name for tokenizer/config.",
    )
    parser.add_argument(
        "--sep",
        default="\n\n",
        help="Separator appended between texts (must match BDH bin builder).",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=32,
        help="Batch size for tokenizer calls.",
    )

    # Pretrain dataset options (match pretrain_tinystories.py defaults)
    parser.add_argument("--pretrain-dataset", default="noanabeshima/TinyStoriesV2")
    parser.add_argument("--pretrain-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)

    # Instruction dataset options (match finetune_instruct.py defaults)
    parser.add_argument("--alpaca-dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dolly-dataset", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-examples", type=int, default=None)

    return parser.parse_args()


def count_tokens_for_batch(tokenizer, batch):
    enc = tokenizer(
        batch,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return sum(len(ids) for ids in enc["input_ids"])


def count_bytes_and_tokens(text_iter, tokenizer, sep, batch_size):
    sep_bytes = sep.encode("utf-8") if sep else b""
    total_bytes = 0
    total_tokens = 0
    batch = []
    for text in text_iter:
        if not text:
            continue
        payload = text + sep if sep else text
        total_bytes += len(payload.encode("utf-8"))
        if tokenizer is not None:
            batch.append(payload)
            if len(batch) >= batch_size:
                total_tokens += count_tokens_for_batch(tokenizer, batch)
                batch = []
    if batch and tokenizer is not None:
        total_tokens += count_tokens_for_batch(tokenizer, batch)
    return total_bytes, total_tokens


def iter_pretrain_texts(args, split):
    datasets = ensure_package("datasets")
    load_dataset = datasets.load_dataset
    dataset_config = args.pretrain_config
    if dataset_config is not None:
        stripped = dataset_config.strip().lower()
        if stripped in ("", "none", "null", "default"):
            dataset_config = None
    load_kwargs = {"split": split, "streaming": True}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    ds = load_dataset(args.pretrain_dataset, **load_kwargs)
    max_examples = (
        args.max_train_examples if split == args.train_split else args.max_val_examples
    )
    return iter_dataset_texts(ds, max_examples=max_examples)


def build_instruction_splits(args):
    datasets = ensure_package("datasets")
    load_dataset = datasets.load_dataset
    concatenate_datasets = datasets.concatenate_datasets

    alpaca = load_dataset(args.alpaca_dataset, split="train")
    alpaca = alpaca.map(normalize_alpaca, remove_columns=alpaca.column_names)
    datasets_list = [alpaca]
    if args.dolly_dataset:
        dolly = load_dataset(args.dolly_dataset, split="train")
        dolly = dolly.map(normalize_dolly, remove_columns=dolly.column_names)
        datasets_list.append(dolly)

    if len(datasets_list) == 1:
        combined = datasets_list[0].shuffle(seed=args.seed)
    else:
        combined = concatenate_datasets(datasets_list).shuffle(seed=args.seed)

    if args.max_examples:
        combined = combined.select(range(args.max_examples))

    split = combined.train_test_split(test_size=args.val_ratio, seed=args.seed)
    return split["train"], split["test"]


def iter_instruction_texts(dataset):
    for example in dataset:
        text = format_alpaca_prompt(example)
        if text:
            yield text


def pick_attr(config, *names):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return None


def format_int(value):
    return f"{value:,}"


def format_int_or_na(value):
    if value is None:
        return "n/a"
    return format_int(value)


def format_float(value):
    return f"{value:.6f}"


def compute_vocab_token_stats(tokenizer):
    vocab = tokenizer.get_vocab()
    if not vocab:
        return None, None
    total_bytes = 0
    total_chars = 0
    count = 0
    for token in vocab.keys():
        try:
            token_text = tokenizer.convert_tokens_to_string([token])
        except Exception:
            token_text = token
        total_bytes += len(token_text.encode("utf-8"))
        total_chars += len(token_text)
        count += 1
    if count == 0:
        return None, None
    return total_bytes / count, total_chars / count


def main():
    args = parse_args()

    transformers = ensure_package("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    AutoConfig = transformers.AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    hf_config = AutoConfig.from_pretrained(args.hf_model)
    avg_token_bytes, avg_token_chars = compute_vocab_token_stats(tokenizer)

    bdh_config = build_config(model_size=args.bdh_model_size)

    want_train = args.split in ("train", "both")
    want_val = args.split in ("val", "both")

    train_bytes = None
    train_tokens = None
    val_bytes = None
    val_tokens = None

    if args.dataset == "pretrain":
        if want_train:
            train_texts = iter_pretrain_texts(args, args.train_split)
            train_bytes, train_tokens = count_bytes_and_tokens(
                train_texts, tokenizer, args.sep, args.tokenizer_batch_size
            )
        if want_val:
            val_texts = iter_pretrain_texts(args, args.val_split)
            val_bytes, val_tokens = count_bytes_and_tokens(
                val_texts, tokenizer, args.sep, args.tokenizer_batch_size
            )
    else:
        train_ds, val_ds = build_instruction_splits(args)
        if want_train:
            train_texts = iter_instruction_texts(train_ds)
            train_bytes, train_tokens = count_bytes_and_tokens(
                train_texts, tokenizer, args.sep, args.tokenizer_batch_size
            )
        if want_val:
            val_texts = iter_instruction_texts(val_ds)
            val_bytes, val_tokens = count_bytes_and_tokens(
                val_texts, tokenizer, args.sep, args.tokenizer_batch_size
            )

    if not want_train and not want_val:
        raise ValueError("--split must include train and/or val.")

    total_bytes = sum(v for v in (train_bytes, val_bytes) if v is not None)
    total_tokens = sum(v for v in (train_tokens, val_tokens) if v is not None)
    avg_bytes_per_token = (
        total_bytes / total_tokens if total_tokens else float("nan")
    )

    bdh_head_dim = (
        bdh_config.n_embd // bdh_config.n_head if bdh_config.n_head else None
    )
    bdh_mlp_dim = bdh_config.n_embd * bdh_config.mlp_internal_dim_multiplier
    bdh_mlp_dim_per_head = (
        bdh_mlp_dim // bdh_config.n_head if bdh_config.n_head else None
    )

    hf_hidden = pick_attr(hf_config, "hidden_size", "n_embd", "d_model")
    hf_layers = pick_attr(hf_config, "num_hidden_layers", "n_layer", "num_layers")
    hf_heads = pick_attr(hf_config, "num_attention_heads", "n_head", "num_heads")
    hf_kv_heads = pick_attr(hf_config, "num_key_value_heads", "n_kv_head")
    hf_intermediate = pick_attr(
        hf_config, "intermediate_size", "n_inner", "ffn_dim", "mlp_hidden_size"
    )
    hf_vocab = pick_attr(hf_config, "vocab_size")
    hf_head_dim = None
    if hf_hidden is not None and hf_heads:
        hf_head_dim = hf_hidden // hf_heads

    print("Dataset stats")
    print(f"  dataset_type: {args.dataset}")
    print(f"  bdh_bytes_train: {format_int_or_na(train_bytes)}")
    print(f"  bdh_bytes_val: {format_int_or_na(val_bytes)}")
    print(f"  bdh_bytes_total: {format_int(total_bytes)}")
    print(f"  hf_tokens_train: {format_int_or_na(train_tokens)}")
    print(f"  hf_tokens_val: {format_int_or_na(val_tokens)}")
    print(f"  hf_tokens_total: {format_int(total_tokens)}")
    print(f"  avg_bdh_bytes_per_hf_token: {format_float(avg_bytes_per_token)}")
    print(f"  sep: {repr(args.sep)}")

    print("\nBDH config")
    print(f"  model_size: {args.bdh_model_size}")
    print(f"  n_layer: {bdh_config.n_layer}")
    print(f"  n_head: {bdh_config.n_head}")
    print(f"  n_embd: {bdh_config.n_embd}")
    print(f"  head_dim: {bdh_head_dim}")
    print(f"  mlp_internal_dim_multiplier: {bdh_config.mlp_internal_dim_multiplier}")
    print(f"  mlp_internal_dim: {bdh_mlp_dim}")
    print(f"  mlp_internal_dim_per_head: {bdh_mlp_dim_per_head}")
    print(f"  vocab_size: {bdh_config.vocab_size}")
    print(f"  attn_window: {bdh_config.attn_window}")

    print("\nHF transformer config")
    print(f"  model_name: {args.hf_model}")
    print(f"  model_type: {getattr(hf_config, 'model_type', None)}")
    print(f"  num_hidden_layers: {hf_layers}")
    print(f"  num_attention_heads: {hf_heads}")
    if hf_kv_heads is not None:
        print(f"  num_key_value_heads: {hf_kv_heads}")
    print(f"  hidden_size: {hf_hidden}")
    print(f"  head_dim: {hf_head_dim}")
    if hf_intermediate is not None:
        print(f"  intermediate_size: {hf_intermediate}")
    if hf_vocab is not None:
        print(f"  vocab_size: {hf_vocab}")
    if avg_token_bytes is not None:
        print(f"  avg_vocab_token_bytes: {format_float(avg_token_bytes)}")
    if avg_token_chars is not None:
        print(f"  avg_vocab_token_chars: {format_float(avg_token_chars)}")


if __name__ == "__main__":
    main()
