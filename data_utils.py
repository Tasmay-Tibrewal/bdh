import itertools
import json
import os

from datasets import concatenate_datasets, load_dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **_kwargs):
        return iterable


TEXT_KEYS = ("text", "story", "content")


def _extract_text(example):
    for key in TEXT_KEYS:
        if key in example and example[key]:
            return example[key]
    return ""


def iter_dataset_texts(dataset, max_examples=None):
    if max_examples is None:
        for example in dataset:
            text = _extract_text(example)
            if text:
                yield text
        return
    for example in itertools.islice(dataset, max_examples):
        text = _extract_text(example)
        if text:
            yield text


def write_texts_to_bin(text_iter, out_path, sep="\n\n", desc=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sep_bytes = sep.encode("utf-8") if sep else b""
    total_bytes = 0
    with open(out_path, "wb") as f:
        for text in tqdm(text_iter, desc=desc):
            if not text:
                continue
            payload = text.encode("utf-8")
            f.write(payload)
            if sep_bytes:
                f.write(sep_bytes)
            total_bytes += len(payload) + len(sep_bytes)
    return total_bytes


def build_tinystories_bins(
    out_dir,
    dataset_name,
    dataset_config,
    train_split,
    val_split,
    max_train_examples=None,
    max_val_examples=None,
    force=False,
):
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    if (
        not force
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(meta_path)
    ):
        return train_path, val_path, meta_path

    os.makedirs(out_dir, exist_ok=True)
    load_kwargs = {"split": train_split, "streaming": True}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    train_ds = load_dataset(dataset_name, **load_kwargs)

    val_kwargs = {"split": val_split, "streaming": True}
    if dataset_config:
        val_kwargs["name"] = dataset_config
    val_ds = load_dataset(dataset_name, **val_kwargs)

    train_bytes = write_texts_to_bin(
        iter_dataset_texts(train_ds, max_train_examples),
        train_path,
        desc="writing train",
    )
    val_bytes = write_texts_to_bin(
        iter_dataset_texts(val_ds, max_val_examples),
        val_path,
        desc="writing val",
    )

    meta = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "train_split": train_split,
        "val_split": val_split,
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    return train_path, val_path, meta_path


def normalize_alpaca(example):
    return {
        "instruction": (example.get("instruction") or "").strip(),
        "input": (example.get("input") or "").strip(),
        "output": (example.get("output") or "").strip(),
    }


def normalize_dolly(example):
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    if context:
        instruction = f"{instruction}\n\n**Context:**\n{context}"
    return {
        "instruction": instruction,
        "input": "",
        "output": (example.get("response") or "").strip(),
    }


def format_alpaca_prompt(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()
    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output_text}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output_text}"
    )


def build_instruction_bins(
    out_dir,
    alpaca_dataset,
    dolly_dataset,
    val_ratio,
    seed,
    max_examples=None,
    force=False,
):
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    if (
        not force
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(meta_path)
    ):
        return train_path, val_path, meta_path

    os.makedirs(out_dir, exist_ok=True)
    alpaca = load_dataset(alpaca_dataset, split="train")
    alpaca = alpaca.map(normalize_alpaca, remove_columns=alpaca.column_names)
    dolly = load_dataset(dolly_dataset, split="train")
    dolly = dolly.map(normalize_dolly, remove_columns=dolly.column_names)
    combined = concatenate_datasets([alpaca, dolly]).shuffle(seed=seed)

    if max_examples:
        combined = combined.select(range(max_examples))

    split = combined.train_test_split(test_size=val_ratio, seed=seed)
    train_bytes = write_texts_to_bin(
        (format_alpaca_prompt(ex) for ex in split["train"]),
        train_path,
        desc="writing train",
    )
    val_bytes = write_texts_to_bin(
        (format_alpaca_prompt(ex) for ex in split["test"]),
        val_path,
        desc="writing val",
    )

    meta = {
        "alpaca_dataset": alpaca_dataset,
        "dolly_dataset": dolly_dataset,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_examples": max_examples,
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    return train_path, val_path, meta_path
