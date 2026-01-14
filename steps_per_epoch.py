import argparse
import os

from train_utils import steps_per_epoch


DATASET_DEFAULT_DIRS = {
    "pretrain": "data/tinystories_v2_gpt4",
    "instruction": "data/instruction/alpaca",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute steps per epoch for a byte-level dataset."
    )
    parser.add_argument(
        "--data-bin",
        default=None,
        help="Path to train.bin (byte-level dataset).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.bin.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DEFAULT_DIRS.keys()),
        default=None,
        help="Use the default data dir for a known dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    if not (args.data_bin or args.data_dir or args.dataset):
        parser.error("Provide --data-bin, --data-dir, or --dataset.")
    return args


def resolve_data_bin(args):
    if args.data_bin:
        return args.data_bin
    if args.data_dir:
        return os.path.join(args.data_dir, "train.bin")
    data_dir = DATASET_DEFAULT_DIRS[args.dataset]
    return os.path.join(data_dir, "train.bin")


def main():
    args = parse_args()
    data_bin = resolve_data_bin(args)
    steps = steps_per_epoch(data_bin, args.block_size, args.batch_size)
    print(steps)


if __name__ == "__main__":
    main()
