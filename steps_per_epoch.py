import argparse

from train_utils import steps_per_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute steps per epoch for a byte-level dataset."
    )
    parser.add_argument(
        "--data-bin",
        required=True,
        help="Path to train.bin (byte-level dataset).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    steps = steps_per_epoch(args.data_bin, args.block_size, args.batch_size)
    print(steps)


if __name__ == "__main__":
    main()
