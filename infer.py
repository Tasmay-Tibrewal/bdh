import argparse
import dataclasses

import torch

import bdh


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a BDH checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--prompt", default="Once upon a time, ")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Compute dtype override.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg, device):
    if dtype_arg != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_arg]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_config(ckpt):
    if "config" not in ckpt:
        return bdh.BDHConfig()
    config = ckpt["config"]
    if isinstance(config, dict):
        return bdh.BDHConfig(**config)
    if dataclasses.is_dataclass(config):
        return config
    return bdh.BDHConfig()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = load_config(ckpt)
    model = bdh.BDH(config).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt_bytes = bytearray(args.prompt, "utf-8")
    prompt_tensor = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(
        0
    )

    with torch.no_grad():
        output = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    decoded = bytes(output.to(torch.uint8).cpu().squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(decoded)


if __name__ == "__main__":
    main()
