"""Export deterministic PyTorch inputs, outputs, and intermediates to NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.pytorch_visionzip import visionzip_compress_torch
from visionzip_jittor.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/visionzip_64.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=577)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--metric-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.validate(args.sequence_length)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    hidden_states = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_dim,
        generator=generator,
        dtype=dtype,
    )
    attention_logits = torch.randn(
        args.batch_size,
        args.heads,
        args.sequence_length,
        args.sequence_length,
        generator=generator,
        dtype=torch.float32,
    )
    attentions = attention_logits.softmax(dim=-1).to(dtype)
    metric = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.metric_dim,
        generator=generator,
        dtype=dtype,
    )

    output = visionzip_compress_torch(hidden_states, attentions, metric, config)
    arrays = {
        "hidden_states": hidden_states.numpy(),
        "attentions": attentions.numpy(),
        "metric": metric.numpy(),
    }
    arrays.update({key: value.detach().cpu().numpy() for key, value in output.as_dict().items()})

    metadata = {
        "seed": args.seed,
        "dtype": args.dtype,
        "config": config.to_dict(),
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "torch_version": torch.__version__,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved reference: {args.output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
