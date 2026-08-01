"""Benchmark native Jittor VisionZip core after JIT warm-up."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jittor as jt

from visionzip_jittor.config import load_config
from visionzip_jittor.core import visionzip_compress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/visionzip_64.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=577)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--metric-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def synchronize(output):
    output["compressed_tokens"].sync()


def main():
    args = parse_args()
    jt.flags.use_cuda = 1
    np.random.seed(args.seed)
    config = load_config(args.config)
    config.validate(args.sequence_length)

    hidden = jt.array(
        np.random.randn(
            args.batch_size, args.sequence_length, args.hidden_dim
        ).astype("float32")
    )
    logits = np.random.randn(
        args.batch_size,
        args.heads,
        args.sequence_length,
        args.sequence_length,
    ).astype("float32")
    logits -= logits.max(axis=-1, keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=-1, keepdims=True)
    attentions = jt.array(attention)
    metric = jt.array(
        np.random.randn(
            args.batch_size, args.sequence_length, args.metric_dim
        ).astype("float32")
    )

    for _ in range(args.warmup):
        output = visionzip_compress(hidden, attentions, metric, config)
        synchronize(output)

    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = visionzip_compress(hidden, attentions, metric, config)
        synchronize(output)
        samples_ms.append((time.perf_counter() - start) * 1000)

    report = {
        "config": config.to_dict(),
        "input_shape": {
            "hidden_states": list(hidden.shape),
            "attentions": list(attentions.shape),
            "metric": list(metric.shape),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": {
            "mean": statistics.mean(samples_ms),
            "median": statistics.median(samples_ms),
            "min": min(samples_ms),
            "max": max(samples_ms),
            "stdev": statistics.pstdev(samples_ms),
        },
        "jittor_version": jt.__version__,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
