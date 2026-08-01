"""Run native Jittor VisionZip on a PyTorch-exported NPZ and report alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jittor as jt

from visionzip_jittor.config import VisionZipConfig
from visionzip_jittor.core import visionzip_compress


FLOAT_KEYS = (
    "compressed_tokens",
    "contextual_tokens",
    "cls_attention_sum",
    "assignment_counts",
)
INDEX_KEYS = (
    "selected_indices",
    "dominant_ordered_indices",
    "remaining_indices",
    "target_positions",
    "merge_positions",
    "assignments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--cpu", action="store_true", help="Disable Jittor CUDA")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def floating_metrics(actual: np.ndarray, expected: np.ndarray) -> Dict[str, float]:
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    difference = np.abs(actual64 - expected64)
    flat_actual = actual64.reshape(-1)
    flat_expected = expected64.reshape(-1)
    denominator = np.linalg.norm(flat_actual) * np.linalg.norm(flat_expected)
    cosine = float(np.dot(flat_actual, flat_expected) / denominator) if denominator else 1.0
    return {
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    jt.flags.use_cuda = 0 if args.cpu else 1
    data = np.load(args.reference, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    config = VisionZipConfig.from_dict(metadata["config"])

    hidden = jt.array(data["hidden_states"])
    attentions = jt.array(data["attentions"])
    metric = jt.array(data["metric"])
    output = visionzip_compress(hidden, attentions, metric, config)
    output["compressed_tokens"].sync()

    report = {
        "reference": str(args.reference),
        "jittor_version": jt.__version__,
        "use_cuda": int(jt.flags.use_cuda),
        "config": config.to_dict(),
        "float": {},
        "indices": {},
        "passed": True,
    }

    for key in FLOAT_KEYS:
        actual = output[key].numpy()
        expected = data[key]
        metrics = floating_metrics(actual, expected)
        metrics["allclose"] = bool(
            np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
        )
        report["float"][key] = metrics
        report["passed"] = report["passed"] and metrics["allclose"]

    for key in INDEX_KEYS:
        actual = output[key].numpy()
        expected = data[key]
        exact = bool(np.array_equal(actual, expected))
        agreement = float((actual == expected).mean()) if actual.size else 1.0
        report["indices"][key] = {
            "exact": exact,
            "agreement": agreement,
            "shape": list(actual.shape),
        }
        report["passed"] = report["passed"] and exact

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
