"""Run the two-environment smoke alignment used on AutoDL."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-python",
        default="/root/miniconda3/bin/python",
        help="Python executable containing PyTorch 2.1.2",
    )
    parser.add_argument(
        "--jittor-python",
        default="/root/autodl-tmp/envs/visionzip-jittor/bin/python",
        help="Python executable containing Jittor 1.3.11.0",
    )
    return parser.parse_args()


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = parse_args()
    output = ROOT / "outputs/smoke_reference.npz"
    report = ROOT / "logs/smoke_alignment.json"
    run(
        [
            args.torch_python,
            "scripts/export_pytorch_reference.py",
            "--config",
            "configs/visionzip_64.json",
            "--output",
            str(output),
            "--batch-size",
            "1",
            "--sequence-length",
            "97",
            "--hidden-dim",
            "32",
            "--metric-dim",
            "8",
            "--heads",
            "4",
        ]
    )
    run(
        [
            args.jittor_python,
            "scripts/run_jittor_alignment.py",
            "--reference",
            str(output),
            "--output-json",
            str(report),
        ]
    )


if __name__ == "__main__":
    main()
