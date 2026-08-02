"""Run the complete real-CLIP PyTorch/Jittor alignment pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ROOT / "configs/visionzip_64.json",
    ROOT / "configs/visionzip_128.json",
    ROOT / "configs/visionzip_192.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--jittor-python", required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "outputs/real_clip")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/real_clip")
    parser.add_argument(
        "--model-name-or-path", default="openai/clip-vit-large-patch14-336"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def config_name(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        return str(json.load(handle)["name"])


def sanitize(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "artifact"


def reference_path(artifact_dir: Path, config_path: Path, dtype: str) -> Path:
    return artifact_dir / f"reference_{sanitize(config_name(config_path))}_{dtype}_real_clip.npz"


def run_and_tee(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(item) for item in command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    configs = args.config or list(DEFAULT_CONFIGS)
    if not args.image and args.image_dir is None:
        raise ValueError("provide --image or --image-dir")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    export_command: List[str] = [
        args.torch_python,
        str(ROOT / "scripts/export_real_clip_reference.py"),
        "--output-dir",
        str(args.artifact_dir),
        "--model-name-or-path",
        args.model_name_or_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    for image in args.image:
        export_command.extend(("--image", str(image)))
    if args.image_dir is not None:
        export_command.extend(("--image-dir", str(args.image_dir)))
    for config in configs:
        export_command.extend(("--config", str(config)))
    if args.cache_dir is not None:
        export_command.extend(("--cache-dir", str(args.cache_dir)))
    if args.local_files_only:
        export_command.append("--local-files-only")
    run_and_tee(export_command, args.log_dir / "export_real_clip.log")

    reports = []
    for config in configs:
        reference = reference_path(args.artifact_dir, config, args.dtype)
        label = sanitize(config_name(config))
        report_path = args.log_dir / f"alignment_{label}_{args.dtype}.json"
        alignment_command = [
            args.jittor_python,
            str(ROOT / "scripts/run_jittor_alignment.py"),
            "--reference",
            str(reference),
            "--output-json",
            str(report_path),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
        ]
        run_and_tee(alignment_command, args.log_dir / f"alignment_{label}_{args.dtype}.log")
        reports.append(str(report_path))
        if not args.skip_visualization:
            visualization_command = [
                args.torch_python,
                str(ROOT / "scripts/visualize_real_clip_tokens.py"),
                "--reference",
                str(reference),
                "--output-dir",
                str(args.artifact_dir / "visualizations"),
            ]
            run_and_tee(
                visualization_command,
                args.log_dir / f"visualize_{label}_{args.dtype}.log",
            )

    summary = {
        "passed": True,
        "references": [str(reference_path(args.artifact_dir, config, args.dtype)) for config in configs],
        "alignment_reports": reports,
        "visualization_dir": None if args.skip_visualization else str(args.artifact_dir / "visualizations"),
    }
    summary_path = args.log_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nReal CLIP alignment pipeline completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
