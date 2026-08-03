#!/usr/bin/env python3
"""Build compact, reviewable submission assets from preserved evidence archives.

The raw experiment logs and model artifacts are intentionally not committed. This
script reads the independently archived Phase 2/4A/4B/5A evidence packages and
writes small CSV/JSON summaries plus presentation-ready PNG figures under docs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-archive", type=Path, required=True)
    parser.add_argument("--phase4a-archive", type=Path, required=True)
    parser.add_argument("--phase4b-archive", type=Path, required=True)
    parser.add_argument("--phase5a-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceArchive:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._tar = tarfile.open(self.path, "r:gz")
        self._files = [member.name for member in self._tar.getmembers() if member.isfile()]

    def close(self) -> None:
        self._tar.close()

    def find_suffix(self, suffix: str) -> str:
        matches = [name for name in self._files if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one archive member ending with {suffix!r} "
                f"in {self.path}, found {matches}"
            )
        return matches[0]

    def read_bytes(self, suffix: str) -> bytes:
        member = self.find_suffix(suffix)
        stream = self._tar.extractfile(member)
        if stream is None:
            raise FileNotFoundError(member)
        return stream.read()

    def read_text(self, suffix: str) -> str:
        return self.read_bytes(suffix).decode("utf-8")

    def read_json(self, suffix: str) -> dict[str, Any]:
        return json.loads(self.read_text(suffix))

    def read_jsonl(self, suffix: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.read_text(suffix).splitlines() if line.strip()]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float] = []
    cumulative = np.cumsum(np.asarray([0.0, *values], dtype=np.float64))
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result.append(float((cumulative[index + 1] - cumulative[start]) / (index + 1 - start)))
    return result


def configure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install the project with the submission extra"
        ) from exc
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def extract_phase2(archive: EvidenceArchive) -> list[dict[str, Any]]:
    rows = []
    for budget in (64, 128, 192):
        report = archive.read_json(
            f"alignment_clip_{budget}_code_exact_float32.json"
        )
        rows.append(
            {
                "budget": budget,
                "output_tokens_including_cls": budget + 1,
                "selected_indices_exact": report["indices"]["selected_indices"]["exact"],
                "assignments_exact": report["indices"]["assignments"]["exact"],
                "compressed_max_abs_error": report["float"]["compressed_tokens"]["max_abs_error"],
                "compressed_mean_abs_error": report["float"]["compressed_tokens"]["mean_abs_error"],
                "contextual_max_abs_error": report["float"]["contextual_tokens"]["max_abs_error"],
                "cls_attention_max_abs_error": report["float"]["cls_attention_sum"]["max_abs_error"],
                "passed": report["passed"],
            }
        )
    return rows


def extract_phase4a(archive: EvidenceArchive) -> dict[str, Any]:
    summary = archive.read_json("logs/phase4a/phase4a_summary.json")
    return {
        "passed": summary["passed"],
        "initial_train_loss": summary["initial_train_loss"],
        "final_train_loss": summary["final_train_loss"],
        "initial_validation_loss": summary["initial_validation_loss"],
        "final_validation_loss": summary["final_validation_loss"],
        "language_unchanged": summary["language_unchanged"],
        "projector_optimizer_scope_exact": summary["projector_optimizer_scope_exact"],
    }


def extract_phase4b(
    archive: EvidenceArchive,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    metrics = archive.read_jsonl("training/phase4b_train_metrics.jsonl")
    summary = archive.read_json("training/phase4b_training_summary.json")
    train_rows = [row for row in metrics if row["artifact_type"] == "phase4b_train_metric_v1"]
    validation_rows = [
        row for row in metrics if row["artifact_type"] == "phase4b_held_out_nll_v1"
    ]
    losses = [float(row["target_token_weighted_nll"]) for row in train_rows]
    smoothed = rolling_mean(losses, window=32)
    compact_train = [
        {
            "optimizer_step": int(row["optimizer_step"]),
            "train_target_nll": float(row["target_token_weighted_nll"]),
            "train_target_nll_rolling32": smoothed[index],
            "learning_rate": float(row["learning_rate"]),
            "optimizer_step_ms": float(row["elapsed_ms"]),
            "finite_update": bool(row["finite_update"]),
        }
        for index, row in enumerate(train_rows)
    ]
    compact_validation = [
        {
            "optimizer_step": int(row["optimizer_step"]),
            "held_out_target_nll": float(row["held_out_target_nll"]),
            "held_out_target_perplexity": float(row["held_out_target_perplexity"]),
            "sample_count": int(row["sample_count"]),
            "target_token_count": int(row["target_token_count"]),
        }
        for row in validation_rows
    ]
    benchmark = summary["training_benchmark"]
    key_summary = {
        "passed": summary["passed"],
        "completed_training": summary["completed_training"],
        "start_optimizer_step": summary["start_optimizer_step"],
        "final_optimizer_step": summary["final_optimizer_step"],
        "initial_held_out_target_nll": summary["initial_validation"]["held_out_target_nll"],
        "final_held_out_target_nll": summary["final_validation"]["held_out_target_nll"],
        "initial_held_out_perplexity": summary["initial_validation"]["held_out_target_perplexity"],
        "final_held_out_perplexity": summary["final_validation"]["held_out_target_perplexity"],
        "language_unchanged": summary["language_unchanged"],
        "projector_optimizer_scope_exact": summary["projector_optimizer_scope_exact"],
        "projector_trainability_restored_after_evaluation": summary[
            "projector_trainability_restored_after_evaluation"
        ],
        "all_updates_finite": summary["all_updates_finite"],
        "effective_samples_per_second": benchmark["effective_samples_per_second"],
        "target_tokens_per_second": benchmark["target_tokens_per_second"],
        "mean_optimizer_step_ms": benchmark["mean_optimizer_step_ms"],
        "peak_process_gpu_memory_mib": benchmark["peak_process_gpu_memory_mib"],
    }
    return compact_train, compact_validation, key_summary, summary


def extract_phase5a(archive: EvidenceArchive) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = archive.read_json("kv_cache_benchmark_7227717.json")
    rows = []
    for budget_result in report["results"]:
        samples = budget_result["sample_results"]
        benchmark = budget_result["aggregate_benchmark"]
        rows.append(
            {
                "budget": budget_result["budget"],
                "cases": len(samples),
                "exact_greedy_ids": all(
                    sample["correctness"]["token_ids_exact"] for sample in samples
                ),
                "exact_cache_contracts": all(
                    sample["acceptance"]["cache_exact"] for sample in samples
                ),
                "max_total_variation_distance": max(
                    sample["acceptance"]["max_total_variation_distance"]
                    for sample in samples
                ),
                "cached_total_mean_ms": benchmark["cached_total_generation"]["mean_ms"],
                "uncached_total_mean_ms": benchmark["uncached_total_generation"]["mean_ms"],
                "cached_vs_uncached_speedup": benchmark[
                    "uncached_over_cached_total_speedup"
                ],
                "peak_process_gpu_memory_mib": budget_result[
                    "peak_process_gpu_memory_mib"
                ],
                "passed": budget_result["passed"],
            }
        )
    summary = {
        "passed": report["passed"],
        "invariants_passed": report["invariants_passed"],
        "source_commit": report["source_commit"],
        "gpu_name": report["gpu_name"],
        "max_new_tokens": report["config"]["max_new_tokens"],
        "warmup_runs": report["config"]["warmup_runs"],
        "measured_runs": report["config"]["measured_runs"],
        "max_total_variation_threshold": report["config"][
            "max_total_variation_distance"
        ],
        "claim_boundary": report["claim_boundary"],
    }
    return rows, summary


def plot_phase2(plt: Any, rows: list[dict[str, Any]], path: Path) -> None:
    budgets = [str(row["budget"]) for row in rows]
    compressed = [row["compressed_max_abs_error"] for row in rows]
    contextual = [row["contextual_max_abs_error"] for row in rows]
    x = np.arange(len(budgets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, compressed, width, label="Compressed tokens")
    ax.bar(x + width / 2, contextual, width, label="Contextual tokens")
    ax.axhline(1e-5, color="#c23b23", linestyle="--", linewidth=1.2, label="atol = 1e-5")
    ax.set_yscale("log")
    ax.set_xticks(x, budgets)
    ax.set_xlabel("Vision token budget (CLS excluded)")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title("Real CLIP PyTorch/Jittor alignment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_phase4b_loss(
    plt: Any,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    steps = [row["optimizer_step"] for row in train_rows]
    raw = [row["train_target_nll"] for row in train_rows]
    smooth = [row["train_target_nll_rolling32"] for row in train_rows]
    val_steps = [row["optimizer_step"] for row in validation_rows]
    val_loss = [row["held_out_target_nll"] for row in validation_rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(steps, raw, color="#4c78a8", alpha=0.16, linewidth=0.7, label="Train NLL (per step)")
    ax.plot(steps, smooth, color="#1f5a94", linewidth=2.0, label="Train NLL (rolling mean, 32)")
    ax.plot(val_steps, val_loss, color="#e45756", marker="o", markersize=4, linewidth=1.8, label="Held-out NLL (1,024 samples)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Target-token negative log-likelihood")
    ax.set_title("Phase 4B Projector-only training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate(plt: Any, train_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(
        [row["optimizer_step"] for row in train_rows],
        [row["learning_rate"] for row in train_rows],
        color="#59a14f",
        linewidth=1.8,
    )
    ax.axvline(67, color="#777777", linestyle="--", linewidth=1, label="Warm-up end")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Phase 4B learning-rate schedule")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_phase5a(plt: Any, rows: list[dict[str, Any]], path: Path) -> None:
    budgets = [str(row["budget"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].bar(x, [row["cached_vs_uncached_speedup"] for row in rows], color="#f28e2b")
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, budgets)
    axes[0].set_xlabel("Vision token budget")
    axes[0].set_ylabel("Uncached / cached total latency")
    axes[0].set_title("Pinned-protocol speedup")
    axes[1].bar(x, [row["peak_process_gpu_memory_mib"] for row in rows], color="#76b7b2")
    axes[1].set_xticks(x, budgets)
    axes[1].set_xlabel("Vision token budget")
    axes[1].set_ylabel("Peak process GPU memory (MiB)")
    axes[1].set_title("Peak memory")
    fig.suptitle("Phase 5A native Jittor GPT-2 KV-cache benchmark")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_visualization_montage(archive: EvidenceArchive, path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required; install the project with the submission extra"
        ) from exc
    budgets = (64, 128, 192)
    images = ("dense", "scene", "text")
    thumb_size = (520, 290)
    header = 44
    margin = 12
    canvas = Image.new(
        "RGB",
        (
            margin + len(images) * (thumb_size[0] + margin),
            margin + len(budgets) * (thumb_size[1] + header + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1 has no size parameter.
        font = ImageFont.load_default()
    for row_index, budget in enumerate(budgets):
        for column_index, image_name in enumerate(images):
            suffix = (
                f"reference_clip_{budget}_code_exact_float32_real_clip_"
                f"{image_name}_tokens.png"
            )
            image = Image.open(io.BytesIO(archive.read_bytes(suffix))).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x = margin + column_index * (thumb_size[0] + margin)
            y = margin + row_index * (thumb_size[1] + header + margin)
            label = f"Budget {budget} | {image_name}.png"
            draw.text((x, y + 8), label, fill="black", font=font)
            paste_y = y + header
            paste_x = x + (thumb_size[0] - image.width) // 2
            canvas.paste(image, (paste_x, paste_y))
    canvas.save(path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"
    results_dir = output_dir / "results"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    archives = {
        "phase2": EvidenceArchive(args.phase2_archive),
        "phase4a": EvidenceArchive(args.phase4a_archive),
        "phase4b": EvidenceArchive(args.phase4b_archive),
        "phase5a": EvidenceArchive(args.phase5a_archive),
    }
    try:
        phase2 = extract_phase2(archives["phase2"])
        phase4a = extract_phase4a(archives["phase4a"])
        phase4b_train, phase4b_validation, phase4b, _ = extract_phase4b(
            archives["phase4b"]
        )
        phase5a_rows, phase5a = extract_phase5a(archives["phase5a"])

        write_csv(
            results_dir / "phase2_real_clip_alignment.csv",
            list(phase2[0].keys()),
            phase2,
        )
        write_csv(
            results_dir / "phase4b_training_trace.csv",
            list(phase4b_train[0].keys()),
            phase4b_train,
        )
        write_csv(
            results_dir / "phase4b_validation_curve.csv",
            list(phase4b_validation[0].keys()),
            phase4b_validation,
        )
        write_csv(
            results_dir / "phase5a_kv_cache_summary.csv",
            list(phase5a_rows[0].keys()),
            phase5a_rows,
        )

        consolidated = {
            "artifact_type": "visionzip_jittor_submission_results_v1",
            "generated_from": {
                key: {
                    "file": archive.path.name,
                    "sha256": sha256_file(archive.path),
                }
                for key, archive in archives.items()
            },
            "phase2_real_clip_alignment": phase2,
            "phase4a_infrastructure_training": phase4a,
            "phase4b_real_paired_training": phase4b,
            "phase5a_kv_cache": {
                **phase5a,
                "budgets": phase5a_rows,
            },
            "claim_boundary": {
                "validated": [
                    "Native Jittor VisionZip core and real-CLIP feature alignment",
                    "Projector-only training with a frozen native Jittor GPT-2",
                    "Held-out target NLL reduction on the pinned 8,192-sample pilot",
                    "Pinned-protocol native Jittor GPT-2 KV-cache correctness and timing",
                ],
                "not_validated": [
                    "All VisionZip paper tables, datasets, tasks, and model scales",
                    "LLaVA-equivalent end-to-end quality",
                    "Human-caption quality or COCO multi-reference benchmark quality",
                    "Universal strict 1e-5 raw-logit equality or universal speedup",
                ],
            },
        }
        (results_dir / "submission_results.json").write_text(
            json.dumps(consolidated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        plt = configure_matplotlib()
        plot_phase2(plt, phase2, assets_dir / "phase2_alignment_errors.png")
        plot_phase4b_loss(
            plt,
            phase4b_train,
            phase4b_validation,
            assets_dir / "phase4b_loss_curve.png",
        )
        plot_learning_rate(
            plt,
            phase4b_train,
            assets_dir / "phase4b_learning_rate.png",
        )
        plot_phase5a(plt, phase5a_rows, assets_dir / "phase5a_kv_cache.png")
        build_visualization_montage(
            archives["phase2"], assets_dir / "visionzip_token_visualizations.png"
        )
    finally:
        for archive in archives.values():
            archive.close()

    print(f"Saved submission assets to: {output_dir}")


if __name__ == "__main__":
    main()
