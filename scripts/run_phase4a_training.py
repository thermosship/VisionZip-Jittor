#!/usr/bin/env python3
"""Train the Phase 4A Projector on real paired image-caption samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import jittor as jt
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionzip_jittor.gpt2 import (
    NativeGPT2LMHeadModel,
    masked_causal_language_loss,
    parameter_count as gpt2_parameter_count,
)
from visionzip_jittor.gpt2_config import load_gpt2_config
from visionzip_jittor.phase4_config import load_phase4a_config
from visionzip_jittor.phase4_data import (
    PairedSample,
    batch_indices_for_step,
    deterministic_split,
    load_paired_manifest,
    load_precomputed_visual_tokens,
)
from visionzip_jittor.phase4_training import (
    build_jittor_training_batch,
    generate_caption,
    gradient_statistics,
    load_phase4_checkpoint,
    max_parameter_delta,
    parameter_sha256,
    save_phase4_checkpoint,
    snapshot_parameters,
)
from visionzip_jittor.projector import MultimodalProjector, parameter_count
from visionzip_jittor.projector_config import ProjectorConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase4a_tiny_overfit.json",
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        default=ROOT / "manifests/phase4a_tiny_pairs.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "outputs/phase3b/gpt2",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "outputs/real_clip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase4a/tiny_overfit",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs/phase4a",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_gpt2_artifacts(artifact_dir: Path, manifest: Dict[str, Any]):
    files = manifest.get("files", {})
    expected = manifest.get("sha256", {})
    actual = {
        "weights": file_sha256(artifact_dir / files["weights"]),
        "config": file_sha256(artifact_dir / files["config"]),
        "tokenizer": directory_sha256(artifact_dir / files["tokenizer"]),
        "reference": file_sha256(artifact_dir / files["reference"]),
    }
    checks = {key: actual[key] == expected.get(key) for key in actual}
    if not all(checks.values()):
        raise ValueError(f"GPT-2 artifact checksum failure: {checks}")
    return {"actual": actual, "matches_manifest": checks}


def reference_path(reference_dir: Path, budget: int) -> Path:
    return reference_dir / (
        f"reference_clip_{budget}_code_exact_float32_real_clip.npz"
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def select_rows(
    visual_tokens: np.ndarray,
    all_samples: Sequence[PairedSample],
    selected: Sequence[PairedSample],
) -> np.ndarray:
    row_by_id = {
        sample.sample_id: index for index, sample in enumerate(all_samples)
    }
    return visual_tokens[
        np.asarray([row_by_id[sample.sample_id] for sample in selected])
    ]


def optimizer_scope_exact(optimizer, parameters: Iterable[jt.Var]) -> bool:
    expected = list(parameters)
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    return (
        len(actual) == len(expected)
        and {id(item) for item in actual} == {id(item) for item in expected}
    )


def compute_loss(
    projector,
    model,
    tokenizer,
    visual_tokens: np.ndarray,
    samples: Sequence[PairedSample],
    config,
) -> float:
    if not samples:
        return float("nan")
    with jt.no_grad():
        visual = jt.array(visual_tokens.astype(np.float32, copy=False))
        visual.stop_grad()
        projected = projector(visual)
        batch = build_jittor_training_batch(
            tokenizer,
            model,
            projected,
            [sample.caption for sample in samples],
            config.prompt,
            config.max_caption_tokens,
        )
        logits = model(
            inputs_embeds=batch["packed_embeddings"],
            attention_mask=batch["attention_mask"],
        )
        loss = masked_causal_language_loss(
            logits,
            batch["labels"],
            batch["label_mask"],
        )
        loss.sync()
        value = float(loss.numpy().item())
    return value


def train_one_step(
    global_step: int,
    projector,
    model,
    tokenizer,
    optimizer,
    train_visual: np.ndarray,
    train_samples: Sequence[PairedSample],
    config,
) -> Dict[str, Any]:
    indices = batch_indices_for_step(
        len(train_samples),
        config.batch_size,
        config.seed,
        global_step,
    )
    batch_samples = [train_samples[index] for index in indices]
    visual = jt.array(train_visual[np.asarray(indices, dtype=np.int64)])
    visual.stop_grad()
    before = snapshot_parameters(projector.parameters())
    started = time.perf_counter()
    projected = projector(visual)
    batch = build_jittor_training_batch(
        tokenizer,
        model,
        projected,
        [sample.caption for sample in batch_samples],
        config.prompt,
        config.max_caption_tokens,
    )
    logits = model(
        inputs_embeds=batch["packed_embeddings"],
        attention_mask=batch["attention_mask"],
    )
    loss = masked_causal_language_loss(
        logits,
        batch["labels"],
        batch["label_mask"],
    )
    loss.sync()
    loss_value = float(loss.numpy().item())
    optimizer.zero_grad()
    optimizer.backward(loss)
    gradients = gradient_statistics(projector.parameters(), optimizer)
    optimizer.step()
    jt.sync_all()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    delta = max_parameter_delta(before, projector.parameters())
    finite = bool(
        math.isfinite(loss_value)
        and gradients["finite"]
        and gradients["l2_norm"] > 0.0
        and delta > 0.0
    )
    return {
        "artifact_type": "phase4a_train_metric_v1",
        "global_step": global_step + 1,
        "batch_indices": list(indices),
        "sample_ids": [sample.sample_id for sample in batch_samples],
        "loss": loss_value,
        "gradient": gradients,
        "projector_max_parameter_delta": delta,
        "elapsed_ms": elapsed_ms,
        "finite_update": finite,
    }


def checkpoint_metadata(
    global_step: int,
    config,
    manifest,
    train_samples,
    validation_samples,
    reference: Path,
) -> Dict[str, Any]:
    return {
        "global_step": global_step,
        "phase4a_config": config.to_dict(),
        "paired_manifest_name": manifest.name,
        "train_sample_ids": [sample.sample_id for sample in train_samples],
        "validation_sample_ids": [
            sample.sample_id for sample in validation_samples
        ],
        "phase2_reference": str(reference),
        "budget": config.budget,
    }


def verify_checkpoint_resume(
    checkpoint: Path,
    global_step: int,
    projector,
    model,
    tokenizer,
    optimizer,
    train_visual,
    train_samples,
    config,
) -> Dict[str, Any]:
    restored = load_phase4_checkpoint(checkpoint, projector, optimizer)
    first = train_one_step(
        global_step,
        projector,
        model,
        tokenizer,
        optimizer,
        train_visual,
        train_samples,
        config,
    )
    first_parameters = snapshot_parameters(projector.parameters())
    first_hash = parameter_sha256(projector.parameters())

    load_phase4_checkpoint(checkpoint, projector, optimizer)
    second = train_one_step(
        global_step,
        projector,
        model,
        tokenizer,
        optimizer,
        train_visual,
        train_samples,
        config,
    )
    second_parameters = snapshot_parameters(projector.parameters())
    second_hash = parameter_sha256(projector.parameters())
    parameter_max_abs_error = max(
        float(
            np.max(
                np.abs(
                    first_value.astype(np.float64)
                    - second_value.astype(np.float64)
                )
            )
        )
        for first_value, second_value in zip(
            first_parameters,
            second_parameters,
        )
    )
    loss_abs_error = abs(first["loss"] - second["loss"])
    bit_exact = bool(
        first["loss"] == second["loss"]
        and first["sample_ids"] == second["sample_ids"]
        and first_hash == second_hash
    )
    numerically_reproduced = bool(
        first["sample_ids"] == second["sample_ids"]
        and loss_abs_error <= config.resume_atol
        and parameter_max_abs_error <= config.resume_atol
    )
    load_phase4_checkpoint(checkpoint, projector, optimizer)
    return {
        "checkpoint": str(checkpoint),
        "restored_global_step": restored.get("global_step"),
        "next_loss_first": first["loss"],
        "next_loss_resumed": second["loss"],
        "loss_abs_error": loss_abs_error,
        "next_projector_sha256_first": first_hash,
        "next_projector_sha256_resumed": second_hash,
        "projector_max_abs_error": parameter_max_abs_error,
        "tolerance": config.resume_atol,
        "bit_exact": bit_exact,
        "numerically_reproduced": numerically_reproduced,
        "passed": numerically_reproduced,
    }


def generate_validation_samples(
    projector,
    model,
    tokenizer,
    visual_tokens,
    samples,
    config,
) -> List[Dict[str, Any]]:
    generated: List[Dict[str, Any]] = []
    with jt.no_grad():
        for index, sample in enumerate(samples):
            visual = jt.array(visual_tokens[index : index + 1])
            visual.stop_grad()
            projected = projector(visual)
            output = generate_caption(
                tokenizer,
                model,
                projected,
                config.prompt,
                config.generation_prompt,
                config.max_new_tokens,
            )
            generated.append(
                {
                    "sample_id": sample.sample_id,
                    "image_name": sample.image_name,
                    "target": sample.caption,
                    **output,
                }
            )
    return generated


def main():
    args = parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    config = load_phase4a_config(args.config)
    paired_manifest = load_paired_manifest(args.paired_manifest)
    train_samples, validation_samples = deterministic_split(
        paired_manifest.samples,
        config.validation_fraction,
        config.seed,
    )
    reference = reference_path(args.reference_dir, config.budget)
    visual_tokens, visual_report = load_precomputed_visual_tokens(
        reference,
        paired_manifest.samples,
    )
    if list(visual_tokens.shape[1:]) != [config.budget + 1, config.vision_hidden_size]:
        raise ValueError(
            "Phase 2 token shape does not match Phase 4A config: "
            f"{list(visual_tokens.shape)}"
        )
    train_visual = select_rows(
        visual_tokens,
        paired_manifest.samples,
        train_samples,
    )
    validation_visual = select_rows(
        visual_tokens,
        paired_manifest.samples,
        validation_samples,
    )

    artifact_manifest = json.loads(
        (args.artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not artifact_manifest.get("real_llm"):
        raise ValueError("GPT-2 artifact manifest must declare real_llm=true")
    artifact_checks = verify_gpt2_artifacts(
        args.artifact_dir,
        artifact_manifest,
    )
    gpt2_config = load_gpt2_config(args.artifact_dir / "hf_config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        args.artifact_dir / "tokenizer",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 72)
    print("Phase 4A: loading frozen native Jittor GPT-2")
    print("=" * 72)
    model = NativeGPT2LMHeadModel(gpt2_config)
    load_report = model.load_npz_weights(
        args.artifact_dir / "gpt2_float32_weights.npz"
    )
    model.freeze_parameters()
    model.eval()
    language_parameters = list(model.parameters())
    language_hash_before = parameter_sha256(language_parameters)

    projector_config = ProjectorConfig(
        projector_type=config.projector_type,
        vision_hidden_size=config.vision_hidden_size,
        language_hidden_size=gpt2_config.n_embd,
        vocab_size=gpt2_config.vocab_size,
        prefix_tokens=0,
        suffix_tokens=0,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    jt.set_global_seed(config.seed + config.budget)
    projector = MultimodalProjector(projector_config)
    projector_parameters = list(projector.parameters())
    optimizer = jt.optim.Adam(
        projector_parameters,
        lr=config.learning_rate,
    )
    scope_exact = optimizer_scope_exact(optimizer, projector_parameters)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    metrics_path = args.log_dir / "train_metrics.jsonl"
    summary_path = args.log_dir / "phase4a_summary.json"
    if args.resume is None:
        metrics_path.write_text("", encoding="utf-8")
        start_step = 0
    else:
        restored = load_phase4_checkpoint(args.resume, projector, optimizer)
        start_step = int(restored["global_step"])
        if restored.get("phase4a_config") != config.to_dict():
            raise ValueError("resume checkpoint config does not match current config")
        if restored.get("train_sample_ids") != [
            sample.sample_id for sample in train_samples
        ]:
            raise ValueError("resume checkpoint train split does not match")

    initial_train_loss = compute_loss(
        projector,
        model,
        tokenizer,
        train_visual,
        train_samples,
        config,
    )
    initial_validation_loss = compute_loss(
        projector,
        model,
        tokenizer,
        validation_visual,
        validation_samples,
        config,
    )
    print(
        json.dumps(
            {
                "start_step": start_step,
                "initial_train_loss": initial_train_loss,
                "initial_validation_loss": initial_validation_loss,
                "train_sample_ids": [sample.sample_id for sample in train_samples],
                "validation_sample_ids": [
                    sample.sample_id for sample in validation_samples
                ],
            },
            indent=2,
        )
    )

    step_results: List[Dict[str, Any]] = []
    all_updates_finite = True
    for global_step in range(start_step, config.max_steps):
        metric = train_one_step(
            global_step,
            projector,
            model,
            tokenizer,
            optimizer,
            train_visual,
            train_samples,
            config,
        )
        all_updates_finite = all_updates_finite and metric["finite_update"]
        if metric["global_step"] % config.log_every == 0:
            metric["train_loss_full"] = compute_loss(
                projector,
                model,
                tokenizer,
                train_visual,
                train_samples,
                config,
            )
            metric["validation_loss"] = compute_loss(
                projector,
                model,
                tokenizer,
                validation_visual,
                validation_samples,
                config,
            )
            append_jsonl(metrics_path, metric)
            step_results.append(metric)
            print(json.dumps(metric, ensure_ascii=False))
        if metric["global_step"] % config.checkpoint_every == 0:
            checkpoint = checkpoint_dir / (
                f"projector_step_{metric['global_step']:06d}.npz"
            )
            save_phase4_checkpoint(
                checkpoint,
                projector,
                optimizer,
                checkpoint_metadata(
                    metric["global_step"],
                    config,
                    paired_manifest,
                    train_samples,
                    validation_samples,
                    reference,
                ),
            )
        jt.gc()

    final_step = config.max_steps
    final_checkpoint = checkpoint_dir / f"projector_step_{final_step:06d}.npz"
    save_phase4_checkpoint(
        final_checkpoint,
        projector,
        optimizer,
        checkpoint_metadata(
            final_step,
            config,
            paired_manifest,
            train_samples,
            validation_samples,
            reference,
        ),
    )
    final_train_loss = compute_loss(
        projector,
        model,
        tokenizer,
        train_visual,
        train_samples,
        config,
    )
    final_validation_loss = compute_loss(
        projector,
        model,
        tokenizer,
        validation_visual,
        validation_samples,
        config,
    )
    resume_check = (
        verify_checkpoint_resume(
            final_checkpoint,
            final_step,
            projector,
            model,
            tokenizer,
            optimizer,
            train_visual,
            train_samples,
            config,
        )
        if config.verify_resume
        else {"exact": None, "skipped": True}
    )
    generation_samples = validation_samples or train_samples
    generation_visual = validation_visual if validation_samples else train_visual
    generated = generate_validation_samples(
        projector,
        model,
        tokenizer,
        generation_visual,
        generation_samples,
        config,
    )
    language_hash_after = parameter_sha256(language_parameters)
    language_unchanged = language_hash_before == language_hash_after
    loss_improvement = initial_train_loss - final_train_loss
    loss_decreased = bool(
        math.isfinite(initial_train_loss)
        and math.isfinite(final_train_loss)
        and loss_improvement > config.minimum_loss_improvement
    )
    passed = bool(
        artifact_manifest.get("architecture") == "GPT2LMHeadModel"
        and artifact_manifest.get("model_name_or_path")
        == config.model_name_or_path
        and load_report["tensor_count"]
        == artifact_manifest.get("exported_tensor_count")
        and gpt2_parameter_count(language_parameters)
        == artifact_manifest.get("parameter_count")
        and model.all_parameters_stop_grad()
        and language_unchanged
        and scope_exact
        and all(not item.is_stop_grad() for item in projector_parameters)
        and all_updates_finite
        and loss_decreased
        and (resume_check["passed"] is True or not config.verify_resume)
        and all(item["token_ids"] for item in generated)
    )
    summary = {
        "artifact_type": "phase4a_real_paired_projector_training_v1",
        "passed": passed,
        "device": args.device,
        "jittor_version": jt.__version__,
        "phase4a_config": config.to_dict(),
        "paired_manifest": paired_manifest.to_dict(),
        "split": {
            "train_sample_ids": [sample.sample_id for sample in train_samples],
            "validation_sample_ids": [
                sample.sample_id for sample in validation_samples
            ],
        },
        "precomputed_visual_features": visual_report,
        "gpt2_artifact_checks": artifact_checks,
        "gpt2_weight_loading": load_report,
        "language_parameter_count": gpt2_parameter_count(language_parameters),
        "language_all_stop_grad": model.all_parameters_stop_grad(),
        "language_sha256_before": language_hash_before,
        "language_sha256_after": language_hash_after,
        "language_unchanged": language_unchanged,
        "projector_parameter_count": parameter_count(projector),
        "projector_optimizer_scope_exact": scope_exact,
        "projector_all_trainable": all(
            not item.is_stop_grad() for item in projector_parameters
        ),
        "start_step": start_step,
        "final_step": final_step,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "train_loss_improvement": loss_improvement,
        "train_loss_decreased": loss_decreased,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "all_updates_finite": all_updates_finite,
        "logged_steps": step_results,
        "final_checkpoint": str(final_checkpoint),
        "checkpoint_resume": resume_check,
        "generated": generated,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("Phase 4A summary")
    print("=" * 72)
    print(text)
    summary_path.write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(
        text + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
