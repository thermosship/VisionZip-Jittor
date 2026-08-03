#!/usr/bin/env python3
"""Train the Phase 4B Projector on frozen licensed real-image features."""

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
from visionzip_jittor.phase4_training import (
    build_jittor_training_batch,
    generate_caption,
    gradient_statistics,
    load_phase4_checkpoint,
    max_parameter_delta,
    parameter_sha256,
    save_phase4_checkpoint,
    snapshot_parameters,
    step_adam_after_gradient_accumulation,
)
from visionzip_jittor.phase4b_config import load_phase4b_config
from visionzip_jittor.phase4b_data import canonical_json_sha256, file_sha256
from visionzip_jittor.phase4b_training import (
    GpuMemorySampler,
    Phase4BLoadedFeatures,
    batch_indices_for_optimizer_step,
    caption_metrics,
    checkpoints_to_remove,
    deterministic_subset_indices,
    learning_rate_for_optimizer_step,
    load_phase4b_training_features,
    summarize_training_benchmark,
    training_benchmark_is_acceptable,
)
from visionzip_jittor.projector import MultimodalProjector, parameter_count
from visionzip_jittor.projector_config import ProjectorConfig


CHECKPOINT_ARTIFACT_TYPE = "phase4b_projector_checkpoint_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json",
    )
    parser.add_argument(
        "--prepared-manifest",
        type=Path,
        default=(
            ROOT
            / "datasets/phase4b/commoncatalog_cc_by_8k/manifest.json"
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/phase4b/commoncatalog_cc_by_8k/features/manifest.json"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "outputs/phase3b/gpt2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase4b/commoncatalog_cc_by_8k/training",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs/phase4b",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stop-after-optimizer-step",
        type=int,
        default=None,
        help="Intentional step-boundary stop used only for resume validation.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


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
    matches = {key: actual[key] == expected.get(key) for key in actual}
    if not all(matches.values()):
        raise ValueError(f"GPT-2 artifact checksum failure: {matches}")
    return {"actual": actual, "matches_manifest": matches}


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


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def select_global_rows(
    loaded: Phase4BLoadedFeatures,
    split_indices: np.ndarray,
    local_indices: Sequence[int],
) -> np.ndarray:
    return split_indices[np.asarray(local_indices, dtype=np.int64)]


def evaluate_target_nll(
    projector,
    model,
    tokenizer,
    loaded: Phase4BLoadedFeatures,
    global_indices: np.ndarray,
    config,
) -> Dict[str, Any]:
    """Compute target-token-weighted held-out NLL and perplexity."""

    if len(global_indices) == 0:
        raise ValueError("evaluation split must not be empty")
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    projector.eval()
    with jt.no_grad():
        for start in range(0, len(global_indices), config.training.micro_batch_size):
            rows = global_indices[start : start + config.training.micro_batch_size]
            visual = jt.array(
                loaded.compressed_tokens[rows].astype(np.float32, copy=False)
            )
            visual.stop_grad()
            projected = projector(visual)
            samples = [loaded.samples[int(index)] for index in rows]
            batch = build_jittor_training_batch(
                tokenizer,
                model,
                projected,
                [sample.caption for sample in samples],
                config.training.prompt,
                config.training.max_caption_tokens,
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
            token_count = int(sum(batch["target_token_counts"]))
            total_nll += float(loss.numpy().item()) * token_count
            total_tokens += token_count
            jt.gc()
    mean_nll = total_nll / total_tokens
    return {
        "artifact_type": "phase4b_held_out_nll_v1",
        "sample_count": int(len(global_indices)),
        "target_token_count": total_tokens,
        "target_nll_sum": total_nll,
        "held_out_target_nll": mean_nll,
        "held_out_target_perplexity": (
            math.exp(mean_nll) if mean_nll < 709.0 else float("inf")
        ),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def train_optimizer_step(
    completed_steps: int,
    projector,
    model,
    tokenizer,
    optimizer,
    loaded: Phase4BLoadedFeatures,
    config,
) -> Dict[str, Any]:
    settings = config.training
    microbatches = batch_indices_for_optimizer_step(
        len(loaded.train_indices),
        settings.micro_batch_size,
        settings.gradient_accumulation_steps,
        completed_steps,
        settings.seed,
    )
    learning_rate = learning_rate_for_optimizer_step(
        completed_steps + 1,
        settings.learning_rate,
        settings.warmup_steps,
        settings.max_optimizer_steps,
    )
    optimizer.lr = learning_rate
    optimizer.zero_grad()
    projector.train()
    losses: List[float] = []
    token_counts: List[int] = []
    sample_ids: List[str] = []
    microbatch_rows = [
        select_global_rows(loaded, loaded.train_indices, local_indices)
        for local_indices in microbatches
    ]
    expected_token_counts = []
    for rows in microbatch_rows:
        count = 0
        for index in rows:
            encoded = tokenizer.encode(
                loaded.samples[int(index)].caption,
                add_special_tokens=False,
            )
            count += min(len(encoded), settings.max_caption_tokens - 1) + 1
        expected_token_counts.append(count)
    effective_target_tokens = sum(expected_token_counts)
    started = time.perf_counter()
    for rows, expected_token_count in zip(
        microbatch_rows,
        expected_token_counts,
    ):
        samples = [loaded.samples[int(index)] for index in rows]
        sample_ids.extend(sample.sample_id for sample in samples)
        visual = jt.array(
            loaded.compressed_tokens[rows].astype(np.float32, copy=False)
        )
        visual.stop_grad()
        projected = projector(visual)
        batch = build_jittor_training_batch(
            tokenizer,
            model,
            projected,
            [sample.caption for sample in samples],
            settings.prompt,
            settings.max_caption_tokens,
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
        losses.append(float(loss.numpy().item()))
        token_count = int(sum(batch["target_token_counts"]))
        if token_count != expected_token_count:
            raise ValueError("precomputed caption token count mismatch")
        token_counts.append(token_count)
        optimizer.backward(loss * token_count / effective_target_tokens)
    gradients = gradient_statistics(projector.parameters(), optimizer)
    corrected_n_step = step_adam_after_gradient_accumulation(
        optimizer,
        completed_steps,
    )
    jt.sync_all()
    weighted_loss = sum(
        loss * count for loss, count in zip(losses, token_counts)
    ) / sum(token_counts)
    finite = bool(
        math.isfinite(weighted_loss)
        and gradients["finite"]
        and gradients["l2_norm"] > 0.0
        and corrected_n_step == completed_steps + 1
    )
    return {
        "artifact_type": "phase4b_train_metric_v1",
        "optimizer_step": completed_steps + 1,
        "learning_rate": learning_rate,
        "microbatch_local_indices": [list(item) for item in microbatches],
        "sample_ids": sample_ids,
        "microbatch_losses": losses,
        "microbatch_target_token_counts": token_counts,
        "target_token_weighted_nll": weighted_loss,
        "gradient": gradients,
        "optimizer_n_step": corrected_n_step,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "finite_update": finite,
    }


def checkpoint_metadata(
    completed_steps: int,
    config,
    loaded: Phase4BLoadedFeatures,
    feature_manifest_path: Path,
) -> Dict[str, Any]:
    return {
        "completed_optimizer_steps": completed_steps,
        "phase4b_config": config.to_dict(),
        "phase4b_config_sha256": canonical_json_sha256(config.to_dict()),
        "prepared_samples_sha256": loaded.prepared_manifest.samples_sha256,
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "feature_model_name_or_path": loaded.feature_manifest.model_name_or_path,
        "feature_model_revision": loaded.feature_manifest.model_revision,
        "train_sample_count": int(len(loaded.train_indices)),
        "validation_sample_count": int(len(loaded.validation_indices)),
        "scheduler": "linear_warmup_then_cosine_decay_v1",
        "checkpoint_boundary": "complete_optimizer_step",
    }


def save_training_checkpoint(
    path: Path,
    completed_steps: int,
    projector,
    optimizer,
    config,
    loaded: Phase4BLoadedFeatures,
    feature_manifest_path: Path,
) -> None:
    save_phase4_checkpoint(
        path,
        projector,
        optimizer,
        checkpoint_metadata(
            completed_steps,
            config,
            loaded,
            feature_manifest_path,
        ),
        artifact_type=CHECKPOINT_ARTIFACT_TYPE,
    )


def validate_resume_metadata(
    metadata: Dict[str, Any],
    config,
    loaded: Phase4BLoadedFeatures,
    feature_manifest_path: Path,
) -> int:
    expected = checkpoint_metadata(
        int(metadata["completed_optimizer_steps"]),
        config,
        loaded,
        feature_manifest_path,
    )
    for key in (
        "phase4b_config_sha256",
        "prepared_samples_sha256",
        "feature_manifest_sha256",
        "feature_model_name_or_path",
        "feature_model_revision",
        "train_sample_count",
        "validation_sample_count",
        "scheduler",
        "checkpoint_boundary",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"resume checkpoint identity mismatch for {key}")
    completed = int(metadata["completed_optimizer_steps"])
    if not 0 <= completed <= config.training.max_optimizer_steps:
        raise ValueError("resume completed_optimizer_steps is outside the run")
    if int(metadata["optimizer_n_step"]) != completed:
        raise ValueError("resume Adam n_step does not match optimizer-step boundary")
    return completed


def generate_held_out_captions(
    projector,
    model,
    tokenizer,
    loaded: Phase4BLoadedFeatures,
    config,
) -> Dict[str, Any]:
    settings = config.training
    local_indices = deterministic_subset_indices(
        len(loaded.validation_indices),
        settings.generation_eval_samples,
        settings.seed,
    )
    generated: List[Dict[str, Any]] = []
    projector.eval()
    with jt.no_grad():
        for local_index in local_indices:
            global_index = int(loaded.validation_indices[local_index])
            sample = loaded.samples[global_index]
            visual = jt.array(
                loaded.compressed_tokens[global_index : global_index + 1].astype(
                    np.float32,
                    copy=False,
                )
            )
            visual.stop_grad()
            projected = projector(visual)
            output = generate_caption(
                tokenizer,
                model,
                projected,
                settings.prompt,
                settings.generation_prompt,
                settings.max_new_tokens,
            )
            generated.append(
                {
                    "sample_id": sample.sample_id,
                    "reference": sample.caption,
                    "hypothesis": output["text"],
                    "token_ids": output["token_ids"],
                }
            )
            jt.gc()
    metrics = caption_metrics(
        [item["reference"] for item in generated],
        [item["hypothesis"] for item in generated],
    )
    return {
        "artifact_type": "phase4b_held_out_generation_v1",
        "subset_local_indices": list(local_indices),
        "metrics": metrics,
        "samples": generated,
    }


def main():
    args = parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    config = load_phase4b_config(args.config)
    settings = config.training
    if args.stop_after_optimizer_step is not None and not (
        0 < args.stop_after_optimizer_step <= settings.max_optimizer_steps
    ):
        raise ValueError("--stop-after-optimizer-step is outside the configured run")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.log_dir / "phase4b_train_metrics.jsonl"
    summary_path = args.log_dir / "phase4b_training_summary.json"

    print("=" * 72)
    print("Phase 4B: validating prepared data and frozen feature store")
    print("=" * 72)
    loaded = load_phase4b_training_features(
        args.prepared_manifest,
        args.feature_manifest,
        verify_images=False,
        verify_shards=True,
    )
    feature_identity_checks = {
        "config_sha256_exact": (
            loaded.feature_manifest.config_sha256
            == canonical_json_sha256(config.to_dict())
        ),
        "model_name_exact": (
            loaded.feature_manifest.model_name_or_path
            == config.features.model_name_or_path
        ),
        "model_revision_exact": (
            loaded.feature_manifest.model_revision
            == config.features.model_revision
        ),
        "requested_layer_exact": (
            loaded.feature_manifest.requested_layer_index
            == config.features.requested_layer_index
        ),
        "storage_dtype_exact": (
            loaded.feature_manifest.storage_dtype
            == config.features.storage_dtype
        ),
        "token_shape_exact": (
            loaded.feature_manifest.token_shape
            == (settings.budget + 1, settings.vision_hidden_size)
        ),
        "train_count_exact": (
            len(loaded.train_indices) == config.dataset.train_sample_count
        ),
        "validation_count_exact": (
            len(loaded.validation_indices)
            == config.dataset.validation_sample_count
        ),
    }
    if not all(feature_identity_checks.values()):
        raise ValueError(f"Phase 4B feature identity failure: {feature_identity_checks}")

    artifact_manifest = json.loads(
        (args.artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not artifact_manifest.get("real_llm"):
        raise ValueError("GPT-2 artifact manifest must declare real_llm=true")
    artifact_checksums = verify_gpt2_artifacts(args.artifact_dir, artifact_manifest)
    if artifact_manifest.get("model_name_or_path") != settings.model_name_or_path:
        raise ValueError("GPT-2 artifact model identity does not match Phase 4B config")
    gpt2_config = load_gpt2_config(args.artifact_dir / "hf_config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        args.artifact_dir / "tokenizer",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 72)
    print("Phase 4B: loading frozen native Jittor GPT-2")
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
        projector_type=settings.projector_type,
        vision_hidden_size=settings.vision_hidden_size,
        language_hidden_size=gpt2_config.n_embd,
        vocab_size=gpt2_config.vocab_size,
        prefix_tokens=0,
        suffix_tokens=0,
        learning_rate=settings.learning_rate,
        seed=settings.seed,
    )
    jt.set_global_seed(settings.seed + settings.budget)
    projector = MultimodalProjector(projector_config)
    projector_parameters = list(projector.parameters())
    optimizer = jt.optim.Adam(
        projector_parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scope_exact = optimizer_scope_exact(optimizer, projector_parameters)

    if args.resume is None:
        metrics_path.write_text("", encoding="utf-8")
        start_step = 0
    else:
        restored = load_phase4_checkpoint(
            args.resume,
            projector,
            optimizer,
            expected_artifact_type=CHECKPOINT_ARTIFACT_TYPE,
        )
        start_step = validate_resume_metadata(
            restored,
            config,
            loaded,
            args.feature_manifest,
        )
    initial_projector = snapshot_parameters(projector_parameters)
    requested_stop = args.stop_after_optimizer_step or settings.max_optimizer_steps
    if requested_stop < start_step:
        raise ValueError("requested stop step is before the resume checkpoint")

    initial_validation = evaluate_target_nll(
        projector,
        model,
        tokenizer,
        loaded,
        loaded.validation_indices,
        config,
    )
    initial_validation["optimizer_step"] = start_step
    append_jsonl(metrics_path, initial_validation)
    print(json.dumps(initial_validation, ensure_ascii=False))
    best_nll = initial_validation["held_out_target_nll"]
    best_step = start_step
    best_checkpoint = args.output_dir / "best_projector.npz"
    save_training_checkpoint(
        best_checkpoint,
        start_step,
        projector,
        optimizer,
        config,
        loaded,
        args.feature_manifest,
    )

    all_updates_finite = True
    evaluation_history: List[Dict[str, Any]] = [initial_validation]
    invocation_train_metrics: List[Dict[str, Any]] = []
    memory_sampler = GpuMemorySampler(
        enabled=args.device == "cuda",
        interval=0.1,
    )
    memory_sampling_started = False
    final_step = start_step
    try:
        for completed_steps in range(start_step, requested_stop):
            # A one-based optimizer step is post-warm-up only when it is
            # strictly greater than warmup_steps. Start sampling immediately
            # before that first update and keep setup/warm-up peaks excluded.
            if (
                not memory_sampling_started
                and completed_steps >= settings.warmup_steps
            ):
                memory_sampler.start()
                memory_sampling_started = True
            metric = train_optimizer_step(
                completed_steps,
                projector,
                model,
                tokenizer,
                optimizer,
                loaded,
                config,
            )
            invocation_train_metrics.append(metric)
            final_step = metric["optimizer_step"]
            all_updates_finite = all_updates_finite and metric["finite_update"]
            append_jsonl(metrics_path, metric)
            print(json.dumps(metric, ensure_ascii=False))

            if final_step % settings.evaluation_every == 0:
                evaluation = evaluate_target_nll(
                    projector,
                    model,
                    tokenizer,
                    loaded,
                    loaded.validation_indices,
                    config,
                )
                evaluation["optimizer_step"] = final_step
                append_jsonl(metrics_path, evaluation)
                evaluation_history.append(evaluation)
                print(json.dumps(evaluation, ensure_ascii=False))
                if evaluation["held_out_target_nll"] < best_nll:
                    best_nll = evaluation["held_out_target_nll"]
                    best_step = final_step
                    save_training_checkpoint(
                        best_checkpoint,
                        final_step,
                        projector,
                        optimizer,
                        config,
                        loaded,
                        args.feature_manifest,
                    )

            if final_step % settings.checkpoint_every == 0:
                checkpoint = checkpoint_dir / f"projector_step_{final_step:06d}.npz"
                save_training_checkpoint(
                    checkpoint,
                    final_step,
                    projector,
                    optimizer,
                    config,
                    loaded,
                    args.feature_manifest,
                )
                for stale in checkpoints_to_remove(
                    checkpoint_dir.glob("projector_step_*.npz"),
                    keep_last=settings.keep_last_checkpoints,
                ):
                    stale.unlink()
            jt.gc()
    finally:
        memory_sampler.stop()

    final_checkpoint = checkpoint_dir / f"projector_step_{final_step:06d}.npz"
    save_training_checkpoint(
        final_checkpoint,
        final_step,
        projector,
        optimizer,
        config,
        loaded,
        args.feature_manifest,
    )
    for stale in checkpoints_to_remove(
        checkpoint_dir.glob("projector_step_*.npz"),
        keep_last=settings.keep_last_checkpoints,
    ):
        stale.unlink()

    if not evaluation_history or evaluation_history[-1].get("optimizer_step") != final_step:
        final_validation = evaluate_target_nll(
            projector,
            model,
            tokenizer,
            loaded,
            loaded.validation_indices,
            config,
        )
        final_validation["optimizer_step"] = final_step
        append_jsonl(metrics_path, final_validation)
        evaluation_history.append(final_validation)
        if final_validation["held_out_target_nll"] < best_nll:
            best_nll = final_validation["held_out_target_nll"]
            best_step = final_step
            save_training_checkpoint(
                best_checkpoint,
                final_step,
                projector,
                optimizer,
                config,
                loaded,
                args.feature_manifest,
            )
    else:
        final_validation = evaluation_history[-1]

    completed_training = final_step == settings.max_optimizer_steps
    training_benchmark = summarize_training_benchmark(
        invocation_train_metrics,
        settings.warmup_steps,
        peak_process_gpu_memory_mib=memory_sampler.peak_mib,
        gpu_memory_sample_count=memory_sampler.sample_count,
        gpu_memory_sampling_interval_seconds=memory_sampler.interval,
    )
    training_benchmark_accepted = training_benchmark_is_acceptable(
        training_benchmark,
        require_gpu_memory=args.device == "cuda",
    )
    generation = (
        generate_held_out_captions(
            projector,
            model,
            tokenizer,
            loaded,
            config,
        )
        if completed_training
        else None
    )

    # Jittor 1.3.11 implements Module.eval() by marking parameters as
    # stop-grad. Evaluation above is intentionally read-only, but the final
    # Projector trainability invariant must be checked after restoring the
    # module to training mode. This changes state only, not parameter values.
    projector.train()
    projector_trainability_restored = all(
        not item.is_stop_grad() for item in projector_parameters
    )

    language_hash_after = parameter_sha256(language_parameters)
    language_unchanged = language_hash_before == language_hash_after
    projector_delta = max_parameter_delta(initial_projector, projector_parameters)
    artifact_model_checks = {
        "architecture_is_gpt2_lm_head": (
            artifact_manifest.get("architecture") == "GPT2LMHeadModel"
        ),
        "loaded_tensor_count_matches_manifest": (
            load_report["tensor_count"]
            == artifact_manifest.get("exported_tensor_count")
        ),
        "parameter_count_matches_manifest": (
            gpt2_parameter_count(language_parameters)
            == artifact_manifest.get("parameter_count")
        ),
    }
    run_integrity_passed = bool(
        all(feature_identity_checks.values())
        and all(artifact_checksums["matches_manifest"].values())
        and all(artifact_model_checks.values())
        and model.all_parameters_stop_grad()
        and language_unchanged
        and scope_exact
        and projector_trainability_restored
        and all_updates_finite
        and projector_delta > 0.0
        and int(optimizer.n_step) == final_step
        and math.isfinite(final_validation["held_out_target_nll"])
        and (not completed_training or training_benchmark_accepted)
        and (
            generation is None
            or all(item["token_ids"] for item in generation["samples"])
        )
    )
    summary = {
        "artifact_type": "phase4b_real_paired_projector_training_v1",
        "passed": run_integrity_passed,
        "completed_training": completed_training,
        "claim_boundary": (
            "Phase 4B held-out results are reportable only when completed_training=true."
        ),
        "device": args.device,
        "jittor_version": jt.__version__,
        "phase4b_config": config.to_dict(),
        "phase4b_config_sha256": canonical_json_sha256(config.to_dict()),
        "prepared_manifest": loaded.prepared_manifest.to_dict(),
        "feature_manifest": loaded.feature_manifest.to_dict(),
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "feature_identity_checks": feature_identity_checks,
        "gpt2_artifact_checksums": artifact_checksums,
        "gpt2_artifact_model_checks": artifact_model_checks,
        "gpt2_weight_loading": load_report,
        "language_parameter_count": gpt2_parameter_count(language_parameters),
        "language_all_stop_grad": model.all_parameters_stop_grad(),
        "language_sha256_before": language_hash_before,
        "language_sha256_after": language_hash_after,
        "language_unchanged": language_unchanged,
        "projector_parameter_count": parameter_count(projector),
        "projector_optimizer_scope_exact": scope_exact,
        "projector_all_trainable": projector_trainability_restored,
        "projector_trainability_restored_after_evaluation": (
            projector_trainability_restored
        ),
        "projector_max_parameter_delta_from_run_start": projector_delta,
        "start_optimizer_step": start_step,
        "final_optimizer_step": final_step,
        "optimizer_n_step": int(optimizer.n_step),
        "intentional_stop_after_optimizer_step": args.stop_after_optimizer_step,
        "all_updates_finite": all_updates_finite,
        "training_benchmark_accepted": training_benchmark_accepted,
        "training_benchmark": training_benchmark,
        "initial_validation": initial_validation,
        "evaluation_history": evaluation_history,
        "final_validation": final_validation,
        "best_validation": {
            "optimizer_step": best_step,
            "held_out_target_nll": best_nll,
            "checkpoint": str(best_checkpoint),
        },
        "held_out_generation": generation,
        "final_checkpoint": str(final_checkpoint),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("Phase 4B training summary")
    print("=" * 72)
    print(text)
    summary_path.write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(
        text + "\n",
        encoding="utf-8",
    )
    if not run_integrity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
