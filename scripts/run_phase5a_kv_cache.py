#!/usr/bin/env python3
"""Validate and benchmark native Jittor GPT-2 KV-cache generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

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
    greedy_generate_from_embeddings,
    greedy_generate_from_embeddings_cached,
    parameter_count as gpt2_parameter_count,
)
from visionzip_jittor.gpt2_config import load_gpt2_config
from visionzip_jittor.phase4_training import (
    build_generation_embeddings,
    load_phase4_checkpoint,
    parameter_sha256,
)
from visionzip_jittor.phase4b_config import load_phase4b_config
from visionzip_jittor.phase4b_data import canonical_json_sha256
from visionzip_jittor.phase4b_features import load_feature_manifest
from visionzip_jittor.phase4b_training import GpuMemorySampler
from visionzip_jittor.projector import MultimodalProjector, parameter_count
from visionzip_jittor.projector_config import ProjectorConfig


CHECKPOINT_ARTIFACT_TYPE = "phase4b_projector_checkpoint_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase5a_kv_cache.json",
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
        "--phase4b-config",
        type=Path,
        default=ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json",
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
        "--projector-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/phase4b/commoncatalog_cc_by_8k/"
            "training_benchmark_0f53a93/best_projector.npz"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "logs/phase5a/kv_cache_benchmark.json",
    )
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


def source_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def load_benchmark_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {
        "artifact_type",
        "name",
        "budgets",
        "sample_indices",
        "max_new_tokens",
        "warmup_runs",
        "measured_runs",
        "atol",
        "rtol",
        "require_exact_token_ids",
        "require_language_unchanged",
        "require_projector_unchanged",
        "require_speedup",
        "gpu_memory_sampling_interval_seconds",
    }
    unknown = sorted(set(payload) - required)
    missing = sorted(required - set(payload))
    if unknown or missing:
        raise ValueError(
            f"Phase 5A config keys mismatch: missing={missing}, unknown={unknown}"
        )
    if payload["artifact_type"] != "phase5a_kv_cache_benchmark_config_v1":
        raise ValueError("unsupported Phase 5A config artifact_type")
    budgets = [int(item) for item in payload["budgets"]]
    sample_indices = [int(item) for item in payload["sample_indices"]]
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be non-empty and unique")
    if any(item <= 0 for item in budgets):
        raise ValueError("budgets must be positive")
    if not sample_indices or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must be non-empty and unique")
    if any(item < 0 for item in sample_indices):
        raise ValueError("sample_indices must be non-negative")
    for name in ("max_new_tokens", "measured_runs"):
        if int(payload[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(payload["warmup_runs"]) < 0:
        raise ValueError("warmup_runs must be non-negative")
    for name in ("atol", "rtol"):
        value = float(payload[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "require_exact_token_ids",
        "require_language_unchanged",
        "require_projector_unchanged",
        "require_speedup",
    ):
        if not isinstance(payload[name], bool):
            raise ValueError(f"{name} must be a JSON boolean")
    interval = float(payload["gpu_memory_sampling_interval_seconds"])
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("GPU memory sampling interval must be positive")
    payload["budgets"] = budgets
    payload["sample_indices"] = sample_indices
    return payload


def reference_path(reference_dir: Path, budget: int) -> Path:
    return reference_dir / (
        f"reference_clip_{budget}_code_exact_float32_real_clip.npz"
    )


def load_reference_samples(
    path: Path,
    budget: int,
    sample_indices: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 reference: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"compressed_tokens", "metadata_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Phase 2 reference is missing arrays: {missing}")
        tokens = archive["compressed_tokens"].astype(np.float32, copy=True)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("artifact_type") != "real_clip_reference_v1":
        raise ValueError("Phase 2 reference artifact type mismatch")
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (budget + 1, 1024):
        raise ValueError(
            f"Budget {budget} reference token shape mismatch: {tokens.shape}"
        )
    if not np.all(np.isfinite(tokens)):
        raise ValueError(f"Budget {budget} reference contains non-finite tokens")
    images = metadata.get("images")
    if not isinstance(images, list) or len(images) != len(tokens):
        raise ValueError("Phase 2 image metadata does not match token rows")
    if any(index >= len(tokens) for index in sample_indices):
        raise ValueError("sample index is outside the Phase 2 reference")
    selected = tokens[np.asarray(sample_indices, dtype=np.int64)]
    selected_images = [dict(images[index]) for index in sample_indices]
    return selected, metadata, selected_images


def verify_gpt2_artifacts(
    artifact_dir: Path,
    manifest: Dict[str, Any],
) -> Dict[str, str]:
    files = manifest.get("files", {})
    expected = manifest.get("sha256", {})
    required = {"weights", "config", "tokenizer", "reference"}
    if set(files) != required or set(expected) != required:
        raise ValueError("GPT-2 artifact manifest file keys mismatch")
    actual = {
        "weights": file_sha256(artifact_dir / files["weights"]),
        "config": file_sha256(artifact_dir / files["config"]),
        "tokenizer": directory_sha256(artifact_dir / files["tokenizer"]),
        "reference": file_sha256(artifact_dir / files["reference"]),
    }
    if actual != expected:
        raise ValueError(
            f"GPT-2 artifact checksum mismatch: expected={expected}, actual={actual}"
        )
    return actual


def summarize_timings(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(item) for item in values]
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("timing samples must be finite and positive")
    return {
        "runs": len(values),
        "raw_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95_ms": float(np.percentile(np.asarray(values), 95)),
    }


def timed_runs(function, warmup_runs: int, measured_runs: int) -> Dict[str, Any]:
    for _ in range(warmup_runs):
        function()
        jt.sync_all()
    elapsed = []
    for _ in range(measured_runs):
        jt.sync_all()
        started = time.perf_counter()
        function()
        jt.sync_all()
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return summarize_timings(elapsed)


def uncached_trace(
    model: NativeGPT2LMHeadModel,
    initial_embeddings: jt.Var,
    max_new_tokens: int,
) -> Tuple[List[int], List[np.ndarray]]:
    embeddings = initial_embeddings
    token_ids: List[int] = []
    logits_trace: List[np.ndarray] = []
    for _ in range(max_new_tokens):
        logits = model(inputs_embeds=embeddings)
        next_logits = logits[:, -1, :]
        next_ids, _ = next_logits.argmax(dim=-1)
        next_token = int(next_ids.numpy()[0])
        logits_trace.append(next_logits.numpy().copy())
        token_ids.append(next_token)
        token_var = jt.array(np.asarray([[next_token]], dtype=np.int32))
        token_var.stop_grad()
        embeddings = jt.concat(
            [embeddings, model.embed_tokens(token_var)],
            dim=1,
        )
    return token_ids, logits_trace


def cached_trace(
    model: NativeGPT2LMHeadModel,
    initial_embeddings: jt.Var,
    max_new_tokens: int,
) -> Tuple[List[int], List[np.ndarray], List[Tuple[jt.Var, jt.Var]]]:
    output = model.forward_with_cache(
        inputs_embeds=initial_embeddings,
        use_cache=True,
    )
    token_ids: List[int] = []
    logits_trace: List[np.ndarray] = []
    for step in range(max_new_tokens):
        next_logits = output["logits"][:, -1, :]
        next_ids, _ = next_logits.argmax(dim=-1)
        next_token = int(next_ids.numpy()[0])
        logits_trace.append(next_logits.numpy().copy())
        token_ids.append(next_token)
        if step + 1 >= max_new_tokens:
            break
        token_var = jt.array(np.asarray([[next_token]], dtype=np.int32))
        token_var.stop_grad()
        output = model.forward_with_cache(
            input_ids=token_var,
            past_key_values=output["past_key_values"],
            use_cache=True,
        )
    return token_ids, logits_trace, output["past_key_values"]


def compare_traces(
    uncached_ids: Sequence[int],
    uncached_logits: Sequence[np.ndarray],
    cached_ids: Sequence[int],
    cached_logits: Sequence[np.ndarray],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    if len(uncached_logits) != len(cached_logits):
        raise ValueError("cached and uncached logit trace lengths differ")
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    allclose = True
    step_results = []
    for step, (left, right) in enumerate(zip(uncached_logits, cached_logits)):
        difference = np.abs(
            left.astype(np.float64) - right.astype(np.float64)
        )
        step_maximum = float(np.max(difference))
        step_mean = float(np.mean(difference))
        step_allclose = bool(np.allclose(left, right, atol=atol, rtol=rtol))
        maximum = max(maximum, step_maximum)
        absolute_sum += float(np.sum(difference))
        element_count += int(difference.size)
        allclose = allclose and step_allclose
        step_results.append(
            {
                "step": step,
                "max_abs_error": step_maximum,
                "mean_abs_error": step_mean,
                "allclose": step_allclose,
                "uncached_token_id": int(uncached_ids[step]),
                "cached_token_id": int(cached_ids[step]),
            }
        )
    return {
        "token_ids_exact": list(uncached_ids) == list(cached_ids),
        "uncached_token_ids": [int(item) for item in uncached_ids],
        "cached_token_ids": [int(item) for item in cached_ids],
        "max_abs_error": maximum,
        "mean_abs_error": absolute_sum / element_count if element_count else 0.0,
        "allclose": allclose,
        "steps": step_results,
    }


def validate_cache(
    past_key_values: Sequence[Tuple[jt.Var, jt.Var]],
    model: NativeGPT2LMHeadModel,
    expected_tokens: int,
) -> Dict[str, Any]:
    layer_shapes = []
    exact = len(past_key_values) == model.config.n_layer
    expected_shape = (
        1,
        model.config.n_head,
        expected_tokens,
        model.config.n_embd // model.config.n_head,
    )
    for key, value in past_key_values:
        key_shape = tuple(int(item) for item in key.shape)
        value_shape = tuple(int(item) for item in value.shape)
        layer_shapes.append(
            {"key": list(key_shape), "value": list(value_shape)}
        )
        exact = exact and key_shape == expected_shape and value_shape == expected_shape
    return {
        "layer_count": len(past_key_values),
        "expected_layer_count": model.config.n_layer,
        "expected_shape": list(expected_shape),
        "layer_shapes": layer_shapes,
        "exact": bool(exact),
    }


def cached_decode_from_prefill(
    model: NativeGPT2LMHeadModel,
    past_key_values: Sequence[Tuple[jt.Var, jt.Var]],
    first_token: int,
    decode_steps: int,
) -> None:
    """Decode only post-prefill tokens from an immutable initial cache."""

    next_token = int(first_token)
    cache = list(past_key_values)
    for _ in range(decode_steps):
        token_var = jt.array(np.asarray([[next_token]], dtype=np.int32))
        token_var.stop_grad()
        output = model.forward_with_cache(
            input_ids=token_var,
            past_key_values=cache,
            use_cache=True,
        )
        cache = output["past_key_values"]
        next_ids, _ = output["logits"][:, -1, :].argmax(dim=-1)
        next_token = int(next_ids.numpy()[0])


def benchmark_sample(
    model: NativeGPT2LMHeadModel,
    initial_embeddings: jt.Var,
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    prefill = timed_runs(
        lambda: model.forward_with_cache(
            inputs_embeds=initial_embeddings,
            use_cache=True,
        ),
        warmup_runs,
        measured_runs,
    )
    prefill_output = model.forward_with_cache(
        inputs_embeds=initial_embeddings,
        use_cache=True,
    )
    prefill_next_ids, _ = prefill_output["logits"][:, -1, :].argmax(dim=-1)
    prefill_next_token = int(prefill_next_ids.numpy()[0])
    prefill_cache = prefill_output["past_key_values"]
    jt.sync_all()
    decode_steps = max_new_tokens - 1
    cached_decode = timed_runs(
        lambda: cached_decode_from_prefill(
            model,
            prefill_cache,
            prefill_next_token,
            decode_steps,
        ),
        warmup_runs,
        measured_runs,
    )
    cached_total = timed_runs(
        lambda: greedy_generate_from_embeddings_cached(
            model,
            initial_embeddings,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
        ),
        warmup_runs,
        measured_runs,
    )
    uncached_total = timed_runs(
        lambda: greedy_generate_from_embeddings(
            model,
            initial_embeddings,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
        ),
        warmup_runs,
        measured_runs,
    )
    cached_decode["decode_steps_per_run"] = decode_steps
    cached_decode["tokens_per_second"] = (
        decode_steps * 1000.0 / cached_decode["mean_ms"]
        if decode_steps
        else None
    )
    cached_total["generated_tokens_per_run"] = max_new_tokens
    cached_total["tokens_per_second"] = (
        max_new_tokens * 1000.0 / cached_total["mean_ms"]
    )
    uncached_total["generated_tokens_per_run"] = max_new_tokens
    uncached_total["tokens_per_second"] = (
        max_new_tokens * 1000.0 / uncached_total["mean_ms"]
    )
    return {
        "cached_prefill": prefill,
        "cached_decode_only": cached_decode,
        "cached_total_generation": cached_total,
        "uncached_total_generation": uncached_total,
        "uncached_over_cached_total_speedup": (
            uncached_total["mean_ms"] / cached_total["mean_ms"]
        ),
    }


def aggregate_budget_benchmark(sample_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fields = (
        "cached_prefill",
        "cached_decode_only",
        "cached_total_generation",
        "uncached_total_generation",
    )
    aggregate = {}
    for field in fields:
        values = []
        for result in sample_results:
            values.extend(result["benchmark"][field]["raw_ms"])
        aggregate[field] = summarize_timings(values)
    decode_steps = int(
        sample_results[0]["benchmark"]["cached_decode_only"]
        ["decode_steps_per_run"]
    )
    generated_tokens = int(
        sample_results[0]["benchmark"]["cached_total_generation"]
        ["generated_tokens_per_run"]
    )
    aggregate["cached_decode_only"]["decode_steps_per_run"] = decode_steps
    aggregate["cached_decode_only"]["tokens_per_second"] = (
        decode_steps * 1000.0 / aggregate["cached_decode_only"]["mean_ms"]
        if decode_steps
        else None
    )
    for field in ("cached_total_generation", "uncached_total_generation"):
        aggregate[field]["generated_tokens_per_run"] = generated_tokens
        aggregate[field]["tokens_per_second"] = (
            generated_tokens * 1000.0 / aggregate[field]["mean_ms"]
        )
    aggregate["uncached_over_cached_total_speedup"] = (
        aggregate["uncached_total_generation"]["mean_ms"]
        / aggregate["cached_total_generation"]["mean_ms"]
    )
    return aggregate


def run_budget(
    budget: int,
    reference_dir: Path,
    sample_indices: Sequence[int],
    model: NativeGPT2LMHeadModel,
    tokenizer,
    projector: MultimodalProjector,
    prompt: str,
    generation_prompt: str,
    config: Dict[str, Any],
    device: str,
    expected_image_names: Sequence[str] | None,
) -> Tuple[Dict[str, Any], List[str]]:
    path = reference_path(reference_dir, budget)
    visual_tokens, metadata, images = load_reference_samples(
        path,
        budget,
        sample_indices,
    )
    image_names = [str(item["name"]) for item in images]
    if expected_image_names is not None and image_names != list(expected_image_names):
        raise ValueError("Phase 2 reference image order differs across budgets")

    sampler = GpuMemorySampler(
        enabled=device == "cuda",
        interval=float(config["gpu_memory_sampling_interval_seconds"]),
    )
    sampler.start()
    sample_results = []
    try:
        for row, (sample_index, image, compressed) in enumerate(
            zip(sample_indices, images, visual_tokens)
        ):
            compressed_var = jt.array(compressed[None, :, :])
            compressed_var.stop_grad()
            projected = projector(compressed_var)
            projected.stop_grad()
            initial_embeddings = build_generation_embeddings(
                tokenizer,
                model,
                projected,
                prompt,
                generation_prompt,
            )
            initial_embeddings.stop_grad()
            uncached_ids, uncached_logits = uncached_trace(
                model,
                initial_embeddings,
                int(config["max_new_tokens"]),
            )
            cached_ids, cached_logits, cache = cached_trace(
                model,
                initial_embeddings,
                int(config["max_new_tokens"]),
            )
            comparison = compare_traces(
                uncached_ids,
                uncached_logits,
                cached_ids,
                cached_logits,
                float(config["atol"]),
                float(config["rtol"]),
            )
            expected_cache_tokens = (
                int(initial_embeddings.shape[1])
                + int(config["max_new_tokens"])
                - 1
            )
            cache_check = validate_cache(cache, model, expected_cache_tokens)
            benchmark = benchmark_sample(
                model,
                initial_embeddings,
                int(config["max_new_tokens"]),
                int(config["warmup_runs"]),
                int(config["measured_runs"]),
            )
            token_id_requirement_satisfied = bool(
                comparison["token_ids_exact"]
                or not bool(config["require_exact_token_ids"])
            )
            sample_passed = bool(
                comparison["allclose"]
                and token_id_requirement_satisfied
                and cache_check["exact"]
            )
            sample_results.append(
                {
                    "row": row,
                    "sample_index": int(sample_index),
                    "image": image,
                    "compressed_shape": list(compressed_var.shape),
                    "projected_shape": list(projected.shape),
                    "initial_embedding_shape": list(initial_embeddings.shape),
                    "generated_text": tokenizer.decode(
                        cached_ids,
                        skip_special_tokens=True,
                    ),
                    "correctness": comparison,
                    "token_id_requirement_satisfied": (
                        token_id_requirement_satisfied
                    ),
                    "cache": cache_check,
                    "benchmark": benchmark,
                    "passed": sample_passed,
                }
            )
            jt.gc()
    finally:
        sampler.stop()

    aggregate = aggregate_budget_benchmark(sample_results)
    speedup_ok = (
        not bool(config["require_speedup"])
        or aggregate["uncached_over_cached_total_speedup"] > 1.0
    )
    result = {
        "budget": budget,
        "reference": str(path),
        "reference_sha256": file_sha256(path),
        "reference_metadata": metadata,
        "sample_results": sample_results,
        "aggregate_benchmark": aggregate,
        "peak_process_gpu_memory_mib": sampler.peak_mib,
        "gpu_memory_sample_count": sampler.sample_count,
        "speedup_requirement_satisfied": speedup_ok,
        "passed": bool(
            sample_results
            and all(item["passed"] for item in sample_results)
            and speedup_ok
        ),
    }
    return result, image_names


def main():
    args = parse_args()
    config = load_benchmark_config(args.config)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0

    artifact_manifest_path = args.artifact_dir / "manifest.json"
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    if not artifact_manifest.get("real_llm"):
        raise ValueError("GPT-2 artifact manifest must declare real_llm=true")
    artifact_checksums = verify_gpt2_artifacts(
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
    print("Phase 5A: loading frozen native Jittor GPT-2")
    print("=" * 72)
    model = NativeGPT2LMHeadModel(gpt2_config)
    load_report = model.load_npz_weights(
        args.artifact_dir / "gpt2_float32_weights.npz"
    )
    model.freeze_parameters()
    model.eval()
    language_parameters = list(model.parameters())
    language_hash_before = parameter_sha256(language_parameters)

    phase4b_config = load_phase4b_config(args.phase4b_config)
    settings = phase4b_config.training
    if settings.model_name_or_path != artifact_manifest.get("model_name_or_path"):
        raise ValueError("Phase 4B and GPT-2 model identities differ")
    feature_manifest = load_feature_manifest(args.feature_manifest)
    feature_manifest_hash = file_sha256(args.feature_manifest)

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
    projector = MultimodalProjector(projector_config)
    projector_parameters = list(projector.parameters())
    optimizer = jt.optim.Adam(
        projector_parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    checkpoint_metadata = load_phase4_checkpoint(
        args.projector_checkpoint,
        projector,
        optimizer,
        expected_artifact_type=CHECKPOINT_ARTIFACT_TYPE,
    )
    projector.eval()
    for parameter in projector_parameters:
        parameter.stop_grad()
    projector_hash_before = parameter_sha256(projector_parameters)
    del optimizer

    checkpoint_checks = {
        "artifact_type_exact": (
            checkpoint_metadata.get("artifact_type") == CHECKPOINT_ARTIFACT_TYPE
        ),
        "completed_training_exact": (
            int(checkpoint_metadata.get("completed_optimizer_steps", -1))
            == settings.max_optimizer_steps
        ),
        "projector_hash_exact": (
            projector_hash_before
            == checkpoint_metadata.get("projector_parameter_sha256")
        ),
        "feature_manifest_hash_exact": (
            feature_manifest_hash
            == checkpoint_metadata.get("feature_manifest_sha256")
        ),
        "phase4b_config_hash_exact": (
            canonical_json_sha256(phase4b_config.to_dict())
            == checkpoint_metadata.get("phase4b_config_sha256")
        ),
        "feature_model_exact": (
            feature_manifest.model_name_or_path
            == checkpoint_metadata.get("feature_model_name_or_path")
        ),
        "feature_revision_exact": (
            feature_manifest.model_revision
            == checkpoint_metadata.get("feature_model_revision")
        ),
    }
    if not all(checkpoint_checks.values()):
        raise ValueError(
            f"Phase 4B checkpoint provenance failure: {checkpoint_checks}"
        )

    results = []
    expected_image_names = None
    with jt.no_grad():
        for budget in config["budgets"]:
            print("=" * 72)
            print(f"Phase 5A KV-cache correctness/benchmark: budget {budget}")
            print("=" * 72)
            result, image_names = run_budget(
                budget,
                args.reference_dir,
                config["sample_indices"],
                model,
                tokenizer,
                projector,
                settings.prompt,
                settings.generation_prompt,
                config,
                args.device,
                expected_image_names,
            )
            if expected_image_names is None:
                expected_image_names = image_names
            results.append(result)
            print(
                json.dumps(
                    {
                        "budget": budget,
                        "aggregate_benchmark": result["aggregate_benchmark"],
                        "peak_process_gpu_memory_mib": (
                            result["peak_process_gpu_memory_mib"]
                        ),
                        "passed": result["passed"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            jt.gc()

    language_hash_after = parameter_sha256(language_parameters)
    projector_hash_after = parameter_sha256(projector_parameters)
    language_unchanged = language_hash_before == language_hash_after
    projector_unchanged = projector_hash_before == projector_hash_after
    language_all_stop_grad = model.all_parameters_stop_grad()
    projector_all_stop_grad = bool(projector_parameters) and all(
        item.is_stop_grad() for item in projector_parameters
    )
    invariant_passed = bool(
        language_all_stop_grad
        and projector_all_stop_grad
        and (
            language_unchanged
            or not bool(config["require_language_unchanged"])
        )
        and (
            projector_unchanged
            or not bool(config["require_projector_unchanged"])
        )
    )
    report = {
        "artifact_type": "phase5a_native_jittor_gpt2_kv_cache_v1",
        "source_commit": source_commit(),
        "device": args.device,
        "jittor_version": jt.__version__,
        "gpu_name": (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            if args.device == "cuda"
            else None
        ),
        "config": config,
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "gpt2": {
            "real_llm": True,
            "model_name_or_path": artifact_manifest["model_name_or_path"],
            "architecture": artifact_manifest["architecture"],
            "parameter_count": gpt2_parameter_count(language_parameters),
            "load_report": load_report,
            "artifact_manifest": str(artifact_manifest_path),
            "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
            "artifact_checksums": artifact_checksums,
            "all_stop_grad": language_all_stop_grad,
            "sha256_before": language_hash_before,
            "sha256_after": language_hash_after,
            "unchanged": language_unchanged,
        },
        "projector": {
            "checkpoint": str(args.projector_checkpoint),
            "checkpoint_sha256": file_sha256(args.projector_checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
            "checkpoint_checks": checkpoint_checks,
            "parameter_count": parameter_count(projector),
            "all_stop_grad": projector_all_stop_grad,
            "sha256_before": projector_hash_before,
            "sha256_after": projector_hash_after,
            "unchanged": projector_unchanged,
        },
        "phase4b_feature_manifest": {
            "path": str(args.feature_manifest),
            "sha256": feature_manifest_hash,
            "sample_count": feature_manifest.sample_count,
            "token_shape": list(feature_manifest.token_shape),
            "shard_count": len(feature_manifest.shards),
        },
        "image_names": expected_image_names,
        "results": results,
        "invariants_passed": invariant_passed,
        "claim_boundary": (
            "Phase 5A validates cached decoding correctness and reports runtime "
            "measurements; it does not claim caption-quality improvement."
        ),
        "passed": bool(
            invariant_passed
            and all(checkpoint_checks.values())
            and results
            and all(item["passed"] for item in results)
        ),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("Phase 5A summary")
    print("=" * 72)
    print(text)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
