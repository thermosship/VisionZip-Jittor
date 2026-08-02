#!/usr/bin/env python3
"""Run Phase 3B with native Jittor VisionZip, Projector and real GPT-2."""

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
import threading
import time
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import jittor as jt
import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionzip_jittor.config import VisionZipConfig
from visionzip_jittor.core import visionzip_compress
from visionzip_jittor.gpt2 import (
    NativeGPT2LMHeadModel,
    greedy_generate_from_embeddings,
    masked_causal_language_loss,
    parameter_count,
)
from visionzip_jittor.gpt2_config import (
    load_gpt2_config,
    load_phase3b_config,
)
from visionzip_jittor.multimodal import pack_multimodal_embeddings
from visionzip_jittor.projector import MultimodalProjector
from visionzip_jittor.projector_config import ProjectorConfig


DEFAULT_BUDGETS = (64, 128, 192)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase3b_gpt2.json",
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
    parser.add_argument("--budgets", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "logs/phase3b/gpt2_smoke.json",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def reference_path(reference_dir: Path, budget: int) -> Path:
    return reference_dir / (
        f"reference_clip_{budget}_code_exact_float32_real_clip.npz"
    )


def snapshot(parameters: Iterable[jt.Var]) -> List[np.ndarray]:
    return [parameter.numpy().copy() for parameter in parameters]


def max_parameter_delta(before, after) -> float:
    maximum = 0.0
    for old, new in zip(before, after):
        value = new.numpy()
        if value.size:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            old.astype(np.float64)
                            - value.astype(np.float64)
                        )
                    )
                ),
            )
    return maximum


def parameter_sha256(parameters: Iterable[jt.Var]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        array = np.ascontiguousarray(parameter.numpy())
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative names and contents for a deterministic artifact tree."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_artifact_files(artifact_dir: Path, manifest):
    checks = {}
    for label in ("weights", "config", "tokenizer", "reference"):
        relative = manifest["files"][label]
        path = artifact_dir / relative
        expected = manifest["sha256"][label]
        actual = directory_sha256(path) if path.is_dir() else file_sha256(path)
        checks[label] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "exact": actual == expected,
        }
    if not all(item["exact"] for item in checks.values()):
        raise ValueError(f"Phase 3B artifact checksum failure: {checks}")
    return checks


def gradient_statistics(parameters: Iterable[jt.Var], optimizer):
    squared_norm = 0.0
    maximum = 0.0
    finite = True
    found = 0
    for parameter in parameters:
        gradient_var = parameter.opt_grad(optimizer)
        if gradient_var is None:
            continue
        gradient = gradient_var.numpy().astype(np.float64)
        found += 1
        finite = finite and bool(np.isfinite(gradient).all())
        if gradient.size:
            maximum = max(maximum, float(np.max(np.abs(gradient))))
        squared_norm += float(np.sum(gradient * gradient))
    return {
        "parameter_tensors_with_grad": found,
        "l2_norm": math.sqrt(squared_norm),
        "max_abs": maximum,
        "finite": finite,
    }


class GpuMemorySampler:
    """Sample this process's nvidia-smi memory without an extra dependency."""

    def __init__(self, enabled: bool, interval: float = 0.1):
        self.enabled = enabled
        self.interval = interval
        self.peak_mib: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return
        own_pid = os.getpid()
        for line in result.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                pid, memory = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if pid == own_pid:
                self.peak_mib = max(self.peak_mib or 0, memory)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.enabled:
            self._sample()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
            self._sample()


def repeat_ids(ids: List[int], batch_size: int) -> np.ndarray:
    return np.repeat(np.asarray([ids], dtype=np.int32), batch_size, axis=0)


def build_training_batch(tokenizer, model, projected_visual, phase3b_config):
    batch_size = int(projected_visual.shape[0])
    if len(phase3b_config.targets) != batch_size:
        raise ValueError(
            "Phase 3B target count must equal the real-CLIP batch size: "
            f"{len(phase3b_config.targets)} != {batch_size}"
        )
    prompt_ids_list = tokenizer.encode(
        phase3b_config.prompt,
        add_special_tokens=False,
    )
    target_lists = [
        tokenizer.encode(text, add_special_tokens=False)
        + [int(tokenizer.eos_token_id)]
        for text in phase3b_config.targets
    ]
    maximum_target = max(len(item) for item in target_lists)
    pad_id = int(tokenizer.eos_token_id)
    target_ids_np = np.full(
        (batch_size, maximum_target),
        pad_id,
        dtype=np.int32,
    )
    target_mask_np = np.zeros(
        (batch_size, maximum_target),
        dtype=np.float32,
    )
    for index, target in enumerate(target_lists):
        target_ids_np[index, : len(target)] = target
        target_mask_np[index, : len(target)] = 1.0

    prompt_ids = jt.array(repeat_ids(prompt_ids_list, batch_size))
    target_ids = jt.array(target_ids_np)
    prompt_ids.stop_grad()
    target_ids.stop_grad()
    prompt_embeddings = model.embed_tokens(prompt_ids)
    target_embeddings = model.embed_tokens(target_ids)
    packed = pack_multimodal_embeddings(
        prompt_embeddings,
        projected_visual,
        target_embeddings,
    )

    prompt_tokens = len(prompt_ids_list)
    visual_tokens = int(projected_visual.shape[1])
    total_tokens = int(packed.shape[1])
    labels_np = np.zeros((batch_size, total_tokens), dtype=np.int32)
    label_mask_np = np.zeros((batch_size, total_tokens), dtype=np.float32)
    target_start = prompt_tokens + visual_tokens
    labels_np[:, target_start:] = target_ids_np
    label_mask_np[:, target_start:] = target_mask_np
    attention_mask_np = np.ones((batch_size, total_tokens), dtype=np.float32)
    attention_mask_np[:, target_start:] = target_mask_np

    labels = jt.array(labels_np)
    label_mask = jt.array(label_mask_np)
    attention_mask = jt.array(attention_mask_np)
    for tensor in (labels, label_mask, attention_mask):
        tensor.stop_grad()
    return {
        "packed_embeddings": packed,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_mask": label_mask,
        "prompt_tokens": prompt_tokens,
        "target_tokens": maximum_target,
        "target_token_counts": [len(item) for item in target_lists],
    }


def benchmark_prefill(model, batch, warmup: int, iterations: int):
    for _ in range(warmup):
        logits = model(
            inputs_embeds=batch["packed_embeddings"],
            attention_mask=batch["attention_mask"],
        )
        logits.sync()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        logits = model(
            inputs_embeds=batch["packed_embeddings"],
            attention_mask=batch["attention_mask"],
        )
        logits.sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def generate_samples(tokenizer, model, projected_visual, phase3b_config):
    prefix = tokenizer.encode(phase3b_config.prompt, add_special_tokens=False)
    suffix = tokenizer.encode(
        phase3b_config.generation_prompt,
        add_special_tokens=False,
    )
    results = []
    with jt.no_grad():
        for index in range(int(projected_visual.shape[0])):
            prefix_var = jt.array(np.asarray([prefix], dtype=np.int32))
            suffix_var = jt.array(np.asarray([suffix], dtype=np.int32))
            initial = pack_multimodal_embeddings(
                model.embed_tokens(prefix_var),
                projected_visual[index : index + 1],
                model.embed_tokens(suffix_var),
            )
            generated_ids = greedy_generate_from_embeddings(
                model,
                initial,
                phase3b_config.max_new_tokens,
                eos_token_id=int(tokenizer.eos_token_id),
            )
            results.append(
                {
                    "sample_index": index,
                    "token_ids": generated_ids,
                    "text": tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ),
                }
            )
    return results


def validate_text_reference(model, artifact_dir, atol: float, rtol: float):
    path = artifact_dir / "text_reference.npz"
    with np.load(path, allow_pickle=False) as reference:
        input_ids_np = reference["input_ids"].astype(np.int32)
        attention_mask_np = reference["attention_mask"].astype(np.float32)
        expected = reference["logits"].astype(np.float32)
        prompt = str(reference["prompt"].item())
    input_ids = jt.array(input_ids_np)
    attention_mask = jt.array(attention_mask_np)
    with jt.no_grad():
        actual_var = model(input_ids=input_ids, attention_mask=attention_mask)
        actual_var.sync()
        actual = actual_var.numpy()
    maximum = float(
        np.max(
            np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        )
    )
    return {
        "prompt": prompt,
        "shape": list(actual.shape),
        "max_abs_error": maximum,
        "allclose": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def run_budget(
    budget,
    path,
    model,
    tokenizer,
    phase3b_config,
    atol,
    rtol,
    device,
):
    with np.load(path, allow_pickle=False) as reference:
        metadata = json.loads(str(reference["metadata_json"].item()))
        visionzip_config = VisionZipConfig.from_dict(metadata["config"])
        hidden_np = reference["hidden_states"].astype(np.float32)
        attention_np = reference["attentions"].astype(np.float32)
        metric_np = reference["metric"].astype(np.float32)
        compressed_reference = reference["compressed_tokens"].astype(np.float32)
        assignments_reference = reference["assignments"].astype(np.int64)

    hidden = jt.array(hidden_np)
    attentions = jt.array(attention_np)
    metric = jt.array(metric_np)
    for tensor in (hidden, attentions, metric):
        tensor.stop_grad()
    visionzip_output = visionzip_compress(
        hidden,
        attentions,
        metric,
        visionzip_config,
    )
    compressed = visionzip_output["compressed_tokens"]
    compressed.sync()
    compressed_np = compressed.numpy()
    assignments_np = visionzip_output["assignments"].numpy().astype(np.int64)
    compressed_error = float(
        np.max(
            np.abs(
                compressed_np.astype(np.float64)
                - compressed_reference.astype(np.float64)
            )
        )
    )
    compressed_allclose = bool(
        np.allclose(compressed_np, compressed_reference, atol=atol, rtol=rtol)
    )
    assignments_exact = bool(
        np.array_equal(assignments_np, assignments_reference)
    )

    projector_config = ProjectorConfig(
        projector_type=phase3b_config.projector_type,
        vision_hidden_size=phase3b_config.vision_hidden_size,
        language_hidden_size=model.config.n_embd,
        vocab_size=model.config.vocab_size,
        prefix_tokens=0,
        suffix_tokens=0,
        learning_rate=phase3b_config.learning_rate,
        seed=phase3b_config.seed,
    )
    jt.set_global_seed(phase3b_config.seed + budget)
    projector = MultimodalProjector(projector_config)
    projector_parameters = list(projector.parameters())
    optimizer = jt.optim.Adam(
        projector_parameters,
        lr=phase3b_config.learning_rate,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    before = snapshot(projector_parameters)

    with GpuMemorySampler(device == "cuda") as memory:
        projected = projector(compressed)
        batch = build_training_batch(
            tokenizer,
            model,
            projected,
            phase3b_config,
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
        backward_start = time.perf_counter()
        optimizer.backward(loss)
        jt.sync_all()
        backward_ms = (time.perf_counter() - backward_start) * 1000.0
        gradients = gradient_statistics(projector_parameters, optimizer)
        optimizer.step()
        jt.sync_all()
        projector_delta = max_parameter_delta(before, projector_parameters)

        with jt.no_grad():
            projected_after = projector(compressed)
            benchmark_batch = build_training_batch(
                tokenizer,
                model,
                projected_after,
                phase3b_config,
            )
            prefill = benchmark_prefill(
                model,
                benchmark_batch,
                phase3b_config.warmup,
                phase3b_config.iterations,
            )
            generations = generate_samples(
                tokenizer,
                model,
                projected_after,
                phase3b_config,
            )

    optimizer_scope_exact = (
        len(optimizer_parameters) == len(projector_parameters)
        and {id(item) for item in optimizer_parameters}
        == {id(item) for item in projector_parameters}
    )
    gradient_ok = bool(
        gradients["finite"]
        and gradients["l2_norm"] > 0.0
        and gradients["parameter_tensors_with_grad"]
        == len(projector_parameters)
    )
    projector_changed = projector_delta > 0.0
    generated_nonempty = all(item["token_ids"] for item in generations)
    expected_visual_tokens = budget + 1
    shape_checks = {
        "visionzip_output": list(compressed.shape)
        == [int(compressed.shape[0]), expected_visual_tokens, 1024],
        "projector_output": list(projected.shape)
        == [int(projected.shape[0]), expected_visual_tokens, model.config.n_embd],
        "logits": int(logits.shape[0]) == int(compressed.shape[0])
        and int(logits.shape[2]) == model.config.vocab_size,
    }
    passed = bool(
        all(shape_checks.values())
        and compressed_allclose
        and assignments_exact
        and optimizer_scope_exact
        and all(not item.is_stop_grad() for item in projector_parameters)
        and gradient_ok
        and projector_changed
        and math.isfinite(loss_value)
        and generated_nonempty
    )
    return {
        "budget": budget,
        "reference": str(path),
        "shapes": {
            "visionzip_output": list(compressed.shape),
            "projector_output": list(projected.shape),
            "packed_embeddings": list(batch["packed_embeddings"].shape),
            "logits": list(logits.shape),
        },
        "shape_checks": shape_checks,
        "phase2_regression": {
            "compressed_max_abs_error": compressed_error,
            "compressed_allclose": compressed_allclose,
            "assignments_exact": assignments_exact,
        },
        "loss": loss_value,
        "projector_parameter_count": parameter_count(projector_parameters),
        "optimizer_scope_exact": optimizer_scope_exact,
        "projector_all_trainable": all(
            not item.is_stop_grad() for item in projector_parameters
        ),
        "gradient": gradients,
        "projector_max_parameter_delta": projector_delta,
        "projector_changed": projector_changed,
        "target_token_counts": batch["target_token_counts"],
        "generated": generations,
        "prefill_benchmark": prefill,
        "backward_ms": backward_ms,
        "peak_process_gpu_memory_mib": memory.peak_mib,
        "passed": passed,
    }


def main():
    args = parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    phase3b_config = load_phase3b_config(args.config)
    manifest = json.loads(
        (args.artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("real_llm"):
        raise ValueError("Phase 3B artifact manifest must declare real_llm=true")
    artifact_checksums = verify_artifact_files(args.artifact_dir, manifest)
    gpt2_config = load_gpt2_config(args.artifact_dir / "hf_config.json")
    if gpt2_config.n_embd <= 0:
        raise ValueError("Invalid GPT-2 hidden size")

    tokenizer = AutoTokenizer.from_pretrained(
        args.artifact_dir / "tokenizer",
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 72)
    print("Loading native Jittor GPT-2")
    print("=" * 72)
    model = NativeGPT2LMHeadModel(gpt2_config)
    load_report = model.load_npz_weights(
        args.artifact_dir / "gpt2_float32_weights.npz"
    )
    model.freeze_parameters()
    model.eval()
    frozen_parameters = list(model.parameters())
    language_parameter_count = parameter_count(frozen_parameters)
    artifact_model_checks = {
        "architecture_is_gpt2_lm_head": manifest.get("architecture")
        == "GPT2LMHeadModel",
        "configured_model_matches_manifest": (
            phase3b_config.model_name_or_path == manifest.get("model_name_or_path")
        ),
        "loaded_tensor_count_matches_manifest": (
            load_report["tensor_count"] == manifest.get("exported_tensor_count")
        ),
        "parameter_count_matches_manifest": (
            language_parameter_count == manifest.get("parameter_count")
        ),
    }
    if not all(artifact_model_checks.values()):
        raise ValueError(
            f"GPT-2 artifact/model integrity failure: {artifact_model_checks}"
        )
    frozen_hash_before = parameter_sha256(frozen_parameters)
    text_alignment = validate_text_reference(
        model,
        args.artifact_dir,
        phase3b_config.logit_atol,
        phase3b_config.logit_rtol,
    )
    print(json.dumps(text_alignment, indent=2, ensure_ascii=False))

    results = []
    for budget in tuple(args.budgets or DEFAULT_BUDGETS):
        print("=" * 72)
        print(f"Phase 3B real GPT-2 smoke: budget {budget}")
        print("=" * 72)
        result = run_budget(
            budget,
            reference_path(args.reference_dir, budget),
            model,
            tokenizer,
            phase3b_config,
            args.atol,
            args.rtol,
            args.device,
        )
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        jt.gc()

    frozen_hash_after = parameter_sha256(frozen_parameters)
    frozen_unchanged = frozen_hash_before == frozen_hash_after
    report = {
        "artifact_type": "phase3b_native_jittor_real_gpt2_smoke_v1",
        "real_llm": True,
        "model_name_or_path": manifest["model_name_or_path"],
        "architecture": manifest["architecture"],
        "device": args.device,
        "jittor_version": jt.__version__,
        "gpt2_config": gpt2_config.to_dict(),
        "phase3b_config": phase3b_config.to_dict(),
        "artifact_checksums": artifact_checksums,
        "weight_loading": load_report,
        "artifact_model_checks": artifact_model_checks,
        "language_parameter_count": language_parameter_count,
        "language_all_stop_grad": model.all_parameters_stop_grad(),
        "language_sha256_before": frozen_hash_before,
        "language_sha256_after": frozen_hash_after,
        "language_unchanged": frozen_unchanged,
        "text_logit_alignment": text_alignment,
        "results": results,
        "passed": bool(
            all(artifact_model_checks.values())
            and text_alignment["allclose"]
            and model.all_parameters_stop_grad()
            and frozen_unchanged
            and all(item["passed"] for item in results)
        ),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("Phase 3B summary")
    print("=" * 72)
    print(text)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
