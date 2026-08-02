"""Run Phase 3A Projector + frozen-language-stub forward/backward smoke tests.

This runner deliberately uses a small frozen embedding/output-head surrogate,
not a real LLM. It validates token plumbing, parameter freezing, gradient
isolation, and one Projector optimizer step for the 64/128/192 real-CLIP
artifacts produced in Phase 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jittor as jt

from visionzip_jittor.config import VisionZipConfig
from visionzip_jittor.core import visionzip_compress
from visionzip_jittor.multimodal import (
    FrozenLanguageStub,
    ProjectorFrozenLanguageBridge,
)
from visionzip_jittor.projector import MultimodalProjector, parameter_count
from visionzip_jittor.projector_config import load_projector_config


DEFAULT_BUDGETS = (64, 128, 192)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projector-config",
        type=Path,
        default=ROOT / "configs/phase3_projector_smoke.json",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "outputs/real_clip",
    )
    parser.add_argument(
        "--budget",
        action="append",
        type=int,
        dest="budgets",
        help="Nominal visual token budget; repeat for multiple budgets",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "logs/phase3/projector_smoke.json",
    )
    return parser.parse_args()


def reference_path(reference_dir: Path, budget: int) -> Path:
    return reference_dir / (
        f"reference_clip_{budget}_code_exact_float32_real_clip.npz"
    )


def scalar_parameter_count(parameters: Iterable[jt.Var]) -> int:
    total = 0
    for parameter in parameters:
        size = 1
        for dimension in parameter.shape:
            size *= int(dimension)
        total += size
    return total


def snapshot(parameters: Iterable[jt.Var]) -> List[np.ndarray]:
    return [parameter.numpy().copy() for parameter in parameters]


def max_parameter_delta(
    before: Iterable[np.ndarray],
    after: Iterable[jt.Var],
) -> float:
    maximum = 0.0
    for old, new in zip(before, after):
        delta = np.max(
            np.abs(old.astype(np.float64) - new.numpy().astype(np.float64))
        )
        maximum = max(maximum, float(delta))
    return maximum


def gradient_statistics(
    parameters: Iterable[jt.Var],
    optimizer,
) -> Dict[str, float | bool]:
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


def deterministic_token_ids(
    batch_size: int,
    token_count: int,
    vocab_size: int,
    offset: int,
) -> jt.Var:
    values = (
        np.arange(batch_size * token_count, dtype=np.int32).reshape(
            batch_size, token_count
        )
        + offset
    ) % vocab_size
    return jt.array(values)


def run_budget(
    budget: int,
    path: Path,
    projector_config,
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 2 reference for budget {budget}: {path}"
        )

    with np.load(path, allow_pickle=False) as reference:
        metadata = json.loads(str(reference["metadata_json"].item()))
        visionzip_config = VisionZipConfig.from_dict(metadata["config"])
        hidden_np = reference["hidden_states"].astype(np.float32)
        attention_np = reference["attentions"].astype(np.float32)
        metric_np = reference["metric"].astype(np.float32)
        compressed_reference = reference["compressed_tokens"].astype(np.float32)
        assignments_reference = reference["assignments"].astype(np.int64)

    if visionzip_config.nominal_visual_tokens != budget:
        raise ValueError(
            f"Reference budget mismatch: filename={budget}, "
            f"config={visionzip_config.nominal_visual_tokens}"
        )
    if hidden_np.shape[-1] != projector_config.vision_hidden_size:
        raise ValueError(
            "Projector vision_hidden_size does not match the CLIP reference"
        )

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
    compressed_max_abs_error = float(
        np.max(
            np.abs(
                compressed_np.astype(np.float64)
                - compressed_reference.astype(np.float64)
            )
        )
    )
    compressed_allclose = bool(
        np.allclose(
            compressed_np,
            compressed_reference,
            atol=atol,
            rtol=rtol,
        )
    )
    assignments_exact = bool(
        np.array_equal(assignments_np, assignments_reference)
    )

    jt.set_global_seed(projector_config.seed + budget)
    projector = MultimodalProjector(projector_config)
    language_stub = FrozenLanguageStub(projector_config)
    bridge = ProjectorFrozenLanguageBridge(projector, language_stub)
    projector_parameters = list(projector.parameters())
    frozen_parameters = list(language_stub.parameters())
    optimizer = jt.optim.Adam(
        projector_parameters,
        lr=projector_config.learning_rate,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    batch_size = int(compressed.shape[0])
    visual_tokens = int(compressed.shape[1])
    prefix_ids = deterministic_token_ids(
        batch_size,
        projector_config.prefix_tokens,
        projector_config.vocab_size,
        offset=budget,
    )
    suffix_ids = deterministic_token_ids(
        batch_size,
        projector_config.suffix_tokens,
        projector_config.vocab_size,
        offset=budget + 17,
    )
    prefix_ids.stop_grad()
    suffix_ids.stop_grad()

    # Compile the forward graph before timing the checked optimizer step.
    warmup = bridge(compressed, prefix_ids, suffix_ids)
    warmup["logits"].sync()
    del warmup
    jt.gc()

    projector_before = snapshot(projector_parameters)
    frozen_before = snapshot(frozen_parameters)

    forward_start = time.perf_counter()
    outputs = bridge(compressed, prefix_ids, suffix_ids)
    visual_logits = outputs["logits"][
        :,
        projector_config.prefix_tokens : (
            projector_config.prefix_tokens + visual_tokens
        ),
        :,
    ]
    loss = (visual_logits * visual_logits).mean()
    loss.sync()
    forward_ms = (time.perf_counter() - forward_start) * 1000.0
    loss_value = float(loss.numpy().item())

    optimizer.zero_grad()
    backward_start = time.perf_counter()
    optimizer.backward(loss)
    jt.sync_all()
    backward_ms = (time.perf_counter() - backward_start) * 1000.0
    gradients = gradient_statistics(projector_parameters, optimizer)

    optimizer.step()
    jt.sync_all()
    projector_delta = max_parameter_delta(
        projector_before,
        projector_parameters,
    )
    frozen_delta = max_parameter_delta(frozen_before, frozen_parameters)

    expected_visual_tokens = visionzip_config.actual_output_tokens
    expected_packed_tokens = (
        projector_config.prefix_tokens
        + expected_visual_tokens
        + projector_config.suffix_tokens
    )
    expected_shapes = {
        "visionzip_output": [
            batch_size,
            expected_visual_tokens,
            projector_config.vision_hidden_size,
        ],
        "projector_output": [
            batch_size,
            expected_visual_tokens,
            projector_config.language_hidden_size,
        ],
        "packed_embeddings": [
            batch_size,
            expected_packed_tokens,
            projector_config.language_hidden_size,
        ],
        "logits": [
            batch_size,
            expected_packed_tokens,
            projector_config.vocab_size,
        ],
    }
    actual_shapes = {
        "visionzip_output": list(compressed.shape),
        "projector_output": list(outputs["projected_visual"].shape),
        "packed_embeddings": list(outputs["packed_embeddings"].shape),
        "logits": list(outputs["logits"].shape),
    }
    shape_checks = {
        name: actual_shapes[name] == expected
        for name, expected in expected_shapes.items()
    }
    inputs_stop_grad = all(
        tensor.is_stop_grad()
        for tensor in (hidden, attentions, metric, prefix_ids, suffix_ids)
    )
    frozen_all_stop_grad = language_stub.all_parameters_stop_grad()
    projector_all_trainable = all(
        not parameter.is_stop_grad() for parameter in projector_parameters
    )
    optimizer_scope_exact = (
        len(optimizer_parameters) == len(projector_parameters)
        and {id(parameter) for parameter in optimizer_parameters}
        == {id(parameter) for parameter in projector_parameters}
    )
    projector_changed = projector_delta > 0.0
    frozen_changed = frozen_delta > 0.0
    gradient_ok = bool(
        gradients["finite"]
        and gradients["l2_norm"] > 0.0
        and gradients["parameter_tensors_with_grad"] == len(projector_parameters)
    )
    passed = bool(
        all(shape_checks.values())
        and compressed_allclose
        and assignments_exact
        and inputs_stop_grad
        and frozen_all_stop_grad
        and projector_all_trainable
        and optimizer_scope_exact
        and gradient_ok
        and projector_changed
        and not frozen_changed
        and math.isfinite(loss_value)
    )

    return {
        "budget": budget,
        "reference": str(path),
        "visionzip_config": visionzip_config.to_dict(),
        "shapes": actual_shapes,
        "expected_shapes": expected_shapes,
        "shape_checks": shape_checks,
        "phase2_regression": {
            "compressed_max_abs_error": compressed_max_abs_error,
            "compressed_allclose": compressed_allclose,
            "assignments_exact": assignments_exact,
        },
        "loss": loss_value,
        "projector_parameter_count": parameter_count(projector),
        "optimizer_parameter_count": scalar_parameter_count(
            optimizer_parameters
        ),
        "optimizer_scope_exact": optimizer_scope_exact,
        "projector_all_trainable": projector_all_trainable,
        "gradient": gradients,
        "projector_max_parameter_delta": projector_delta,
        "projector_changed": projector_changed,
        "frozen_language_parameter_count": scalar_parameter_count(
            frozen_parameters
        ),
        "frozen_language_all_stop_grad": frozen_all_stop_grad,
        "frozen_language_max_parameter_delta": frozen_delta,
        "frozen_language_changed": frozen_changed,
        "reference_inputs_all_stop_grad": inputs_stop_grad,
        "timing_ms": {
            "forward": forward_ms,
            "backward": backward_ms,
        },
        "passed": passed,
    }


def main():
    args = parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    projector_config = load_projector_config(args.projector_config)
    budgets = tuple(args.budgets or DEFAULT_BUDGETS)

    results = []
    for budget in budgets:
        print("=" * 72)
        print(f"Phase 3A Projector smoke: budget {budget}")
        print("=" * 72)
        result = run_budget(
            budget,
            reference_path(args.reference_dir, budget),
            projector_config,
            args.atol,
            args.rtol,
        )
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        jt.gc()

    report = {
        "artifact_type": "phase3_projector_frozen_language_stub_smoke_v1",
        "real_llm": False,
        "scope": (
            "Native Jittor VisionZip + trainable Projector + frozen "
            "embedding/output-head surrogate"
        ),
        "projector_config": projector_config.to_dict(),
        "device": args.device,
        "jittor_version": jt.__version__,
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("Phase 3A summary")
    print("=" * 72)
    print(text)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
