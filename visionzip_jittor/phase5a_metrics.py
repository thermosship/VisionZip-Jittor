"""Pure NumPy correctness metrics for Phase 5A cached decoding."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def centered_logits(logits: np.ndarray) -> np.ndarray:
    """Remove the per-row vocabulary mean using float64 arithmetic."""

    values = np.asarray(logits, dtype=np.float64)
    return values - np.mean(values, axis=-1, keepdims=True)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a stable per-row softmax using float64 accumulation."""

    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _finite_arrays(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError(
            "cached and uncached trace value shapes differ: "
            f"{left_values.shape} != {right_values.shape}"
        )
    if not np.all(np.isfinite(left_values)) or not np.all(
        np.isfinite(right_values)
    ):
        raise ValueError("cached and uncached traces must contain finite values")
    return left_values, right_values


def _compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    left_values, right_values = _finite_arrays(left, right)
    difference = np.abs(left_values - right_values)
    return {
        "shape": [int(item) for item in left_values.shape],
        "max_abs_error": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs_error": float(np.mean(difference)) if difference.size else 0.0,
        "allclose": bool(
            np.allclose(left_values, right_values, atol=atol, rtol=rtol)
        ),
    }


def _top1_margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[-1] < 2:
        return np.full(probabilities.shape[:-1], np.inf, dtype=np.float64)
    top_two = np.partition(probabilities, -2, axis=-1)[..., -2:]
    return np.max(top_two, axis=-1) - np.min(top_two, axis=-1)


def _probability_distribution_metrics(
    uncached_probabilities: np.ndarray,
    cached_probabilities: np.ndarray,
    *,
    max_total_variation_distance: float,
) -> Dict[str, Any]:
    uncached, cached = _finite_arrays(
        uncached_probabilities,
        cached_probabilities,
    )
    difference = np.abs(uncached - cached)
    total_variation = 0.5 * np.sum(difference, axis=-1)
    uncached_argmax = np.argmax(uncached, axis=-1)
    cached_argmax = np.argmax(cached, axis=-1)
    uncached_top1 = np.take_along_axis(
        uncached,
        uncached_argmax[..., None],
        axis=-1,
    )[..., 0]
    cached_at_uncached_top1 = np.take_along_axis(
        cached,
        uncached_argmax[..., None],
        axis=-1,
    )[..., 0]
    top1_probability_error = np.abs(
        uncached_top1 - cached_at_uncached_top1
    )
    uncached_margin = _top1_margin(uncached)
    cached_margin = _top1_margin(cached)
    within_bound = total_variation <= max_total_variation_distance
    return {
        "shape": [int(item) for item in uncached.shape],
        "total_variation_threshold": float(max_total_variation_distance),
        "max_total_variation_distance": (
            float(np.max(total_variation)) if total_variation.size else 0.0
        ),
        "mean_total_variation_distance": (
            float(np.mean(total_variation)) if total_variation.size else 0.0
        ),
        "within_total_variation_bound": bool(np.all(within_bound)),
        "max_top1_probability_abs_error": (
            float(np.max(top1_probability_error))
            if top1_probability_error.size
            else 0.0
        ),
        "min_uncached_top1_margin": (
            float(np.min(uncached_margin)) if uncached_margin.size else 0.0
        ),
        "min_cached_top1_margin": (
            float(np.min(cached_margin)) if cached_margin.size else 0.0
        ),
        "argmax_ids_exact": bool(
            np.array_equal(uncached_argmax, cached_argmax)
        ),
    }


def _aggregate_step_metrics(
    steps: Sequence[Dict[str, Any]],
    metric_name: str,
    *,
    atol: float,
    rtol: float,
    role: str,
) -> Dict[str, Any]:
    reports = [item[metric_name] for item in steps]
    element_counts = [int(np.prod(item["shape"])) for item in reports]
    total_elements = sum(element_counts)
    weighted_absolute_sum = sum(
        report["mean_abs_error"] * count
        for report, count in zip(reports, element_counts)
    )
    return {
        "role": role,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": max(
            (float(item["max_abs_error"]) for item in reports),
            default=0.0,
        ),
        "mean_abs_error": (
            weighted_absolute_sum / total_elements if total_elements else 0.0
        ),
        "allclose": all(bool(item["allclose"]) for item in reports),
        "failed_steps": [
            int(step["step"])
            for step in steps
            if not bool(step[metric_name]["allclose"])
        ],
    }


def _aggregate_distribution_metrics(
    steps: Sequence[Dict[str, Any]],
    *,
    max_total_variation_distance: float,
) -> Dict[str, Any]:
    reports = [item["probability_distribution"] for item in steps]
    return {
        "role": "acceptance",
        "metric": "per_step_total_variation_distance",
        "threshold": float(max_total_variation_distance),
        "max_total_variation_distance": max(
            (
                float(item["max_total_variation_distance"])
                for item in reports
            ),
            default=0.0,
        ),
        "mean_total_variation_distance": (
            float(
                np.mean(
                    [
                        float(item["mean_total_variation_distance"])
                        for item in reports
                    ]
                )
            )
            if reports
            else 0.0
        ),
        "within_bound": all(
            bool(item["within_total_variation_bound"])
            for item in reports
        ),
        "failed_steps": [
            int(step["step"])
            for step in steps
            if not bool(
                step["probability_distribution"][
                    "within_total_variation_bound"
                ]
            )
        ],
        "max_top1_probability_abs_error": max(
            (
                float(item["max_top1_probability_abs_error"])
                for item in reports
            ),
            default=0.0,
        ),
        "min_uncached_top1_margin": min(
            (float(item["min_uncached_top1_margin"]) for item in reports),
            default=0.0,
        ),
        "min_cached_top1_margin": min(
            (float(item["min_cached_top1_margin"]) for item in reports),
            default=0.0,
        ),
        "argmax_ids_exact": all(
            bool(item["argmax_ids_exact"]) for item in reports
        ),
    }


def compare_generation_traces(
    uncached_ids: Sequence[int],
    uncached_logits: Sequence[np.ndarray],
    cached_ids: Sequence[int],
    cached_logits: Sequence[np.ndarray],
    *,
    raw_atol: float,
    raw_rtol: float,
    centered_atol: float,
    centered_rtol: float,
    probability_atol: float,
    probability_rtol: float,
    max_total_variation_distance: float,
) -> Dict[str, Any]:
    """Compare cached and full-recompute generation without hiding drift.

    Raw logits, centered logits, and coordinatewise probability ``allclose``
    remain diagnostics. The versioned probability-distribution acceptance
    metric is the maximum per-step total variation distance.
    """

    if len(uncached_ids) != len(cached_ids):
        raise ValueError("cached and uncached token trace lengths differ")
    if len(uncached_logits) != len(cached_logits):
        raise ValueError("cached and uncached logit trace lengths differ")
    if len(uncached_ids) != len(uncached_logits):
        raise ValueError("token and logit trace lengths differ")
    if not np.isfinite(max_total_variation_distance) or not (
        0.0 <= max_total_variation_distance <= 1.0
    ):
        raise ValueError(
            "max_total_variation_distance must be finite and in [0, 1]"
        )

    steps: List[Dict[str, Any]] = []
    for step, (left, right) in enumerate(zip(uncached_logits, cached_logits)):
        raw = _compare_arrays(
            left,
            right,
            atol=raw_atol,
            rtol=raw_rtol,
        )
        centered = _compare_arrays(
            centered_logits(left),
            centered_logits(right),
            atol=centered_atol,
            rtol=centered_rtol,
        )
        uncached_probabilities = stable_softmax(left)
        cached_probabilities = stable_softmax(right)
        probabilities = _compare_arrays(
            uncached_probabilities,
            cached_probabilities,
            atol=probability_atol,
            rtol=probability_rtol,
        )
        distribution = _probability_distribution_metrics(
            uncached_probabilities,
            cached_probabilities,
            max_total_variation_distance=max_total_variation_distance,
        )
        steps.append(
            {
                "step": int(step),
                "uncached_token_id": int(uncached_ids[step]),
                "cached_token_id": int(cached_ids[step]),
                "raw_logits": raw,
                "centered_logits": centered,
                "softmax_probabilities": probabilities,
                "probability_distribution": distribution,
            }
        )

    return {
        "token_ids_exact": list(uncached_ids) == list(cached_ids),
        "uncached_token_ids": [int(item) for item in uncached_ids],
        "cached_token_ids": [int(item) for item in cached_ids],
        "raw_logits": _aggregate_step_metrics(
            steps,
            "raw_logits",
            atol=raw_atol,
            rtol=raw_rtol,
            role="diagnostic_only",
        ),
        "centered_logits": _aggregate_step_metrics(
            steps,
            "centered_logits",
            atol=centered_atol,
            rtol=centered_rtol,
            role="diagnostic_only",
        ),
        "softmax_probabilities": _aggregate_step_metrics(
            steps,
            "softmax_probabilities",
            atol=probability_atol,
            rtol=probability_rtol,
            role="diagnostic_only",
        ),
        "probability_distribution": _aggregate_distribution_metrics(
            steps,
            max_total_variation_distance=max_total_variation_distance,
        ),
        "steps": steps,
    }


def evaluate_trace_acceptance(
    comparison: Dict[str, Any],
    *,
    require_exact_token_ids: bool,
    require_total_variation_bound: bool,
) -> Dict[str, Any]:
    """Evaluate only the declared trace-level acceptance checks."""

    token_ids_exact = bool(comparison["token_ids_exact"])
    distribution_within_bound = bool(
        comparison["probability_distribution"]["within_bound"]
    )
    token_requirement = token_ids_exact or not require_exact_token_ids
    distribution_requirement = (
        distribution_within_bound or not require_total_variation_bound
    )
    return {
        "raw_logits_are_diagnostic_only": True,
        "centered_logits_are_diagnostic_only": True,
        "coordinatewise_probabilities_are_diagnostic_only": True,
        "require_exact_token_ids": bool(require_exact_token_ids),
        "token_ids_exact": token_ids_exact,
        "token_id_requirement_satisfied": bool(token_requirement),
        "require_total_variation_bound": bool(
            require_total_variation_bound
        ),
        "total_variation_threshold": float(
            comparison["probability_distribution"]["threshold"]
        ),
        "max_total_variation_distance": float(
            comparison["probability_distribution"][
                "max_total_variation_distance"
            ]
        ),
        "total_variation_within_bound": distribution_within_bound,
        "probability_distribution_requirement_satisfied": bool(
            distribution_requirement
        ),
        "passed": bool(token_requirement and distribution_requirement),
    }
