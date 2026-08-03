"""Deterministic training-data and evaluation utilities for Phase 4B.

This module intentionally has no Jittor dependency so dataset/feature validation,
batch scheduling, learning-rate policy, checkpoint retention, and caption metrics
can be tested on the Windows working copy before an expensive AutoDL run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from .phase4b_data import (
    Phase4BPreparedManifest,
    Phase4BPreparedSample,
    load_prepared_dataset,
)
from .phase4b_features import (
    Phase4BFeatureManifest,
    iter_feature_shards,
    load_feature_manifest,
)


_WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", flags=re.UNICODE)
_CHECKPOINT_RE = re.compile(r"projector_step_(\d{6,})\.npz$")


@dataclass(frozen=True)
class Phase4BLoadedFeatures:
    """Prepared samples and dense frozen features in identical row order."""

    prepared_manifest: Phase4BPreparedManifest
    feature_manifest: Phase4BFeatureManifest
    samples: Tuple[Phase4BPreparedSample, ...]
    compressed_tokens: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def sample_ids(self) -> Tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)


def load_phase4b_training_features(
    prepared_manifest_path: Union[str, Path],
    feature_manifest_path: Union[str, Path],
    *,
    verify_images: bool = False,
    verify_shards: bool = True,
) -> Phase4BLoadedFeatures:
    """Load all frozen shards once while enforcing exact dataset row identity.

    The destination array is allocated once and filled shard-by-shard. This
    avoids the temporary second multi-gigabyte allocation caused by
    ``np.concatenate`` for the real 8,192-sample feature store.
    """

    prepared_manifest, samples = load_prepared_dataset(
        prepared_manifest_path,
        verify_images=verify_images,
    )
    feature_manifest = load_feature_manifest(
        feature_manifest_path,
        verify_shards=verify_shards,
    )
    if feature_manifest.dataset_samples_sha256 != prepared_manifest.samples_sha256:
        raise ValueError("feature store dataset samples SHA256 mismatch")
    if feature_manifest.config_sha256 != prepared_manifest.config_sha256:
        raise ValueError("feature store config SHA256 mismatch")
    if feature_manifest.sample_count != len(samples):
        raise ValueError("feature store sample count does not match prepared dataset")

    dtype = np.dtype(feature_manifest.storage_dtype)
    shape = (
        feature_manifest.sample_count,
        feature_manifest.token_shape[0],
        feature_manifest.token_shape[1],
    )
    compressed = np.empty(shape, dtype=dtype)
    expected_ids = [sample.sample_id for sample in samples]
    cursor = 0
    for shard, arrays in iter_feature_shards(
        feature_manifest_path,
        verify_shards=verify_shards,
    ):
        count = shard.sample_count
        actual_ids = [str(item) for item in arrays["sample_ids"].tolist()]
        expected_slice = expected_ids[cursor : cursor + count]
        if actual_ids != expected_slice:
            raise ValueError(
                f"feature sample order mismatch in {shard.path} at row {cursor}"
            )
        values = arrays["compressed_tokens"]
        if values.dtype != dtype:
            raise ValueError(
                f"feature dtype mismatch in {shard.path}: "
                f"expected {dtype}, got {values.dtype}"
            )
        compressed[cursor : cursor + count] = values
        cursor += count
    if cursor != len(samples):
        raise ValueError("feature shards did not fill the prepared sample sequence")
    if not np.isfinite(compressed).all():
        raise ValueError("feature store contains non-finite compressed tokens")

    train_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.split == "train"],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if sample.split == "validation"
        ],
        dtype=np.int64,
    )
    if len(train_indices) != prepared_manifest.train_sample_count:
        raise ValueError("prepared train split count mismatch")
    if len(validation_indices) != prepared_manifest.validation_sample_count:
        raise ValueError("prepared validation split count mismatch")
    return Phase4BLoadedFeatures(
        prepared_manifest=prepared_manifest,
        feature_manifest=feature_manifest,
        samples=samples,
        compressed_tokens=compressed,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def _epoch_permutation(sample_count: int, seed: int, epoch: int) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if seed < 0 or epoch < 0:
        raise ValueError("seed and epoch must be non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return rng.permutation(sample_count).astype(np.int64, copy=False)


def batch_indices_for_optimizer_step(
    sample_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    optimizer_step: int,
    seed: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Return deterministic microbatches for a zero-based optimizer step.

    The conceptual shuffled stream is continuous across epoch boundaries, so
    resume needs only the completed optimizer-step count and not a mutable RNG
    snapshot. Every epoch uses a separate SeedSequence derived from seed/epoch.
    """

    if micro_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("micro batch and accumulation counts must be positive")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    effective = micro_batch_size * gradient_accumulation_steps
    start = optimizer_step * effective
    remaining = effective
    flat: List[int] = []
    while remaining:
        epoch = start // sample_count
        offset = start % sample_count
        permutation = _epoch_permutation(sample_count, seed, epoch)
        take = min(remaining, sample_count - offset)
        flat.extend(int(item) for item in permutation[offset : offset + take])
        start += take
        remaining -= take
    return tuple(
        tuple(flat[index : index + micro_batch_size])
        for index in range(0, effective, micro_batch_size)
    )


def learning_rate_for_optimizer_step(
    optimizer_step: int,
    base_learning_rate: float,
    warmup_steps: int,
    max_optimizer_steps: int,
) -> float:
    """Linear warmup followed by cosine decay, using a one-based step."""

    if not 1 <= optimizer_step <= max_optimizer_steps:
        raise ValueError("optimizer_step must be within [1, max_optimizer_steps]")
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if not 0 <= warmup_steps < max_optimizer_steps:
        raise ValueError("warmup_steps must be within the run")
    if warmup_steps and optimizer_step <= warmup_steps:
        return base_learning_rate * optimizer_step / warmup_steps
    decay_steps = max_optimizer_steps - warmup_steps
    # Optimizer step warmup_steps + 1 is the first cosine-decay update and
    # starts at the base LR. The final configured update remains positive;
    # the conceptual next step would reach exactly zero.
    decay_index = optimizer_step - warmup_steps - 1
    progress = decay_index / decay_steps
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def deterministic_subset_indices(
    population_size: int,
    sample_count: int,
    seed: int,
) -> Tuple[int, ...]:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 0 < sample_count <= population_size:
        raise ValueError("sample_count must be within the population")
    permutation = _epoch_permutation(population_size, seed, 0)
    return tuple(sorted(int(item) for item in permutation[:sample_count]))


def checkpoint_step(path: Union[str, Path]) -> int:
    match = _CHECKPOINT_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"unrecognized Phase 4 checkpoint name: {path}")
    return int(match.group(1))


def checkpoints_to_remove(
    checkpoint_paths: Iterable[Union[str, Path]],
    keep_last: int,
    protected: Iterable[Union[str, Path]] = (),
) -> Tuple[Path, ...]:
    """Choose old rolling checkpoints for deletion without touching best/final."""

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    paths = sorted((Path(item) for item in checkpoint_paths), key=checkpoint_step)
    protected_resolved = {Path(item).resolve() for item in protected}
    removable = [item for item in paths if item.resolve() not in protected_resolved]
    excess = max(0, len(removable) - keep_last)
    return tuple(removable[:excess])


def tokenize_caption(text: str) -> Tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _WORD_RE.finditer(str(text)))


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(max(0, len(tokens) - order + 1))
    )


def corpus_bleu(
    references: Sequence[str],
    hypotheses: Sequence[str],
    max_order: int,
    *,
    add_one_smoothing: bool,
) -> float:
    """Single-reference corpus BLEU with explicit optional add-one smoothing."""

    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and aligned")
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    matches = [0] * max_order
    totals = [0] * max_order
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = tokenize_caption(reference)
        hypothesis_tokens = tokenize_caption(hypothesis)
        reference_length += len(reference_tokens)
        hypothesis_length += len(hypothesis_tokens)
        for order in range(1, max_order + 1):
            reference_counts = _ngram_counts(reference_tokens, order)
            hypothesis_counts = _ngram_counts(hypothesis_tokens, order)
            matches[order - 1] += sum(
                min(count, reference_counts[ngram])
                for ngram, count in hypothesis_counts.items()
            )
            totals[order - 1] += sum(hypothesis_counts.values())
    if hypothesis_length == 0:
        return 0.0
    precisions: List[float] = []
    for matched, total in zip(matches, totals):
        if add_one_smoothing:
            precisions.append((matched + 1.0) / (total + 1.0))
        elif total == 0 or matched == 0:
            return 0.0
        else:
            precisions.append(matched / total)
    brevity_penalty = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return float(
        brevity_penalty
        * math.exp(sum(math.log(value) for value in precisions) / max_order)
    )


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = tokenize_caption(reference)
    hypothesis_tokens = tokenize_caption(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def caption_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> Dict[str, object]:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and aligned")
    rouge_values = [
        rouge_l_f1(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]
    return {
        "bleu_1_single_synthetic_reference": corpus_bleu(
            references,
            hypotheses,
            1,
            add_one_smoothing=False,
        ),
        "bleu_4_single_synthetic_reference": corpus_bleu(
            references,
            hypotheses,
            4,
            add_one_smoothing=True,
        ),
        "rouge_l_single_synthetic_reference": float(np.mean(rouge_values)),
        "tokenization": "lowercase Unicode word tokens",
        "bleu_4_smoothing": "add-one smoothing for every n-gram order",
        "reference_type": "one BLIP-2 synthetic caption per held-out image",
        "sample_count": len(references),
    }
