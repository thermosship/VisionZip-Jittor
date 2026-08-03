"""Deterministic paired-data and supervision helpers for Phase 4A."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class PairedSample:
    """One image/caption pair linked to a real-CLIP reference row."""

    sample_id: str
    image_name: str
    caption: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PairedSample":
        allowed = {"id", "image_name", "caption"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown paired-sample keys: {unknown}")
        missing = sorted(allowed - set(payload))
        if missing:
            raise ValueError(f"Missing paired-sample keys: {missing}")
        sample = cls(
            sample_id=str(payload["id"]),
            image_name=str(payload["image_name"]),
            caption=str(payload["caption"]),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample id must not be empty")
        if not self.image_name.strip():
            raise ValueError("image_name must not be empty")
        if not self.caption.strip():
            raise ValueError("caption must contain non-whitespace text")

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.sample_id,
            "image_name": self.image_name,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class PairedManifest:
    artifact_type: str
    name: str
    description: str
    license: str
    samples: Tuple[PairedSample, ...]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PairedManifest":
        allowed = {
            "artifact_type",
            "name",
            "description",
            "license",
            "samples",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown paired-manifest keys: {unknown}")
        missing = sorted(allowed - set(payload))
        if missing:
            raise ValueError(f"Missing paired-manifest keys: {missing}")
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list):
            raise ValueError("manifest samples must be a list")
        manifest = cls(
            artifact_type=str(payload["artifact_type"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            license=str(payload["license"]),
            samples=tuple(PairedSample.from_dict(item) for item in raw_samples),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.artifact_type != "phase4a_paired_manifest_v1":
            raise ValueError(
                "artifact_type must be phase4a_paired_manifest_v1"
            )
        if not self.name.strip():
            raise ValueError("manifest name must not be empty")
        if not self.description.strip():
            raise ValueError("manifest description must not be empty")
        if not self.license.strip():
            raise ValueError("manifest license must not be empty")
        if not self.samples:
            raise ValueError("manifest must contain at least one sample")
        sample_ids = [sample.sample_id for sample in self.samples]
        image_names = [sample.image_name for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("manifest sample ids must be unique")
        if len(image_names) != len(set(image_names)):
            raise ValueError("manifest image names must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "name": self.name,
            "description": self.description,
            "license": self.license,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def load_paired_manifest(path: Union[str, Path]) -> PairedManifest:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return PairedManifest.from_dict(json.load(handle))


def deterministic_split(
    samples: Sequence[PairedSample],
    validation_fraction: float,
    seed: int,
) -> Tuple[Tuple[PairedSample, ...], Tuple[PairedSample, ...]]:
    """Return stable train/validation subsets without global RNG state."""

    if not samples:
        raise ValueError("cannot split an empty sample sequence")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be within [0, 1)")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    if validation_fraction == 0.0 or len(shuffled) == 1:
        validation_count = 0
    else:
        validation_count = max(
            1,
            int(math.floor(len(shuffled) * validation_fraction)),
        )
        validation_count = min(validation_count, len(shuffled) - 1)
    validation_ids = {
        sample.sample_id for sample in shuffled[:validation_count]
    }
    train = tuple(
        sample for sample in ordered if sample.sample_id not in validation_ids
    )
    validation = tuple(
        sample for sample in ordered if sample.sample_id in validation_ids
    )
    return train, validation


def batch_indices_for_step(
    sample_count: int,
    batch_size: int,
    seed: int,
    step: int,
) -> Tuple[int, ...]:
    """Resolve a deterministic epoch-shuffled mini-batch for a global step."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    batches_per_epoch = int(math.ceil(sample_count / batch_size))
    epoch = step // batches_per_epoch
    batch_in_epoch = step % batches_per_epoch
    order = list(range(sample_count))
    random.Random(seed + epoch).shuffle(order)
    start = batch_in_epoch * batch_size
    return tuple(order[start : start + batch_size])


def load_precomputed_visual_tokens(
    reference_path: Union[str, Path],
    samples: Sequence[PairedSample],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load compressed Phase 2 tokens and reorder them to match a manifest."""

    reference_path = Path(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 reference: {reference_path}")
    with np.load(reference_path, allow_pickle=False) as archive:
        required = {"compressed_tokens", "metadata_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Phase 2 reference is missing arrays: {missing}")
        compressed = archive["compressed_tokens"].astype(np.float32, copy=True)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("artifact_type") != "real_clip_reference_v1":
        raise ValueError("reference is not a real_clip_reference_v1 artifact")
    images = metadata.get("images")
    if not isinstance(images, list) or len(images) != len(compressed):
        raise ValueError("reference image metadata does not match token rows")
    row_by_name: Dict[str, int] = {}
    for index, image in enumerate(images):
        name = str(image.get("name", ""))
        if not name or name in row_by_name:
            raise ValueError("reference image names must be non-empty and unique")
        row_by_name[name] = index
    missing_images = [
        sample.image_name
        for sample in samples
        if sample.image_name not in row_by_name
    ]
    if missing_images:
        raise ValueError(
            f"manifest images are absent from Phase 2 reference: {missing_images}"
        )
    rows = [row_by_name[sample.image_name] for sample in samples]
    selected = compressed[np.asarray(rows, dtype=np.int64)]
    return selected, {
        "reference_path": str(reference_path),
        "reference_metadata": metadata,
        "source_rows": rows,
        "sample_ids": [sample.sample_id for sample in samples],
        "image_names": [sample.image_name for sample in samples],
    }


def prepare_text_supervision(
    tokenizer,
    captions: Sequence[str],
    prompt: str,
    max_caption_tokens: int,
) -> Dict[str, Any]:
    """Tokenize a shared prompt and padded caption targets using NumPy."""

    if not captions:
        raise ValueError("captions must not be empty")
    if max_caption_tokens <= 0:
        raise ValueError("max_caption_tokens must be positive")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    eos_token_id = int(eos_token_id)
    prompt_ids = [
        int(item)
        for item in tokenizer.encode(prompt, add_special_tokens=False)
    ]
    if not prompt_ids:
        raise ValueError("prompt must tokenize to at least one token")
    target_lists: List[List[int]] = []
    for caption in captions:
        encoded = [
            int(item)
            for item in tokenizer.encode(caption, add_special_tokens=False)
        ]
        encoded = encoded[: max_caption_tokens - 1] + [eos_token_id]
        target_lists.append(encoded)
    maximum_target = max(len(item) for item in target_lists)
    target_ids = np.full(
        (len(captions), maximum_target),
        eos_token_id,
        dtype=np.int32,
    )
    target_mask = np.zeros(
        (len(captions), maximum_target),
        dtype=np.float32,
    )
    for index, target in enumerate(target_lists):
        target_ids[index, : len(target)] = target
        target_mask[index, : len(target)] = 1.0
    return {
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int32),
        "target_ids": target_ids,
        "target_mask": target_mask,
        "target_token_counts": [len(item) for item in target_lists],
    }


def build_label_arrays(
    prompt_tokens: int,
    visual_tokens: int,
    target_ids: np.ndarray,
    target_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build target-only causal labels and full-sequence attention masks."""

    if prompt_tokens <= 0 or visual_tokens <= 0:
        raise ValueError("prompt_tokens and visual_tokens must be positive")
    target_ids = np.asarray(target_ids, dtype=np.int32)
    target_mask = np.asarray(target_mask, dtype=np.float32)
    if target_ids.ndim != 2 or target_mask.shape != target_ids.shape:
        raise ValueError("target ids/mask must be matching rank-2 arrays")
    batch_size, target_tokens = target_ids.shape
    total_tokens = prompt_tokens + visual_tokens + target_tokens
    target_start = prompt_tokens + visual_tokens
    labels = np.zeros((batch_size, total_tokens), dtype=np.int32)
    label_mask = np.zeros((batch_size, total_tokens), dtype=np.float32)
    attention_mask = np.ones((batch_size, total_tokens), dtype=np.float32)
    labels[:, target_start:] = target_ids
    label_mask[:, target_start:] = target_mask
    attention_mask[:, target_start:] = target_mask
    return {
        "labels": labels,
        "label_mask": label_mask,
        "attention_mask": attention_mask,
        "target_start": np.asarray(target_start, dtype=np.int32),
    }
