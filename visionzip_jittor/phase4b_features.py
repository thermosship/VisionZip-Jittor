"""Sharded, hashed visual-feature storage for Phase 4B."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple, Union

import numpy as np

from .phase4b_data import canonical_json_sha256, file_sha256


@dataclass(frozen=True)
class Phase4BFeatureShard:
    path: str
    sha256: str
    sample_count: int
    sample_ids_sha256: str
    first_sample_id: str
    last_sample_id: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BFeatureShard":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown feature-shard keys: {unknown}")
        if missing:
            raise ValueError(f"Missing feature-shard keys: {missing}")
        shard = cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            sample_count=int(payload["sample_count"]),
            sample_ids_sha256=str(payload["sample_ids_sha256"]),
            first_sample_id=str(payload["first_sample_id"]),
            last_sample_id=str(payload["last_sample_id"]),
        )
        shard.validate()
        return shard

    def validate(self) -> None:
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("feature shard path must be a safe relative path")
        if not self.path.endswith(".npz"):
            raise ValueError("feature shard path must end with .npz")
        for name in ("sha256", "sample_ids_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"feature shard {name} must be lowercase SHA256")
        if self.sample_count <= 0:
            raise ValueError("feature shard sample_count must be positive")
        if not self.first_sample_id or not self.last_sample_id:
            raise ValueError("feature shard boundary sample ids must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class Phase4BFeatureManifest:
    artifact_type: str
    config_sha256: str
    dataset_samples_sha256: str
    model_name_or_path: str
    model_revision: str
    visionzip_config: Dict[str, Any]
    requested_layer_index: int
    resolved_layer_index: int
    storage_dtype: str
    token_shape: Tuple[int, int]
    sample_count: int
    shards: Tuple[Phase4BFeatureShard, ...]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BFeatureManifest":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown feature-manifest keys: {unknown}")
        if missing:
            raise ValueError(f"Missing feature-manifest keys: {missing}")
        manifest = cls(
            artifact_type=str(payload["artifact_type"]),
            config_sha256=str(payload["config_sha256"]),
            dataset_samples_sha256=str(payload["dataset_samples_sha256"]),
            model_name_or_path=str(payload["model_name_or_path"]),
            model_revision=str(payload["model_revision"]),
            visionzip_config=dict(payload["visionzip_config"]),
            requested_layer_index=int(payload["requested_layer_index"]),
            resolved_layer_index=int(payload["resolved_layer_index"]),
            storage_dtype=str(payload["storage_dtype"]),
            token_shape=tuple(int(item) for item in payload["token_shape"]),
            sample_count=int(payload["sample_count"]),
            shards=tuple(Phase4BFeatureShard.from_dict(item) for item in payload["shards"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.artifact_type != "phase4b_feature_manifest_v1":
            raise ValueError("feature artifact_type must be phase4b_feature_manifest_v1")
        for name in ("config_sha256", "dataset_samples_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"feature manifest {name} must be lowercase SHA256")
        if (
            not self.model_name_or_path
            or not re.fullmatch(r"[0-9a-f]{40}", self.model_revision)
        ):
            raise ValueError("feature model identity is invalid")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("feature storage_dtype must be float16 or float32")
        if len(self.token_shape) != 2 or any(item <= 0 for item in self.token_shape):
            raise ValueError("feature token_shape must contain two positive dimensions")
        if self.sample_count <= 0 or not self.shards:
            raise ValueError("feature manifest must contain samples and shards")
        if sum(item.sample_count for item in self.shards) != self.sample_count:
            raise ValueError("feature shard counts do not sum to sample_count")
        if len({item.path for item in self.shards}) != len(self.shards):
            raise ValueError("feature shard paths must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "config_sha256": self.config_sha256,
            "dataset_samples_sha256": self.dataset_samples_sha256,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "visionzip_config": self.visionzip_config,
            "requested_layer_index": self.requested_layer_index,
            "resolved_layer_index": self.resolved_layer_index,
            "storage_dtype": self.storage_dtype,
            "token_shape": list(self.token_shape),
            "sample_count": self.sample_count,
            "shards": [item.to_dict() for item in self.shards],
        }


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return canonical_json_sha256(list(sample_ids))


def validate_feature_arrays(arrays: Mapping[str, np.ndarray]) -> Tuple[int, Tuple[int, int]]:
    required = {"sample_ids", "compressed_tokens", "selected_indices", "assignments"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"feature shard arrays are missing: {missing}")
    sample_ids = np.asarray(arrays["sample_ids"])
    tokens = np.asarray(arrays["compressed_tokens"])
    selected = np.asarray(arrays["selected_indices"])
    assignments = np.asarray(arrays["assignments"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("feature sample_ids must be a one-dimensional string array")
    sample_count = int(sample_ids.shape[0])
    if sample_count <= 0 or len(set(sample_ids.astype(str).tolist())) != sample_count:
        raise ValueError("feature sample_ids must be non-empty and unique")
    if tokens.ndim != 3 or tokens.shape[0] != sample_count:
        raise ValueError("compressed_tokens must have shape [samples, tokens, hidden]")
    if tokens.dtype not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("compressed_tokens must use float16 or float32")
    if selected.ndim != 2 or selected.shape[0] != sample_count or selected.shape[1] <= 0:
        raise ValueError("selected_indices must have shape [samples, selected]")
    if assignments.ndim != 2 or assignments.shape[0] != sample_count or assignments.shape[1] <= 0:
        raise ValueError("assignments must have shape [samples, merge]")
    if selected.dtype.kind not in {"i", "u"}:
        raise ValueError("selected_indices must use an integer dtype")
    if assignments.dtype.kind not in {"i", "u"}:
        raise ValueError("assignments must use an integer dtype")
    if np.any(selected < 0) or np.any(assignments < 0):
        raise ValueError("feature indices and assignments must be non-negative")
    if not np.isfinite(tokens).all():
        raise ValueError("compressed_tokens must be finite")
    return sample_count, (int(tokens.shape[1]), int(tokens.shape[2]))


def write_feature_shard(
    output_dir: Union[str, Path],
    shard_index: int,
    arrays: Mapping[str, np.ndarray],
) -> Phase4BFeatureShard:
    output_dir = Path(output_dir)
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    sample_count, _ = validate_feature_arrays(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"features-{shard_index:05d}.npz"
    path = output_dir / name
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))
    ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
    shard = Phase4BFeatureShard(
        path=name,
        sha256=file_sha256(path),
        sample_count=sample_count,
        sample_ids_sha256=sample_ids_sha256(ids),
        first_sample_id=ids[0],
        last_sample_id=ids[-1],
    )
    shard.validate()
    return shard


def write_feature_manifest(
    output_dir: Union[str, Path],
    config_payload: Dict[str, Any],
    dataset_samples_sha256: str,
    model_name_or_path: str,
    model_revision: str,
    visionzip_config: Dict[str, Any],
    requested_layer_index: int,
    resolved_layer_index: int,
    storage_dtype: str,
    token_shape: Sequence[int],
    shards: Sequence[Phase4BFeatureShard],
) -> Phase4BFeatureManifest:
    output_dir = Path(output_dir)
    manifest = Phase4BFeatureManifest(
        artifact_type="phase4b_feature_manifest_v1",
        config_sha256=canonical_json_sha256(config_payload),
        dataset_samples_sha256=dataset_samples_sha256,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        visionzip_config=dict(visionzip_config),
        requested_layer_index=requested_layer_index,
        resolved_layer_index=resolved_layer_index,
        storage_dtype=storage_dtype,
        token_shape=tuple(int(item) for item in token_shape),
        sample_count=sum(item.sample_count for item in shards),
        shards=tuple(shards),
    )
    manifest.validate()
    path = output_dir / "manifest.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return manifest


def load_feature_manifest(
    path: Union[str, Path],
    verify_shards: bool = True,
) -> Phase4BFeatureManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        manifest = Phase4BFeatureManifest.from_dict(json.load(handle))
    if verify_shards:
        for shard in manifest.shards:
            shard_path = path.parent / shard.path
            if not shard_path.is_file():
                raise ValueError(f"feature shard is missing: {shard.path}")
            if file_sha256(shard_path) != shard.sha256:
                raise ValueError(f"feature shard SHA256 mismatch: {shard.path}")
    return manifest


def iter_feature_shards(
    manifest_path: Union[str, Path],
    verify_shards: bool = True,
) -> Iterator[Tuple[Phase4BFeatureShard, Dict[str, np.ndarray]]]:
    manifest_path = Path(manifest_path)
    manifest = load_feature_manifest(manifest_path, verify_shards=verify_shards)
    expected_shape = manifest.token_shape
    for shard in manifest.shards:
        with np.load(manifest_path.parent / shard.path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        sample_count, token_shape = validate_feature_arrays(arrays)
        if sample_count != shard.sample_count or token_shape != expected_shape:
            raise ValueError(f"feature shard shape mismatch: {shard.path}")
        ids = arrays["sample_ids"].astype(str).tolist()
        if sample_ids_sha256(ids) != shard.sample_ids_sha256:
            raise ValueError(f"feature shard sample-id hash mismatch: {shard.path}")
        yield shard, arrays


def build_feature_index(
    manifest_path: Union[str, Path],
    verify_shards: bool = True,
) -> Dict[str, Tuple[str, int]]:
    index: Dict[str, Tuple[str, int]] = {}
    for shard, arrays in iter_feature_shards(manifest_path, verify_shards=verify_shards):
        for row, sample_id in enumerate(arrays["sample_ids"].astype(str).tolist()):
            if sample_id in index:
                raise ValueError(f"duplicate feature sample id: {sample_id}")
            index[sample_id] = (shard.path, row)
    return index
