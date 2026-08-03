"""Deterministic preparation and integrity helpers for Phase 4B datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from .phase4b_config import Phase4BConfig


_WHITESPACE = re.compile(r"\s+")


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_caption(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def source_image_bytes(value: Any) -> bytes:
    """Return embedded Hugging Face Image bytes from a PyArrow/Python value."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        embedded = value.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)):
            return bytes(embedded)
        source_path = value.get("path")
        if source_path:
            return Path(str(source_path)).read_bytes()
    raise ValueError("source image is neither embedded bytes nor a readable path")


def source_sample_id(row: Mapping[str, Any]) -> str:
    image_hash = str(row.get("sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", image_hash):
        return "ccby-" + image_hash[:24]
    fallback = "{}:{}".format(row.get("uid", ""), row.get("photoid", ""))
    return "ccby-" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]


def source_row_rejection(
    row: Mapping[str, Any],
    config: Phase4BConfig,
) -> Union[str, None]:
    settings = config.dataset
    if str(row.get(settings.status_field) or "") != settings.accepted_status:
        return "status"
    caption = normalize_caption(row.get(settings.caption_field))
    words = caption.split()
    if not settings.min_caption_words <= len(words) <= settings.max_caption_words:
        return "caption_words"
    if len(caption) > settings.max_caption_chars:
        return "caption_chars"
    try:
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
    except (TypeError, ValueError):
        return "dimensions"
    if min(width, height) < settings.min_dimension:
        return "dimensions"
    license_url = str(row.get("licenseurl") or "")
    if not any(
        license_url.startswith(prefix)
        for prefix in settings.allowed_license_url_prefixes
    ):
        return "license"
    if not normalize_caption(row.get("licensename")):
        return "license_name"
    if not normalize_caption(row.get("unickname")):
        return "creator"
    source_page_url = str(row.get("pageurl") or "")
    if not source_page_url.startswith(("http://", "https://")):
        return "source_page"
    return None


@dataclass(frozen=True)
class Phase4BPreparedSample:
    sample_id: str
    split: str
    image_path: str
    image_sha256: str
    caption: str
    source_shard: str
    source_row: int
    source_photo_id: str
    source_uid: str
    source_image_sha256: str
    creator_name: str
    title: str
    source_page_url: str
    license_name: str
    license_url: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BPreparedSample":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown prepared-sample keys: {unknown}")
        if missing:
            raise ValueError(f"Missing prepared-sample keys: {missing}")
        sample = cls(
            sample_id=str(payload["sample_id"]),
            split=str(payload["split"]),
            image_path=str(payload["image_path"]),
            image_sha256=str(payload["image_sha256"]),
            caption=str(payload["caption"]),
            source_shard=str(payload["source_shard"]),
            source_row=int(payload["source_row"]),
            source_photo_id=str(payload["source_photo_id"]),
            source_uid=str(payload["source_uid"]),
            source_image_sha256=str(payload["source_image_sha256"]),
            creator_name=str(payload["creator_name"]),
            title=str(payload["title"]),
            source_page_url=str(payload["source_page_url"]),
            license_name=str(payload["license_name"]),
            license_url=str(payload["license_url"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("prepared sample_id must not be empty")
        if self.split not in {"train", "validation"}:
            raise ValueError("prepared sample split must be train or validation")
        if Path(self.image_path).is_absolute() or ".." in Path(self.image_path).parts:
            raise ValueError("prepared image_path must be a safe relative path")
        for name in ("image_sha256", "source_image_sha256"):
            value = getattr(self, name)
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"prepared {name} must be lowercase SHA256")
        if self.image_sha256 != self.source_image_sha256:
            raise ValueError("prepared image and source SHA256 values must match")
        if not self.caption.strip():
            raise ValueError("prepared caption must not be empty")
        if self.source_row < 0:
            raise ValueError("prepared source_row must be non-negative")
        if not self.source_shard.endswith(".parquet"):
            raise ValueError("prepared source_shard must end with .parquet")
        if not self.source_page_url.startswith(("http://", "https://")):
            raise ValueError("prepared source_page_url must be an HTTP URL")
        if not self.license_url.startswith(("http://", "https://")):
            raise ValueError("prepared license_url must be an HTTP URL")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("prepared image dimensions must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class Phase4BPreparedManifest:
    artifact_type: str
    config_sha256: str
    dataset_id: str
    dataset_revision: str
    samples_file: str
    samples_sha256: str
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    source_file_count: int
    source_size_bytes: int
    source_files: Tuple[Dict[str, Any], ...]
    rejection_counts: Dict[str, int]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BPreparedManifest":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown prepared-manifest keys: {unknown}")
        if missing:
            raise ValueError(f"Missing prepared-manifest keys: {missing}")
        manifest = cls(
            artifact_type=str(payload["artifact_type"]),
            config_sha256=str(payload["config_sha256"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_revision=str(payload["dataset_revision"]),
            samples_file=str(payload["samples_file"]),
            samples_sha256=str(payload["samples_sha256"]),
            sample_count=int(payload["sample_count"]),
            train_sample_count=int(payload["train_sample_count"]),
            validation_sample_count=int(payload["validation_sample_count"]),
            source_file_count=int(payload["source_file_count"]),
            source_size_bytes=int(payload["source_size_bytes"]),
            source_files=tuple(dict(item) for item in payload["source_files"]),
            rejection_counts={str(key): int(value) for key, value in payload["rejection_counts"].items()},
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.artifact_type != "phase4b_prepared_dataset_v1":
            raise ValueError("prepared artifact_type must be phase4b_prepared_dataset_v1")
        for name in ("config_sha256", "samples_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"prepared manifest {name} must be lowercase SHA256")
        if not self.dataset_id or len(self.dataset_revision) != 40:
            raise ValueError("prepared manifest dataset identity is invalid")
        if Path(self.samples_file).is_absolute() or ".." in Path(self.samples_file).parts:
            raise ValueError("prepared samples_file must be a safe relative path")
        if self.sample_count != self.train_sample_count + self.validation_sample_count:
            raise ValueError("prepared split counts do not sum to sample_count")
        if self.sample_count <= 0 or self.source_file_count <= 0 or self.source_size_bytes <= 0:
            raise ValueError("prepared manifest counts and source size must be positive")
        if len(self.source_files) != self.source_file_count:
            raise ValueError("prepared source_file_count does not match source_files")
        if any(value < 0 for value in self.rejection_counts.values()):
            raise ValueError("prepared rejection counts must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "config_sha256": self.config_sha256,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "samples_file": self.samples_file,
            "samples_sha256": self.samples_sha256,
            "sample_count": self.sample_count,
            "train_sample_count": self.train_sample_count,
            "validation_sample_count": self.validation_sample_count,
            "source_file_count": self.source_file_count,
            "source_size_bytes": self.source_size_bytes,
            "source_files": list(self.source_files),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
        }


def assign_exact_splits(
    samples: Sequence[Phase4BPreparedSample],
    validation_sample_count: int,
    seed: int,
) -> Tuple[Phase4BPreparedSample, ...]:
    if not 0 < validation_sample_count < len(samples):
        raise ValueError("validation_sample_count must be within the sample sequence")
    ranked = sorted(
        samples,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.sample_id}".encode("utf-8")
        ).digest(),
    )
    validation_ids = {
        item.sample_id for item in ranked[:validation_sample_count]
    }
    return tuple(
        replace(
            item,
            split="validation" if item.sample_id in validation_ids else "train",
        )
        for item in samples
    )


def prepared_sample_from_row(
    row: Mapping[str, Any],
    config: Phase4BConfig,
    source_shard: str,
    source_row: int,
    image_path: str,
    image_sha256: str,
) -> Phase4BPreparedSample:
    return Phase4BPreparedSample(
        sample_id=source_sample_id(row),
        split="train",
        image_path=image_path,
        image_sha256=image_sha256,
        caption=normalize_caption(row.get(config.dataset.caption_field)),
        source_shard=source_shard,
        source_row=source_row,
        source_photo_id=str(row.get("photoid") or ""),
        source_uid=str(row.get("uid") or ""),
        source_image_sha256=str(row.get("sha256") or image_sha256).lower(),
        creator_name=normalize_caption(row.get("unickname")),
        title=normalize_caption(row.get("title")),
        source_page_url=str(row.get("pageurl") or ""),
        license_name=normalize_caption(row.get("licensename")),
        license_url=str(row.get("licenseurl") or ""),
        width=int(row.get("width")),
        height=int(row.get("height")),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_prepared_dataset_manifest(
    output_dir: Union[str, Path],
    config: Phase4BConfig,
    samples: Sequence[Phase4BPreparedSample],
    source_files: Sequence[Dict[str, Any]],
    rejection_counts: Mapping[str, int],
) -> Phase4BPreparedManifest:
    output_dir = Path(output_dir)
    if not samples:
        raise ValueError("cannot write an empty prepared dataset")
    sample_ids = [item.sample_id for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prepared sample ids must be unique")
    samples_name = "samples.jsonl"
    sample_text = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in samples
    )
    samples_path = output_dir / samples_name
    _atomic_write_text(samples_path, sample_text)
    train_count = sum(item.split == "train" for item in samples)
    validation_count = sum(item.split == "validation" for item in samples)
    manifest = Phase4BPreparedManifest(
        artifact_type="phase4b_prepared_dataset_v1",
        config_sha256=canonical_json_sha256(config.to_dict()),
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        samples_file=samples_name,
        samples_sha256=file_sha256(samples_path),
        sample_count=len(samples),
        train_sample_count=train_count,
        validation_sample_count=validation_count,
        source_file_count=len(source_files),
        source_size_bytes=sum(int(item["size_bytes"]) for item in source_files),
        source_files=tuple(dict(item) for item in source_files),
        rejection_counts={str(key): int(value) for key, value in rejection_counts.items()},
    )
    manifest.validate()
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return manifest


def load_prepared_dataset(
    manifest_path: Union[str, Path],
    verify_images: bool = False,
) -> Tuple[Phase4BPreparedManifest, Tuple[Phase4BPreparedSample, ...]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = Phase4BPreparedManifest.from_dict(json.load(handle))
    samples_path = manifest_path.parent / manifest.samples_file
    if file_sha256(samples_path) != manifest.samples_sha256:
        raise ValueError("prepared samples.jsonl SHA256 mismatch")
    samples: List[Phase4BPreparedSample] = []
    with samples_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(Phase4BPreparedSample.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(f"invalid prepared sample at line {line_number}") from error
    if len(samples) != manifest.sample_count:
        raise ValueError("prepared sample count does not match manifest")
    ids = [item.sample_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("prepared sample ids are not unique")
    train_count = sum(item.split == "train" for item in samples)
    validation_count = sum(item.split == "validation" for item in samples)
    if (train_count, validation_count) != (
        manifest.train_sample_count,
        manifest.validation_sample_count,
    ):
        raise ValueError("prepared split counts do not match manifest")
    if verify_images:
        for item in samples:
            image_path = manifest_path.parent / item.image_path
            if not image_path.is_file():
                raise ValueError(f"prepared image is missing: {item.image_path}")
            if file_sha256(image_path) != item.image_sha256:
                raise ValueError(f"prepared image SHA256 mismatch: {item.image_path}")
    return manifest, tuple(samples)


def preflight_report(config: Phase4BConfig, free_bytes: Union[int, None] = None) -> Dict[str, Any]:
    source_bytes = config.dataset.source_size_bytes
    estimated_image_bytes = source_bytes
    feature_bytes = (
        config.dataset.target_sample_count
        * (config.training.budget + 1)
        * config.training.vision_hidden_size
        * (2 if config.features.storage_dtype == "float16" else 4)
    )
    working_headroom = 4 * 1024 ** 3
    estimated_required = source_bytes + estimated_image_bytes + feature_bytes + working_headroom
    report = {
        "artifact_type": "phase4b_preflight_v1",
        "dataset_id": config.dataset.dataset_id,
        "dataset_revision": config.dataset.revision,
        "source_file_count": len(config.dataset.source_shards),
        "source_row_count": config.dataset.source_row_count,
        "source_size_bytes": source_bytes,
        "target_sample_count": config.dataset.target_sample_count,
        "train_sample_count": config.dataset.train_sample_count,
        "validation_sample_count": config.dataset.validation_sample_count,
        "feature_storage_dtype": config.features.storage_dtype,
        "estimated_feature_bytes": feature_bytes,
        "working_headroom_bytes": working_headroom,
        "estimated_required_bytes": estimated_required,
    }
    if free_bytes is not None:
        report["free_bytes"] = int(free_bytes)
        report["disk_preflight_passed"] = int(free_bytes) >= estimated_required
    return report
