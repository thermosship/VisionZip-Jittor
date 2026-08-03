"""Versioned configuration for Phase 4B licensed paired-data training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union


@dataclass(frozen=True)
class Phase4BSourceShard:
    """One immutable Parquet object selected from the source dataset."""

    path: str
    size_bytes: int
    num_rows: int
    repository_oid: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BSourceShard":
        allowed = {"path", "size_bytes", "num_rows", "repository_oid"}
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B source-shard keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B source-shard keys: {missing}")
        shard = cls(
            path=str(payload["path"]),
            size_bytes=int(payload["size_bytes"]),
            num_rows=int(payload["num_rows"]),
            repository_oid=str(payload["repository_oid"]),
        )
        shard.validate()
        return shard

    def validate(self) -> None:
        if not self.path.endswith(".parquet"):
            raise ValueError("source shard path must end with .parquet")
        if self.size_bytes <= 0:
            raise ValueError("source shard size_bytes must be positive")
        if self.num_rows <= 0:
            raise ValueError("source shard num_rows must be positive")
        if not all(character in "0123456789abcdef" for character in self.repository_oid) or len(self.repository_oid) != 40:
            raise ValueError("source shard repository_oid must be a lowercase 40-character Git oid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "num_rows": self.num_rows,
            "repository_oid": self.repository_oid,
        }


@dataclass(frozen=True)
class Phase4BDatasetSettings:
    dataset_id: str
    revision: str
    split: str
    image_field: str
    caption_field: str
    status_field: str
    accepted_status: str
    source_shards: Tuple[Phase4BSourceShard, ...]
    target_sample_count: int
    validation_sample_count: int
    split_seed: int
    min_dimension: int
    min_caption_words: int
    max_caption_words: int
    max_caption_chars: int
    allowed_license_url_prefixes: Tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BDatasetSettings":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B dataset keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B dataset keys: {missing}")
        settings = cls(
            dataset_id=str(payload["dataset_id"]),
            revision=str(payload["revision"]),
            split=str(payload["split"]),
            image_field=str(payload["image_field"]),
            caption_field=str(payload["caption_field"]),
            status_field=str(payload["status_field"]),
            accepted_status=str(payload["accepted_status"]),
            source_shards=tuple(
                Phase4BSourceShard.from_dict(item)
                for item in payload["source_shards"]
            ),
            target_sample_count=int(payload["target_sample_count"]),
            validation_sample_count=int(payload["validation_sample_count"]),
            split_seed=int(payload["split_seed"]),
            min_dimension=int(payload["min_dimension"]),
            min_caption_words=int(payload["min_caption_words"]),
            max_caption_words=int(payload["max_caption_words"]),
            max_caption_chars=int(payload["max_caption_chars"]),
            allowed_license_url_prefixes=tuple(
                str(item) for item in payload["allowed_license_url_prefixes"]
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        for name in (
            "dataset_id", "revision", "split", "image_field", "caption_field",
            "status_field", "accepted_status",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"dataset {name} must not be empty")
        if len(self.revision) != 40 or not all(
            character in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("dataset revision must be a lowercase 40-character commit hash")
        if not self.source_shards:
            raise ValueError("at least one source shard is required")
        if len({item.path for item in self.source_shards}) != len(self.source_shards):
            raise ValueError("source shard paths must be unique")
        total_rows = sum(item.num_rows for item in self.source_shards)
        if self.target_sample_count <= 0 or self.target_sample_count > total_rows:
            raise ValueError("target_sample_count must fit within declared source rows")
        if not 0 < self.validation_sample_count < self.target_sample_count:
            raise ValueError("validation_sample_count must be within the target sample count")
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        if self.min_dimension <= 0:
            raise ValueError("min_dimension must be positive")
        if not 0 < self.min_caption_words <= self.max_caption_words:
            raise ValueError("caption word limits are invalid")
        if self.max_caption_chars <= 0:
            raise ValueError("max_caption_chars must be positive")
        if not self.allowed_license_url_prefixes:
            raise ValueError("allowed_license_url_prefixes must not be empty")
        if any(not item.startswith(("http://", "https://")) for item in self.allowed_license_url_prefixes):
            raise ValueError("license URL prefixes must be absolute HTTP URLs")

    @property
    def train_sample_count(self) -> int:
        return self.target_sample_count - self.validation_sample_count

    @property
    def source_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.source_shards)

    @property
    def source_row_count(self) -> int:
        return sum(item.num_rows for item in self.source_shards)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"source_shards", "allowed_license_url_prefixes"}
        }
        payload["source_shards"] = [item.to_dict() for item in self.source_shards]
        payload["allowed_license_url_prefixes"] = list(self.allowed_license_url_prefixes)
        return payload


@dataclass(frozen=True)
class Phase4BFeatureSettings:
    model_name_or_path: str
    model_revision: str
    visionzip_config: str
    requested_layer_index: int
    batch_size: int
    shard_size: int
    storage_dtype: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BFeatureSettings":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B feature keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B feature keys: {missing}")
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("feature model_name_or_path must not be empty")
        if len(self.model_revision) != 40:
            raise ValueError("feature model_revision must be a 40-character commit hash")
        if not self.visionzip_config.strip():
            raise ValueError("visionzip_config must not be empty")
        if self.requested_layer_index >= 0:
            raise ValueError("requested_layer_index must use a negative hidden-layer offset")
        if self.batch_size <= 0 or self.shard_size <= 0:
            raise ValueError("feature batch_size and shard_size must be positive")
        if self.shard_size % self.batch_size != 0:
            raise ValueError("feature shard_size must be divisible by batch_size")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("feature storage_dtype must be float16 or float32")

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class Phase4BTrainingSettings:
    model_name_or_path: str
    budget: int
    projector_type: str
    vision_hidden_size: int
    learning_rate: float
    weight_decay: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    warmup_steps: int
    checkpoint_every: int
    evaluation_every: int
    keep_last_checkpoints: int
    generation_eval_samples: int
    max_caption_tokens: int
    max_new_tokens: int
    seed: int
    prompt: str
    generation_prompt: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BTrainingSettings":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B training keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B training keys: {missing}")
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("training model_name_or_path must not be empty")
        if self.budget not in {64, 128, 192}:
            raise ValueError("training budget must be 64, 128, or 192")
        if self.projector_type not in {"linear", "mlp2x_gelu"}:
            raise ValueError("training projector_type is invalid")
        if self.vision_hidden_size <= 0:
            raise ValueError("training vision_hidden_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer settings are invalid")
        positive = (
            "micro_batch_size", "gradient_accumulation_steps",
            "max_optimizer_steps", "checkpoint_every", "evaluation_every",
            "keep_last_checkpoints", "generation_eval_samples",
            "max_caption_tokens", "max_new_tokens",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"training {name} must be positive")
        if not 0 <= self.warmup_steps < self.max_optimizer_steps:
            raise ValueError("training warmup_steps must be within the run")
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")
        if not self.prompt or not self.generation_prompt:
            raise ValueError("training prompts must not be empty")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class Phase4BEvaluationSettings:
    primary_metrics: Tuple[str, ...]
    secondary_metrics: Tuple[str, ...]
    reference_type: str
    quality_claim_policy: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BEvaluationSettings":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B evaluation keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B evaluation keys: {missing}")
        settings = cls(
            primary_metrics=tuple(str(item) for item in payload["primary_metrics"]),
            secondary_metrics=tuple(str(item) for item in payload["secondary_metrics"]),
            reference_type=str(payload["reference_type"]),
            quality_claim_policy=str(payload["quality_claim_policy"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.primary_metrics:
            raise ValueError("at least one primary evaluation metric is required")
        if not self.reference_type.strip() or not self.quality_claim_policy.strip():
            raise ValueError("evaluation reference and claim policy must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_metrics": list(self.primary_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "reference_type": self.reference_type,
            "quality_claim_policy": self.quality_claim_policy,
        }


@dataclass(frozen=True)
class Phase4BConfig:
    artifact_type: str
    name: str
    dataset: Phase4BDatasetSettings
    features: Phase4BFeatureSettings
    training: Phase4BTrainingSettings
    evaluation: Phase4BEvaluationSettings

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4BConfig":
        allowed = {"artifact_type", "name", "dataset", "features", "training", "evaluation"}
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown:
            raise ValueError(f"Unknown Phase 4B config keys: {unknown}")
        if missing:
            raise ValueError(f"Missing Phase 4B config keys: {missing}")
        config = cls(
            artifact_type=str(payload["artifact_type"]),
            name=str(payload["name"]),
            dataset=Phase4BDatasetSettings.from_dict(payload["dataset"]),
            features=Phase4BFeatureSettings.from_dict(payload["features"]),
            training=Phase4BTrainingSettings.from_dict(payload["training"]),
            evaluation=Phase4BEvaluationSettings.from_dict(payload["evaluation"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.artifact_type != "phase4b_config_v1":
            raise ValueError("artifact_type must be phase4b_config_v1")
        if not self.name.strip():
            raise ValueError("Phase 4B config name must not be empty")
        if self.training.budget != 64:
            raise ValueError("the initial Phase 4B pilot is fixed to nominal budget 64")
        if self.training.vision_hidden_size != 1024:
            raise ValueError("CLIP ViT-L/14-336 features require vision_hidden_size=1024")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "features": self.features.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def load_phase4b_config(path: Union[str, Path]) -> Phase4BConfig:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return Phase4BConfig.from_dict(json.load(handle))
