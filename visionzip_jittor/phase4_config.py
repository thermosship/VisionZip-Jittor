"""Configuration for Phase 4A paired image-text Projector training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


@dataclass(frozen=True)
class Phase4AConfig:
    """Serializable runtime settings for the first real paired-data trainer."""

    model_name_or_path: str = "openai-community/gpt2"
    budget: int = 64
    projector_type: str = "mlp2x_gelu"
    vision_hidden_size: int = 1024
    learning_rate: float = 1e-4
    seed: int = 2026
    prompt: str = "Describe the image in a short phrase:"
    generation_prompt: str = " This image shows"
    validation_fraction: float = 1.0 / 3.0
    batch_size: int = 2
    max_steps: int = 30
    max_caption_tokens: int = 24
    max_new_tokens: int = 12
    log_every: int = 1
    checkpoint_every: int = 10
    minimum_loss_improvement: float = 0.0
    resume_atol: float = 1e-5
    verify_resume: bool = True

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty")
        if self.budget not in {64, 128, 192}:
            raise ValueError("budget must be one of 64, 128, or 192")
        if self.projector_type not in {"linear", "mlp2x_gelu"}:
            raise ValueError("projector_type must be linear or mlp2x_gelu")
        if not isinstance(self.vision_hidden_size, int) or self.vision_hidden_size <= 0:
            raise ValueError("vision_hidden_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.generation_prompt:
            raise ValueError("generation_prompt must not be empty")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within [0, 1)")
        for name in (
            "batch_size",
            "max_steps",
            "max_caption_tokens",
            "max_new_tokens",
            "log_every",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_loss_improvement < 0:
            raise ValueError("minimum_loss_improvement must be non-negative")
        if self.resume_atol < 0:
            raise ValueError("resume_atol must be non-negative")
        if not isinstance(self.verify_resume, bool):
            raise ValueError("verify_resume must be a boolean")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase4AConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Phase 4A config keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def load_phase4a_config(path: Union[str, Path]) -> Phase4AConfig:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return Phase4AConfig.from_dict(json.load(handle))
