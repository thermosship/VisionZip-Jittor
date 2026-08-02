"""Configuration for the Phase 3 native Jittor projector smoke path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


@dataclass(frozen=True)
class ProjectorConfig:
    """Small, serializable configuration for Projector + frozen stub tests."""

    projector_type: str = "mlp2x_gelu"
    vision_hidden_size: int = 1024
    language_hidden_size: int = 4096
    vocab_size: int = 128
    prefix_tokens: int = 8
    suffix_tokens: int = 8
    learning_rate: float = 1e-4
    seed: int = 2026

    def validate(self) -> None:
        if self.projector_type not in {"linear", "mlp2x_gelu"}:
            raise ValueError(
                "projector_type must be 'linear' or 'mlp2x_gelu', got "
                f"{self.projector_type!r}"
            )
        for name in (
            "vision_hidden_size",
            "language_hidden_size",
            "vocab_size",
            "prefix_tokens",
            "suffix_tokens",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"prefix_tokens", "suffix_tokens"} else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectorConfig":
        allowed = {
            "projector_type",
            "vision_hidden_size",
            "language_hidden_size",
            "vocab_size",
            "prefix_tokens",
            "suffix_tokens",
            "learning_rate",
            "seed",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown projector config keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projector_type": self.projector_type,
            "vision_hidden_size": self.vision_hidden_size,
            "language_hidden_size": self.language_hidden_size,
            "vocab_size": self.vocab_size,
            "prefix_tokens": self.prefix_tokens,
            "suffix_tokens": self.suffix_tokens,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


def load_projector_config(path: Union[str, Path]) -> ProjectorConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return ProjectorConfig.from_dict(json.load(handle))
