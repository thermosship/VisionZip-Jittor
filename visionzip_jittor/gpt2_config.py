"""Configuration helpers for the Phase 3B native Jittor GPT-2 path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


@dataclass(frozen=True)
class GPT2Config:
    """Subset of Hugging Face GPT-2 configuration used by this project."""

    model_type: str = "gpt2"
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"
    bos_token_id: int = 50256
    eos_token_id: int = 50256

    def validate(self) -> None:
        if self.model_type != "gpt2":
            raise ValueError(f"model_type must be 'gpt2', got {self.model_type!r}")
        for name in ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.activation_function not in {"gelu_new", "gelu_fast"}:
            raise ValueError(
                "Phase 3B supports GPT-2 gelu_new/gelu_fast only, got "
                f"{self.activation_function!r}"
            )
        for name in ("bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be within the vocabulary")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GPT2Config":
        aliases = {
            "hidden_size": "n_embd",
            "num_hidden_layers": "n_layer",
            "num_attention_heads": "n_head",
            "max_position_embeddings": "n_positions",
        }
        normalized = dict(payload)
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in normalized.items() if key in allowed}
        config = cls(**filtered)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "activation_function": self.activation_function,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }


@dataclass(frozen=True)
class Phase3BConfig:
    """Runtime and acceptance settings for the real frozen GPT-2 smoke test."""

    model_name_or_path: str = "openai-community/gpt2"
    projector_type: str = "mlp2x_gelu"
    vision_hidden_size: int = 1024
    learning_rate: float = 1e-4
    seed: int = 2026
    prompt: str = "Describe the image in a short phrase:"
    targets: tuple[str, ...] = (
        " dense geometric pattern",
        " simple outdoor scene",
        " printed text",
    )
    generation_prompt: str = "This image shows"
    max_new_tokens: int = 8
    warmup: int = 2
    iterations: int = 5
    logit_atol: float = 5e-4
    logit_rtol: float = 5e-4

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty")
        if self.projector_type not in {"linear", "mlp2x_gelu"}:
            raise ValueError("projector_type must be linear or mlp2x_gelu")
        if self.vision_hidden_size <= 0:
            raise ValueError("vision_hidden_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.targets or any(not item for item in self.targets):
            raise ValueError("targets must contain non-empty strings")
        if not self.generation_prompt:
            raise ValueError("generation_prompt must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("warmup must be >= 0 and iterations must be > 0")
        if self.logit_atol < 0 or self.logit_rtol < 0:
            raise ValueError("logit tolerances must be non-negative")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Phase3BConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Phase 3B config keys: {unknown}")
        normalized = dict(payload)
        if "targets" in normalized:
            normalized["targets"] = tuple(normalized["targets"])
        config = cls(**normalized)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "projector_type": self.projector_type,
            "vision_hidden_size": self.vision_hidden_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "prompt": self.prompt,
            "targets": list(self.targets),
            "generation_prompt": self.generation_prompt,
            "max_new_tokens": self.max_new_tokens,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "logit_atol": self.logit_atol,
            "logit_rtol": self.logit_rtol,
        }


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_gpt2_config(path: Union[str, Path]) -> GPT2Config:
    return GPT2Config.from_dict(_load_json(path))


def load_phase3b_config(path: Union[str, Path]) -> Phase3BConfig:
    return Phase3BConfig.from_dict(_load_json(path))
