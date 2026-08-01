"""Configuration objects shared by the PyTorch and Jittor implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class VisionZipConfig:
    """Core VisionZip token-budget configuration.

    ``dominant_tokens`` counts selected image patch tokens and does not count
    CLIP's CLS token. The official CLIP implementation prepends CLS to the
    returned dominant tokens, so ``actual_output_tokens`` is one larger than
    the nominal visual-token budget when ``include_cls`` is true.
    """

    dominant_tokens: int
    contextual_tokens: int
    merge_mode: str = "code_exact"
    include_cls: bool = True
    normalization_eps: float = 0.0
    name: str = "custom"
    source: str = "custom"

    def validate(self, sequence_length: Optional[int] = None) -> None:
        if self.dominant_tokens < 0:
            raise ValueError("dominant_tokens must be non-negative")
        if self.contextual_tokens <= 0:
            raise ValueError("contextual_tokens must be positive")
        if self.merge_mode not in {"code_exact", "paper_avg"}:
            raise ValueError(
                "merge_mode must be either 'code_exact' or 'paper_avg'"
            )
        if self.normalization_eps < 0:
            raise ValueError("normalization_eps must be non-negative")
        if sequence_length is not None:
            cls_count = 1 if self.include_cls else 0
            selected = self.dominant_tokens + cls_count
            remaining = sequence_length - selected
            if remaining < self.contextual_tokens:
                raise ValueError(
                    "Not enough non-dominant tokens for contextual targets: "
                    f"remaining={remaining}, contextual={self.contextual_tokens}"
                )

    @property
    def nominal_visual_tokens(self) -> int:
        """Reported patch/context token budget, excluding CLIP CLS."""

        return self.dominant_tokens + self.contextual_tokens

    @property
    def actual_output_tokens(self) -> int:
        """Actual output sequence length produced by the CLIP code path."""

        return self.nominal_visual_tokens + (1 if self.include_cls else 0)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisionZipConfig":
        allowed = {
            "dominant_tokens",
            "contextual_tokens",
            "merge_mode",
            "include_cls",
            "normalization_eps",
            "name",
            "source",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown VisionZip config keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dominant_tokens": self.dominant_tokens,
            "contextual_tokens": self.contextual_tokens,
            "merge_mode": self.merge_mode,
            "include_cls": self.include_cls,
            "normalization_eps": self.normalization_eps,
            "source": self.source,
        }


def load_config(path: Union[str, Path]) -> VisionZipConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return VisionZipConfig.from_dict(json.load(handle))
