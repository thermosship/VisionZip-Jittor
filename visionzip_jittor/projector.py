"""Native Jittor multimodal projectors used by Phase 3."""

from __future__ import annotations

import jittor as jt
from jittor import nn

from .projector_config import ProjectorConfig


class MultimodalProjector(nn.Module):
    """Map compressed CLIP tokens into the language hidden dimension."""

    def __init__(self, config: ProjectorConfig):
        config.validate()
        self.config = config
        if config.projector_type == "linear":
            self.linear = nn.Linear(
                config.vision_hidden_size,
                config.language_hidden_size,
            )
            self.linear_1 = None
            self.linear_2 = None
        else:
            self.linear = None
            self.linear_1 = nn.Linear(
                config.vision_hidden_size,
                config.language_hidden_size,
            )
            self.linear_2 = nn.Linear(
                config.language_hidden_size,
                config.language_hidden_size,
            )

    def execute(self, visual_tokens: jt.Var) -> jt.Var:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, D]")
        if int(visual_tokens.shape[-1]) != self.config.vision_hidden_size:
            raise ValueError(
                "visual hidden size mismatch: expected "
                f"{self.config.vision_hidden_size}, got "
                f"{int(visual_tokens.shape[-1])}"
            )
        if self.linear is not None:
            return self.linear(visual_tokens)
        hidden = nn.gelu(self.linear_1(visual_tokens))
        return self.linear_2(hidden)


def parameter_count(module: nn.Module) -> int:
    """Return the number of scalar parameters registered by a module."""

    total = 0
    for parameter in module.parameters():
        size = 1
        for dimension in parameter.shape:
            size *= int(dimension)
        total += size
    return total
