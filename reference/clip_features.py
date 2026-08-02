"""Helpers for extracting and naming real CLIP alignment artifacts."""

from __future__ import annotations

import re
from typing import Tuple

import torch


def resolve_layer_index(num_layers: int, requested: int) -> int:
    """Resolve a possibly-negative encoder-layer index."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    resolved = requested if requested >= 0 else num_layers + requested
    if resolved < 0 or resolved >= num_layers:
        raise IndexError(
            f"layer index {requested} resolves to {resolved}, but the model has "
            f"{num_layers} encoder layers"
        )
    return resolved


def key_projection_to_metric(
    key_projection: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Reproduce VisionZip's raw-key metric from CLIP's k_proj output.

    Hugging Face CLIP's k_proj returns [batch, sequence, embed_dim]. The
    official VisionZip attention patch reshapes this to
    [batch, heads, sequence, head_dim] and averages across heads.
    """

    if key_projection.ndim != 3:
        raise ValueError(
            "key_projection must have shape [batch, sequence, embed_dim]"
        )
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    batch, sequence, embed_dim = key_projection.shape
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}"
        )
    head_dim = embed_dim // num_heads
    keys = key_projection.reshape(batch, sequence, num_heads, head_dim)
    keys = keys.permute(0, 2, 1, 3)
    return keys.mean(dim=1)


def contextual_target_original_indices(
    remaining_indices: torch.Tensor, target_positions: torch.Tensor
) -> torch.Tensor:
    """Map contextual target positions back to original CLIP token indices."""

    if remaining_indices.ndim != 2 or target_positions.ndim != 2:
        raise ValueError("remaining_indices and target_positions must be rank-2")
    if remaining_indices.shape[0] != target_positions.shape[0]:
        raise ValueError("batch dimensions must match")
    return torch.gather(remaining_indices, dim=1, index=target_positions.long())


def merge_original_indices(
    remaining_indices: torch.Tensor, merge_positions: torch.Tensor
) -> torch.Tensor:
    """Map merge positions back to original CLIP token indices."""

    if remaining_indices.ndim != 2 or merge_positions.ndim != 2:
        raise ValueError("remaining_indices and merge_positions must be rank-2")
    if remaining_indices.shape[0] != merge_positions.shape[0]:
        raise ValueError("batch dimensions must match")
    return torch.gather(remaining_indices, dim=1, index=merge_positions.long())


def infer_patch_grid(sequence_length: int, include_cls: bool = True) -> Tuple[int, int]:
    """Infer a square CLIP patch grid from a sequence length."""

    patch_count = sequence_length - (1 if include_cls else 0)
    if patch_count <= 0:
        raise ValueError("sequence length does not contain any patch tokens")
    side = int(round(patch_count ** 0.5))
    if side * side != patch_count:
        raise ValueError(
            f"cannot infer a square patch grid from {patch_count} patch tokens"
        )
    return side, side


def sanitize_artifact_name(value: str) -> str:
    """Convert a config/model label into a stable filename fragment."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "artifact"


def real_reference_filename(config_name: str, dtype: str) -> str:
    return f"reference_{sanitize_artifact_name(config_name)}_{dtype}_real_clip.npz"
