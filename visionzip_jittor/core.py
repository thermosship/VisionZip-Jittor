"""Native Jittor implementation of VisionZip's CLIP token compression.

This module intentionally uses Jittor tensor operations only. It mirrors the
official PyTorch CLIP code path pinned in ``references/UPSTREAM.md``.
"""

from __future__ import annotations

from typing import Dict

import jittor as jt
from jittor import nn

from .config import VisionZipConfig


def _batch_gather(x: jt.Var, indices: jt.Var) -> jt.Var:
    """Gather ``[B, K]`` token indices from a ``[B, N, D]`` tensor."""

    batch, count = int(indices.shape[0]), int(indices.shape[1])
    width = int(x.shape[2])
    expanded = indices.unsqueeze(-1).broadcast((batch, count, width))
    return x.gather(1, expanded)


def _ascending_values(values: jt.Var) -> jt.Var:
    """Sort each row of unique non-negative integer values ascending.

    Jittor ``topk`` is used instead of relying on boolean masked-select order.
    The official PyTorch implementation selects by a mask, which emits tokens
    in original sequence order rather than attention-score order.
    """

    count = int(values.shape[1])
    _, order = jt.topk(-values, count, dim=1, largest=True, sorted=True)
    return values.gather(1, order)


def _remaining_indices(selected_indices: jt.Var, sequence_length: int) -> jt.Var:
    """Return the complement of selected indices in original token order."""

    batch, selected_count = (
        int(selected_indices.shape[0]),
        int(selected_indices.shape[1]),
    )
    remaining_count = sequence_length - selected_count
    token_ids = jt.arange(sequence_length).reshape((1, sequence_length))
    token_ids = token_ids.broadcast((batch, sequence_length))
    comparisons = token_ids.unsqueeze(-1) == selected_indices.unsqueeze(1)
    is_selected = comparisons.sum(dim=2) > 0

    # Every remaining token gets a positive priority and lower original token
    # ids get higher priority. Selected tokens get non-positive priority.
    priorities = (is_selected == 0).cast("int32") * (sequence_length + 1)
    priorities = priorities - token_ids
    _, indices = jt.topk(
        priorities,
        remaining_count,
        dim=1,
        largest=True,
        sorted=True,
    )
    return indices


def _target_and_merge_positions(
    remaining_count: int, contextual_tokens: int, batch: int
) -> tuple[jt.Var, jt.Var]:
    step = max(1, remaining_count // contextual_tokens)
    target_list = list(range(0, remaining_count, step))[:contextual_tokens]
    if len(target_list) != contextual_tokens:
        raise ValueError(
            "Unable to create the requested number of contextual targets: "
            f"remaining={remaining_count}, contextual={contextual_tokens}"
        )

    target_1d = jt.array(target_list, dtype="int32")
    positions = jt.arange(remaining_count)
    is_target = (positions.unsqueeze(-1) == target_1d.unsqueeze(0)).sum(dim=1) > 0
    priorities = (is_target == 0).cast("int32") * (remaining_count + 1)
    priorities = priorities - positions
    _, merge_1d = jt.topk(
        priorities,
        remaining_count - contextual_tokens,
        dim=0,
        largest=True,
        sorted=True,
    )

    targets = target_1d.reshape((1, contextual_tokens)).broadcast(
        (batch, contextual_tokens)
    )
    merge = merge_1d.reshape((1, remaining_count - contextual_tokens)).broadcast(
        (batch, remaining_count - contextual_tokens)
    )
    return targets, merge


def visionzip_compress(
    hidden_states: jt.Var,
    attentions: jt.Var,
    metric: jt.Var,
    config: VisionZipConfig,
    cls_index: int = 0,
) -> Dict[str, jt.Var]:
    """Compress CLIP tokens with the official VisionZip algorithm.

    Args:
        hidden_states: CLIP hidden states ``[B, N, C]``.
        attentions: Multi-head attention probabilities ``[B, H, N, N]``.
        metric: Mean-head key features ``[B, N, D]``.
        config: Token budget and merge semantics.
        cls_index: CLIP CLS position. The exact official path uses zero.

    Returns:
        A dictionary containing compressed tokens and intermediate tensors used
        for PyTorch/Jittor numerical alignment.
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, N, C]")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape [B, H, N, N]")
    if metric.ndim != 3:
        raise ValueError("metric must have shape [B, N, D]")

    batch = int(hidden_states.shape[0])
    sequence_length = int(hidden_states.shape[1])
    if int(attentions.shape[0]) != batch or int(metric.shape[0]) != batch:
        raise ValueError("Batch dimensions do not match")
    if (
        int(attentions.shape[2]) != sequence_length
        or int(attentions.shape[3]) != sequence_length
    ):
        raise ValueError("Attention sequence dimensions do not match hidden_states")
    if int(metric.shape[1]) != sequence_length:
        raise ValueError("Metric sequence dimension does not match hidden_states")
    if cls_index != 0:
        raise NotImplementedError("The exact CLIP path currently requires cls_index=0")

    config.validate(sequence_length)
    cls_count = 1 if config.include_cls else 0

    cls_attention = attentions[:, :, cls_index, cls_index + 1 :]
    cls_attention_sum = cls_attention.sum(dim=1)
    _, topk_patch = jt.topk(
        cls_attention_sum,
        config.dominant_tokens,
        dim=1,
        largest=True,
        sorted=True,
    )
    topk_patch = topk_patch + (cls_index + 1)

    if config.include_cls:
        cls = jt.zeros((batch, 1), dtype=topk_patch.dtype)
        selected_indices = jt.concat((cls, topk_patch), dim=1)
    else:
        selected_indices = topk_patch

    dominant_ordered_indices = _ascending_values(selected_indices)
    dominant_tokens = _batch_gather(hidden_states, dominant_ordered_indices)

    remaining_indices = _remaining_indices(selected_indices, sequence_length)
    hidden_filtered = _batch_gather(hidden_states, remaining_indices)
    metric_filtered = _batch_gather(metric, remaining_indices)

    norm = (metric_filtered * metric_filtered).sum(dim=-1, keepdims=True).sqrt()
    if config.normalization_eps > 0:
        norm = norm.maximum(config.normalization_eps)
    metric_normalized = metric_filtered / norm

    remaining_count = int(remaining_indices.shape[1])
    target_positions, merge_positions = _target_and_merge_positions(
        remaining_count, config.contextual_tokens, batch
    )
    target_metric = _batch_gather(metric_normalized, target_positions)
    merge_metric = _batch_gather(metric_normalized, merge_positions)
    similarity = nn.bmm(merge_metric, target_metric.transpose(1, 2))
    assignments, _ = similarity.argmax(dim=2)

    context_ids = jt.arange(config.contextual_tokens).reshape(
        (1, 1, config.contextual_tokens)
    )
    context_ids = context_ids.broadcast(
        (batch, int(assignments.shape[1]), config.contextual_tokens)
    )
    one_hot = (assignments.unsqueeze(-1) == context_ids).cast(hidden_states.dtype)

    raw_counts = one_hot.sum(dim=1)
    merge_hidden = _batch_gather(hidden_filtered, merge_positions)
    aggregated_sum = nn.bmm(one_hot.transpose(1, 2), merge_hidden)
    target_hidden = _batch_gather(hidden_filtered, target_positions)

    if config.merge_mode == "code_exact":
        counts = raw_counts.maximum(1.0)
        contextual_tokens = target_hidden + aggregated_sum / counts.unsqueeze(-1)
    else:
        counts = raw_counts + 1.0
        contextual_tokens = (target_hidden + aggregated_sum) / counts.unsqueeze(-1)

    compressed_tokens = jt.concat((dominant_tokens, contextual_tokens), dim=1)

    return {
        "compressed_tokens": compressed_tokens,
        "selected_indices": selected_indices,
        "dominant_ordered_indices": dominant_ordered_indices,
        "remaining_indices": remaining_indices,
        "target_positions": target_positions,
        "merge_positions": merge_positions,
        "assignments": assignments,
        "assignment_counts": raw_counts,
        "contextual_tokens": contextual_tokens,
        "cls_attention_sum": cls_attention_sum,
    }


class VisionZip(nn.Module):
    """Small ``nn.Module`` wrapper around :func:`visionzip_compress`."""

    def __init__(self, config: VisionZipConfig):
        super().__init__()
        config.validate()
        self.config = config

    def execute(
        self, hidden_states: jt.Var, attentions: jt.Var, metric: jt.Var
    ) -> Dict[str, jt.Var]:
        return visionzip_compress(hidden_states, attentions, metric, self.config)
