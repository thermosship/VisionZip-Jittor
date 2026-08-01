"""PyTorch reference implementation pinned to the official VisionZip CLIP path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from visionzip_jittor.config import VisionZipConfig


@dataclass
class VisionZipTorchOutput:
    compressed_tokens: torch.Tensor
    selected_indices: torch.Tensor
    dominant_ordered_indices: torch.Tensor
    remaining_indices: torch.Tensor
    target_positions: torch.Tensor
    merge_positions: torch.Tensor
    assignments: torch.Tensor
    assignment_counts: torch.Tensor
    contextual_tokens: torch.Tensor
    cls_attention_sum: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }


def _validate_inputs(
    hidden_states: torch.Tensor,
    attentions: torch.Tensor,
    metric: torch.Tensor,
    config: VisionZipConfig,
    cls_index: int,
) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, N, C]")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape [B, H, N, N]")
    if metric.ndim != 3:
        raise ValueError("metric must have shape [B, N, D]")
    batch, sequence_length, _ = hidden_states.shape
    if attentions.shape[0] != batch or metric.shape[0] != batch:
        raise ValueError("Batch dimensions do not match")
    if attentions.shape[2:] != (sequence_length, sequence_length):
        raise ValueError("Attention sequence dimensions do not match hidden_states")
    if metric.shape[1] != sequence_length:
        raise ValueError("Metric sequence dimension does not match hidden_states")
    if cls_index != 0:
        raise NotImplementedError("The exact CLIP path currently requires cls_index=0")
    config.validate(sequence_length)


def visionzip_compress_torch(
    hidden_states: torch.Tensor,
    attentions: torch.Tensor,
    metric: torch.Tensor,
    config: VisionZipConfig,
    cls_index: int = 0,
) -> VisionZipTorchOutput:
    """Execute VisionZip and retain all alignment-critical intermediates.

    ``code_exact`` reproduces the official repository's CLIP implementation:
    contextual targets are added to the mean of their assigned merge tokens.
    ``paper_avg`` averages the target together with all assigned tokens and is
    provided only as a documented ablation.
    """

    _validate_inputs(hidden_states, attentions, metric, config, cls_index)
    batch, sequence_length, hidden_width = hidden_states.shape

    cls_attention = attentions[:, :, cls_index, cls_index + 1 :]
    cls_attention_sum = cls_attention.sum(dim=1)
    topk_patch = cls_attention_sum.topk(
        config.dominant_tokens, dim=1, largest=True, sorted=True
    ).indices + (cls_index + 1)

    if config.include_cls:
        cls = torch.zeros(
            (batch, 1), dtype=topk_patch.dtype, device=topk_patch.device
        )
        selected_indices = torch.cat((cls, topk_patch), dim=1)
    else:
        selected_indices = topk_patch

    mask = torch.ones(
        (batch, sequence_length), dtype=torch.bool, device=hidden_states.device
    )
    mask.scatter_(1, selected_indices, False)

    # masked_select follows the original token order, not top-k score order.
    selected_count = selected_indices.shape[1]
    dominant_tokens = hidden_states.masked_select((~mask).unsqueeze(-1)).view(
        batch, selected_count, hidden_width
    )
    dominant_ordered_indices = torch.arange(
        sequence_length, device=hidden_states.device
    ).expand(batch, -1)[~mask].view(batch, selected_count)

    remaining_count = sequence_length - selected_count
    remaining_indices = torch.arange(
        sequence_length, device=hidden_states.device
    ).expand(batch, -1)[mask].view(batch, remaining_count)
    hidden_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(
        batch, remaining_count, hidden_width
    )
    metric_filtered = metric.masked_select(mask.unsqueeze(-1)).view(
        batch, remaining_count, metric.shape[-1]
    )

    norm = metric_filtered.norm(dim=-1, keepdim=True)
    if config.normalization_eps > 0:
        norm = norm.clamp_min(config.normalization_eps)
    metric_normalized = metric_filtered / norm

    step = max(1, remaining_count // config.contextual_tokens)
    target_1d = torch.arange(
        0, remaining_count, step, device=hidden_states.device
    )[: config.contextual_tokens]
    if target_1d.numel() != config.contextual_tokens:
        raise ValueError(
            "Unable to create the requested number of contextual targets"
        )
    target_positions = target_1d.expand(batch, -1)

    position_ids = torch.arange(remaining_count, device=hidden_states.device)
    merge_mask = ~torch.isin(position_ids, target_1d)
    merge_1d = position_ids[merge_mask]
    merge_positions = merge_1d.expand(batch, -1)

    target_metric = metric_normalized[:, target_1d, :]
    merge_metric = metric_normalized[:, merge_1d, :]
    similarity = torch.bmm(merge_metric, target_metric.transpose(1, 2))
    assignments = similarity.argmax(dim=2)

    one_hot = torch.zeros(
        batch,
        merge_1d.numel(),
        config.contextual_tokens,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    one_hot.scatter_(2, assignments.unsqueeze(-1), 1)
    raw_counts = one_hot.sum(dim=1)

    merge_hidden = hidden_filtered[:, merge_1d, :]
    aggregated_sum = torch.bmm(one_hot.transpose(1, 2), merge_hidden)
    target_hidden = hidden_filtered[:, target_1d, :]

    if config.merge_mode == "code_exact":
        contextual_tokens = target_hidden + aggregated_sum / raw_counts.clamp_min(
            1
        ).unsqueeze(-1)
    else:
        contextual_tokens = (target_hidden + aggregated_sum) / (
            raw_counts + 1
        ).unsqueeze(-1)

    compressed_tokens = torch.cat((dominant_tokens, contextual_tokens), dim=1)

    return VisionZipTorchOutput(
        compressed_tokens=compressed_tokens,
        selected_indices=selected_indices,
        dominant_ordered_indices=dominant_ordered_indices,
        remaining_indices=remaining_indices,
        target_positions=target_positions,
        merge_positions=merge_positions,
        assignments=assignments,
        assignment_counts=raw_counts,
        contextual_tokens=contextual_tokens,
        cls_attention_sum=cls_attention_sum,
    )
