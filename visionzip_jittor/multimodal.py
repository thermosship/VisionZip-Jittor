"""Minimal frozen-language integration for Phase 3A smoke validation.

The classes in this module intentionally do not implement a real language
model. They validate embedding packing, parameter freezing, and gradient flow
from a frozen output head back into the trainable multimodal Projector.
"""

from __future__ import annotations

from typing import Dict

import jittor as jt
from jittor import nn

from .projector import MultimodalProjector
from .projector_config import ProjectorConfig


class FrozenLanguageStub(nn.Module):
    """Frozen embedding + output head surrogate; explicitly not a real LLM."""

    def __init__(self, config: ProjectorConfig):
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.language_hidden_size,
        )
        self.lm_head = nn.Linear(
            config.language_hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.freeze_parameters()

    def freeze_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.stop_grad()

    def all_parameters_stop_grad(self) -> bool:
        parameters = list(self.parameters())
        return bool(parameters) and all(
            parameter.is_stop_grad() for parameter in parameters
        )

    def embed_tokens(self, token_ids: jt.Var) -> jt.Var:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        return self.token_embedding(token_ids)

    def execute(self, inputs_embeds: jt.Var) -> jt.Var:
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [B, T, D]")
        if int(inputs_embeds.shape[-1]) != self.config.language_hidden_size:
            raise ValueError("language hidden size mismatch")
        return self.lm_head(inputs_embeds)


def pack_multimodal_embeddings(
    prefix_embeddings: jt.Var,
    visual_embeddings: jt.Var,
    suffix_embeddings: jt.Var,
) -> jt.Var:
    """Concatenate prefix text, visual and suffix text embeddings."""

    tensors = (prefix_embeddings, visual_embeddings, suffix_embeddings)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("all embeddings must have shape [B, T, D]")
    batch_sizes = {int(tensor.shape[0]) for tensor in tensors}
    hidden_sizes = {int(tensor.shape[2]) for tensor in tensors}
    if len(batch_sizes) != 1:
        raise ValueError("embedding batch sizes must match")
    if len(hidden_sizes) != 1:
        raise ValueError("embedding hidden sizes must match")
    return jt.concat(list(tensors), dim=1)


class ProjectorFrozenLanguageBridge(nn.Module):
    """Trainable Projector connected to a frozen language surrogate."""

    def __init__(
        self,
        projector: MultimodalProjector,
        language_stub: FrozenLanguageStub,
    ):
        if projector.config.language_hidden_size != (
            language_stub.config.language_hidden_size
        ):
            raise ValueError("projector and language hidden sizes must match")
        self.projector = projector
        self.language_stub = language_stub

    def execute(
        self,
        visual_tokens: jt.Var,
        prefix_token_ids: jt.Var,
        suffix_token_ids: jt.Var,
    ) -> Dict[str, jt.Var]:
        projected_visual = self.projector(visual_tokens)
        prefix_embeddings = self.language_stub.embed_tokens(prefix_token_ids)
        suffix_embeddings = self.language_stub.embed_tokens(suffix_token_ids)
        packed_embeddings = pack_multimodal_embeddings(
            prefix_embeddings,
            projected_visual,
            suffix_embeddings,
        )
        logits = self.language_stub(packed_embeddings)
        return {
            "projected_visual": projected_visual,
            "packed_embeddings": packed_embeddings,
            "logits": logits,
        }
