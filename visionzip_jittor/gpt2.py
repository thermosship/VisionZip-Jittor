"""Native Jittor GPT-2 used by the Phase 3B real frozen-LLM path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jittor as jt
import numpy as np
from jittor import nn

from .gpt2_config import GPT2Config


class HFConv1D(nn.Module):
    """Hugging Face GPT-2 Conv1D with weight layout [input, output]."""

    def __init__(self, input_size: int, output_size: int):
        self.weight = jt.zeros((input_size, output_size), dtype="float32")
        self.bias = jt.zeros((output_size,), dtype="float32")

    def execute(self, x: jt.Var) -> jt.Var:
        original_shape = tuple(int(value) for value in x.shape)
        flat = x.reshape((-1, original_shape[-1]))
        output = jt.matmul(flat, self.weight) + self.bias
        return output.reshape(original_shape[:-1] + (int(self.bias.shape[0]),))


class GPT2LayerNorm(nn.Module):
    """LayerNorm with explicit GPT-2 epsilon and affine parameters."""

    def __init__(self, hidden_size: int, epsilon: float):
        self.weight = jt.ones((hidden_size,), dtype="float32")
        self.bias = jt.zeros((hidden_size,), dtype="float32")
        self.epsilon = float(epsilon)

    def execute(self, x: jt.Var) -> jt.Var:
        mean = x.mean(dim=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdims=True)
        normalized = centered / jt.sqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


def gelu_new(x: jt.Var) -> jt.Var:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + jt.tanh(coefficient * (x + 0.044715 * x * x * x)))


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.c_attn = HFConv1D(config.n_embd, 3 * config.n_embd)
        self.c_proj = HFConv1D(config.n_embd, config.n_embd)

    def _split_heads(self, x: jt.Var) -> jt.Var:
        batch, tokens, _ = (int(value) for value in x.shape)
        return x.reshape(
            (batch, tokens, self.num_heads, self.head_dim)
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x: jt.Var) -> jt.Var:
        batch, _, tokens, _ = (int(value) for value in x.shape)
        return x.permute(0, 2, 1, 3).reshape(
            (batch, tokens, self.hidden_size)
        )

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        batch, tokens, _ = (int(value) for value in hidden_states.shape)
        qkv = self.c_attn(hidden_states)
        query = qkv[:, :, : self.hidden_size]
        key = qkv[:, :, self.hidden_size : 2 * self.hidden_size]
        value = qkv[:, :, 2 * self.hidden_size :]

        query = self._split_heads(query).reshape(
            (batch * self.num_heads, tokens, self.head_dim)
        )
        key = self._split_heads(key).reshape(
            (batch * self.num_heads, tokens, self.head_dim)
        )
        value = self._split_heads(value).reshape(
            (batch * self.num_heads, tokens, self.head_dim)
        )
        scores = nn.bmm_transpose(query, key) * self.scale
        scores = scores.reshape((batch, self.num_heads, tokens, tokens))

        causal_np = np.tril(np.ones((tokens, tokens), dtype=np.float32))
        causal = jt.array(causal_np).reshape((1, 1, tokens, tokens))
        scores = scores + (1.0 - causal) * -10000.0
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape [B, T]")
            if tuple(int(value) for value in attention_mask.shape) != (
                batch,
                tokens,
            ):
                raise ValueError("attention_mask shape mismatch")
            key_mask = attention_mask.reshape((batch, 1, 1, tokens))
            scores = scores + (1.0 - key_mask) * -10000.0

        probabilities = nn.softmax(scores, dim=-1).reshape(
            (batch * self.num_heads, tokens, tokens)
        )
        context = nn.bmm(probabilities, value).reshape(
            (batch, self.num_heads, tokens, self.head_dim)
        )
        return self.c_proj(self._merge_heads(context))


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        intermediate = 4 * config.n_embd
        self.c_fc = HFConv1D(config.n_embd, intermediate)
        self.c_proj = HFConv1D(intermediate, config.n_embd)

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        return self.c_proj(gelu_new(self.c_fc(hidden_states)))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        self.ln_1 = GPT2LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = GPT2LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states), attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class NativeGPT2LMHeadModel(nn.Module):
    """GPT-2 LM head model with tied embeddings and frozen-weight helpers."""

    def __init__(self, config: GPT2Config):
        config.validate()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.block_names = []
        for index in range(config.n_layer):
            name = f"block_{index}"
            setattr(self, name, GPT2Block(config))
            self.block_names.append(name)
        self.ln_f = GPT2LayerNorm(config.n_embd, config.layer_norm_epsilon)

    def blocks(self) -> List[GPT2Block]:
        return [getattr(self, name) for name in self.block_names]

    def embed_tokens(self, token_ids: jt.Var) -> jt.Var:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        return self.wte(token_ids)

    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [B, T, D]")
        batch, tokens, hidden = (int(value) for value in inputs_embeds.shape)
        if hidden != self.config.n_embd:
            raise ValueError("GPT-2 hidden size mismatch")
        if tokens > self.config.n_positions:
            raise ValueError(
                f"sequence length {tokens} exceeds {self.config.n_positions}"
            )
        position_ids = jt.arange(tokens).reshape((1, tokens)).broadcast(
            (batch, tokens)
        )
        hidden_states = inputs_embeds + self.wpe(position_ids)
        for block in self.blocks():
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.ln_f(hidden_states)
        flat = hidden_states.reshape((-1, self.config.n_embd))
        logits = jt.matmul(flat, self.wte.weight.transpose(0, 1))
        return logits.reshape((batch, tokens, self.config.vocab_size))

    def weight_targets(self) -> Dict[str, jt.Var]:
        targets: Dict[str, jt.Var] = {
            "transformer.wte.weight": self.wte.weight,
            "transformer.wpe.weight": self.wpe.weight,
            "transformer.ln_f.weight": self.ln_f.weight,
            "transformer.ln_f.bias": self.ln_f.bias,
        }
        for index, block in enumerate(self.blocks()):
            prefix = f"transformer.h.{index}"
            targets.update(
                {
                    f"{prefix}.ln_1.weight": block.ln_1.weight,
                    f"{prefix}.ln_1.bias": block.ln_1.bias,
                    f"{prefix}.attn.c_attn.weight": block.attn.c_attn.weight,
                    f"{prefix}.attn.c_attn.bias": block.attn.c_attn.bias,
                    f"{prefix}.attn.c_proj.weight": block.attn.c_proj.weight,
                    f"{prefix}.attn.c_proj.bias": block.attn.c_proj.bias,
                    f"{prefix}.ln_2.weight": block.ln_2.weight,
                    f"{prefix}.ln_2.bias": block.ln_2.bias,
                    f"{prefix}.mlp.c_fc.weight": block.mlp.c_fc.weight,
                    f"{prefix}.mlp.c_fc.bias": block.mlp.c_fc.bias,
                    f"{prefix}.mlp.c_proj.weight": block.mlp.c_proj.weight,
                    f"{prefix}.mlp.c_proj.bias": block.mlp.c_proj.bias,
                }
            )
        return targets

    def load_npz_weights(self, path: Path | str) -> Dict[str, object]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing GPT-2 weights: {path}")
        targets = self.weight_targets()
        loaded: List[str] = []
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(targets) - set(archive.files))
            unexpected = sorted(set(archive.files) - set(targets))
            if missing:
                raise ValueError(f"Missing GPT-2 weight tensors: {missing}")
            if unexpected:
                raise ValueError(f"Unexpected GPT-2 weight tensors: {unexpected}")
            for name, target in targets.items():
                value = archive[name].astype(np.float32, copy=False)
                expected_shape = tuple(int(item) for item in target.shape)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"Weight shape mismatch for {name}: "
                        f"expected {expected_shape}, got {value.shape}"
                    )
                target.assign(value)
                loaded.append(name)
        return {"tensor_count": len(loaded), "names": loaded}

    def freeze_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.stop_grad()

    def all_parameters_stop_grad(self) -> bool:
        parameters = list(self.parameters())
        return bool(parameters) and all(
            parameter.is_stop_grad() for parameter in parameters
        )


def parameter_count(parameters: Iterable[jt.Var]) -> int:
    total = 0
    for parameter in parameters:
        size = 1
        for dimension in parameter.shape:
            size *= int(dimension)
        total += size
    return total


def masked_causal_language_loss(
    logits: jt.Var,
    labels: jt.Var,
    label_mask: jt.Var,
) -> jt.Var:
    """Mean next-token NLL over positions selected by label_mask."""

    if logits.ndim != 3 or labels.ndim != 2 or label_mask.ndim != 2:
        raise ValueError("invalid language-loss tensor rank")
    batch, tokens, _ = (int(value) for value in logits.shape)
    if tuple(int(value) for value in labels.shape) != (batch, tokens):
        raise ValueError("labels shape mismatch")
    if tuple(int(value) for value in label_mask.shape) != (batch, tokens):
        raise ValueError("label_mask shape mismatch")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = label_mask[:, 1:]
    log_probs = nn.log_softmax(shift_logits, dim=-1)
    selected = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    denominator = shift_mask.sum() + 1e-12
    return -(selected * shift_mask).sum() / denominator


def greedy_generate_from_embeddings(
    model: NativeGPT2LMHeadModel,
    initial_embeddings: jt.Var,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Greedy decode from pre-packed text/visual embeddings without KV cache."""

    if int(initial_embeddings.shape[0]) != 1:
        raise ValueError("generation currently requires batch size 1")
    embeddings = initial_embeddings
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(inputs_embeds=embeddings)
        next_ids, _ = logits[:, -1, :].argmax(dim=-1)
        next_token = int(next_ids.numpy()[0])
        generated.append(next_token)
        token_var = jt.array(np.asarray([[next_token]], dtype=np.int32))
        embeddings = jt.concat([embeddings, model.embed_tokens(token_var)], dim=1)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return generated
