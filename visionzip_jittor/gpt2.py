"""Native Jittor GPT-2 used by the Phase 3B real frozen-LLM path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import jittor as jt
import numpy as np
from jittor import nn

from .gpt2_config import GPT2Config


KeyValueCache = Tuple[jt.Var, jt.Var]
PastKeyValues = List[KeyValueCache]


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

    def forward_with_cache(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
        past_key_value: Optional[KeyValueCache] = None,
        use_cache: bool = True,
    ) -> Tuple[jt.Var, Optional[KeyValueCache]]:
        batch, query_tokens, _ = (
            int(value) for value in hidden_states.shape
        )
        qkv = self.c_attn(hidden_states)
        query = self._split_heads(qkv[:, :, : self.hidden_size])
        key = self._split_heads(
            qkv[:, :, self.hidden_size : 2 * self.hidden_size]
        )
        value = self._split_heads(qkv[:, :, 2 * self.hidden_size :])

        past_tokens = 0
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError("past_key_value must contain key and value")
            past_key, past_value = past_key_value
            expected_prefix = (batch, self.num_heads)
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("cached key/value must have shape [B, H, T, D]")
            if tuple(int(item) for item in past_key.shape[:2]) != expected_prefix:
                raise ValueError("cached key batch/head shape mismatch")
            if tuple(int(item) for item in past_value.shape[:2]) != expected_prefix:
                raise ValueError("cached value batch/head shape mismatch")
            if int(past_key.shape[3]) != self.head_dim:
                raise ValueError("cached key head dimension mismatch")
            if int(past_value.shape[3]) != self.head_dim:
                raise ValueError("cached value head dimension mismatch")
            if int(past_key.shape[2]) != int(past_value.shape[2]):
                raise ValueError("cached key/value lengths must match")
            past_tokens = int(past_key.shape[2])
            key = jt.concat([past_key, key], dim=2)
            value = jt.concat([past_value, value], dim=2)

        total_tokens = past_tokens + query_tokens
        query_flat = query.reshape(
            (batch * self.num_heads, query_tokens, self.head_dim)
        )
        key_flat = key.reshape(
            (batch * self.num_heads, total_tokens, self.head_dim)
        )
        value_flat = value.reshape(
            (batch * self.num_heads, total_tokens, self.head_dim)
        )
        scores = nn.bmm_transpose(query_flat, key_flat) * self.scale
        scores = scores.reshape(
            (batch, self.num_heads, query_tokens, total_tokens)
        )

        query_positions = np.arange(
            past_tokens,
            total_tokens,
            dtype=np.int64,
        ).reshape((query_tokens, 1))
        key_positions = np.arange(total_tokens, dtype=np.int64).reshape(
            (1, total_tokens)
        )
        causal_np = (key_positions <= query_positions).astype(np.float32)
        causal = jt.array(causal_np).reshape(
            (1, 1, query_tokens, total_tokens)
        )
        scores = scores + (1.0 - causal) * -10000.0
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape [B, T]")
            if tuple(int(item) for item in attention_mask.shape) != (
                batch,
                total_tokens,
            ):
                raise ValueError("attention_mask shape mismatch")
            key_mask = attention_mask.reshape((batch, 1, 1, total_tokens))
            scores = scores + (1.0 - key_mask) * -10000.0

        probabilities = nn.softmax(scores, dim=-1).reshape(
            (batch * self.num_heads, query_tokens, total_tokens)
        )
        context = nn.bmm(probabilities, value_flat).reshape(
            (batch, self.num_heads, query_tokens, self.head_dim)
        )
        output = self.c_proj(self._merge_heads(context))
        present = (key, value) if use_cache else None
        return output, present

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        output, _ = self.forward_with_cache(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return output


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

    def forward_with_cache(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
        past_key_value: Optional[KeyValueCache] = None,
        use_cache: bool = True,
    ) -> Tuple[jt.Var, Optional[KeyValueCache]]:
        attention_output, present = self.attn.forward_with_cache(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present

    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        output, _ = self.forward_with_cache(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return output


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

    def forward_with_cache(
        self,
        input_ids: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
        past_key_values: Optional[PastKeyValues] = None,
        use_cache: bool = True,
    ) -> Dict[str, object]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [B, T, D]")
        batch, tokens, hidden = (int(value) for value in inputs_embeds.shape)
        if hidden != self.config.n_embd:
            raise ValueError("GPT-2 hidden size mismatch")

        if past_key_values is None:
            layer_past = [None] * self.config.n_layer
            past_tokens = 0
        else:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError("past_key_values layer count mismatch")
            layer_past = list(past_key_values)
            lengths = {int(item[0].shape[2]) for item in layer_past}
            if len(lengths) != 1:
                raise ValueError("past_key_values lengths must match")
            past_tokens = lengths.pop()

        total_tokens = past_tokens + tokens
        if total_tokens > self.config.n_positions:
            raise ValueError(
                f"sequence length {total_tokens} exceeds "
                f"{self.config.n_positions}"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape [B, T]")
            if tuple(int(item) for item in attention_mask.shape) != (
                batch,
                total_tokens,
            ):
                raise ValueError("attention_mask shape mismatch")

        position_ids = (jt.arange(tokens) + past_tokens).reshape(
            (1, tokens)
        ).broadcast((batch, tokens))
        hidden_states = inputs_embeds + self.wpe(position_ids)
        presents: PastKeyValues = []
        for block, past_key_value in zip(self.blocks(), layer_past):
            hidden_states, present = block.forward_with_cache(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                if present is None:
                    raise RuntimeError("cache was requested but not returned")
                presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        flat = hidden_states.reshape((-1, self.config.n_embd))
        logits = jt.matmul(flat, self.wte.weight.transpose(0, 1)).reshape(
            (batch, tokens, self.config.vocab_size)
        )
        return {
            "logits": logits,
            "past_key_values": presents if use_cache else None,
        }

    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        output = self.forward_with_cache(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return output["logits"]

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


def greedy_generate_from_embeddings_cached(
    model: NativeGPT2LMHeadModel,
    initial_embeddings: jt.Var,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Greedy decode from packed embeddings while reusing per-layer KV state."""

    if initial_embeddings.ndim != 3:
        raise ValueError("initial_embeddings must have shape [B, T, D]")
    if int(initial_embeddings.shape[0]) != 1:
        raise ValueError("generation currently requires batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return []

    output = model.forward_with_cache(
        inputs_embeds=initial_embeddings,
        use_cache=True,
    )
    generated: List[int] = []
    for step in range(max_new_tokens):
        logits = output["logits"]
        next_ids, _ = logits[:, -1, :].argmax(dim=-1)
        next_token = int(next_ids.numpy()[0])
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if step + 1 >= max_new_tokens:
            break
        token_var = jt.array(np.asarray([[next_token]], dtype=np.int32))
        output = model.forward_with_cache(
            input_ids=token_var,
            past_key_values=output["past_key_values"],
            use_cache=True,
        )
    return generated
