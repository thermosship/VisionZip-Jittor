"""Native Jittor training and checkpoint helpers for Phase 4A."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

import jittor as jt
import numpy as np

from .gpt2 import greedy_generate_from_embeddings
from .multimodal import pack_multimodal_embeddings
from .phase4_data import build_label_arrays, prepare_text_supervision


def snapshot_parameters(parameters: Iterable[jt.Var]) -> List[np.ndarray]:
    return [parameter.numpy().copy() for parameter in parameters]


def parameter_sha256(parameters: Iterable[jt.Var]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = np.ascontiguousarray(parameter.numpy())
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(optimizer) -> str:
    """Hash Adam step and moment tensors in parameter-group order."""

    digest = hashlib.sha256()
    digest.update(str(int(optimizer.n_step)).encode("ascii"))
    for group in optimizer.param_groups:
        for field in ("values", "m"):
            for tensor in group[field]:
                value = np.ascontiguousarray(tensor.numpy())
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(value.shape).encode("ascii"))
                digest.update(value.tobytes())
    return digest.hexdigest()


def max_parameter_delta(
    before: Sequence[np.ndarray],
    parameters: Iterable[jt.Var],
) -> float:
    maximum = 0.0
    for previous, current in zip(before, parameters):
        difference = np.max(
            np.abs(
                previous.astype(np.float64)
                - current.numpy().astype(np.float64)
            )
        )
        maximum = max(maximum, float(difference))
    return maximum


def gradient_statistics(
    parameters: Iterable[jt.Var],
    optimizer,
) -> Dict[str, Any]:
    squared_norm = 0.0
    maximum = 0.0
    tensor_count = 0
    finite = True
    for parameter in parameters:
        gradient = parameter.opt_grad(optimizer).numpy()
        finite = finite and bool(np.isfinite(gradient).all())
        squared_norm += float(
            np.sum(gradient.astype(np.float64) ** 2)
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(gradient.astype(np.float64)))),
        )
        if np.any(gradient != 0):
            tensor_count += 1
    return {
        "finite": finite,
        "l2_norm": float(np.sqrt(squared_norm)),
        "max_abs": maximum,
        "parameter_tensors_with_nonzero_grad": tensor_count,
    }


def build_jittor_training_batch(
    tokenizer,
    model,
    projected_visual: jt.Var,
    captions: Sequence[str],
    prompt: str,
    max_caption_tokens: int,
) -> Dict[str, Any]:
    """Pack prompt, visual tokens, and teacher-forced caption embeddings."""

    batch_size = int(projected_visual.shape[0])
    if len(captions) != batch_size:
        raise ValueError("caption count must equal projected visual batch size")
    supervision = prepare_text_supervision(
        tokenizer,
        captions,
        prompt,
        max_caption_tokens,
    )
    prompt_ids_np = np.repeat(
        supervision["prompt_ids"][None, :],
        batch_size,
        axis=0,
    )
    prompt_ids = jt.array(prompt_ids_np)
    target_ids = jt.array(supervision["target_ids"])
    prompt_ids.stop_grad()
    target_ids.stop_grad()
    prompt_embeddings = model.embed_tokens(prompt_ids)
    target_embeddings = model.embed_tokens(target_ids)
    packed = pack_multimodal_embeddings(
        prompt_embeddings,
        projected_visual,
        target_embeddings,
    )
    arrays = build_label_arrays(
        prompt_tokens=int(prompt_ids_np.shape[1]),
        visual_tokens=int(projected_visual.shape[1]),
        target_ids=supervision["target_ids"],
        target_mask=supervision["target_mask"],
    )
    labels = jt.array(arrays["labels"])
    label_mask = jt.array(arrays["label_mask"])
    attention_mask = jt.array(arrays["attention_mask"])
    for tensor in (labels, label_mask, attention_mask):
        tensor.stop_grad()
    return {
        "packed_embeddings": packed,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_mask": label_mask,
        "prompt_tokens": int(prompt_ids_np.shape[1]),
        "visual_tokens": int(projected_visual.shape[1]),
        "target_tokens": int(supervision["target_ids"].shape[1]),
        "target_token_counts": supervision["target_token_counts"],
    }


def build_generation_embeddings(
    tokenizer,
    model,
    projected_visual: jt.Var,
    prompt: str,
    generation_prompt: str,
) -> jt.Var:
    """Pack one prompt/visual/generation-prefix sequence."""

    if int(projected_visual.shape[0]) != 1:
        raise ValueError("generation embedding builder requires batch size 1")
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    suffix_ids = tokenizer.encode(generation_prompt, add_special_tokens=False)
    if not prefix_ids or not suffix_ids:
        raise ValueError("generation prompts must tokenize to non-empty ids")
    prefix = jt.array(np.asarray([prefix_ids], dtype=np.int32))
    suffix = jt.array(np.asarray([suffix_ids], dtype=np.int32))
    prefix.stop_grad()
    suffix.stop_grad()
    return pack_multimodal_embeddings(
        model.embed_tokens(prefix),
        projected_visual,
        model.embed_tokens(suffix),
    )


def generate_caption(
    tokenizer,
    model,
    projected_visual: jt.Var,
    prompt: str,
    generation_prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    embeddings = build_generation_embeddings(
        tokenizer,
        model,
        projected_visual,
        prompt,
        generation_prompt,
    )
    token_ids = greedy_generate_from_embeddings(
        model,
        embeddings,
        max_new_tokens=max_new_tokens,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
    }


def save_phase4_checkpoint(
    path: Union[str, Path],
    projector,
    optimizer,
    metadata: Dict[str, Any],
    artifact_type: str = "phase4a_projector_checkpoint_v1",
) -> None:
    """Atomically save Projector and complete Jittor Adam state to NPZ."""

    if not artifact_type.strip():
        raise ValueError("checkpoint artifact_type must not be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for name, value in projector.state_dict(to="numpy").items():
        arrays[f"projector::{name}"] = np.asarray(value)
    for group_index, group in enumerate(optimizer.param_groups):
        for value_index, value in enumerate(group["values"]):
            arrays[f"adam_v::{group_index}::{value_index}"] = value.numpy()
        for value_index, value in enumerate(group["m"]):
            arrays[f"adam_m::{group_index}::{value_index}"] = value.numpy()
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "artifact_type": artifact_type,
            "optimizer": "jittor.optim.Adam",
            "optimizer_n_step": int(optimizer.n_step),
            "optimizer_lr": float(optimizer.lr),
            "optimizer_eps": float(optimizer.eps),
            "optimizer_betas": [float(item) for item in optimizer.betas],
            "optimizer_weight_decay": float(optimizer.weight_decay),
            "projector_parameter_sha256": parameter_sha256(
                projector.parameters()
            ),
            "optimizer_state_sha256": optimizer_state_sha256(optimizer),
        }
    )
    arrays["metadata_json"] = np.asarray(
        json.dumps(checkpoint_metadata, ensure_ascii=False, sort_keys=True)
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)



def step_adam_after_gradient_accumulation(
    optimizer,
    completed_optimizer_steps: int,
) -> int:
    """Apply one Adam update with bias correction based on optimizer steps.

    Jittor 1.3.11 increments ``optimizer.n_step`` inside every
    ``optimizer.backward`` call. During gradient accumulation that would make
    Adam bias correction count microbatches instead of updates. Reset it to the
    one-based update number immediately before ``step``.
    """

    if completed_optimizer_steps < 0:
        raise ValueError("completed_optimizer_steps must be non-negative")
    optimizer.n_step = completed_optimizer_steps + 1
    optimizer.step()
    return int(optimizer.n_step)

def load_phase4_checkpoint(
    path: Union[str, Path],
    projector,
    optimizer,
    expected_artifact_type: str = "phase4a_projector_checkpoint_v1",
) -> Dict[str, Any]:
    """Restore Projector and Jittor Adam state, validating every tensor."""

    if not expected_artifact_type.strip():
        raise ValueError("expected checkpoint artifact_type must not be empty")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 4 checkpoint: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("checkpoint is missing metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("artifact_type") != expected_artifact_type:
            raise ValueError(
                "unsupported Phase 4 checkpoint artifact type: "
                f"expected {expected_artifact_type!r}, "
                f"got {metadata.get('artifact_type')!r}"
            )
        state = projector.state_dict()
        expected_projector = {f"projector::{name}" for name in state}
        actual_projector = {
            name for name in archive.files if name.startswith("projector::")
        }
        if actual_projector != expected_projector:
            raise ValueError(
                "checkpoint Projector keys mismatch: "
                f"missing={sorted(expected_projector - actual_projector)}, "
                f"unexpected={sorted(actual_projector - expected_projector)}"
            )
        for name, target in state.items():
            value = archive[f"projector::{name}"].astype(
                np.float32,
                copy=False,
            )
            if tuple(value.shape) != tuple(int(item) for item in target.shape):
                raise ValueError(f"checkpoint shape mismatch for {name}")
            target.assign(value)
        for group_index, group in enumerate(optimizer.param_groups):
            for field, prefix in (("values", "adam_v"), ("m", "adam_m")):
                for value_index, target in enumerate(group[field]):
                    key = f"{prefix}::{group_index}::{value_index}"
                    if key not in archive.files:
                        raise ValueError(f"checkpoint is missing {key}")
                    value = archive[key].astype(np.float32, copy=False)
                    if tuple(value.shape) != tuple(
                        int(item) for item in target.shape
                    ):
                        raise ValueError(f"checkpoint shape mismatch for {key}")
                    target.assign(value)
    optimizer.n_step = int(metadata["optimizer_n_step"])
    optimizer.lr = float(metadata["optimizer_lr"])
    optimizer.eps = float(metadata["optimizer_eps"])
    optimizer.betas = tuple(float(item) for item in metadata["optimizer_betas"])
    optimizer.weight_decay = float(metadata["optimizer_weight_decay"])
    jt.sync_all()
    restored_hash = parameter_sha256(projector.parameters())
    if restored_hash != metadata["projector_parameter_sha256"]:
        raise ValueError("restored Projector hash does not match checkpoint")
    restored_optimizer_hash = optimizer_state_sha256(optimizer)
    if restored_optimizer_hash != metadata["optimizer_state_sha256"]:
        raise ValueError("restored Adam state hash does not match checkpoint")
    return metadata
