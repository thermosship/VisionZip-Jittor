#!/usr/bin/env python3
"""Export Hugging Face GPT-2 weights/tokenizer/reference for native Jittor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# This exporter requires PyTorch. Override variables that may remain set after
# using the separate Jittor environment in the same shell.
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="openai-community/gpt2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase3b/gpt2",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-prompt",
        default="A native Jittor language model",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative names and contents for a deterministic artifact tree."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def expected_weight_names(layer_count: int):
    names = [
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
    ]
    suffixes = [
        "ln_1.weight",
        "ln_1.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ]
    for index in range(layer_count):
        names.extend(f"transformer.h.{index}.{suffix}" for suffix in suffixes)
    return names


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.output_dir / "tokenizer"

    load_kwargs = {
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
    }
    load_kwargs = {
        key: value for key, value in load_kwargs.items() if value is not None
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **load_kwargs)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = False

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(tokenizer_dir)

    full_config_path = args.output_dir / "hf_config.json"
    full_config_path.write_text(
        json.dumps(model.config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    state = model.state_dict()
    names = expected_weight_names(model.config.n_layer)
    missing = sorted(set(names) - set(state))
    if missing:
        raise RuntimeError(f"Hugging Face state dict is missing: {missing}")
    arrays = {
        name: state[name].detach().cpu().float().numpy()
        for name in names
    }
    weights_path = args.output_dir / "gpt2_float32_weights.npz"
    np.savez(weights_path, **arrays)

    encoded = tokenizer(args.reference_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        token_embeddings = model.transformer.wte(input_ids)
        logits_from_ids = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        logits_from_embeds = model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    if not torch.equal(logits_from_ids, logits_from_embeds):
        maximum = float((logits_from_ids - logits_from_embeds).abs().max())
        raise RuntimeError(f"GPT-2 reference paths differ: max error={maximum}")

    reference_path = args.output_dir / "text_reference.npz"
    np.savez(
        reference_path,
        input_ids=input_ids.cpu().numpy().astype(np.int32),
        attention_mask=attention_mask.cpu().numpy().astype(np.float32),
        logits=logits_from_ids.cpu().numpy().astype(np.float32),
        prompt=np.asarray(args.reference_prompt),
    )

    manifest = {
        "artifact_type": "phase3b_hf_gpt2_to_jittor_v1",
        "model_name_or_path": args.model_name_or_path,
        "real_llm": True,
        "architecture": "GPT2LMHeadModel",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": "float32",
        "tied_lm_head": True,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "exported_tensor_count": len(arrays),
        "reference_prompt": args.reference_prompt,
        "files": {
            "weights": weights_path.name,
            "config": full_config_path.name,
            "tokenizer": tokenizer_dir.name,
            "reference": reference_path.name,
        },
        "sha256": {
            "weights": file_sha256(weights_path),
            "config": file_sha256(full_config_path),
            "tokenizer": directory_sha256(tokenizer_dir),
            "reference": file_sha256(reference_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved Phase 3B GPT-2 artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
