#!/usr/bin/env python3
"""Precompute sharded real-CLIP/VisionZip features for a Phase 4B dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.clip_features import key_projection_to_metric, resolve_layer_index
from reference.pytorch_visionzip import visionzip_compress_torch
from visionzip_jittor.config import load_config
from visionzip_jittor.phase4b_config import load_phase4b_config
from visionzip_jittor.phase4b_data import (
    canonical_json_sha256,
    load_prepared_dataset,
)
from visionzip_jittor.phase4b_features import (
    Phase4BFeatureShard,
    iter_feature_shards,
    load_feature_manifest,
    sample_ids_sha256,
    validate_feature_arrays,
    write_feature_manifest,
    write_feature_shard,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase4b_commoncatalog_cc_by_8k.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "datasets/phase4b/commoncatalog_cc_by_8k/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase4b/commoncatalog_cc_by_8k/features",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/cache/huggingface"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify and reuse an already completed feature manifest.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    return torch.device(requested)


def existing_shard_metadata(
    path: Path,
    expected_ids: Sequence[str],
    expected_token_shape: Sequence[int],
):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    count, token_shape = validate_feature_arrays(arrays)
    if token_shape != tuple(int(item) for item in expected_token_shape):
        raise ValueError(f"Existing feature shard token shape does not match: {path}")
    actual_ids = arrays["sample_ids"].astype(str).tolist()
    if actual_ids != list(expected_ids):
        raise ValueError(f"Existing feature shard sample ids do not match: {path}")
    from visionzip_jittor.phase4b_data import file_sha256

    return Phase4BFeatureShard(
        path=path.name,
        sha256=file_sha256(path),
        sample_count=count,
        sample_ids_sha256=sample_ids_sha256(actual_ids),
        first_sample_id=actual_ids[0],
        last_sample_id=actual_ids[-1],
    )


def save_progress(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def compute_batch(
    samples,
    dataset_root: Path,
    processor,
    model,
    attention_module,
    layer_index: int,
    visionzip_config,
    device: torch.device,
    storage_dtype: str,
):
    opened = []
    for item in samples:
        with Image.open(dataset_root / item.image_path) as image:
            opened.append(image.convert("RGB"))
    captured = {}

    def capture_key_projection(_module, _inputs, output):
        captured["key_projection"] = output.detach()

    hook = attention_module.k_proj.register_forward_hook(capture_key_projection)
    try:
        pixel_values = processor(images=opened, return_tensors="pt")["pixel_values"].to(
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        hook.remove()
        for image in opened:
            image.close()
    if "key_projection" not in captured:
        raise RuntimeError("CLIP k_proj hook did not capture a tensor")
    if outputs.attentions is None or outputs.attentions[layer_index] is None:
        raise RuntimeError("CLIP did not return attention weights")
    hidden_states = outputs.hidden_states[layer_index + 1]
    attentions = outputs.attentions[layer_index]
    metric = key_projection_to_metric(
        captured["key_projection"],
        attention_module.num_heads,
    )
    compressed = visionzip_compress_torch(
        hidden_states,
        attentions,
        metric,
        visionzip_config,
    )
    target_dtype = np.float16 if storage_dtype == "float16" else np.float32
    arrays = {
        "sample_ids": np.asarray([item.sample_id for item in samples]),
        "compressed_tokens": compressed.compressed_tokens.detach().cpu().numpy().astype(target_dtype),
        "selected_indices": compressed.selected_indices.detach().cpu().numpy().astype(np.int16),
        "assignments": compressed.assignments.detach().cpu().numpy().astype(np.int8),
    }
    validate_feature_arrays(arrays)
    del outputs, hidden_states, attentions, metric, compressed, pixel_values
    return arrays


def main() -> None:
    args = parse_args()
    config = load_phase4b_config(args.config)
    prepared_manifest, samples = load_prepared_dataset(
        args.dataset_manifest,
        verify_images=False,
    )
    expected_config_hash = canonical_json_sha256(config.to_dict())
    if prepared_manifest.config_sha256 != expected_config_hash:
        raise ValueError("Prepared dataset was created from a different Phase 4B config")
    if len(samples) != config.dataset.target_sample_count:
        raise ValueError("Prepared dataset sample count does not match Phase 4B config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = args.output_dir / "manifest.json"
    if final_manifest_path.exists():
        if not args.verify_existing:
            raise SystemExit(
                "Feature manifest already exists; pass --verify-existing to verify and reuse it."
            )
        manifest = load_feature_manifest(final_manifest_path, verify_shards=True)
        if manifest.config_sha256 != expected_config_hash:
            raise ValueError("Existing feature manifest config hash mismatch")
        if manifest.dataset_samples_sha256 != prepared_manifest.samples_sha256:
            raise ValueError("Existing feature manifest dataset hash mismatch")
        if manifest.model_name_or_path != config.features.model_name_or_path:
            raise ValueError("Existing feature manifest model name mismatch")
        if manifest.model_revision != config.features.model_revision:
            raise ValueError("Existing feature manifest model revision mismatch")
        if manifest.sample_count != len(samples):
            raise ValueError("Existing feature manifest sample count mismatch")
        expected_ids = [item.sample_id for item in samples]
        actual_ids = []
        for _shard, arrays in iter_feature_shards(final_manifest_path, verify_shards=False):
            actual_ids.extend(arrays["sample_ids"].astype(str).tolist())
        if actual_ids != expected_ids:
            raise ValueError("Existing feature manifest sample order mismatch")
        print(json.dumps({"passed": True, "reused": True, "manifest": manifest.to_dict()}, indent=2))
        return

    device = select_device(args.device)
    torch.manual_seed(config.training.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    load_kwargs = {
        "cache_dir": str(args.cache_dir),
        "revision": config.features.model_revision,
        "torch_dtype": torch.float32,
    }
    processor = CLIPImageProcessor.from_pretrained(
        config.features.model_name_or_path,
        cache_dir=str(args.cache_dir),
        revision=config.features.model_revision,
    )
    model = CLIPVisionModel.from_pretrained(
        config.features.model_name_or_path,
        **load_kwargs,
    ).to(device)
    model.eval()
    layer_count = len(model.vision_model.encoder.layers)
    layer_index = resolve_layer_index(
        num_layers=layer_count,
        requested=config.features.requested_layer_index,
    )
    attention_module = model.vision_model.encoder.layers[layer_index].self_attn
    visionzip_path = ROOT / config.features.visionzip_config
    visionzip_config = load_config(visionzip_path)
    if visionzip_config.nominal_visual_tokens != config.training.budget:
        raise ValueError("VisionZip config budget does not match Phase 4B training budget")

    expected_token_shape = (
        visionzip_config.actual_output_tokens,
        config.training.vision_hidden_size,
    )
    feature_shards: List[Phase4BFeatureShard] = []
    shard_size = config.features.shard_size
    micro_batch = config.features.batch_size
    dataset_root = args.dataset_manifest.parent
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        shard_samples = samples[start : start + shard_size]
        expected_ids = [item.sample_id for item in shard_samples]
        shard_path = args.output_dir / f"features-{shard_index:05d}.npz"
        if shard_path.exists():
            shard = existing_shard_metadata(
                shard_path, expected_ids, expected_token_shape
            )
            feature_shards.append(shard)
            print(f"Reused feature shard {shard.path} ({shard.sample_count} samples)")
        else:
            batch_arrays = []
            for batch_start in range(0, len(shard_samples), micro_batch):
                batch = shard_samples[batch_start : batch_start + micro_batch]
                batch_arrays.append(
                    compute_batch(
                        batch,
                        dataset_root,
                        processor,
                        model,
                        attention_module,
                        layer_index,
                        visionzip_config,
                        device,
                        config.features.storage_dtype,
                    )
                )
            arrays = {
                key: np.concatenate([item[key] for item in batch_arrays], axis=0)
                for key in batch_arrays[0]
            }
            shard = write_feature_shard(args.output_dir, shard_index, arrays)
            feature_shards.append(shard)
            print(f"Saved feature shard {shard.path} ({shard.sample_count} samples)")
        save_progress(
            args.output_dir / "progress.json",
            {
                "artifact_type": "phase4b_feature_progress_v1",
                "completed_samples": sum(item.sample_count for item in feature_shards),
                "target_samples": len(samples),
                "completed_shards": [item.to_dict() for item in feature_shards],
            },
        )

    manifest = write_feature_manifest(
        output_dir=args.output_dir,
        config_payload=config.to_dict(),
        dataset_samples_sha256=prepared_manifest.samples_sha256,
        model_name_or_path=config.features.model_name_or_path,
        model_revision=config.features.model_revision,
        visionzip_config=visionzip_config.to_dict(),
        requested_layer_index=config.features.requested_layer_index,
        resolved_layer_index=layer_index,
        storage_dtype=config.features.storage_dtype,
        token_shape=expected_token_shape,
        shards=feature_shards,
    )
    actual_ids = []
    for _shard, arrays in iter_feature_shards(
        args.output_dir / "manifest.json",
        verify_shards=True,
    ):
        actual_ids.extend(arrays["sample_ids"].astype(str).tolist())
    expected_ids = [item.sample_id for item in samples]
    if actual_ids != expected_ids:
        raise ValueError("Final feature shard sample order does not match prepared data")

    result = {
        "artifact_type": "phase4b_feature_precompute_result_v1",
        "passed": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "manifest": manifest.to_dict(),
    }
    save_progress(args.output_dir / "precompute_summary.json", result)
    print("=" * 72)
    print("Phase 4B feature precompute complete")
    print("=" * 72)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
