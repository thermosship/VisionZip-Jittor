"""Export real CLIP features and PyTorch VisionZip references to NPZ."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.clip_features import (
    contextual_target_original_indices,
    infer_patch_grid,
    key_projection_to_metric,
    merge_original_indices,
    real_reference_filename,
    resolve_layer_index,
)
from reference.pytorch_visionzip import visionzip_compress_torch
from visionzip_jittor.config import load_config

DEFAULT_CONFIGS = (
    ROOT / "configs/visionzip_64.json",
    ROOT / "configs/visionzip_128.json",
    ROOT / "configs/visionzip_192.json",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPSTREAM_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real CLIP tensors and export PyTorch VisionZip references."
    )
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/real_clip")
    parser.add_argument(
        "--model-name-or-path",
        default="openai/clip-vit-large-patch14-336",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-2,
        help="CLIP encoder layer used by VisionZip; -2 matches the official code.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def collect_images(explicit: Sequence[Path], image_dir: Path) -> List[Path]:
    candidates: List[Path] = list(explicit)
    if image_dir is not None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {image_dir}")
        candidates.extend(
            sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"image not found: {candidate}")
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {resolved}")
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("provide at least one --image or a non-empty --image-dir")
    return unique


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot see CUDA")
    return torch.device(requested)


def numpy_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    image_paths = collect_images(args.image, args.image_dir)
    config_paths = args.config or list(DEFAULT_CONFIGS)
    configs = [load_config(path) for path in config_paths]
    device = choose_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CLIP extraction requires CUDA")

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    try:
        import transformers
        from transformers import CLIPImageProcessor, CLIPVisionModel
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install requirements/real_clip.txt "
            "in the PyTorch baseline environment."
        ) from exc

    load_kwargs = {
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    processor = CLIPImageProcessor.from_pretrained(
        args.model_name_or_path, **load_kwargs
    )
    model = CLIPVisionModel.from_pretrained(args.model_name_or_path, **load_kwargs)
    model = model.to(device=device, dtype=dtype).eval()

    layers = model.vision_model.encoder.layers
    layer_index = resolve_layer_index(len(layers), args.layer_index)
    attention_module = layers[layer_index].self_attn
    captured: Dict[str, torch.Tensor] = {}

    def capture_key_projection(module, inputs, output):
        del module, inputs
        captured["key_projection"] = output.detach()

    hook = attention_module.k_proj.register_forward_hook(capture_key_projection)
    opened_images = []
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                opened_images.append(image.convert("RGB"))
        processor_output = processor(images=opened_images, return_tensors="pt")
        pixel_values = processor_output["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            outputs = model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        hook.remove()
        for image in opened_images:
            image.close()

    if "key_projection" not in captured:
        raise RuntimeError("the k_proj forward hook did not capture any tensor")
    if outputs.attentions is None or outputs.attentions[layer_index] is None:
        raise RuntimeError(
            "CLIP did not return attention weights. Use transformers==4.31.0 "
            "from requirements/real_clip.txt."
        )

    hidden_states = outputs.hidden_states[layer_index + 1]
    attentions = outputs.attentions[layer_index]
    metric = key_projection_to_metric(
        captured["key_projection"], attention_module.num_heads
    )
    captured.clear()
    del outputs, model, attention_module
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if hidden_states.shape[:2] != attentions.shape[:1] + attentions.shape[2:3]:
        raise RuntimeError("hidden-state and attention sequence shapes do not match")
    if hidden_states.shape[:2] != metric.shape[:2]:
        raise RuntimeError("hidden-state and metric sequence shapes do not match")

    sequence_length = int(hidden_states.shape[1])
    grid_height, grid_width = infer_patch_grid(sequence_length, include_cls=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor_mean = list(getattr(processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]))
    processor_std = list(getattr(processor, "image_std", [0.26862954, 0.26130258, 0.27577711]))
    common_arrays = {
        "hidden_states": numpy_array(hidden_states),
        "attentions": numpy_array(attentions),
        "metric": numpy_array(metric),
        "pixel_values": numpy_array(pixel_values),
    }

    written = []
    for config_path, config in zip(config_paths, configs):
        config.validate(sequence_length)
        reference = visionzip_compress_torch(
            hidden_states, attentions, metric, config
        )
        output_dict = reference.as_dict()
        contextual_original = contextual_target_original_indices(
            output_dict["remaining_indices"], output_dict["target_positions"]
        )
        merge_original = merge_original_indices(
            output_dict["remaining_indices"], output_dict["merge_positions"]
        )
        arrays = dict(common_arrays)
        arrays.update({key: numpy_array(value) for key, value in output_dict.items()})
        arrays["contextual_target_indices"] = numpy_array(contextual_original)
        arrays["merge_original_indices"] = numpy_array(merge_original)

        metadata = {
            "artifact_type": "real_clip_reference_v1",
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
            "images": [
                {"name": path.name, "path": str(path)} for path in image_paths
            ],
            "config_path": str(config_path),
            "config": config.to_dict(),
            "model_name_or_path": args.model_name_or_path,
            "layer_index": layer_index,
            "requested_layer_index": args.layer_index,
            "hidden_states_index": layer_index + 1,
            "attention_index": layer_index,
            "metric_source": "k_proj -> [B,H,N,D] -> mean(heads)",
            "patch_grid": [grid_height, grid_width],
            "processor": {
                "image_mean": processor_mean,
                "image_std": processor_std,
                "crop_size": getattr(processor, "crop_size", None),
                "size": getattr(processor, "size", None),
            },
            "official_upstream_commit": UPSTREAM_COMMIT,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
        arrays["metadata_json"] = np.array(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        )
        output_path = args.output_dir / real_reference_filename(config.name, args.dtype)
        np.savez_compressed(output_path, **arrays)
        written.append(str(output_path))
        print(f"Saved real CLIP reference: {output_path}")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    manifest = {
        "artifact_type": "real_clip_export_manifest_v1",
        "references": written,
        "images": [str(path) for path in image_paths],
        "model_name_or_path": args.model_name_or_path,
        "dtype": args.dtype,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
