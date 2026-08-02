"""Visualize dominant-token selection and contextual-token merging."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/visualizations"
    )
    parser.add_argument("--sample-index", type=int, action="append", default=[])
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def load_font(size: int):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def denormalize_pixel_values(
    pixel_values: np.ndarray, mean: Sequence[float], std: Sequence[float]
) -> Image.Image:
    image = pixel_values.astype(np.float32).transpose(1, 2, 0)
    image = image * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB")


def patch_box(token_index: int, grid: Tuple[int, int], image_size: int):
    patch_index = token_index - 1
    rows, cols = grid
    row, col = divmod(patch_index, cols)
    x0 = round(col * image_size / cols)
    y0 = round(row * image_size / rows)
    x1 = round((col + 1) * image_size / cols)
    y1 = round((row + 1) * image_size / rows)
    return x0, y0, x1, y1


def blend_patch(image: Image.Image, box, color, alpha: int) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, fill=tuple(color) + (alpha,))
    image.alpha_composite(overlay)


def palette(count: int) -> List[Tuple[int, int, int]]:
    values = []
    for index in range(max(count, 1)):
        hue = index / max(count, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
        values.append(tuple(int(channel * 255) for channel in rgb))
    return values


def render_dominant(
    base: Image.Image,
    attention: np.ndarray,
    dominant_indices: Iterable[int],
    grid: Tuple[int, int],
) -> Image.Image:
    panel = base.convert("RGBA")
    scores = attention.astype(np.float64)
    low, high = float(scores.min()), float(scores.max())
    normalized = (scores - low) / (high - low + 1e-12)
    for patch, score in enumerate(normalized, start=1):
        blend_patch(panel, patch_box(patch, grid, panel.width), (255, 45, 45), int(20 + 120 * score))
    draw = ImageDraw.Draw(panel)
    width = max(2, panel.width // 168)
    for token_index in dominant_indices:
        token_index = int(token_index)
        if token_index <= 0:
            continue
        draw.rectangle(patch_box(token_index, grid, panel.width), outline=(255, 230, 0, 255), width=width)
    return panel.convert("RGB")


def render_contextual(
    base: Image.Image,
    dominant_indices: np.ndarray,
    target_indices: np.ndarray,
    merge_indices: np.ndarray,
    assignments: np.ndarray,
    grid: Tuple[int, int],
) -> Image.Image:
    panel = base.convert("RGBA")
    colors = palette(len(target_indices))
    for token_index, assignment in zip(merge_indices, assignments):
        token_index = int(token_index)
        if token_index > 0:
            blend_patch(panel, patch_box(token_index, grid, panel.width), colors[int(assignment)], 92)
    for group, token_index in enumerate(target_indices):
        token_index = int(token_index)
        if token_index > 0:
            blend_patch(panel, patch_box(token_index, grid, panel.width), colors[group], 175)
    draw = ImageDraw.Draw(panel)
    border = max(2, panel.width // 168)
    font = load_font(max(10, panel.width // 42))
    for group, token_index in enumerate(target_indices):
        token_index = int(token_index)
        if token_index <= 0:
            continue
        box = patch_box(token_index, grid, panel.width)
        draw.rectangle(box, outline=(255, 255, 255, 255), width=border)
        draw.text((box[0] + 2, box[1] + 1), str(group), font=font, fill=(0, 0, 0, 255), stroke_width=1, stroke_fill=(255, 255, 255, 255))
    for token_index in dominant_indices:
        token_index = int(token_index)
        if token_index > 0:
            draw.rectangle(patch_box(token_index, grid, panel.width), outline=(255, 30, 30, 255), width=border)
    return panel.convert("RGB")


def add_title(panel: Image.Image, title: str, title_height: int) -> Image.Image:
    canvas = Image.new("RGB", (panel.width, panel.height + title_height), "white")
    canvas.paste(panel, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(14, title_height // 2))
    draw.text((10, max(1, title_height // 5)), title, font=font, fill="#18212f")
    return canvas


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    with np.load(args.reference, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        pixel_values = data["pixel_values"]
        selected_indices = data["dominant_ordered_indices"]
        target_indices = data["contextual_target_indices"]
        merge_indices = data["merge_original_indices"]
        assignments = data["assignments"]
        cls_attention_sum = data["cls_attention_sum"]

    images = metadata.get("images", [])
    batch = int(pixel_values.shape[0])
    requested = args.sample_index or list(range(batch))
    for index in requested:
        if index < 0 or index >= batch:
            raise IndexError(f"sample index {index} is outside batch size {batch}")

    mean = metadata["processor"]["image_mean"]
    std = metadata["processor"]["image_std"]
    grid = tuple(int(value) for value in metadata["patch_grid"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_height = 34 * args.scale

    for index in requested:
        base = denormalize_pixel_values(pixel_values[index], mean, std)
        side = base.width * args.scale
        base = base.resize((side, side), Image.Resampling.BICUBIC)
        dominant = render_dominant(
            base, cls_attention_sum[index], selected_indices[index], grid
        )
        contextual = render_contextual(
            base,
            selected_indices[index],
            target_indices[index],
            merge_indices[index],
            assignments[index],
            grid,
        )
        panels = [
            add_title(base, "Preprocessed CLIP input", title_height),
            add_title(dominant, "CLS attention + dominant tokens", title_height),
            add_title(contextual, "Contextual merge groups", title_height),
        ]
        canvas = Image.new(
            "RGB", (sum(panel.width for panel in panels), panels[0].height), "white"
        )
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        image_name = images[index]["name"] if index < len(images) else f"sample_{index}"
        stem = Path(image_name).stem
        output_path = args.output_dir / f"{args.reference.stem}_{stem}_tokens.png"
        canvas.save(output_path)
        print(f"Saved token visualization: {output_path}")


if __name__ == "__main__":
    main()
