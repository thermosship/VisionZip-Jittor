"""Create deterministic, license-free sample images for real CLIP tests."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "assets/sample_images"
    )
    parser.add_argument("--size", type=int, default=672)
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def create_scene(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "#91c9f7")
    draw = ImageDraw.Draw(image)
    horizon = int(size * 0.58)
    draw.rectangle((0, horizon, size, size), fill="#78b159")
    draw.ellipse((size * 0.72, size * 0.08, size * 0.9, size * 0.26), fill="#ffd45c")
    draw.polygon(
        [(size * 0.18, horizon), (size * 0.42, size * 0.31), (size * 0.66, horizon)],
        fill="#b45f45",
    )
    draw.rectangle((size * 0.25, size * 0.47, size * 0.59, size * 0.79), fill="#f2d39b")
    draw.rectangle((size * 0.38, size * 0.62, size * 0.48, size * 0.79), fill="#6f4e37")
    draw.rectangle((size * 0.28, size * 0.54, size * 0.36, size * 0.64), fill="#9ed5f5")
    draw.rectangle((size * 0.5, size * 0.54, size * 0.58, size * 0.64), fill="#9ed5f5")
    draw.rectangle((size * 0.75, size * 0.48, size * 0.79, size * 0.82), fill="#795548")
    draw.ellipse((size * 0.65, size * 0.32, size * 0.89, size * 0.61), fill="#2f8f46")
    draw.line((0, horizon, size, horizon), fill="#406343", width=max(2, size // 150))
    return image


def create_text(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "#f7f4ea")
    draw = ImageDraw.Draw(image)
    margin = size // 12
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=size // 30,
        fill="#ffffff",
        outline="#253858",
        width=max(3, size // 100),
    )
    title = load_font(size // 10, bold=True)
    body = load_font(size // 18, bold=False)
    small = load_font(size // 24, bold=True)
    draw.text((size * 0.16, size * 0.15), "VISIONZIP", font=title, fill="#d64545")
    draw.text((size * 0.18, size * 0.34), "JITTOR REPRODUCTION", font=body, fill="#253858")
    draw.text((size * 0.18, size * 0.48), "TOKEN COMPRESSION", font=body, fill="#253858")
    draw.text((size * 0.27, size * 0.64), "64   128   192", font=title, fill="#276749")
    draw.text((size * 0.29, size * 0.82), "REAL CLIP ALIGNMENT", font=small, fill="#6b7280")
    return image


def create_dense(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "#1d2433")
    draw = ImageDraw.Draw(image)
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#b983ff", "#ff8fab"]
    cells = 8
    gap = size // 80
    cell = size // cells
    for row in range(cells):
        for col in range(cells):
            x0 = col * cell + gap
            y0 = row * cell + gap
            x1 = (col + 1) * cell - gap
            y1 = (row + 1) * cell - gap
            color = colors[(row * 3 + col * 5) % len(colors)]
            mode = (row + col) % 3
            if mode == 0:
                draw.ellipse((x0, y0, x1, y1), fill=color)
            elif mode == 1:
                draw.rounded_rectangle((x0, y0, x1, y1), radius=cell // 5, fill=color)
            else:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                radius = (x1 - x0) * 0.48
                points = []
                for index in range(10):
                    angle = -math.pi / 2 + index * math.pi / 5
                    r = radius if index % 2 == 0 else radius * 0.45
                    points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
                draw.polygon(points, fill=color)
    return image


def main() -> None:
    args = parse_args()
    if args.size < 224:
        raise ValueError("--size must be at least 224")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "scene.png": create_scene(args.size),
        "text.png": create_text(args.size),
        "dense.png": create_dense(args.size),
    }
    for name, image in images.items():
        output = args.output_dir / name
        image.save(output)
        print(f"Saved sample image: {output}")


if __name__ == "__main__":
    main()
