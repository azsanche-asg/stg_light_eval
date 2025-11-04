"""Placeholder Gaussian splatting baseline for lightweight evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from stg_light_eval.utils.io import ensure_dir


def _load_images(images_dir: Path, limit: int = 10) -> list[np.ndarray]:
    images = []
    for image_path in sorted(images_dir.glob("*.png"))[:limit]:
        images.append(np.array(Image.open(image_path).convert("RGB")))
    return images


def _sample_keypoints(image: np.ndarray, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    coords = np.column_stack((np.random.uniform(0, w, size=num_points), np.random.uniform(0, h, size=num_points)))
    intensities = image.reshape(-1, 3)[
        (coords[:, 1].astype(int).clip(0, h - 1) * w + coords[:, 0].astype(int).clip(0, w - 1))
    ]
    return coords, intensities


def _render_gaussian_proxy(coords: np.ndarray, colors: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    proxy = np.zeros((height, width, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    sigma = max(width, height) * 0.02
    for (x, y), color in zip(coords, colors):
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        proxy += gaussian[..., None] * (color / 255.0)
    proxy = np.clip(proxy, 0.0, 1.0)
    return (proxy * 255).astype(np.uint8)


def _estimate_model_size(num_points: int, attributes: int = 8) -> float:
    return num_points * attributes * 4 / 1e6


def run_stub(images_dir: Path, out_dir: Path, num_points: int = 512) -> dict[str, float]:
    images = _load_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")
    reference = images[0]
    coords, colors = _sample_keypoints(reference, num_points=num_points)
    proxy = _render_gaussian_proxy(coords, colors, reference.shape[:2])

    ensure_dir(out_dir)
    Image.fromarray(proxy).save(out_dir / "proxy.png")

    model_size = _estimate_model_size(num_points)
    metrics = {
        "num_points": float(num_points),
        "model_size_mb": model_size,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gaussian splatting placeholder baseline")
    parser.add_argument("--images_dir", required=True, help="Directory of input PNG images")
    parser.add_argument("--out_dir", required=True, help="Directory to store proxy outputs")
    parser.add_argument("--points", type=int, default=512, help="Number of Gaussian keypoints to sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_stub(Path(args.images_dir).expanduser().resolve(), Path(args.out_dir).expanduser().resolve(), num_points=args.points)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

