"""Slot Attention baseline stub using SLIC superpixels and K-Means grouping."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import color, segmentation
from sklearn.cluster import KMeans

from stg_light_eval.utils.io import ensure_dir


def _load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def _segment_superpixels(image: np.ndarray, target_segments: int) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    image_lab = color.rgb2lab(image_float)
    segments = segmentation.slic(image_lab, n_segments=target_segments, compactness=10.0, start_label=0)
    return segments


def _compute_superpixel_features(image: np.ndarray, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    segment_ids = np.unique(segments)
    features = []
    for seg_id in segment_ids:
        mask = segments == seg_id
        coords = np.column_stack(np.nonzero(mask))
        colors = image[mask]
        centroid_yx = coords.mean(axis=0)
        mean_color = colors.mean(axis=0)
        features.append(np.concatenate([mean_color, centroid_yx]))
    return np.array(features, dtype=np.float32), segment_ids


def _group_into_slots(features: np.ndarray, segment_ids: np.ndarray, slots: int) -> np.ndarray:
    slots = min(slots, len(segment_ids))
    kmeans = KMeans(n_clusters=slots, random_state=0, n_init=10)
    assignments = kmeans.fit_predict(features)
    slot_map = {seg_id: assign for seg_id, assign in zip(segment_ids, assignments)}
    return np.array([slot_map[seg_id] for seg_id in segment_ids], dtype=np.int32)


def _save_slot_masks(
    segments: np.ndarray,
    segment_ids: np.ndarray,
    slot_assignments: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    ensure_dir(out_dir)
    masks = []
    for slot_id in np.unique(slot_assignments):
        mask = np.isin(segments, segment_ids[slot_assignments == slot_id])
        mask_img = (mask.astype(np.uint8) * 255)
        mask_path = out_dir / f"slot_{slot_id:02d}.png"
        Image.fromarray(mask_img).save(mask_path)
        masks.append(mask_path)
    return masks


def _slot_metrics(image: np.ndarray, masks: list[np.ndarray]) -> list[tuple[int, float, float]]:
    h, w = image.shape[:2]
    total_pixels = h * w
    metrics = []
    for idx, mask in enumerate(masks):
        area = mask.sum()
        coverage = float(area) / float(total_pixels)
        if area == 0:
            quality = 0.0
        else:
            slot_pixels = image[mask.astype(bool)]
            color_var = np.mean(slot_pixels.astype(np.float32).var(axis=0))
            quality = float(np.exp(-color_var / 1000.0))
        metrics.append((idx, coverage, quality))
    return metrics


def run_stub(image_path: Path, out_dir: Path, slots: int) -> None:
    ensure_dir(out_dir)
    image = _load_image(image_path)
    segments = _segment_superpixels(image, target_segments=max(slots * 4, 30))
    features, segment_ids = _compute_superpixel_features(image, segments)
    assignments = _group_into_slots(features, segment_ids, slots)

    mask_arrays: list[np.ndarray] = []
    for slot_id in range(slots):
        mask = np.isin(segments, segment_ids[assignments == slot_id]).astype(np.uint8)
        mask_arrays.append(mask)
        Image.fromarray(mask * 255).save(out_dir / f"slot_{slot_id:02d}.png")

    metrics = _slot_metrics(image, mask_arrays)
    csv_path = out_dir / "slot_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slot_id", "coverage", "quality"])
        for slot_id, coverage, quality in metrics:
            writer.writerow([slot_id, f"{coverage:.6f}", f"{quality:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slot Attention placeholder using SLIC + KMeans")
    parser.add_argument("--image", required=True, help="Path to an RGB image")
    parser.add_argument("--out_dir", required=True, help="Directory to save masks and CSV")
    parser.add_argument("--slots", type=int, default=8, help="Number of slots to produce")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stub(Path(args.image).expanduser().resolve(), Path(args.out_dir).expanduser().resolve(), slots=args.slots)


if __name__ == "__main__":
    main()
