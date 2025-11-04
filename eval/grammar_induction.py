"""Minimal placeholder for grammar induction over synthetic scenes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image


def _load_frames(path: Path, limit: int | None = None) -> list[np.ndarray]:
    frames = []
    for frame_file in sorted(path.glob("*.png"))[:limit]:
        frames.append(np.array(Image.open(frame_file).convert("RGB")))
    return frames


def _detect_peaks(signal: np.ndarray, threshold: float = 0.2) -> list[int]:
    fft = np.fft.rfft(signal)
    magnitudes = np.abs(fft)
    magnitudes[0] = 0
    peaks = []
    for idx in range(1, len(magnitudes)):
        if magnitudes[idx] > threshold * magnitudes.max():
            peaks.append(idx)
    return peaks


def _analyze_axis(image: np.ndarray, axis: int, threshold: float = 0.2) -> tuple[list[int], float]:
    projection = image.mean(axis=axis)
    flattened = projection.mean(axis=-1)
    peaks = _detect_peaks(flattened, threshold=threshold)
    spacing = 0.0
    if len(peaks) >= 2:
        spacing = float(np.mean(np.diff(peaks)))
    return peaks, spacing


def _build_rules(image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    h_peaks, h_spacing = _analyze_axis(image, axis=1, threshold=0.2)
    v_peaks, v_spacing = _analyze_axis(image, axis=0, threshold=0.2)

    rules: list[dict[str, Any]] = []
    tree_children: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    if h_spacing > 1:
        rules.append({
            "type": "Repeat",
            "parent": "Scene",
            "axis": "y",
            "count": len(h_peaks),
            "spacing": h_spacing,
        })
        nodes.append({"axis": "y", "spacing": h_spacing, "pos": h_peaks})

    if v_spacing > 1:
        rules.append({
            "type": "Repeat",
            "parent": "Scene",
            "axis": "x",
            "count": len(v_peaks),
            "spacing": v_spacing,
        })
        nodes.append({"axis": "x", "spacing": v_spacing, "pos": v_peaks})

    splits = []
    if h_peaks:
        splits.append({"axis": "y", "positions": h_peaks})
    if v_peaks:
        splits.append({"axis": "x", "positions": v_peaks})

    for split in splits:
        rules.append({"type": "Split", "parent": "Scene", **split})
        tree_children.append({"symbol": f"Split_{split['axis']}", "positions": split["positions"]})

    tree = {"symbol": "Scene", "children": tree_children or [{"symbol": "Region"}]}

    return rules, tree, nodes


def _compute_motion(mask_stack: np.ndarray) -> dict[str, dict[str, float]]:
    track_centroids: dict[int, list[Tuple[float, float]]] = defaultdict(list)
    num_frames = mask_stack.shape[0]
    for t in range(num_frames):
        mask = mask_stack[t]
        ids = np.unique(mask)
        for instance_id in ids:
            if instance_id <= 0:
                continue
            ys, xs = np.where(mask == instance_id)
            if xs.size == 0:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            track_centroids[instance_id].append((t, cx, cy))

    motion_params: dict[str, dict[str, float]] = {}
    for instance_id, points in track_centroids.items():
        if len(points) < 2:
            continue
        times = np.array([p[0] for p in points], dtype=np.float32)
        xs = np.array([p[1] for p in points], dtype=np.float32)
        ys = np.array([p[2] for p in points], dtype=np.float32)
        vx, x0 = np.polyfit(times, xs, 1)
        vy, y0 = np.polyfit(times, ys, 1)
        motion_params[str(instance_id)] = {
            "start": [float(x0), float(y0)],
            "velocity": [float(vx), float(vy)],
        }
    return motion_params


def induce_grammar(scene_path: Path, dynamic: bool = False) -> dict[str, Any]:
    frames = _load_frames(scene_path / "frames", limit=5)
    if not frames:
        raise FileNotFoundError(f"No frames found in {scene_path}")
    reference = frames[0]
    pred_rules, pred_tree, pred_nodes = _build_rules(reference)

    pred_motion = {}
    if dynamic and (scene_path / "instance_masks.npy").exists():
        masks = np.load(scene_path / "instance_masks.npy")
        pred_motion = _compute_motion(masks[: len(frames)])

    return {
        "pred_rules": pred_rules,
        "pred_tree": pred_tree,
        "pred_nodes": pred_nodes,
        "pred_motion": pred_motion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal grammar induction placeholder")
    parser.add_argument("--scene", required=True, help="Path to a scene directory")
    parser.add_argument("--dynamic", action="store_true", help="Treat scene as dynamic and estimate motion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    result = induce_grammar(scene_path, dynamic=args.dynamic)
    print(result)


if __name__ == "__main__":
    main()
