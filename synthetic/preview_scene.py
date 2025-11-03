"""Visualization helper to preview generated synthetic scenes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stg_light_eval.utils.io import load_json
from stg_light_eval.utils.vis import draw_boxes, montage

try:
    from matplotlib import cm
except ImportError as exc:  # pragma: no cover - matplotlib should exist but fail gracefully.
    raise SystemExit("matplotlib is required for preview_scene.py") from exc


def _load_frame(scene_dir: Path, frame_index: int) -> Image.Image:
    frame_path = scene_dir / "frames" / f"{frame_index:03d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    return Image.open(frame_path).convert("RGB")


def _load_depth(scene_dir: Path, frame_index: int) -> np.ndarray:
    depth_path = scene_dir / "depth.npy"
    depth = np.load(depth_path)
    if depth.ndim == 3:
        return depth[frame_index]
    return depth


def _load_masks(scene_dir: Path, frame_index: int) -> np.ndarray:
    mask_path = scene_dir / "instance_masks.npy"
    masks = np.load(mask_path)
    if masks.ndim == 3:
        return masks[frame_index]
    return masks


def _mask_bboxes(mask: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for instance_id in np.unique(mask):
        if instance_id <= 0:
            continue
        ys, xs = np.where(mask == instance_id)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        boxes.append((int(instance_id), (int(x1), int(y1), int(x2), int(y2))))
    return boxes


def _format_grammar(grammar: dict) -> list[str]:
    lines = [f"Scene type: {grammar.get('scene_type', 'unknown')}", ""]
    for prod in grammar.get("productions", []):
        ptype = prod.get("type", "?")
        if ptype == "Split":
            parent = prod.get("parent", "?")
            axis = prod.get("axis", "?")
            children = ", ".join(prod.get("children", []))
            lines.append(f"Split[{axis}] {parent} -> {children}")
        elif ptype == "Repeat":
            parent = prod.get("parent", "?")
            axis = prod.get("axis", "?")
            count = prod.get("count", '?')
            symbol = prod.get("symbol", "?")
            lines.append(f"Repeat[{axis}] {parent} x{count} ({symbol})")
        else:
            lines.append(str(prod))
    return lines


def _text_panel(lines: Sequence[str], width: int, height: int) -> np.ndarray:
    panel = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - default font usually available.
        font = None  # type: ignore[assignment]
    margin = 8
    line_height = 14 if font is None else font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
    y = margin
    for line in lines:
        if y > height - margin:
            break
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return np.array(panel)


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    norm = (depth - d_min) / (d_max - d_min + 1e-6)
    colored = cm.get_cmap("viridis")(norm)
    return (colored[..., :3] * 255).astype(np.uint8)


def build_preview(scene_dir: Path, frame_index: int) -> tuple[np.ndarray, Path]:
    frame = _load_frame(scene_dir, frame_index)
    depth = _load_depth(scene_dir, frame_index)
    mask = _load_masks(scene_dir, frame_index)
    grammar = load_json(scene_dir / "gt_grammar.json")

    frame_np = np.array(frame)
    instance_boxes = _mask_bboxes(mask)
    boxes = [bbox for _, bbox in instance_boxes]
    labels = [f"id={instance_id}" for instance_id, _ in instance_boxes]
    overlay_np = draw_boxes(frame_np, boxes, labels=labels, copy=True)

    text_lines = _format_grammar(grammar)
    text_np = _text_panel(text_lines, frame_np.shape[1], frame_np.shape[0])

    depth_np = _depth_to_rgb(depth)

    grid = montage([frame_np, overlay_np, text_np, depth_np], cols=2, tile_shape=frame_np.shape[:2])
    output_path = scene_dir / f"preview_{frame_index:03d}.png"
    Image.fromarray(grid).save(output_path)
    return grid, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 2x2 preview montage for a generated scene.")
    parser.add_argument("--scene", required=True, help="Path to a synthetic scene directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene).expanduser().resolve()
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    _, output = build_preview(scene_dir, args.frame)
    print(f"Preview saved to {output}")


if __name__ == "__main__":
    main()
