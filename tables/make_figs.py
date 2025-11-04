"""Generate illustrative figures for the stg_light_eval project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from stg_light_eval.utils.io import ensure_dir
from stg_light_eval.utils.vis import draw_boxes, overlay_repeat_grid

try:
    from stg_light_eval.metrics import feature_recon_cosine
except Exception:  # pragma: no cover - optional heavy deps
    feature_recon_cosine = None  # type: ignore


FIG_DIR = ensure_dir("tables/figs")


def _load_frame(scene_dir: Path, frame_idx: int = 0) -> np.ndarray:
    frame_path = sorted((scene_dir / "frames").glob("*.png"))[frame_idx]
    return np.array(Image.open(frame_path).convert("RGB"))


def _load_mask(scene_dir: Path, frame_idx: int = 0) -> np.ndarray | None:
    mask_path = scene_dir / "instance_masks.npy"
    if mask_path.exists():
        masks = np.load(mask_path)
        if masks.ndim == 3:
            return masks[frame_idx]
        return masks
    return None


def _mask_bboxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for instance_id in np.unique(mask):
        if instance_id <= 0:
            continue
        ys, xs = np.where(mask == instance_id)
        if ys.size == 0 or xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append((x1, y1, x2, y2))
    return boxes


def _find_scene(root: Path, keyword: str) -> Path | None:
    for path in sorted(root.glob("scene_*")):
        if keyword in path.name:
            return path
    return None


def _build_fig1(data_root: Path, city_root: Path) -> None:
    synth_dir = Path(data_root)
    city_dir = Path(city_root)

    facade_scene = _find_scene(synth_dir, "facade")
    street_scene = _find_scene(synth_dir, "street")
    dynamic_scene = _find_scene(synth_dir, "dynamic")

    if facade_scene is None or street_scene is None:
        print("[warn] Missing facade or street scenes for Fig1")
        return

    panels: List[np.ndarray] = []
    titles = ["Facade", "Street", "Cityscapes A", "Cityscapes B"]

    for scene in [facade_scene, street_scene]:
        frame = _load_frame(scene)
        mask = _load_mask(scene)
        if mask is not None:
            boxes = _mask_bboxes(mask)
            labels = [f"id={i}" for i in range(1, len(boxes) + 1)]
            frame = draw_boxes(frame, boxes, labels=labels)
        panels.append(frame)

    city_frames = sorted((city_dir / "frames").glob("*.png"))[:2]
    for frame_path in city_frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        panels.append(overlay_repeat_grid(img, step=(48, 48)))

    while len(panels) < 4:
        filler = np.zeros_like(panels[0]) if panels else np.zeros((320, 480, 3), dtype=np.uint8)
        panels.append(filler)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig1_grammar_overlays.png", dpi=200)
    plt.close(fig)


def _persistence_curves(masks: np.ndarray) -> Dict[int, np.ndarray]:
    tracks: Dict[int, np.ndarray] = {}
    frames = masks.shape[0]
    ids = [int(i) for i in np.unique(masks) if i > 0]
    for instance_id in ids:
        presence = []
        for t in range(frames):
            presence.append(1 if (masks[t] == instance_id).any() else 0)
        tracks[instance_id] = np.asarray(presence)
    return tracks


def _extract_motion(tree_path: Path) -> Dict[str, Dict[str, List[float]]]:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    motions: Dict[str, Dict[str, List[float]]] = {}

    def _traverse(node: Dict[str, Any]) -> None:  # type: ignore[name-defined]
        if isinstance(node, dict):
            symbol = str(node.get("symbol", len(motions)))
            if "motion" in node:
                motions[symbol] = node["motion"]
            for child in node.get("children", []) or []:
                _traverse(child)
        elif isinstance(node, list):
            for child in node:
                _traverse(child)

    _traverse(tree)
    return motions


def _build_fig2(data_root: Path) -> None:
    dynamic_scene = _find_scene(Path(data_root), "dynamic")
    if dynamic_scene is None:
        print("[warn] No dynamic scene found for Fig2")
        return

    masks = _load_mask(dynamic_scene, frame_idx=0)
    mask_stack = np.load(dynamic_scene / "instance_masks.npy") if (dynamic_scene / "instance_masks.npy").exists() else None
    if mask_stack is None:
        print("[warn] Dynamic scene lacks masks for Fig2")
        return

    tracks = _persistence_curves(mask_stack)
    motions = _extract_motion(dynamic_scene / "gt_tree.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for instance_id, series in tracks.items():
        ax.plot(series, label=f"id {instance_id}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Presence")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Node persistence")
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    ax2 = axes[1]
    frames = np.arange(mask_stack.shape[0])
    for symbol, motion in motions.items():
        start = motion.get("start", [0.0, 0.0])
        velocity = motion.get("velocity", [0.0, 0.0])
        xs = start[0] + velocity[0] * frames
        ys = start[1] + velocity[1] * frames
        ax2.plot(frames, xs, label=f"{symbol} x")
        ax2.plot(frames, ys, linestyle="--", label=f"{symbol} y")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Position (px)")
    ax2.set_title("Motion fits")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_persistence_motion.png", dpi=200)
    plt.close(fig)


def _paste_patch(image: np.ndarray, bbox: Tuple[int, int, int, int], fill_color: Tuple[int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    edited = image.copy()
    edited[y1:y2, x1:x2] = fill_color
    return edited


def _shift_patch(image: np.ndarray, bbox: Tuple[int, int, int, int], dx: int, dy: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    edited = image.copy()
    patch = image[y1:y2, x1:x2].copy()
    edited[y1:y2, x1:x2] = np.median(image, axis=(0, 1)).astype(np.uint8)
    nx1 = np.clip(x1 + dx, 0, image.shape[1] - (x2 - x1))
    ny1 = np.clip(y1 + dy, 0, image.shape[0] - (y2 - y1))
    edited[ny1:ny1 + patch.shape[0], nx1:nx1 + patch.shape[1]] = patch
    return edited


def _clip_delta(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if feature_recon_cosine is None:
        return float("nan")
    try:
        base = feature_recon_cosine(img_a, img_a)
        variant = feature_recon_cosine(img_a, img_b)
        return variant - base
    except Exception:  # pragma: no cover
        return float("nan")


def _build_fig3(data_root: Path) -> None:
    facade_scene = _find_scene(Path(data_root), "facade")
    dynamic_scene = _find_scene(Path(data_root), "dynamic")
    if facade_scene is None or dynamic_scene is None:
        print("[warn] Missing scenes for Fig3")
        return

    facade_frame = _load_frame(facade_scene)
    facade_mask = _load_mask(facade_scene)
    facade_boxes = _mask_bboxes(facade_mask) if facade_mask is not None else []

    # Remove the largest repeated element (heuristic)
    if facade_boxes:
        areas = [(idx, (x2 - x1) * (y2 - y1)) for idx, (x1, y1, x2, y2) in enumerate(facade_boxes)]
        remove_idx = max(areas, key=lambda p: p[1])[0]
        bbox = facade_boxes[remove_idx]
        fill_color = tuple(int(c) for c in np.median(facade_frame, axis=(0, 1)))
        edited_facade = _paste_patch(facade_frame, bbox, fill_color)
    else:
        edited_facade = facade_frame
        bbox = None

    grammar = json.loads((facade_scene / "gt_grammar.json").read_text(encoding="utf-8"))
    rule_count_original = len(grammar.get("productions", []))
    rule_count_edited = max(rule_count_original - 1, 0)

    dynamic_frame = _load_frame(dynamic_scene)
    dynamic_mask = _load_mask(dynamic_scene)
    dynamic_boxes = _mask_bboxes(dynamic_mask) if dynamic_mask is not None else []

    if dynamic_boxes:
        actor_bbox = dynamic_boxes[0]
        zero_motion_frame = _shift_patch(dynamic_frame, actor_bbox, dx=0, dy=0)
        double_motion_frame = _shift_patch(dynamic_frame, actor_bbox, dx=20, dy=5)
    else:
        zero_motion_frame = dynamic_frame
        double_motion_frame = dynamic_frame

    clip_delta = _clip_delta(zero_motion_frame, double_motion_frame)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(facade_frame)
    axes[0, 0].set_title(f"Original rules: {rule_count_original}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edited_facade)
    subtitle = f"After deletion: {rule_count_edited}"
    if bbox is not None:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor="red", linewidth=2)
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(subtitle)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(zero_motion_frame)
    axes[1, 0].set_title("Counterfactual motion ×0")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(double_motion_frame)
    axes[1, 1].set_title(f"Counterfactual motion ×2\nΔCLIP ≈ {clip_delta:.3f}" if np.isfinite(clip_delta) else "Counterfactual motion ×2")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_edits_counterfactuals.png", dpi=200)
    plt.close(fig)


def main() -> None:
    data_root = Path("data/synth")
    city_root = Path("data/cityscapes")

    if not data_root.exists():
        print("[warn] data/synth not found; figures will be skipped.")
        return

    ensure_dir(FIG_DIR)
    _build_fig1(data_root, city_root)
    _build_fig2(data_root)
    _build_fig3(data_root)
    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
