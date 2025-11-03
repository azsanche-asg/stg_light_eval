"""Procedural synthetic scene generator for the stg_light_eval toolkit."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from stg_light_eval.utils.io import ensure_dir, save_json, set_seed

WIDTH = 480
HEIGHT = 320


@dataclass
class StaticRect:
    label: str
    bbox: tuple[int, int, int, int]
    color: tuple[int, int, int]
    depth: float
    mask_id: int


@dataclass
class MovingRect:
    label: str
    color: tuple[int, int, int]
    size: tuple[int, int]
    start: tuple[float, float]
    velocity: tuple[float, float]
    depth: float
    mask_id: int

    def bbox(self, frame_idx: int) -> tuple[int, int, int, int]:
        cx = self.start[0] + self.velocity[0] * frame_idx
        cy = self.start[1] + self.velocity[1] * frame_idx
        half_w = self.size[0] / 2
        half_h = self.size[1] / 2
        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_h))
        return x1, y1, x2, y2


def _vertical_gradient(width: int, height: int, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> Image.Image:
    top = np.array(top_color, dtype=np.float32)
    bottom = np.array(bottom_color, dtype=np.float32)
    ratios = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = top + (bottom - top) * ratios
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    gradient = np.repeat(gradient[:, None, :], width, axis=1)
    return Image.fromarray(gradient, mode="RGB")


def _base_depth_plane(width: int, height: int, rng: np.random.Generator, base_depth: float = 12.0) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    grad_x = rng.uniform(-1.0, 1.0)
    grad_y = rng.uniform(-1.0, 1.0)
    plane = base_depth + grad_x * xx + grad_y * yy
    return plane.astype(np.float32)


def _add_noise(depth: np.ndarray, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=depth.shape).astype(np.float32)
    return depth + noise


def _clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _render_static_frame(
    background: Image.Image,
    objects: Sequence[StaticRect],
    base_depth: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = background.copy()
    if jitter > 0:
        arr = np.array(image, dtype=np.float32)
        arr += rng.normal(0.0, jitter, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(image)

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    depth = base_depth.copy()

    for rect in objects:
        x1, y1, x2, y2 = rect.bbox
        draw.rectangle(rect.bbox, fill=rect.color)
        mask[y1:y2, x1:x2] = rect.mask_id
        depth[y1:y2, x1:x2] = rect.depth

    depth = _add_noise(depth, rng)
    return np.array(image, dtype=np.uint8), depth.astype(np.float32), mask


def _render_dynamic_frame(
    background: Image.Image,
    static_objects: Sequence[StaticRect],
    movers: Sequence[MovingRect],
    frame_idx: int,
    base_depth: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    image = background.copy()
    draw = ImageDraw.Draw(image)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    depth = base_depth.copy()
    boxes: list[tuple[int, int, int, int]] = []

    for rect in static_objects:
        draw.rectangle(rect.bbox, fill=rect.color)
        x1, y1, x2, y2 = rect.bbox
        mask[y1:y2, x1:x2] = rect.mask_id
        depth[y1:y2, x1:x2] = rect.depth

    for mover in movers:
        raw_bbox = mover.bbox(frame_idx)
        bbox = _clamp_bbox(raw_bbox, WIDTH, HEIGHT)
        x1, y1, x2, y2 = bbox
        draw.rectangle(bbox, fill=mover.color)
        mask[y1:y2, x1:x2] = mover.mask_id
        depth[y1:y2, x1:x2] = mover.depth
        boxes.append(bbox)

    depth = _add_noise(depth, rng)
    return np.array(image, dtype=np.uint8), depth.astype(np.float32), mask, boxes


def _facade_scene(rng: np.random.Generator) -> tuple[Image.Image, list[StaticRect], dict, dict]:
    building_color = tuple(int(x) for x in rng.integers(140, 220, size=3))
    window_color = tuple(int(x) for x in rng.integers(40, 120, size=3))
    accent_color = tuple(int(x) for x in rng.integers(100, 160, size=3))

    background = _vertical_gradient(WIDTH, HEIGHT, (120, 160, 200), (200, 210, 220))
    draw = ImageDraw.Draw(background)
    ground_height = int(HEIGHT * 0.15)
    draw.rectangle((0, HEIGHT - ground_height, WIDTH, HEIGHT), fill=(90, 100, 110))

    margin_x = int(WIDTH * 0.08)
    margin_y = int(HEIGHT * 0.1)
    building_bbox = (margin_x, margin_y, WIDTH - margin_x, HEIGHT - ground_height)

    objects: list[StaticRect] = []
    mask_id = 1
    objects.append(
        StaticRect(
            label="Building",
            bbox=building_bbox,
            color=building_color,
            depth=8.0,
            mask_id=mask_id,
        )
    )

    n_floors = int(rng.integers(3, 6))
    n_columns = int(rng.integers(3, 6))
    floor_height = (building_bbox[3] - building_bbox[1]) / n_floors
    col_width = (building_bbox[2] - building_bbox[0]) / n_columns

    mask_id += 1
    for floor in range(n_floors):
        for col in range(n_columns):
            inset_x = col_width * 0.15
            inset_y = floor_height * 0.15
            x1 = int(building_bbox[0] + col * col_width + inset_x)
            y1 = int(building_bbox[1] + floor * floor_height + inset_y)
            x2 = int(building_bbox[0] + (col + 1) * col_width - inset_x)
            y2 = int(building_bbox[1] + (floor + 1) * floor_height - inset_y)
            objects.append(
                StaticRect(
                    label=f"Window_{floor}_{col}",
                    bbox=(x1, y1, x2, y2),
                    color=window_color,
                    depth=7.0 - 0.1 * floor,
                    mask_id=mask_id,
                )
            )
            mask_id += 1

    door_width = int(col_width * 0.6)
    door_height = int(floor_height * 1.2)
    door_x1 = int((building_bbox[0] + building_bbox[2]) / 2 - door_width / 2)
    door_x2 = door_x1 + door_width
    door_y2 = building_bbox[3]
    door_y1 = door_y2 - door_height
    objects.append(
        StaticRect(
            label="Door",
            bbox=(door_x1, door_y1, door_x2, door_y2),
            color=accent_color,
            depth=7.5,
            mask_id=mask_id,
        )
    )

    productions = [
        {
            "type": "Split",
            "parent": "Scene",
            "axis": "y",
            "children": ["Sky", "Building", "Ground"],
        },
        {
            "type": "Repeat",
            "parent": "Building",
            "axis": "y",
            "count": n_floors,
            "symbol": "Floor",
        },
        {
            "type": "Repeat",
            "parent": "Floor",
            "axis": "x",
            "count": n_columns,
            "symbol": "Window",
        },
    ]

    tree = {
        "symbol": "Scene",
        "children": [
            {"symbol": "Sky"},
            {
                "symbol": "Building",
                "children": [
                    {
                        "symbol": f"Floor_{floor}",
                        "children": [
                            {"symbol": f"Window_{floor}_{col}"} for col in range(n_columns)
                        ],
                    }
                    for floor in range(n_floors)
                ]
                + [{"symbol": "Door"}],
            },
            {"symbol": "Ground"},
        ],
    }

    return background, objects, {"scene_type": "facade", "productions": productions}, tree


def _street_scene(rng: np.random.Generator) -> tuple[Image.Image, list[StaticRect], dict, dict]:
    sky = _vertical_gradient(WIDTH, HEIGHT, (90, 140, 200), (130, 170, 210))
    image = sky.copy()
    draw = ImageDraw.Draw(image)

    horizon = int(HEIGHT * 0.45)
    draw.rectangle((0, horizon, WIDTH, HEIGHT), fill=(60, 60, 60))

    sidewalk_h = int(HEIGHT * 0.08)
    draw.rectangle((0, HEIGHT - sidewalk_h, WIDTH, HEIGHT), fill=(110, 110, 110))

    lane_color = (240, 220, 120)
    n_lanes = int(rng.integers(2, 4))
    lane_width = WIDTH / (n_lanes + 1)

    objects: list[StaticRect] = []
    mask_id = 1

    road_depth = 12.0
    sidewalk_depth = 11.0
    building_depth = 9.0

    sidewalk_rect = StaticRect(
        label="Sidewalk",
        bbox=(0, HEIGHT - sidewalk_h, WIDTH, HEIGHT),
        color=(110, 110, 110),
        depth=sidewalk_depth,
        mask_id=mask_id,
    )
    objects.append(sidewalk_rect)
    mask_id += 1

    building_band_h = int(HEIGHT * 0.2)
    building_color = tuple(int(x) for x in rng.integers(100, 160, size=3))
    buildings_rect = StaticRect(
        label="Buildings",
        bbox=(0, 0, WIDTH, building_band_h),
        color=building_color,
        depth=building_depth,
        mask_id=mask_id,
    )
    objects.append(buildings_rect)
    mask_id += 1

    lane_rects: list[StaticRect] = []
    for lane in range(n_lanes):
        cx = (lane + 1) * lane_width
        lane_box = (
            int(cx - 3),
            horizon,
            int(cx + 3),
            HEIGHT - sidewalk_h,
        )
        lane_rects.append(
            StaticRect(
                label=f"Lane_{lane}",
                bbox=lane_box,
                color=lane_color,
                depth=road_depth - 0.2,
                mask_id=mask_id,
            )
        )
        mask_id += 1

    objects.extend(lane_rects)

    productions = [
        {
            "type": "Split",
            "parent": "Scene",
            "axis": "y",
            "children": ["Sky", "Buildings", "Road", "Sidewalk"],
        },
        {
            "type": "Repeat",
            "parent": "Road",
            "axis": "x",
            "count": n_lanes,
            "symbol": "Lane",
        },
    ]

    tree = {
        "symbol": "Scene",
        "children": [
            {"symbol": "Sky"},
            {"symbol": "Buildings"},
            {
                "symbol": "Road",
                "children": [{"symbol": f"Lane_{lane}"} for lane in range(n_lanes)],
            },
            {"symbol": "Sidewalk"},
        ],
    }

    return image, objects, {"scene_type": "street", "productions": productions}, tree


def _dynamic_scene(rng: np.random.Generator) -> tuple[Image.Image, list[StaticRect], list[MovingRect], dict, dict]:
    base = _vertical_gradient(WIDTH, HEIGHT, (100, 160, 210), (120, 180, 220))
    image = base.copy()
    draw = ImageDraw.Draw(image)

    horizon = int(HEIGHT * 0.4)
    draw.rectangle((0, horizon, WIDTH, HEIGHT), fill=(70, 70, 70))
    draw.rectangle((0, HEIGHT - int(HEIGHT * 0.1), WIDTH, HEIGHT), fill=(120, 120, 120))

    static_objects: list[StaticRect] = []
    mask_id = 1
    static_objects.append(
        StaticRect(
            label="BackgroundRoad",
            bbox=(0, horizon, WIDTH, HEIGHT),
            color=(70, 70, 70),
            depth=13.0,
            mask_id=mask_id,
        )
    )
    mask_id += 1

    movers: list[MovingRect] = []
    categories = [
        ("vehicle", (220, 80, 60), (80, 40), (-60.0, HEIGHT * 0.7), (6.0, 0.0), 8.5),
        ("pedestrian", (60, 200, 120), (30, 60), (WIDTH * 0.2, HEIGHT * 0.78), (1.2, 0.0), 9.5),
        ("drone", (200, 200, 80), (40, 25), (WIDTH * 0.3, HEIGHT * 0.25), (2.4, 0.3), 7.5),
    ]

    for label, color, size, start, velocity, depth in categories:
        mover = MovingRect(
            label=label,
            color=color,
            size=size,
            start=(start[0] + rng.uniform(-15, 15), start[1] + rng.uniform(-10, 10)),
            velocity=(velocity[0] * rng.uniform(0.8, 1.2), velocity[1] * rng.uniform(0.8, 1.2)),
            depth=depth,
            mask_id=mask_id,
        )
        movers.append(mover)
        mask_id += 1

    productions = [
        {
            "type": "Split",
            "parent": "Scene",
            "axis": "y",
            "children": ["Sky", "Traffic"],
        },
        {
            "type": "Repeat",
            "parent": "Traffic",
            "axis": "time",
            "count": len(movers),
            "symbol": "Actor",
        },
    ]

    tree = {
        "symbol": "Scene",
        "children": [
            {"symbol": "Sky"},
            {
                "symbol": "Traffic",
                "children": [
                    {
                        "symbol": f"{mover.label.title()}_{idx}",
                        "motion": {
                            "start": [float(mover.start[0]), float(mover.start[1])],
                            "velocity": [float(mover.velocity[0]), float(mover.velocity[1])],
                        },
                    }
                    for idx, mover in enumerate(movers)
                ],
            },
        ],
    }

    return image, static_objects, movers, {"scene_type": "dynamic", "productions": productions}, tree


def _scene_types_from_args(n_scenes: int, types: Sequence[str]) -> list[str]:
    order: list[str] = []
    while len(order) < n_scenes:
        order.extend(types)
    return order[:n_scenes]


def _save_metadata(scene_dir: Path, grammar: dict, tree: dict) -> None:
    save_json(grammar, scene_dir / "gt_grammar.json")
    save_json(tree, scene_dir / "gt_tree.json")


def _generate_facade(scene_dir: Path, frame_indices: Sequence[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    background, objects, grammar, tree = _facade_scene(rng)
    base_depth = _base_depth_plane(WIDTH, HEIGHT, rng, base_depth=12.0)

    frames: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for _ in frame_indices:
        frame, depth, mask = _render_static_frame(background, objects, base_depth, rng, jitter=1.2)
        frames.append(frame)
        depths.append(depth)
        masks.append(mask)

    frames_np = np.stack(frames, axis=0)
    depth_np = np.stack(depths, axis=0)
    mask_np = np.stack(masks, axis=0)

    _save_metadata(scene_dir, grammar, tree)
    return depth_np, mask_np, frames_np


def _generate_street(scene_dir: Path, frame_indices: Sequence[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    background, objects, grammar, tree = _street_scene(rng)
    base_depth = _base_depth_plane(WIDTH, HEIGHT, rng, base_depth=14.0)

    frames: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for _ in frame_indices:
        frame, depth, mask = _render_static_frame(background, objects, base_depth, rng, jitter=2.0)
        frames.append(frame)
        depths.append(depth)
        masks.append(mask)

    frames_np = np.stack(frames, axis=0)
    depth_np = np.stack(depths, axis=0)
    mask_np = np.stack(masks, axis=0)

    _save_metadata(scene_dir, grammar, tree)
    return depth_np, mask_np, frames_np


def _generate_dynamic(scene_dir: Path, frame_indices: Sequence[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    background, statics, movers, grammar, tree = _dynamic_scene(rng)
    base_depth = _base_depth_plane(WIDTH, HEIGHT, rng, base_depth=16.0)

    frames: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    tracks: list[list[tuple[int, int, int, int]]] = []

    for idx in frame_indices:
        frame, depth, mask, boxes = _render_dynamic_frame(background, statics, movers, idx, base_depth, rng)
        frames.append(frame)
        depths.append(depth)
        masks.append(mask)
        tracks.append([list(map(int, box)) for box in boxes])

    frames_np = np.stack(frames, axis=0)
    depth_np = np.stack(depths, axis=0)
    mask_np = np.stack(masks, axis=0)

    tree_with_tracks = tree.copy()
    tree_with_tracks = dict(tree_with_tracks)
    traffic = tree_with_tracks["children"][1]
    for entry, track in zip(traffic["children"], tracks[0]):
        entry["bbox_first_frame"] = [int(v) for v in track]

    _save_metadata(scene_dir, grammar, tree_with_tracks)
    return depth_np, mask_np, frames_np


SCENE_GENERATORS = {
    "facade": _generate_facade,
    "street": _generate_street,
    "dynamic": _generate_dynamic,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate procedural synthetic scenes.")
    parser.add_argument("--out", default="data/synth", help="Output directory for generated scenes")
    parser.add_argument("--n", type=int, default=30, help="Number of scenes to generate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--types",
        default="facade,street,dynamic",
        help="Comma separated list of scene types to cycle through",
    )
    parser.add_argument("--frames", type=int, default=30, help="Number of frames per sequence (inclusive index)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    if not types:
        raise ValueError("At least one scene type must be specified")

    for scene_type in types:
        if scene_type not in SCENE_GENERATORS:
            raise ValueError(f"Unsupported scene type: {scene_type}")

    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    frame_indices = list(range(args.frames + 1))

    scene_types = _scene_types_from_args(args.n, types)

    for idx, scene_type in enumerate(scene_types):
        scene_name = f"scene_{idx:03d}_{scene_type}"
        scene_dir = ensure_dir(Path(out_dir) / scene_name)
        frames_dir = ensure_dir(scene_dir / "frames")
        rng = np.random.default_rng(args.seed + idx)
        random.seed(args.seed + idx)

        generator = SCENE_GENERATORS[scene_type]
        depth_np, mask_np, frames_np = generator(scene_dir, frame_indices, rng)

        for frame_idx, frame in zip(frame_indices, frames_np):
            frame_image = Image.fromarray(frame, mode="RGB")
            frame_path = frames_dir / f"{frame_idx:03d}.png"
            frame_image.save(frame_path)

        np.save(scene_dir / "depth.npy", depth_np.astype(np.float32))
        np.save(scene_dir / "instance_masks.npy", mask_np.astype(np.int16))
        print(f"Generated {scene_name} -> frames: {len(frame_indices)}, depth: {depth_np.shape}, masks: {mask_np.shape}")


if __name__ == "__main__":
    main()
