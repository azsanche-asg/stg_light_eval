"""Visualization utilities for debugging scene grammar experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import networkx as nx
import numpy as np


Box = Sequence[float | int]


def _to_np(image: np.ndarray | Sequence) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(image)


def draw_boxes(
    image: np.ndarray | Sequence,
    boxes: Iterable[Box],
    labels: Sequence[str] | None = None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """Overlay axis-aligned boxes (x1, y1, x2, y2) onto an image."""
    canvas = _to_np(image)
    if copy:
        canvas = canvas.copy()

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and idx < len(labels):
            text = labels[idx]
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            text_origin = (x1, max(0, y1 - baseline - 2))
            cv2.rectangle(canvas, (text_origin[0], text_origin[1] - th - baseline), (text_origin[0] + tw, text_origin[1] + baseline), color, -1)
            cv2.putText(
                canvas,
                text,
                (text_origin[0], text_origin[1] - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    return canvas


def overlay_repeat_grid(
    image: np.ndarray | Sequence,
    *,
    step: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 1,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a regular grid to debug spatial tiling assumptions."""
    canvas = _to_np(image)
    overlay = canvas.copy()
    height, width = overlay.shape[:2]
    step_x, step_y = step

    for x in range(0, width, step_x):
        cv2.line(overlay, (x, 0), (x, height), color, thickness)
    for y in range(0, height, step_y):
        cv2.line(overlay, (0, y), (width, y), color, thickness)

    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def draw_parse_tree(
    tree: nx.DiGraph | Mapping[str, Sequence[str]] | Sequence,
    *,
    root: str | None = None,
    indent: int = 2,
) -> str:
    """Return a textual representation of a parse tree."""

    def _render(node: str, depth: int, graph: nx.DiGraph, lines: list[str]) -> None:
        lines.append(" " * (indent * depth) + str(node))
        for child in graph.successors(node):
            _render(child, depth + 1, graph, lines)

    if isinstance(tree, nx.DiGraph):
        graph = tree
    elif isinstance(tree, Mapping):
        graph = nx.DiGraph()
        for parent, children in tree.items():
            for child in children:
                graph.add_edge(parent, child)
    else:
        graph = nx.DiGraph()
        for parent, children in tree:  # type: ignore[assignment]
            for child in children:
                graph.add_edge(parent, child)

    if root is None:
        candidates = [node for node in graph.nodes if graph.in_degree(node) == 0]
        if not candidates:
            raise ValueError("cannot infer root from cyclic graph")
        root = candidates[0]

    lines: list[str] = []
    _render(root, 0, graph, lines)
    return "\n".join(lines)


def montage(
    images: Sequence[np.ndarray | Sequence],
    *,
    cols: int | None = None,
    tile_shape: tuple[int, int] | None = None,
    fill_value: int = 0,
    border: int = 2,
) -> np.ndarray:
    """Tile images into a simple grid for qualitative inspection."""
    if not images:
        raise ValueError("at least one image is required")

    np_images = [_to_np(img) for img in images]
    first = np_images[0]

    if tile_shape is None:
        tile_shape = first.shape[:2]

    resized = []
    for img in np_images:
        if img.shape[:2] != tile_shape:
            resized_img = cv2.resize(img, (tile_shape[1], tile_shape[0]))
        else:
            resized_img = img
        resized.append(resized_img)

    num_images = len(resized)
    if cols is None:
        cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))

    channels = resized[0].shape[2] if resized[0].ndim == 3 else 1
    tile_h, tile_w = tile_shape
    canvas_shape = (
        rows * tile_h + border * (rows + 1),
        cols * tile_w + border * (cols + 1),
        channels,
    )
    canvas = np.full(canvas_shape, fill_value, dtype=resized[0].dtype)

    for idx, img in enumerate(resized):
        r = idx // cols
        c = idx % cols
        top = border + r * (tile_h + border)
        left = border + c * (tile_w + border)
        region = canvas[top : top + tile_h, left : left + tile_w]
        if img.ndim == 2 and channels == 1:
            region[:, :] = img[:, :, None]
        elif img.ndim == 2:
            region[:, :, 0] = img
        else:
            region[:, :, : img.shape[2]] = img

    if channels == 1 and first.ndim == 2:
        return canvas[:, :, 0]
    return canvas

