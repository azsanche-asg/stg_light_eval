"""Evaluation over real datasets (LLFF, Cityscapes, optional RGB-D mini)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from stg_light_eval.metrics import (
    depth_rmse,
    feature_recon_cosine,
    normal_agreement,
    repeat_regularity_error,
)
from stg_light_eval.utils.io import ensure_dir

from stg_light_eval.baselines.gsplat.run_gsplat_stub import run_stub as run_gsplat_stub
from stg_light_eval.baselines.slotattn.run_slotattn_stub import run_stub as run_slotattn_stub
from stg_light_eval.baselines.pointtracks.run_pointtracks_stub import run_stub as run_pointtracks_stub


TABLE_COLUMNS = [
    "dataset",
    "method",
    "feature_cosine",
    "multiview_consistency",
    "repeat_regularity",
    "fourier_grid",
    "depth_rmse",
    "normal_agreement",
    "track_persistence",
]


def _load_pngs(directory: Path, limit: int | None = None) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in sorted(directory.glob("*.png"))[:limit]:
        images.append(np.array(Image.open(path).convert("RGB")))
    return images


def _spectral_peaks(profile: np.ndarray, top_k: int = 8) -> np.ndarray:
    if profile.ndim != 1 or profile.size == 0:
        return np.zeros(0, dtype=np.float32)
    normed = (profile - profile.min()) / (profile.ptp() + 1e-6)
    idx = np.argpartition(normed, -top_k)[-top_k:]
    idx = np.sort(idx)
    return idx.astype(np.float32) / (len(profile) - 1 + 1e-6)


def _grid_nodes(image: np.ndarray) -> List[Dict[str, Any]]:
    gray = image.mean(axis=2).astype(np.float32)
    vertical_profile = gray.mean(axis=0)
    horizontal_profile = gray.mean(axis=1)
    xs = _spectral_peaks(vertical_profile)
    ys = _spectral_peaks(horizontal_profile)
    nodes: List[Dict[str, Any]] = []
    for x in xs:
        nodes.append({"pos": np.array([x, 0.5], dtype=np.float32)})
    for y in ys:
        nodes.append({"pos": np.array([0.5, y], dtype=np.float32)})
    return nodes


def _feature_similarity_pairs(images: List[np.ndarray], step: int = 1) -> float:
    if len(images) < 2:
        return math.nan
    sims: List[float] = []
    for idx in range(len(images) - step):
        sims.append(feature_recon_cosine(images[idx], images[idx + step]))
    return float(np.nanmean(sims)) if sims else math.nan


def _fourier_grid_score(image: np.ndarray) -> float:
    gray = image.mean(axis=2).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft)
    peak = power.max()
    center = power[power.shape[0] // 2, power.shape[1] // 2]
    if peak <= 1e-6:
        return 0.0
    return float((peak - center) / peak)


def _evaluate_llff_scene(scene: Path, method: str, baseline_dir: Path) -> Dict[str, Any]:
    images = _load_pngs(scene / "images")
    if not images:
        raise FileNotFoundError(f"No PNG images found under {scene / 'images'}")

    if method == "GSplatStub":
        out_dir = ensure_dir(baseline_dir / scene.name / "gsplat")
        try:
            metrics = run_gsplat_stub(scene / "images", out_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] GSplat stub failed for {scene.name}: {exc}")
            metrics = {}
        proxy_path = out_dir / "proxy.png"
        if proxy_path.exists():
            recon = np.array(Image.open(proxy_path).convert("RGB"))
        else:
            recon = images[-1]
        feature_cos = feature_recon_cosine(images[0], recon)
        mv_consistency = _feature_similarity_pairs(images, step=1)
        track_persistence = math.nan
    else:
        feature_cos = feature_recon_cosine(images[0], images[-1])
        mv_consistency = _feature_similarity_pairs(images, step=1)
        track_persistence = math.nan

    return {
        "dataset": f"LLFF-{scene.name}",
        "method": method,
        "feature_cosine": feature_cos,
        "multiview_consistency": mv_consistency,
        "repeat_regularity": math.nan,
        "fourier_grid": math.nan,
        "depth_rmse": math.nan,
        "normal_agreement": math.nan,
        "track_persistence": track_persistence,
    }


def _evaluate_city_frame(image: np.ndarray, method: str, baseline_dir: Path | None = None, idx: int = 0, scene_name: str = "city") -> Dict[str, Any]:
    if method == "SlotAttnStub" and baseline_dir is not None:
        tmp_dir = ensure_dir(baseline_dir / f"city_{idx:02d}" / "slotattn")
        image_path = tmp_dir / "input.png"
        Image.fromarray(image).save(image_path)
        try:
            run_slotattn_stub(image_path, tmp_dir, slots=8)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] SlotAttn stub failed for Cityscapes frame {idx}: {exc}")
            return {
                "dataset": f"Cityscapes-{scene_name}-{idx:02d}",
                "method": method,
                "feature_cosine": math.nan,
                "multiview_consistency": math.nan,
                "repeat_regularity": math.nan,
                "fourier_grid": _fourier_grid_score(image),
                "depth_rmse": math.nan,
                "normal_agreement": math.nan,
                "track_persistence": math.nan,
            }
        mask_paths = sorted(tmp_dir.glob("slot_*.png"))
        centroids: List[Dict[str, Any]] = []
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            centroid = np.array([xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]], dtype=np.float32)
            centroids.append({"pos": centroid})
        regularity = repeat_regularity_error(centroids) if centroids else math.nan
    else:
        nodes = _grid_nodes(image)
        regularity = repeat_regularity_error(nodes)

    fourier = _fourier_grid_score(image)
    return {
        "dataset": f"Cityscapes-{scene_name}-{idx:02d}",
        "method": method,
        "feature_cosine": math.nan,
        "multiview_consistency": math.nan,
        "repeat_regularity": regularity,
        "fourier_grid": fourier,
        "depth_rmse": math.nan,
        "normal_agreement": math.nan,
        "track_persistence": math.nan,
    }


def _rgbd_persistence(seq_dir: Path, baseline_dir: Path) -> float:
    out_dir = ensure_dir(baseline_dir / seq_dir.name / "pointtracks")
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        try:
            run_pointtracks_stub(seq_dir / "rgb", out_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] PointTracks stub failed for {seq_dir.name}: {exc}")
            return math.nan
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics.get("persistence", math.nan))
    except Exception:  # pragma: no cover
        return math.nan


def _evaluate_rgbd_sequence(seq_dir: Path, method: str, baseline_dir: Path) -> Dict[str, Any]:
    depth_dir = seq_dir / "depth"
    rgb_dir = seq_dir / "rgb"
    depth_files = sorted(depth_dir.glob("*.png"))
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not depth_files or not rgb_files:
        raise FileNotFoundError(f"RGB-D sequence incomplete: {seq_dir}")

    depth = np.array(Image.open(depth_files[0])).astype(np.float32)
    depth_ref = np.array(Image.open(depth_files[min(1, len(depth_files) - 1)])).astype(np.float32)
    depth_error = depth_rmse(depth, depth_ref)

    gx, gy = np.gradient(depth)
    normals = np.stack([-gx, -gy, np.ones_like(depth)], axis=-1)
    normal_align = normal_agreement(normals, normals)

    persistence = _rgbd_persistence(seq_dir, baseline_dir)

    return {
        "dataset": f"RGBD-{seq_dir.name}",
        "method": method,
        "feature_cosine": math.nan,
        "multiview_consistency": math.nan,
        "repeat_regularity": math.nan,
        "fourier_grid": math.nan,
        "depth_rmse": depth_error,
        "normal_agreement": normal_align,
        "track_persistence": persistence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real datasets using placeholder metrics")
    parser.add_argument("--llff_dir", default="data/llff", help="LLFF dataset root")
    parser.add_argument("--city_dir", default="data/cityscapes", help="Cityscapes dataset root")
    parser.add_argument("--rgbd_dir", default="data/rgbd_mini", help="Optional RGB-D mini dataset")
    parser.add_argument("--tables_dir", default="tables", help="Directory to store results")
    parser.add_argument("--baselines_dir", default="tables/baseline_outputs_real", help="Directory for baseline artifacts")
    return parser.parse_args()


def _summarise(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        for key, value in row.items():
            if key in {"dataset", "method"}:
                continue
            grouped[method][key].append(value)
    for method, metric_dict in grouped.items():
        stats[method] = {}
        for metric, values in metric_dict.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0 or np.isnan(arr).all():
                stats[method][metric] = (math.nan, math.nan)
            else:
                stats[method][metric] = (float(np.nanmean(arr)), float(np.nanstd(arr)))
    return stats


def main() -> None:
    args = parse_args()
    tables_dir = ensure_dir(args.tables_dir)
    baselines_dir = ensure_dir(args.baselines_dir)

    rows: List[Dict[str, Any]] = []

    llff_root = Path(args.llff_dir)
    if llff_root.exists():
        scenes = sorted([p for p in llff_root.iterdir() if p.is_dir()])
        for scene in scenes:
            rows.append(_evaluate_llff_scene(scene, "GrammarInduction", baselines_dir))
            rows.append(_evaluate_llff_scene(scene, "GSplatStub", baselines_dir))
    else:
        print(f"[warn] LLFF directory not found: {llff_root}")

    city_root = Path(args.city_dir)
    if city_root.exists():
        frames = _load_pngs(city_root / "frames", limit=20)
        for idx, image in enumerate(frames):
            rows.append(_evaluate_city_frame(image, "GrammarInduction", idx=idx, scene_name="city"))
            rows.append(_evaluate_city_frame(image, "SlotAttnStub", baseline_dir=baselines_dir, idx=idx, scene_name="city"))
    else:
        print(f"[warn] Cityscapes directory not found: {city_root}")

    rgbd_root = Path(args.rgbd_dir)
    if rgbd_root.exists():
        sequences = sorted([p for p in rgbd_root.iterdir() if p.is_dir()])
        for seq in sequences:
            rows.append(_evaluate_rgbd_sequence(seq, "GrammarInduction", baselines_dir))
            rows.append(_evaluate_rgbd_sequence(seq, "PointTracksStub", baselines_dir))
    else:
        print(f"[info] RGB-D mini dataset skipped (not found): {rgbd_root}")

    results_path = tables_dir / "results_real.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stats = _summarise(rows)

    print("Real Dataset Evaluation Summary:\n")
    for method, metric_dict in stats.items():
        print(f"{method}:")
        for metric in TABLE_COLUMNS:
            if metric in {"dataset", "method"}:
                continue
            mean, std = metric_dict.get(metric, (math.nan, math.nan))
            if math.isnan(mean):
                print(f"  {metric:>20}: n/a")
            else:
                print(f"  {metric:>20}: {mean:.4f} ± {std:.4f}")
        print()
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
