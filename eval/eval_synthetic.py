"""Evaluate synthetic scenes using grammar induction and baseline stubs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from stg_light_eval.eval.grammar_induction import induce_grammar
from stg_light_eval.metrics import (
    mdl_score,
    motion_error,
    persistence_score,
    repeat_regularity_error,
    rule_f1,
    tree_edit_distance,
)
from stg_light_eval.utils.io import ensure_dir, load_json

from stg_light_eval.baselines.gsplat.run_gsplat_stub import run_stub as run_gsplat_stub
from stg_light_eval.baselines.slotattn.run_slotattn_stub import run_stub as run_slotattn_stub
from stg_light_eval.baselines.pointtracks.run_pointtracks_stub import run_stub as run_pointtracks_stub


METRIC_COLUMNS = [
    "rule_precision",
    "rule_recall",
    "rule_f1",
    "tree_edit",
    "regularity",
    "mdl",
    "persistence_error",
    "motion_error",
    "model_size_mb",
    "slot_quality_mean",
    "slot_quality_std",
    "track_persistence",
]


def _load_instance_masks(scene_path: Path) -> np.ndarray | None:
    mask_path = scene_path / "instance_masks.npy"
    if mask_path.exists():
        return np.load(mask_path)
    return None


def _compute_sequences(masks: np.ndarray | None) -> Dict[int, List[int | None]]:
    if masks is None:
        return {}
    total_frames = masks.shape[0]
    ids = sorted(int(i) for i in np.unique(masks) if i > 0)
    sequences: Dict[int, List[int | None]] = {i: [] for i in ids}
    for frame_idx in range(total_frames):
        frame = masks[frame_idx]
        present = set(int(i) for i in np.unique(frame) if i > 0)
        for instance_id in ids:
            sequences[instance_id].append(instance_id if instance_id in present else None)
    return sequences


def _extract_motion(tree: Any) -> Dict[str, Dict[str, Iterable[float]]]:
    motions: Dict[str, Dict[str, Iterable[float]]] = {}

    def _traverse(node: Any) -> None:
        if isinstance(node, dict):
            if "motion" in node:
                motions[str(node.get("symbol", len(motions)))] = node["motion"]
            for child in node.get("children", []) or []:
                _traverse(child)
        elif isinstance(node, list):
            for child in node:
                _traverse(child)

    _traverse(tree)
    return motions


def _flatten_motion(motion: Dict[str, Dict[str, Iterable[float]]]) -> np.ndarray:
    if not motion:
        return np.zeros(0, dtype=np.float32)
    entries: List[float] = []
    for key in sorted(motion.keys()):
        params = motion[key]
        start = params.get("start", [0.0, 0.0])
        velocity = params.get("velocity", [0.0, 0.0])
        entries.extend(float(x) for x in start)
        entries.extend(float(x) for x in velocity)
    return np.asarray(entries, dtype=np.float32)


def _compute_mdl(pred_rules: List[dict[str, Any]], tree_cost: float) -> float:
    structure_bits = float(len(pred_rules) * 16)
    residual_bits = max(tree_cost * 100.0, 0.0)
    return mdl_score(structure_bits, residual_bits, cap_bits=1e3)


def _prepare_variant(pred: Dict[str, Any], *, drop_repeat: bool = False, drop_split: bool = False, drop_temporal: bool = False) -> Dict[str, Any]:
    variant = deepcopy(pred)
    if drop_repeat:
        variant["pred_rules"] = [rule for rule in variant["pred_rules"] if rule.get("type") != "Repeat"]
        filtered_nodes = []
        for node in variant.get("pred_nodes", []):
            if isinstance(node, dict):
                axis = node.get("axis")
                if axis in {"x", "y"}:
                    continue
            filtered_nodes.append(node)
        variant["pred_nodes"] = filtered_nodes
    if drop_split:
        variant["pred_rules"] = [rule for rule in variant["pred_rules"] if rule.get("type") != "Split"]
    if drop_temporal:
        variant["pred_motion"] = {}
    return variant


def _evaluate_prediction(
    scene_name: str,
    pred: Dict[str, Any],
    grammar: Dict[str, Any],
    tree: Dict[str, Any],
    masks: np.ndarray | None,
) -> Dict[str, float]:
    gt_rules = grammar.get("productions", [])
    pred_rules = pred.get("pred_rules", [])

    precision, recall, f1 = rule_f1(gt_rules, pred_rules)
    ted = tree_edit_distance(tree, pred.get("pred_tree", {}))
    regularity = repeat_regularity_error(pred.get("pred_nodes", []))
    mdl = _compute_mdl(pred_rules, ted)

    sequences = _compute_sequences(masks)
    gt_persistence = persistence_score(sequences.values()) if sequences else 0.0

    pred_motion = pred.get("pred_motion", {})
    if masks is not None and pred_motion:
        pred_ids = {int(i) for i in pred_motion.keys()}
        pred_sequences = {i: seq for i, seq in sequences.items() if i in pred_ids}
        pred_persistence = persistence_score(pred_sequences.values()) if pred_sequences else 0.0
    else:
        pred_persistence = gt_persistence
    persistence_err = abs(pred_persistence - gt_persistence)

    gt_motion = _extract_motion(tree)
    gt_motion_vec = _flatten_motion(gt_motion)
    pred_motion_vec = _flatten_motion(pred_motion)
    motion_err = motion_error(gt_motion_vec, pred_motion_vec) if gt_motion_vec.size or pred_motion_vec.size else 0.0

    return {
        "rule_precision": precision,
        "rule_recall": recall,
        "rule_f1": f1,
        "tree_edit": ted,
        "regularity": regularity,
        "mdl": mdl,
        "persistence_error": persistence_err,
        "motion_error": motion_err,
        "model_size_mb": math.nan,
        "slot_quality_mean": math.nan,
        "slot_quality_std": math.nan,
        "track_persistence": math.nan,
    }


def _evaluate_gsplat(scene_path: Path, out_root: Path) -> Dict[str, float]:
    out_dir = ensure_dir(out_root / scene_path.name / "gsplat")
    try:
        metrics = run_gsplat_stub(scene_path / "frames", out_dir)
    except Exception as exc:  # pragma: no cover - baseline failure fallback
        print(f"[warn] GSplat stub failed for {scene_path.name}: {exc}")
        metrics = {}
    return {
        "rule_precision": math.nan,
        "rule_recall": math.nan,
        "rule_f1": math.nan,
        "tree_edit": math.nan,
        "regularity": math.nan,
        "mdl": math.nan,
        "persistence_error": math.nan,
        "motion_error": math.nan,
        "model_size_mb": float(metrics.get("model_size_mb", math.nan)),
        "slot_quality_mean": math.nan,
        "slot_quality_std": math.nan,
        "track_persistence": math.nan,
    }


def _evaluate_slotattn(scene_path: Path, out_root: Path, slots: int = 8) -> Dict[str, float]:
    out_dir = ensure_dir(out_root / scene_path.name / "slotattn")
    frame_paths = sorted((scene_path / "frames").glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {scene_path / 'frames'}")
    image_path = frame_paths[0]
    try:
        run_slotattn_stub(image_path, out_dir, slots)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] SlotAttn stub failed for {scene_path.name}: {exc}")
        return {
            "rule_precision": math.nan,
            "rule_recall": math.nan,
            "rule_f1": math.nan,
            "tree_edit": math.nan,
            "regularity": math.nan,
            "mdl": math.nan,
            "persistence_error": math.nan,
            "motion_error": math.nan,
            "model_size_mb": math.nan,
            "slot_quality_mean": math.nan,
            "slot_quality_std": math.nan,
            "track_persistence": math.nan,
        }
    csv_path = out_dir / "slot_metrics.csv"
    qualities: List[float] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    qualities.append(float(row.get("quality", "nan")))
                except ValueError:
                    continue
    mean_quality = float(np.mean(qualities)) if qualities else math.nan
    std_quality = float(np.std(qualities)) if qualities else math.nan
    return {
        "rule_precision": math.nan,
        "rule_recall": math.nan,
        "rule_f1": math.nan,
        "tree_edit": math.nan,
        "regularity": math.nan,
        "mdl": math.nan,
        "persistence_error": math.nan,
        "motion_error": math.nan,
        "model_size_mb": math.nan,
        "slot_quality_mean": mean_quality,
        "slot_quality_std": std_quality,
        "track_persistence": math.nan,
    }


def _evaluate_pointtracks(scene_path: Path, out_root: Path, clusters: int = 5) -> Dict[str, float]:
    out_dir = ensure_dir(out_root / scene_path.name / "pointtracks")
    try:
        run_pointtracks_stub(scene_path / "frames", out_dir, clusters=clusters)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] PointTracks stub failed for {scene_path.name}: {exc}")
        return {
            "rule_precision": math.nan,
            "rule_recall": math.nan,
            "rule_f1": math.nan,
            "tree_edit": math.nan,
            "regularity": math.nan,
            "mdl": math.nan,
            "persistence_error": math.nan,
            "motion_error": math.nan,
            "model_size_mb": math.nan,
            "slot_quality_mean": math.nan,
            "slot_quality_std": math.nan,
            "track_persistence": math.nan,
        }
    metrics_path = out_dir / "metrics.json"
    persistence = math.nan
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        persistence = float(metrics.get("persistence", math.nan))
    return {
        "rule_precision": math.nan,
        "rule_recall": math.nan,
        "rule_f1": math.nan,
        "tree_edit": math.nan,
        "regularity": math.nan,
        "mdl": math.nan,
        "persistence_error": math.nan,
        "motion_error": math.nan,
        "model_size_mb": math.nan,
        "slot_quality_mean": math.nan,
        "slot_quality_std": math.nan,
        "track_persistence": persistence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic scenes with grammar induction and baselines")
    parser.add_argument("--data_dir", default="data/synth", help="Directory containing synthetic scenes")
    parser.add_argument("--tables_dir", default="tables", help="Directory to store result tables")
    parser.add_argument("--baselines_dir", default="tables/baseline_outputs", help="Directory for baseline artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    tables_dir = ensure_dir(args.tables_dir)
    baselines_dir = ensure_dir(args.baselines_dir)

    per_scene_rows: List[Dict[str, Any]] = []
    method_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not scenes:
        raise FileNotFoundError(f"No scenes found in {data_dir}")

    for scene_path in scenes:
        grammar = load_json(scene_path / "gt_grammar.json")
        tree = load_json(scene_path / "gt_tree.json")
        masks = _load_instance_masks(scene_path)
        dynamic = grammar.get("scene_type", "static") == "dynamic"

        base_pred = induce_grammar(scene_path, dynamic=dynamic)
        variants = {
            "GrammarInduction": base_pred,
            "GrammarInduction-NoRepeat": _prepare_variant(base_pred, drop_repeat=True),
            "GrammarInduction-NoSplit": _prepare_variant(base_pred, drop_split=True),
            "GrammarInduction-NoTemporal": _prepare_variant(base_pred, drop_temporal=True),
        }

        for method, pred in variants.items():
            metrics_dict = _evaluate_prediction(scene_path.name, pred, grammar, tree, masks)
            metrics_dict.update({"scene": scene_path.name, "method": method})
            per_scene_rows.append(metrics_dict)
            for metric_name in METRIC_COLUMNS:
                method_metrics[method][metric_name].append(metrics_dict[metric_name])

        # Baseline evaluations
        baseline_results = {
            "GSplatStub": _evaluate_gsplat(scene_path, baselines_dir),
            "SlotAttnStub": _evaluate_slotattn(scene_path, baselines_dir),
            "PointTracksStub": _evaluate_pointtracks(scene_path, baselines_dir),
        }
        for method, metrics_dict in baseline_results.items():
            row = {"scene": scene_path.name, "method": method}
            row.update(metrics_dict)
            per_scene_rows.append(row)
            for metric_name in METRIC_COLUMNS:
                method_metrics[method][metric_name].append(row[metric_name])

    per_scene_path = tables_dir / "per_scene_synth.csv"
    with per_scene_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scene", "method"] + METRIC_COLUMNS)
        writer.writeheader()
        for row in per_scene_rows:
            writer.writerow(row)

    summary_rows: List[Dict[str, Any]] = []
    for method, metric_dict in method_metrics.items():
        for metric_name, values in metric_dict.items():
            arr = np.asarray(values, dtype=float)
            mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
            std = float(np.nanstd(arr)) if np.isfinite(arr).any() else math.nan
            summary_rows.append({
                "method": method,
                "metric": metric_name,
                "mean": mean,
                "std": std,
            })

    results_path = tables_dir / "results_synth.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "metric", "mean", "std"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Synthetic Evaluation Summary:\n")
    methods = sorted(method_metrics.keys())
    display_metrics = ["rule_f1", "tree_edit", "regularity", "mdl", "persistence_error", "motion_error"]
    for method in methods:
        print(f"{method}:")
        for metric in display_metrics:
            arr = np.asarray(method_metrics[method][metric], dtype=float)
            if arr.size == 0 or (np.isnan(arr).all()):
                value = "n/a"
            else:
                value = f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"
            print(f"  {metric:>17}: {value}")
        print()
    print(f"Per-scene metrics saved to {per_scene_path}")
    print(f"Aggregate results saved to {results_path}")


if __name__ == "__main__":
    main()
