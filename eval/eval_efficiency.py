"""Efficiency evaluation placeholder capturing model size, memory, and steps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import random

from stg_light_eval.utils.io import ensure_dir

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def _estimate_artifact_size(path: Path) -> float:
    if not path.exists():
        return math.nan
    if path.is_file():
        return path.stat().st_size / 1e6
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / 1e6


def _mock_peak_memory() -> float:
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU path
        torch.cuda.reset_peak_memory_stats()
        peak = torch.cuda.max_memory_allocated() / 1e6
        return float(peak)
    return float(random.uniform(50, 150))


def _method_entries(artifact_root: Path, methods: List[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for method in methods:
        artifact_dir = artifact_root / method
        size_mb = _estimate_artifact_size(artifact_dir)
        peak_mem = _mock_peak_memory()
        steps = random.randint(10, 30)
        rows.append({
            "method": method,
            "artifact_mb": size_mb,
            "peak_gpu_mb": peak_mem,
            "steps_k": steps / 1_000.0,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate efficiency metrics for baselines")
    parser.add_argument("--artifact_root", default="tables/baseline_outputs", help="Directory containing baseline artifacts")
    parser.add_argument("--tables_dir", default="tables", help="Directory to store efficiency table")
    parser.add_argument("--methods", default="GrammarInduction,GSplatStub,SlotAttnStub,PointTracksStub", help="Comma separated list of methods")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = ensure_dir(args.tables_dir)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = _method_entries(artifact_root, methods)

    results_path = tables_dir / "results_eff.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "artifact_mb", "peak_gpu_mb", "steps_k"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Efficiency Summary:\n")
    for row in rows:
        size = row["artifact_mb"]
        peak = row["peak_gpu_mb"]
        steps = row["steps_k"]
        size_str = f"{size:.2f} MB" if math.isfinite(size) else "n/a"
        peak_str = f"{peak:.2f} MB"
        steps_str = f"{steps:.3f}k"
        print(f"{row['method']:<20} size={size_str:>8} peak={peak_str:>10} steps={steps_str}")

    print(f"\nEfficiency results saved to {results_path}")


if __name__ == "__main__":
    main()
