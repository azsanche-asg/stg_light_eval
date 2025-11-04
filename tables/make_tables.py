"""Generate LaTeX tables from evaluation CSV summaries."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from stg_light_eval.utils.io import ensure_dir


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool

    @property
    def arrow(self) -> str:
        return "\\uparrow" if self.higher_is_better else "\\downarrow"


SYNTH_METRICS = [
    MetricSpec("rule_f1", "Rule F1", True),
    MetricSpec("tree_edit", "Tree Edit", False),
    MetricSpec("regularity", "Repeat Reg.", False),
    MetricSpec("mdl", "MDL", False),
    MetricSpec("persistence_error", "Persistence Err.", False),
    MetricSpec("motion_error", "Motion Err.", False),
]

REAL_METRICS = [
    MetricSpec("feature_cosine", "Feature Cosine", True),
    MetricSpec("multiview_consistency", "Multi-view Cons.", True),
    MetricSpec("repeat_regularity", "Repeat Reg.", False),
    MetricSpec("fourier_grid", "Fourier Grid", True),
    MetricSpec("depth_rmse", "Depth RMSE", False),
    MetricSpec("normal_agreement", "Normal Agree.", True),
    MetricSpec("track_persistence", "Track Persist.", True),
]

EFF_METRICS = [
    MetricSpec("artifact_mb", "Artifact (MB)", False),
    MetricSpec("peak_gpu_mb", "Peak GPU (MB)", False),
    MetricSpec("steps_k", "Steps (k)", False),
]


def _format_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean) or not math.isfinite(std):
        return "\\text{n/a}"
    return f"{mean:.2f} \\pm {std:.2f}"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _pivot_synth(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    table: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for row in rows:
        method = row["method"]
        metric = row["metric"]
        mean = float(row.get("mean", "nan"))
        std = float(row.get("std", "nan"))
        table.setdefault(method, {})[metric] = (mean, std)
    return table


def _aggregate_real(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    accum: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        method = row["method"]
        accum.setdefault(method, {})
        for spec in REAL_METRICS:
            value = float(row.get(spec.key, "nan"))
            accum[method].setdefault(spec.key, []).append(value)
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for method, metric_map in accum.items():
        summary[method] = {}
        for key, values in metric_map.items():
            arr = [v for v in values if math.isfinite(v)]
            if not arr:
                summary[method][key] = (math.nan, math.nan)
            else:
                mean = sum(arr) / len(arr)
                variance = sum((v - mean) ** 2 for v in arr) / len(arr)
                summary[method][key] = (mean, math.sqrt(variance))
    return summary


def _aggregate_eff(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for row in rows:
        method = row["method"]
        summary.setdefault(method, {})
        for spec in EFF_METRICS:
            value = float(row.get(spec.key, "nan"))
            summary[method][spec.key] = (value, 0.0)
    return summary


def _render_table(caption: str, label: str, specs: List[MetricSpec], data: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
    methods = sorted(data.keys())
    headers = " & ".join(["Method"] + [f"{spec.label} ({spec.arrow})" for spec in specs])
    lines = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{l{'c' * len(specs)}}}", "\\toprule", headers + " \\\", "\\midrule"]
    for method in methods:
        entries: List[str] = [method]
        for spec in specs:
            mean, std = data.get(method, {}).get(spec.key, (math.nan, math.nan))
            entries.append(_format_mean_std(mean, std))
        lines.append(" & ".join(entries) + " \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    tables_dir = ensure_dir("tables")
    synth_rows = _read_csv(tables_dir / "results_synth.csv")
    real_rows = _read_csv(tables_dir / "results_real.csv")
    eff_rows = _read_csv(tables_dir / "results_eff.csv")

    synth_table = _pivot_synth(synth_rows)
    real_table = _aggregate_real(real_rows)
    eff_table = _aggregate_eff(eff_rows)

    tab1 = _render_table(
        "Synthetic evaluation (higher is better for Rule F1; arrows denote metric direction)",
        "tab:synthetic",
        SYNTH_METRICS,
        synth_table,
    )
    (tables_dir / "Tab1_synthetic.tex").write_text(tab1, encoding="utf-8")

    tab2 = _render_table(
        "Real-data evaluation (↑ higher is better, ↓ lower is better)",
        "tab:real",
        REAL_METRICS,
        real_table,
    )
    (tables_dir / "Tab2_real.tex").write_text(tab2, encoding="utf-8")

    tab3 = _render_table(
        "Efficiency comparison (smaller is better)",
        "tab:efficiency",
        EFF_METRICS,
        eff_table,
    )
    (tables_dir / "Tab3_efficiency.tex").write_text(tab3, encoding="utf-8")

    print("Generated Tab1_synthetic.tex, Tab2_real.tex, Tab3_efficiency.tex")


if __name__ == "__main__":
    main()

