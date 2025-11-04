#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

echo "[smoke] Generating synthetic scenes"
python "$ROOT/stg_light_eval/synthetic/generate_scenes.py" --out "$ROOT/data/synth" --n 6 --frames 10

echo "[smoke] Downloading LLFF subset"
python -m stg_light_eval.real_data_prep.download_llff --out "$ROOT/data/llff" --scenes 3 || echo "[warn] LLFF download failed"

echo "[smoke] Downloading Cityscapes frames"
python -m stg_light_eval.real_data_prep.download_cityscapes_frames --out "$ROOT/data/cityscapes" || echo "[warn] Cityscapes download failed"

echo "[smoke] Evaluating synthetic scenes"
if ! python "$ROOT/stg_light_eval/eval/eval_synthetic.py" --data_dir "$ROOT/data/synth" --tables_dir "$ROOT/tables" --baselines_dir "$ROOT/tables/baseline_outputs" --quick; then
  python "$ROOT/stg_light_eval/eval/eval_synthetic.py" --data_dir "$ROOT/data/synth" --tables_dir "$ROOT/tables" --baselines_dir "$ROOT/tables/baseline_outputs"
fi

echo "[smoke] Evaluating real datasets"
if ! python "$ROOT/stg_light_eval/eval/eval_real.py" --llff_dir "$ROOT/data/llff" --city_dir "$ROOT/data/cityscapes" --rgbd_dir "$ROOT/data/rgbd_mini" --tables_dir "$ROOT/tables" --baselines_dir "$ROOT/tables/baseline_outputs_real" --quick; then
  python "$ROOT/stg_light_eval/eval/eval_real.py" --llff_dir "$ROOT/data/llff" --city_dir "$ROOT/data/cityscapes" --rgbd_dir "$ROOT/data/rgbd_mini" --tables_dir "$ROOT/tables" --baselines_dir "$ROOT/tables/baseline_outputs_real"
fi

echo "[smoke] Evaluating efficiency"
if ! python "$ROOT/stg_light_eval/eval/eval_efficiency.py" --artifact_root "$ROOT/tables/baseline_outputs" --tables_dir "$ROOT/tables" --quick; then
  python "$ROOT/stg_light_eval/eval/eval_efficiency.py" --artifact_root "$ROOT/tables/baseline_outputs" --tables_dir "$ROOT/tables"
fi

echo "[smoke] Building tables"
python "$ROOT/stg_light_eval/tables/make_tables.py"

echo "[smoke] Building figures"
python "$ROOT/stg_light_eval/tables/make_figs.py"

echo "[smoke] Generated tables:"
find "$ROOT/tables" -maxdepth 1 -name 'Tab*.tex' -print

echo "[smoke] Generated figures:"
find "$ROOT/tables/figs" -maxdepth 1 -name 'Fig*.png' -print
