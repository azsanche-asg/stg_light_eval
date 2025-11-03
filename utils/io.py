"""Lightweight IO helpers for the evaluation toolkit."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import simplejson as sjson

try:  # Optional dependency handling for reproducibility helpers.
    import numpy as np
except ImportError:  # pragma: no cover - numpy is in requirements but play safe.
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover - torch availability varies.
    torch = None  # type: ignore


def _to_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path, *, exist_ok: bool = True) -> Path:
    """Create the directory if needed and return it as a Path."""
    directory = _to_path(path)
    directory.mkdir(parents=True, exist_ok=exist_ok)
    return directory


def load_json(path: str | Path, *, use_simplejson: bool = True) -> Any:
    """Load JSON content using simplejson when available."""
    json_path = _to_path(path)
    loader = sjson if use_simplejson else json
    with json_path.open("r", encoding="utf-8") as fh:
        return loader.load(fh)


def save_json(data: Any, path: str | Path, *, indent: int = 2, sort_keys: bool = True) -> Path:
    """Serialize data to JSON with UTF-8 encoding."""
    json_path = _to_path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as fh:
        sjson.dump(data, fh, indent=indent, sort_keys=sort_keys)
        fh.write("\n")
    return json_path


def save_csv(
    rows: Iterable[Mapping[str, Any]] | Sequence[Sequence[Any]],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
    delimiter: str = ",",
) -> Path:
    """Write rows to CSV, supporting dict or sequence records."""
    import csv

    csv_path = _to_path(path)
    ensure_dir(csv_path.parent)

    row_list = list(rows)
    if not row_list:
        if fieldnames is None:
            raise ValueError("fieldnames must be provided when rows are empty")
        header = list(fieldnames)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(header)
        return csv_path

    first_row = row_list[0]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if isinstance(first_row, Mapping):
            dict_rows = row_list  # type: ignore[assignment]
            header = fieldnames or sorted({key for row in dict_rows for key in row.keys()})
            writer = csv.DictWriter(fh, fieldnames=header, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(dict_rows)
        else:
            if fieldnames is None:
                raise ValueError("fieldnames must be provided for sequence rows")
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(fieldnames)
            writer.writerows(row_list)
    return csv_path


def set_seed(seed: int) -> int:
    """Seed common RNGs to improve reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        backend = getattr(torch, "backends", None)
        cudnn = getattr(backend, "cudnn", None) if backend else None
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
    return seed

