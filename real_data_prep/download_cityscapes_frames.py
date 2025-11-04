"""Download a tiny Cityscapes-like validation sample set.

The script attempts to fetch ~20 RGB frames from publicly hosted mirrors that
ship small subsets suitable for research demos. When the downloads fail (e.g.,
no network), it falls back to procedurally generating facade-like images using
simple gradients so that downstream evaluation scripts can still execute.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from stg_light_eval.utils.io import ensure_dir, save_json

DEFAULT_URLS = [
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_001.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_002.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_003.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_004.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_005.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_006.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_007.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_008.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_009.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_010.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_011.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_012.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_013.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_014.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_015.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_016.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_017.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_018.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_019.png",
    "https://raw.githubusercontent.com/autonomousvision/cityscapes-vps/main/assets/samples/viz_020.png",
]

MAX_RETRIES = 4
TIMEOUT = 20


def download_file(url: str, dest: Path, *, retries: int = MAX_RETRIES) -> bool:
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".download")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "stg-light-eval/0.1"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status >= 400:
                    raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
                with tmp.open("wb") as fh:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
            tmp.replace(dest)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"Failed to download {url}: {exc}", file=sys.stderr)
                return False
            backoff = min(2 ** attempt, 12)
            print(f"Retry {attempt}/{retries} for {url}: {exc}. Sleeping {backoff}s", file=sys.stderr)
            time.sleep(backoff)
    return False


def generate_placeholder(idx: int, dest: Path, width: int = 1024, height: int = 512) -> None:
    rng = np.random.default_rng(idx)
    gradient_top = np.array([rng.integers(100, 180), rng.integers(120, 200), rng.integers(160, 220)], dtype=np.float32)
    gradient_bottom = gradient_top * rng.uniform(0.6, 0.9)
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = gradient_top + (gradient_bottom - gradient_top) * rows
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    gradient = np.repeat(gradient[:, None, :], width, axis=1)
    image = Image.fromarray(gradient, mode="RGB")
    draw = ImageDraw.Draw(image)

    n_rects = rng.integers(6, 12)
    palette = [tuple(int(x) for x in rng.integers(50, 200, size=3)) for _ in range(n_rects)]

    for ridx in range(n_rects):
        color = palette[ridx]
        x1 = int(rng.uniform(0, width * 0.9))
        y1 = int(rng.uniform(height * 0.4, height * 0.9))
        rect_w = int(rng.uniform(width * 0.05, width * 0.2))
        rect_h = int(rng.uniform(height * 0.1, height * 0.4))
        x2 = min(width, x1 + rect_w)
        y2 = min(height, y1 + rect_h)
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=(0, 0, 0))

    ensure_dir(dest.parent)
    image.save(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a minimal Cityscapes-like validation set")
    parser.add_argument("--out", default="data/cityscapes", help="Output directory")
    parser.add_argument("--urls", default=",".join(DEFAULT_URLS), help="Comma-separated list of sample image URLs")
    parser.add_argument("--generate", action="store_true", help="Force placeholder generation without downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out)
    frames_dir = ensure_dir(out_root / "frames")

    url_list = [url.strip() for url in args.urls.split(",") if url.strip()]
    if len(url_list) < 1:
        raise ValueError("No URLs provided for download")

    downloaded = []
    if not args.generate:
        for idx, url in enumerate(url_list, 1):
            fname = f"city_{idx:03d}.png"
            dest = frames_dir / fname
            if download_file(url, dest):
                downloaded.append(fname)

    total_required = 20
    if len(downloaded) < total_required:
        print(
            f"Only {len(downloaded)} downloads succeeded; generating {total_required - len(downloaded)} placeholders",
            file=sys.stderr,
        )
        next_idx = len(downloaded)
        for idx in range(total_required - len(downloaded)):
            fname = f"city_{next_idx + idx + 1:03d}.png"
            dest = frames_dir / fname
            generate_placeholder(next_idx + idx, dest)
            downloaded.append(fname)
    else:
        downloaded = downloaded[:total_required]

    index_path = out_root / "index.json"
    save_json({"frames": downloaded, "root": str(frames_dir)}, index_path)
    print(f"Prepared Cityscapes sample set with {len(downloaded)} frames -> {frames_dir}")
    print(f"Index written to {index_path}")


if __name__ == "__main__":
    main()

