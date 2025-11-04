"""Download a lightweight subset of LLFF example scenes.

The script pulls a handful of RGB frames for each LLFF scene from the
`bmild/nerf` GitHub repository (nerf_llff_data/images_4). Files are fetched
via raw HTTPS links to keep the total footprint under ~200 MB.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from stg_light_eval.utils.io import ensure_dir, save_json

DEFAULT_SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns",
    "leaves",
    "orchids",
    "room",
    "trex",
)

RAW_BASE = "https://raw.githubusercontent.com/bmild/nerf/master/nerf_llff_data/{scene}/images_4/{filename}"
FRAME_COUNT_PER_SCENE = 24  # 24 * ~8MB ~= 192MB across 8 scenes


def iter_scene_filenames(count: int = FRAME_COUNT_PER_SCENE) -> Iterable[str]:
    for idx in range(count):
        yield f"{idx:03d}.png"


def download_file(url: str, destination: Path, *, retries: int = 4, timeout: int = 20) -> Path:
    ensure_dir(destination.parent)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stg-light-eval/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status >= 400:
                    raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
                with tmp_path.open("wb") as fh:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
            tmp_path.replace(destination)
            return destination
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == retries:
                raise
            backoff = min(2 ** attempt, 10)
            print(f"Retry {attempt}/{retries} for {url}: {exc}. Backing off {backoff}s", file=sys.stderr)
            time.sleep(backoff)
    return destination


def download_scene(
    scene: str,
    out_root: Path,
    *,
    frame_count: int = FRAME_COUNT_PER_SCENE,
    retries: int = 4,
) -> None:
    scene_dir = ensure_dir(out_root / scene / "images")
    print(f"Scene {scene}: downloading {frame_count} frames -> {scene_dir}")
    for filename in iter_scene_filenames(frame_count):
        url = RAW_BASE.format(scene=scene, filename=filename)
        dest = scene_dir / filename
        if dest.exists():
            continue
        try:
            download_file(url, dest, retries=retries)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to download {url}: {exc}", file=sys.stderr)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download lightweight LLFF RGB samples")
    parser.add_argument("--out", default="data/llff", help="Directory to store the LLFF scenes")
    parser.add_argument(
        "--scenes",
        default=",".join(DEFAULT_SCENES),
        help="Comma-separated subset of scenes to download",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=FRAME_COUNT_PER_SCENE,
        help="Number of frames per scene to fetch (default: 24).",
    )
    parser.add_argument("--retries", type=int, default=4, help="Retry attempts per file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    unknown = sorted(set(selected) - set(DEFAULT_SCENES))
    if unknown:
        raise ValueError(f"Unknown scenes requested: {unknown}. Available: {', '.join(DEFAULT_SCENES)}")

    out_root = ensure_dir(args.out)
    downloaded: list[str] = []

    for scene in selected:
        try:
            download_scene(scene, out_root, frame_count=args.frames, retries=args.retries)
            downloaded.append(scene)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Scene {scene} failed: {exc}", file=sys.stderr)
            raise

    if downloaded:
        index_path = out_root / "index.json"
        save_json({"scenes": downloaded, "frames": args.frames}, index_path)
        print(f"Wrote index with {len(downloaded)} scenes -> {index_path}")
    else:
        print("No scenes downloaded.")


if __name__ == "__main__":
    main()
