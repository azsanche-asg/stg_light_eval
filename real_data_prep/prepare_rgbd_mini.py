"""Prepare a tiny RGB-D evaluation subset from user-provided archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

from stg_light_eval.utils.io import ensure_dir

SUPPORTED_EXTS = {".zip", ".tar", ".tar.gz", ".tgz"}
MAX_SEQUENCES = 10


def iter_archives(src: Path) -> list[Path]:
    archives = []
    for path in sorted(src.glob("*")):
        if path.is_file() and (
            path.suffix in SUPPORTED_EXTS or path.name.endswith((".tar.gz", ".tgz"))
        ):
            archives.append(path)
    return archives


def extract_archive(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif archive.suffix == ".tar" or archive.name.endswith((".tar.gz", ".tgz")):
        mode = "r:gz" if archive.name.endswith(('.tar.gz', '.tgz')) else "r"
        with tarfile.open(archive, mode) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def prepare_sequences(extracted_root: Path, target_root: Path, max_sequences: int = MAX_SEQUENCES) -> list[str]:
    sequences = []
    for folder in sorted(extracted_root.glob("**/rgb")):
        seq_dir = folder.parent
        depth_dir = seq_dir / "depth" if (seq_dir / "depth").exists() else seq_dir / "depth_registered"
        if not depth_dir.exists():
            continue
        seq_name = seq_dir.name
        out_dir = ensure_dir(target_root / seq_name)
        rgb_target = ensure_dir(out_dir / "rgb")
        depth_target = ensure_dir(out_dir / "depth")

        # Copy first 30 frames (rgb + depth)
        rgb_frames = sorted(folder.glob("*.png"))[:30]
        depth_frames = sorted(depth_dir.glob("*.png"))[:30]
        for src_frames, dst_root in ((rgb_frames, rgb_target), (depth_frames, depth_target)):
            for frame in src_frames:
                dst = dst_root / frame.name
                if dst.exists():
                    continue
                dst.write_bytes(frame.read_bytes())

        intrinsics_candidates = list(seq_dir.glob("*intrinsics*.txt")) + list(seq_dir.glob("camera.txt"))
        if intrinsics_candidates:
            (out_dir / "intrinsics.txt").write_bytes(intrinsics_candidates[0].read_bytes())
        else:
            (out_dir / "intrinsics.txt").write_text("fx fy cx cy\n500 500 320 240\n")
        sequences.append(seq_name)
        if len(sequences) >= max_sequences:
            break
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a tiny RGB-D dataset from provided archives")
    parser.add_argument("--src", default="data/rgbd_src", help="Directory containing RGB-D archives (zip/tar)")
    parser.add_argument("--out", default="data/rgbd_mini", help="Destination directory for prepared sequences")
    parser.add_argument("--tmp", default="data/rgbd_tmp", help="Temporary extraction directory")
    parser.add_argument("--max", type=int, default=MAX_SEQUENCES, help="Maximum sequences to prepare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = ensure_dir(args.src)
    out_root = ensure_dir(args.out)
    tmp_root = ensure_dir(args.tmp)

    archives = iter_archives(src_dir)
    if not archives:
        print(f"No RGB-D archives found in {src_dir}. Place 7-Scenes or TUM zips there and rerun.")
        return

    prepared = []
    for archive in archives:
        extract_dir = tmp_root / archive.stem
        extract_archive(archive, extract_dir)
        sequences = prepare_sequences(extract_dir, out_root, max_sequences=args.max - len(prepared))
        prepared.extend(sequences)
        if len(prepared) >= args.max:
            break

    if prepared:
        print(f"Prepared {len(prepared)} sequences in {out_root}:")
        for seq in prepared:
            print(f" - {seq}")
    else:
        print("No sequences prepared. Check archive structure (expected rgb/ and depth/ folders).")


if __name__ == "__main__":
    main()
