"""Point tracks baseline stub using OpenCV corner tracking and k-means clustering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from sklearn.cluster import KMeans

from stg_light_eval.utils.io import ensure_dir


def _load_frames(frames_dir: Path, limit: int | None = None) -> List[np.ndarray]:
    frames = []
    for frame_path in sorted(frames_dir.glob("*.png"))[:limit]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)
    return frames


def _track_points(frames: List[np.ndarray], max_corners: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < 2:
        raise ValueError("Need at least two frames for tracking")
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    if points is None:
        raise ValueError("No corners detected")
    tracks = [points.squeeze(1)]
    survival_mask = np.ones(points.shape[0], dtype=bool)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
        valid = status.squeeze(1).astype(bool)
        survival_mask &= valid
        points = next_pts
        tracks.append(points.squeeze(1))
        prev_gray = gray

    tracks_array = np.stack(tracks, axis=1)  # [num_points, num_frames, 2]
    return tracks_array, survival_mask


def _cluster_tracks(tracks: np.ndarray, clusters: int = 5) -> np.ndarray:
    if tracks.shape[0] < clusters:
        clusters = tracks.shape[0]
    if clusters <= 0:
        return np.zeros(tracks.shape[0], dtype=np.int32)

    features = tracks.reshape(tracks.shape[0], -1)
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10)
    assignments = kmeans.fit_predict(features)
    return assignments


def _compute_persistence(survival_mask: np.ndarray) -> float:
    if survival_mask.size == 0:
        return 0.0
    return float(np.mean(survival_mask))


def run_stub(frames_dir: Path, out_dir: Path, clusters: int = 5) -> None:
    ensure_dir(out_dir)
    frames = _load_frames(frames_dir)
    if len(frames) < 2:
        raise FileNotFoundError("Need at least two frames in frames_dir")

    tracks, survival = _track_points(frames)
    assignments = _cluster_tracks(tracks, clusters=clusters)
    persistence = _compute_persistence(survival)

    csv_path = out_dir / "point_tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["track_id", "cluster", "survived"])
        for idx, (cluster_id, survived) in enumerate(zip(assignments, survival)):
            writer.writerow([idx, cluster_id, int(survived)])

    metrics_path = out_dir / "metrics.json"
    metrics = {"num_tracks": int(tracks.shape[0]), "clusters": int(clusters), "persistence": persistence}
    metrics_path.write_text(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point tracking baseline stub")
    parser.add_argument("--frames_dir", required=True, help="Directory with sequential PNG frames")
    parser.add_argument("--out_dir", required=True, help="Directory to store track outputs")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for track grouping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stub(Path(args.frames_dir).expanduser().resolve(), Path(args.out_dir).expanduser().resolve(), clusters=args.clusters)


if __name__ == "__main__":
    main()
