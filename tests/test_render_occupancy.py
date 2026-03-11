"""Render occupied voxels from a .npy occupancy grid into UE4 via AirSim."""

from __future__ import annotations
import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import json
from pathlib import Path
import numpy as np

try:
    import airsim
except Exception:  # pragma: no cover
    airsim = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render occupied voxels from occupancy npy into UE4.")
    parser.add_argument("--npy", type=Path, default=Path("artifacts/level_occupancy.npy"), help="Occupancy .npy path.")
    parser.add_argument("--voxel-size", type=float, default=6.0, help="Meters per voxel.")
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=None,
        metavar=("OX", "OY", "OZ"),
        help="Grid origin; if omitted will try to read from <npy>.json origin.",
    )
    parser.add_argument("--color", type=float, nargs=4, default=[1.0, 0.8, 0.1, 0.8], help="RGBA color for occupied voxels.")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for plotting points.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration for markers (0 = persistent).")
    return parser.parse_args()


def load_origin(meta_path: Path, override: np.ndarray | None) -> np.ndarray:
    if override is not None:
        return override
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            return np.array(data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        except Exception:
            pass
    return np.zeros(3, dtype=np.float64)


def main() -> None:
    if airsim is None:
        raise SystemExit("airsim package not installed.")
    args = parse_args()
    occ = np.load(args.npy)
    origin = load_origin(args.npy.with_suffix(".json"), np.array(args.origin, dtype=np.float64) if args.origin else None)
    voxel = float(args.voxel_size)

    occupied = np.argwhere(occ == 1)
    if occupied.size == 0:
        print("No occupied voxels found.")
        return
    if args.downsample > 1:
        occupied = occupied[:: args.downsample]
    points = (occupied.astype(np.float64) + 0.5) * voxel + origin
    ned = points.copy()
    ned[:, 2] *= -1.0

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.simFlushPersistentMarkers()
    client.simPlotPoints(
        [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in ned],
        color_rgba=args.color,
        size=12.0,
        is_persistent=args.duration == 0.0,
        duration=args.duration,
    )
    print(f"Rendered {len(points)} occupied voxels as points in UE4.")


if __name__ == "__main__":
    main()
