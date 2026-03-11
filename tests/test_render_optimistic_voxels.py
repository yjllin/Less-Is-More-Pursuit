"""Render optimistic downsampled occupancy (BFS grid) into UE4 via AirSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np
import torch

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import airsim  # type: ignore
except Exception:  # pragma: no cover
    airsim = None

from src.navigation.gpu_path_guider_v2 import optimistic_downsample_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render optimistic downsampled occupancy grid into UE4.")
    parser.add_argument("--npy", type=Path, default=Path("artifacts/level_occupancy.npy"), help="Occupancy .npy path.")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Meters per voxel (original grid).")
    parser.add_argument("--factor", type=int, default=3, help="Optimistic downsample factor.")
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=None,
        metavar=("OX", "OY", "OZ"),
        help="Grid origin; if omitted will try to read from <npy>.json origin.",
    )
    parser.add_argument("--color", type=float, nargs=4, default=[0.0, 1.0, 1.0, 1.0], help="RGBA color for occupied voxels.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for markers (0 = persistent).")
    parser.add_argument("--downsample-plot", type=int, default=1, help="Further subsample voxels for plotting.")
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

    occ_tensor = torch.from_numpy(occ.astype(np.float32))
    ds = optimistic_downsample_3d(occ_tensor, factor=args.factor).cpu().numpy()
    effective_voxel = voxel * args.factor

    occupied = np.argwhere(ds >= 0.5)
    if occupied.size == 0:
        print("No occupied voxels in downsampled grid.")
        return
    if args.downsample_plot > 1:
        occupied = occupied[:: args.downsample_plot]

    # Render voxel centers as points (consistent with test_frontier_allocation.py usage)
    centers = (occupied.astype(np.float64) + 0.5) * effective_voxel + origin
    ned = centers.copy()
    ned[:, 2] *= -1.0
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.simFlushPersistentMarkers()
    client.simPlotPoints(
        [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in ned],
        color_rgba=args.color,
        size=9.0,
        is_persistent=args.duration == 0.0,
        duration=args.duration,
    )
    print(
        f"Rendered {len(centers)} optimistic voxels as points (factor={args.factor}, voxel={effective_voxel}m) "
        f"from {args.npy} into UE4."
    )


if __name__ == "__main__":
    main()
