"""Minimal BFS path render: given start/goal, compute GPU BFS path and render in UE4 via AirSim."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

try:
    import airsim  # type: ignore
except Exception:  # pragma: no cover
    airsim = None

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch  BFS  torch AirSim  CUDA ")

from src.config import load_config
from src.navigation import GPUPathField


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a BFS path between two points and render in AirSim.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("SX", "SY", "SZ"), help="Start position (world, meters).")
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("GX", "GY", "GZ"), help="Goal position (world, meters).")
    parser.add_argument("--npy", type=Path, default=Path("artifacts/level_occupancy.npy"), help="Occupancy grid path.")
    parser.add_argument("--voxel-size", type=float, default=None, help="Override voxel size (defaults to config/environment).")
    parser.add_argument("--color", type=float, nargs=4, default=[0.0, 1.0, 1.0, 1.0], help="RGBA color for path.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for rendered markers (0 = persistent).")
    return parser.parse_args()


def load_origin(meta_path: Path) -> np.ndarray:
    if meta_path.exists():
        try:
            import json

            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return np.array(data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        except Exception:
            pass
    return np.zeros(3, dtype=np.float64)


def main() -> None:
    if airsim is None:
        raise SystemExit("airsim package not installed.")

    args = parse_args()
    cfg = load_config()

    occ = np.load(args.npy)
    voxel_size = float(args.voxel_size) if args.voxel_size is not None else float(cfg.environment.voxel_size_m)
    origin = load_origin(args.npy.with_suffix(".json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_field = GPUPathField(
        occupancy_grid=occ,
        voxel_size=voxel_size,
        origin=origin,
        num_envs=1,
        device=device,
        downsample_factor=3,
    )

    start = np.array(args.start, dtype=np.float64)
    goal = np.array(args.goal, dtype=np.float64)
    print(f"start:{start},goal:{goal}")
    path = path_field.plan_path(start_world=start, goal_world=goal, max_steps=500)
    print(f"path first/last: {path[0]} -> {path[-1]}, length={len(path)}")
    print(f"path:{path}")
    if len(path) < 2:
        print("No path found.")
        return

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.simFlushPersistentMarkers()

    ned_path = []
    for p in path:
        ned_path.append(airsim.Vector3r(float(p[0]), float(p[1]), float(-p[2])))

    client.simPlotPoints(
        ned_path,
        color_rgba=args.color,
        size=8.0,
        is_persistent=args.duration == 0.0,
        duration=args.duration,
    )
    print(f"Rendered BFS path with {len(path)} points.")


if __name__ == "__main__":
    main()
