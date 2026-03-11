"""Benchmark/availability test using project AStar3D on artifacts/level_occupancy.npy."""

from __future__ import annotations

import time
import sys
import os
from pathlib import Path

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np

from src.navigation import AStar3D


def run_benchmark():
    npy_path = Path("artifacts/level_occupancy.npy")
    occ = np.load(npy_path).astype(np.int8)
    meta_path = npy_path.with_suffix(".json")
    origin = (0.0, 0.0, 0.0)
    voxel_size = 2.0
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        origin = tuple(meta.get("origin", origin))
        voxel_size = float(meta.get("voxel_size", voxel_size))

    astar = AStar3D(
        grid_shape=occ.shape,
        voxel_size=voxel_size,
        cache_refresh_steps=10,
        lookahead_m=5.0,
        heuristic_weight=1.5,
        origin=origin,
    )
    astar.update_grid(occ)

    start = np.array(origin, dtype=np.float64)
    goal = start + np.array([140.0, 140.0, 80.0])

    # Warmup
    astar.compute_direction(start, goal, current_step=0)

    iters = 200
    t0 = time.perf_counter()
    for i in range(iters):
        astar.compute_direction(start, goal, current_step=i)
    elapsed = time.perf_counter() - t0
    print(f"AStar3D {iters} queries on occupancy grid {occ.shape}: {elapsed*1000:.2f} ms, {iters/elapsed:.1f} qps")


if __name__ == "__main__":
    run_benchmark()
