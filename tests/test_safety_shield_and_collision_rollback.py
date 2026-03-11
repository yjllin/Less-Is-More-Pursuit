"""Sanity tests for safety shield action adjustment and collision rollback.

These tests target the same kernels used by VectorizedMutualAStarEnvV2 to keep
behavior aligned with training.
"""

from __future__ import annotations
import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np

from src.environment.batch_kernels import (
    batch_apply_safety_shield,
    batch_rollback_collisions_v2,
)


def _find_dir_index(target_dir: tuple[int, int, int]) -> int:
    """Return index in 26-direction order used by batch_apply_safety_shield."""
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if (dx, dy, dz) == target_dir:
                    return idx
                idx += 1
    raise ValueError(f"Direction not found: {target_dir}")


def test_safety_shield_modifies_actions() -> None:
    actions = np.zeros((1, 1, 4), dtype=np.float64)
    actions[0, 0, 0] = 5.0
    lidar = np.ones((1, 1, 26), dtype=np.float64)
    front_idx = _find_dir_index((1, 0, 0))
    lidar[0, 0, front_idx] = 0.0
    # Target far away, so normal shield threshold applies
    target_dist = np.array([[100.0]], dtype=np.float64)
    capture_radius = np.array([5.0], dtype=np.float64)

    triggered = batch_apply_safety_shield(
        actions,
        lidar,
        min_dist_m=5.0,
        lidar_max_range=50.0,
        target_dist=target_dist,
        capture_radius=capture_radius,
    )

    assert bool(triggered[0, 0]) is True
    assert actions[0, 0, 0] != 5.0


def test_collision_rollback() -> None:
    pos = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    prev_pos = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    vel = np.array([[[1.0, -1.0, 0.5]]], dtype=np.float64)
    collisions = np.array([[True]], dtype=np.bool_)

    batch_rollback_collisions_v2(pos, vel, prev_pos, collisions)

    assert np.allclose(pos, prev_pos)
    assert np.allclose(vel, 0.0)


def test_no_collision_no_rollback() -> None:
    pos = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    prev_pos = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    vel = np.array([[[1.0, -1.0, 0.5]]], dtype=np.float64)
    collisions = np.array([[False]], dtype=np.bool_)

    batch_rollback_collisions_v2(pos, vel, prev_pos, collisions)

    assert np.allclose(pos, np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64))
    assert np.allclose(vel, np.array([[[1.0, -1.0, 0.5]]], dtype=np.float64))


if __name__ == "__main__":
    test_safety_shield_modifies_actions()
    test_collision_rollback()
    test_no_collision_no_rollback()
    print("OK")
