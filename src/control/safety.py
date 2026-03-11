"""Safety shield to prevent imminent collisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def safety_shield(
    lidar_points: NDArray[np.float64],
    intended_action: NDArray[np.float64],
    *,
    min_distance: float,
) -> tuple[NDArray[np.float64], bool]:
    """
    Damping Safety Shield: Limit inward velocity based on obstacle distance.

    lidar_points: point cloud in RL Body Frame (Z-up).
    intended_action: [vx, vy, vz, yaw_rate] in RL Body Frame.
    """

    if lidar_points.size == 0:
        return intended_action, False

    # Find the closest obstacle point.
    distances = np.linalg.norm(lidar_points, axis=1)
    min_idx = int(np.argmin(distances))
    min_dist = float(distances[min_idx])
    
    if min_dist >= min_distance or min_dist <= 1e-6:
        return intended_action, False

    closest_vec = lidar_points[min_idx]
    

    obstacle_dir = closest_vec / min_dist 


    intended_vel = intended_action[:3]

    proj = np.dot(intended_vel, obstacle_dir)
    
    safe_vel = intended_vel.copy()
    
    if proj > 0:
        damp = max(0.0, min_dist / min_distance)
        new_proj = proj * damp
        delta = proj - new_proj
        safe_vel -= delta * obstacle_dir
    
    final_action = intended_action.copy()
    final_action[:3] = safe_vel
    
    if closest_vec[0] > 0 and min_dist < min_distance * 0.4:
        avoid_turn = -1.5 if closest_vec[1] > 0 else 1.5
        final_action[3] = avoid_turn  # Override the original turn command.

    # Apply a global speed clamp after the avoidance update.
    final_vel_norm = np.linalg.norm(final_action[:3])
    if final_vel_norm > 8.0:
        final_action[:3] = final_action[:3] / final_vel_norm * 8.0

    return final_action, True
