"""Unified coordinate system utilities for 3D navigation.

This module provides consistent coordinate transformations between:
- World coordinates (meters, absolute position)
- Grid/voxel coordinates (integer indices)
- Local coordinates (relative to origin)
- Body frame coordinates (relative to agent orientation)

Coordinate System Convention:
- X: Forward (North in world frame)
- Y: Right (East in world frame)  
- Z: Up (altitude)
- Yaw: Rotation around Z axis, 0 = facing +X, positive = counter-clockwise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CoordinateFrame:
    """Defines a coordinate frame with origin and optional rotation."""
    origin: NDArray[np.float64]
    yaw: float = 0.0  # Rotation around Z axis in radians
    
    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64)
        if self.origin.shape != (3,):
            raise ValueError(f"Origin must be 3D, got shape {self.origin.shape}")


def world_to_grid(
    position: NDArray[np.float64],
    origin: NDArray[np.float64],
    voxel_size: float,
    grid_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Convert world coordinates to grid indices.
    
    Args:
        position: 3D position in world coordinates (meters)
        origin: World position of grid origin
        voxel_size: Size of each voxel in meters
        grid_shape: Shape of the grid (nx, ny, nz)
        
    Returns:
        Tuple of (x, y, z) grid indices, clamped to valid range
    """
    local = position - origin
    idx = np.floor(local / voxel_size).astype(int)
    idx = np.clip(idx, [0, 0, 0], np.array(grid_shape) - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


def grid_to_world(
    indices: Tuple[int, int, int],
    origin: NDArray[np.float64],
    voxel_size: float,
) -> NDArray[np.float64]:
    """
    Convert grid indices to world coordinates (center of voxel).
    
    Args:
        indices: Grid indices (x, y, z)
        origin: World position of grid origin
        voxel_size: Size of each voxel in meters
        
    Returns:
        3D position in world coordinates (center of voxel)
    """
    return (np.array(indices, dtype=np.float64) + 0.5) * voxel_size + origin


def world_to_body(
    world_vec: NDArray[np.float64],
    yaw: float,
) -> NDArray[np.float64]:
    """
    Transform a vector from world frame to body frame.
    
    Args:
        world_vec: 3D vector in world coordinates
        yaw: Agent yaw angle in radians (rotation around Z)
        
    Returns:
        3D vector in body frame (X=forward, Y=right, Z=up)
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Rotation matrix for yaw (around Z axis)
    # Body X = World X * cos(yaw) + World Y * sin(yaw)
    # Body Y = -World X * sin(yaw) + World Y * cos(yaw)
    # Body Z = World Z
    return np.array([
        cos_yaw * world_vec[0] + sin_yaw * world_vec[1],
        -sin_yaw * world_vec[0] + cos_yaw * world_vec[1],
        world_vec[2],
    ], dtype=np.float64)


def body_to_world(
    body_vec: NDArray[np.float64],
    yaw: float,
) -> NDArray[np.float64]:
    """
    Transform a vector from body frame to world frame.
    
    Args:
        body_vec: 3D vector in body coordinates
        yaw: Agent yaw angle in radians (rotation around Z)
        
    Returns:
        3D vector in world frame
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Inverse rotation matrix
    return np.array([
        cos_yaw * body_vec[0] - sin_yaw * body_vec[1],
        sin_yaw * body_vec[0] + cos_yaw * body_vec[1],
        body_vec[2],
    ], dtype=np.float64)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def angle_difference(target: float, current: float) -> float:
    """
    Compute shortest angular difference from current to target.
    
    Returns value in [-pi, pi], positive = counter-clockwise.
    """
    diff = target - current
    return normalize_angle(diff)


def heading_from_velocity(velocity: NDArray[np.float64]) -> float:
    """
    Compute heading angle from velocity vector.
    
    Args:
        velocity: 3D velocity vector
        
    Returns:
        Heading angle in radians (0 = +X direction)
    """
    speed_xy = np.sqrt(velocity[0]**2 + velocity[1]**2)
    if speed_xy < 1e-6:
        return 0.0
    return float(np.arctan2(velocity[1], velocity[0]))


def distance_3d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


def distance_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute horizontal (XY) distance between two 3D points."""
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def interpolate_path(
    path: list[NDArray[np.float64]],
    max_segment_length: float,
) -> list[NDArray[np.float64]]:
    """
    Interpolate path to ensure no segment exceeds max length.
    
    Args:
        path: List of 3D waypoints
        max_segment_length: Maximum distance between consecutive points
        
    Returns:
        Interpolated path with additional waypoints
    """
    if len(path) < 2:
        return path
    
    result: list[NDArray[np.float64]] = [path[0]]
    
    for i in range(1, len(path)):
        start = path[i - 1]
        end = path[i]
        dist = distance_3d(start, end)
        
        if dist <= max_segment_length:
            result.append(end)
        else:
            # Add intermediate points
            num_segments = int(np.ceil(dist / max_segment_length))
            for j in range(1, num_segments + 1):
                t = j / num_segments
                point = start + t * (end - start)
                result.append(point)
    
    return result


__all__ = [
    "CoordinateFrame",
    "world_to_grid",
    "grid_to_world",
    "world_to_body",
    "body_to_world",
    "normalize_angle",
    "angle_difference",
    "heading_from_velocity",
    "distance_3d",
    "distance_2d",
    "interpolate_path",
]
