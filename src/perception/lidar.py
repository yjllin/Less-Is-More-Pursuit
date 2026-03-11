"""
Lidar preprocessing utilities.
Numba-optimized for high-frequency RL training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# =============================================================================
# Numba Kernels
# =============================================================================

@njit(cache=True)
def _bin_lidar_kernel(
    points: np.ndarray,
    max_range: float,
    floor: float
) -> np.ndarray:
    """
    Core logic for 3x3x3 binning.
    Iterates through points in C-speed instead of Python.
    """
    # Initialize sectors with max_range
    # 26 sectors (3*3*3 - 1 center)
    sector_dists = np.full(26, max_range, dtype=np.float64)
    
    n_points = points.shape[0]
    
    # Pre-calculate constants
    scale = 1.5 / max_range
    offset = 1.5
    
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        
        # Calculate Euclidean distance
        dist = np.sqrt(x*x + y*y + z*z)
        
        # Determine bin coordinates (0, 1, 2)
        # Logic: ((val / max) * 1.5 + 1.5) clipped to [0, 2]
        # This maps [-max, -max/3] -> 0, [-max/3, max/3] -> 1, [max/3, max] -> 2
        
        nx = int(x * scale + offset)
        ny = int(y * scale + offset)
        nz = int(z * scale + offset)
        
        # Manual clip to be safe
        if nx < 0: nx = 0
        elif nx > 2: nx = 2
        
        if ny < 0: ny = 0
        elif ny > 2: ny = 2
        
        if nz < 0: nz = 0
        elif nz > 2: nz = 2
        
        # Calculate flat index (0-26)
        # x*9 + y*3 + z
        flat_idx = nx * 9 + ny * 3 + nz
        
        # Center bin is (1, 1, 1) -> 1*9 + 1*3 + 1 = 13
        if flat_idx == 13:
            continue
            
        # Map flat index (0..26) to sector index (0..25)
        # Skip 13
        if flat_idx > 13:
            sector_idx = flat_idx - 1
        else:
            sector_idx = flat_idx
            
        # Min reduction
        if dist < sector_dists[sector_idx]:
            sector_dists[sector_idx] = dist
            
    # Clip floor (min distance)
    for i in range(26):
        if sector_dists[i] < floor:
            sector_dists[i] = floor
            
    return sector_dists

# =============================================================================
# Python Interfaces
# =============================================================================

def downsample_lidar(points: NDArray[np.float64], max_points: int) -> NDArray[np.float64]:
    """
    Randomly subsample point cloud if it exceeds max_points.
    Optimized to use shuffling instead of choice(replace=False) for speed.
    """
    n = points.shape[0]
    if n <= max_points:
        return points
    
    # Method 1: Shuffle indices (Faster than random.choice for large arrays)
    # indices = np.arange(n)
    # np.random.shuffle(indices)
    # return points[indices[:max_points]]

    # Method 2: Fast random selection (might have duplicates, but faster)
    # For Lidar binning, duplicates don't matter much.
    # idx = np.random.randint(0, n, size=max_points)
    # return points[idx]

    # Method 3: Choice (Standard) - keeping it correct but assuming N isn't massive
    # If N is huge (e.g. 20k), this is slow. 
    # Since we aim for speed:
    idx = np.random.choice(n, size=max_points, replace=False)
    return points[idx]


def bin_lidar_points(
    points: NDArray[np.float64],
    bin_shape: tuple[int, int, int],
    max_range: float,
    floor: float = 0.05,
) -> NDArray[np.float64]:
    """
    Convert a point cloud into 26-direction nearest distances.
    The space is divided into a 3x3x3 neighborhood; the center bin is skipped.
    """
    if points.size == 0:
        return np.full(26, max_range, dtype=np.float64)

    # Validation (Removed for speed in loop, assume correct input from env)
    # if bin_shape != (3, 3, 3):
    #     raise ValueError("Only 3x3x3 binning is supported.")

    # Call Numba Kernel
    return _bin_lidar_kernel(points, float(max_range), float(floor))