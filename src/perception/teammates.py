"""
Teammate feature extraction.
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
def _build_teammate_feats_kernel(
    agent_pos: np.ndarray,
    teammate_positions: np.ndarray, # Shape (M, 3)
    top_k: int,
    visibility_radius: float
) -> np.ndarray:
    """
    Computes top-k nearest teammates features.
    Features: [dx, dy, dz, distance, is_visible]
    """
    m_teammates = teammate_positions.shape[0]
    # Initialize with zeros (padding)
    feats = np.zeros((top_k, 5), dtype=np.float64)
    
    if m_teammates == 0:
        return feats
        
    # 1. Compute distances
    # Avoid creating intermediate diff arrays, calculate squared dist first
    dists = np.empty(m_teammates, dtype=np.float64)
    
    for i in range(m_teammates):
        dx = teammate_positions[i, 0] - agent_pos[0]
        dy = teammate_positions[i, 1] - agent_pos[1]
        dz = teammate_positions[i, 2] - agent_pos[2]
        dists[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
        
    # 2. Sort to find nearest
    # argsort is supported in Numba
    order = np.argsort(dists)
    
    # 3. Fill features
    count = min(m_teammates, top_k)
    
    for i in range(count):
        idx = order[i]
        d = dists[idx]
        
        # [dx, dy, dz]
        feats[i, 0] = teammate_positions[idx, 0] - agent_pos[0]
        feats[i, 1] = teammate_positions[idx, 1] - agent_pos[1]
        feats[i, 2] = teammate_positions[idx, 2] - agent_pos[2]
        
        # [dist]
        feats[i, 3] = d
        
        # [is_visible]
        if d <= visibility_radius:
            feats[i, 4] = 1.0
        else:
            feats[i, 4] = 0.0
            
    return feats

# =============================================================================
# Python Interface
# =============================================================================

def build_teammate_features(
    agent_position: NDArray[np.float64],
    teammates: list[NDArray[np.float64]] | NDArray[np.float64],
    *,
    top_k: int,
    visibility_radius: float,
) -> NDArray[np.float64]:
    """
    Return an array of shape (top_k, 5) with `[dx, dy, dz, dist, is_visible]`.
    Optimized to handle both list of arrays and numpy arrays.
    """
    
    # Fast path: handle empty inputs
    if len(teammates) == 0:
        return np.zeros((top_k, 5), dtype=np.float64)

    # Convert list to array if necessary
    # Ideally, the caller should pass a single (M, 3) array to avoid this stacking overhead
    if isinstance(teammates, list):
        teammates_arr = np.stack(teammates)
    else:
        teammates_arr = teammates

    # Call Numba Kernel
    return _build_teammate_feats_kernel(
        agent_position,
        teammates_arr,
        int(top_k),
        float(visibility_radius)
    )