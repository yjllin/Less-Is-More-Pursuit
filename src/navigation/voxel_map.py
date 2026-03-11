"""
Numba-optimized 3D Voxel Map.
Eliminates NumPy overhead for grid updates and raycasting to achieve >1000 FPS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback for dev environments
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# Numba Kernels (The Engine)
# =============================================================================

@njit(cache=True, parallel=True)
def _kernel_mark_agent_presence(
    visits: np.ndarray,
    positions: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    radius: float,
    grid_shape: np.ndarray
):
    """
    Updates 'visits' grid around agents using direct iteration.
    Parallelized over agents.
    """
    n_agents = positions.shape[0]
    r_vox = radius / voxel_size
    r_vox_sq = r_vox * r_vox
    ceil_r = int(np.ceil(r_vox))
    
    # Iterate over agents in parallel
    for i in prange(n_agents):
        # Local coordinates relative to map origin
        lx = positions[i, 0] - origin[0]
        ly = positions[i, 1] - origin[1]
        lz = positions[i, 2] - origin[2]
        
        # Center voxel index
        cx = int(np.floor(lx / voxel_size))
        cy = int(np.floor(ly / voxel_size))
        cz = int(np.floor(lz / voxel_size))
        
        # Bounding box clamping
        min_x = max(0, cx - ceil_r)
        max_x = min(grid_shape[0], cx + ceil_r + 1)
        min_y = max(0, cy - ceil_r)
        max_y = min(grid_shape[1], cy + ceil_r + 1)
        min_z = max(0, cz - ceil_r)
        max_z = min(grid_shape[2], cz + ceil_r + 1)
        
        # Iterate bounding box
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                for z in range(min_z, max_z):
                    # Distance check (voxel coordinates)
                    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
                    if dist_sq <= r_vox_sq:
                        # Atomic add is not strictly necessary if agents don't overlap much,
                        # but safer. However, race condition here just means undersounting,
                        # which is acceptable for 'visits'.
                        visits[x, y, z] += 1


@njit(cache=True)
def _kernel_integrate_hits(
    hits: np.ndarray,
    visits: np.ndarray,
    points: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    grid_shape: np.ndarray
):
    """
    Directly maps lidar points to voxels and increments hit counts.
    """
    n_points = points.shape[0]
    for i in range(n_points):
        # Index calculation
        lx = points[i, 0] - origin[0]
        ly = points[i, 1] - origin[1]
        lz = points[i, 2] - origin[2]
        
        ix = int(np.floor(lx / voxel_size))
        iy = int(np.floor(ly / voxel_size))
        iz = int(np.floor(lz / voxel_size))
        
        # Bounds check
        if 0 <= ix < grid_shape[0] and \
           0 <= iy < grid_shape[1] and \
           0 <= iz < grid_shape[2]:
            hits[ix, iy, iz] += 1
            visits[ix, iy, iz] += 1


@njit(cache=True)
def _bresenham_trace(
    visits: np.ndarray,
    x0: int, y0: int, z0: int,
    x1: int, y1: int, z1: int,
    grid_shape: np.ndarray
):
    """Helper: 3D Bresenham line algorithm to mark visited cells along a ray."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    # Check dominant axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            if 0 <= x0 < grid_shape[0] and 0 <= y0 < grid_shape[1] and 0 <= z0 < grid_shape[2]:
                visits[x0, y0, z0] += 1
            x0 += sx
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            if 0 <= x0 < grid_shape[0] and 0 <= y0 < grid_shape[1] and 0 <= z0 < grid_shape[2]:
                visits[x0, y0, z0] += 1
            y0 += sy
            if p1 >= 0:
                x0 += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            if 0 <= x0 < grid_shape[0] and 0 <= y0 < grid_shape[1] and 0 <= z0 < grid_shape[2]:
                visits[x0, y0, z0] += 1
            z0 += sz
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += sx
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx


@njit(cache=True, parallel=True)
def _kernel_mark_free_along_paths(
    visits: np.ndarray,
    origins: np.ndarray,
    ends: np.ndarray,
    map_origin: np.ndarray,
    voxel_size: float,
    grid_shape: np.ndarray
):
    """
    Marks cells along rays as visited (free) using Bresenham.
    Parallelized over rays.
    """
    n_rays = ends.shape[0]
    single_origin = (origins.shape[0] == 1)
    
    for i in prange(n_rays):
        start_pos = origins[0] if single_origin else origins[i]
        end_pos = ends[i]
        
        # Calculate start indices
        x0 = int(np.floor((start_pos[0] - map_origin[0]) / voxel_size))
        y0 = int(np.floor((start_pos[1] - map_origin[1]) / voxel_size))
        z0 = int(np.floor((start_pos[2] - map_origin[2]) / voxel_size))
        
        # Calculate end indices
        x1 = int(np.floor((end_pos[0] - map_origin[0]) / voxel_size))
        y1 = int(np.floor((end_pos[1] - map_origin[1]) / voxel_size))
        z1 = int(np.floor((end_pos[2] - map_origin[2]) / voxel_size))
        
        _bresenham_trace(visits, x0, y0, z0, x1, y1, z1, grid_shape)


@njit(cache=True)
def _kernel_frontier_mask(
    visits: np.ndarray,
    hits: np.ndarray,
    threshold: float,
    grid_shape: np.ndarray
) -> np.ndarray:
    """
    Computes frontier mask in a single pass.
    Frontier = Cell is Free (Known) AND Cell has an Unknown Neighbor.
    """
    frontier = np.zeros(visits.shape, dtype=np.bool_)
    
    # Iterate internal volume to simplify boundary checks
    # Boundaries are rarely frontiers in this context anyway
    for x in range(1, grid_shape[0]-1):
        for y in range(1, grid_shape[1]-1):
            for z in range(1, grid_shape[2]-1):
                
                v_curr = visits[x, y, z]
                
                # Condition 1: Must be Visited (Known)
                if v_curr == 0:
                    continue
                
                # Condition 2: Must be Free (Occupancy < Threshold)
                # ratio = hits / visits
                if (hits[x, y, z] / v_curr) > threshold:
                    continue
                
                # Condition 3: Must have at least one Unknown (Unvisited) neighbor
                # 6-connectivity check
                if (visits[x+1, y, z] == 0 or visits[x-1, y, z] == 0 or
                    visits[x, y+1, z] == 0 or visits[x, y-1, z] == 0 or
                    visits[x, y, z+1] == 0 or visits[x, y, z-1] == 0):
                    
                    frontier[x, y, z] = True
                    
    return frontier


# =============================================================================
# VoxelMap3D Class
# =============================================================================

class VoxelMap3D:
    def __init__(
        self,
        world_size_m: tuple[float, float, float],
        voxel_size_m: float,
        occupancy_threshold: float,
        origin: tuple[float, float, float] | None = None,
    ) -> None:
        self.world_size = world_size_m
        self.voxel_size = float(voxel_size_m)
        self.origin = np.array(origin if origin is not None else (0.0, 0.0, 0.0), dtype=np.float64)
        self.shape = (
            int(round(world_size_m[0] / voxel_size_m)),
            int(round(world_size_m[1] / voxel_size_m)),
            int(round(world_size_m[2] / voxel_size_m)),
        )
        self.occupancy_threshold = float(occupancy_threshold)
        
        # Precompute shape array for Numba
        self._grid_shape_arr = np.array(self.shape, dtype=np.int64)
        
        self.reset()

    def reset(self) -> None:
        self.hits = np.zeros(self.shape, dtype=np.int32)
        self.visits = np.zeros(self.shape, dtype=np.int32)

    def _indices_from_points(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """Convert world coordinates to grid indices."""
        coords = np.floor((points - self.origin) / self.voxel_size).astype(int)
        max_idx = np.array(self.shape, dtype=int) - 1
        coords = np.clip(coords, 0, max_idx)
        return coords

    def pos_to_idx(self, pos: NDArray[np.float64]) -> tuple[int, int, int]:
        """Convert a single world position to grid indices (no clamping)."""
        coords = np.floor((pos - self.origin) / self.voxel_size).astype(int)
        return (int(coords[0]), int(coords[1]), int(coords[2]))

    def mark_agent_presence(self, positions: NDArray[np.float64], radius: float | None = None) -> None:
        """Mark voxels around agents as visited."""
        if positions.size == 0:
            return
        
        r = float(radius) if radius is not None else 0.0
        
        # Call Numba Kernel
        _kernel_mark_agent_presence(
            self.visits,
            positions.astype(np.float64),
            self.origin,
            self.voxel_size,
            r,
            self._grid_shape_arr
        )

    def integrate_lidar_hits(self, points: NDArray[np.float64]) -> None:
        """Increment hit count for voxels containing lidar points."""
        if points.size == 0:
            return
        
        # Call Numba Kernel
        _kernel_integrate_hits(
            self.hits,
            self.visits,
            points.astype(np.float64),
            self.origin,
            self.voxel_size,
            self._grid_shape_arr
        )

    def mark_free_along_paths(self, origins: NDArray[np.float64], points: NDArray[np.float64]) -> None:
        """Mark voxels along the ray from origin to point as visited (free)."""
        if points.size == 0 or origins.size == 0:
            return
        
        # Call Numba Kernel
        _kernel_mark_free_along_paths(
            self.visits,
            origins.astype(np.float64),
            points.astype(np.float64),
            self.origin,
            self.voxel_size,
            self._grid_shape_arr
        )

    def occupancy_grid(self) -> NDArray[np.int_]:
        """
        Returns binary occupancy grid (1=Occupied, 0=Free/Unknown).
        """
        # Vectorized calculation is fast enough here
        visited_mask = self.visits > 0
        grid = np.zeros(self.shape, dtype=np.int32)
        
        if np.any(visited_mask):
            # Avoid division by zero warnings
            # ratio = hits / visits
            hits_masked = self.hits[visited_mask]
            visits_masked = self.visits[visited_mask]
            # Floating point division
            ratios = hits_masked.astype(np.float32) / visits_masked.astype(np.float32)
            
            occupied_indices = ratios > self.occupancy_threshold
            
            # Create a temp flat array to assign
            flat_grid = grid[visited_mask]
            flat_grid[occupied_indices] = 1
            grid[visited_mask] = flat_grid
            
        return grid

    def free_mask(self) -> NDArray[np.bool_]:
        """Returns boolean mask where True = Known Free Space."""
        # Free = Visited AND (Hits/Visits <= Threshold)
        # Avoid float division for speed: Hits <= Visits * Threshold
        is_visited = self.visits > 0
        threshold_hits = self.visits * self.occupancy_threshold
        is_free_ratio = self.hits <= threshold_hits
        return np.logical_and(is_visited, is_free_ratio)

    def unknown_mask(self) -> NDArray[np.bool_]:
        """Returns boolean mask where True = Unknown (Unvisited)."""
        return self.visits == 0

    def frontier_mask(self) -> NDArray[np.bool_]:
        """
        Returns mask of Frontier voxels (Known Free adjacent to Unknown).
        Uses Numba kernel for high speed 3D convolution-like logic.
        """
        return _kernel_frontier_mask(
            self.visits, 
            self.hits, 
            self.occupancy_threshold, 
            self._grid_shape_arr
        )

    def world_from_index(self, idx: tuple[int, int, int]) -> NDArray[np.float64]:
        """Convert grid index to world center coordinate."""
        return (np.array(idx, dtype=np.float64) + 0.5) * self.voxel_size + self.origin
    
    def decay_visits(self, decay_factor: float = 0.999) -> None:
        """
        Simulate uncertainty increase over time (optional).
        Currently just a placeholder or simple decrement if using int visits.
        For logic consistency with paper, usually implemented in upper layer or by resetting.
        If using integers, decay is hard. 
        """
        # If needed for 're-exploration', logic would go here.
        # For int array, maybe set visits = visits * factor?
        pass
