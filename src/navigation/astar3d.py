"""
High-Performance 3D A* Planner using Numba.
Optimized for 150x150x50 grids (2m voxels in 300m world).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, int32, float32
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback for dev environments without GPU/Numba
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Mock NumbaList for pure python fallback
    def NumbaList(): return []

GridIndex = Tuple[int, int, int]

# =============================================================================
# Numba Optimized Kernels (The Engine)
# =============================================================================

@njit(cache=True)
def _flatten_idx(x, y, z, stride_y, stride_z):
    return x * stride_y * stride_z + y * stride_z + z

@njit(cache=True)
def _unflatten_idx(idx, stride_y, stride_z):
    x = idx // (stride_y * stride_z)
    rem = idx % (stride_y * stride_z)
    y = rem // stride_z
    z = rem % stride_z
    return x, y, z

@njit(cache=True)
def _heuristic_euclidean(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

@njit(cache=True)
def astar_kernel(
    start_node: tuple, 
    goal_node: tuple, 
    grid: np.ndarray, 
    heuristic_weight: float,
    max_nodes: int
) -> List[Tuple[int, int, int]]:
    """
    Core A* logic compiled with Numba.
    Uses flat arrays instead of dicts for extreme speed.
    """
    shape_x, shape_y, shape_z = grid.shape
    
    # Precompute strides for flat indexing
    stride_y = shape_y
    stride_z = shape_z
    total_size = shape_x * shape_y * shape_z
    
    start_flat = _flatten_idx(start_node[0], start_node[1], start_node[2], stride_y, stride_z)
    goal_flat = _flatten_idx(goal_node[0], goal_node[1], goal_node[2], stride_y, stride_z)
    
    # Allocations
    # g_score initialized to infinity
    g_score = np.full(total_size, np.inf, dtype=np.float32)
    g_score[start_flat] = 0.0
    
    # came_from stores parent index (init to -1)
    came_from = np.full(total_size, -1, dtype=np.int32)
    
    # Priority Queue: stores (f_score, flat_index)
    # Numba supports heapq on lists
    open_set = [(0.0, start_flat)] # Start with f=0 is fine, heuristic added later or ignored for start
    
    # Track visited to avoid re-adding to heap excessively 
    # (Optional but good for dense graphs, here we rely on g_score check)
    
    gx, gy, gz = goal_node
    
    nodes_explored = 0
    
    # Neighbors: 26-connectivity
    # Pre-allocate deltas
    dx_list = np.array([1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int8)
    dy_list = np.array([0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 0, 0, 1, 1], dtype=np.int8)
    dz_list = np.array([0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1], dtype=np.int8)
    # Note: 6-connectivity is faster if diagonal movement isn't strictly required.
    # To switch to 6-conn, just keep the first 6 elements.
    # Let's assume 26 for smoothness, but you can slice [:6] for speed.
    num_neighbors = 26 

    path_found = False
    
    while len(open_set) > 0:
        # Check node limit
        if nodes_explored >= max_nodes:
            break
        nodes_explored += 1
        
        # Pop smallest f
        curr_f, curr_flat = heapq.heappop(open_set)
        
        # Lazy deletion: if we found a better path to this node already, skip
        if curr_f > g_score[curr_flat] + heuristic_weight * 200.0: # Loose bound check or exact check
             # Exact: if curr_f > g_score[curr_flat] + h(curr, goal), but h is static.
             # Simply checking g is enough if we store (f, idx)
             pass 

        if curr_flat == goal_flat:
            path_found = True
            break
            
        cx, cy, cz = _unflatten_idx(curr_flat, stride_y, stride_z)
        
        # Expand neighbors
        for i in range(num_neighbors):
            nx = cx + dx_list[i]
            ny = cy + dy_list[i]
            nz = cz + dz_list[i]
            
            # Bounds check
            if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                # Collision check
                if grid[nx, ny, nz] == 0:
                    # Strictly prevent corner-cutting on diagonal moves:
                    # require all axis-aligned and face-diagonal neighbor cells to be free.
                    dx = dx_list[i]
                    dy = dy_list[i]
                    dz = dz_list[i]
                    if (dx != 0 and dy != 0) or (dx != 0 and dz != 0) or (dy != 0 and dz != 0):
                        # Strictly prevent corner-cutting:
                        # 2D diagonal: both axis-aligned neighbors must be free.
                        if dx != 0 and dy != 0 and (grid[cx + dx, cy, cz] == 1 or grid[cx, cy + dy, cz] == 1):
                            continue
                        if dx != 0 and dz != 0 and (grid[cx + dx, cy, cz] == 1 or grid[cx, cy, cz + dz] == 1):
                            continue
                        if dy != 0 and dz != 0 and (grid[cx, cy + dy, cz] == 1 or grid[cx, cy, cz + dz] == 1):
                            continue
                        # 3D diagonal: all three axis-aligned neighbors must be free.
                        if dx != 0 and dy != 0 and dz != 0:
                            if grid[cx + dx, cy, cz] == 1 or grid[cx, cy + dy, cz] == 1 or grid[cx, cy, cz + dz] == 1:
                                continue
                    # Cost: 1.0 for straight, 1.414 for diagonal, 1.732 for 3d diag
                    # Quick calculation:
                    dist = math.sqrt(dx_list[i]**2 + dy_list[i]**2 + dz_list[i]**2)
                    tentative_g = g_score[curr_flat] + dist
                    
                    neighbor_flat = _flatten_idx(nx, ny, nz, stride_y, stride_z)
                    
                    if tentative_g < g_score[neighbor_flat]:
                        came_from[neighbor_flat] = curr_flat
                        g_score[neighbor_flat] = tentative_g
                        
                        h = _heuristic_euclidean(nx, ny, nz, gx, gy, gz)
                        f = tentative_g + heuristic_weight * h
                        heapq.heappush(open_set, (f, neighbor_flat))
    
    # Reconstruct path
    path = NumbaList()
    if path_found:
        curr = goal_flat
        while curr != -1:
            cx, cy, cz = _unflatten_idx(curr, stride_y, stride_z)
            # Numba requires typed tuples for list
            # We append to list then reverse later or here
            path.append((cx, cy, cz))
            curr = came_from[curr]
    
    return path

@njit(cache=True)
def _bresenham_3d_blocked(
    x0: int, y0: int, z0: int,
    x1: int, y1: int, z1: int,
    grid: np.ndarray,
) -> bool:
    """3D Bresenham line algorithm to check if path is blocked."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            x0 += sx
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            if grid[x0, y0, z0] == 1: return True
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            y0 += sy
            if p1 >= 0:
                x0 += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            if grid[x0, y0, z0] == 1: return True
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            z0 += sz
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += sx
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            if grid[x0, y0, z0] == 1: return True
    return False

# =============================================================================
# High-Level Wrapper
# =============================================================================

@dataclass
class _PathCache:
    start: GridIndex
    goal: GridIndex
    path: List[GridIndex]
    planned_step: int

class AStar3D:
    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        voxel_size: float,
        *,
        cache_refresh_steps: int,
        lookahead_m: float,
        heuristic_weight: float = 1.0,
        origin: tuple[float, float, float] | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.voxel_size = float(voxel_size)
        self.origin = np.array(origin if origin is not None else (0.0, 0.0, 0.0), dtype=np.float64)
        self.cache_refresh_steps = int(cache_refresh_steps)
        self.lookahead_m = float(lookahead_m)
        self.heuristic_weight = max(1.0, float(heuristic_weight))
        self._grid: Optional[NDArray[np.int_]] = None
        self._cache: Optional[_PathCache] = None
        self._cache_by_key: Dict[object, _PathCache] = {}
        self._last_world_path: List[NDArray[np.float64]] = []
        self._last_world_path_by_key: Dict[object, List[NDArray[np.float64]]] = {}
        
        # Max nodes limit to prevent freezing on unreachable targets (e.g. goal inside wall)
        self.max_nodes_explored = 50000 

    def update_grid(self, occupancy_grid: NDArray[np.int_]) -> None:
        if occupancy_grid.shape != self.grid_shape:
            raise ValueError(f"Expected grid shape {self.grid_shape}, got {occupancy_grid.shape}")
        self._grid = occupancy_grid.copy().astype(np.int8) # Ensure int8 for Numba efficiency
        self._cache = None

    def clear_cache_for_env(self, env_id: int) -> None:
        """Clear A* cache entries for a specific environment to prevent memory growth.
        
        Should be called on environment reset to avoid stale paths and memory fragmentation.
        
        Args:
            env_id: Environment ID whose cache entries should be cleared
        """
        keys_to_remove = [k for k in self._cache_by_key.keys() if isinstance(k, tuple) and k[0] == env_id]
        for k in keys_to_remove:
            del self._cache_by_key[k]
        
        # Also clear last_world_path entries
        keys_to_remove = [k for k in self._last_world_path_by_key.keys() if isinstance(k, tuple) and k[0] == env_id]
        for k in keys_to_remove:
            del self._last_world_path_by_key[k]
    
    def clear_all_caches(self) -> None:
        """Clear all A* caches. Call periodically to prevent memory growth."""
        self._cache = None
        self._cache_by_key.clear()
        self._last_world_path.clear()
        self._last_world_path_by_key.clear()

    def compute_direction(
        self,
        start_position: NDArray[np.float64],
        goal_position: NDArray[np.float64],
        current_step: int,
        current_velocity: Optional[NDArray[np.float64]] = None,
        smooth: bool = True,
        cache_key: object | None = None,
    ) -> tuple[NDArray[np.float64], List[NDArray[np.float64]], float]:
        """
        Compute navigation direction and path to goal.
        
        Returns:
            direction: Unit vector pointing towards next waypoint
            world_path: List of waypoints in world coordinates
            path_length: Total path length in meters (computed in Numba)
        """
        if self._grid is None:
            raise RuntimeError("A* grid not initialized.")

        start_cell = self._world_to_cell(start_position)
        goal_cell = self._world_to_cell(goal_position)

        cache = self._cache_by_key.get(cache_key) if cache_key is not None else self._cache
        if self._should_replan(start_cell, goal_cell, current_step, cache):
            path = self._plan_path(start_cell, goal_cell)

            # Smoothing (still done in Python but calls Numba Bresenham)
            # Since path length is small (e.g. < 500 points), Python loop is acceptable here.
            if smooth and len(path) >= 3:
                path = self.smooth_path(path)

            cache = _PathCache(start=start_cell, goal=goal_cell, path=path, planned_step=current_step)
            if cache_key is not None:
                self._cache_by_key[cache_key] = cache
            else:
                self._cache = cache

        path = cache.path if cache else []
        world_path = self._path_to_world(path)
        
        # Compute path length directly (avoid Python loop in caller)
        path_length = self._compute_path_length(world_path)
        
        # Forward filtering (Python/Numpy is fast enough for small arrays)
        if current_velocity is not None and len(world_path) > 1:
            world_path = self._filter_forward_path(start_position, current_velocity, world_path)
        
        if cache_key is not None:
            self._last_world_path_by_key[cache_key] = world_path
        else:
            self._last_world_path = world_path
        direction = self._select_direction(start_position, goal_position, world_path)
        return direction, world_path, path_length
    
    def _compute_path_length(self, world_path: List[NDArray[np.float64]]) -> float:
        """Compute total path length from world path points."""
        if not world_path or len(world_path) < 2:
            return 0.0
        
        total_length = 0.0
        prev = world_path[0]
        for p in world_path[1:]:
            dx = p[0] - prev[0]
            dy = p[1] - prev[1]
            dz = p[2] - prev[2]
            total_length += math.sqrt(dx*dx + dy*dy + dz*dz)
            prev = p
        return total_length

    def last_path(self, cache_key: object | None = None) -> Optional[List[NDArray[np.float64]]]:
        if cache_key is not None:
            path = self._last_world_path_by_key.get(cache_key)
            return [p.copy() for p in path] if path else None
        return [p.copy() for p in self._last_world_path] if self._last_world_path else None

    def _should_replan(self, start: GridIndex, goal: GridIndex, step: int, cache: Optional[_PathCache]) -> bool:
        if cache is None: return True
        if cache.goal != goal: return True
        if cache.start != start: return True
        # Time-based refresh to avoid stale paths
        if step - cache.planned_step >= self.cache_refresh_steps: return True
        return False

    def _plan_path(self, start: GridIndex, goal: GridIndex) -> List[GridIndex]:
        # Snap start/goal to valid bounds
        start = self._clip_index(start)
        goal = self._clip_index(goal)
        
        # Check if inside obstacle
        if self._grid[start] == 1:
            # Simple local search for nearest free (Python is ok for very local search)
            start = self._nearest_free_local(start)
        if self._grid[goal] == 1:
            goal = self._nearest_free_local(goal)
            
        if start == goal:
            return [start]

        # CALL NUMBA KERNEL
        # Note: Numba returns a typed List, we convert to standard Python list
        # Reversing happens here because Numba reconstructed it backwards (Goal -> Start)
        raw_path = astar_kernel(
            start, goal, self._grid, self.heuristic_weight, self.max_nodes_explored
        )
        
        path = list(raw_path)
        # The kernel reconstructs from Goal -> Start, so path[0] is Goal.
        # We want Start -> Goal.
        path.reverse() 
        
        return path

    def _nearest_free_local(self, cell: GridIndex) -> GridIndex:
        # Very simple BFS to find nearest free cell if spawn/target is in wall
        # Cap range to avoid hanging
        x, y, z = cell
        for r in range(1, 5): # Check radius 1 to 4
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if 0 <= nx < self.grid_shape[0] and \
                           0 <= ny < self.grid_shape[1] and \
                           0 <= nz < self.grid_shape[2]:
                            if self._grid[nx, ny, nz] == 0:
                                return (nx, ny, nz)
        return cell

    def _world_to_cell(self, position: NDArray[np.float64]) -> GridIndex:
        idx = np.floor((position - self.origin) / self.voxel_size).astype(int)
        return self._clip_index(tuple(idx))
    
    def _clip_index(self, idx: Tuple[int,int,int]) -> GridIndex:
        ix = min(max(idx[0], 0), self.grid_shape[0]-1)
        iy = min(max(idx[1], 0), self.grid_shape[1]-1)
        iz = min(max(idx[2], 0), self.grid_shape[2]-1)
        return (ix, iy, iz)

    def _cell_to_world(self, cell: GridIndex) -> NDArray[np.float64]:
        return (np.array(cell, dtype=np.float64) + 0.5) * self.voxel_size + self.origin

    def _path_to_world(self, path: List[GridIndex]) -> List[NDArray[np.float64]]:
        return [self._cell_to_world(p) for p in path]

    def _select_direction(
        self,
        start_position: NDArray[np.float64],
        goal_position: NDArray[np.float64],
        world_path: List[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        if not world_path:
            return np.zeros(3, dtype=np.float64)

      
      
      
        best_point = None
        best_dist = 0.0
        
        for point in world_path:
            delta = point - start_position
            dist = float(np.linalg.norm(delta))
            # Skip points that are too close and may already be behind the agent.
            if dist < 1.0:
                continue
            # Return immediately once the lookahead distance is satisfied.
            if dist >= self.lookahead_m:
                return delta / dist
            # Otherwise keep the farthest valid point seen so far.
            if dist > best_dist:
                best_dist = dist
                best_point = point
        # Fall back to the farthest valid point if nothing exceeds the lookahead threshold.
        if best_point is not None:
            delta = best_point - start_position
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                return delta / dist

        # Final fallback: point directly at the goal.
        to_goal = goal_position - start_position
        norm = float(np.linalg.norm(to_goal))
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float64)
        return to_goal / norm

    def smooth_path(self, path: List[GridIndex]) -> List[GridIndex]:
        """Greedy raycast shortcutting."""
        if self._grid is None or len(path) < 3:
            return path
        
        smoothed: List[GridIndex] = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            furthest_visible = current_idx + 1
            # Check ahead a max of 20 points to save compute
            search_end = min(len(path), current_idx + 20)
            
            for check_idx in range(search_end - 1, current_idx, -1):
                x0, y0, z0 = path[current_idx]
                x1, y1, z1 = path[check_idx]
                
                if not _bresenham_3d_blocked(x0, y0, z0, x1, y1, z1, self._grid):
                    furthest_visible = check_idx
                    break
            
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible
            
            # If we reached the end, stop
            if current_idx == len(path) - 1:
                break
        
        return smoothed

    def _filter_forward_path(
        self,
        current_pos: NDArray[np.float64],
        current_velocity: NDArray[np.float64],
        world_path: List[NDArray[np.float64]],
    ) -> List[NDArray[np.float64]]:
        if len(world_path) <= 1:
            return world_path
        
        speed = float(np.linalg.norm(current_velocity))
        if speed < 1e-6:
            return world_path
        
        vel_norm = current_velocity / speed
        filtered: List[NDArray[np.float64]] = []
        
        for i, wp in enumerate(world_path):
          
            if i == len(world_path) - 1:
                filtered.append(wp)
                continue
            
            direction = wp - current_pos
            dist = float(np.linalg.norm(direction))
            # Skip points that are too close and may already be behind the agent.
            if dist < 1.0:
                continue
            
          
            if dist > 1e-6:
                dir_norm = direction / dist
                if np.dot(dir_norm, vel_norm) >= 0.0:
                    filtered.append(wp)
        
        if not filtered:
            return [world_path[-1]]
        return filtered
