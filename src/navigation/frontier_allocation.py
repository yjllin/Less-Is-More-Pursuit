"""
Frontier detection and assignment for 3D exploration.
Numba-optimized version for >1000 FPS training.
Enhanced with "Hunting Instinct" - LKP (Last Known Position) bias for target search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .voxel_map import VoxelMap3D

try:
    from numba import njit, float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Mock njit
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

if TYPE_CHECKING:
    from .astar3d import AStar3D


# =============================================================================
# Numba Kernels
# =============================================================================

@njit(cache=True)
def detect_frontiers_kernel(
    visits: np.ndarray,
    occupancy_grid: np.ndarray, # 1=Occupied, 0=Free
    allowed_mask: np.ndarray,   # 1=Allowed, 0=Disallowed
    origin: np.ndarray,
    voxel_size: float,
    max_z_idx: int,
    max_candidates: int,
    confidence_threshold: float = 0.5,
    unvisited_only: boolean = False
) -> np.ndarray:
    """
    Scan grid to find frontier voxels (Free & Low-Confidence & Allowed).
    Returns (N, 3) world coordinates.
    """
    shape = visits.shape
    candidates = np.zeros((max_candidates, 3), dtype=np.float64)
    count = 0
    
    stride = 2  # Skip every other voxel for speed
    
    for x in range(0, shape[0], stride):
        for y in range(0, shape[1], stride):
            for z in range(0, min(shape[2], max_z_idx), stride):
                # Check allowed (allowed_mask is int8: 1=allowed, 0=disallowed)
                if allowed_mask.size > 0:
                    if allowed_mask[x, y, z] == 0:
                        continue
                
                # Check exploration state
                if unvisited_only:
                    if visits[x, y, z] != 0.0:
                        continue
                elif visits[x, y, z] >= confidence_threshold:
                    continue

                # Check free (occupancy == 0)
                if occupancy_grid[x, y, z] == 0:
                    # Found frontier
                    wx = (x + 0.5) * voxel_size + origin[0]
                    wy = (y + 0.5) * voxel_size + origin[1]
                    wz = (z + 0.5) * voxel_size + origin[2]
                    
                    candidates[count, 0] = wx
                    candidates[count, 1] = wy
                    candidates[count, 2] = wz
                    count += 1
                    
                    if count >= max_candidates:
                        return candidates
                            
    return candidates[:count]


@njit(cache=True)
def select_nearest_to_lkp_kernel(
    frontiers: np.ndarray,
    lkp_pos: np.ndarray,
    n_nearest: int
) -> np.ndarray:
    """
    Select N frontiers nearest to LKP (Last Known Position).
    These are "intel points" for hunting instinct.
    """
    n_points = frontiers.shape[0]
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_points <= n_nearest:
        return frontiers.copy()
    
    # Calculate distances to LKP
    dists = np.zeros(n_points, dtype=np.float64)
    for i in range(n_points):
        dx = frontiers[i, 0] - lkp_pos[0]
        dy = frontiers[i, 1] - lkp_pos[1]
        dz = frontiers[i, 2] - lkp_pos[2]
        dists[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Find n_nearest smallest distances using partial sort
    indices = np.argsort(dists)[:n_nearest]
    
    result = np.zeros((n_nearest, 3), dtype=np.float64)
    for i in range(n_nearest):
        result[i] = frontiers[indices[i]]
    
    return result


@njit(cache=True)
def fps_select_kernel(
    points: np.ndarray,
    agent_pos: np.ndarray,
    k: int,
    cos_threshold: float
) -> np.ndarray:
    """
    Farthest Point Sampling with angular inhibition.
    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_points <= k:
        return points
        
    # Calculate all distances
    dists = np.zeros(n_points, dtype=np.float64)
    vecs = np.zeros_like(points)
    
    for i in range(n_points):
        dx = points[i, 0] - agent_pos[0]
        dy = points[i, 1] - agent_pos[1]
        dz = points[i, 2] - agent_pos[2]
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        dists[i] = d
        if d > 1e-6:
            vecs[i, 0] = dx / d
            vecs[i, 1] = dy / d
            vecs[i, 2] = dz / d
            
    available = np.ones(n_points, dtype=np.bool_)
    selected_indices = np.empty(k, dtype=np.int64)
    num_selected = 0
    
    while num_selected < k:
        # Argmax distance among available
        max_d = -1.0
        best_idx = -1
        
        has_avail = False
        for i in range(n_points):
            if available[i]:
                has_avail = True
                if dists[i] > max_d:
                    max_d = dists[i]
                    best_idx = i
        
        if best_idx == -1 or not has_avail:
            break
            
        selected_indices[num_selected] = best_idx
        num_selected += 1
        available[best_idx] = False
        
        # Angular inhibition
        sel_vec = vecs[best_idx]
        
        for i in range(n_points):
            if available[i]:
                # Dot product
                dot = sel_vec[0]*vecs[i,0] + sel_vec[1]*vecs[i,1] + sel_vec[2]*vecs[i,2]
                if dot > cos_threshold: # Too close in angle
                    available[i] = False
                    
    # Return selected points
    result = np.zeros((num_selected, 3), dtype=np.float64)
    for i in range(num_selected):
        idx = selected_indices[i]
        result[i] = points[idx]
        
    return result


@njit(cache=True)
def check_line_obstacle(
    start: np.ndarray,
    end: np.ndarray,
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    grid_shape: np.ndarray
) -> bool:
    """3D Bresenham line check."""
    # Convert to indices
    x0 = int((start[0] - origin[0]) / voxel_size)
    y0 = int((start[1] - origin[1]) / voxel_size)
    z0 = int((start[2] - origin[2]) / voxel_size)
    
    x1 = int((end[0] - origin[0]) / voxel_size)
    y1 = int((end[1] - origin[1]) / voxel_size)
    z1 = int((end[2] - origin[2]) / voxel_size)
    
    # Check bounds
    if x0 < 0 or x0 >= grid_shape[0] or y0 < 0 or y0 >= grid_shape[1] or z0 < 0 or z0 >= grid_shape[2]: return True
    if x1 < 0 or x1 >= grid_shape[0] or y1 < 0 or y1 >= grid_shape[1] or z1 < 0 or z1 >= grid_shape[2]: return True

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
            if p1 >= 0: y0 += sy; p1 -= 2 * dx
            if p2 >= 0: z0 += sz; p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            if occupancy_grid[x0, y0, z0] == 1: return True
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            y0 += sy
            if p1 >= 0: x0 += sx; p1 -= 2 * dy
            if p2 >= 0: z0 += sz; p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            if occupancy_grid[x0, y0, z0] == 1: return True
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            z0 += sz
            if p1 >= 0: y0 += sy; p1 -= 2 * dz
            if p2 >= 0: x0 += sx; p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            if occupancy_grid[x0, y0, z0] == 1: return True
            
    return False


@njit(cache=True)
def build_cost_matrix_kernel(
    agent_pos: np.ndarray,
    agent_head: np.ndarray,
    candidates: np.ndarray,
    free_indices: np.ndarray, # indices of free agents
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    w_z: float, w_turn: float, w_climb: float, w_obstacle: float,
    angle_cos_limit: float,
    preferred_layer_z: np.ndarray,
    layer_penalty_weight: float,
    layer_z_bounds: np.ndarray,
    agent_layer_idx: np.ndarray,
    lkp_pos: np.ndarray,
    w_scent: float,
    w_dispersion: float = 50.0  # 
) -> np.ndarray:
    """
    Compute cost matrix in Numba with LKP "scent" attraction and agent dispersion.
    
    Cost formula:
        cost = dist + turn_cost + climb_cost + obs_cost + (dist_to_lkp * w_scent) + dispersion_penalty
    
    Args:
        lkp_pos: Last Known Position of target (3D)
        w_scent: Scent weight - higher means stronger attraction to LKP
                 Suggested: max(0.0, 2.0 * (1.0 - target_lost_steps / 300))
        w_dispersion: Dispersion weight - penalty for agents choosing nearby candidates
    
    Performance: O(N) pre-computation for LKP distance instead of O(N*M) in inner loop.
    """
    n_agents = len(free_indices)
    n_cands = candidates.shape[0]
    costs = np.empty((n_agents, n_cands), dtype=np.float64)
    grid_shape = np.array(occupancy_grid.shape)
    
    # --- Performance: Pre-compute scent cost for each candidate (O(N) instead of O(N*M)) ---
    cand_scent_costs = np.zeros(n_cands, dtype=np.float64)
    lkp_valid = (lkp_pos[0] != 0.0 or lkp_pos[1] != 0.0 or lkp_pos[2] != 0.0)
    if lkp_valid and w_scent > 0.0:
        for j in range(n_cands):
            cand = candidates[j]
            ldx = cand[0] - lkp_pos[0]
            ldy = cand[1] - lkp_pos[1]
            ldz = cand[2] - lkp_pos[2]
            # Apply w_z to maintain 3D distance consistency
            dist_to_lkp = np.sqrt(ldx**2 + ldy**2 + (ldz * w_z)**2)
            cand_scent_costs[j] = dist_to_lkp * w_scent
    
    # --- Main assignment loop ---
    for i in range(n_agents):
        a_idx = free_indices[i]
        pos = agent_pos[a_idx]
        head = agent_head[a_idx]
        head_norm = np.sqrt(head[0]**2 + head[1]**2) # 2D heading norm
        has_layer_constraint = False
        z_min = 0.0
        z_max = 0.0
        if agent_layer_idx.size > 0 and layer_z_bounds.size > 0:
            layer_idx = agent_layer_idx[a_idx]
            if layer_idx >= 0 and layer_idx < layer_z_bounds.shape[0]:
                z_min = layer_z_bounds[layer_idx, 0]
                z_max = layer_z_bounds[layer_idx, 1]
                has_layer_constraint = True
        
        for j in range(n_cands):
            cand = candidates[j]
            if has_layer_constraint and (cand[2] < z_min or cand[2] > z_max):
                costs[i, j] = 1e9
                continue
            dx = cand[0] - pos[0]
            dy = cand[1] - pos[1]
            dz = cand[2] - pos[2]
            
            # 1. Distance
            dist = np.sqrt(dx**2 + dy**2 + (dz * w_z)**2)
            
            # 2. Turn Cost
            turn_cost = 0.0
            if head_norm > 1e-6:
                d_xy = np.sqrt(dx**2 + dy**2)
                if d_xy > 1e-6:
                    dot = (dx * head[0] + dy * head[1]) / (d_xy * head_norm)
                    if dot < -1.0: dot = -1.0
                    elif dot > 1.0: dot = 1.0
                    if dot < angle_cos_limit:
                        costs[i, j] = 1e9
                        continue
                    angle = np.arccos(dot) * 57.2958 # rad to deg
                    turn_cost = angle * w_turn
                    
            # 3. Climb Cost
            climb_cost = abs(dz) * w_climb
            
            # 4. Obstacle Cost
            obs_cost = 0.0
            is_blocked = check_line_obstacle(pos, cand, occupancy_grid, origin, voxel_size, grid_shape)
            if is_blocked:
                obs_cost = w_obstacle
            
            # 5. Dispersion Cost - penalty for candidates too close to other agents
            dispersion_cost = 0.0
            if w_dispersion > 0.0:
                min_agent_dist = 1e9
                for k in range(len(free_indices)):
                    if k != i:  # Don't compare with self
                        other_idx = free_indices[k]
                        other_pos = agent_pos[other_idx]
                        other_dx = cand[0] - other_pos[0]
                        other_dy = cand[1] - other_pos[1]
                        other_dz = cand[2] - other_pos[2]
                        other_dist = np.sqrt(other_dx**2 + other_dy**2 + (other_dz * w_z)**2)
                        if other_dist < min_agent_dist:
                            min_agent_dist = other_dist
                
                # Apply penalty if candidate is too close to other agents (< 50m)
                if min_agent_dist < 50.0:
                    dispersion_cost = w_dispersion * (50.0 - min_agent_dist) / 50.0
            
            # 6. Layer preference penalty
            layer_penalty = 0.0
            if layer_penalty_weight > 0.0:
                pref_z = preferred_layer_z[a_idx]
                layer_penalty = abs(cand[2] - pref_z) * layer_penalty_weight

            # 7. LKP Scent Cost (pre-computed)
            costs[i, j] = dist + turn_cost + climb_cost + obs_cost + dispersion_cost + layer_penalty + cand_scent_costs[j]
            
    return costs


@njit(cache=True)
def unique_points_kernel(points: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """
    Remove duplicate points within tolerance (Numba optimized).
    """
    n = points.shape[0]
    if n == 0:
        return points
    
    keep = np.ones(n, dtype=np.bool_)
    tol_sq = tolerance * tolerance
    
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < tol_sq:
                keep[j] = False
    
    count = 0
    for i in range(n):
        if keep[i]:
            count += 1
    
    result = np.zeros((count, 3), dtype=np.float64)
    idx = 0
    for i in range(n):
        if keep[i]:
            result[idx] = points[i]
            idx += 1
    
    return result


# =============================================================================
# Main Class
# =============================================================================

@dataclass
class FrontierAllocator:
    fps_count: int
    cosine_margin: float
    vertical_weight: float
    lock_steps: int
    angle_min_deg: float
    sweep_trigger_m: float
    sweep_radius_m: float
    sweep_max_steps: int
    sweep_min_distance_m: float = 40.0
    confidence_threshold: float = 0.5  # Voxels with confidence < threshold are frontiers
    min_frontier_distance_m: float = 30.0
    use_unvisited_only: bool = True
    lkp_nearest_count: int = 20  # Number of "intel points" near LKP
    scent_decay_steps: int = 300  # Steps until scent attraction decays to zero
    allowed_mask: Optional[NDArray[np.bool_]] = field(default=None)
    astar: Optional["AStar3D"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._locks: Dict[int, Dict[str, object]] = {}
        self._sweeps: Dict[int, Dict[str, object]] = {}
        self._agent_state: Dict[int, str] = {}
        self._debug_enabled = False
        
        angle = max(0.0, min(180.0, self.angle_min_deg))
        self._angle_cos_limit = float(np.cos(np.deg2rad(angle)))
        
        # Convert mask to numpy for Numba if present
        if self.allowed_mask is not None:
            self.allowed_mask = self.allowed_mask.astype(np.int8) # Use int8 for Numba bool
        else:
            self.allowed_mask = np.array([], dtype=np.int8) # Empty indicator
            
        self._last_candidates = np.zeros((0, 3), dtype=np.float64)
        self._per_agent_candidates: dict[int, np.ndarray] = {}
        self._occupancy_grid = None
        self._preferred_layer_z: Optional[np.ndarray] = None
        self._layer_penalty_weight: float = 0.0
        self._layer_z_bounds: Optional[np.ndarray] = None
        self._agent_layer_idx: Optional[np.ndarray] = None
        self._override_frontiers: Optional[np.ndarray] = None
        
        # Track agents that just switched from Tracking to Searching
        self._just_lost_target: Dict[int, bool] = {}

    def set_astar(self, astar: "AStar3D") -> None:
        self.astar = astar

    def set_occupancy_grid(self, occupancy: NDArray[np.int8]) -> None:
        self._occupancy_grid = occupancy

    def set_layer_preferences(self, preferred_layer_z: Optional[np.ndarray], penalty_weight: float) -> None:
        self._preferred_layer_z = preferred_layer_z
        self._layer_penalty_weight = float(penalty_weight)

    def set_layer_constraints(self, layer_z_bounds: Optional[np.ndarray], agent_layer_idx: Optional[np.ndarray]) -> None:
        self._layer_z_bounds = layer_z_bounds
        self._agent_layer_idx = agent_layer_idx

    def set_layer_frontiers(self, frontiers: Optional[np.ndarray]) -> None:
        self._override_frontiers = frontiers

    def _get_agent_layer_bounds(self, agent_idx: int) -> Optional[np.ndarray]:
        if self._layer_z_bounds is None or self._agent_layer_idx is None:
            return None
        if self._agent_layer_idx.size == 0 or self._layer_z_bounds.size == 0:
            return None
        layer_idx = int(self._agent_layer_idx[agent_idx])
        if layer_idx < 0 or layer_idx >= self._layer_z_bounds.shape[0]:
            return None
        return self._layer_z_bounds[layer_idx]

    def _check_frontier_still_valid(self, point: np.ndarray, voxel_map: VoxelMap3D) -> bool:
        """Check whether a frontier point still needs exploration.

        A frontier becomes invalid once the neighborhood around it has high
        enough confidence. A small 1-2 voxel radius is used for the check.

        Args:
            point: Frontier point in world coordinates.
            voxel_map: Voxel map containing confidence values.

        Returns:
            True if the frontier is still worth exploring, else False.
        """
      
        local = point - voxel_map.origin
        idx = np.floor(local / voxel_map.voxel_size).astype(int)
        
      
        shape = voxel_map.shape
        if (idx[0] < 0 or idx[0] >= shape[0] or
            idx[1] < 0 or idx[1] >= shape[1] or
            idx[2] < 0 or idx[2] >= shape[2]):
            return False  # Out of bounds, treat as invalid.
        
      
        check_radius = 1  # Radius in voxels.
        low_confidence_count = 0
        total_checked = 0
        
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                for dz in range(-check_radius, check_radius + 1):
                    nx, ny, nz = idx[0] + dx, idx[1] + dy, idx[2] + dz
                    if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                      
                        if self._occupancy_grid is not None and self._occupancy_grid[nx, ny, nz] == 1:
                            continue
                        total_checked += 1
                      
                        if voxel_map.visits[nx, ny, nz] < self.confidence_threshold:
                            low_confidence_count += 1
        
      
        if total_checked == 0:
            return False
        return low_confidence_count > total_checked * 0.3  # Require at least 30% low-confidence voxels.


    def detect_frontiers(self, voxel_map: VoxelMap3D, max_candidates: int = 400) -> NDArray[np.float64]:
        """Detect frontiers using Numba."""
        if self._occupancy_grid is None:
            # Fallback (slow path)
            occ = voxel_map.occupancy_grid()
        else:
            occ = self._occupancy_grid
            
        # Limits
        voxel_size = voxel_map.voxel_size
        max_z_idx = int(95.0 / voxel_size)
        
        return detect_frontiers_kernel(
            voxel_map.visits,
            occ,
            self.allowed_mask,
            voxel_map.origin,
            voxel_size,
            max_z_idx,
            max_candidates,
            self.confidence_threshold,
            self.use_unvisited_only
        )

    def mark_agent_lost_target(self, agent_idx: int, just_lost: bool = True) -> None:
        """
        Mark an agent as having just lost its target (Tracking -> Searching transition).
        This agent will get priority for LKP-nearby candidates in Hungarian assignment.
        """
        self._just_lost_target[agent_idx] = just_lost

    def allocate(
        self,
        agent_positions: NDArray[np.float64],
        agent_headings: NDArray[np.float64],
        voxel_map: VoxelMap3D,
        current_step: int,
        lkp_pos: Optional[NDArray[np.float64]] = None,
        target_lost_steps: int = 0,
    ) -> Dict[int, Optional[NDArray[np.float64]]]:
        """
        Allocate frontier targets to agents with "hunting instinct".
        
        Args:
            agent_positions: (N, 3) array of agent positions
            agent_headings: (N, 3) array of agent heading vectors
            voxel_map: VoxelMap3D instance
            current_step: Current simulation step
            lkp_pos: Last Known Position of target (3D). If None, pure exploration mode.
            target_lost_steps: Steps since target was lost. Used to decay scent attraction.
        
        Returns:
            Dict mapping agent_idx -> target position (or None)
        """
        num_agents = agent_positions.shape[0]
        assignments: Dict[int, Optional[NDArray[np.float64]]] = {idx: None for idx in range(num_agents)}

        self._expire_locks(current_step)

        # 1. Handle Sweeps (Fast Python logic)
        for idx in list(self._sweeps.keys()):
            if idx >= num_agents: del self._sweeps[idx]; continue
            layer_bounds = self._get_agent_layer_bounds(idx)
            if layer_bounds is not None:
                z_min = layer_bounds[0]
                z_max = layer_bounds[1]
                pos = agent_positions[idx]
                if pos[2] < z_min or pos[2] > z_max:
                    del self._sweeps[idx]
                    continue
            
            target = self._next_sweep_target(idx, agent_positions[idx], agent_headings[idx], voxel_map)
            if target is None:
                del self._sweeps[idx]
            else:
                assignments[idx] = target

      
        for idx in list(self._locks.keys()):
            if idx >= num_agents: del self._locks[idx]; continue
            if idx in self._sweeps: continue

            lock = self._locks[idx]
            locked_point = np.array(lock["point"])
            layer_bounds = self._get_agent_layer_bounds(idx)
            if layer_bounds is not None:
                z_min = layer_bounds[0]
                z_max = layer_bounds[1]
                if locked_point[2] < z_min or locked_point[2] > z_max:
                    del self._locks[idx]
                    continue
            dist = np.linalg.norm(locked_point - agent_positions[idx])
            
          
            lock_still_valid = self._check_frontier_still_valid(locked_point, voxel_map)
            
            if not lock_still_valid:
              
                if self._debug_enabled:
                    print(f"[Lock] Agent {idx}: releasing an explored frontier lock")
                del self._locks[idx]
                continue

            if dist <= self.sweep_trigger_m:
              
                del self._locks[idx]
                target = self._nearest_unknown_for_sweep(agent_positions[idx], voxel_map, heading=agent_headings[idx], agent_idx=idx)
                if target is not None:
                    self._sweeps[idx] = {
                        "steps": 1,
                        "start_pos": agent_positions[idx].copy(),
                        "start_step": current_step
                    }
                    assignments[idx] = target
                    if self._debug_enabled:
                        print(f"[Sweep] Agent {idx}: entering sweep mode from {agent_positions[idx][:2]}")
            else:
                assignments[idx] = locked_point

        # 3. Identify Free Agents
        free_agents = [idx for idx in range(num_agents) if assignments[idx] is None]
        if not free_agents:
            return assignments

        # 4. Detect Frontiers (Numba Optimized)
        if self._override_frontiers is not None:
            frontiers = self._override_frontiers
        else:
            frontiers = self.detect_frontiers(voxel_map)
        if frontiers.shape[0] == 0:
            return assignments

        # 5. Per-Agent FPS Selection, then merge
        all_candidates_list = []
        
      
        for idx in free_agents:
            cos_threshold = max(0.5, self.cosine_margin)
            cands = fps_select_kernel(frontiers, agent_positions[idx], self.fps_count, cos_threshold)
            if cands.shape[0] > 0:
                all_candidates_list.append(cands)
            self._per_agent_candidates[idx] = cands
        
        # 5b. LKP-nearest selection (intel points) - only if LKP is valid
        if lkp_pos is not None and target_lost_steps < self.scent_decay_steps:
            lkp_arr = np.asarray(lkp_pos, dtype=np.float64)
            if lkp_arr.shape[0] >= 3:
                intel_points = select_nearest_to_lkp_kernel(
                    frontiers, lkp_arr, self.lkp_nearest_count
                )
                if intel_points.shape[0] > 0:
                    all_candidates_list.append(intel_points)
            
        if not all_candidates_list:
            return assignments
            
      
        candidates = np.vstack(all_candidates_list)
        candidates = unique_points_kernel(candidates, tolerance=1.0)
        
      
        if self.min_frontier_distance_m > 0.0 and candidates.shape[0] > 0:
            dists = np.linalg.norm(candidates[:, None, :] - agent_positions[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            candidates = candidates[min_dists >= self.min_frontier_distance_m]
            if candidates.shape[0] == 0:
                return assignments
        
        self._last_candidates = candidates

        # 6. Cost Matrix with LKP Scent (Numba Optimized)
        if self._occupancy_grid is None:
            self._occupancy_grid = voxel_map.occupancy_grid().astype(np.int8)

        # Calculate dynamic scent weight
        if lkp_pos is not None and target_lost_steps < self.scent_decay_steps:
            w_scent = max(0.0, 2.0 * (1.0 - target_lost_steps / self.scent_decay_steps))
            lkp_arr = np.asarray(lkp_pos, dtype=np.float64)
        else:
            w_scent = 0.0
            lkp_arr = np.zeros(3, dtype=np.float64)

        if self._preferred_layer_z is None or self._preferred_layer_z.shape[0] != agent_positions.shape[0]:
            preferred_layer_z = np.zeros(agent_positions.shape[0], dtype=np.float64)
            layer_penalty_weight = 0.0
        else:
            preferred_layer_z = self._preferred_layer_z
            layer_penalty_weight = self._layer_penalty_weight
        if self._layer_z_bounds is None or self._agent_layer_idx is None:
            layer_z_bounds = np.zeros((0, 2), dtype=np.float64)
            agent_layer_idx = np.zeros((0,), dtype=np.int64)
        else:
            layer_z_bounds = self._layer_z_bounds.astype(np.float64)
            agent_layer_idx = self._agent_layer_idx.astype(np.int64)

        costs = build_cost_matrix_kernel(
            agent_positions, agent_headings, candidates, 
            np.array(free_agents), 
            self._occupancy_grid,
            voxel_map.origin, voxel_map.voxel_size,
            self.vertical_weight, 0.3, 2.0, 50.0,
            self._angle_cos_limit,
            preferred_layer_z,
            layer_penalty_weight,
            layer_z_bounds,
            agent_layer_idx,
            lkp_arr, w_scent, 50.0
        )

        # 7. Role Protection: Agents that just lost target get priority for LKP-nearby candidates
        if lkp_pos is not None and w_scent > 0.0:
            self._apply_role_protection(costs, candidates, free_agents, lkp_arr)

        # 8. Hungarian Assignment (Scipy - Fast enough for small matrix)
        row_ind, col_ind = linear_sum_assignment(costs)

      
        initial_assignments = {}
        for r, c in zip(row_ind, col_ind):
            if costs[r, c] < 1e9: # check inf
                actual_idx = free_agents[int(r)]
                target = candidates[int(c)]
                initial_assignments[actual_idx] = (target, int(c), int(r))
        
      
        MIN_TARGET_SEPARATION = 30.0  # Minimum distance between assigned targets in meters.
        assigned_targets = {}  # agent_idx -> target
        used_candidate_indices = set()
        
      
        for actual_idx in sorted(initial_assignments.keys()):
            target, cand_idx, agent_row = initial_assignments[actual_idx]
            
          
            too_close = cand_idx in used_candidate_indices
            if not too_close:
                for other_idx, other_target in assigned_targets.items():
                    dist = np.linalg.norm(target - other_target)
                    if dist < MIN_TARGET_SEPARATION:
                        too_close = True
                        break
            
            if not too_close:
              
                assigned_targets[actual_idx] = target
                used_candidate_indices.add(cand_idx)
                assignments[actual_idx] = target
                self._locks[actual_idx] = {"point": target, "expiry": current_step + self.lock_steps}
            else:
              
                best_alt_idx = -1
                best_alt_cost = 1e9
                
                for alt_c in range(len(candidates)):
                    if alt_c in used_candidate_indices:
                        continue
                    
                    alt_target = candidates[alt_c]
                    
                  
                    far_enough = True
                    for other_target in assigned_targets.values():
                        if np.linalg.norm(alt_target - other_target) < MIN_TARGET_SEPARATION:
                            far_enough = False
                            break
                    
                    if far_enough and costs[agent_row, alt_c] < best_alt_cost:
                        best_alt_cost = costs[agent_row, alt_c]
                        best_alt_idx = alt_c
                
                if best_alt_idx >= 0:
                    alt_target = candidates[best_alt_idx]
                    assigned_targets[actual_idx] = alt_target
                    used_candidate_indices.add(best_alt_idx)
                    assignments[actual_idx] = alt_target
                    self._locks[actual_idx] = {"point": alt_target, "expiry": current_step + self.lock_steps}
                else:
                  
                    if cand_idx not in used_candidate_indices:
                        assigned_targets[actual_idx] = target
                        used_candidate_indices.add(cand_idx)
                        assignments[actual_idx] = target
                        self._locks[actual_idx] = {"point": target, "expiry": current_step + self.lock_steps}
                  

        # Clear just_lost_target flags after allocation
        self._just_lost_target.clear()

        return assignments


    def _apply_role_protection(
        self,
        costs: np.ndarray,
        candidates: np.ndarray,
        free_agents: List[int],
        lkp_pos: np.ndarray
    ) -> None:
        """
        Apply negative bias to cost matrix rows for agents that just lost target.
        This ensures they stay near LKP instead of being reassigned far away by Hungarian.
        
        Strategy:
        - Find candidates closest to LKP
        - Apply negative bias to those columns for agents with just_lost_target flag
        """
        # Find indices of candidates closest to LKP
        dists_to_lkp = np.linalg.norm(candidates - lkp_pos, axis=1)
        
        # Get top 5 closest candidates to LKP
        n_priority = min(5, len(dists_to_lkp))
        priority_cols = np.argsort(dists_to_lkp)[:n_priority]
        
        # Apply negative bias for agents that just lost target
        bias = -100.0  # Strong negative bias to ensure priority
        
        for row_idx, agent_idx in enumerate(free_agents):
            if self._just_lost_target.get(agent_idx, False):
                for col in priority_cols:
                    costs[row_idx, col] += bias
                if self._debug_enabled:
                    print(f"[RoleProtection] Agent {agent_idx}: applied bias to {n_priority} LKP-nearby candidates")

    def last_candidates(self) -> np.ndarray:
        """Return last merged FPS candidate set."""
        return self._last_candidates

    def last_per_agent_candidates(self) -> dict[int, np.ndarray]:
        """Return last per-agent FPS candidates (dict: agent_idx -> np.ndarray)."""
        return self._per_agent_candidates

    def _expire_locks(self, step: int) -> None:
        for idx in list(self._locks.keys()):
            if step >= int(self._locks[idx]["expiry"]):
                del self._locks[idx]


    def _next_sweep_target(self, idx: int, pos: np.ndarray, heading: np.ndarray, voxel_map: VoxelMap3D) -> Optional[np.ndarray]:
        """Get the next target while an agent is in sweep mode.

        Sweep mode searches locally for low-confidence voxels, prefers points in
        front of the current heading, and stops once the step budget is exhausted.
        """
        sweep_info = self._sweeps.get(idx)
        if sweep_info is None:
            return None
        
      
        current_steps = sweep_info.get("steps", 0)
        if current_steps >= self.sweep_max_steps:
            if self._debug_enabled:
                print(f"[Sweep] Agent {idx}: reached the sweep step limit {self.sweep_max_steps}")
            return None
        
      
        target = self._find_sweep_target(pos, heading, voxel_map, idx)
        
        if target is None:
            if self._debug_enabled:
                print(f"[Sweep] Agent {idx}: local area fully explored, leaving sweep mode")
            return None
        
      
        sweep_info["steps"] = current_steps + 1
        
        if self._debug_enabled:
            dist = np.linalg.norm(target - pos)
            print(f"[Sweep] Agent {idx}: step={current_steps + 1}/{self.sweep_max_steps}, "
                  f"target_dist={dist:.1f}m")
        
        return target

    def _find_sweep_target(self, pos: np.ndarray, heading: np.ndarray, voxel_map: VoxelMap3D, agent_idx: int | None = None) -> Optional[np.ndarray]:
        """Search the local sweep area for a valid low-confidence target."""
        radius = self.sweep_radius_m
        min_dist = self.sweep_min_distance_m
        vsize = voxel_map.voxel_size
        local = pos - voxel_map.origin
        
      
        min_corner = np.floor((local - radius) / vsize).astype(int)
        max_corner = np.ceil((local + radius) / vsize).astype(int)
        max_idx = np.array(voxel_map.shape, dtype=int) - 1
        min_corner = np.clip(min_corner, 0, max_idx)
        max_corner = np.clip(max_corner, 0, max_idx)
        
      
        sub_visits = voxel_map.visits[
            min_corner[0]:max_corner[0] + 1,
            min_corner[1]:max_corner[1] + 1,
            min_corner[2]:max_corner[2] + 1
        ]
        
      
        indices = np.argwhere(sub_visits < self.confidence_threshold)
        if indices.size == 0:
            return None
        
      
        indices += min_corner
        candidates = (indices + 0.5) * vsize + voxel_map.origin
        
      
        dists = np.linalg.norm(candidates - pos, axis=1)
        
      
        valid_radius = dists <= radius
        
      
        valid_min_dist = dists >= min_dist
        
      
        valid = valid_radius & valid_min_dist
        
        if not np.any(valid):
          
            valid = valid_radius
            if not np.any(valid):
                return None
        
        candidates = candidates[valid]
        dists = dists[valid]
        layer_bounds = self._get_agent_layer_bounds(agent_idx) if agent_idx is not None else None
        if layer_bounds is not None and candidates.shape[0] > 0:
            z_min = layer_bounds[0]
            z_max = layer_bounds[1]
            layer_mask = (candidates[:, 2] >= z_min) & (candidates[:, 2] <= z_max)
            if not np.any(layer_mask):
                return None
            candidates = candidates[layer_mask]
            dists = dists[layer_mask]
        
      
        if heading is not None:
            norm_h = np.linalg.norm(heading[:2])  # Use horizontal heading for the forward preference.
            if norm_h > 1e-6:
                h_dir = heading / (np.linalg.norm(heading) + 1e-6)
                vecs = candidates - pos
                vecs_norm = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs_norm = np.where(vecs_norm > 1e-6, vecs_norm, 1.0)
                vecs_unit = vecs / vecs_norm
                
              
                dots = np.sum(vecs_unit * h_dir, axis=1)
                
              
                forward_mask = dots > 0
                if np.any(forward_mask):
                    candidates = candidates[forward_mask]
                    dists = dists[forward_mask]
                    dots = dots[forward_mask]
                    
                  
                    return candidates[np.argmin(dists)]
        
      
        return candidates[np.argmin(dists)]


    def _nearest_unknown_for_sweep(self, pos: np.ndarray, voxel_map: VoxelMap3D, heading: np.ndarray = None, agent_idx: int | None = None) -> Optional[np.ndarray]:
        """Find the first sweep target when entering sweep mode.

        This is similar to `_find_sweep_target` but skips the minimum-distance
        filter because the sweep phase has just started.
        """
        radius = self.sweep_radius_m
        vsize = voxel_map.voxel_size
        local = pos - voxel_map.origin
        
        min_corner = np.floor((local - radius) / vsize).astype(int)
        max_corner = np.ceil((local + radius) / vsize).astype(int)
        max_idx = np.array(voxel_map.shape, dtype=int) - 1
        min_corner = np.clip(min_corner, 0, max_idx)
        max_corner = np.clip(max_corner, 0, max_idx)
        
        sub_visits = voxel_map.visits[
            min_corner[0]:max_corner[0] + 1,
            min_corner[1]:max_corner[1] + 1,
            min_corner[2]:max_corner[2] + 1
        ]
        
        indices = np.argwhere(sub_visits < self.confidence_threshold)
        if indices.size == 0:
            return None
        
        indices += min_corner
        candidates = (indices + 0.5) * vsize + voxel_map.origin
        
        dists = np.linalg.norm(candidates - pos, axis=1)
        valid = dists <= radius
        if not np.any(valid):
            return None
        
        candidates = candidates[valid]
        dists = dists[valid]
        layer_bounds = self._get_agent_layer_bounds(agent_idx) if agent_idx is not None else None
        if layer_bounds is not None and candidates.shape[0] > 0:
            z_min = layer_bounds[0]
            z_max = layer_bounds[1]
            layer_mask = (candidates[:, 2] >= z_min) & (candidates[:, 2] <= z_max)
            if not np.any(layer_mask):
                return None
            candidates = candidates[layer_mask]
            dists = dists[layer_mask]
        
      
        if heading is not None:
            norm_h = np.linalg.norm(heading)
            if norm_h > 1e-6:
                h_dir = heading / norm_h
                vecs = candidates - pos
                dots = np.sum(vecs * h_dir, axis=1)
                forward_mask = dots > 0
                if np.any(forward_mask):
                    candidates = candidates[forward_mask]
                    dists = dists[forward_mask]
        
        return candidates[np.argmin(dists)]
