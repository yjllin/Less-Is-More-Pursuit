"""Force AirSim drones to spawn at separated corners and hover in place.

This script exists because settings.json spawns every vehicle at (0, 0, -2).
It overrides the spawn pose in Python so that four drones start at:
(-20, -20, -6), (20, -20, -6), (20, 20, -6), (-20, 20, -6).
"""

from __future__ import annotations
import argparse
import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import json
import time
import importlib.util
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation

from src.config import load_config
from src.navigation import AStar3D, FrontierAllocator, VoxelMap3D

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[override]
        def wrapper(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return wrapper

try:
    import airsim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    airsim = None


SPAWN_POINTS = [
    np.array([-20.0, -20.0, -6.0], dtype=np.float64),
    np.array([20.0, -20.0, -6.0], dtype=np.float64),
    np.array([20.0, 20.0, -6.0], dtype=np.float64),
    np.array([-20.0, 20.0, -6.0], dtype=np.float64),
]
TOLERANCE_M = 0.35


def _to_world_from_pose(pose) -> np.ndarray:
    """Convert AirSim pose (NED) to UE/world coords expected by voxel map (flip Z)."""
    return np.array([pose.position.x_val, pose.position.y_val, -pose.position.z_val], dtype=np.float64)


def _to_airsim_vec(world_point: np.ndarray) -> "airsim.Vector3r":
    """Convert UE/world coords (Z up) to AirSim NED vector (Z down)."""
    return airsim.Vector3r(float(world_point[0]), float(world_point[1]), float(-world_point[2]))


def _heading_from_quaternion(q) -> np.ndarray:
    """Extract planar forward heading from AirSim quaternion (Z-up yaw)."""
    w = float(q.w_val)
    x = float(q.x_val)
    y = float(q.y_val)
    z = float(q.z_val)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)


def _settings_vehicle_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        data = path.read_text(encoding="utf-8")
        vehicles = json.loads(data).get("Vehicles") or {}
    except Exception:
        return []
    if not isinstance(vehicles, dict):
        return []
    return list(vehicles.keys())


def _resolve_vehicle_names(client, expected: int) -> List[str]:
    names: List[str] = []
    try:
        names = client.listVehicles() or []
    except Exception:
        names = []
    if len(names) < expected:
        settings_names = _settings_vehicle_names(Path.home() / "Documents" / "AirSim" / "settings.json")
        names.extend([n for n in settings_names if n not in names])
    if len(names) < expected:
        names.extend([f"Drone{i}" for i in range(expected - len(names))])
    return names[:expected]


def _arm_and_hover(client, vehicle_name: str, ned_target: np.ndarray) -> None:
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    pose = airsim.Pose(
        airsim.Vector3r(float(ned_target[0]), float(ned_target[1]), float(ned_target[2])),
        airsim.to_quaternion(0.0, 0.0, 0.0),
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
    client.hoverAsync(vehicle_name=vehicle_name).join()
    client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=1.0, vehicle_name=vehicle_name).join()


def _start_follow_path(client, vehicle_name: str, world_path: list[np.ndarray], speed: float = 8.0):
    """Start asynchronous flight along an A* path and return the future."""
    if len(world_path) < 2:
        return None
    ned_path = [_to_airsim_vec(p) for p in world_path]
    return client.moveOnPathAsync(
        ned_path,
        velocity=float(speed),
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        lookahead=-1,
        adaptive_lookahead=1,
        vehicle_name=vehicle_name,
    )


def _tick_update_visibility(client, vehicle_names: list[str], voxel_map: VoxelMap3D, occupancy: np.ndarray, visited: np.ndarray, view_radius: float) -> list[np.ndarray]:
    """Poll current drone positions, update visited voxels, and return latest world positions."""
    positions = []
    for name in vehicle_names:
        pose = client.simGetVehiclePose(vehicle_name=name)
        positions.append(_to_world_from_pose(pose))
    _mark_visible_raycast(voxel_map, occupancy, visited, positions, view_radius)
    return positions


def _load_level_map(cfg) -> tuple[VoxelMap3D, np.ndarray]:
    meta_path = Path("artifacts") / "level_occupancy.json"
    grid_path = Path("artifacts") / "level_occupancy.npy"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        origin = tuple(meta.get("origin", (0.0, 0.0, 0.0)))
        world_size = tuple(meta.get("world_size", tuple(cfg.environment.world_size_m)))
        voxel_size = float(meta.get("voxel_size", cfg.environment.voxel_size_m))
    else:
        origin = (0.0, 0.0, 0.0)
        world_size = tuple(cfg.environment.world_size_m)
        voxel_size = float(cfg.environment.voxel_size_m)
    voxel_map = VoxelMap3D(world_size, voxel_size, cfg.navigation.occupancy_threshold, origin=origin)
    if not grid_path.exists():
        raise FileNotFoundError(f"Occupancy grid not found at {grid_path}")
    occupancy = np.load(grid_path)
    if occupancy.shape != voxel_map.shape:
        raise ValueError(f"Occupancy grid shape {occupancy.shape} does not match voxel map shape {voxel_map.shape}")
  
  
    voxel_map.visits = np.zeros(voxel_map.shape, dtype=np.float32)
    voxel_map.visits[occupancy == 1] = 1.0
    voxel_map.hits[occupancy == 1] = 1
    return voxel_map, occupancy


@njit(cache=True)
def _raycast_visible(start_idx: np.ndarray, end_idx: np.ndarray, occupancy: np.ndarray) -> bool:
    """
    Bresenham-like integer ray march between start and end voxels.
    Returns False if any occupied voxel (value==1) blocks the path.
    """
    x0, y0, z0 = int(start_idx[0]), int(start_idx[1]), int(start_idx[2])
    x1, y1, z1 = int(end_idx[0]), int(end_idx[1]), int(end_idx[2])

    if x0 == x1 and y0 == y1 and z0 == z1:
        return occupancy[x0, y0, z0] != 1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1 if x1 < x0 else 0
    sy = 1 if y1 > y0 else -1 if y1 < y0 else 0
    sz = 1 if z1 > z0 else -1 if z1 < z0 else 0

    if dx >= dy and dx >= dz:
        err_y = dx / 2.0
        err_z = dx / 2.0
        while x0 != x1:
            x0 += sx
            err_y -= dy
            err_z -= dz
            if err_y < 0:
                y0 += sy
                err_y += dx
            if err_z < 0:
                z0 += sz
                err_z += dx
            if occupancy[x0, y0, z0] == 1:
                return False
    elif dy >= dx and dy >= dz:
        err_x = dy / 2.0
        err_z = dy / 2.0
        while y0 != y1:
            y0 += sy
            err_x -= dx
            err_z -= dz
            if err_x < 0:
                x0 += sx
                err_x += dy
            if err_z < 0:
                z0 += sz
                err_z += dy
            if occupancy[x0, y0, z0] == 1:
                return False
    else:
        err_x = dz / 2.0
        err_y = dz / 2.0
        while z0 != z1:
            z0 += sz
            err_x -= dx
            err_y -= dy
            if err_x < 0:
                x0 += sx
                err_x += dz
            if err_y < 0:
                y0 += sy
                err_y += dz
            if occupancy[x0, y0, z0] == 1:
                return False
    return True


@njit(cache=True)
def _raycast_check(start_idx: np.ndarray, targets: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    visible = np.zeros(targets.shape[0], dtype=np.bool_)
    for i in range(targets.shape[0]):
        visible[i] = _raycast_visible(start_idx, targets[i], occupancy)
    return visible


def raycast_check(start_idx: np.ndarray, targets: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """
    Vectorized wrapper to run visibility checks for many candidate voxels.
    Falls back to pure Python when numba is unavailable.
    """
    if targets.size == 0:
        return np.zeros(0, dtype=bool)
    start_idx_i = start_idx.astype(np.int64)
    targets_i = targets.astype(np.int64)
    if NUMBA_AVAILABLE:
        return _raycast_check(start_idx_i, targets_i, occupancy)
    if hasattr(_raycast_check, "py_func"):  # type: ignore[attr-defined]
        return _raycast_check.py_func(start_idx_i, targets_i, occupancy)  # pragma: no cover - numba fallback
    return _raycast_check(start_idx_i, targets_i, occupancy)

def simplify_path_by_raycast(path: list[np.ndarray], occupancy: np.ndarray, voxel_map: VoxelMap3D) -> list[np.ndarray]:
    """
    Greedy Raycast Shortcutting:
    If point A can see point C directly (no obstacles), remove intermediate point B.
    """
    if len(path) < 3:
        return path
        
    path_indices = np.array([voxel_map._indices_from_points(np.array([p]))[0] for p in path])
    smoothed_path = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Check visibility from current point to all subsequent points
        check_indices = path_indices[current_idx+1:]
        
        # Use existing vectorized raycast_check
        visible_mask = raycast_check(path_indices[current_idx], check_indices, occupancy)
        
        # Find the furthest visible point index
        true_indices = np.where(visible_mask)[0]
        
        if true_indices.size > 0:
            # Jump to the furthest visible point
            next_hop_idx = current_idx + 1 + true_indices[-1]
        else:
            next_hop_idx = current_idx + 1
            
        smoothed_path.append(path[next_hop_idx])
        current_idx = next_hop_idx
        
    return smoothed_path


def smooth_and_filter_path(
    current_pos: np.ndarray,
    current_velocity: np.ndarray,
    world_path: list[np.ndarray],
    occupancy: np.ndarray,
    voxel_map: VoxelMap3D,
) -> list[np.ndarray]:
    """Combine raycast smoothing with forward filtering based on current velocity."""
    if not world_path:
        return []
    simplified = simplify_path_by_raycast(world_path, occupancy, voxel_map)
    filtered = _filter_forward_path(current_pos, current_velocity, simplified)
    return filtered


def _detect_target(agent_position: np.ndarray) -> np.ndarray | None:
    """
    Placeholder target detection; return a target world position when detected, else None.
    Integrate with real perception as needed.
    """
    return None


def _filter_forward_path(current_pos: np.ndarray, current_velocity: np.ndarray, world_path: list[np.ndarray]) -> list[np.ndarray]:
    """
    Keep waypoints that lie ahead of the drone along its velocity direction; always keep the final goal.
    """
    if len(world_path) <= 1:
        return world_path
    speed = np.linalg.norm(current_velocity)
    if speed < 1e-6:
        return world_path
    vel_norm = current_velocity / speed
    filtered: list[np.ndarray] = []
    for i, wp in enumerate(world_path):
        if i == len(world_path) - 1:
            filtered.append(wp)
            continue
        direction = wp - current_pos
        if np.dot(direction, vel_norm) >= 0:
            filtered.append(wp)
    if not filtered:
        return [world_path[-1]]
    return filtered



def _mark_visible_raycast(voxel_map: VoxelMap3D, occupancy: np.ndarray, visited: np.ndarray, positions: list[np.ndarray], radius: float) -> None:
    """
    Mark voxels that are both within view radius and unobstructed by occupied cells.
    Uses float32 confidence (0.0 = unexplored, 1.0 = just explored).
    """
    if radius <= 0.0 or not positions:
        return
    radius = float(radius)
    radius_sq = radius * radius
    max_idx = np.array(voxel_map.shape, dtype=int) - 1
    for pos in positions:
        local = pos - voxel_map.origin
        drone_idx = np.floor(local / voxel_map.voxel_size).astype(int)
        drone_idx = np.clip(drone_idx, 0, max_idx)

        min_corner = np.floor((local - radius) / voxel_map.voxel_size).astype(int)
        max_corner = np.floor((local + radius) / voxel_map.voxel_size).astype(int)
        min_corner = np.clip(min_corner, 0, max_idx)
        max_corner = np.clip(max_corner, 0, max_idx)

        xs = np.arange(min_corner[0], max_corner[0] + 1, dtype=int)
        ys = np.arange(min_corner[1], max_corner[1] + 1, dtype=int)
        zs = np.arange(min_corner[2], max_corner[2] + 1, dtype=int)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        centers = (grid.astype(np.float64) + 0.5) * voxel_map.voxel_size + voxel_map.origin
        dist_sq = np.sum((centers - pos) ** 2, axis=1)
        candidates = grid[dist_sq <= radius_sq]
        if candidates.size == 0:
            continue

        visible_mask = raycast_check(drone_idx, candidates, occupancy)
        if not visible_mask.any():
            continue
        visible_cells = candidates[visible_mask]
        for idx in visible_cells:
            cell = tuple(int(i) for i in idx)
            visited[cell] = 1  # 
            if occupancy[cell]:
                continue
          
            voxel_map.visits[cell] = 1.0


def _render_layers(client, voxel_map: VoxelMap3D, occupancy: np.ndarray, visited: np.ndarray, duration: float = 30.0) -> None:
    # Visited voxels (seen by drones) that are free space
    visited_idx = np.argwhere(np.logical_and(visited == 1, occupancy == 0))
    if visited_idx.size > 0:
        visit_points = []
        for idx in visited_idx:
            world = voxel_map.world_from_index(tuple(int(i) for i in idx))
            visit_points.append(_to_airsim_vec(world))
        seen_yellow = [1.0, 0.9, 0.2, 0.7]  # lighter yellow to distinguish from static occupancy
        client.simPlotPoints(visit_points, color_rgba=seen_yellow, size=8.0, duration=duration)


def _compute_global_exploration_ratio(allowed_mask: np.ndarray, visited: np.ndarray) -> float:
    """ / """
    total_allowed = int(np.sum(allowed_mask))
    if total_allowed == 0:
        return 1.0
    explored = int(np.sum(np.logical_and(visited > 0, allowed_mask)))
    return explored / total_allowed


def _load_policy_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "test_policy_airsim.py"
    if not module_path.exists():
        raise FileNotFoundError(f"policy test module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("test_policy_airsim", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _run_policy_exploration(
    cfg,
    policy_path: str,
    stage_name: str | None,
    steps: int,
    device: torch.device,
    benchmark_mode: bool,
) -> None:
    policy_mod = _load_policy_module()
    stage_cfg = policy_mod.get_stage_config(cfg, stage_name) if stage_name else None
    env = policy_mod.AirSimEnv(cfg, stage_config=stage_cfg, debug=not benchmark_mode, debug_frontier=False)
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    obs_dim = obs.shape[-1]
    policy = policy_mod.load_policy(policy_path, obs_dim, cfg, device)
    num_agents = obs.shape[0]
    hidden = policy.init_hidden(num_envs=num_agents).to(device)
    masks = torch.ones(num_agents, 1, device=device)

    voxel_map, occupancy = _load_level_map(cfg)
    visited = np.zeros_like(occupancy, dtype=np.int8)
    if stage_cfg is not None:
        view_radius = float(stage_cfg.view_radius_m)
    elif len(cfg.curriculum.stages) > 2:
        view_radius = float(cfg.curriculum.stages[2].view_radius_m)
    else:
        view_radius = float(cfg.curriculum.stages[0].view_radius_m)

    start_time = time.time()
    exploration_target = 0.99

    for step in range(steps):
        obs_tensor = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            actions, _, _, hidden = policy.act(obs_tensor, hidden, masks)
        actions_np = actions.cpu().numpy().astype(np.float64)
        obs, done, info = env.step(actions_np)
        obs = np.asarray(obs, dtype=np.float32)
        positions, _, _ = env._get_agent_states()
        _mark_visible_raycast(voxel_map, occupancy, visited, list(positions), view_radius)
        if step % 10 == 0:
            ratio = _compute_global_exploration_ratio(occupancy == 0, visited)
            elapsed = time.time() - start_time
            print(f"[policy-explore] step={step} ratio={ratio:.2%} elapsed={elapsed:.1f}s")
        if _compute_global_exploration_ratio(occupancy == 0, visited) >= exploration_target:
            print(f"[policy-explore] reached {exploration_target:.0%} at step {step}")
            break

    if not benchmark_mode:
        _render_layers(env.client, voxel_map, occupancy, visited)


@pytest.mark.skipif(airsim is None, reason="AirSim package not installed")
def test_force_spawn_positions_and_hover(
    benchmark_mode: bool = False,
    cfg_path: Path | str | None = None,
) -> None:
    """
    
    
    Args:
        benchmark_mode:  True99%
                        False
    """
    cfg = load_config(cfg_path) if cfg_path else load_config()
    client = airsim.MultirotorClient(ip=cfg.air_sim.ip, port=cfg.air_sim.port)
    client.confirmConnection()
    client.simPause(False)

    vehicle_names = _resolve_vehicle_names(client, len(SPAWN_POINTS))
    assert len(vehicle_names) >= len(SPAWN_POINTS), "Need at least four vehicles registered in AirSim."

    for name, target in zip(vehicle_names, SPAWN_POINTS):
        _arm_and_hover(client, name, target)
    time.sleep(0.5)

    positions: list[np.ndarray] = []
    for name, target in zip(vehicle_names, SPAWN_POINTS):
        pose = client.simGetVehiclePose(vehicle_name=name)
        position = _to_world_from_pose(pose)
        positions.append(position)
        expected_world = target.copy()
        expected_world[2] *= -1  # convert NED target Z to world-up for comparison
        assert np.allclose(position, expected_world, atol=TOLERANCE_M), f"{name} not at expected spawn {expected_world}, got {position}"

    voxel_map, occupancy = _load_level_map(cfg)
    view_radius = cfg.curriculum.stages[2].view_radius_m  # stage 3 (0-indexed) -> 16m
    visited = np.zeros_like(occupancy, dtype=np.int8)
    _mark_visible_raycast(voxel_map, occupancy, visited, positions, view_radius)
    
  
    if not benchmark_mode:
        _render_layers(client, voxel_map, occupancy, visited)

    # Frontier allocator and A* planner (single continuous loop)
  
    dilate_radius_m = 10.0
    dilate_radius_vox = int(np.ceil(dilate_radius_m / voxel_map.voxel_size))
    xs = np.arange(-dilate_radius_vox, dilate_radius_vox + 1)
    ys = np.arange(-dilate_radius_vox, dilate_radius_vox + 1)
    zs = np.arange(-dilate_radius_vox, dilate_radius_vox + 1)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    struct = np.linalg.norm(grid, axis=-1) * voxel_map.voxel_size <= dilate_radius_m
    dilated_occupancy = binary_dilation(occupancy, structure=struct)
    allowed_mask = np.logical_not(dilated_occupancy)
  
    astar = AStar3D(
        grid_shape=voxel_map.shape,
        voxel_size=voxel_map.voxel_size,
        cache_refresh_steps=cfg.navigation.cache_refresh_steps,
        lookahead_m=cfg.navigation.lookahead_distance_m,
        heuristic_weight=cfg.navigation.heuristic_weight,
        origin=voxel_map.origin,
    )
    static_grid = np.where(occupancy, 1, 0).astype(int)
    astar.update_grid(static_grid)
    
  
    confidence_threshold = getattr(cfg.environment, 'confidence_threshold', 0.5)
    
  
    allocator = FrontierAllocator(
        fps_count=cfg.navigation.frontier_fps_count,
        cosine_margin=cfg.navigation.frontier_cosine_margin,
        vertical_weight=cfg.navigation.frontier_vertical_weight,
        lock_steps=cfg.navigation.frontier_lock_steps,
        angle_min_deg=cfg.navigation.frontier_angle_min_deg,
        sweep_trigger_m=cfg.navigation.frontier_sweep_trigger_m,
        sweep_radius_m=cfg.navigation.frontier_sweep_radius_m,
        sweep_max_steps=cfg.navigation.frontier_sweep_max_steps,
        sweep_min_distance_m=getattr(cfg.navigation, 'frontier_sweep_min_distance_m', 40.0),
        confidence_threshold=confidence_threshold,
        allowed_mask=allowed_mask,
    )
    allocator.set_astar(astar)  #  A* 
    allocator.set_occupancy_grid(occupancy)  # 
    allocator._debug_enabled = True

    path_color = [0.0, 1.0, 1.0, 1.0]  # 
    red = [1.0, 0.0, 0.0, 1.0]
    pink = [1.0, 0.0, 1.0, 1.0]
    move_speed = 8.0
    max_steps = cfg.environment.max_steps  # 
    
  
    start_time = time.time()
    exploration_target = 0.99  # 99% 
    exploration_complete = False
    
    agent_positions = np.stack(positions)
    prev_positions = {i: agent_positions[i].copy() for i in range(len(agent_positions))}
    target_found = {i: False for i in range(len(agent_positions))}  # 0: , 1: 
    target_points: dict[int, np.ndarray | None] = {i: None for i in range(len(agent_positions))}
    need_replan = {i: True for i in range(len(agent_positions))}  # 0: , 1: 
    last_goal: dict[int, np.ndarray | None] = {i: None for i in range(len(agent_positions))}
    reach_threshold = 6.0
    
  
    confidence_decay = getattr(cfg.environment, 'confidence_decay', 0.999)
    
    for step in range(max_steps):
      
        free_mask = occupancy == 0
        voxel_map.visits[free_mask] *= confidence_decay
        
      
        for idx in range(len(agent_positions)):
            detected = _detect_target(agent_positions[idx])
            if detected is not None:
                target_found[idx] = True
                target_points[idx] = detected
                need_replan[idx] = True

      
        agent_headings = []
        for name in vehicle_names:
            pose = client.simGetVehiclePose(vehicle_name=name)
            agent_headings.append(_heading_from_quaternion(pose.orientation))
        agent_headings_arr = np.stack(agent_headings)

        print(f"\n=== Step {step} ===")
        print("")
        assignments = allocator.allocate(agent_positions, agent_headings_arr, voxel_map, current_step=step)
      
        candidates = allocator._last_candidates if isinstance(allocator._last_candidates, np.ndarray) else allocator.last_candidates()
        per_agent_candidates = allocator.last_per_agent_candidates()
        
      
        print(f" ( {len([a for a in assignments.values() if a is not None])} ):")
        for idx, point in assignments.items():
            if point is not None:
                dist = float(np.linalg.norm(point - agent_positions[idx]))
                print(f"  Agent {idx}:  = {dist:.1f}m,  = {agent_positions[idx][:2]},  = {point[:2]}")
            else:
                print(f"  Agent {idx}: ")
      
        if not benchmark_mode:
            # Visualize all candidates (pink)
            if candidates.size > 0:
                cand_points = [_to_airsim_vec(c) for c in candidates]
                client.simPlotPoints(cand_points, color_rgba=pink, size=9.0, duration=10.0)
            
            # Visualize per-agent candidates with agent-specific colors
            agent_colors = [
                [0.0, 0.5, 1.0, 0.8],  # Blue for agent 0
                [1.0, 0.5, 0.0, 0.8],  # Orange for agent 1
                [0.0, 1.0, 0.5, 0.8],  # Green for agent 2
                [0.8, 0.0, 0.8, 0.8],  # Purple for agent 3
            ]
            for agent_idx, agent_cands in per_agent_candidates.items():
                if agent_cands.size > 0:
                    color = agent_colors[agent_idx % len(agent_colors)]
                    cand_pts = [_to_airsim_vec(c) for c in agent_cands]
                    client.simPlotPoints(cand_pts, color_rgba=color, size=7.0, duration=10.0)

            assigned_points = []
            for idx, point in assignments.items():
                if point is None:
                    continue
                assigned_points.append(_to_airsim_vec(point))
            if assigned_points:
                client.simPlotPoints(assigned_points, color_rgba=red, size=12.0, duration=10.0)

        flight_futures = []
        traversed_positions: list[np.ndarray] = []
        endpoints: list[np.ndarray] = []
        
      
        CONTROL_LOOP_DURATION = 1 

        for idx, point in assignments.items():
            if point is None:
                continue

          
            goal = target_points[idx] if target_found[idx] and target_points[idx] is not None else point

          
            if not need_replan[idx] and last_goal[idx] is not None:
                dist_to_last = np.linalg.norm(agent_positions[idx] - last_goal[idx])
                if dist_to_last <= reach_threshold and goal is not None and not np.allclose(goal, last_goal[idx]):
                    need_replan[idx] = True

            if not need_replan[idx] or goal is None:
                continue
            need_replan[idx] = False

            current_pos = agent_positions[idx]
            current_velocity = np.zeros(3)
            if idx in prev_positions:
                # Estimate velocity over the last control loop duration
                current_velocity = (current_pos - prev_positions[idx]) / max(CONTROL_LOOP_DURATION, 1e-6)

          
            predicted_pos = current_pos + current_velocity * 0.2

          
            _, world_path = astar.compute_direction(predicted_pos, goal, current_step=step)
            if not world_path:
                # print(f"[astar] agent {idx} could not find a path")
                last_goal[idx] = None
                continue

          
            world_path = smooth_and_filter_path(current_pos, current_velocity, world_path, occupancy, voxel_map)

          
            last_goal[idx] = goal
            
          
          
          
            target_path = world_path 
            print(f"target_path:{target_path}")
          
            if not benchmark_mode:
                path_vis = [_to_airsim_vec(p) for p in target_path]
                client.simPlotPoints(path_vis, color_rgba=path_color, size=6.0, duration=CONTROL_LOOP_DURATION * 2)
            
          
          
            fut = _start_follow_path(client, vehicle_names[idx], target_path, speed=move_speed)
            
            if fut is not None:
                flight_futures.append(fut)

      
        if traversed_positions:
            _mark_visible_raycast(voxel_map, occupancy, visited, traversed_positions, view_radius)
        if endpoints:
            _mark_visible_raycast(voxel_map, occupancy, visited, endpoints, view_radius * 0.25)

      
        if not benchmark_mode and step % 50 == 0:
            explored_idx = np.argwhere(np.logical_and(visited == 1, allowed_mask))
            if explored_idx.size > 0:
                explored_points = [
                    _to_airsim_vec(voxel_map.world_from_index(tuple(int(i) for i in idx))) for idx in explored_idx
                ]
                client.simPlotPoints(explored_points, color_rgba=[0.0, 1.0, 0.0, 0.6], size=10.0, duration=CONTROL_LOOP_DURATION * 2)

        unknown_exists = bool(np.logical_and(allowed_mask, visited == 0).any())
        
        start_ts = time.time()
        while time.time() - start_ts < CONTROL_LOOP_DURATION:
          
            updated_positions = _tick_update_visibility(client, vehicle_names, voxel_map, occupancy, visited, view_radius)
            agent_positions = np.stack(updated_positions)
            time.sleep(0.1) # 
        else:
            updated_positions = _tick_update_visibility(client, vehicle_names, voxel_map, occupancy, visited, view_radius)
            agent_positions = np.stack(updated_positions)
            

        prev_positions = {i: agent_positions[i].copy() for i in range(len(agent_positions))}

      
        current_exploration = _compute_global_exploration_ratio(allowed_mask, visited)
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[exploration] step={step} ratio={current_exploration:.2%} elapsed={elapsed:.1f}s")
        
      
        if current_exploration >= exploration_target:
            elapsed_time = time.time() - start_time
            exploration_complete = True
            print("\n" + "=" * 60)
            print(f"[EXPLORATION COMPLETE]  {exploration_target:.0%} !")
            print(f"  : {step + 1}")
            print(f"  : {elapsed_time:.2f}  ({elapsed_time / 60:.2f} )")
            print(f"  : {current_exploration:.2%}")
            print(f"  : {elapsed_time / (step + 1):.3f} ")
            print("=" * 60 + "\n")
          
            client.simFlushPersistentMarkers()
            break

        if not unknown_exists:
            break

  
    final_exploration = _compute_global_exploration_ratio(allowed_mask, visited)
    total_time = time.time() - start_time
    total_steps = step + 1
    
    if not exploration_complete:
      
        print("\n" + "=" * 60)
        print(f"[EXPLORATION ENDED]  {exploration_target:.0%} ")
        print(f"  : {total_steps}")
        print(f"  : {total_time:.2f}  ({total_time / 60:.2f} )")
        print(f"  : {final_exploration:.2%}")
        print(f"  : {total_time / total_steps:.3f} ")
        print("=" * 60 + "\n")
      
        client.simFlushPersistentMarkers()
    
  
    if not benchmark_mode:
        final_positions = agent_positions
        white = [1.0, 1.0, 1.0, 1.0]
        drone_points = [_to_airsim_vec(c) for c in final_positions]
        client.simPlotPoints(drone_points, color_rgba=white, size=18.0, duration=30.0)
        _render_layers(client, voxel_map, occupancy, visited)

def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier allocation test for multi-drone exploration")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml or training config.json to load model parameters.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="99%%",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional policy.pt checkpoint to drive exploration instead of frontier allocation.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage name for policy-driven exploration.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max steps for policy-driven exploration (defaults to env max_steps).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for policy inference: auto/cuda/cpu.",
    )
    args = parser.parse_args()
    
  
    benchmark_mode = args.benchmark and not args.debug
    
    if benchmark_mode:
        print("[MODE]  - 99%% ")
    else:
        print("[MODE]  - ")
    
    if airsim is None:
        raise SystemExit("AirSim package not installed; cannot execute spawn test.")

    if args.policy:
        cfg = load_config(args.config) if args.config else load_config()
        device = _pick_device(args.device)
        max_steps = int(args.steps) if args.steps is not None else int(cfg.environment.max_steps)
        _run_policy_exploration(
            cfg=cfg,
            policy_path=args.policy,
            stage_name=args.stage,
            steps=max_steps,
            device=device,
            benchmark_mode=benchmark_mode,
        )
        return
    try:
        test_force_spawn_positions_and_hover(benchmark_mode=benchmark_mode, cfg_path=args.config)
    except AssertionError as exc:
        print(f"[spawn-check] FAILED: {exc}")
        raise SystemExit(1)
    print("[spawn-check] SUCCESS: all drones spawned at expected positions and are hovering.")


if __name__ == "__main__":
    main()
