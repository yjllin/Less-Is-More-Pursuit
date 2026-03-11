"""
High-Performance Vectorized MutualAStar3D Environment V2.

Performance Target: 100+ SPS with 32 parallel environments.
Key Optimizations:
1. All operations batched over (B, N, ...) tensors
2. Zero Python loops in hot path - all Numba parallel
3. Pre-allocated memory buffers
4. Staggered navigation updates (every 10 steps)
5. Simple direct guidance as fallback (A* optional)

Usage:
    env = VectorizedMutualAStarEnvV2(num_envs=32, cfg=cfg)
    obs, info = env.reset()
    for _ in range(1000):
        actions = policy(obs)  # (B, N, 4)
        obs, rewards, dones, infos = env.step(actions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:
        # Minimal fallback to allow running tests without gym installed
        class _DummyBox:
            def __init__(self, low, high, shape=None, dtype=None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        class _DummySpaces:
            Box = _DummyBox
        spaces = _DummySpaces()


from src.config import ThreeDConfig
from src.navigation.astar3d import AStar3D
from src.navigation.frontier_allocation import FrontierAllocator
from src.navigation.voxel_map import VoxelMap3D

from .batch_kernels import (
    batch_update_agents_v2,
    batch_update_targets_v2,
    batch_check_collisions_v2,
    batch_check_team_collisions,
    batch_resolve_collisions_sliding,
    batch_simulate_lidar_v2,
    batch_build_teammate_features,
    batch_build_observations_v2,
    batch_compute_rewards_v2,
    batch_respawn_envs_v2,
    batch_compute_simple_guidance,
    batch_visibility_mask,
    batch_compute_agent_centroids,
    batch_apply_action_smoothing,
    batch_update_exploration_grid_optimized,
    batch_update_exploration_grid_2p5d,
    batch_apply_safety_shield,
    batch_init_position_history,
    batch_update_path_lengths,
    batch_check_lkp_reached,
    batch_update_obs_targets,
    NUMBA_AVAILABLE,
)

def _inflate_occupancy_grid(occupancy: np.ndarray, radius_vox: int) -> np.ndarray:
    """Inflate occupied voxels by a spherical radius (in voxels)."""
    if radius_vox <= 0:
        return occupancy.copy()
    gx, gy, gz = occupancy.shape
    inflated = occupancy.copy()
    offsets = []
    r2 = radius_vox * radius_vox
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    offsets.append((dx, dy, dz))
    for x in range(gx):
        for y in range(gy):
            for z in range(gz):
                if occupancy[x, y, z] != 1:
                    continue
                for dx, dy, dz in offsets:
                    nx = x + dx
                    ny = y + dy
                    nz = z + dz
                    if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                        inflated[nx, ny, nz] = 1
    return inflated


def compute_encirclement_features(
    agents_pos: np.ndarray,
    target_pos: np.ndarray,
    active_masks: np.ndarray,
    arena_bounds: np.ndarray | dict,
    wall_threshold: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute 3D target-centric topological features for each agent.

    Output per agent:
    - global_uniformity_with_walls: ||sum_i u_i + sum(wall_vectors)|| / (active_count + wall_count)
    - max_pairwise_cosine: max_j dot(u_i, u_j), masked by active agents
    """
    if target_pos.ndim == 2:
        target = target_pos[:, None, :]
    elif target_pos.ndim == 3 and target_pos.shape[1] == 1:
        target = target_pos
    else:
        raise ValueError(f"target_pos shape must be (E,3) or (E,1,3), got {target_pos.shape}")

    if isinstance(arena_bounds, dict):
        try:
            x_min = float(arena_bounds["x_min"])
            x_max = float(arena_bounds["x_max"])
            y_min = float(arena_bounds["y_min"])
            y_max = float(arena_bounds["y_max"])
            z_min = float(arena_bounds["z_min"])
            z_max = float(arena_bounds["z_max"])
        except KeyError as exc:
            raise ValueError("arena_bounds dict must include x_min/x_max/y_min/y_max/z_min/z_max") from exc
    else:
        bounds = np.asarray(arena_bounds, dtype=np.float64)
        if bounds.shape == (6,):
            x_min, x_max, y_min, y_max, z_min, z_max = bounds.tolist()
        elif bounds.shape == (2, 3):
            x_min, y_min, z_min = bounds[0].tolist()
            x_max, y_max, z_max = bounds[1].tolist()
        else:
            raise ValueError("arena_bounds must be dict, shape (6,), or shape (2,3)")

    active = active_masks.astype(bool)
    num_envs, num_agents, _ = agents_pos.shape

    # Unit vectors from target to agents in 3D.
    rel = agents_pos - target
    rel_norm = np.linalg.norm(rel, axis=-1, keepdims=True)
    unit = rel / (rel_norm + eps)
    unit = np.where(active[:, :, None], unit, 0.0)

    # Feature 1: global coverage deviation with virtual wall vectors.
    v_agents = np.sum(unit, axis=1)  # (E, 3)
    tgt = target[:, 0, :]  # (E, 3)
    near_x_min = tgt[:, 0] < (x_min + wall_threshold)
    near_x_max = tgt[:, 0] > (x_max - wall_threshold)
    near_y_min = tgt[:, 1] < (y_min + wall_threshold)
    near_y_max = tgt[:, 1] > (y_max - wall_threshold)
    near_z_min = tgt[:, 2] < (z_min + wall_threshold)
    near_z_max = tgt[:, 2] > (z_max - wall_threshold)
    # Wall vectors follow target->wall direction to fill blocked escape directions.
    wall_x = near_x_max.astype(np.float64) - near_x_min.astype(np.float64)
    wall_y = near_y_max.astype(np.float64) - near_y_min.astype(np.float64)
    wall_z = near_z_max.astype(np.float64) - near_z_min.astype(np.float64)
    v_walls = np.stack((wall_x, wall_y, wall_z), axis=-1)
    v_total = v_agents + v_walls
    v_sum_norm = np.linalg.norm(v_total, axis=-1, keepdims=True)  # (E, 1)
    active_count = np.sum(active, axis=1, keepdims=True).astype(np.float64)
    wall_count = (
        near_x_min.astype(np.float64) + near_x_max.astype(np.float64)
        + near_y_min.astype(np.float64) + near_y_max.astype(np.float64)
        + near_z_min.astype(np.float64) + near_z_max.astype(np.float64)
    )[:, None]
    norm_denom = np.maximum(active_count + wall_count, 1.0)
    v_sum_norm = v_sum_norm / norm_denom
    uniformity = np.broadcast_to(v_sum_norm, (num_envs, num_agents))
    uniformity = np.where(active, uniformity, 0.0)

    # Feature 2: max pairwise cosine for each agent.
    gram = np.matmul(unit, np.transpose(unit, (0, 2, 1)))  # (E, A, A)
    active_i = active[:, :, None]
    active_j = active[:, None, :]
    pair_mask = active_i & active_j
    pair_mask &= ~np.eye(num_agents, dtype=bool)[None, :, :]
    gram_masked = np.where(pair_mask, gram, -2.0)
    max_cos = np.max(gram_masked, axis=-1)
    has_neighbor = np.any(pair_mask, axis=-1)
    max_cos = np.where(has_neighbor, max_cos, 0.0)
    max_cos = np.where(active, max_cos, 0.0)

    return np.stack((uniformity.astype(np.float32), max_cos.astype(np.float32)), axis=-1)


def collision_aware_sliding_prediction(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    pred_seconds: float | np.ndarray,
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    world_min: np.ndarray,
    world_max: np.ndarray,
    step_distance_m: float,
    max_substeps: int = 64,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Predict target position with collision-aware sliding.

    This function intentionally does not use target policy internals (e.g., repulse formula).
    It only uses current kinematics + occupancy.

    Args:
        target_pos: (B, 3) current target positions.
        target_vel: (B, 3) current target velocities.
        pred_seconds: scalar or (B,) prediction horizon in seconds.
        occupancy_grid: (Gx, Gy, Gz), 1=occupied, 0=free.
        origin: world origin for occupancy grid.
        voxel_size: voxel size in meters.
        world_min/world_max: world bounds.
        step_distance_m: max move distance per sub-step.
        max_substeps: cap for simulation iterations.
    Returns:
        predicted_pos: (B, 3)
    """
    pos0 = np.asarray(target_pos, dtype=np.float64)
    vel0 = np.asarray(target_vel, dtype=np.float64)
    if pos0.ndim != 2 or pos0.shape[1] != 3:
        raise ValueError(f"target_pos must be (B,3), got {pos0.shape}")
    if vel0.shape != pos0.shape:
        raise ValueError(f"target_vel shape {vel0.shape} must match target_pos {pos0.shape}")

    B = pos0.shape[0]
    if B == 0:
        return pos0.copy()

    if np.isscalar(pred_seconds):
        remaining = np.full(B, float(pred_seconds), dtype=np.float64)
    else:
        pred_arr = np.asarray(pred_seconds, dtype=np.float64).reshape(-1)
        if pred_arr.shape[0] != B:
            raise ValueError(f"pred_seconds must be scalar or (B,), got {pred_arr.shape} for B={B}")
        remaining = pred_arr.copy()
    remaining = np.maximum(remaining, 0.0)

    grid = np.asarray(occupancy_grid)
    gx, gy, gz = grid.shape

    pred = pos0.copy()
    vel = vel0.copy()
    step_distance_m = max(float(step_distance_m), float(voxel_size) * 0.25, 0.1)

    def _blocked_mask(points: np.ndarray) -> np.ndarray:
        world_ok = np.all(points >= world_min[None, :], axis=1) & np.all(points <= world_max[None, :], axis=1)
        idx = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
        idx_ok = (
            (idx[:, 0] >= 0) & (idx[:, 0] < gx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < gy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < gz)
        )
        occ = np.zeros(points.shape[0], dtype=np.bool_)
        valid = world_ok & idx_ok
        if np.any(valid):
            occ[valid] = grid[idx[valid, 0], idx[valid, 1], idx[valid, 2]] > 0
        return (~world_ok) | (~idx_ok) | occ

    for _ in range(max_substeps):
        active = remaining > eps
        if not np.any(active):
            break

        speed = np.linalg.norm(vel, axis=1)
        dt_cap = step_distance_m / np.maximum(speed, eps)
        dt = np.minimum(remaining, dt_cap)
        dt = np.where(active, dt, 0.0)

        candidate = pred + vel * dt[:, None]
        blocked = _blocked_mask(candidate)

        no_hit = active & (~blocked)
        if np.any(no_hit):
            pred[no_hit] = candidate[no_hit]
            remaining[no_hit] -= dt[no_hit]

        hit_idx = np.where(active & blocked)[0]
        for i in hit_idx:
            t_step = float(dt[i])
            if t_step <= eps:
                remaining[i] = 0.0
                continue

            # Axis-wise sliding approximation: zero blocked component(s), keep tangent motion.
            blocked_axis = np.zeros(3, dtype=np.bool_)
            for ax in range(3):
                probe = pred[i].copy()
                probe[ax] += vel[i, ax] * t_step
                blocked_axis[ax] = bool(_blocked_mask(probe[None, :])[0])

            slide_v = vel[i].copy()
            slide_v[blocked_axis] = 0.0

            slid = pred[i] + slide_v * t_step
            if bool(_blocked_mask(slid[None, :])[0]):
                slid = pred[i]
                slide_v[:] = 0.0

            pred[i] = slid
            vel[i] = slide_v
            remaining[i] -= t_step

    pred = np.clip(pred, world_min[None, :], world_max[None, :])
    return pred


@dataclass
class StageConfig:
    """Flattened stage configuration for vectorized access."""
    name: str
    target_speed: float
    view_radius: float
    guidance_enabled: bool
    target_behavior: int  # 0=static, 1=wander, 2=repulse
    capture_radius: float
    collision_penalty: float


class VectorizedMutualAStarEnvV2:
    """
    High-performance vectorized multi-agent pursuit-evasion environment.
    
    All operations are batched and parallelized using Numba.
    Achieves 100+ SPS with 32 parallel environments on modern hardware.
    """
    
    # Behavior mapping
    BEHAVIOR_MAP = {"static": 0, "wander": 1, "repulse": 2}
    
    def __init__(
        self,
        num_envs: int,
        cfg: ThreeDConfig,
        device: str = "cpu",
    ) -> None:
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments (recommended: 32-64)
            cfg: Configuration object
            device: Compute device ("cpu" or "cuda" for future GPU support)
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_agents = cfg.environment.num_agents
        self.device = device
        exp_cfg = getattr(cfg, "experiment", None)
        self.experiment_mode = str(getattr(exp_cfg, "mode", "ours"))
        self.observation_profile = str(getattr(exp_cfg, "observation_profile", "full83")).lower()
        self.guidance_backend = str(getattr(exp_cfg, "guidance_backend", "astar")).lower()
        if self.observation_profile not in {"full83", "local50"}:
            raise ValueError(f"Unsupported observation_profile: {self.observation_profile}")
        if self.guidance_backend not in {"astar", "euclidean"}:
            raise ValueError(f"Unsupported guidance_backend: {self.guidance_backend}")
        self._init_debug = bool(
            getattr(cfg.logging, "numba_warmup", False) or getattr(cfg.logging, "performance_debug", False)
        )
        
        # Timing
        env_cfg = cfg.environment
        self.dt = 1.0 / float(env_cfg.step_hz)
        self.max_steps = env_cfg.max_steps
        
        # Physics limits
        self.max_speed = cfg.control.action_bounds["vx"]
        self.max_accel = getattr(cfg.control, "max_accel", 5.0)
        self.target_max_accel = float(getattr(cfg.control, "target_max_accel", self.max_accel * 1.8))
        self.max_yaw_rate = cfg.control.action_bounds["yaw_rate"]
        self.target_max_vx = float(cfg.control.action_bounds.get("vx", 0.0))
        self.target_max_vy = float(cfg.control.action_bounds.get("vy", 0.0))
        self.target_max_vz = float(cfg.control.action_bounds.get("vz", 0.0))
        self.target_axis_limit_enabled = False
        self.action_delay_min_steps = int(getattr(cfg.control, "action_delay_min_steps", 0))
        self.action_delay_max_steps = int(getattr(cfg.control, "action_delay_max_steps", 0))
        if self.action_delay_min_steps <= 0 and self.action_delay_max_steps <= 0:
            fixed_delay = int(getattr(cfg.control, "action_delay_steps", 0))
            self.action_delay_min_steps = fixed_delay
            self.action_delay_max_steps = fixed_delay
        self.action_delay_buffer = None
        self.action_delay_cursor = 0
        self.action_delay_steps_env = np.zeros(self.num_envs, dtype=np.int32)
        max_delay = max(self.action_delay_max_steps, 0)
        if max_delay > 0:
            self.action_delay_buffer = np.zeros(
                (max_delay + 1, num_envs, self.num_agents, 4),
                dtype=np.float64,
            )
        
        # World bounds
        self.world_min = np.array([
            -env_cfg.world_size_m[0] / 2,
            -env_cfg.world_size_m[1] / 2,
            0.0
        ], dtype=np.float64)
        self.world_max = self.world_min + np.array(env_cfg.world_size_m, dtype=np.float64)
        self._arena_bounds_flat = np.array(
            [
                self.world_min[0], self.world_max[0],
                self.world_min[1], self.world_max[1],
                self.world_min[2], self.world_max[2],
            ],
            dtype=np.float64,
        )
        
        # Spawn bounds
        self.spawn_min = np.array([-140.0, -140.0, 10.0], dtype=np.float64)
        self.spawn_max = np.array([140.0, 140.0, 90.0], dtype=np.float64)
        self.min_spawn_separation = env_cfg.min_spawn_separation_m
        self.target_spawn_min_dist = float(getattr(env_cfg, "target_spawn_min_dist_m", 20.0))
        # <= 0 disables upper-bound filtering in target spawn.
        self.target_spawn_max_dist = float(getattr(env_cfg, "target_spawn_max_dist_m", -1.0))
        self.team_collision_radius = getattr(env_cfg, "team_collision_radius_m", self.cfg.shield.min_distance_m * 0.5)
        
        # Load voxel map
        if self._init_debug:
            print("[Init] Loading voxel map...", flush=True)
        self._init_voxel_map(env_cfg)
        if self._init_debug:
            print("[Init] Voxel map loaded.", flush=True)
        self.target_avoid_radius_vox = max(1, int(np.ceil(6.0 / max(self.voxel_size, 1e-6))))
        
        # 2.5D layered exploration state (B, Gx, Gy, 3)
        grid_shape = self.voxel_map.shape
        default_bins = [[0.0, 33.0], [34.0, 67.0], [68.0, 110.0]]
        self.layer_z_bounds = np.asarray(
            getattr(cfg.navigation, "layer_z_bins", default_bins),
            dtype=np.float64,
        )
        self.layer_centers = np.mean(self.layer_z_bounds, axis=1)
        if self._init_debug:
            print("[Init] Building layer projections...", flush=True)
        self.layer_z_indices = self._compute_layer_z_indices()
        self.static_layer_occ = self._generate_layer_projections()
        if self._init_debug:
            print("[Init] Layer projections ready.", flush=True)
        self.batch_layer_timestamps = np.zeros((num_envs, grid_shape[0], grid_shape[1], 3), dtype=np.int32)
        self.explore_layer_radius_m = float(getattr(cfg.navigation, "explore_layer_radius_m", 50.0))
        self.explore_sample_count = int(getattr(cfg.navigation, "explore_sample_count", 300))
        self.layer_sample_offsets = self._build_layer_sample_offsets(
            self.explore_layer_radius_m,
            self.explore_sample_count,
        )
        self.layer_preference_weight = float(getattr(cfg.navigation, "layer_preference_weight", 2.0))
        self.layer_switch_cooldown_steps = int(getattr(cfg.navigation, "layer_switch_cooldown_steps", 30))
        self.layer_goal_blend = float(getattr(cfg.navigation, "layer_goal_blend", 0.4))
        
      
        self.confidence_decay = getattr(env_cfg, 'confidence_decay', 0.999)
        self.confidence_threshold = getattr(env_cfg, 'confidence_threshold', 0.5)
        steps = int(np.ceil(np.log(self.confidence_threshold) / np.log(self.confidence_decay)))
        self.steps_to_reexplore = max(1, steps)

        
        if self._init_debug:
            print("[Init] Initializing navigation helpers...", flush=True)
        self._init_navigation_helpers()
        if self._init_debug:
            print("[Init] Navigation helpers ready.", flush=True)

        self.visibility_hysteresis_on_steps = int(
            getattr(cfg.navigation, "visibility_hysteresis_on_steps", 1)
        )
        self.visibility_hysteresis_off_steps = int(
            getattr(cfg.navigation, "visibility_hysteresis_off_steps", 2)
        )
        
        # Initialize stage configs
        self._init_stages()
        self.lidar_max_range = float(self.stages[0].view_radius) if self.stages else 0.0
        # Curriculum-adaptive hybrid predictor controls.
        self.hybrid_predictor_stage_start = int(getattr(cfg.navigation, "hybrid_predictor_stage_start", 2))
        self.hybrid_predictor_gate_scale = float(getattr(cfg.navigation, "hybrid_predictor_gate_scale", 0.5))
        self.hybrid_predictor_ema_beta = float(getattr(cfg.navigation, "hybrid_predictor_ema_beta", 0.95))
        self.hybrid_predictor_init_error = float(getattr(cfg.navigation, "hybrid_predictor_init_error_m", 20.0))
        self._pending_nn_target_rel: Optional[np.ndarray] = None
        self._pending_nn_target_rel_valid = False

        # Per-agent goal offset directions to reduce path overlap.
        self.goal_offsets = self._init_goal_offsets()
        
        # Pre-allocate state arrays
        self._init_state_arrays()
        if self._init_debug:
            print("[Init] State arrays ready.", flush=True)
        
        # Pre-allocate buffers
        self._init_buffers()
        if self._init_debug:
            print("[Init] Buffers ready.", flush=True)
        
        # Action/observation spaces
        self._init_spaces()
        if self._init_debug:
            print("[Init] Spaces ready.", flush=True)
        
        # Navigation update interval (staggered across envs for smooth CPU load).
        # Keep configurable to balance SPS and target-tracking responsiveness.
        self.nav_update_interval = max(1, int(getattr(cfg.navigation, "nav_update_interval", 10)))
        self.explore_update_interval = int(getattr(cfg.navigation, "Explore_update_interval", 10))
        self.lidar_update_interval = max(1, int(getattr(env_cfg, "lidar_update_interval", 1)))
        self._lidar_step = 0
        self.offset_decay_distance = float(getattr(cfg.navigation, "Offset_decay_distance", 30.0))
        # Keep a minimum offset near target so formation does not collapse.
        self.goal_offset_scale_floor = 0.45
        self.goal_offset_capture_mult = 1.2
        self.goal_offset_min_radius = 4.0
        self.goal_offset_max_radius = 18.0
        # Dynamic Tactical Slotting (tetrahedron interception around predicted target).
        self.tactical_slot_radius = float(getattr(cfg.navigation, "tactical_slot_radius_m", 16.0))
        self.tactical_slot_radius_offset = float(getattr(cfg.navigation, "tactical_slot_radius_offset_m", 6.0))
        self.tactical_slot_min_dist_to_target = float(
            getattr(cfg.navigation, "tactical_slot_min_dist_to_target_m", 1.0)
        )
        self.tactical_slot_pullback_step = float(
            getattr(cfg.navigation, "tactical_slot_pullback_step_m", max(self.voxel_size * 0.5, 0.5))
        )
        self.tactical_slot_max_pullback_steps = int(
            getattr(cfg.navigation, "tactical_slot_max_pullback_steps", 96)
        )
        # Frontier/exploration bookkeeping
        self.prev_visited_count = 0.0
        
        # Pre-compute confidence LUT for O(1) lookup (replaces expensive np.power)
        # confidence_lut[delta_steps] = decay^delta_steps
        self.confidence_lut = np.power(self.confidence_decay, np.arange(self.max_steps + 1, dtype=np.float32))
        # Reusable exploration mask scratch buffers to avoid per-env allocations in navigation.
        gx, gy, gz = self.voxel_map.shape
        self._exploration_conf2d_scratch = np.zeros((gx, gy, 3), dtype=np.float32)
        self._exploration_mask_scratch = np.zeros((gx, gy, gz), dtype=np.float32)
        
        # Random number generator
        self.rng = np.random.default_rng()
        # Metrics for info logging
        self._metric_shield = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_smoothness = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_align_err = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_far_idle_ratio = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_capture_contribution = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_dist_p50 = np.zeros(self.num_envs, dtype=np.float64)
        self._metric_dist_p90 = np.zeros(self.num_envs, dtype=np.float64)
        self._frontier_debug = [None for _ in range(self.num_envs)]
        # Training fast-path toggles (set by trainer). Keep eval/debug behavior unchanged by default.
        self._store_frontier_debug = True
        self._minimal_train_info = False
        self._diag_metrics_interval_steps = 1
        self.visibility_gate_threshold = np.zeros(self.num_envs, dtype=np.float64)
        self.visibility_gate_min_ratio = 0.0
        self.visibility_gate_max_ratio = 0.0
        self.visibility_gate_open_prob = 0.0
        self.visibility_gate_timeout_steps = 0
        self.visibility_override = 0  # 0=auto, 1=force_visible, -1=force_hidden
        
        # Initialize reward core coefficients for default stage (stage 0)
        self._update_reward_core(self.current_stage_idx)
        self._perf_debug = bool(getattr(cfg.logging, "performance_debug", False))
        self._perf_last: Dict[str, float] = {}
        self._perf_nav: Dict[str, float] = {}
        if self._perf_debug and not NUMBA_AVAILABLE:
            print("[Perf] NUMBA unavailable: running slow Python kernels.")
        
        # Warm-up Numba compilation to avoid 4-minute delay on first step
        warmup_enabled = getattr(cfg.logging, "numba_warmup", True)
        if NUMBA_AVAILABLE and warmup_enabled:
            print("[Init] Warming up Numba kernels (this may take 1-2 minutes)...", flush=True)
            self._warmup_numba_kernels()
            print("[Init] Numba warm-up complete!", flush=True)
        elif not NUMBA_AVAILABLE:
            print("[Init] Numba not available - using slower Python fallbacks", flush=True)

    def set_runtime_mode(self, *, training: bool) -> None:
        """Toggle training/eval runtime behavior without changing env dynamics."""
        is_training = bool(training)
        self._minimal_train_info = is_training
        # Frontier debug snapshots are only used by visualization.
        self._store_frontier_debug = not is_training
        # Downsample expensive diagnostics during training only.
        self._diag_metrics_interval_steps = 4 if is_training else 1

    
    def _init_voxel_map(self, env_cfg) -> None:
        """Initialize voxel map and occupancy grid."""
        self.voxel_map = VoxelMap3D(
            tuple(env_cfg.world_size_m),
            env_cfg.voxel_size_m,
            self.cfg.navigation.occupancy_threshold,
            origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        
        # Load static occupancy grid
        import json
        from pathlib import Path
        
        mask_path = Path(self.cfg.navigation.frontier_mask_path)
        if mask_path.exists():
            self.occupancy_grid = np.load(mask_path).astype(np.int8)
            meta_path = mask_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                self.voxel_size = float(meta.get("voxel_size", env_cfg.voxel_size_m))
            else:
                self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                self.voxel_size = float(env_cfg.voxel_size_m)
        else:
            # Empty grid fallback
            self.occupancy_grid = np.zeros(self.voxel_map.shape, dtype=np.int8)
            self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.voxel_size = float(env_cfg.voxel_size_m)
        
        # Keep voxel map origin/voxel_size aligned with loaded metadata
        self.voxel_map.origin = self.origin
        self.voxel_map.voxel_size = self.voxel_size
        
        # Derive world/spawn bounds from occupancy grid metadata
        occ_world_size = np.array(self.occupancy_grid.shape, dtype=np.float64) * self.voxel_size
        self.world_min = self.origin.copy()
        self.world_max = self.origin + occ_world_size
        req_min = np.array([-140.0, -140.0, 10.0], dtype=np.float64)
        req_max = np.array([140.0, 140.0, 90.0], dtype=np.float64)
        self.spawn_min = np.maximum(self.world_min, req_min)
        self.spawn_max = np.minimum(self.world_max, req_max)
        # Target movement bounds (same as spawn bounds to keep target in valid area)
        self.target_bounds_min = self.spawn_min.copy()
        self.target_bounds_max = self.spawn_max.copy()
        self.min_spawn_separation = env_cfg.min_spawn_separation_m
        
        self.map_size = float(max(occ_world_size[:2]))
        self._free_mask = (self.occupancy_grid <= 0)
        self._free_voxel_count = int(np.count_nonzero(self._free_mask))

    def _compute_layer_z_indices(self) -> np.ndarray:
        gz = self.voxel_map.shape[2]
        indices = np.zeros((3, 2), dtype=np.int32)
        for idx in range(3):
            z_min = float(self.layer_z_bounds[idx, 0])
            z_max = float(self.layer_z_bounds[idx, 1])
            z_min_idx = int(np.floor((z_min - self.origin[2]) / self.voxel_size))
            z_max_idx = int(np.floor((z_max - self.origin[2]) / self.voxel_size))
            z_min_idx = max(0, min(gz - 1, z_min_idx))
            z_max_idx = max(0, min(gz - 1, z_max_idx))
            if z_max_idx < z_min_idx:
                z_min_idx, z_max_idx = z_max_idx, z_min_idx
            indices[idx, 0] = z_min_idx
            indices[idx, 1] = z_max_idx
        return indices

    def _dilate_2d(self, grid: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return grid
        pad = np.pad(grid, radius, mode="constant", constant_values=0)
        out = np.zeros_like(grid)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if np.any(pad[x:x + 2 * radius + 1, y:y + 2 * radius + 1]):
                    out[x, y] = 1
        return out

    def _generate_layer_projections(self) -> np.ndarray:
        gx, gy, _ = self.occupancy_grid.shape
        static_layers = np.zeros((gx, gy, 3), dtype=np.int8)
        # Build 2.5D layer occupancy from per-column occupancy ratio instead of max projection.
        # Max projection is overly conservative and can make high layers appear fully blocked.
        occ_ratio_threshold = float(
            getattr(self.cfg.navigation, "occupancy_threshold", 0.3)
        )
        occ_ratio_threshold = float(np.clip(occ_ratio_threshold, 0.0, 1.0))
        for layer_idx in range(3):
            z_min, z_max = self.layer_z_indices[layer_idx]
            if z_min > z_max:
                continue
            layer_slice = self.occupancy_grid[:, :, z_min:z_max + 1]
            occ_ratio = np.mean(layer_slice, axis=2)
            layer_occ = (occ_ratio >= occ_ratio_threshold).astype(np.int8)
            static_layers[:, :, layer_idx] = layer_occ

        dilate_radius_m = float(getattr(self.cfg.navigation, "layer_dilate_radius_m", 6.0))
        dilate_cells = int(np.ceil(dilate_radius_m / max(self.voxel_size, 1e-6)))
        if dilate_cells > 0:
            for layer_idx in range(3):
                static_layers[:, :, layer_idx] = self._dilate_2d(static_layers[:, :, layer_idx], dilate_cells)
        return static_layers

    def _build_layer_sample_offsets(self, radius_m: float, count: int) -> np.ndarray:
        radius_cells = int(np.ceil(radius_m / max(self.voxel_size, 1e-6)))
        if radius_cells <= 0 or count <= 0:
            return np.zeros((0, 2), dtype=np.int32)
        max_points = int(np.pi * radius_cells * radius_cells)
        if count > max_points:
            if self._init_debug:
                print(f"[Init] explore_sample_count clipped {count} -> {max_points}", flush=True)
            count = max_points
        rng = np.random.default_rng(12345)
        offsets = set()
        max_tries = max(1000, count * 50)
        tries = 0
        while len(offsets) < count and tries < max_tries:
            dx = int(rng.integers(-radius_cells, radius_cells + 1))
            dy = int(rng.integers(-radius_cells, radius_cells + 1))
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.add((dx, dy))
            tries += 1
        result = np.zeros((len(offsets), 2), dtype=np.int32)
        for i, (dx, dy) in enumerate(offsets):
            result[i, 0] = dx
            result[i, 1] = dy
        return result

    def _layer_index_from_z(self, z_val: float) -> int:
        if z_val >= self.layer_z_bounds[2, 0]:
            return 2
        if z_val >= self.layer_z_bounds[1, 0]:
            return 1
        return 0

    def _compute_layer_unvisited_ratio(self, env_idx: int) -> np.ndarray:
        ratios = np.zeros(3, dtype=np.float64)
        ts = self.batch_layer_timestamps[env_idx]
        for layer_idx in range(3):
            free_mask = self.static_layer_occ[:, :, layer_idx] == 0
            free_count = int(np.sum(free_mask))
            if free_count == 0:
                ratios[layer_idx] = 0.0
                continue
            unvisited = np.logical_and(ts[:, :, layer_idx] == 0, free_mask)
            ratios[layer_idx] = float(np.sum(unvisited)) / float(free_count)
        return ratios

    def _assign_preferred_layers(self, env_idx: int) -> np.ndarray:
        """Return preferred layer centers per agent (hard coverage before soft cost)."""
        N = self.num_agents
        costs = np.zeros((N, 3), dtype=np.float64)
        layer_ratios = self._compute_layer_unvisited_ratio(env_idx)
        needs_agent = layer_ratios > 0.05  # explored <95%
        current_layers = np.zeros(N, dtype=np.int32)
        for n in range(N):
            current_layers[n] = self._layer_index_from_z(self.pos[env_idx, n, 2])
            z = self.pos[env_idx, n, 2]
            for layer_idx in range(3):
                climb = abs(z - self.layer_centers[layer_idx])
                ratio_penalty = 1.0 - layer_ratios[layer_idx]
                costs[n, layer_idx] = climb + self.layer_preference_weight * ratio_penalty

        layer_has_agent = np.zeros(3, dtype=np.bool_)
        for layer_idx in range(3):
            layer_has_agent[layer_idx] = np.any(current_layers == layer_idx)

        preferred = np.full(N, -1, dtype=np.int32)
        unassigned = set(range(N))
        required_layers = [i for i in range(3) if needs_agent[i] and not layer_has_agent[i]]
        for layer_idx in required_layers:
            if not unassigned:
                break
            best_agent = None
            best_cost = 1e12
            for n in list(unassigned):
                c = costs[n, layer_idx]
                if c < best_cost:
                    best_cost = c
                    best_agent = n
            if best_agent is not None:
                preferred[best_agent] = int(layer_idx)
                unassigned.discard(best_agent)

        for n in list(unassigned):
            preferred[n] = int(np.argmin(costs[n]))

        # cooldown to avoid frequent switching
        for n in range(N):
            prev_layer = self.preferred_layers[env_idx, n]
            if prev_layer == preferred[n]:
                continue
            last_step = self.preferred_layer_steps[env_idx, n]
            if (self.steps[env_idx] - last_step) < self.layer_switch_cooldown_steps:
                preferred[n] = prev_layer
            else:
                self.preferred_layers[env_idx, n] = preferred[n]
                self.preferred_layer_steps[env_idx, n] = self.steps[env_idx]

        return self.layer_centers[preferred].astype(np.float64)

    def _update_layer_state(self, env_idx: int) -> None:
        for n in range(self.num_agents):
            desired = int(self.preferred_layers[env_idx, n])
            current = self._layer_index_from_z(self.pos[env_idx, n, 2])
            if desired < 0:
                self.layer_state[env_idx, n] = 0
            elif desired == current:
                self.layer_state[env_idx, n] = desired + 1
            else:
                self.layer_state[env_idx, n] = 0

    def _build_layer_frontiers(self, env_idx: int) -> np.ndarray:
        """Build 2.5D frontier candidates from layered confidence maps.
        
        
        FPSagent
        """
        layer_conf = self.get_layer_confidence(env_idx)
        frontiers = []
        max_candidates_per_layer = 200
        rng = np.random.default_rng(int(self.steps[env_idx]) + 123)
        
        for layer_idx in range(3):
            conf = layer_conf[:, :, layer_idx]
            free = self.static_layer_occ[:, :, layer_idx] == 0
            
          
          
            unknown = (conf < self.confidence_threshold) & free
            
            if not np.any(unknown):
              
                continue
            
          
            idx = np.argwhere(unknown)
            
          
            if idx.shape[0] > max_candidates_per_layer:
                pick = rng.choice(idx.shape[0], size=max_candidates_per_layer, replace=False)
                idx = idx[pick]
            
          
            pts = np.zeros((idx.shape[0], 3), dtype=np.float64)
            pts[:, 0] = (idx[:, 0] + 0.5) * self.voxel_size + self.origin[0]
            pts[:, 1] = (idx[:, 1] + 0.5) * self.voxel_size + self.origin[1]
            pts[:, 2] = self.layer_centers[layer_idx]
            frontiers.append(pts)
            
        if not frontiers:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(frontiers)

    def _init_navigation_helpers(self) -> None:
        """Initialize A* navigator and per-env frontier allocators."""
        occ_grid = self.occupancy_grid.astype(np.int8)
        inflate_vox = 1
        if self._init_debug:
            print(f"[Init] Inflating occupancy grid (vox={inflate_vox})...", flush=True)
        self.occupancy_grid_astar = _inflate_occupancy_grid(occ_grid, inflate_vox)
        if self._init_debug:
            print("[Init] Inflation done. Initializing A*...", flush=True)
        self.navigator = AStar3D(
            grid_shape=occ_grid.shape,
            voxel_size=self.voxel_size,
            cache_refresh_steps=self.cfg.navigation.cache_refresh_steps,
            lookahead_m=self.cfg.navigation.lookahead_distance_m,
            heuristic_weight=self.cfg.navigation.heuristic_weight,
            origin=self.origin,
        )
        if self._init_debug:
            print("[Init] A* created. Updating grid...", flush=True)
        self.navigator.update_grid(self.occupancy_grid_astar)
        if self._init_debug:
            print("[Init] A* grid updated. Building allocators...", flush=True)
        
        allowed_mask = self.occupancy_grid == 0
        self.allocators: List[FrontierAllocator] = []
        for _ in range(self.num_envs):
            allocator = FrontierAllocator(
                fps_count=self.cfg.navigation.frontier_fps_count,
                cosine_margin=self.cfg.navigation.frontier_cosine_margin,
                vertical_weight=self.cfg.navigation.frontier_vertical_weight,
                lock_steps=self.cfg.navigation.frontier_lock_steps,
                angle_min_deg=self.cfg.navigation.frontier_angle_min_deg,
                sweep_trigger_m=self.cfg.navigation.frontier_sweep_trigger_m,
                sweep_radius_m=self.cfg.navigation.frontier_sweep_radius_m,
                sweep_max_steps=self.cfg.navigation.frontier_sweep_max_steps,
                sweep_min_distance_m=self.cfg.navigation.frontier_sweep_min_distance_m,
                confidence_threshold=self.confidence_threshold,
                min_frontier_distance_m=float(
                    getattr(self.cfg.navigation, "frontier_min_distance_m", 30.0)
                ),
                use_unvisited_only=bool(
                    getattr(self.cfg.navigation, "frontier_unvisited_only", True)
                ),
                allowed_mask=allowed_mask,
            )
            allocator.set_astar(self.navigator)
            allocator.set_occupancy_grid(self.occupancy_grid)
            self.allocators.append(allocator)
        if self._init_debug:
            print("[Init] Allocators ready.", flush=True)

        self.astar_start_free_radius_vox = int(
            getattr(self.cfg.navigation, "astar_start_free_radius_vox", 2)
        )
        self.astar_goal_free_radius_vox = int(
            getattr(self.cfg.navigation, "astar_goal_free_radius_vox", 2)
        )

    def _snap_astar_start(self, position: np.ndarray) -> np.ndarray:
        """Snap A* start to nearest free voxel if inside inflated occupancy."""
        if self.occupancy_grid_astar is None:
            return position
        idx = np.floor((position - self.origin) / self.voxel_size).astype(int)
        gx, gy, gz = self.occupancy_grid_astar.shape
        if (idx < 0).any() or idx[0] >= gx or idx[1] >= gy or idx[2] >= gz:
            return position
        if self.occupancy_grid_astar[idx[0], idx[1], idx[2]] == 0:
            return position
        radius = max(1, self.astar_start_free_radius_vox)
        best = None
        best_dist = 1e9
        for dx in range(-radius, radius + 1):
            nx = idx[0] + dx
            if nx < 0 or nx >= gx:
                continue
            for dy in range(-radius, radius + 1):
                ny = idx[1] + dy
                if ny < 0 or ny >= gy:
                    continue
                for dz in range(-radius, radius + 1):
                    nz = idx[2] + dz
                    if nz < 0 or nz >= gz:
                        continue
                    if self.occupancy_grid_astar[nx, ny, nz] != 0:
                        continue
                    dist = dx * dx + dy * dy + dz * dz
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny, nz)
        if best is None:
            return position
        return (np.array(best, dtype=np.float64) + 0.5) * self.voxel_size + self.origin

    def _snap_astar_goal(self, position: np.ndarray) -> np.ndarray:
        """Snap A* goal to nearest free voxel if inside inflated occupancy."""
        if self.occupancy_grid_astar is None:
            return position
        idx = np.floor((position - self.origin) / self.voxel_size).astype(int)
        gx, gy, gz = self.occupancy_grid_astar.shape
        if (idx < 0).any() or idx[0] >= gx or idx[1] >= gy or idx[2] >= gz:
            return position
        if self.occupancy_grid_astar[idx[0], idx[1], idx[2]] == 0:
            return position
        radius = max(1, self.astar_goal_free_radius_vox)
        best = None
        best_dist = 1e9
        for dx in range(-radius, radius + 1):
            nx = idx[0] + dx
            if nx < 0 or nx >= gx:
                continue
            for dy in range(-radius, radius + 1):
                ny = idx[1] + dy
                if ny < 0 or ny >= gy:
                    continue
                for dz in range(-radius, radius + 1):
                    nz = idx[2] + dz
                    if nz < 0 or nz >= gz:
                        continue
                    if self.occupancy_grid_astar[nx, ny, nz] != 0:
                        continue
                    dist = dx * dx + dy * dy + dz * dz
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny, nz)
        if best is None:
            return position
        return (np.array(best, dtype=np.float64) + 0.5) * self.voxel_size + self.origin

    def _predict_target_collision_aware(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        pred_seconds: float | np.ndarray,
    ) -> np.ndarray:
        """Collision-aware target prediction wrapper using inflated A* occupancy when available."""
        occ = self.occupancy_grid_astar if self.occupancy_grid_astar is not None else self.occupancy_grid
        return collision_aware_sliding_prediction(
            target_pos=target_pos,
            target_vel=target_vel,
            pred_seconds=pred_seconds,
            occupancy_grid=occ,
            origin=self.origin,
            voxel_size=self.voxel_size,
            world_min=self.world_min,
            world_max=self.world_max,
            step_distance_m=max(self.voxel_size * 0.75, 1.0),
            max_substeps=64,
        )

    def set_nn_target_predictions(self, pred_target_rel_norm: np.ndarray) -> None:
        """
        Set policy-predicted normalized target-relative vectors for current step.

        Args:
            pred_target_rel_norm: (B, N, 3), normalized to [-1, 1], body-frame rel target.
        """
        pred = np.asarray(pred_target_rel_norm, dtype=np.float64)
        expected = (self.num_envs, self.num_agents, 3)
        if pred.shape != expected:
            raise ValueError(f"pred_target_rel_norm shape {pred.shape} != expected {expected}")
        self._pending_nn_target_rel = np.clip(pred, -1.0, 1.0)
        self._pending_nn_target_rel_valid = True

    def _prediction_horizon_seconds(self) -> float:
        if self.current_stage_idx >= 3:
            return 2.8
        if self.current_stage_idx >= 2:
            return 2.0
        return self.dt * 5

    def _decode_nn_target_predictions(self, env_ids: np.ndarray) -> Optional[np.ndarray]:
        """
        Decode pending policy predictions into world-frame target positions per env.

        Returns:
            (len(env_ids), 3) predicted positions, or None when unavailable.
        """
        if (not self._pending_nn_target_rel_valid) or self._pending_nn_target_rel is None:
            return None
        if env_ids.size == 0:
            return np.zeros((0, 3), dtype=np.float64)

        rel_norm = self._pending_nn_target_rel[env_ids]  # (E, N, 3)
        yaws = self.yaw[env_ids]  # (E, N)
        pos = self.pos[env_ids]  # (E, N, 3)
        alive = self.agent_alive[env_ids]  # (E, N)

        z_den = float(self.world_max[2] - self.world_min[2])
        z_scale = (z_den / 3.0) if z_den > 1e-6 else float(self.map_size)
        b_dx = rel_norm[:, :, 0] * float(self.map_size)
        b_dy = rel_norm[:, :, 1] * float(self.map_size)
        t_dz = rel_norm[:, :, 2] * z_scale

        cy = np.cos(yaws)
        sy = np.sin(yaws)
        t_dx = b_dx * cy - b_dy * sy
        t_dy = b_dx * sy + b_dy * cy

        pred_world = np.empty_like(pos)
        pred_world[:, :, 0] = pos[:, :, 0] + t_dx
        pred_world[:, :, 1] = pos[:, :, 1] + t_dy
        pred_world[:, :, 2] = pos[:, :, 2] + t_dz
        pred_world = np.clip(pred_world, self.world_min[None, None, :], self.world_max[None, None, :])

        alive_f = alive.astype(np.float64)
        counts = np.maximum(np.sum(alive_f, axis=1, keepdims=True), 1.0)
        env_pred = np.sum(pred_world * alive_f[:, :, None], axis=1) / counts
        return env_pred

    def _blend_hybrid_target_prediction(
        self,
        env_ids: np.ndarray,
        pred_phys: np.ndarray,
        pred_nn: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend physics and NN predictions with curriculum-adaptive gating."""
        alpha = np.zeros(env_ids.shape[0], dtype=np.float64)
        if pred_nn is None or self.current_stage_idx < self.hybrid_predictor_stage_start:
            return pred_phys, alpha
        phys_err = self.error_phys_ema[env_ids]
        nn_err = self.error_nn_ema[env_ids]
        alpha = np.clip((phys_err - nn_err) * self.hybrid_predictor_gate_scale, 0.0, 1.0)
        blended = (1.0 - alpha[:, None]) * pred_phys + alpha[:, None] * pred_nn
        blended = np.clip(blended, self.world_min[None, :], self.world_max[None, :])
        return blended, alpha

    def _update_prediction_error_ema(self) -> None:
        """Update per-env prediction error EMA using current target as ground truth."""
        valid = self.last_pred_valid
        if not np.any(valid):
            return
        beta = self.hybrid_predictor_ema_beta
        actual = self.target_pos[valid]
        phys_err = np.linalg.norm(self.last_pred_phys[valid] - actual, axis=1)
        nn_err = np.linalg.norm(self.last_pred_nn[valid] - actual, axis=1)
        self.error_phys_ema[valid] = beta * self.error_phys_ema[valid] + (1.0 - beta) * phys_err
        self.error_nn_ema[valid] = beta * self.error_nn_ema[valid] + (1.0 - beta) * nn_err

    def _generate_slots(self, predicted_target_pos: np.ndarray, encircle_radius: float) -> np.ndarray:
        """
        Generate fixed-world-frame regular tetrahedron slots around predicted targets.

        Args:
            predicted_target_pos: (E, 3)
            encircle_radius: scalar radius in meters
        Returns:
            slots: (E, 4, 3)
        """
        targets = np.asarray(predicted_target_pos, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"predicted_target_pos must be (E,3), got {targets.shape}")
        r = max(float(encircle_radius), 0.1)
        sq2 = math.sqrt(2.0)
        sq6 = math.sqrt(6.0)
        # Regular tetrahedron on unit sphere:
        # slot0 up, slot1/2/3 on lower plane 120 deg apart.
        dirs = np.array(
            [
                [0.0, 0.0, 1.0],
                [2.0 * sq2 / 3.0, 0.0, -1.0 / 3.0],
                [-sq2 / 3.0, sq6 / 3.0, -1.0 / 3.0],
                [-sq2 / 3.0, -sq6 / 3.0, -1.0 / 3.0],
            ],
            dtype=np.float64,
        )
        return targets[:, None, :] + r * dirs[None, :, :]

    def _points_blocked_mask(self, points: np.ndarray, occupancy_grid: np.ndarray) -> np.ndarray:
        """Return blocked mask for world-frame points: out-of-bounds or occupied voxel."""
        pts = np.asarray(points, dtype=np.float64)
        world_ok = np.all(pts >= self.world_min[None, :], axis=1) & np.all(pts <= self.world_max[None, :], axis=1)
        idx = np.floor((pts - self.origin[None, :]) / float(self.voxel_size)).astype(np.int64)
        gx, gy, gz = occupancy_grid.shape
        idx_ok = (
            (idx[:, 0] >= 0) & (idx[:, 0] < gx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < gy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < gz)
        )
        occ = np.zeros(pts.shape[0], dtype=np.bool_)
        valid = world_ok & idx_ok
        if np.any(valid):
            occ[valid] = occupancy_grid[idx[valid, 0], idx[valid, 1], idx[valid, 2]] > 0
        return (~world_ok) | (~idx_ok) | occ

    def _rectify_slots(
        self,
        slots: np.ndarray,
        target_pos: np.ndarray,
        occupancy_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Rectify tactical slots to valid free space with boundary clip + obstacle pull-back.

        Args:
            slots: (E, M, 3)
            target_pos: (E, 3)
            occupancy_grid: optional occupancy grid; defaults to inflated A* grid.
        Returns:
            rectified slots: (E, M, 3)
        """
        occ = self.occupancy_grid_astar if occupancy_grid is None and self.occupancy_grid_astar is not None else (
            occupancy_grid if occupancy_grid is not None else self.occupancy_grid
        )
        out = np.asarray(slots, dtype=np.float64).copy()
        targets = np.asarray(target_pos, dtype=np.float64)
        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(f"slots must be (E,M,3), got {out.shape}")
        if targets.ndim != 2 or targets.shape[1] != 3 or targets.shape[0] != out.shape[0]:
            raise ValueError(f"target_pos must be (E,3) and align with slots, got {targets.shape}")

        out = np.clip(out, self.world_min[None, None, :], self.world_max[None, None, :])
        flat = out.reshape(-1, 3)
        blocked = self._points_blocked_mask(flat, occ)
        if not np.any(blocked):
            return out

        step = max(float(self.tactical_slot_pullback_step), float(self.voxel_size) * 0.25, 0.1)
        max_steps = max(int(self.tactical_slot_max_pullback_steps), 1)
        min_dist = max(float(self.tactical_slot_min_dist_to_target), 0.0)
        num_envs, num_slots, _ = out.shape

        blocked_idx = np.where(blocked)[0]
        for flat_i in blocked_idx:
            e = flat_i // num_slots
            s = flat_i % num_slots
            slot = out[e, s].copy()
            tgt = targets[e]
            ray = tgt - slot
            ray_dist = float(np.linalg.norm(ray))
            if ray_dist <= 1e-6:
                out[e, s] = tgt
                continue
            direction = ray / ray_dist
            n_steps = min(max_steps, max(1, int(math.ceil(ray_dist / step))))
            found = False
            best = tgt.copy()
            for k in range(1, n_steps + 1):
                cand = slot + direction * (k * ray_dist / n_steps)
                cand = np.clip(cand, self.world_min, self.world_max)
                if not self._points_blocked_mask(cand[None, :], occ)[0]:
                    best = cand
                    found = True
                    break
            if not found:
                best = tgt.copy()
            if np.linalg.norm(best - tgt) < min_dist:
                best = tgt.copy()
            out[e, s] = best
        return out

    def _assign_slots_hungarian(
        self,
        agent_positions: np.ndarray,
        slots: np.ndarray,
        active_mask: np.ndarray,
        fallback_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign active agents to tactical slots by Hungarian minimum-distance matching.

        Args:
            agent_positions: (N,3)
            slots: (M,3)
            active_mask: (N,) bool
            fallback_target: (3,)
        Returns:
            per-agent goals: (N,3)
            per-agent slot assignment id: (N,), -1 means unassigned
        """
        agents = np.asarray(agent_positions, dtype=np.float64)
        slot_pts = np.asarray(slots, dtype=np.float64)
        mask = np.asarray(active_mask, dtype=np.bool_)
        goals = np.repeat(np.asarray(fallback_target, dtype=np.float64)[None, :], agents.shape[0], axis=0)
        slot_ids = np.full(agents.shape[0], -1, dtype=np.int32)
        active_idx = np.where(mask)[0]
        if active_idx.size == 0 or slot_pts.shape[0] == 0:
            return goals, slot_ids

        active_agents = agents[active_idx]
        diff = active_agents[:, None, :] - slot_pts[None, :, :]
        cost = np.linalg.norm(diff, axis=-1)
        row_idx, col_idx = linear_sum_assignment(cost)
        assigned_rows = np.zeros(active_idx.size, dtype=np.bool_)
        for rr, cc in zip(row_idx, col_idx):
            goals[active_idx[rr]] = slot_pts[cc]
            slot_ids[active_idx[rr]] = int(cc)
            assigned_rows[rr] = True
        # If active agents outnumber slots, give remaining agents nearest slot.
        if np.any(~assigned_rows):
            for rr in np.where(~assigned_rows)[0]:
                nearest = int(np.argmin(cost[rr]))
                goals[active_idx[rr]] = slot_pts[nearest]
                slot_ids[active_idx[rr]] = nearest
        return goals, slot_ids

    def _init_stages(self) -> None:
        """Initialize stage configurations."""
        self.stages: List[StageConfig] = []
        for stage_cfg in self.cfg.curriculum.stages:
            behavior = self.BEHAVIOR_MAP.get(stage_cfg.target_behavior, 2)
            self.stages.append(StageConfig(
                name=stage_cfg.name,
                target_speed=stage_cfg.target_speed,
                view_radius=stage_cfg.view_radius_m,
                guidance_enabled=stage_cfg.guidance_enabled,
                target_behavior=behavior,
                capture_radius=getattr(stage_cfg, "capture_radius_m", self.cfg.environment.capture_radius_m),
                collision_penalty=stage_cfg.collision_penalty or float(self.cfg.reward_core_defaults.collision_cost),
            ))
        self.current_stage_idx = 0

    def _init_goal_offsets(self) -> np.ndarray:
        """Initialize per-agent unit offset directions to reduce path overlap."""
        offsets = np.zeros((self.num_agents, 3), dtype=np.float64)
        if self.num_agents <= 1:
            return offsets
        for i in range(self.num_agents):
            angle = (2.0 * np.pi * i) / max(self.num_agents, 1)
            offsets[i, 0] = np.cos(angle)
            offsets[i, 1] = np.sin(angle)
            offsets[i, 2] = 0.0
        return offsets
    
    def _init_state_arrays(self) -> None:
        """Pre-allocate all state arrays."""
        B, N = self.num_envs, self.num_agents
        
        # Agent state
        self.pos = np.zeros((B, N, 3), dtype=np.float64)
        self.vel = np.zeros((B, N, 3), dtype=np.float64)
        self.yaw = np.zeros((B, N), dtype=np.float64)
        self.prev_actions = np.zeros((B, N, 4), dtype=np.float64)
        self.agent_alive = np.ones((B, N), dtype=np.bool_)
        
        # Target state
        self.target_pos = np.zeros((B, 3), dtype=np.float64)
        self.target_vel = np.zeros((B, 3), dtype=np.float64)
        
        # Navigation state
        self.current_goals = np.zeros((B, N, 3), dtype=np.float64)
        self.guidance_vectors = np.zeros((B, N, 3), dtype=np.float64)
        # Tactical slotting state
        self.slot_positions = np.zeros((B, N, 3), dtype=np.float64)  # per-env slot coordinates
        self.slot_assignments = np.full((B, N), -1, dtype=np.int32)  # per-agent assigned slot id (-1 means none)
        
        # Target tracking state
        self.last_known_target_pos = np.zeros((B, 3), dtype=np.float64)  # 
        self.target_lost_steps = np.zeros(B, dtype=np.int32)  # 
        self.lkp_countdown = np.zeros(B, dtype=np.int32)  #  (30 steps)
        
        # Target mode tracking (per agent)
        self.target_mode_flag = np.ones((B, N), dtype=np.float32)  # 1.0=tracking, 0.0=searching
        self.prev_target_mode_flag = np.ones((B, N), dtype=np.float32)  # 
        self.preferred_layers = np.zeros((B, N), dtype=np.int32)
        self.preferred_layer_steps = np.zeros((B, N), dtype=np.int32)
        self.layer_state = np.zeros((B, N), dtype=np.int32)
        
        # Physics tracking for stagnation detection
        self.position_history = np.zeros((B, N, 5, 3), dtype=np.float64)  # 5
        self.history_idx = np.zeros((B, N), dtype=np.int32)  # 
        
        # Visibility state (cached from batch_visibility_mask)
        self._any_visible = np.zeros(B, dtype=np.bool_)
        self.team_has_target = np.zeros(B, dtype=np.bool_)
        self.nav_visible_mask = np.zeros((B, N), dtype=np.bool_)
        self.nav_visible_streak = np.zeros((B, N), dtype=np.int32)
        self.nav_invisible_streak = np.zeros((B, N), dtype=np.int32)
        
        # Reward tracking
        self.prev_target_dist = np.zeros((B, N), dtype=np.float64)
        self.prev_frontier_dist = np.zeros((B, N), dtype=np.float64)
        self.frontier_deltas = np.zeros((B, N), dtype=np.float64)
        self.prev_frontier_goals = np.zeros((B, N, 3), dtype=np.float64)
        self.prev_frontier_mask = np.zeros((B, N), dtype=np.bool_)
        self.path_lengths = np.zeros((B, N), dtype=np.float64)
        
        # Environment state
        self.steps = np.zeros(B, dtype=np.int32)
        self.stage_ids = np.zeros(B, dtype=np.int32)
        
        # Per-env stage config arrays (for vectorized access)
        self.env_target_speeds = np.zeros(B, dtype=np.float64)
        self.env_view_radius = np.zeros(B, dtype=np.float64)
        self.env_guidance_enabled = np.zeros(B, dtype=np.bool_)
        self.env_behaviors = np.zeros(B, dtype=np.int32)
        self.env_capture_radius = np.zeros(B, dtype=np.float64)
        # Hybrid predictor state (per-env EMAs and last predictions).
        self.error_phys_ema = np.full(B, self.hybrid_predictor_init_error, dtype=np.float64)
        self.error_nn_ema = np.full(B, self.hybrid_predictor_init_error, dtype=np.float64)
        self.last_pred_phys = np.zeros((B, 3), dtype=np.float64)
        self.last_pred_nn = np.zeros((B, 3), dtype=np.float64)
        self.last_pred_valid = np.zeros(B, dtype=np.bool_)
        self.last_pred_alpha = np.zeros(B, dtype=np.float64)

    
    def _init_buffers(self) -> None:
        """Pre-allocate computation buffers."""
        B, N = self.num_envs, self.num_agents
        
        # Observation buffer
        # Frame stacking removed: always use single-frame observations.
        self.frame_stack = 1
        self.base_obs_dim = self._calc_base_obs_dim()
        obs_dim = self._calc_obs_dim()
        self.imu_accel_noise_std = float(getattr(self.cfg.environment, "imu_accel_noise_std", 0.0))
        self.imu_gyro_noise_std = float(getattr(self.cfg.environment, "imu_gyro_noise_std", 0.0))
        self.imu_linear_acc_body = np.zeros((B, N, 3), dtype=np.float64)
        self.imu_angular_vel_body = np.zeros((B, N, 3), dtype=np.float64)
        self.lidar_sector_subrays = int(getattr(self.cfg.perception, "lidar_sector_subrays", 5))
        self.lidar_sector_spread_deg = float(getattr(self.cfg.perception, "lidar_sector_spread_deg", 10.0))
        self.lidar_dropout_prob = float(getattr(self.cfg.perception, "lidar_dropout_prob", 0.0))
        self.lidar_noise_std = float(getattr(self.cfg.perception, "lidar_noise_std", 0.0))
        self.target_obs_noise_std = float(getattr(self.cfg.perception, "target_obs_noise_std", 0.0))
        
        # LiDAR buffer
        self.lidar_buffer = np.zeros((B, N, 26), dtype=np.float64)
        
        # Teammate features buffer
        top_k = self.cfg.perception.teammate_top_k
        self.teammate_buffer = np.zeros((B, N, top_k * 8), dtype=np.float64)
        
        # Observation target buffers (per-agent targets for observations)
        self.obs_target_pos = np.zeros((B, N, 3), dtype=np.float64)
        self.prev_obs_target_pos = np.zeros((B, N, 3), dtype=np.float64)  # obs_target_pos
        self.obs_target_vel = np.zeros((B, N, 3), dtype=np.float64)

        # Collision rearm state (impact only on first contact)
        self.collision_armed = np.ones((B, N), dtype=np.bool_)
        self.collision_rearm_count = np.zeros((B, N), dtype=np.int32)
        
        # 9-core reward coefficients (single-layer scaling).
        self.reward_core_coefs = np.zeros(9, dtype=np.float64)
        
        # Random number buffers
        max_attempts = 200
        self.spawn_rng_buffer = np.zeros((B, N + 1, max_attempts, 3), dtype=np.float64)
        self.yaw_rng_buffer = np.zeros((B, N), dtype=np.float64)
        self.target_rng_buffer = np.zeros((B, 3), dtype=np.float64)

    def _apply_target_observation_noise(self, visible_tracking_mask: np.ndarray) -> None:
        if self.target_obs_noise_std <= 0.0:
            return
        mask = np.asarray(visible_tracking_mask, dtype=np.bool_)
        if mask.shape != self.obs_target_pos.shape[:2] or not np.any(mask):
            return
        noise = self.rng.normal(0.0, self.target_obs_noise_std, size=self.obs_target_pos.shape)
        noisy = self.obs_target_pos + noise * mask[..., None]
        self.obs_target_pos = np.clip(noisy, self.world_min[None, None, :], self.world_max[None, None, :])
    
    def _init_spaces(self) -> None:
        """Initialize action and observation spaces."""
        obs_dim = self._calc_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        bounds = self.cfg.control.action_bounds
        self.action_space = spaces.Box(
            low=np.array([-bounds["vx"], -bounds["vy"], -bounds["vz"], -bounds["yaw_rate"]], dtype=np.float32),
            high=np.array([bounds["vx"], bounds["vy"], bounds["vz"], bounds["yaw_rate"]], dtype=np.float32),
            dtype=np.float32,
        )
    
    def _calc_base_obs_dim(self) -> int:
        """Calculate single-frame observation dimension."""
        if self.observation_profile == "local50":
            return 50
        top_k = self.cfg.perception.teammate_top_k
        num_agents = self.num_agents
        # lidar(26) + target_rel(3) + target_vel(3) + self_vel(3) +
        # teammate(top_k*8) + guidance(3) + agent_id(num_agents) +
        # target_mode_flag(1) + action_delay(1) +
        # tactical_slot_id_onehot(num_agents) + tactical_slot_rel_vec(3) +
        # encircle_topology(2)
        return 26 + 3 + 3 + 3 + 3 + 3 + (top_k * 8) + 3 + num_agents + 1 + 1 + num_agents + 3 + 2

    def _calc_obs_dim(self) -> int:
        """Calculate stacked observation dimension."""
        return self._calc_base_obs_dim()

    def _use_euclidean_guidance(self) -> bool:
        return self.guidance_backend == "euclidean"

    def _project_full_obs_to_local50(self, obs_full: np.ndarray) -> np.ndarray:
        """Project full 83D observation to local 50D profile."""
        # Full layout: [0:41]=lidar+target/self/imu, [41:65]=teammates, [65:74]=guidance+id+mode+delay, [74:81]=slot, [81:83]=encircle
        if obs_full.shape[-1] != 83:
            raise ValueError(f"Expected full observation dim 83 before local projection, got {obs_full.shape[-1]}")
        head = obs_full[..., :41]
        tail = obs_full[..., 65:74]
        return np.concatenate((head, tail), axis=-1)

    @staticmethod
    def _compute_euclidean_guidance(agent_pos: np.ndarray, final_goal: np.ndarray) -> tuple[np.ndarray, float]:
        delta = final_goal - agent_pos
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            return (delta / norm).astype(np.float64, copy=False), norm
        return np.zeros(3, dtype=np.float64), 0.0
    
    def _warmup_numba_kernels(self) -> None:
        """Warm up Numba kernels with dummy data to avoid JIT compilation delays during runtime."""
        import time
        from . import batch_kernels
        
        # Create minimal dummy data for compilation
        B, N = 2, 2  # Small batch for warmup
        dummy_pos = np.zeros((B, N, 3), dtype=np.float64)
        dummy_vel = np.zeros((B, N, 3), dtype=np.float64)
        dummy_yaw = np.zeros((B, N), dtype=np.float64)
        dummy_actions = np.zeros((B, N, 4), dtype=np.float64)
        dummy_target_pos = np.zeros((B, 3), dtype=np.float64)
        dummy_target_vel = np.zeros((B, 3), dtype=np.float64)
        dummy_lidar = np.ones((B, N, 26), dtype=np.float64)
        dummy_grid = np.zeros((10, 10, 10), dtype=np.int8)
        dummy_origin = np.zeros(3, dtype=np.float64)
        dummy_world_bounds = np.array([[-10, -10, 0], [10, 10, 10]], dtype=np.float64)
        dummy_bool = np.zeros((B, N), dtype=np.bool_)
        dummy_timestamps = np.zeros((B, 10, 10, 10), dtype=np.int32)
        dummy_steps = np.zeros(B, dtype=np.int32)
        
        start_time = time.time()
        
        # Warm up critical kernels that cause the most delay
        kernels_to_warmup = [
            # Physics kernels
              (batch_kernels.batch_update_agents_v2, (
                  dummy_pos, dummy_vel, dummy_yaw, dummy_actions, 0.1, 10.0, 5.0, 1.0,
                  dummy_world_bounds[0], dummy_world_bounds[1],
                  np.zeros((B, N, 3), dtype=np.float64),
                  np.zeros((B, N, 3), dtype=np.float64),
                  np.zeros((B, N, 3), dtype=np.float64),
                  np.zeros((B, N, 3), dtype=np.float64),
              )),
            
            # Collision kernels  
            (batch_kernels.batch_check_collisions_v2, (
                dummy_pos, dummy_grid, dummy_origin, 1.0
            )),
            
            # LiDAR kernel
              (batch_kernels.batch_simulate_lidar_v2, (
                  dummy_pos, dummy_yaw, dummy_grid, dummy_origin, 1.0, 50.0, 5, 10.0
              )),
            
            # Observation kernel
            (batch_kernels.batch_build_teammate_features, (
                dummy_pos, dummy_vel, dummy_yaw, 3, 100.0
            )),
            
            # Exploration kernel
            (batch_kernels.batch_update_exploration_grid_optimized, (
                dummy_pos, dummy_lidar, dummy_timestamps, dummy_steps,
                dummy_origin, 1.0, 50.0, 100
            )),
        ]
        
        for i, (kernel_func, args) in enumerate(kernels_to_warmup):
            try:
                print(f"[Init] Compiling kernel {i+1}/{len(kernels_to_warmup)}: {kernel_func.__name__}...", flush=True)
                kernel_func(*args)
            except Exception as e:
                print(f"[Init] Warning: Failed to warm up {kernel_func.__name__}: {e}")
        
        elapsed = time.time() - start_time
        print(f"[Init] Numba warmup completed in {elapsed:.1f}s", flush=True)
    
    def _update_stage_arrays(self, stage_idx: int) -> None:
        """Update per-env stage configuration arrays."""
        stage = self.stages[stage_idx]
        stage_cfg = self.cfg.curriculum.stages[stage_idx]
        self.env_target_speeds[:] = stage.target_speed
        self.env_view_radius[:] = stage.view_radius
        self.env_guidance_enabled[:] = stage.guidance_enabled
        self.env_behaviors[:] = stage.target_behavior
        self.env_capture_radius[:] = stage.capture_radius
        self.stage_ids[:] = stage_idx
        self.lidar_max_range = float(stage.view_radius)
        self.target_axis_limit_enabled = bool(getattr(stage_cfg, "target_axis_limit", False))

        # Update reward core coefficients based on stage config.
        self._update_reward_core(stage_idx)
        self.visibility_gate_min_ratio = float(getattr(stage, "visibility_gate_min_ratio", 0.0))
        self.visibility_gate_max_ratio = float(getattr(stage, "visibility_gate_max_ratio", 0.0))
        self.visibility_gate_open_prob = float(getattr(stage, "visibility_gate_open_prob", 0.0))
        self.visibility_gate_timeout_steps = int(getattr(stage, "visibility_gate_timeout_steps", 0))
        self._resample_visibility_gate_thresholds(np.ones(self.num_envs, dtype=np.bool_))

    def _resample_visibility_gate_thresholds(self, mask: np.ndarray) -> None:
        if self.visibility_gate_max_ratio <= 0.0 or not np.any(mask):
            self.visibility_gate_threshold[mask] = 0.0
            return
        low = min(self.visibility_gate_min_ratio, self.visibility_gate_max_ratio)
        high = max(self.visibility_gate_min_ratio, self.visibility_gate_max_ratio)
        self.visibility_gate_threshold[mask] = self.rng.uniform(low, high, size=int(np.sum(mask)))

    def _compute_explored_ratio(self) -> np.ndarray:
        visited = self.batch_layer_timestamps > 0
        if self._free_voxel_count > 0:
            visited = visited & self._free_mask[None, ...]
            denom = float(self._free_voxel_count)
        else:
            denom = float(np.prod(self.batch_layer_timestamps.shape[1:]))
        return visited.reshape(self.num_envs, -1).sum(axis=1) / max(denom, 1.0)

    def _apply_visibility_gate(self, all_visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.visibility_override == 1:
            gate_open = np.ones(self.num_envs, dtype=np.bool_)
            return np.ones_like(all_visible), gate_open
        if self.visibility_override == -1:
            gate_open = np.zeros(self.num_envs, dtype=np.bool_)
            return np.zeros_like(all_visible), gate_open
        if self.visibility_gate_max_ratio <= 0.0:
            gate_open = np.ones(self.num_envs, dtype=np.bool_)
            return all_visible, gate_open
        explored_ratio = self._compute_explored_ratio()
        gate_open = explored_ratio >= self.visibility_gate_threshold
        # Time-based gate opening disabled: require exploration to open visibility gate.
        if self.visibility_gate_open_prob > 0.0:
            gate_open |= (self.rng.random(self.num_envs) < self.visibility_gate_open_prob)
        return all_visible & gate_open[:, None], gate_open

    def _reset_visibility_hysteresis(self, all_visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.nav_visible_mask[:] = all_visible
        self.nav_visible_streak[:] = all_visible.astype(np.int32)
        self.nav_invisible_streak[:] = (~all_visible).astype(np.int32)
        any_visible = np.any(self.nav_visible_mask, axis=1)
        return self.nav_visible_mask, any_visible

    def _apply_visibility_hysteresis(self, all_visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        on_steps = max(1, int(self.visibility_hysteresis_on_steps))
        off_steps = max(1, int(self.visibility_hysteresis_off_steps))
        if on_steps <= 1 and off_steps <= 1:
            self.nav_visible_mask[:] = all_visible
            self.nav_visible_streak[:] = all_visible.astype(np.int32)
            self.nav_invisible_streak[:] = (~all_visible).astype(np.int32)
            any_visible = np.any(self.nav_visible_mask, axis=1)
            return self.nav_visible_mask, any_visible
        self.nav_visible_streak[all_visible] += 1
        self.nav_invisible_streak[all_visible] = 0
        self.nav_invisible_streak[~all_visible] += 1
        self.nav_visible_streak[~all_visible] = 0
        turn_on = all_visible & (self.nav_visible_streak >= on_steps)
        turn_off = (~all_visible) & (self.nav_invisible_streak >= off_steps)
        self.nav_visible_mask[turn_on] = True
        self.nav_visible_mask[turn_off] = False
        any_visible = np.any(self.nav_visible_mask, axis=1)
        return self.nav_visible_mask, any_visible

    def set_target_spawn_range(self, min_dist: float, max_dist: float) -> None:
        self.target_spawn_min_dist = float(min_dist)
        self.target_spawn_max_dist = float(max_dist)

    def set_stage_params(
        self,
        stage_idx: int,
        target_speed: Optional[float] = None,
        view_radius: Optional[float] = None,
    ) -> None:
        if stage_idx < 0 or stage_idx >= len(self.stages):
            return
        stage = self.stages[stage_idx]
        if target_speed is not None:
            stage.target_speed = float(target_speed)
            self.cfg.curriculum.stages[stage_idx].target_speed = float(target_speed)
        if view_radius is not None:
            stage.view_radius = float(view_radius)
            self.cfg.curriculum.stages[stage_idx].view_radius_m = float(view_radius)
        if stage_idx == self.current_stage_idx:
            self._update_stage_arrays(stage_idx)

    def _update_reward_core(self, stage_idx: int) -> None:
        """Update 9-core reward coefficients for the active stage."""
        stage_cfg = self.cfg.curriculum.stages[stage_idx]
        overrides = stage_cfg.reward_core or {}
        defaults = self.cfg.reward_core_defaults

        def core_value(key: str, default: float) -> float:
            if key in overrides:
                return float(overrides[key])
            return float(getattr(defaults, key, default))

        self.reward_core_coefs = np.array(
            [
                core_value("step_cost", -0.05),
                core_value("progress_gain", 0.8),
                core_value("exploration_gain", 0.2),
                core_value("proximity_cost", -0.4),
                core_value("collision_cost", -1.2),
                core_value("direction_gain", 0.15),
                core_value("control_cost", -0.25),
                core_value("capture_gain", 1.5),
                core_value("capture_quality_gain", 0.6),
            ],
            dtype=np.float64,
        )

        rr = self.cfg.reward_runtime
        self.frontier_reach_dist = float(rr.frontier_reach_dist_m)
        self.lidar_safe_distance_m = float(rr.lidar_safe_distance_m)
        self.search_speed_floor_mps = float(rr.search_speed_floor_mps)
        self.direction_gate_active_radius_m = float(getattr(rr, "direction_gate_active_radius_m", 80.0))
        self.potential_unstable_obs_jump_m = float(rr.potential_unstable_obs_jump_m)
        self.potential_unstable_path_jump_m = float(rr.potential_unstable_path_jump_m)

    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        stage_index: Optional[int] = None,
        env_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environments.
        
        Args:
            seed: Random seed
            stage_index: Curriculum stage to use
            env_ids: Specific environment IDs to reset (None = all)
        
        Returns:
            obs: (B, N, obs_dim) observations
            info: Dictionary with stage info
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        if stage_index is not None:
            self.current_stage_idx = stage_index
        
        self._update_stage_arrays(self.current_stage_idx)
        
        # Determine which envs to reset
        if env_ids is None:
            reset_mask = np.ones(self.num_envs, dtype=np.bool_)
        else:
            reset_mask = np.zeros(self.num_envs, dtype=np.bool_)
            reset_mask[env_ids] = True
        self._resample_visibility_gate_thresholds(reset_mask)
        if self.action_delay_buffer is not None:
            self.action_delay_buffer[:, reset_mask, :, :] = 0.0
            low = min(self.action_delay_min_steps, self.action_delay_max_steps)
            high = max(self.action_delay_min_steps, self.action_delay_max_steps)
            if high > 0:
                self.action_delay_steps_env[reset_mask] = self.rng.integers(low, high + 1, size=int(np.sum(reset_mask)))
            else:
                self.action_delay_steps_env[reset_mask] = 0
        self.last_pred_valid[reset_mask] = False
        self.last_pred_alpha[reset_mask] = 0.0
        self.error_phys_ema[reset_mask] = self.hybrid_predictor_init_error
        self.error_nn_ema[reset_mask] = self.hybrid_predictor_init_error
        self._pending_nn_target_rel_valid = False

        # Reset collision rearm state for selected envs
        self.collision_armed[reset_mask, :] = True
        self.collision_rearm_count[reset_mask, :] = 0
        self.agent_alive[reset_mask, :] = True
        
        # Generate random numbers
        self._generate_spawn_rng()
        
        # Batch respawn
        batch_respawn_envs_v2(
            reset_mask,
            self.pos, self.vel, self.yaw, self.prev_actions,
            self.target_pos, self.target_vel,
            self.steps, self.prev_target_dist, self.current_goals,
            self.occupancy_grid_astar, self.origin, self.voxel_size,
            self.spawn_min, self.spawn_max, self.min_spawn_separation,
            self.target_spawn_min_dist, self.target_spawn_max_dist,
            self.spawn_rng_buffer, self.yaw_rng_buffer,
        )

        # Initialize preferred layers based on current positions
        for env_id in np.where(reset_mask)[0]:
            for n in range(self.num_agents):
                layer_idx = self._layer_index_from_z(self.pos[env_id, n, 2])
                self.preferred_layers[env_id, n] = layer_idx
                self.preferred_layer_steps[env_id, n] = int(self.steps[env_id])
                self.layer_state[env_id, n] = layer_idx + 1

        self.imu_linear_acc_body[reset_mask, :, :] = 0.0
        self.imu_angular_vel_body[reset_mask, :, :] = 0.0
        
        # Clear A* cache for reset environments to prevent memory growth
        reset_indices = np.where(reset_mask)[0]
        for env_id in reset_indices:
            self.navigator.clear_cache_for_env(int(env_id))
        
        self.prev_frontier_dist[:] = 0.0
        if reset_indices.size > 0:
            self.batch_layer_timestamps[reset_indices] = 0
        self.prev_visited_count = float(np.sum(self.batch_layer_timestamps))
        
        # Initial guidance
        self.guidance_vectors = batch_compute_simple_guidance(self.pos, self.current_goals)
        
        # Compute initial visibility for observations
        stage = self.stages[self.current_stage_idx]
        force_visible = np.zeros(self.num_envs, dtype=np.bool_)
        all_visible, _ = batch_visibility_mask(
            self.pos,
            self.target_pos,
            self.target_vel,
            np.full(self.num_envs, stage.view_radius, dtype=np.float64),
            force_visible,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
        )
        all_visible, gate_open = self._apply_visibility_gate(all_visible)
        all_visible, any_visible = self._reset_visibility_hysteresis(all_visible)
        self._any_visible = any_visible
        
        # Initialize observation targets and last known positions
        self.last_known_target_pos[:] = self.target_pos
        self.target_lost_steps[:] = 0
        self.lkp_countdown[:] = 30
        self.target_mode_flag[:] = 1.0  # tracking
        self.prev_target_mode_flag[:] = 1.0
        self.slot_assignments[reset_mask, :] = -1
        for env_id in np.where(reset_mask)[0]:
            self.slot_positions[env_id, :, :] = self.target_pos[env_id]
        if self.visibility_gate_max_ratio > 0.0:
            gate_closed = ~gate_open
            self.lkp_countdown[gate_closed] = 0
            self.target_mode_flag[gate_closed, :] = 0.0
            self.prev_target_mode_flag[gate_closed, :] = 0.0
        
        # Initialize position history properly to fix stagnation detection
        batch_init_position_history(self.position_history, self.history_idx, self.pos)
        
        # Initially all agents observe the real target
      
        for b in range(self.num_envs):
            for n in range(self.num_agents):
                self.obs_target_pos[b, n] = self.target_pos[b]
                self.prev_obs_target_pos[b, n] = self.target_pos[b]  # prev_obs_target_pos
                self.obs_target_vel[b, n] = self.target_vel[b]
                
              
              
                try:
                    if self._use_euclidean_guidance():
                        _, path_length = self._compute_euclidean_guidance(self.pos[b, n], self.target_pos[b])
                    else:
                        astar_start = self._snap_astar_start(self.pos[b, n])
                        astar_goal = self._snap_astar_goal(self.target_pos[b])
                        _, _, path_length = self.navigator.compute_direction(
                            astar_start,
                            astar_goal,
                            current_step=0,
                            current_velocity=self.vel[b, n],
                            cache_key=(b, n),
                        )
                    if path_length > 0:
                        self.path_lengths[b, n] = path_length
                        self.prev_target_dist[b, n] = path_length
                    else:
                      
                        dx = self.pos[b, n, 0] - self.obs_target_pos[b, n, 0]
                        dy = self.pos[b, n, 1] - self.obs_target_pos[b, n, 1]
                        dz = self.pos[b, n, 2] - self.obs_target_pos[b, n, 2]
                        euclidean_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                        self.path_lengths[b, n] = euclidean_dist
                        self.prev_target_dist[b, n] = euclidean_dist
                except Exception:
                  
                    dx = self.pos[b, n, 0] - self.obs_target_pos[b, n, 0]
                    dy = self.pos[b, n, 1] - self.obs_target_pos[b, n, 1]
                    dz = self.pos[b, n, 2] - self.obs_target_pos[b, n, 2]
                    euclidean_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    self.path_lengths[b, n] = euclidean_dist
                    self.prev_target_dist[b, n] = euclidean_dist
        
        # Build observations
        obs = self._build_observations()
        
        info = {
            "stage": [self.stages[self.current_stage_idx].name] * self.num_envs,
        }
        
        return obs, info
    
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one environment step.
        
        Args:
            actions: (B, N, 4) actions [vx, vy, vz, yaw_rate]
        
        Returns:
            obs: (B, N, obs_dim) observations
            rewards: (B, N) rewards
            dones: (B,) done flags
            infos: List of info dicts
        """
        actions = np.asarray(actions, dtype=np.float64)
        if self._perf_debug:
            t_step = time.perf_counter()
        B, N = self.num_envs, self.num_agents
        alive_before = self.agent_alive.copy()
        if np.any(~alive_before):
            actions[~alive_before] = 0.0
            self.prev_actions[~alive_before] = 0.0
            self.vel[~alive_before] = 0.0

        if self.action_delay_buffer is not None:
            buffer_len = self.action_delay_buffer.shape[0]
            self.action_delay_buffer[self.action_delay_cursor] = actions
            idxs = (self.action_delay_cursor - self.action_delay_steps_env) % buffer_len
            actions = self.action_delay_buffer[idxs, np.arange(B)]
            self.action_delay_cursor = (self.action_delay_cursor + 1) % buffer_len
        
        # 1. Apply action smoothing
        if self._perf_debug:
            t0 = time.perf_counter()
        alpha = self.cfg.control.smoothing_alpha
        batch_apply_action_smoothing(actions, self.prev_actions, alpha, 1e-3)

        # Clip actions
        bounds = self.cfg.control.action_bounds
        max_v = np.array([bounds["vx"], bounds["vy"], bounds["vz"], bounds["yaw_rate"]])
        actions = np.clip(actions, -max_v, max_v)

        # Compute distance to target for adaptive safety shield
        target_delta = self.pos - self.target_pos[:, None, :]
        target_dist = np.linalg.norm(target_delta, axis=2)

        # Safety shield using previous LiDAR (lidar_buffer)
        # Adaptive: reduces threshold when close to target to allow capture
        shield_flags = batch_apply_safety_shield(
            actions,
            self.lidar_buffer,
            self.cfg.shield.min_distance_m,
            self.lidar_max_range,
            target_dist,
            self.env_capture_radius,
        )
        if self._perf_debug:
            t1 = time.perf_counter()

        # Store for next step
        prev_actions_copy = self.prev_actions.copy()
        self.prev_actions = actions.copy()
        
      
        self.prev_obs_target_pos = self.obs_target_pos.copy()
        
        # Update position history for stagnation detection
        b_idx = np.arange(B)[:, None]
        n_idx = np.arange(N)[None, :]
        idx = self.history_idx
        self.position_history[b_idx, n_idx, idx] = self.pos
        self.history_idx = (idx + 1) % 5
      
        smooth_penalty = np.sum((actions - prev_actions_copy) ** 2, axis=-1)  # (B, N)
        # Guidance alignment error: 1 - max(0, cos)
        vel_norm = np.linalg.norm(self.vel, axis=-1)
        guide_norm = np.linalg.norm(self.guidance_vectors, axis=-1)
        dot = np.sum(self.vel * self.guidance_vectors, axis=-1)
        mask = (vel_norm > 1e-6) & (guide_norm > 1e-6)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_raw = np.where(mask, dot / (vel_norm * guide_norm + 1e-8), 0.0)
        cos = np.nan_to_num(cos_raw, nan=0.0, posinf=0.0, neginf=0.0)
        align_err = 1.0 - np.maximum(0.0, cos)
        self._metric_shield = np.sum(shield_flags, axis=1).astype(np.float64)
        self._metric_smoothness = np.mean(smooth_penalty, axis=1)
        self._metric_align_err = np.mean(align_err, axis=1)
        
        # 2. Physics update - agents
        if self._perf_debug:
            t2 = time.perf_counter()
        prev_pos = self.pos.copy()
        imu_accel_noise = np.zeros((B, N, 3), dtype=np.float64)
        imu_gyro_noise = np.zeros((B, N, 3), dtype=np.float64)
        if self.imu_accel_noise_std > 0.0:
            imu_accel_noise = self.rng.normal(0.0, self.imu_accel_noise_std, size=(B, N, 3))
        if self.imu_gyro_noise_std > 0.0:
            imu_gyro_noise = self.rng.normal(0.0, self.imu_gyro_noise_std, size=(B, N, 3))

        boundary_collisions = batch_update_agents_v2(
            self.pos, self.vel, self.yaw, actions,
            self.dt, self.max_speed, self.max_accel, self.max_yaw_rate,
            self.world_min, self.world_max,
            self.imu_linear_acc_body,
            self.imu_angular_vel_body,
            imu_accel_noise,
            imu_gyro_noise,
        )
        if self._perf_debug:
            t3 = time.perf_counter()
        
        # 3. Collision check
        obstacle_collisions = batch_check_collisions_v2(
            self.pos, self.occupancy_grid, self.origin, self.voxel_size
        )
        collisions_raw = np.logical_or(boundary_collisions, obstacle_collisions)
        
        # Team collision check using Numba kernel (replaces Python loops)
        if self.team_collision_radius > 0.0 and N > 1:
            team_collisions = batch_check_team_collisions(self.pos, self.team_collision_radius, alive_before)
            collisions_raw = np.logical_or(collisions_raw, team_collisions)
        collisions = np.logical_and(collisions_raw, alive_before)
        
        batch_resolve_collisions_sliding(
            self.pos,
            self.vel,
            prev_pos,
            collisions,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
            self.world_min,
            self.world_max,
        )
        if self._perf_debug:
            t4 = time.perf_counter()
        
        # 4. Physics update - targets
        self.target_rng_buffer = self.rng.normal(0.0, 1.0, size=(B, 3))
        if self.target_axis_limit_enabled:
            # Scale per-axis limits by current target speed while preserving
            # the same directional capability ratio as agents (X:Y:Z = 8:4:3).
            base_vx = max(float(self.target_max_vx), 1e-6)
            ratio_y = float(self.target_max_vy) / base_vx
            ratio_z = float(self.target_max_vz) / base_vx
            speed_ref = float(np.max(self.env_target_speeds))
            max_vx = speed_ref
            max_vy = speed_ref * ratio_y
            max_vz = speed_ref * ratio_z
        else:
            max_vx = 0.0
            max_vy = 0.0
            max_vz = 0.0
        batch_update_targets_v2(
            self.target_pos, self.target_vel, self.pos,
            self.env_behaviors, self.env_target_speeds,
            max_vx, max_vy, max_vz,
            self.dt, self.target_max_accel, self.target_bounds_min, self.target_bounds_max,
            self.target_rng_buffer,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
            self.target_avoid_radius_vox,
        )
        # Update EMAs with previous-step predictions against current realized target state.
        self._update_prediction_error_ema()
        if self._perf_debug:
            t5 = time.perf_counter()
        
        # 5. LiDAR simulation
        lidar_obs = self.lidar_buffer
        if self._lidar_step % self.lidar_update_interval == 0:
            lidar_obs = batch_simulate_lidar_v2(
                self.pos,
                self.yaw,
                self.occupancy_grid,
                self.origin,
                self.voxel_size,
                self.lidar_max_range,
                self.lidar_sector_subrays,
                self.lidar_sector_spread_deg,
            )
            if self.lidar_dropout_prob > 0.0:
                drop_mask = self.rng.random(lidar_obs.shape) < self.lidar_dropout_prob
                lidar_obs = np.where(drop_mask, 1.0, lidar_obs)
            if self.lidar_noise_std > 0.0:
                noise = self.rng.normal(0.0, self.lidar_noise_std, size=lidar_obs.shape)
                lidar_obs = np.clip(lidar_obs + noise, 0.0, 1.0)
            self.lidar_buffer = lidar_obs
        self._lidar_step += 1
        if self._perf_debug:
            t6 = time.perf_counter()

        # 6. Collision rearm + shield/frontier/exploration bookkeeping
        min_lidar = np.min(lidar_obs, axis=2)
        min_dist_m = min_lidar * self.lidar_max_range
        collision_impact_flags = collisions & alive_before & self.collision_armed

        dead_mask = ~alive_before
        if np.any(dead_mask):
            self.collision_armed[dead_mask] = False
            self.collision_rearm_count[dead_mask] = 0

        collided_alive = collisions & alive_before
        if np.any(collided_alive):
            self.collision_armed[collided_alive] = False
            self.collision_rearm_count[collided_alive] = 0

        no_collision_alive = alive_before & ~collisions
        safe_mask = no_collision_alive & (min_dist_m > 0.5)
        unsafe_mask = no_collision_alive & ~safe_mask
        if np.any(unsafe_mask):
            self.collision_rearm_count[unsafe_mask] = 0
        if np.any(safe_mask):
            self.collision_rearm_count[safe_mask] += 1
            rearm_ready = safe_mask & (self.collision_rearm_count >= 3)
            if np.any(rearm_ready):
                self.collision_armed[rearm_ready] = True
                self.collision_rearm_count[rearm_ready] = 0
        self.frontier_deltas[:, :] = 0.0
        exploration_deltas = np.zeros((B, N), dtype=np.float64)
        if np.any(self.steps % self.explore_update_interval == 0):
            new_voxels = batch_update_exploration_grid_2p5d(
                self.pos,
                self.batch_layer_timestamps,
                self.steps,
                self.voxel_map.origin,
                self.voxel_map.voxel_size,
                self.layer_z_bounds,
                self.layer_sample_offsets,
                self.static_layer_occ,
                self.steps_to_reexplore,
            )
            exploration_deltas = new_voxels.astype(np.float64)
        if self._perf_debug:
            t7 = time.perf_counter()

        # 7. Navigation update (staggered per-env to smooth CPU load)
        # Instead of all envs updating at once, each env updates at different frames
        # stagger_offset = b % nav_update_interval, update when steps[b] % interval == offset
        # This spreads ~3.2 envs per frame (32/10) instead of 32 envs every 10 frames
        self._update_navigation_staggered(prev_pos)
        if self._perf_debug:
            t8 = time.perf_counter()

        # 8. Compute rewards
      
        is_switching_frame = (self.target_mode_flag != self.prev_target_mode_flag)
        
      
        obs_target_jump_threshold = float(self.potential_unstable_obs_jump_m)
        obs_target_changed = np.linalg.norm(
            self.obs_target_pos - self.prev_obs_target_pos, axis=2) > obs_target_jump_threshold
        
      
      
        path_length_jump_threshold = float(self.potential_unstable_path_jump_m)
        path_length_changed = np.abs(self.path_lengths - self.prev_target_dist) > path_length_jump_threshold
        
      
        is_potential_unstable = is_switching_frame | obs_target_changed | path_length_changed
        
      
      
        unstable_path = is_potential_unstable & path_length_changed
        if np.any(unstable_path):
            self.prev_target_dist[unstable_path] = self.path_lengths[unstable_path]

        unstable_non_path = is_potential_unstable & ~path_length_changed
        unstable_tracking = unstable_non_path & (self.target_mode_flag > 0.5)
        if np.any(unstable_tracking):
            dist_to_obs = np.linalg.norm(self.pos - self.obs_target_pos, axis=2)
            self.prev_target_dist[unstable_tracking] = dist_to_obs[unstable_tracking]

        unstable_search = unstable_non_path & (self.target_mode_flag <= 0.5)
        if np.any(unstable_search):
            dist_to_goal = np.linalg.norm(self.current_goals - self.pos, axis=2)
            self.prev_frontier_dist[unstable_search] = dist_to_goal[unstable_search]
        
        rewards, self.prev_target_dist, self.prev_frontier_dist, captured, reward_breakdown_arr = batch_compute_rewards_v2(
            self.pos, self.vel, self.target_pos, self.target_vel,
            self.obs_target_pos,
            self.path_lengths,
            self.prev_target_dist,
            self.prev_frontier_dist,
            self.lidar_buffer, self.guidance_vectors,
            actions, prev_actions_copy,
            collisions,
            alive_before,
            collision_impact_flags,
            shield_flags,
            exploration_deltas,
            self.frontier_deltas,
            self.reward_core_coefs,
            self.env_capture_radius,
            self.lidar_max_range,
            self.frontier_reach_dist,
            self.lidar_safe_distance_m,
            self.search_speed_floor_mps,
            self.direction_gate_active_radius_m,
            self.target_mode_flag,
            is_potential_unstable,
            self.position_history,
            self.current_goals,
            self.world_min,
            self.world_max,
        )
        if self._perf_debug:
            t9 = time.perf_counter()
        
        # Stage-agnostic collision termination handled by dones logic.
        # Build reward breakdown dicts only when full info is requested (eval/debug).
        reward_breakdown = None
        if not self._minimal_train_info:
            reward_breakdown = []
            for b in range(B):
                # breakdown order matches batch_compute_rewards_v2:
                # [0:step_cost, 1:progress_gain, 2:exploration_gain, 3:proximity_cost,
                #  4:collision_cost, 5:direction_gain, 6:control_cost, 7:capture_gain,
                #  8:capture_quality_gain, 9:reward_total]
                vals = reward_breakdown_arr[b]
                rb = {
                    "step_cost": float(vals[0]),
                    "progress_gain": float(vals[1]),
                    "exploration_gain": float(vals[2]),
                    "proximity_cost": float(vals[3]),
                    "collision_cost": float(vals[4]),
                    "direction_gain": float(vals[5]),
                    "control_cost": float(vals[6]),
                    "capture_gain": float(vals[7]),
                    "capture_quality_gain": float(vals[8]),
                    "reward_total": float(vals[9]),
                }
                reward_breakdown.append(rb)

        # 9. Check done conditions
        raw_captured = captured.copy()
        self.steps += 1
        timeout = self.steps >= self.max_steps
        collision_done = np.any(collisions, axis=1)
        clean_capture = np.logical_and(raw_captured, ~collision_done)
        success_capture = raw_captured
        delta_all = self.pos - self.target_pos[:, None, :]
        d3_all = np.linalg.norm(delta_all, axis=2)
        abs_dz_all = np.abs(delta_all[:, :, 2])
        alive_counts = np.sum(alive_before, axis=1).astype(np.float64)
        alive_env = alive_counts > 0.0

        masked_d3 = np.where(alive_before, d3_all, np.inf)
        team_target_min_dist_3d = np.where(alive_env, np.min(masked_d3, axis=1), 1e9).astype(np.float64)

        team_target_mean_abs_dz = np.sum(abs_dz_all * alive_before, axis=1) / np.maximum(alive_counts, 1.0)
        team_target_mean_abs_dz = np.where(alive_env, team_target_mean_abs_dz, 0.0).astype(np.float64)

        diag_interval = max(1, int(getattr(self, "_diag_metrics_interval_steps", 1)))
        compute_heavy_diag = (diag_interval <= 1) or ((int(self._lidar_step) % diag_interval) == 0)
        if compute_heavy_diag:
            d3_nan = np.where(alive_before, d3_all, np.nan)
            if np.any(~alive_env):
                d3_nan[~alive_env, 0] = 0.0
            dist_to_target_p50 = np.nanpercentile(d3_nan, 50, axis=1).astype(np.float64)
            dist_to_target_p90 = np.nanpercentile(d3_nan, 90, axis=1).astype(np.float64)
            dist_to_target_p50[~alive_env] = 0.0
            dist_to_target_p90[~alive_env] = 0.0

            # P2 diagnostic: far-and-idle behavior ratio.
            to_target_all = self.target_pos[:, None, :] - self.pos
            d_safe_all = np.maximum(d3_all[..., None], 1e-6)
            target_dir_all = to_target_all / d_safe_all
            rel_vel_all = self.vel - self.target_vel[:, None, :]
            closing_speed_all = np.sum(rel_vel_all * target_dir_all, axis=2)
            far_threshold = np.maximum(2.5 * self.env_capture_radius, 25.0)[:, None]
            far_idle = alive_before & (d3_all > far_threshold) & (closing_speed_all < 0.5)
            far_idle_ratio = np.sum(far_idle, axis=1).astype(np.float64) / np.maximum(alive_counts, 1.0)
            far_idle_ratio = np.where(alive_env, far_idle_ratio, 0.0).astype(np.float64)

            # P2 diagnostic: capture contribution concentration (top-1 share).
            capture_contribution = np.zeros(B, dtype=np.float64)
            if np.any(raw_captured):
                contrib_radius = np.maximum(3.0 * self.env_capture_radius, 30.0)[:, None]
                closenorm = np.clip(closing_speed_all / 4.0, 0.0, 1.0)
                prox = np.clip((contrib_radius - d3_all) / np.maximum(contrib_radius, 1e-6), 0.0, 1.0)
                raw = (0.05 + 0.70 * prox + 0.25 * closenorm) * alive_before.astype(np.float64)
                raw += 0.30 * ((d3_all <= self.env_capture_radius[:, None]) & alive_before)
                raw_sum = np.sum(raw, axis=1)
                max_share = np.max(raw / np.maximum(raw_sum[:, None], 1e-8), axis=1)
                fallback_share = 1.0 / np.maximum(alive_counts, 1.0)
                capture_val = np.where(raw_sum > 1e-8, max_share, fallback_share)
                capture_contribution = np.where(raw_captured, capture_val, 0.0).astype(np.float64)
        else:
            dist_to_target_p50 = self._metric_dist_p50.copy()
            dist_to_target_p90 = self._metric_dist_p90.copy()
            far_idle_ratio = self._metric_far_idle_ratio.copy()
            capture_contribution = self._metric_capture_contribution.copy()

        self._metric_far_idle_ratio[:] = far_idle_ratio
        self._metric_capture_contribution[:] = capture_contribution
        self._metric_dist_p50[:] = dist_to_target_p50
        self._metric_dist_p90[:] = dist_to_target_p90
        dones = np.logical_or(success_capture, timeout)
        dones = np.logical_or(dones, collision_done)
        done_reason = np.full(B, "running", dtype=object)
        done_reason[timeout] = "timeout"
        done_reason[success_capture] = "capture"
        done_reason[collision_done] = "collision"

        # Preserve terminal observations for done envs before auto-reset.
        terminal_obs = None
        terminal_agent_pos = None
        terminal_target_pos = None
        terminal_target_vel = None
        # Terminal snapshots are only needed for value bootstrap overrides on
        # truncated endings (timeout / collision teammate truncation). Capture
        # episodes are true terminal and do not need terminal observations.
        need_terminal_snapshots = bool(np.any(timeout) or np.any(collision_done))
        if np.any(dones):
            terminal_agent_pos = self.pos.copy()
            terminal_target_pos = self.target_pos.copy()
            terminal_target_vel = self.target_vel.copy()
        if np.any(dones) and need_terminal_snapshots:
            done_env_ids = np.where(dones)[0]
            terminal_obs = np.zeros((B, N, self.get_obs_dim()), dtype=np.float32)
            terminal_obs[done_env_ids] = self._build_observations_for_envs(done_env_ids).astype(np.float32, copy=False)

        # 10. Auto-reset done environments
        if np.any(dones):
            self._generate_spawn_rng()
            batch_respawn_envs_v2(
                dones,
                self.pos, self.vel, self.yaw, self.prev_actions,
                self.target_pos, self.target_vel,
                self.steps, self.prev_target_dist, self.current_goals,
                self.occupancy_grid_astar, self.origin, self.voxel_size,
                self.spawn_min, self.spawn_max, self.min_spawn_separation,
                self.target_spawn_min_dist, self.target_spawn_max_dist,
                self.spawn_rng_buffer, self.yaw_rng_buffer,
            )
            self.prev_frontier_dist[dones] = 0.0
            self.batch_layer_timestamps[dones] = 0
            self.prev_visited_count = float(np.sum(self.batch_layer_timestamps))
            self.agent_alive[dones] = True
            self.slot_assignments[dones, :] = -1
            for env_id in np.where(dones)[0]:
                self.slot_positions[env_id, :, :] = self.target_pos[env_id]
            self.imu_linear_acc_body[dones, :, :] = 0.0
            self.imu_angular_vel_body[dones, :, :] = 0.0
            self.last_pred_valid[dones] = False
            self.last_pred_alpha[dones] = 0.0
            self.error_phys_ema[dones] = self.hybrid_predictor_init_error
            self.error_nn_ema[dones] = self.hybrid_predictor_init_error
            
            # Clear A* cache for reset environments to prevent memory growth
            done_indices = np.where(dones)[0]
            for env_id in done_indices:
                self.navigator.clear_cache_for_env(int(env_id))
            
            # Initialize position history for respawned environments
            for b in range(self.num_envs):
                if dones[b]:
                    batch_init_position_history(
                        self.position_history[b:b+1], 
                        self.history_idx[b:b+1], 
                        self.pos[b:b+1]
                    )
            
            # Update guidance for reset envs
            self.guidance_vectors = batch_compute_simple_guidance(self.pos, self.current_goals)
        if self._perf_debug:
            t10 = time.perf_counter()

        # 11. Build observations
        obs = self._build_observations()
        if self._perf_debug:
            t11 = time.perf_counter()

        # 12. Build info dicts
        infos = self._build_infos(
            collisions,
            success_capture,
            dones,
            timeout,
            done_reason,
            reward_breakdown,
            reward_breakdown_arr,
            raw_captured,
            clean_capture,
            team_target_min_dist_3d,
            team_target_mean_abs_dz,
            far_idle_ratio,
            capture_contribution,
            dist_to_target_p50,
            dist_to_target_p90,
            terminal_obs,
            terminal_agent_pos,
            terminal_target_pos,
            terminal_target_vel,
        )
        if self._perf_debug:
            t12 = time.perf_counter()
            self._perf_last = {
                "pre": t1 - t0,
                "agents": t3 - t2,
                "collisions": t4 - t3,
                "targets": t5 - t4,
                "lidar": t6 - t5,
                "explore": t7 - t6,
                "nav": t8 - t7,
                "rewards": t9 - t8,
                "respawn": t10 - t9,
                "obs": t11 - t10,
                "infos": t12 - t11,
                "total": t12 - t_step,
            }
            if self._perf_nav:
                self._perf_last.update(self._perf_nav)

        # Consume one-step NN predictions to avoid stale reuse.
        self._pending_nn_target_rel_valid = False
        
        return obs, rewards, dones, infos

    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _generate_spawn_rng(self) -> None:
        """Generate random numbers for spawning."""
        B, N = self.num_envs, self.num_agents
        max_attempts = self.spawn_rng_buffer.shape[2]
        self.spawn_rng_buffer = self.rng.random((B, N + 1, max_attempts, 3))
        self.yaw_rng_buffer = self.rng.random((B, N))
    
    def _update_navigation_staggered(self, prev_pos: np.ndarray) -> None:
        """
        Staggered navigation update - spreads computation across frames.
        
        Instead of updating all envs every nav_update_interval steps,
        each env updates at a different offset: env b updates when
        steps[b] % nav_update_interval == b % nav_update_interval.
        
        This spreads ~3.2 envs per frame (32/10) for smooth CPU load.
        """
        B, N = self.num_envs, self.num_agents
        
        # Compute which envs need full navigation update this frame
        stagger_offsets = np.arange(B, dtype=np.int32) % self.nav_update_interval
        needs_nav_update = (self.steps % self.nav_update_interval) == stagger_offsets
        
        # Always update target tracking for all envs (cheap operation)
        stage = self.stages[self.current_stage_idx]
        force_visible = np.zeros(B, dtype=np.bool_)
        all_visible, _ = batch_visibility_mask(
            self.pos,
            self.target_pos,
            self.target_vel,
            np.full(B, stage.view_radius, dtype=np.float64),
            force_visible,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
        )
        all_visible, gate_open = self._apply_visibility_gate(all_visible)
        all_visible, any_visible = self._apply_visibility_hysteresis(all_visible)
        self._any_visible = any_visible
        
        # Save previous mode for all envs
        self.prev_target_mode_flag[:] = self.target_mode_flag
        
        # Vectorized team_has_target computation
        team_has_target = np.any(all_visible, axis=1)  # (B,)
        self.team_has_target = team_has_target
        self.team_has_target = team_has_target
        
        # Batch check LKP reached using Numba kernel
        lkp_breakout_dist = self.cfg.environment.lkp_breakout_dist_m
        lkp_reached = batch_check_lkp_reached(self.pos, self.last_known_target_pos, lkp_breakout_dist)
        
        # Vectorized update for environments where team sees target
        visible_envs = team_has_target
        self.last_known_target_pos[visible_envs] = self.target_pos[visible_envs]
        self.target_lost_steps[visible_envs] = 0
        self.lkp_countdown[visible_envs] = self.cfg.environment.lkp_countdown_steps
        self.target_mode_flag[visible_envs, :] = 1.0
        
        # Update for environments where team lost target
        lost_envs = ~team_has_target
        self.target_lost_steps[lost_envs] += 1
        
        # LKP countdown logic (vectorized)
        in_lkp_phase = lost_envs & (self.lkp_countdown > 0)
        self.lkp_countdown[in_lkp_phase] -= 1
        
        # LKP breakout: if any agent reached LKP, force countdown to 0
        lkp_breakout_envs = in_lkp_phase & lkp_reached
        self.lkp_countdown[lkp_breakout_envs] = 0
        self.target_mode_flag[lkp_breakout_envs, :] = 0.0
        
        # Remaining LKP phase envs stay in tracking mode
        still_in_lkp = in_lkp_phase & ~lkp_reached
        self.target_mode_flag[still_in_lkp, :] = 1.0
        
        # LKP timeout: switch to search mode
        lkp_timeout_envs = lost_envs & (self.lkp_countdown <= 0)
        self.target_mode_flag[lkp_timeout_envs, :] = 0.0
        
        # Always smooth path lengths for all envs (cheap Numba kernel)
        batch_update_path_lengths(self.path_lengths, self.pos, prev_pos)
        
        # Batch update obs_target using Numba kernel (replaces Python loops)
        batch_update_obs_targets(
            self.obs_target_pos,
            self.obs_target_vel,
            self.target_pos,
            self.target_vel,
            self.last_known_target_pos,
            self.current_goals,
            self.target_mode_flag,
            team_has_target,
            needs_nav_update,
            stage.guidance_enabled,
        )
        
        # For envs doing full nav update: run A* and frontier allocation
        nav_env_ids = np.where(needs_nav_update)[0]
        if len(nav_env_ids) > 0:
            self._update_navigation_for_envs(nav_env_ids, all_visible, team_has_target)

        visible_tracking_mask = (self.target_mode_flag > 0.5) & team_has_target[:, None]
        self._apply_target_observation_noise(visible_tracking_mask)
    
    def _update_navigation_for_envs(
        self, 
        env_ids: np.ndarray, 
        all_visible: np.ndarray,
        team_has_target: np.ndarray
    ) -> None:
        """
        Run full navigation update (A* + frontier allocation) for specific envs.
        Called by staggered update for ~3.2 envs per frame.
        """
        N = self.num_agents
        agent_positions = self.pos
        agent_yaws = self.yaw
        global_targets = self.target_pos
        stage = self.stages[self.current_stage_idx]
        pred_seconds = self._prediction_horizon_seconds()
        pred_phys = self._predict_target_collision_aware(
            target_pos=global_targets[env_ids],
            target_vel=self.target_vel[env_ids],
            pred_seconds=pred_seconds,
        )
        pred_nn = self._decode_nn_target_predictions(env_ids)
        predicted_targets, pred_alpha = self._blend_hybrid_target_prediction(env_ids, pred_phys, pred_nn)
        self.last_pred_phys[env_ids] = pred_phys
        self.last_pred_nn[env_ids] = pred_nn if pred_nn is not None else pred_phys
        self.last_pred_valid[env_ids] = True
        self.last_pred_alpha[env_ids] = pred_alpha
        
        if self._perf_debug:
            nav_astar = 0.0
            nav_path_len = 0.0
            nav_frontier_envs = 0.0
            nav_visible = 0.0
            nav_total = float(len(env_ids) * N)
        
        for env_offset, b in enumerate(env_ids):
            allocator = self.allocators[b]
            
            # Reuse scratch buffers to avoid per-env exploration mask allocations.
            self.voxel_map.visits = self._fill_exploration_mask(
                b,
                self._exploration_conf2d_scratch,
                self._exploration_mask_scratch,
            )
            
            dists = np.linalg.norm(agent_positions[b] - global_targets[b], axis=1)
            visible_mask = all_visible[b]
            has_target = team_has_target[b]
            slot_goals = None
            slot_assign = None
            if has_target and stage.guidance_enabled:
                pred_center = predicted_targets[env_offset:env_offset + 1]
                # Stage-dynamic slot radius: capture_radius + offset.
                slot_radius = max(
                    float(self.env_capture_radius[b]) + self.tactical_slot_radius_offset,
                    0.5,
                )
                slots = self._generate_slots(pred_center, slot_radius)
                slots = self._rectify_slots(slots, pred_center)
                tracking_alive = self.agent_alive[b] & (self.target_mode_flag[b] > 0.5)
                slot_goals, slot_assign = self._assign_slots_hungarian(
                    agent_positions=agent_positions[b],
                    slots=slots[0],
                    active_mask=tracking_alive,
                    fallback_target=predicted_targets[env_offset],
                )
                self.slot_positions[b, :, :] = slots[0]
                self.slot_assignments[b, :] = slot_assign
            else:
                self.slot_positions[b, :, :] = predicted_targets[env_offset]
                self.slot_assignments[b, :] = -1
            
            if self._perf_debug:
                nav_visible += float(np.sum(visible_mask))
            
            # Frontier allocation
            headings = np.zeros((N, 3))
            headings[:, 0] = np.cos(agent_yaws[b])
            headings[:, 1] = np.sin(agent_yaws[b])

            frontier_goals = {}
            has_search_agents = np.any(self.target_mode_flag[b] <= 0.5)
            if stage.guidance_enabled and has_search_agents:
                preferred_layer_z = self._assign_preferred_layers(b)
                allocator.set_layer_preferences(preferred_layer_z, self.layer_preference_weight)
                allocator.set_layer_constraints(self.layer_z_bounds, self.preferred_layers[b])
                allocator.set_layer_frontiers(self._build_layer_frontiers(b))
                for n in range(N):
                    if self.target_mode_flag[b, n] < 0.5 and self.prev_target_mode_flag[b, n] > 0.5:
                        allocator.mark_agent_lost_target(n, True)
                
                frontier_goals = allocator.allocate(
                    agent_positions[b], headings, self.voxel_map, int(self.steps[b]),
                    lkp_pos=self.last_known_target_pos[b],
                    target_lost_steps=int(self.target_lost_steps[b])
                )
                for n in range(N):
                    if self.target_mode_flag[b, n] > 0.5:
                        continue
                    if frontier_goals.get(n) is None:
                        layer_idx = int(self.preferred_layers[b, n])
                        layer_idx = int(np.clip(layer_idx, 0, 2))
                        anchor = np.array(
                            [agent_positions[b, n, 0], agent_positions[b, n, 1], self.layer_centers[layer_idx]],
                            dtype=np.float64,
                        )
                        frontier_goals[n] = anchor
                if self._perf_debug:
                    nav_frontier_envs += 1.0
            
            if stage.guidance_enabled:
                frontier_goals_arr = np.repeat(global_targets[b][None, :], N, axis=0)
                if frontier_goals:
                    for n_idx, goal in frontier_goals.items():
                        if goal is not None:
                            frontier_goals_arr[n_idx] = goal
                selected_goals = frontier_goals_arr.copy()
                selected_goals[visible_mask] = global_targets[b]
            else:
                selected_goals = None

            # Debug capture for visualization (disabled in training fast-path).
            if self._store_frontier_debug:
                self._frontier_debug[b] = {
                    "candidates": allocator._last_candidates.copy(),
                    "per_agent_candidates": {k: v.copy() for k, v in allocator._per_agent_candidates.items()},
                    "frontier_goals": {k: (None if v is None else v.copy()) for k, v in frontier_goals.items()},
                    "selected_goals": None if selected_goals is None else selected_goals.copy(),
                    "visible_mask": visible_mask.copy(),
                }
            else:
                self._frontier_debug[b] = None
            self._update_layer_state(b)
            
            # Frontier bonus bookkeeping
            if stage.guidance_enabled and selected_goals is not None:
                for n in range(N):
                    is_searching = self.target_mode_flag[b, n] <= 0.5
                    if not is_searching:
                        self.prev_frontier_mask[b, n] = False
                        continue
                    goal = selected_goals[n]
                    dist_to_target = np.linalg.norm(goal - global_targets[b])
                    is_frontier = dist_to_target > 1e-6
                    if is_frontier:
                        prev_goal = self.prev_frontier_goals[b, n]
                        moved = np.linalg.norm(goal - prev_goal) > self.voxel_size * 0.5
                        if not self.prev_frontier_mask[b, n] or moved:
                            self.frontier_deltas[b, n] = 1.0
                        self.prev_frontier_goals[b, n] = goal
                        self.prev_frontier_mask[b, n] = True
                    else:
                        self.prev_frontier_mask[b, n] = False
            
            # Per-agent navigation
            for n in range(N):
                final_goal = global_targets[b]
                
                if self.target_mode_flag[b, n] > 0.5:  # Tracking mode
                    if has_target:
                        if slot_goals is not None:
                            final_goal = slot_goals[n].copy()
                        else:
                            final_goal = predicted_targets[env_offset].copy()
                        obs_target_pos = global_targets[b]
                        obs_target_vel = self.target_vel[b]
                    else:
                        obs_target_pos = self.last_known_target_pos[b]
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = self.last_known_target_pos[b]
                else:  # Searching mode
                    if stage.guidance_enabled and selected_goals is not None:
                        raw_goal = selected_goals[n].copy()
                        layer_idx = self._layer_index_from_z(raw_goal[2])
                        target_z = self.layer_centers[layer_idx]
                        blend = max(0.0, min(1.0, self.layer_goal_blend))
                        blended_z = self.pos[b, n, 2] + blend * (target_z - self.pos[b, n, 2])
                        blended_z = float(np.clip(blended_z, self.world_min[2], self.world_max[2]))
                        raw_goal[2] = blended_z
                        obs_target_pos = raw_goal
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = raw_goal
                    else:
                        obs_target_pos = self.last_known_target_pos[b]
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = self.last_known_target_pos[b]
                
                self.obs_target_pos[b, n] = obs_target_pos
                self.obs_target_vel[b, n] = obs_target_vel
                
                if stage.guidance_enabled:
                    if visible_mask[n]:
                        # Do not add legacy radial offsets in tactical-slot tracking mode.
                        use_legacy_offset = not (
                            has_target and slot_goals is not None and self.target_mode_flag[b, n] > 0.5
                        )
                        if use_legacy_offset:
                            capture_r = max(float(self.env_capture_radius[b]), 1.0)
                            offset_radius = np.clip(
                                capture_r * self.goal_offset_capture_mult,
                                self.goal_offset_min_radius,
                                self.goal_offset_max_radius,
                            )
                            scale = min(
                                1.0,
                                max(self.goal_offset_scale_floor, dists[n] / self.offset_decay_distance),
                            )
                            final_goal = final_goal + self.goal_offsets[n] * (offset_radius * scale)
                    
                    direct_dist = dists[n]
                    if self._use_euclidean_guidance():
                        direction, path_len = self._compute_euclidean_guidance(agent_positions[b, n], final_goal)
                        path = None
                    else:
                        astar_start = self._snap_astar_start(agent_positions[b, n])
                        astar_goal = self._snap_astar_goal(final_goal)
                        direction, path, path_len = self.navigator.compute_direction(
                            start_position=astar_start,
                            goal_position=astar_goal,
                            current_step=int(self.steps[b]),
                            current_velocity=self.vel[b, n],
                            smooth=False,
                            cache_key=(int(b), int(n)),
                        )
                        if not np.allclose(astar_start, agent_positions[b, n]):
                            if path is not None and len(path) >= 2:
                                anchor = path[1]
                            elif path is not None and len(path) == 1:
                                anchor = path[0]
                            else:
                                anchor = astar_start
                            delta = anchor - agent_positions[b, n]
                            norm = np.linalg.norm(delta)
                            if norm > 1e-6:
                                direction = delta / norm
                        if self._perf_debug:
                            nav_astar += 1.0
                            nav_path_len += float(len(path))
                        if visible_mask[n] and path_len <= 0.0:
                            direction, path_len = self._compute_euclidean_guidance(agent_positions[b, n], final_goal)
                    
                    self.guidance_vectors[b, n] = direction
                    self.current_goals[b, n] = final_goal
                    if path_len <= 0.0:
                        dx = final_goal[0] - agent_positions[b, n, 0]
                        dy = final_goal[1] - agent_positions[b, n, 1]
                        dz = final_goal[2] - agent_positions[b, n, 2]
                        path_len = np.sqrt(dx*dx + dy*dy + dz*dz)
                    self.path_lengths[b, n] = path_len
                else:
                    self.guidance_vectors[b, n] = 0.0
                    dx = final_goal[0] - agent_positions[b, n, 0]
                    dy = final_goal[1] - agent_positions[b, n, 1]
                    dz = final_goal[2] - agent_positions[b, n, 2]
                    self.path_lengths[b, n] = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if self._perf_debug:
            visible_frac = nav_visible / nav_total if nav_total > 0 else 0.0
            self._perf_nav = {
                "nav_astar": nav_astar,
                "nav_path_len": nav_path_len,
                "nav_frontier_envs": nav_frontier_envs,
                "nav_visible_frac": visible_frac,
            }
    
    def _update_navigation(self) -> None:
        """
        Update navigation goals and guidance vectors.
        Run every 'nav_update_interval' steps.
        Uses Python loop over envs (acceptable for 10Hz interval).
        """
        B, N = self.num_envs, self.num_agents
        
      
        agent_positions = self.pos
        agent_yaws = self.yaw
        global_targets = self.target_pos
        stage = self.stages[self.current_stage_idx]
        pred_seconds = self._prediction_horizon_seconds()
        pred_phys = self._predict_target_collision_aware(
            target_pos=global_targets,
            target_vel=self.target_vel,
            pred_seconds=pred_seconds,
        )
        all_env_ids = np.arange(B, dtype=np.int32)
        pred_nn = self._decode_nn_target_predictions(all_env_ids)
        predicted_targets, pred_alpha = self._blend_hybrid_target_prediction(all_env_ids, pred_phys, pred_nn)
        self.last_pred_phys[all_env_ids] = pred_phys
        self.last_pred_nn[all_env_ids] = pred_nn if pred_nn is not None else pred_phys
        self.last_pred_valid[all_env_ids] = True
        self.last_pred_alpha[all_env_ids] = pred_alpha
        force_visible = np.zeros(B, dtype=np.bool_)
        # Batch compute visibility mask and any_visible flag (single LOS pass)
        all_visible, _ = batch_visibility_mask(
            agent_positions,
            global_targets,
            self.target_vel,
            np.full(B, stage.view_radius, dtype=np.float64),
            force_visible,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
        )
        all_visible, gate_open = self._apply_visibility_gate(all_visible)
        all_visible, any_visible = self._apply_visibility_hysteresis(all_visible)
        self._any_visible = any_visible
        
        if self._perf_debug:
            nav_astar = 0.0
            nav_path_len = 0.0
            nav_frontier_envs = 0.0
            nav_visible = 0.0
            nav_total = float(B * N)

      
        self.prev_target_mode_flag[:] = self.target_mode_flag
        
        for b in range(B):
            stage = self.stages[self.current_stage_idx] #  env 
            allocator = self.allocators[b]
            
          
          
            self.voxel_map.visits = self._fill_exploration_mask(
                b,
                self._exploration_conf2d_scratch,
                self._exploration_mask_scratch,
            )
            
          
          
            dists = np.linalg.norm(agent_positions[b] - global_targets[b], axis=1)
            visible_mask = all_visible[b]
            team_has_target = np.any(visible_mask)
            
          
            if team_has_target:
              
                self.last_known_target_pos[b] = global_targets[b].copy()
                self.target_lost_steps[b] = 0
                self.lkp_countdown[b] = self.cfg.environment.lkp_countdown_steps
                self.target_mode_flag[b, :] = 1.0
            else:
              
                self.target_lost_steps[b] += 1
                
                if self.lkp_countdown[b] > 0:
                  
                    self.lkp_countdown[b] -= 1
                    self.target_mode_flag[b, :] = 1.0
                else:
                  
                    self.target_mode_flag[b, :] = 0.0
            if self.visibility_gate_max_ratio > 0.0 and not gate_open[b]:
                self.lkp_countdown[b] = 0
                self.target_mode_flag[b, :] = 0.0
            
            if self._perf_debug:
                nav_visible += float(np.sum(visible_mask))
                
            #  Frontier Allocator 
            #  headings (N, 3)
            headings = np.zeros((N, 3))
            headings[:, 0] = np.cos(agent_yaws[b])
            headings[:, 1] = np.sin(agent_yaws[b])
            
            #  ""  "Guidance Enabled" 
            # (50) allocate
            #  10 
            
            frontier_goals = {}
            #  allocation
            has_search_agents = np.any(self.target_mode_flag[b] <= 0.5)
            if stage.guidance_enabled and has_search_agents:
                # Mark agents that just switched from Tracking to Searching (role protection)
                for n in range(N):
                    if self.target_mode_flag[b, n] < 0.5 and self.prev_target_mode_flag[b, n] > 0.5:
                        allocator.mark_agent_lost_target(n, True)
                
                frontier_goals = allocator.allocate(
                    agent_positions[b], headings, self.voxel_map, int(self.steps[b]),
                    lkp_pos=self.last_known_target_pos[b],
                    target_lost_steps=int(self.target_lost_steps[b])
                )
                if self._perf_debug:
                    nav_frontier_envs += 1.0
            
            if stage.guidance_enabled:
                frontier_goals_arr = np.repeat(global_targets[b][None, :], N, axis=0)
                if frontier_goals:
                    for n_idx, goal in frontier_goals.items():
                        if goal is not None:
                            frontier_goals_arr[n_idx] = goal
                selected_goals = frontier_goals_arr.copy()
                selected_goals[visible_mask] = global_targets[b]
            else:
                selected_goals = None

            # Frontier bonus bookkeeping: reward when a new frontier goal is assigned
            if stage.guidance_enabled and selected_goals is not None:
                for n in range(N):
                    is_searching = self.target_mode_flag[b, n] <= 0.5
                    if not is_searching:
                        self.prev_frontier_mask[b, n] = False
                        continue
                    goal = selected_goals[n]
                    dist_to_target = np.linalg.norm(goal - global_targets[b])
                    is_frontier = dist_to_target > 1e-6
                    if is_frontier:
                        prev_goal = self.prev_frontier_goals[b, n]
                        moved = np.linalg.norm(goal - prev_goal) > self.voxel_size * 0.5
                        if not self.prev_frontier_mask[b, n] or moved:
                            self.frontier_deltas[b, n] = 1.0
                        self.prev_frontier_goals[b, n] = goal
                        self.prev_frontier_mask[b, n] = True
                    else:
                        self.prev_frontier_mask[b, n] = False
            
            for n in range(N):
              
                final_goal = global_targets[b] # 
                
              
              
                
                if self.target_mode_flag[b, n] > 0.5:  # Tracking
                    if team_has_target:
                        safe_target = predicted_targets[b].copy()
                        
                      
                        try:
                            start_idx = self.voxel_map.pos_to_idx(agent_positions[b, n])
                            if (0 <= start_idx[0] < self.occupancy_grid.shape[0] and
                                0 <= start_idx[1] < self.occupancy_grid.shape[1] and
                                0 <= start_idx[2] < self.occupancy_grid.shape[2]):
                                if self.occupancy_grid[start_idx[0], start_idx[1], start_idx[2]] > 0.5:
                                  
                                    if self._perf_debug:
                                        print(f"Warning: Agent {n} in env {b} is inside obstacle!")
                        except Exception:
                            pass
                        
                        final_goal = safe_target
                        obs_target_pos = global_targets[b]  # 
                        obs_target_vel = self.target_vel[b]
                    else:
                      
                        obs_target_pos = self.last_known_target_pos[b]
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = self.last_known_target_pos[b]
                else:  # Searching
                  
                    if stage.guidance_enabled and selected_goals is not None:
                        obs_target_pos = selected_goals[n]
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = selected_goals[n]
                    else:
                        obs_target_pos = self.last_known_target_pos[b]
                        obs_target_vel = np.zeros(3, dtype=np.float64)
                        final_goal = self.last_known_target_pos[b]
                
              
                self.obs_target_pos[b, n] = obs_target_pos
                self.obs_target_vel[b, n] = obs_target_vel
                
                if stage.guidance_enabled:
                    if visible_mask[n]:
                      
                        capture_r = max(float(self.env_capture_radius[b]), 1.0)
                        offset_radius = np.clip(
                            capture_r * self.goal_offset_capture_mult,
                            self.goal_offset_min_radius,
                            self.goal_offset_max_radius,
                        )
                        scale = min(
                            1.0,
                            max(self.goal_offset_scale_floor, dists[n] / self.offset_decay_distance),
                        )
                        final_goal = final_goal + self.goal_offsets[n] * (offset_radius * scale)
                    
                  
                    direct_dist = dists[n]

                    if self._use_euclidean_guidance():
                        direction, path_len = self._compute_euclidean_guidance(agent_positions[b, n], final_goal)
                        path = None
                    elif visible_mask[n]:
                        direction, path_len = self._compute_euclidean_guidance(agent_positions[b, n], final_goal)
                        path = [agent_positions[b, n], final_goal]
                    else:
                        astar_start = self._snap_astar_start(agent_positions[b, n])
                        astar_goal = self._snap_astar_goal(final_goal)
                        direction, path, path_len = self.navigator.compute_direction(
                            start_position=astar_start,
                            goal_position=astar_goal,
                            current_step=int(self.steps[b]),
                            current_velocity=self.vel[b, n],
                            smooth=False, # smooth 
                            cache_key=(int(b), int(n)),
                        )
                        if not np.allclose(astar_start, agent_positions[b, n]):
                            if path is not None and len(path) >= 2:
                                anchor = path[1]
                            elif path is not None and len(path) == 1:
                                anchor = path[0]
                            else:
                                anchor = astar_start
                            delta = anchor - agent_positions[b, n]
                            norm = np.linalg.norm(delta)
                            if norm > 1e-6:
                                direction = delta / norm

                    self.current_goals[b, n] = final_goal
                    # Use path length from A* directly (no Python loop)
                    if path_len <= 0.0:
                        dx = final_goal[0] - agent_positions[b, n, 0]
                        dy = final_goal[1] - agent_positions[b, n, 1]
                        dz = final_goal[2] - agent_positions[b, n, 2]
                        path_len = np.sqrt(dx*dx + dy*dy + dz*dz)
                    self.path_lengths[b, n] = path_len
                    if self._perf_debug and path is not None:
                        nav_astar += 1.0
                        nav_path_len += float(len(path))
                else:
                  
                    self.guidance_vectors[b, n] = 0.0
                    dx = final_goal[0] - agent_positions[b, n, 0]
                    dy = final_goal[1] - agent_positions[b, n, 1]
                    dz = final_goal[2] - agent_positions[b, n, 2]
                    self.path_lengths[b, n] = np.sqrt(dx*dx + dy*dy + dz*dz)

        if self._perf_debug:
            visible_frac = nav_visible / nav_total if nav_total > 0 else 0.0
            self._perf_nav = {
                "nav_astar": nav_astar,
                "nav_path_len": nav_path_len,
                "nav_frontier_envs": nav_frontier_envs,
                "nav_visible_frac": visible_frac,
            }

      
      
      
    
    def _update_target_tracking(self) -> None:
        """Update target tracking status with LKP (Last Known Position) mechanism."""
        B, N = self.num_envs, self.num_agents
        
        stage = self.stages[self.current_stage_idx]
        force_visible = np.zeros(B, dtype=np.bool_)
        all_visible, _ = batch_visibility_mask(
            self.pos,
            self.target_pos,
            self.target_vel,
            np.full(B, stage.view_radius, dtype=np.float64),
            force_visible,
            self.occupancy_grid,
            self.origin,
            self.voxel_size,
        )
        all_visible, gate_open = self._apply_visibility_gate(all_visible)
        all_visible, any_visible = self._apply_visibility_hysteresis(all_visible)
        self._any_visible = any_visible
        
      
        self.prev_target_mode_flag[:] = self.target_mode_flag
        
        # Vectorized team_has_target computation
        team_has_target = np.any(all_visible, axis=1)  # (B,)
        
        # Batch check LKP reached using Numba kernel
        lkp_breakout_dist = self.cfg.environment.lkp_breakout_dist_m
        lkp_reached = batch_check_lkp_reached(self.pos, self.last_known_target_pos, lkp_breakout_dist)
        
        # Vectorized update for environments where team sees target
        visible_envs = team_has_target
        self.last_known_target_pos[visible_envs] = self.target_pos[visible_envs]
        self.target_lost_steps[visible_envs] = 0
        self.lkp_countdown[visible_envs] = 30
        self.target_mode_flag[visible_envs, :] = 1.0
        
        # Update for environments where team lost target
        lost_envs = ~team_has_target
        self.target_lost_steps[lost_envs] += 1
        
        # LKP countdown logic (vectorized)
        in_lkp_phase = lost_envs & (self.lkp_countdown > 0)
        self.lkp_countdown[in_lkp_phase] -= 1
        
        # LKP breakout: if any agent reached LKP, force countdown to 0
        lkp_breakout_envs = in_lkp_phase & lkp_reached
        self.lkp_countdown[lkp_breakout_envs] = 0
        self.target_mode_flag[lkp_breakout_envs, :] = 0.0
        
        # Remaining LKP phase envs stay in tracking mode
        still_in_lkp = in_lkp_phase & ~lkp_reached
        self.target_mode_flag[still_in_lkp, :] = 1.0
        
        # LKP timeout: switch to search mode
        lkp_timeout_envs = lost_envs & (self.lkp_countdown <= 0)
        self.target_mode_flag[lkp_timeout_envs, :] = 0.0
        if self.visibility_gate_max_ratio > 0.0:
            gate_closed_envs = ~gate_open
            self.lkp_countdown[gate_closed_envs] = 0
            self.target_mode_flag[gate_closed_envs, :] = 0.0
        
        # Update observation targets (vectorized where possible)
        # Tracking mode with team visibility
        tracking_visible = (self.target_mode_flag > 0.5) & team_has_target[:, None]
        for b in range(B):
            for n in range(N):
                if tracking_visible[b, n]:
                    self.obs_target_pos[b, n] = self.target_pos[b]
                    self.obs_target_vel[b, n] = self.target_vel[b]
                elif self.target_mode_flag[b, n] > 0.5:  # Tracking but no visibility
                    self.obs_target_pos[b, n] = self.last_known_target_pos[b]
                    self.obs_target_vel[b, n, :] = 0.0
                else:  # Searching mode
                    if stage.guidance_enabled:
                        self.obs_target_pos[b, n] = self.current_goals[b, n]
                    else:
                        self.obs_target_pos[b, n] = self.last_known_target_pos[b]
                    self.obs_target_vel[b, n, :] = 0.0

        for b in range(B):
            self._update_layer_state(b)

    def _fill_exploration_mask(
        self,
        env_idx: int,
        confidence_2d_out: np.ndarray,
        confidence_3d_out: np.ndarray,
    ) -> np.ndarray:
        """Fill preallocated exploration confidence buffers for one environment."""
        ts = self.batch_layer_timestamps[env_idx]
        confidence_2d_out.fill(0.0)
        visited = ts > 0
        if np.any(visited):
            delta_steps = self.steps[env_idx] - ts[visited]
            delta_steps = np.clip(delta_steps, 0, len(self.confidence_lut) - 1)
            confidence_2d_out[visited] = self.confidence_lut[delta_steps]

        confidence_3d_out.fill(0.0)
        for layer_idx in range(3):
            z_min, z_max = self.layer_z_indices[layer_idx]
            if z_min > z_max:
                continue
            confidence_3d_out[:, :, z_min:z_max + 1] = confidence_2d_out[:, :, layer_idx][:, :, None]
        return confidence_3d_out

    def _get_exploration_mask(self, env_idx: int) -> np.ndarray:
        """Convert layered timestamps to 3D confidence grid for frontier detection."""
        gx, gy, gz = self.voxel_map.shape
        confidence_2d = np.zeros((gx, gy, 3), dtype=np.float32)
        confidence_3d = np.zeros((gx, gy, gz), dtype=np.float32)
        return self._fill_exploration_mask(env_idx, confidence_2d, confidence_3d)

    def get_layer_confidence(self, env_idx: int) -> np.ndarray:
        """Return layered confidence map (Gx, Gy, 3) for visualization."""
        ts = self.batch_layer_timestamps[env_idx]
        confidence_2d = np.zeros_like(ts, dtype=np.float32)
        visited = ts > 0
        if np.any(visited):
            delta_steps = self.steps[env_idx] - ts[visited]
            delta_steps = np.clip(delta_steps, 0, len(self.confidence_lut) - 1)
            confidence_2d[visited] = self.confidence_lut[delta_steps]
        return confidence_2d

    def get_frontier_debug(self, env_idx: int) -> Optional[Dict[str, object]]:
        """Return frontier debug info for visualization."""
        return self._frontier_debug[env_idx]
    
    def _build_base_observations(self) -> np.ndarray:
        """Build single-frame observation vectors for all agents."""
        # Build teammate features
        self.teammate_buffer = batch_build_teammate_features(
            self.pos, self.vel, self.yaw,
            self.cfg.perception.teammate_top_k,
            self.env_view_radius[0],  # Use first env's view radius
        )
        # Normalize action delay steps (per-env) for observability.
        if self.action_delay_buffer is not None and self.action_delay_max_steps > 0:
            delay_norm = (self.action_delay_steps_env.astype(np.float32)
                          / float(self.action_delay_max_steps))
        else:
            delay_norm = np.zeros(self.num_envs, dtype=np.float32)
        
        # Build full observations (use per-agent observation targets)
        obs = batch_build_observations_v2(
            self.pos, self.vel, self.yaw,
            self.target_pos, self.target_vel,
            self.obs_target_pos, self.obs_target_vel,
            self.target_mode_flag,
            self.lidar_buffer, self.guidance_vectors, self.teammate_buffer,
            self.imu_linear_acc_body, self.imu_angular_vel_body,
            delay_norm,
            self.slot_assignments,
            self.slot_positions,
            self.max_speed,
            self.map_size,
            self.num_agents,
            float(self.world_max[2] - self.world_min[2]),
        )

        encircle_feats = compute_encirclement_features(
            agents_pos=self.pos,
            target_pos=self.target_pos,
            active_masks=self.agent_alive,
            arena_bounds=self._arena_bounds_flat,
            wall_threshold=2.0,
        )
        obs_full = np.concatenate((obs, encircle_feats), axis=-1)
        if self.observation_profile == "local50":
            return self._project_full_obs_to_local50(obs_full)
        return obs_full

    def _build_observations_for_envs(self, env_ids: np.ndarray) -> np.ndarray:
        """Build observations only for selected environments (terminal snapshot fast-path)."""
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if env_ids.size == 0:
            return np.zeros((0, self.num_agents, self.get_obs_dim()), dtype=np.float32)

        pos = self.pos[env_ids]
        vel = self.vel[env_ids]
        yaw = self.yaw[env_ids]
        target_pos = self.target_pos[env_ids]
        target_vel = self.target_vel[env_ids]
        obs_target_pos = self.obs_target_pos[env_ids]
        obs_target_vel = self.obs_target_vel[env_ids]
        target_mode_flag = self.target_mode_flag[env_ids]
        lidar_buffer = self.lidar_buffer[env_ids]
        guidance_vectors = self.guidance_vectors[env_ids]
        imu_linear_acc_body = self.imu_linear_acc_body[env_ids]
        imu_angular_vel_body = self.imu_angular_vel_body[env_ids]
        slot_assignments = self.slot_assignments[env_ids]
        slot_positions = self.slot_positions[env_ids]
        agent_alive = self.agent_alive[env_ids]

        teammate_buffer = batch_build_teammate_features(
            pos,
            vel,
            yaw,
            self.cfg.perception.teammate_top_k,
            float(self.env_view_radius[int(env_ids[0])]),
        )
        if self.action_delay_buffer is not None and self.action_delay_max_steps > 0:
            delay_norm = (
                self.action_delay_steps_env[env_ids].astype(np.float32)
                / float(self.action_delay_max_steps)
            )
        else:
            delay_norm = np.zeros(env_ids.shape[0], dtype=np.float32)

        obs = batch_build_observations_v2(
            pos,
            vel,
            yaw,
            target_pos,
            target_vel,
            obs_target_pos,
            obs_target_vel,
            target_mode_flag,
            lidar_buffer,
            guidance_vectors,
            teammate_buffer,
            imu_linear_acc_body,
            imu_angular_vel_body,
            delay_norm,
            slot_assignments,
            slot_positions,
            self.max_speed,
            self.map_size,
            self.num_agents,
            float(self.world_max[2] - self.world_min[2]),
        )
        encircle_feats = compute_encirclement_features(
            agents_pos=pos,
            target_pos=target_pos,
            active_masks=agent_alive,
            arena_bounds=self._arena_bounds_flat,
            wall_threshold=2.0,
        )
        obs_full = np.concatenate((obs, encircle_feats), axis=-1)
        if self.observation_profile == "local50":
            return self._project_full_obs_to_local50(obs_full)
        return obs_full

    def _build_observations(self) -> np.ndarray:
        """Build stacked observation vectors for all agents."""
        base_obs = self._build_base_observations()
        return base_obs
    
    def _build_infos(
        self,
        collisions: np.ndarray,
        captured: np.ndarray,
        dones: np.ndarray,
        timeout: np.ndarray,
        done_reason: np.ndarray,
        reward_breakdown: Optional[List[Dict]],
        reward_breakdown_arr: np.ndarray,
        captured_raw: np.ndarray,
        clean_capture: np.ndarray,
        team_target_min_dist_3d: np.ndarray,
        team_target_mean_abs_dz: np.ndarray,
        far_idle_ratio: np.ndarray,
        capture_contribution: np.ndarray,
        dist_to_target_p50: np.ndarray,
        dist_to_target_p90: np.ndarray,
        terminal_obs: Optional[np.ndarray],
        terminal_agent_pos: Optional[np.ndarray],
        terminal_target_pos: Optional[np.ndarray],
        terminal_target_vel: Optional[np.ndarray],
    ) -> List[Dict]:
        """Build info dictionaries for each environment."""
        infos = []
        stage_name = self.stages[self.current_stage_idx].name

        for b in range(self.num_envs):
            collider_mask = collisions[b].astype(np.bool_)
            if bool(dones[b]):
                if str(done_reason[b]) == "collision":
                    terminated_mask = collider_mask
                elif str(done_reason[b]) == "capture":
                    terminated_mask = np.ones(self.num_agents, dtype=np.bool_)
                elif str(done_reason[b]) == "timeout":
                    terminated_mask = np.zeros(self.num_agents, dtype=np.bool_)
                else:
                    terminated_mask = collider_mask
            else:
                terminated_mask = np.zeros(self.num_agents, dtype=np.bool_)
            terminal_observation = None
            if terminal_obs is not None and bool(dones[b]):
                terminal_observation = terminal_obs[b].astype(np.float32, copy=False)
            terminal_agent_pos_b = None
            terminal_target_pos_b = None
            terminal_target_vel_b = None
            if terminal_agent_pos is not None and bool(dones[b]):
                terminal_agent_pos_b = terminal_agent_pos[b].astype(np.float32, copy=False)
            if terminal_target_pos is not None and bool(dones[b]):
                terminal_target_pos_b = terminal_target_pos[b].astype(np.float32, copy=False)
            if terminal_target_vel is not None and bool(dones[b]):
                terminal_target_vel_b = terminal_target_vel[b].astype(np.float32, copy=False)

            if self._minimal_train_info:
                infos.append({
                    "captured": bool(captured[b]),
                    "captured_raw": bool(captured_raw[b]),
                    "clean_capture": bool(clean_capture[b]),
                    "done_reason": str(done_reason[b]),
                    "collision": bool(np.any(collisions[b])),
                    "collisions": collisions[b].astype(np.bool_, copy=False),
                    "terminated_mask": terminated_mask.astype(np.bool_, copy=False),
                    "terminal_observation": terminal_observation,
                    "terminal_agent_pos": terminal_agent_pos_b,
                    "terminal_target_pos": terminal_target_pos_b,
                    "terminal_target_vel": terminal_target_vel_b,
                    "team_target_min_dist_3d": float(team_target_min_dist_3d[b]),
                    "team_target_mean_abs_dz": float(team_target_mean_abs_dz[b]),
                    "far_idle_ratio": float(far_idle_ratio[b]),
                    "capture_contribution": float(capture_contribution[b]),
                    "dist_to_target_p50": float(dist_to_target_p50[b]),
                    "dist_to_target_p90": float(dist_to_target_p90[b]),
                    "reward_breakdown_row": reward_breakdown_arr[b].astype(np.float64, copy=False),
                })
                continue

            avg_speed = float(np.mean(np.linalg.norm(self.vel[b], axis=-1)))
            infos.append({
                "stage": stage_name,
                "captured": bool(captured[b]),
                "captured_raw": bool(captured_raw[b]),
                "clean_capture": bool(clean_capture[b]),
                "success": bool(captured[b]),
                "done": bool(dones[b]),
                "timeout": bool(timeout[b]),
                "done_reason": str(done_reason[b]),
                "collision": bool(np.any(collisions[b])),
                "collisions": collisions[b].tolist(),
                "is_collider": collider_mask.tolist(),
                "terminated_mask": terminated_mask.tolist(),
                "terminal_observation": terminal_observation,
                "terminal_agent_pos": terminal_agent_pos_b,
                "terminal_target_pos": terminal_target_pos_b,
                "terminal_target_vel": terminal_target_vel_b,
                "team_target_min_dist_3d": float(team_target_min_dist_3d[b]),
                "team_target_mean_abs_dz": float(team_target_mean_abs_dz[b]),
                "far_idle_ratio": float(far_idle_ratio[b]),
                "capture_contribution": float(capture_contribution[b]),
                "dist_to_target_p50": float(dist_to_target_p50[b]),
                "dist_to_target_p90": float(dist_to_target_p90[b]),
                "hybrid_pred_alpha": float(self.last_pred_alpha[b]),
                "pred_error_phys_ema": float(self.error_phys_ema[b]),
                "pred_error_nn_ema": float(self.error_nn_ema[b]),
                "layer_state": self.layer_state[b].tolist(),
                "avg_speed": avg_speed,
                "reward_breakdown": [reward_breakdown[b]] if reward_breakdown is not None else [],
                "metrics": {
                    "Shield_Trigger_Count": float(self._metric_shield[b]),
                    "Action_Smoothness_Score": float(-self._metric_smoothness[b]),
                    "Guidance_Alignment_Error": float(self._metric_align_err[b]),
                    "Far_Idle_Ratio": float(self._metric_far_idle_ratio[b]),
                    "Capture_Contribution": float(self._metric_capture_contribution[b]),
                    "Dist_To_Target_P50": float(self._metric_dist_p50[b]),
                    "Dist_To_Target_P90": float(self._metric_dist_p90[b]),
                },
            })

        return infos
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    def get_obs_dim(self) -> int:
        """Return observation dimension."""
        return self._calc_obs_dim()
    
    @property
    def num_actions(self) -> int:
        """Return action dimension."""
        return 4


# =============================================================================
# Benchmark Utility
# =============================================================================

def benchmark_env(num_envs: int = 32, num_steps: int = 1000, warmup: int = 100):
    """
    Benchmark environment performance.
    
    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps to benchmark
        warmup: Warmup steps (not counted)
    
    Returns:
        sps: Steps per second
    """
    import time
    from src.config import load_config
    
    cfg = load_config()
    env = VectorizedMutualAStarEnvV2(num_envs=num_envs, cfg=cfg)
    
    # Warmup
    obs, _ = env.reset()
    for _ in range(warmup):
        actions = np.random.randn(num_envs, env.num_agents, 4) * 2
        env.step(actions)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        actions = np.random.randn(num_envs, env.num_agents, 4) * 2
        env.step(actions)
    elapsed = time.perf_counter() - start
    
    total_steps = num_envs * num_steps
    sps = total_steps / elapsed
    
    print(f"Benchmark Results:")
    print(f"  Environments: {num_envs}")
    print(f"  Steps: {num_steps}")
    print(f"  Total env steps: {total_steps:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  SPS: {sps:.1f}")
    
    return sps


__all__ = ["VectorizedMutualAStarEnvV2", "benchmark_env"]
