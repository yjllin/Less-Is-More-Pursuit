"""
AirSim policy-visualization script.

Features:
1. Connect to AirSim and control the pursuer drones.
2. Load a trained MAPPO checkpoint.
3. Visualize A* paths and frontier goals.
4. Inspect target motion and capture behavior in 3D.

Usage:
    python scripts/test_policy_airsim.py --policy runs/mappo_3d/default/20251221_165820/policy.pt
    python scripts/test_policy_airsim.py --policy runs/mappo_3d/default/20251221_165820/policy.pt --steps 500
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
import threading
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Add the project root to `sys.path` so the script can be run directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import load_config
from src.controllers.mappo_policy import MAPPOPolicy3D
from src.environment.vectorized_env_v2 import compute_encirclement_features
from src.experiment_modes import BASELINE_CHOICES, get_baseline_override
from src.navigation.astar3d import AStar3D
from src.navigation.frontier_allocation import FrontierAllocator
from src.navigation.voxel_map import VoxelMap3D
from src.environment.batch_kernels import (
    batch_build_observations_v2,
    batch_compute_rewards_v2,
    batch_check_lkp_reached,
    batch_update_exploration_grid_optimized,
    batch_select_goals,
    batch_visibility_mask,
)

# Import Numba only when it is available.
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import airsim
except ImportError:
    print("Error: airsim package not installed. Run: pip install airsim")
    sys.exit(1)

DEFAULT_LIDAR_SENSOR_NAME = "LidarSensor1"


# =============================================================================
# Low-pass filtering and safety-shield helpers
# =============================================================================

def apply_action_smoothing(
    actions: np.ndarray,       # (N, 4) in/out
    prev_actions: np.ndarray,  # (N, 4)
    alpha: float,
    deadband: float = 1e-3,
) -> None:
    """Apply exponential moving-average smoothing to actions.

    Formula:
        smoothed = alpha * raw + (1 - alpha) * prev

    Smaller `alpha` produces stronger smoothing. In practice, values in
    `[0.3, 0.5]` work well for AirSim real-time control.
    """
    N = actions.shape[0]
    for n in range(N):
        for i in range(4):
            # EMA smoothing
            actions[n, i] = alpha * actions[n, i] + (1.0 - alpha) * prev_actions[n, i]
          
            if abs(actions[n, i]) < deadband:
                actions[n, i] = 0.0


def apply_safety_shield(
    actions: np.ndarray,       # (N, 4) In/Out [vx, vy, vz, yaw_rate] in body frame
    lidar_dists: np.ndarray,   # (N, 26) Normalized distances [0, 1]
    min_dist_m: float,         # Safety threshold (e.g. 4.0m)
    lidar_max_range: float,    # Max lidar range (e.g. 50m)
    target_dist: np.ndarray = None,  # (N,) Distance to target per agent
    capture_radius: float = 5.0,     # Capture radius
) -> np.ndarray:               # (N,) bool - Triggered flags
    """
    Apply Safety Shield: Modifies actions in-place to prevent collisions.
    
    Damping shield:
    - If within min_dist, limit inward velocity component.
    - No repulsive force; clamp inward component to 0 at contact.
    - Near target (within 2.5x capture radius): reduce threshold to allow capture
    
    Note: Both LiDAR and actions are in body frame, so we operate directly.
    """
    N = actions.shape[0]
    triggered = np.zeros(N, dtype=bool)
    
    # Pre-compute ray directions (Must match Lidar simulation order!)
    # 26 directions: 3x3x3 grid minus center
    directions = np.zeros((26, 3), dtype=np.float64)
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                norm = np.sqrt(float(dx*dx + dy*dy + dz*dz))
                directions[idx, 0] = dx / norm
                directions[idx, 1] = dy / norm
                directions[idx, 2] = dz / norm
                idx += 1

    for n in range(N):
        # Adaptive threshold: reduce when close to target to allow capture
        if target_dist is not None and target_dist[n] < capture_radius * 2.5:
            effective_min_dist = min_dist_m * 0.5
        else:
            effective_min_dist = min_dist_m
        
        # 1. Find nearest obstacle
        min_val = 1.0
        min_idx = -1
        
        for k in range(26):
            d = lidar_dists[n, k]
            if d < min_val:
                min_val = d
                min_idx = k
        
        # Check threshold
        current_dist_m = min_val * lidar_max_range
        
        if current_dist_m < effective_min_dist and min_idx != -1:
            triggered[n] = True
            
          
            # print(f"[Shield] Agent {n}: obstacle at {current_dist_m:.1f}m < {effective_min_dist:.1f}m, direction_idx={min_idx}")
            
            # Get obstacle direction vector (Body Frame)
            obs_dir = directions[min_idx]  # (3,)
            
            # Current command velocity (body frame)
            cmd_vel = actions[n, :3].copy()
            
            # 2. Limit velocity towards obstacle (Project & Damp)
            # proj = v . n (dot product)
            proj = cmd_vel[0]*obs_dir[0] + cmd_vel[1]*obs_dir[1] + cmd_vel[2]*obs_dir[2]
            
            # If moving towards obstacle (proj > 0), damp it based on distance
            if proj > 0:
                damp = current_dist_m / effective_min_dist
                new_proj = proj * damp
                delta = proj - new_proj
                cmd_vel[0] -= delta * obs_dir[0]
                cmd_vel[1] -= delta * obs_dir[1]
                cmd_vel[2] -= delta * obs_dir[2]
            
            # Update action
            actions[n, 0] = cmd_vel[0]
            actions[n, 1] = cmd_vel[1]
            actions[n, 2] = cmd_vel[2]
            # Yaw rate usually doesn't need modification for simple shield
                
    return triggered

SPAWN_POSITIONS = [
    np.array([-30.0, -30.0, -22.1], dtype=np.float64),  # NED: Z
    np.array([30.0, -30.0, -22.1], dtype=np.float64),
    np.array([30.0, 30.0, -22.1], dtype=np.float64),
    np.array([-30.0, 30.0, -22.1], dtype=np.float64),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained policy in AirSim")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.pt checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=BASELINE_CHOICES,
        help="Apply unified baseline mode overrides to config.experiment.",
    )
    parser.add_argument("--stage", type=str, default=None,
                        help="Curriculum stage name (e.g., stage1_basic_hover, stage2_simple_tracking)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None, 
                        help="Target position [x, y, z] in world coords (default: random)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="eval_logs/airsim_episode_eval.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--show-path", action="store_true", default=True,
                        help="Show A* navigation paths")
    parser.add_argument("--path-interval", type=int, default=10,
                        help="Update path visualization every N steps")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--debug-frontier", action="store_true",
                        help="Print frontier/visibility debug info")
    parser.add_argument("--follow-guidance", action="store_true",
                        help="Ignore policy, directly follow A* guidance (for testing A* correctness)")
    parser.add_argument("--guidance-speed", type=float, default=5.0,
                        help="Speed when following guidance (m/s)")
    parser.add_argument("--show-obstacles", action="store_true",
                        help="Render obstacle voxels in AirSim (for debugging)")
    parser.add_argument("--obstacle-z", type=int, default=None,
                        help="Only show obstacles at specific z layer (grid index)")
    parser.add_argument("--no-shield", action="store_true",
                        help="Disable safety shield")
    parser.add_argument("--lidar-name", type=str, default=DEFAULT_LIDAR_SENSOR_NAME,
                        help="AirSim lidar sensor name (default: LidarSensor1)")
    parser.add_argument("--lidar-diagnostics", action="store_true",
                        help="Log LiDAR point count/min distance/nearest direction periodically.")
    parser.add_argument("--diag-interval", type=int, default=20,
                        help="Diagnostics print interval in steps.")
    parser.add_argument("--smoothing-alpha", type=float, default=None,
                        help="Action smoothing alpha (0-1, higher=less smooth). Default from config.")
    parser.add_argument("--nav-interval", type=int, default=1,
                        help="Navigation update interval (steps). Default: 1")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (mean) actions instead of sampling.")
    return parser.parse_args()


def apply_baseline_overrides(cfg, baseline: str | None):
    if not baseline:
        return cfg
    override = get_baseline_override(baseline)
    assert override is not None
    reward_runtime = cfg.reward_runtime
    if override["direction_gate_active_radius_m"] is not None:
        reward_runtime = replace(
            reward_runtime,
            direction_gate_active_radius_m=override["direction_gate_active_radius_m"],
        )
    return replace(
        cfg,
        reward_runtime=reward_runtime,
        experiment=replace(
            cfg.experiment,
            mode=override["mode"],
            observation_profile=override["observation_profile"],
            guidance_backend=override["guidance_backend"],
            critic_mode=override["critic_mode"],
            eval_controller=override["eval_controller"],
        ),
    )


def resolve_config_path(args) -> str | None:
    if args.config:
        return args.config
    policy_path = Path(args.policy)
    sibling = policy_path.with_name("config.json")
    if sibling.exists():
        print(f"[Config] Using sibling config: {sibling}")
        return str(sibling)
    return None


def get_stage_config(cfg, stage_name):
    """"""
    if stage_name is None:
        return None
    
    for stage in cfg.curriculum.stages:
        if stage.name == stage_name:
            return stage
    
  
    available = [s.name for s in cfg.curriculum.stages]
    print(f"[Warning] Stage '{stage_name}' not found. Available stages: {available}")
    return None


class AirSimEnv:
    """AirSim """
    
    def __init__(
        self,
        cfg,
        stage_config=None,
        target_pos=None,
        debug=False,
        debug_frontier=False,
        lidar_name: str | None = None,
        lidar_diagnostics: bool = False,
        diag_interval: int = 20,
    ):
        self.cfg = cfg
        self.stage_config = stage_config
        self.debug = debug
        self.debug_frontier = debug_frontier
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.lidar_sensor_name = lidar_name or DEFAULT_LIDAR_SENSOR_NAME
        
      
        self._client_lock = threading.Lock()
        
      
      
        
      
        self._lidar_cache = {}
        self._lidar_cache_step = {}
        self._lidar_cache_count = {}
        self._lidar_update_interval = 1  # LiDAR
        self.lidar_diagnostics = bool(lidar_diagnostics)
        self.diag_interval = max(1, int(diag_interval))
        
      
        self._nav_cache = {}
        self._nav_cache_step = {}
        self._nav_cache_interval = 2  # 2
        
      
        self._state_cache = {}
        self._state_cache_timestamp = 0
        self._state_cache_ttl = 0.08  # 80msAirSim
        
      
        self.vehicle_names = self._get_vehicles()
        self.agent_names = [n for n in self.vehicle_names if n != "target"]
        self.num_agents = min(len(self.agent_names), cfg.environment.num_agents)
        self.agent_names = self.agent_names[:self.num_agents]
        print(f"[AirSim] Connected. Using agents: {self.agent_names}")
        self._lidar_last_count = np.zeros(self.num_agents, dtype=np.int32)
        self._lidar_dirs_body = self._build_lidar_dirs_body()
        self.lidar_sector_subrays = int(getattr(cfg.perception, "lidar_sector_subrays", 5))
        self.lidar_sector_spread_deg = float(getattr(cfg.perception, "lidar_sector_spread_deg", 10.0))
        self._lidar_sub_dirs_body = self._build_lidar_sub_dirs_body(
            self.lidar_sector_subrays,
            self.lidar_sector_spread_deg,
        )
        self._lidar_sub_cos = float(np.cos(self.lidar_sector_spread_deg * np.pi / 180.0))
        print(f"[LiDAR] Using sensor: {self.lidar_sensor_name}")
        if self.lidar_diagnostics:
            print(f"[LiDAR] Diagnostics enabled (interval={self.diag_interval} steps)")
        
      
        self.dt = 1.0 / cfg.environment.step_hz
        self.max_speed = cfg.control.action_bounds["vx"]
        # World bounds (match training env)
        world_size = np.array(self.cfg.environment.world_size_m, dtype=np.float64)
        self.world_min = np.array([
            -world_size[0] / 2.0,
            -world_size[1] / 2.0,
            0.0,
        ], dtype=np.float64)
        self.world_max = self.world_min + world_size
        self.top_k = cfg.perception.teammate_top_k
        self.observation_profile = str(getattr(cfg.experiment, "observation_profile", "full83")).lower()
        if self.observation_profile not in {"full83", "local50"}:
            raise ValueError(f"Unsupported observation_profile: {self.observation_profile}")
        # Frame stacking removed: always use single-frame observations.
        self.frame_stack = 1
        self.imu_accel_noise_std = float(getattr(cfg.environment, "imu_accel_noise_std", 0.0))
        self.imu_gyro_noise_std = float(getattr(cfg.environment, "imu_gyro_noise_std", 0.0))
        self.map_size = 300.0  # occupancy
        
        # ============================================================
      
        # ============================================================
        self.action_scale_vx = cfg.control.action_bounds["vx"]      # 8.0 m/s
        self.action_scale_vy = cfg.control.action_bounds["vy"]      # 4.0 m/s
        self.action_scale_vz = cfg.control.action_bounds["vz"]      # 3.0 m/s
        self.action_scale_yaw = cfg.control.action_bounds["yaw_rate"]  # 1.5 rad/s
        print(f"[Control] Action scales: vx={self.action_scale_vx}, vy={self.action_scale_vy}, "
              f"vz={self.action_scale_vz}, yaw={self.action_scale_yaw}")
        
      
        self.smoothing_alpha = cfg.control.smoothing_alpha  # 
        self.ema_clip = cfg.control.ema_clip
        self.prev_actions = np.zeros((self.num_agents, 4), dtype=np.float64)
        
      
        self.shield_min_dist = cfg.shield.min_distance_m
        self.shield_enabled = True  # 
        
      
        if debug:
            self.shield_min_dist = max(self.shield_min_dist, 8.0)  # 8
            print(f"[Shield-Debug] Using conservative parameters: min_dist={self.shield_min_dist}m")
        
      
        if stage_config:
            self.capture_radius = getattr(stage_config, 'capture_radius_m', cfg.environment.capture_radius_m)
            self.view_radius = getattr(stage_config, 'view_radius_m', 300.0)
            self.target_speed = getattr(stage_config, 'target_speed', 0.0)
            self.target_behavior = getattr(stage_config, 'target_behavior', 'static')
            self.guidance_enabled = getattr(stage_config, 'guidance_enabled', True)
            self.target_axis_limit_enabled = bool(getattr(stage_config, "target_axis_limit", False))
            print(f"[Stage] {stage_config.name}: view_radius={self.view_radius}m, "
                  f"capture_radius={self.capture_radius}m, target_speed={self.target_speed}m/s, "
                  f"target_behavior={self.target_behavior}")
        else:
            self.capture_radius = cfg.environment.capture_radius_m
            self.view_radius = 300.0  # default omniscient mode
            self.target_speed = 0.0
            self.target_behavior = "static"
            self.guidance_enabled = True
            self.target_axis_limit_enabled = False
            print(f"[Stage] Default: view_radius={self.view_radius}m, capture_radius={self.capture_radius}m")
      
        base_lidar_range = float(self.view_radius)
      
        if debug:
            self.lidar_max_range = min(base_lidar_range, 30.0)  # 30
            print(f"[LiDAR-Debug] Using shorter range: {self.lidar_max_range}m for better obstacle detection")
        else:
            self.lidar_max_range = base_lidar_range

        # 
        self.target_pos = np.array(target_pos, dtype=np.float64) if target_pos else None
        self.target_vel = np.zeros(3, dtype=np.float64)
        self._target_last_free_pos = None
        
      
        self._load_occupancy()
        self.confidence_decay = getattr(self.cfg.environment, "confidence_decay", 0.999)
        self.confidence_threshold = getattr(self.cfg.environment, "confidence_threshold", 0.5)
        if self.occupancy_grid is not None:
            grid_shape = self.occupancy_grid.shape
            self.batch_timestamps = np.zeros((1, grid_shape[0], grid_shape[1], grid_shape[2]), dtype=np.int32)
            steps = int(np.ceil(np.log(self.confidence_threshold) / np.log(self.confidence_decay)))
            self.steps_to_reexplore = max(1, steps)
        else:
            self.batch_timestamps = None
            self.steps_to_reexplore = 1

        if self.occupancy_grid is not None:
            self.target_avoid_radius_vox = max(1, int(np.ceil(6.0 / max(self.voxel_size, 1e-6))))
        else:
            self.target_avoid_radius_vox = 0
        
      
        self._init_navigator()
        self.goal_offsets = self._init_goal_offsets()
        self.offset_decay_distance = float(getattr(cfg.navigation, "Offset_decay_distance", 30.0))
        self.straight_replace_astar_distance = float(
            getattr(cfg.navigation, "Straight_replace_Astar_distance", 25.0)
        )
        
      
        self.current_goals = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.guidance_vectors = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.current_paths = [None] * self.num_agents
        
      
        self._wander_direction = np.zeros(3)
        self._direction_change_counter = 0
        self.step_count = 0
        
      
        self.nav_update_interval = 1
        self.explore_update_interval = int(getattr(cfg.navigation, "Explore_update_interval", 10))
        self.reward_params = self._build_reward_params(stage_config)
        self.reward_weights = self._build_reward_weights()
        self.prev_target_dist = np.zeros(self.num_agents, dtype=np.float64)
        self.prev_frontier_dist = np.zeros(self.num_agents, dtype=np.float64)
        self.frontier_deltas = np.zeros(self.num_agents, dtype=np.float64)
        self.exploration_deltas = np.zeros(self.num_agents, dtype=np.float64)
        self.path_lengths = np.zeros(self.num_agents, dtype=np.float64)
        self.prev_frontier_goals = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.prev_frontier_mask = np.zeros(self.num_agents, dtype=bool)
        self.last_known_target_pos = np.zeros(3, dtype=np.float64)
        self.target_lost_steps = 0
        self.lkp_countdown = 0
        self.target_mode_flag = np.ones(self.num_agents, dtype=np.float32)
        self.prev_target_mode_flag = np.ones(self.num_agents, dtype=np.float32)
        self.obs_target_pos = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.obs_target_vel = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.prev_obs_target_pos = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.prev_vel = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.prev_yaw = np.zeros(self.num_agents, dtype=np.float64)
        self.imu_linear_acc_body = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.imu_angular_vel_body = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.slot_positions = np.zeros((self.num_agents, 3), dtype=np.float64)
        self.slot_assignments = np.full(self.num_agents, -1, dtype=np.int32)
        self.tactical_slot_radius = float(getattr(cfg.navigation, "tactical_slot_radius_m", 16.0))
        self.tactical_slot_radius_offset = float(getattr(cfg.navigation, "tactical_slot_radius_offset_m", 6.0))
        self.base_obs_dim = self._calc_base_obs_dim()
        self.obs_stack = None
        self.position_history = np.zeros((self.num_agents, 5, 3), dtype=np.float64)
        self.history_idx = 0
        self.collision_armed = np.ones(self.num_agents, dtype=bool)
        self.collision_rearm_count = np.zeros(self.num_agents, dtype=np.int32)
        self.lkp_countdown_steps = int(getattr(cfg.environment, "lkp_countdown_steps", 30))
        self.lkp_breakout_dist_m = float(getattr(cfg.environment, "lkp_breakout_dist_m", 2.0))
        self.team_collision_radius = float(getattr(cfg.environment, "team_collision_radius_m", cfg.shield.min_distance_m * 0.5))
        self._spawn_ned_z = {}
        self._target_z_ned = None
        self._arena_bounds_flat = np.array(
            [
                self.world_min[0], self.world_max[0],
                self.world_min[1], self.world_max[1],
                self.world_min[2], self.world_max[2],
            ],
            dtype=np.float64,
        )

    def _calc_base_obs_dim(self) -> int:
        if self.observation_profile == "local50":
            return 50
        return 26 + 3 + 3 + 3 + 3 + 3 + (self.top_k * 8) + 3 + self.num_agents + 1 + 1 + self.num_agents + 3 + 2

    @staticmethod
    def _project_full_obs_to_local50(obs_full: np.ndarray) -> np.ndarray:
        if obs_full.shape[-1] != 83:
            raise ValueError(f"Expected full83 observation before local projection, got {obs_full.shape[-1]}")
        return np.concatenate((obs_full[..., :41], obs_full[..., 65:74]), axis=-1)

    def _build_tactical_slots(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slots = np.repeat(self.target_pos[None, :], self.num_agents, axis=0).astype(np.float64, copy=True)
        assignments = np.full(self.num_agents, -1, dtype=np.int32)
        if self.num_agents <= 0:
            return slots, assignments
        if not np.any(self.target_mode_flag > 0.5):
            return slots, assignments

        if np.any(np.isfinite(self.obs_target_pos)):
            tracking_mask = self.target_mode_flag > 0.5
            center = np.mean(self.obs_target_pos[tracking_mask], axis=0) if np.any(tracking_mask) else self.target_pos
        else:
            center = self.target_pos
        center = np.asarray(center, dtype=np.float64)
        radius = max(float(self.capture_radius) + self.tactical_slot_radius_offset, self.tactical_slot_radius, 0.1)
        sqrt2 = np.sqrt(2.0)
        sqrt6 = np.sqrt(6.0)
        rel = np.array(
            [
                [0.0, 0.0, 1.0],
                [2.0 * sqrt2 / 3.0, 0.0, -1.0 / 3.0],
                [-sqrt2 / 3.0, sqrt6 / 3.0, -1.0 / 3.0],
                [-sqrt2 / 3.0, -sqrt6 / 3.0, -1.0 / 3.0],
            ],
            dtype=np.float64,
        )
        slots = center[None, :] + radius * rel[: self.num_agents]
        slots = np.clip(slots, self.world_min + 1e-3, self.world_max - 1e-3)
        slots = np.asarray([self._snap_goal_to_free(slot) for slot in slots], dtype=np.float64)

        active_idx = np.flatnonzero(self.target_mode_flag > 0.5)
        if active_idx.size == 0:
            return slots, assignments
        dists = np.linalg.norm(positions[active_idx, None, :] - slots[None, :, :], axis=-1)
        rr, cc = linear_sum_assignment(dists)
        for r_i, c_i in zip(rr, cc):
            assignments[active_idx[r_i]] = int(c_i)
        return slots, assignments

    def _hover_vehicle(self, name: str, duration: float = 5.0, z_ned: float | None = None) -> None:
        """Best-effort hover command to prevent falling after spawn/teleport."""
        try:
            with self._client_lock:
                self.client.hoverAsync(vehicle_name=name)
        except Exception:
            try:
                with self._client_lock:
                    if z_ned is not None:
                        self.client.moveByVelocityZAsync(
                            0, 0, float(z_ned),
                            duration=duration,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                            vehicle_name=name
                        )
                    else:
                        self.client.moveByVelocityAsync(
                            0, 0, 0,
                            duration=duration,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                            vehicle_name=name
                        )
            except Exception as e:
                if self.debug:
                    print(f"[Warning] Failed to hover {name}: {e}")
        
    def _get_vehicles(self):
        try:
            return self.client.listVehicles() or []
        except:
            return ["Drone0", "Drone1", "Drone2", "Drone3"]
    
    def _load_occupancy(self):
        """"""
        import json
        mask_path = Path(self.cfg.navigation.frontier_mask_path)
        if mask_path.exists():
            self.occupancy_grid = np.load(mask_path).astype(np.int8)
            meta_path = mask_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.origin = np.array(meta.get("origin", [0, 0, 0]), dtype=np.float64)
                self.voxel_size = float(meta.get("voxel_size", 2.0))
            else:
                self.origin = np.zeros(3)
                self.voxel_size = 2.0
            occ_world_size = np.array(self.occupancy_grid.shape, dtype=np.float64) * self.voxel_size
            self.map_size = float(max(occ_world_size[:2]))
            print(f"[Map] Loaded occupancy grid: {self.occupancy_grid.shape}, "
                  f"origin={self.origin}, voxel_size={self.voxel_size}")
        else:
            print(f"[Warning] Occupancy grid not found: {mask_path}")
            self.occupancy_grid = None
            self.origin = np.zeros(3)
            self.voxel_size = 2.0
            self.map_size = 300.0
    
    def _init_navigator(self):
        """ A* """
        if self.occupancy_grid is None:
            self.navigator = None
            return
        
      
        self.confidence_decay = getattr(self.cfg.environment, 'confidence_decay', 0.999)
        self.confidence_threshold = getattr(self.cfg.environment, 'confidence_threshold', 0.5)
        print(f"[Nav] Confidence decay={self.confidence_decay}, threshold={self.confidence_threshold}")
        
      
        occ_grid = self.occupancy_grid.copy()
        print(f"[Nav] Downsampling disabled; using full-resolution grid")
        
      
        lookahead = self.cfg.navigation.lookahead_distance_m
        
        self.navigator = AStar3D(
            grid_shape=occ_grid.shape,
            voxel_size=self.voxel_size,
            cache_refresh_steps=self.cfg.navigation.cache_refresh_steps,
            lookahead_m=lookahead,
            heuristic_weight=self.cfg.navigation.heuristic_weight,
            origin=self.origin,
        )
        self.navigator.update_grid(occ_grid)
        print(f"[Nav] A* navigator initialized: grid={occ_grid.shape}, voxel={self.voxel_size}m, lookahead={lookahead}m")
        
      
        world_size = np.array(self.occupancy_grid.shape) * self.voxel_size
        self.voxel_map = VoxelMap3D(
            tuple(world_size),
            self.voxel_size,
            self.cfg.navigation.occupancy_threshold,
            origin=self.origin,
        )
        
      
      
        self.voxel_map.visits = np.zeros(self.occupancy_grid.shape, dtype=np.float32)
        self.voxel_map.visits[self.occupancy_grid == 1] = 1.0
        if self.batch_timestamps is not None:
            self.batch_timestamps[:] = 0
        self.voxel_map.hits[self.occupancy_grid == 1] = 1
        
      
        occ_grid = self.occupancy_grid.astype(np.int8)
        self.allowed_mask = occ_grid == 0
        
      
        self.frontier_allocator = FrontierAllocator(
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
            allowed_mask=self.allowed_mask,
        )
        self.frontier_allocator.set_astar(self.navigator)
        self.frontier_allocator.set_occupancy_grid(self.occupancy_grid)
        self.frontier_allocator._debug_enabled = self.debug
        print(f"[Nav] FrontierAllocator initialized, confidence_threshold={self.confidence_threshold}")
    
    def _init_goal_offsets(self) -> np.ndarray:
        """Initialize per-agent goal offsets to reduce path overlap."""
        offsets = np.zeros((self.num_agents, 3), dtype=np.float64)
        if self.num_agents <= 1:
            return offsets
        radius = 8.0
        for i in range(self.num_agents):
            angle = (2.0 * np.pi * i) / max(self.num_agents, 1)
            offsets[i, 0] = radius * np.cos(angle)
            offsets[i, 1] = radius * np.sin(angle)
            offsets[i, 2] = 0.0
        return offsets

    def _ensure_spawn_separation(self, positions: list[np.ndarray]) -> list[np.ndarray]:
        """Adjust spawn positions to avoid inter-agent overlap."""
        if len(positions) <= 1:
            return positions
        min_sep = float(getattr(self.cfg.environment, "min_spawn_separation_m", 10.0))
        min_sep = max(min_sep, float(self.team_collision_radius) * 2.0)
        adjusted: list[np.ndarray] = []
        for idx, pos in enumerate(positions):
            candidate = pos.copy()
            attempt = 0
            while True:
                ok = True
                for prev in adjusted:
                    if np.linalg.norm(candidate - prev) < min_sep:
                        ok = False
                        break
                if ok or attempt >= 64:
                    break
                angle = (attempt % 12) * (2.0 * np.pi / 12.0)
                radius = min_sep * (1 + attempt // 12)
                candidate = pos.copy()
                candidate[0] += radius * np.cos(angle)
                candidate[1] += radius * np.sin(angle)
                attempt += 1
            if attempt > 0 and self.debug:
                print(f"[Spawn] Adjusted agent {idx} to avoid overlap (attempts={attempt})")
            adjusted.append(candidate)
        return adjusted

    def _spawn_positions_ok(self, positions: np.ndarray, desired_ned: list[np.ndarray]) -> bool:
        """Validate spawn positions against desired poses and separation."""
        if positions is None or len(positions) == 0:
            return False
        tol = 3.0
        min_sep = float(getattr(self.cfg.environment, "min_spawn_separation_m", 10.0))
        min_sep = max(min_sep, float(self.team_collision_radius) * 2.0)
        max_agents = min(len(positions), len(desired_ned))
        for i in range(max_agents):
            desired_world = desired_ned[i].copy()
            desired_world[2] = -desired_world[2]  # NED -> World (z-up)
            if np.linalg.norm(positions[i] - desired_world) > tol:
                return False
        for i in range(max_agents):
            for j in range(i + 1, max_agents):
                if np.linalg.norm(positions[i] - positions[j]) < min_sep:
                    return False
        return True

    def reset(self):
        """"""
        self.client.reset()
        time.sleep(0.5)
        self._state_cache = {}
        self._state_cache_timestamp = 0.0
        
      
        for name in self.vehicle_names:
            try:
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
            except Exception as e:
                print(f"[Warning] Failed to enable control for {name}: {e}")
        
      
        self._spawn_agents()
        
      
        if self.target_pos is None:
            self.target_pos = self._random_target()
        self._target_last_free_pos = self.target_pos.copy()
        
      
        self.target_vel = np.zeros(3, dtype=np.float64)
        
      
        if "target" in self.vehicle_names:
            self._setup_target_drone()
        
        time.sleep(1.0)
        
      
        self.current_goals[:] = self.target_pos
        self.step_count = 0
        
      
        self.prev_actions = np.zeros((self.num_agents, 4), dtype=np.float64)
        
      
        positions, _, _ = self._get_agent_states()
        for i in range(self.num_agents):
            dx = positions[i][0] - self.target_pos[0]
            dy = positions[i][1] - self.target_pos[1]
            dz = positions[i][2] - self.target_pos[2]
            self.prev_target_dist[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
        self.prev_frontier_dist[:] = 0.0

        # LKP tracking state
        self.last_known_target_pos = self.target_pos.copy()
        self.target_lost_steps = 0
        self.lkp_countdown = self.lkp_countdown_steps
        self.target_mode_flag[:] = 1.0
        self.prev_target_mode_flag[:] = 1.0
        self.obs_target_pos[:] = self.target_pos
        self.obs_target_vel[:] = self.target_vel
        self.prev_obs_target_pos[:] = self.target_pos
        dx = positions[:, 0] - self.obs_target_pos[:, 0]
        dy = positions[:, 1] - self.obs_target_pos[:, 1]
        dz = positions[:, 2] - self.obs_target_pos[:, 2]
        self.path_lengths[:] = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Position history for stagnation detection
        self.position_history[:] = positions[:, None, :]
        self.history_idx = 0
        self.collision_armed[:] = True
        self.collision_rearm_count[:] = 0
        
      
        self._frontier_goals = {}
        self._last_frontier_goals = {}
        self.frontier_deltas[:] = 0.0
        self.exploration_deltas[:] = 0.0
        self.prev_frontier_goals[:] = 0.0
        self.prev_frontier_mask[:] = False
        if hasattr(self, 'voxel_map'):
          
            self.voxel_map.visits = np.zeros(self.occupancy_grid.shape, dtype=np.float32)
            self.voxel_map.hits = np.zeros(self.occupancy_grid.shape, dtype=np.int32)
          
            self.voxel_map.hits[self.occupancy_grid == 1] = 1
            self.voxel_map.visits[self.occupancy_grid == 1] = 1.0
        self._stamp_local_timestamps(positions, self.step_count, radius_vox=1)
        
      
        positions, velocities, yaws = self._get_agent_states()
        self._update_navigation(positions, velocities, yaws)
        self.prev_vel = velocities.copy()
        self.prev_yaw = yaws.copy()
        self.imu_linear_acc_body[:] = 0.0
        self.imu_angular_vel_body[:] = 0.0
        
        print(f"[Reset] Target position: {self.target_pos}")
        print(f"[Reset] Target behavior: {self.target_behavior}, speed: {self.target_speed}m/s")
        print(f"[Reset] View radius: {self.view_radius}m")
        print(f"[Reset] Agent positions: {positions}")
        
        return self._get_observations()
    
    def _spawn_agents(self):
        """Spread pursuer drones across the map and hover after teleporting.

        Expected world-frame XY positions:
        - Agent 0: (-30, -30)
        - Agent 1: (30, -30)
        - Agent 2: (30, 30)
        - Agent 3: (-30, 30)
        """
      
        desired_positions: list[np.ndarray] = []
        max_agents = min(len(self.agent_names), len(SPAWN_POSITIONS))
        for i in range(max_agents):
            ned_pos = SPAWN_POSITIONS[i]
            if self.occupancy_grid is not None:
                ned_pos = self._snap_spawn_to_free(ned_pos)
            desired_positions.append(np.array(ned_pos, dtype=np.float64))
        desired_positions = self._ensure_spawn_separation(desired_positions)

        try:
            self.client.simPause(True)
        except Exception:
            pass

        spawn_ok = False
        for attempt in range(3):
            for i, name in enumerate(self.agent_names):
                if i >= len(desired_positions):
                    break
                
                ned_pos = desired_positions[i]
              
                self._spawn_ned_z[name] = float(ned_pos[2])
                
                pose = airsim.Pose(
                    airsim.Vector3r(float(ned_pos[0]), float(ned_pos[1]), float(ned_pos[2])),
                    airsim.to_quaternion(0, 0, 0)
                )
                try:
                    self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=name)
                except Exception as e:
                    print(f"[Warning] Failed to set pose for {name}: {e}")

          
            time.sleep(0.3)
            self._state_cache = {}
            self._state_cache_timestamp = 0.0
            positions, _, _ = self._get_agent_states()
            if self._spawn_positions_ok(positions, desired_positions):
                spawn_ok = True
                break
            if attempt < 2:
                if self.debug:
                    print(f"[Spawn] Pose mismatch, retrying... (attempt {attempt+1})")

        try:
            self.client.simPause(False)
        except Exception:
            pass

        if not spawn_ok:
            if self.debug:
                print("[Spawn] Pose mismatch after retries, using moveToPositionAsync fallback.")
            for i, name in enumerate(self.agent_names):
                if i >= len(desired_positions):
                    break
                ned_pos = desired_positions[i]
                try:
                    self.client.moveToPositionAsync(
                        float(ned_pos[0]),
                        float(ned_pos[1]),
                        float(ned_pos[2]),
                        velocity=3.0,
                        vehicle_name=name,
                    ).join()
                except Exception as e:
                    print(f"[Warning] Failed to move {name} to spawn: {e}")
            time.sleep(0.3)
            self._state_cache = {}
            self._state_cache_timestamp = 0.0
        
      
        time.sleep(0.3)
        
      
        for i, name in enumerate(self.agent_names):
            if i >= len(desired_positions):
                break
            try:
              
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.5, vehicle_name=name)
            except Exception as e:
                print(f"[Warning] Failed to start hover for {name}: {e}")
        
        time.sleep(0.3)
        
      
        for i, name in enumerate(self.agent_names):
            if i >= len(desired_positions):
                break
            try:
                pose = self.client.simGetVehiclePose(vehicle_name=name)
                expected = desired_positions[i]
                actual_world_z = -pose.position.z_val  # NED -> World
                print(f"[Spawn] {name}: expected=({expected[0]:.0f}, {expected[1]:.0f}, {-expected[2]:.0f}), "
                      f"actual=({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {actual_world_z:.1f})")
            except Exception as e:
                print(f"[Warning] Failed to get pose for {name}: {e}")
    
    def _random_target(self):
        """Sample a random target position inside the map and outside obstacles.

        The world-frame z coordinate is sampled from approximately `[10, 90]`.
        """
        max_attempts = 50
        
        for _ in range(max_attempts):
            if self.occupancy_grid is not None:
                grid_size = np.array(self.occupancy_grid.shape) * self.voxel_size
                margin = 30.0
              
                z_min = max(11.1, self.origin[2] + 5)
                z_max = min(88.4, self.origin[2] + grid_size[2] - 5)
                pos = np.array([
                    np.random.uniform(self.origin[0] + margin, self.origin[0] + grid_size[0] - margin),
                    np.random.uniform(self.origin[1] + margin, self.origin[1] + grid_size[1] - margin),
                    np.random.uniform(z_min, z_max)
                ], dtype=np.float64)
                
              
                ix = int((pos[0] - self.origin[0]) / self.voxel_size)
                iy = int((pos[1] - self.origin[1]) / self.voxel_size)
                iz = int((pos[2] - self.origin[2]) / self.voxel_size)
                
                if 0 <= ix < self.occupancy_grid.shape[0] and \
                   0 <= iy < self.occupancy_grid.shape[1] and \
                   0 <= iz < self.occupancy_grid.shape[2]:
                    if self.occupancy_grid[ix, iy, iz] == 0:
                        print(f"[Target] Found free position at attempt {_+1}: {pos}")
                        return pos
            else:
              
                return np.array([
                    np.random.uniform(-60, 60),
                    np.random.uniform(-60, 60),
                    np.random.uniform(10, 90)
                ], dtype=np.float64)
        
      
        print(f"[Target] WARNING: Could not find free position after {max_attempts} attempts, using center")
        if self.occupancy_grid is not None:
            grid_size = np.array(self.occupancy_grid.shape) * self.voxel_size
            center = self.origin + grid_size / 2
            center[2] = 50.0  #  z=50m
            return center
        return np.array([0, 0, 22.1], dtype=np.float64)
    
    def _setup_target_drone(self):
        """"""
        ned_pos = airsim.Vector3r(
            float(self.target_pos[0]),
            float(self.target_pos[1]),
            float(-self.target_pos[2])  # World z-up -> NED z-down
        )
        pose = airsim.Pose(ned_pos, airsim.to_quaternion(0, 0, 0))
        with self._client_lock:
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name="target")
        self._target_z_ned = float(-self.target_pos[2])
        self._hover_vehicle("target", duration=5.0, z_ned=self._target_z_ned)
        
      
        self._wander_direction = np.random.randn(3)
        self._wander_direction[2] *= 0.3
        self._wander_direction = self._wander_direction / (np.linalg.norm(self._wander_direction) + 1e-6)
        self._direction_change_counter = 0

    def _is_target_pos_free(self, pos: np.ndarray) -> bool:
        if self.occupancy_grid is None:
            return True
        if np.any(pos < self.world_min) or np.any(pos > self.world_max):
            return False
        idx = np.floor((pos - self.origin) / self.voxel_size).astype(int)
        grid_shape = self.occupancy_grid.shape
        if np.any(idx < 0) or np.any(idx >= np.array(grid_shape)):
            return False
        return self.occupancy_grid[tuple(idx)] == 0

    def _resolve_target_velocity(self, vel: np.ndarray) -> np.ndarray:
        if np.linalg.norm(vel) < 1e-6:
            return vel
        next_pos = self.target_pos + vel * self.dt
        if self._is_target_pos_free(next_pos):
            return vel
        v_xy = vel.copy()
        v_xy[2] = 0.0
        vxy_norm = np.linalg.norm(v_xy[:2])
        if vxy_norm > 1e-6:
            lateral = np.array([-v_xy[1], v_xy[0], 0.0], dtype=np.float64)
            lateral = lateral / (np.linalg.norm(lateral) + 1e-6) * vxy_norm
            for sign in (1.0, -1.0):
                cand = vel.copy()
                cand[0] = lateral[0] * sign
                cand[1] = lateral[1] * sign
                cand[2] = vel[2] * 0.2
                if self._is_target_pos_free(self.target_pos + cand * self.dt):
                    return cand
        return np.zeros(3, dtype=np.float64)

    def _get_target_clamp_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.occupancy_grid is None:
            return self.world_min, self.world_max
        gx, gy, gz = self.occupancy_grid.shape
        occ_min = self.origin
        occ_max = np.array(
            [origin + dim * self.voxel_size for origin, dim in zip(self.origin, (gx, gy, gz))],
            dtype=np.float64,
        )
        clamp_min = np.maximum(self.world_min, occ_min)
        clamp_max = np.minimum(self.world_max, occ_max)
        return clamp_min, clamp_max

    def _compute_target_state_training_like(self, agent_positions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Mirrors batch_update_targets_v2 in training.
        repulse_radius = 50.0
        min_repulse_dist = 10.0
        boundary_margin = 12.0
        boundary_hard_margin = 3.0
        velocity_smoothing = 0.4
        obstacle_weight = 1.8
        boundary_weight = 1.0

        pos = self.target_pos.copy()
        vel = self.target_vel.copy()
        speed = float(self.target_speed)
        clamp_min, clamp_max = self._get_target_clamp_bounds()

        if self.target_behavior == "static" or speed <= 0.0:
            new_v = np.zeros(3, dtype=np.float64)
        elif self.target_behavior == "wander":
            jitter = np.random.normal(0.0, 1.0, size=3) * 0.2
            jitter[2] = 0.0
            new_v = vel + jitter
        else:
            repulse_force = np.zeros(3, dtype=np.float64)
            for agent_pos in agent_positions:
                delta = pos - agent_pos
                dist = np.linalg.norm(delta)
                if dist < repulse_radius and dist > 1e-6:
                    effective_dist = max(dist, min_repulse_dist)
                    weight = (repulse_radius - dist) / (effective_dist * effective_dist)
                    repulse_force += (delta / dist) * weight

            repulse_norm = np.linalg.norm(repulse_force)
            if repulse_norm > 1e-6:
                repulse_force = repulse_force / repulse_norm
            else:
                repulse_force = np.random.normal(0.0, 1.0, size=3)
                repulse_norm = np.linalg.norm(repulse_force)
                if repulse_norm > 1e-6:
                    repulse_force = repulse_force / repulse_norm

            boundary_force = np.zeros(3, dtype=np.float64)
            for k in range(2):
                dist_to_min = pos[k] - clamp_min[k]
                if dist_to_min < boundary_margin:
                    if dist_to_min < boundary_hard_margin:
                        boundary_force[k] += 3.0
                    else:
                        strength = (boundary_margin - dist_to_min) / boundary_margin
                        boundary_force[k] += strength * boundary_weight

                dist_to_max = clamp_max[k] - pos[k]
                if dist_to_max < boundary_margin:
                    if dist_to_max < boundary_hard_margin:
                        boundary_force[k] -= 3.0
                    else:
                        strength = (boundary_margin - dist_to_max) / boundary_margin
                        boundary_force[k] -= strength * boundary_weight

            dist_to_z_min = pos[2] - clamp_min[2]
            dist_to_z_max = clamp_max[2] - pos[2]
            if dist_to_z_min < boundary_margin * 0.5:
                boundary_force[2] += (boundary_margin * 0.5 - dist_to_z_min) / (boundary_margin * 0.5)
            if dist_to_z_max < boundary_margin * 0.5:
                boundary_force[2] -= (boundary_margin * 0.5 - dist_to_z_max) / (boundary_margin * 0.5)

            obstacle_force = np.zeros(3, dtype=np.float64)
            if self.occupancy_grid is not None and self.target_avoid_radius_vox > 0:
                pos_clamped = np.clip(pos, clamp_min, clamp_max)
                ix = int(np.floor((pos_clamped[0] - self.origin[0]) / self.voxel_size))
                iy = int(np.floor((pos_clamped[1] - self.origin[1]) / self.voxel_size))
                iz = int(np.floor((pos_clamped[2] - self.origin[2]) / self.voxel_size))

                gx, gy, gz = self.occupancy_grid.shape
                radius = self.target_avoid_radius_vox
                radius_m = radius * self.voxel_size
                for ddx in range(-radius, radius + 1):
                    nx = ix + ddx
                    if nx < 0 or nx >= gx:
                        continue
                    for ddy in range(-radius, radius + 1):
                        ny = iy + ddy
                        if ny < 0 or ny >= gy:
                            continue
                        for ddz in range(-radius, radius + 1):
                            nz = iz + ddz
                            if nz < 0 or nz >= gz:
                                continue
                            if self.occupancy_grid[nx, ny, nz] != 1:
                                continue

                            ox = self.origin[0] + (nx + 0.5) * self.voxel_size
                            oy = self.origin[1] + (ny + 0.5) * self.voxel_size
                            oz = self.origin[2] + (nz + 0.5) * self.voxel_size
                            delta = pos - np.array([ox, oy, oz], dtype=np.float64)
                            dist = np.linalg.norm(delta)
                            if dist < 1e-6 or dist > radius_m:
                                continue
                            w = ((radius_m - dist) / radius_m) ** 2
                            obstacle_force += (delta / dist) * w

                obs_norm = np.linalg.norm(obstacle_force)
                if obs_norm > 1e-6:
                    obstacle_force = obstacle_force / obs_norm * obstacle_weight

            target_dir = repulse_force + boundary_force + obstacle_force
            dir_norm = np.linalg.norm(target_dir)
            if dir_norm > 1e-6:
                target_v = target_dir / dir_norm * speed
            else:
                target_v = np.zeros(3, dtype=np.float64)
            new_v = vel * velocity_smoothing + target_v * (1.0 - velocity_smoothing)

        # Match training: clamp per-axis speed to agent action bounds (optional)
        if self.target_axis_limit_enabled:
            if self.action_scale_vx > 0.0:
                new_v[0] = min(max(new_v[0], -self.action_scale_vx), self.action_scale_vx)
            if self.action_scale_vy > 0.0:
                new_v[1] = min(max(new_v[1], -self.action_scale_vy), self.action_scale_vy)
            if self.action_scale_vz > 0.0:
                new_v[2] = min(max(new_v[2], -self.action_scale_vz), self.action_scale_vz)

        v_speed = np.linalg.norm(new_v)
        if v_speed > speed and v_speed > 1e-6:
            new_v = new_v / v_speed * speed

        new_pos = pos + new_v * self.dt
        for i in range(3):
            if new_pos[i] < clamp_min[i]:
                new_pos[i] = clamp_min[i]
                if new_v[i] < 0:
                    new_v[i] = abs(new_v[i]) * 0.5
            elif new_pos[i] > clamp_max[i]:
                new_pos[i] = clamp_max[i]
                if new_v[i] > 0:
                    new_v[i] = -abs(new_v[i]) * 0.5

        if self.occupancy_grid is not None:
            gx, gy, gz = self.occupancy_grid.shape
            ix = int(np.floor((new_pos[0] - self.origin[0]) / self.voxel_size))
            iy = int(np.floor((new_pos[1] - self.origin[1]) / self.voxel_size))
            iz = int(np.floor((new_pos[2] - self.origin[2]) / self.voxel_size))
            if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
                if self.occupancy_grid[ix, iy, iz] == 1:
                    best_dist2 = 1e12
                    best_center = new_pos.copy()
                    radius = self.target_avoid_radius_vox + 2
                    for ddx in range(-radius, radius + 1):
                        nx = ix + ddx
                        if nx < 0 or nx >= gx:
                            continue
                        for ddy in range(-radius, radius + 1):
                            ny = iy + ddy
                            if ny < 0 or ny >= gy:
                                continue
                            for ddz in range(-radius, radius + 1):
                                nz = iz + ddz
                                if nz < 0 or nz >= gz:
                                    continue
                                if self.occupancy_grid[nx, ny, nz] != 0:
                                    continue
                                center = self.origin + (np.array([nx, ny, nz], dtype=np.float64) + 0.5) * self.voxel_size
                                dist2 = np.sum((center - new_pos) ** 2)
                                if dist2 < best_dist2:
                                    best_dist2 = dist2
                                    best_center = center
                    new_pos = best_center
                    new_v = np.zeros(3, dtype=np.float64)

        return new_pos, new_v
    
    def _update_target(self, agent_positions):
        """"""
        try:
            if self.occupancy_grid is not None:
                grid_size = np.array(self.occupancy_grid.shape, dtype=np.float64) * self.voxel_size
                world_min = self.origin
                world_max = self.origin + grid_size
                self.target_pos = np.clip(self.target_pos, world_min, world_max)
                self._target_z_ned = float(-self.target_pos[2])

            new_pos, new_vel = self._compute_target_state_training_like(agent_positions)
            self.target_pos = new_pos
            self.target_vel = new_vel.copy()
            self._target_z_ned = float(-self.target_pos[2])
            vel_ned = np.array([new_vel[0], new_vel[1], -new_vel[2]], dtype=np.float64)
            with self._client_lock:
                self.client.moveByVelocityAsync(
                    float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]),
                    duration=float(self.dt),
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                    vehicle_name="target",
                )
                pose = self.client.simGetVehiclePose(vehicle_name="target")
            self.target_pos = np.array([
                pose.position.x_val,
                pose.position.y_val,
                -pose.position.z_val
            ], dtype=np.float64)
            if not self._is_target_pos_free(self.target_pos):
                safe_pos = self._target_last_free_pos
                if safe_pos is None or not self._is_target_pos_free(safe_pos):
                    safe_pos = self._snap_goal_to_free(self.target_pos)
                self.target_pos = safe_pos
                self._target_z_ned = float(-self.target_pos[2])
                with self._client_lock:
                    pose = airsim.Pose(
                        airsim.Vector3r(float(self.target_pos[0]), float(self.target_pos[1]), float(self._target_z_ned)),
                        airsim.to_quaternion(0, 0, 0)
                    )
                    self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name="target")
                self.target_vel = np.zeros(3, dtype=np.float64)
            else:
                self._target_last_free_pos = self.target_pos.copy()
                
        except Exception as e:
            if self.debug:
                print(f"[Target] Update error: {e}")
    
    def _get_agent_states(self):
        """ agent  - """
        current_time = time.perf_counter()
        
      
        if (current_time - self._state_cache_timestamp) < self._state_cache_ttl and self._state_cache:
            return self._state_cache['positions'], self._state_cache['velocities'], self._state_cache['yaws']
        
      
        positions = []
        velocities = []
        yaws = []
        
        for name in self.agent_names:
            try:
                with self._client_lock:
                  
                    pose = self.client.simGetVehiclePose(vehicle_name=name)
                    state = self.client.getMultirotorState(vehicle_name=name)
                
                vel = state.kinematics_estimated.linear_velocity
                
                # NED -> World (z-up)
                pos = np.array([
                    pose.position.x_val,
                    pose.position.y_val,
                    -pose.position.z_val
                ], dtype=np.float64)
                v = np.array([vel.x_val, vel.y_val, -vel.z_val], dtype=np.float64)
                yaw = self._quat_to_yaw(pose.orientation)
                
                positions.append(pos)
                velocities.append(v)
                yaws.append(yaw)
            except Exception as e:
                if self.debug:
                    print(f"[Warning] Failed to get state for {name}: {e}")
                positions.append(np.zeros(3))
                velocities.append(np.zeros(3))
                yaws.append(0.0)
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        yaws = np.array(yaws)
        
      
        self._state_cache = {
            'positions': positions,
            'velocities': velocities,
            'yaws': yaws
        }
        self._state_cache_timestamp = current_time
        
        return positions, velocities, yaws
    
    def _quat_to_yaw(self, q):
        """ yaw """
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _build_lidar_dirs_body(self) -> np.ndarray:
        """Build 26 unit directions in body frame (match training order)."""
        dirs = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    vec = np.array([dx, dy, dz], dtype=np.float64)
                    vec /= np.linalg.norm(vec)
                    dirs.append(vec)
        return np.stack(dirs, axis=0)

    def _build_lidar_sub_dirs_body(self, sub_rays: int, spread_deg: float) -> np.ndarray:
        """Build sub-ray directions per sector to match training min-pooling logic."""
        rays = max(1, int(sub_rays))
        spread_rad = float(spread_deg) * np.pi / 180.0
        cone_cos = np.cos(spread_rad)
        cone_sin = np.sin(spread_rad)
        base_dirs = self._lidar_dirs_body
        sub_dirs = np.zeros((base_dirs.shape[0], rays, 3), dtype=np.float64)
        for i, d in enumerate(base_dirs):
            dir_x, dir_y, dir_z = float(d[0]), float(d[1]), float(d[2])
            up_x, up_y, up_z = 0.0, 0.0, 1.0
            ux = dir_y * up_z - dir_z * up_y
            uy = dir_z * up_x - dir_x * up_z
            uz = dir_x * up_y - dir_y * up_x
            u_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
            if u_norm < 1e-6:
                up_x, up_y, up_z = 1.0, 0.0, 0.0
                ux = dir_y * up_z - dir_z * up_y
                uy = dir_z * up_x - dir_x * up_z
                uz = dir_x * up_y - dir_y * up_x
                u_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
            inv_un = 1.0 / max(u_norm, 1e-6)
            ux *= inv_un
            uy *= inv_un
            uz *= inv_un
            vx = dir_y * uz - dir_z * uy
            vy = dir_z * ux - dir_x * uz
            vz = dir_x * uy - dir_y * ux
            for s in range(rays):
                if s == 0 or rays == 1:
                    sub_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
                else:
                    angle = 2.0 * np.pi * (s - 1) / max(1, rays - 1)
                    ca = np.cos(angle)
                    sa = np.sin(angle)
                    off_x = ca * ux + sa * vx
                    off_y = ca * uy + sa * vy
                    off_z = ca * uz + sa * vz
                    sub_dir = np.array([
                        dir_x * cone_cos + off_x * cone_sin,
                        dir_y * cone_cos + off_y * cone_sin,
                        dir_z * cone_cos + off_z * cone_sin,
                    ], dtype=np.float64)
                sub_dirs[i, s] = sub_dir
        return sub_dirs

    def _get_lidar_point_cloud(self, agent_idx: int) -> np.ndarray | None:
        if agent_idx < 0 or agent_idx >= len(self.agent_names):
            return None
        name = self.agent_names[agent_idx]
        try:
            with self._client_lock:
                data = self.client.getLidarData(
                    lidar_name=self.lidar_sensor_name,
                    vehicle_name=name,
                )
        except Exception as e:
            if self.debug:
                print(f"[LiDAR] Failed to get data for {name}: {e}")
            return None
        if not data or len(data.point_cloud) < 3:
            return None
        pts = np.array(data.point_cloud, dtype=np.float64).reshape(-1, 3)
        # AirSim uses NED (z-down); convert to z-up to match training frame.
        pts[:, 2] *= -1.0
        return pts
    
    def _simulate_lidar(self, pos, yaw, agent_idx=0, force_refresh=False):
        """Read LiDAR from AirSim and map to 26-sector distances."""
        cache_key = agent_idx
        use_cache = (not self.debug) and (not force_refresh)
        if (use_cache and cache_key in self._lidar_cache_step and
            self.step_count - self._lidar_cache_step[cache_key] < self._lidar_update_interval):
            if cache_key in self._lidar_cache_count:
                self._lidar_last_count[agent_idx] = self._lidar_cache_count[cache_key]
            return self._lidar_cache[cache_key]

        distances_m = np.full(26, self.lidar_max_range, dtype=np.float64)
        pts = self._get_lidar_point_cloud(agent_idx)
        point_count = 0
        if pts is not None and pts.size > 0:
            point_count = int(pts.shape[0])
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 1e-6
            if np.any(mask):
                dists = dists[mask]
                pts = pts[mask] / dists[:, None]
                sub_dirs = self._lidar_sub_dirs_body.reshape(-1, 3)
                dots = pts @ sub_dirs.T
                dots = dots.reshape(pts.shape[0], 26, -1)
                dots = np.where(dots >= self._lidar_sub_cos, dots, -1.0)
                sector_scores = np.max(dots, axis=2)
                idxs = np.argmax(sector_scores, axis=1)
                valid = np.max(sector_scores, axis=1) > 0.0
                for dist, idx, ok in zip(dists, idxs, valid):
                    if ok and dist < distances_m[idx]:
                        distances_m[idx] = dist
        else:
            if self.debug:
                print("[LiDAR] Empty point cloud; falling back to max range")

        distances = np.clip(distances_m / max(self.lidar_max_range, 1e-6), 0.0, 1.0).astype(np.float32)
        self._lidar_cache[cache_key] = distances.copy()
        self._lidar_cache_step[cache_key] = self.step_count
        self._lidar_cache_count[cache_key] = point_count
        self._lidar_last_count[agent_idx] = point_count
        return distances

    def _stamp_local_timestamps(self, positions, step, radius_vox=1):
        """Mark local voxels around each agent as recently visited."""
        if self.batch_timestamps is None:
            return
        step_val = int(step)
        if step_val <= 0:
            step_val = 1
        grid = self.batch_timestamps[0]
        gx, gy, gz = grid.shape
        for pos in positions:
            local = pos - self.origin
            ix = int(np.floor(local[0] / self.voxel_size))
            iy = int(np.floor(local[1] / self.voxel_size))
            iz = int(np.floor(local[2] / self.voxel_size))
            for dx in range(-radius_vox, radius_vox + 1):
                for dy in range(-radius_vox, radius_vox + 1):
                    for dz in range(-radius_vox, radius_vox + 1):
                        nx = ix + dx
                        ny = iy + dy
                        nz = iz + dz
                        if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                            if self.occupancy_grid[nx, ny, nz] == 0:
                                grid[nx, ny, nz] = step_val

    def _has_line_of_sight(self, start, end):
        """Check if line segment is unobstructed in occupancy grid."""
        if self.occupancy_grid is None:
            return True
        gx, gy, gz = self.occupancy_grid.shape
        x0 = int((start[0] - self.origin[0]) / self.voxel_size)
        y0 = int((start[1] - self.origin[1]) / self.voxel_size)
        z0 = int((start[2] - self.origin[2]) / self.voxel_size)
        x1 = int((end[0] - self.origin[0]) / self.voxel_size)
        y1 = int((end[1] - self.origin[1]) / self.voxel_size)
        z1 = int((end[2] - self.origin[2]) / self.voxel_size)

        if x0 < 0 or x0 >= gx or y0 < 0 or y0 >= gy or z0 < 0 or z0 >= gz:
            return False
        if x1 < 0 or x1 >= gx or y1 < 0 or y1 >= gy or z1 < 0 or z1 >= gz:
            return False
        if self.occupancy_grid[x0, y0, z0] == 1 or self.occupancy_grid[x1, y1, z1] == 1:
            return False

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
                if self.occupancy_grid[x0, y0, z0] == 1:
                    return False
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
                if self.occupancy_grid[x0, y0, z0] == 1:
                    return False
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
                if self.occupancy_grid[x0, y0, z0] == 1:
                    return False

        return True
    
    def _update_navigation(self, positions, velocities, yaws):
        """Update A* navigation (aligned with training environment)."""
        # Update exploration mask for frontier allocation
        if self.batch_timestamps is not None and hasattr(self, 'voxel_map'):
            self.voxel_map.visits = self._get_exploration_mask()

        # Visibility mask (batch-style)
        stage_name = self.stage_config.name if self.stage_config else ""
        force_visible = np.array([False], dtype=np.bool_)
        if self.occupancy_grid is not None:
            all_visible, any_visible_arr = batch_visibility_mask(
                positions[None, ...],
                self.target_pos[None, ...],
                self.target_vel[None, ...],
                np.array([self.view_radius], dtype=np.float64),
                force_visible,
                self.occupancy_grid,
                self.origin,
                self.voxel_size,
            )
            visible_mask = all_visible[0]
            any_visible = bool(any_visible_arr[0])
        else:
            dists_vis = np.linalg.norm(positions - self.target_pos[None, :], axis=1)
            visible_mask = dists_vis <= self.view_radius
            any_visible = bool(np.any(visible_mask))
        if self.debug_frontier and self.step_count % 10 == 0:
            visible_count = int(np.sum(visible_mask))
            print(f"[Vis-Debug] step={self.step_count} any_visible={any_visible} "
                  f"visible_count={visible_count} view_radius={self.view_radius:.1f} "
                  f"lidar_max_range={self.lidar_max_range:.1f}")
        self._any_visible = np.array([any_visible], dtype=bool)
        # 
        self.prev_target_mode_flag[:] = self.target_mode_flag

        dists = np.linalg.norm(positions - self.target_pos[None, :], axis=1)

        # LKP tracking update (no breakout here; aligned with training _update_navigation)
        if any_visible:
            self.last_known_target_pos = self.target_pos.copy()
            self.target_lost_steps = 0
            self.lkp_countdown = self.lkp_countdown_steps
            self.target_mode_flag[:] = 1.0
        else:
            self.target_lost_steps += 1
            if self.lkp_countdown > 0:
                self.lkp_countdown -= 1
                self.target_mode_flag[:] = 1.0
            else:
                self.target_mode_flag[:] = 0.0

        # Frontier allocation
        frontier_goals = {}
        selected_goals = None
        has_search_agents = np.any(self.target_mode_flag <= 0.5)
        if self.guidance_enabled and has_search_agents and hasattr(self, 'frontier_allocator'):
            headings = np.zeros((self.num_agents, 3), dtype=np.float64)
            headings[:, 0] = np.cos(yaws)
            headings[:, 1] = np.sin(yaws)

            for n in range(self.num_agents):
                if self.target_mode_flag[n] < 0.5 and self.prev_target_mode_flag[n] > 0.5:
                    self.frontier_allocator.mark_agent_lost_target(n, True)

            frontier_goals = self.frontier_allocator.allocate(
                positions,
                headings,
                self.voxel_map,
                self.step_count,
                lkp_pos=self.last_known_target_pos,
                target_lost_steps=int(self.target_lost_steps),
            )

        self._last_frontier_goals = frontier_goals

        if self.guidance_enabled:
            frontier_goals_arr = np.repeat(self.target_pos[None, :], self.num_agents, axis=0)
            if frontier_goals:
                for n_idx, goal in frontier_goals.items():
                    if goal is not None:
                        frontier_goals_arr[n_idx] = goal
            selected_goals = batch_select_goals(
                positions[None, ...],
                self.target_pos[None, ...],
                frontier_goals_arr[None, ...],
                visible_mask[None, ...],
                np.array([self.guidance_enabled], dtype=np.bool_),
            )[0]

        # Frontier bonus bookkeeping
        if self.guidance_enabled and selected_goals is not None:
            for n in range(self.num_agents):
                is_searching = self.target_mode_flag[n] <= 0.5
                if not is_searching:
                    self.prev_frontier_mask[n] = False
                    continue
                goal = selected_goals[n]
                dist_to_target = np.linalg.norm(goal - self.target_pos)
                is_frontier = dist_to_target > 1e-6
                if is_frontier:
                    prev_goal = self.prev_frontier_goals[n]
                    moved = np.linalg.norm(goal - prev_goal) > self.voxel_size * 0.5
                    if not self.prev_frontier_mask[n] or moved:
                        self.frontier_deltas[n] = 1.0
                    self.prev_frontier_goals[n] = goal
                    self.prev_frontier_mask[n] = True
                else:
                    self.prev_frontier_mask[n] = False

        goals = np.zeros((self.num_agents, 3), dtype=np.float64)
        guidance = np.zeros((self.num_agents, 3), dtype=np.float64)
        paths = [None] * self.num_agents

        for i in range(self.num_agents):
            final_goal = self.target_pos.copy()

            if self.target_mode_flag[i] > 0.5:
                if any_visible:
                    predicted_target = self.target_pos + self.target_vel * self.dt * 5.0
                    safe_target = self.target_pos
                    if self.occupancy_grid is not None:
                        in_bounds = np.all(predicted_target >= self.world_min) and np.all(predicted_target <= self.world_max)
                        if in_bounds:
                            pred_idx = self.voxel_map.pos_to_idx(predicted_target)
                            if (0 <= pred_idx[0] < self.occupancy_grid.shape[0] and
                                0 <= pred_idx[1] < self.occupancy_grid.shape[1] and
                                0 <= pred_idx[2] < self.occupancy_grid.shape[2]):
                                if self.occupancy_grid[pred_idx[0], pred_idx[1], pred_idx[2]] <= 0.5:
                                    safe_target = predicted_target
                    final_goal = safe_target
                    obs_target_pos = self.target_pos
                    obs_target_vel = self.target_vel
                else:
                    obs_target_pos = self.last_known_target_pos
                    obs_target_vel = np.zeros(3, dtype=np.float64)
                    final_goal = self.last_known_target_pos
            else:
                if self.guidance_enabled and selected_goals is not None:
                    obs_target_pos = selected_goals[i]
                    obs_target_vel = np.zeros(3, dtype=np.float64)
                    final_goal = selected_goals[i]
                else:
                    obs_target_pos = self.last_known_target_pos
                    obs_target_vel = np.zeros(3, dtype=np.float64)
                    final_goal = self.last_known_target_pos

            self.obs_target_pos[i] = obs_target_pos
            self.obs_target_vel[i] = obs_target_vel

            if self.guidance_enabled:
                if visible_mask[i]:
                    scale = min(1.0, max(0.0, dists[i] / self.offset_decay_distance))
                    final_goal = final_goal + self.goal_offsets[i] * scale

                direct_dist = dists[i]
                if visible_mask[i] and direct_dist < self.straight_replace_astar_distance:
                    direction = final_goal - positions[i]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                    else:
                        direction = np.zeros(3, dtype=np.float64)
                    path = [positions[i].copy(), final_goal.copy()]
                    path_len = norm
                else:
                    if self.navigator is not None:
                        direction, path, path_len = self.navigator.compute_direction(
                            start_position=positions[i],
                            goal_position=final_goal,
                            current_step=int(self.step_count),
                            current_velocity=velocities[i],
                            smooth=False,
                            cache_key=(0, int(i)),
                        )
                    else:
                        direction = final_goal - positions[i]
                        norm = np.linalg.norm(direction)
                        if norm > 1e-6:
                            direction = direction / norm
                        else:
                            direction = np.zeros(3, dtype=np.float64)
                        path = [positions[i].copy(), final_goal.copy()]
                        path_len = norm

                guidance[i] = direction
                goals[i] = final_goal
                if path_len <= 0.0:
                    dx = final_goal[0] - positions[i, 0]
                    dy = final_goal[1] - positions[i, 1]
                    dz = final_goal[2] - positions[i, 2]
                    path_len = np.sqrt(dx*dx + dy*dy + dz*dz)
                self.path_lengths[i] = path_len
                paths[i] = path
            else:
                goals[i] = final_goal
                guidance[i] = 0.0
                dx = final_goal[0] - positions[i, 0]
                dy = final_goal[1] - positions[i, 1]
                dz = final_goal[2] - positions[i, 2]
                self.path_lengths[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
                paths[i] = [positions[i].copy(), final_goal.copy()]

        self.current_goals = goals
        self.guidance_vectors = guidance
        self.current_paths = paths

    def _update_target_tracking(self, positions):
        """Update LKP target tracking when navigation is not refreshed."""
        stage_name = self.stage_config.name if self.stage_config else ""
        force_visible = np.array([False], dtype=np.bool_)
        if self.occupancy_grid is not None:
            all_visible, any_visible_arr = batch_visibility_mask(
                positions[None, ...],
                self.target_pos[None, ...],
                self.target_vel[None, ...],
                np.array([self.view_radius], dtype=np.float64),
                force_visible,
                self.occupancy_grid,
                self.origin,
                self.voxel_size,
            )
            visible_mask = all_visible[0]
            any_visible = bool(any_visible_arr[0])
        else:
            dists_vis = np.linalg.norm(positions - self.target_pos[None, :], axis=1)
            visible_mask = dists_vis <= self.view_radius
            any_visible = bool(np.any(visible_mask))
        if self.debug_frontier and self.step_count % 10 == 0:
            visible_count = int(np.sum(visible_mask))
            print(f"[Vis-Debug] step={self.step_count} any_visible={any_visible} "
                  f"visible_count={visible_count} view_radius={self.view_radius:.1f} "
                  f"lidar_max_range={self.lidar_max_range:.1f}")
        self._any_visible = np.array([any_visible], dtype=bool)

        # 
        self.prev_target_mode_flag[:] = self.target_mode_flag

        team_has_target = any_visible
        lkp_reached = batch_check_lkp_reached(
            positions[None, ...],
            self.last_known_target_pos[None, ...],
            self.lkp_breakout_dist_m,
        )[0]

        if team_has_target:
            self.last_known_target_pos = self.target_pos.copy()
            self.target_lost_steps = 0
            self.lkp_countdown = self.lkp_countdown_steps
            self.target_mode_flag[:] = 1.0
        else:
            self.target_lost_steps += 1
            if self.lkp_countdown > 0:
                self.lkp_countdown -= 1
                if lkp_reached:
                    self.lkp_countdown = 0
                    self.target_mode_flag[:] = 0.0
                else:
                    self.target_mode_flag[:] = 1.0
            else:
                self.target_mode_flag[:] = 0.0

        for i in range(self.num_agents):
            if self.target_mode_flag[i] > 0.5:
                if team_has_target:
                    self.obs_target_pos[i] = self.target_pos
                    self.obs_target_vel[i] = self.target_vel
                else:
                    self.obs_target_pos[i] = self.last_known_target_pos
                    self.obs_target_vel[i] = np.zeros(3, dtype=np.float64)
            else:
                if self.guidance_enabled:
                    self.obs_target_pos[i] = self.current_goals[i]
                else:
                    self.obs_target_pos[i] = self.last_known_target_pos
                self.obs_target_vel[i] = np.zeros(3, dtype=np.float64)

    def _update_exploration(self, positions):
        """Update exploration confidence with decay and re-observation.

        1. Decay confidence for explored free voxels.
        2. Reset visible voxels to confidence `1.0`.
        3. Treat low-confidence voxels as candidates for re-exploration.
        """
        if not hasattr(self, 'voxel_map'):
            return
        
      
        free_mask = self.occupancy_grid == 0
        self.voxel_map.visits[free_mask] *= self.confidence_decay
        
      
        radius = self.view_radius
        radius_sq = radius * radius
        max_idx = np.array(self.voxel_map.shape, dtype=int) - 1
        
        for pos in positions:
            local = pos - self.voxel_map.origin
            drone_idx = np.floor(local / self.voxel_size).astype(int)
            drone_idx = np.clip(drone_idx, 0, max_idx)
            
          
            min_corner = np.floor((local - radius) / self.voxel_size).astype(int)
            max_corner = np.floor((local + radius) / self.voxel_size).astype(int)
            min_corner = np.clip(min_corner, 0, max_idx)
            max_corner = np.clip(max_corner, 0, max_idx)
            
          
            for x in range(min_corner[0], max_corner[0] + 1):
                for y in range(min_corner[1], max_corner[1] + 1):
                    for z in range(min_corner[2], max_corner[2] + 1):
                      
                        voxel_center = (np.array([x, y, z], dtype=np.float64) + 0.5) * self.voxel_size + self.voxel_map.origin
                        dist_sq = np.sum((voxel_center - pos) ** 2)
                        
                        if dist_sq <= radius_sq:
                          
                            if self.occupancy_grid[x, y, z] == 0:  # 
                                self.voxel_map.visits[x, y, z] = 1.0

    def _get_exploration_mask(self):
        """Convert timestamps to confidence grid for frontier detection."""
        if self.batch_timestamps is None:
            return self.voxel_map.visits
        ts = self.batch_timestamps[0]
        confidence = np.zeros_like(ts, dtype=np.float32)
        visited = ts > 0
        if np.any(visited):
            delta_steps = self.step_count - ts[visited]
            confidence[visited] = np.power(self.confidence_decay, delta_steps)
        return confidence
    
    def _fallback_frontier_goal(self, pos, vel):
        """ FrontierAllocator """
      
        vel_norm = np.linalg.norm(vel[:2])
        if vel_norm > 0.5:
            heading = vel[:2] / vel_norm
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            heading = np.array([np.cos(angle), np.sin(angle)])
        
      
        search_distance = min(60.0, self.view_radius)
        goal = np.array([
            pos[0] + heading[0] * search_distance,
            pos[1] + heading[1] * search_distance,
            pos[2]
        ], dtype=np.float64)
        
      
        if self.occupancy_grid is not None:
            grid_size = np.array(self.occupancy_grid.shape) * self.voxel_size
            margin = 20.0
            goal[0] = np.clip(goal[0], self.origin[0] + margin, self.origin[0] + grid_size[0] - margin)
            goal[1] = np.clip(goal[1], self.origin[1] + margin, self.origin[1] + grid_size[1] - margin)
            goal[2] = np.clip(goal[2], self.origin[2] + 5, self.origin[2] + grid_size[2] - 5)
        
        return goal

    def _snap_goal_to_free(self, goal):
        """Snap planning goal to nearest free cell if it falls inside an occupied voxel."""
        if self.occupancy_grid is None:
            return goal
        grid_shape = self.occupancy_grid.shape
        idx = np.floor((goal - self.origin) / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(grid_shape)):
            return goal
        if self.occupancy_grid[tuple(idx)] == 0:
            return goal
        best = None
        best_dist = float("inf")
        max_r = 8
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        nx = idx[0] + dx
                        ny = idx[1] + dy
                        nz = idx[2] + dz
                        if nx < 0 or ny < 0 or nz < 0 or nx >= grid_shape[0] or ny >= grid_shape[1] or nz >= grid_shape[2]:
                            continue
                        if self.occupancy_grid[nx, ny, nz] == 1:
                            continue
                        dist = dx * dx + dy * dy + dz * dz
                        if dist < best_dist:
                            best_dist = dist
                            best = (nx, ny, nz)
            if best is not None:
                break
        if best is None:
            return goal
        return (np.array(best, dtype=np.float64) + 0.5) * self.voxel_size + self.origin

    def _snap_spawn_to_free(self, pos):
        """Snap spawn position to a nearby free cell in occupancy grid."""
        if self.occupancy_grid is None:
            return pos
        grid_shape = self.occupancy_grid.shape
        idx = np.floor((pos - self.origin) / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(grid_shape)):
            return pos
        if self.occupancy_grid[tuple(idx)] == 0:
            return pos
        best = None
        best_dist = float("inf")
        max_r = 6
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        nx = idx[0] + dx
                        ny = idx[1] + dy
                        nz = idx[2] + dz
                        if nx < 0 or ny < 0 or nz < 0 or nx >= grid_shape[0] or ny >= grid_shape[1] or nz >= grid_shape[2]:
                            continue
                        if self.occupancy_grid[nx, ny, nz] == 1:
                            continue
                        dist = dx * dx + dy * dy + dz * dz
                        if dist < best_dist:
                            best_dist = dist
                            best = (nx, ny, nz)
            if best is not None:
                break
        if best is None:
            return pos
        return (np.array(best, dtype=np.float64) + 0.5) * self.voxel_size + self.origin

    def _get_observations(self):
        """Build observations with the same semantic layout as the training env."""
        positions, velocities, yaws = self._get_agent_states()
        N = self.num_agents

        # Compute IMU signals from velocity/yaw deltas
        if self.dt > 0.0:
            accel_world = (velocities - self.prev_vel) / self.dt
            dyaw = (yaws - self.prev_yaw + np.pi) % (2.0 * np.pi) - np.pi
            yaw_rate = dyaw / self.dt
        else:
            accel_world = np.zeros_like(velocities)
            yaw_rate = np.zeros_like(yaws)
        lidar = np.zeros((1, N, 26), dtype=np.float64)
        teammate_feats = np.zeros((1, N, self.top_k * 8), dtype=np.float64)
        for i in range(N):
            lidar[0, i] = self._simulate_lidar(positions[i], yaws[i])
            teammate_feats[0, i] = self._get_teammate_features(i, positions, velocities, yaws[i])
            cy, sy = np.cos(yaws[i]), np.sin(yaws[i])
            ax = accel_world[i, 0]
            ay = accel_world[i, 1]
            az = accel_world[i, 2]
            acc_bx = ax * cy + ay * sy
            acc_by = -ax * sy + ay * cy
            acc_bz = az
            if self.imu_accel_noise_std > 0.0:
                noise = np.random.normal(0.0, self.imu_accel_noise_std, size=3)
                acc_bx += noise[0]
                acc_by += noise[1]
                acc_bz += noise[2]
            self.imu_linear_acc_body[i] = (acc_bx, acc_by, acc_bz)
            gyro = np.array([0.0, 0.0, float(yaw_rate[i])], dtype=np.float64)
            if self.imu_gyro_noise_std > 0.0:
                gyro += np.random.normal(0.0, self.imu_gyro_noise_std, size=3)
            self.imu_angular_vel_body[i] = gyro

        self.slot_positions, self.slot_assignments = self._build_tactical_slots(positions)
        obs_core = batch_build_observations_v2(
            positions[None, ...],
            velocities[None, ...],
            yaws[None, ...],
            self.target_pos[None, ...],
            self.target_vel[None, ...],
            self.obs_target_pos[None, ...],
            self.obs_target_vel[None, ...],
            self.target_mode_flag[None, ...],
            lidar,
            self.guidance_vectors[None, ...],
            teammate_feats,
            self.imu_linear_acc_body[None, ...],
            self.imu_angular_vel_body[None, ...],
            np.zeros(1, dtype=np.float32),
            self.slot_assignments[None, ...],
            self.slot_positions[None, ...],
            float(self.max_speed),
            float(self.map_size),
            int(self.num_agents),
            float(self.world_max[2] - self.world_min[2]),
        )
        encircle_feats = compute_encirclement_features(
            agents_pos=positions[None, ...],
            target_pos=self.target_pos[None, ...],
            active_masks=np.ones((1, N), dtype=bool),
            arena_bounds=self._arena_bounds_flat,
            wall_threshold=2.0,
        )
        obs_full = np.concatenate((obs_core, encircle_feats), axis=-1).astype(np.float32, copy=False)

        self.prev_vel = velocities.copy()
        self.prev_yaw = yaws.copy()
        if self.observation_profile == "local50":
            return self._project_full_obs_to_local50(obs_full[0])
        return obs_full[0]

    def _get_teammate_features(self, agent_idx, positions, velocities, yaw):
        """ batch_build_teammate_features """
        N = len(positions)
        feats = np.zeros(self.top_k * 8, dtype=np.float32)
        
        pos_self = positions[agent_idx]
        cy, sy = np.cos(yaw), np.sin(yaw)
        map_size = self.map_size
        
      
        dists = []
        for j in range(N):
            if j == agent_idx:
                continue
            dx = positions[j][0] - pos_self[0]
            dy = positions[j][1] - pos_self[1]
            dz = positions[j][2] - pos_self[2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            dists.append((d, j))
        
        dists.sort(key=lambda x: x[0])
        
        for k_idx, (dist, j) in enumerate(dists[:self.top_k]):
            w_dx = positions[j][0] - pos_self[0]
            w_dy = positions[j][1] - pos_self[1]
            w_dz = positions[j][2] - pos_self[2]
            
            # Transform to body frame
            b_dx = w_dx * cy + w_dy * sy
            b_dy = -w_dx * sy + w_dy * cy
            
            rvx = velocities[j][0] - velocities[agent_idx][0]
            rvy = velocities[j][1] - velocities[agent_idx][1]
            rvz = velocities[j][2] - velocities[agent_idx][2]
            b_rvx = rvx * cy + rvy * sy
            b_rvy = -rvx * sy + rvy * cy

            base = k_idx * 8
            feats[base + 0] = b_dx  # 
            feats[base + 1] = b_dy  # 
            feats[base + 2] = w_dz  # 
            feats[base + 3] = dist  # 
            feats[base + 4] = 1.0 if dist <= self.view_radius else 0.0
            feats[base + 5] = b_rvx
            feats[base + 6] = b_rvy
            feats[base + 7] = rvz
        
        return feats

    def _build_reward_params(self, stage_config):
        stage_core = stage_config.reward_core if stage_config and getattr(stage_config, "reward_core", None) else {}
        rc = self.cfg.reward_core_defaults
        rr = self.cfg.reward_runtime

        def core_value(key: str, default: float = 0.0) -> float:
            if key in stage_core:
                return float(stage_core[key])
            return float(getattr(rc, key, default))

        collision_cost = core_value("collision_cost", -1.2)
        if stage_config and getattr(stage_config, "collision_penalty", None) is not None:
            collision_cost = float(stage_config.collision_penalty)

        self.frontier_reach_dist = float(getattr(rr, "frontier_reach_dist_m", 6.0))
        self.lidar_safe_distance_m = float(getattr(rr, "lidar_safe_distance_m", 4.0))
        self.search_speed_floor_mps = float(getattr(rr, "search_speed_floor_mps", 0.0))
        self.direction_gate_active_radius_m = float(getattr(rr, "direction_gate_active_radius_m", 80.0))

        return np.array([
            core_value("step_cost", -0.05),
            core_value("progress_gain", 0.8),
            core_value("exploration_gain", 0.2),
            core_value("proximity_cost", -0.4),
            collision_cost,
            core_value("direction_gain", 0.15),
            core_value("control_cost", -0.25),
            core_value("capture_gain", 1.5),
            core_value("capture_quality_gain", 0.6),
        ], dtype=np.float64)

    def _build_reward_weights(self) -> np.ndarray:
        return np.ones(9, dtype=np.float64)
    
    def step(self, actions):
        """Apply one environment step.

        Pipeline:
        1. Smooth the raw actions.
        2. Run the collision-avoidance shield.
        3. Convert body-frame commands to world-frame velocities.
        4. Send non-blocking AirSim commands with a finite duration.
        5. Compensate for loop drift to keep timing stable.

        Actions are body-frame physical velocities in `(m/s, rad/s)`.
        """
        step_start_time = time.perf_counter()  # High-precision timer.
        self.step_count += 1
        
      
        actions = np.ascontiguousarray(actions, dtype=np.float64).copy()
        actions_raw = actions.copy()
        
      
        positions, _, _ = self._get_agent_states()
        for i in range(self.num_agents):
            z_pos = positions[i][2]
            if z_pos > self.world_max[2] - 2.0:  # 2
                actions[i, 2] = -2.0  # 
                if self.debug:
                    print(f"[EMERGENCY] Agent {i} too high (z={z_pos:.1f}), forcing descent!")
            elif z_pos < self.world_min[2] + 2.0:  # 2
                actions[i, 2] = 2.0   # 
                if self.debug:
                    print(f"[EMERGENCY] Agent {i} too low (z={z_pos:.1f}), forcing ascent!")
        actions_after_emergency = actions.copy()
        prev_actions_copy = self.prev_actions.copy()
        
      
        positions, velocities, yaws = self._get_agent_states()
        if self.debug and self.step_count % 20 == 0 and len(positions) > 0:
            z_world = positions[0][2]
            vz_world = velocities[0][2]
            print(f"[AltCheck] z_world={z_world:.2f}m | vz_world={vz_world:.2f}m/s")
        
      
        lidar_dists = np.zeros((self.num_agents, 26), dtype=np.float32)
        force_lidar = (self.step_count % self.explore_update_interval) == 0
        for i in range(self.num_agents):
            lidar_dists[i] = self._simulate_lidar(
                positions[i],
                yaws[i],
                agent_idx=i,
                force_refresh=force_lidar,
            )

        diag_data = None
        if self.lidar_diagnostics and self.step_count % self.diag_interval == 0:
            diag_data = []
            for i in range(self.num_agents):
                min_idx = int(np.argmin(lidar_dists[i]))
                min_norm = float(lidar_dists[i][min_idx])
                min_dist_m = min_norm * self.lidar_max_range
                dir_body = self._lidar_dirs_body[min_idx]
                yaw = float(yaws[i])
                cy = float(np.cos(yaw))
                sy = float(np.sin(yaw))
                dir_world = np.array([
                    dir_body[0] * cy - dir_body[1] * sy,
                    dir_body[0] * sy + dir_body[1] * cy,
                    dir_body[2],
                ], dtype=np.float64)
                diag_data.append((i, int(self._lidar_last_count[i]), min_dist_m, min_idx, dir_world))
        
      
        if self.debug and self.step_count % 20 == 0:
            for i in range(min(1, self.num_agents)):  # agent
                min_dist = np.min(lidar_dists[i])
                min_dist_m = min_dist * self.lidar_max_range
                print(f"[LiDAR-Debug] Agent {i}: min_normalized={min_dist:.3f}, "
                      f"min_distance_m={min_dist_m:.1f}, threshold={self.shield_min_dist:.1f}")
                if min_dist_m < self.shield_min_dist:
                    print(f"   ")

        # Exploration update (timestamp-based)
        self.frontier_deltas[:] = 0.0
        self.exploration_deltas[:] = 0.0
        if self.batch_timestamps is not None and self.step_count % self.explore_update_interval == 0:
            new_voxels = batch_update_exploration_grid_optimized(
                positions[None, ...],
                lidar_dists[None, ...],
                self.batch_timestamps,
                np.array([self.step_count], dtype=np.int32),
                self.origin,
                self.voxel_size,
                self.lidar_max_range,
                self.steps_to_reexplore,
            )
            self.exploration_deltas = new_voxels[0].astype(np.float64)
        self._stamp_local_timestamps(positions, self.step_count, radius_vox=1)
        
        # ============================================================
      
        # ============================================================
        apply_action_smoothing(
            actions, 
            self.prev_actions, 
            alpha=self.smoothing_alpha,
            deadband=1e-3
        )
        actions_after_smoothing = actions.copy()
        actions_after_shield = actions_after_smoothing.copy()
        
        # ============================================================
      
        # ============================================================
        shield_triggered = np.zeros(self.num_agents, dtype=bool)
        if self.shield_enabled:
          
            actions_before = actions.copy()
            
          
            target_dist = np.zeros(self.num_agents, dtype=np.float64)
            for i in range(self.num_agents):
                dx = positions[i, 0] - self.target_pos[0]
                dy = positions[i, 1] - self.target_pos[1]
                dz = positions[i, 2] - self.target_pos[2]
                target_dist[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            shield_triggered = apply_safety_shield(
                actions,
                lidar_dists,
                min_dist_m=self.shield_min_dist,
                lidar_max_range=self.lidar_max_range,
                target_dist=target_dist,
                capture_radius=self.capture_radius,
            )
            actions_after_shield = actions.copy()
            
          
            if self.debug and np.any(shield_triggered):
                for i in range(self.num_agents):
                    if shield_triggered[i]:
                        action_change = np.linalg.norm(actions[i] - actions_before[i])
                        print(f"[Shield-Debug] Agent {i} triggered! Action change: {action_change:.3f}")
                        print(f"  Before: [{actions_before[i, 0]:.2f}, {actions_before[i, 1]:.2f}, {actions_before[i, 2]:.2f}]")
                        print(f"  After:  [{actions[i, 0]:.2f}, {actions[i, 1]:.2f}, {actions[i, 2]:.2f}]")
        else:
            if self.debug and self.step_count % 50 == 0:
                print("[Shield-Debug] Safety shield is DISABLED!")
        
      
        boundary_triggered = self._apply_boundary_shield(actions, positions)
        shield_triggered = np.logical_or(shield_triggered, boundary_triggered)
        actions_after_boundary = actions.copy()

        if self.lidar_diagnostics and diag_data is not None:
            for agent_idx, point_count, min_dist_m, min_idx, dir_world in diag_data:
                shield_flag = bool(shield_triggered[agent_idx])
                print(
                    f"[LiDAR] step={self.step_count} agent={agent_idx} points={point_count} "
                    f"min_m={min_dist_m:.2f} idx={min_idx} "
                    f"dir_world=[{dir_world[0]:.2f},{dir_world[1]:.2f},{dir_world[2]:.2f}] "
                    f"shield={shield_flag}"
                )
                if point_count == 0:
                    print(f"[LiDAR] Warning: agent {agent_idx} point cloud empty")
        
        if self.debug and self.step_count % 10 == 0 and self.num_agents > 0:
            i = 0
            min_lidar_m = float(np.min(lidar_dists[i]) * self.lidar_max_range)
            print(
                f"[ActionZ] step={self.step_count} "
                f"raw_vz={actions_raw[i, 2]:.2f} "
                f"emg_vz={actions_after_emergency[i, 2]:.2f} "
                f"smooth_vz={actions_after_smoothing[i, 2]:.2f} "
                f"shield_vz={actions_after_shield[i, 2]:.2f} "
                f"boundary_vz={actions_after_boundary[i, 2]:.2f} "
                f"min_lidar_m={min_lidar_m:.1f}"
            )
            
        
      
        self.prev_actions = actions.copy()
        
        # ============================================================
      
        # ============================================================
        cmd_list = []
        for i, name in enumerate(self.agent_names):
            if i >= len(actions):
                break
            
          
            vx_body = float(np.clip(actions[i, 0], -self.action_scale_vx, self.action_scale_vx))
            vy_body = float(np.clip(actions[i, 1], -self.action_scale_vy, self.action_scale_vy))
            vz = float(np.clip(actions[i, 2], -self.action_scale_vz, self.action_scale_vz))
            yaw_rate = float(np.clip(actions[i, 3], -self.action_scale_yaw, self.action_scale_yaw))
            
          
            current_z = positions[i][2]
            z_margin = 5.0  # 
            
          
            if current_z > (self.world_max[2] - z_margin) and vz > 0:
                vz = min(vz, -1.0)  # 
                if self.debug:
                    print(f"[Z-Constraint] Agent {i} near ceiling (z={current_z:.1f}), forcing descent")
            
          
            elif current_z < (self.world_min[2] + z_margin) and vz < 0:
                vz = max(vz, 1.0)  # 
                if self.debug:
                    print(f"[Z-Constraint] Agent {i} near ground (z={current_z:.1f}), forcing ascent")
            
          
            yaw = float(yaws[i])
            cy = float(np.cos(yaw))
            sy = float(np.sin(yaw))
            vx_world = vx_body * cy - vy_body * sy
            vy_world = vx_body * sy + vy_body * cy
            
            # World z-up -> NED z-down
            ned_vx = vx_world
            ned_vy = vy_world
            ned_vz = -vz
            yaw_rate_deg = yaw_rate * 57.2957795  # rad/s -> deg/s
            
            cmd_list.append((name, ned_vx, ned_vy, ned_vz, yaw_rate_deg))
            
          
            if self.debug and i == 0 and self.step_count % 10 == 0:
                print(f"[Z-Debug] Step {self.step_count}: pos_z={positions[i][2]:.1f}, "
                      f"target_z={self.target_pos[2]:.1f}, vz_action={vz:.2f}, "
                      f"world_z_range=[{self.world_min[2]:.1f}, {self.world_max[2]:.1f}]")
            
            if self.debug and i == 0 and self.step_count % 20 == 0:
                t_vec = self.target_pos - positions[i]
                t_bx = t_vec[0] * cy + t_vec[1] * sy
                t_by = -t_vec[0] * sy + t_vec[1] * cy
                t_bz = t_vec[2]
                cy_flip = np.cos(-yaw)
                sy_flip = np.sin(-yaw)
                t_bx_flip = t_vec[0] * cy_flip + t_vec[1] * sy_flip
                t_by_flip = -t_vec[0] * sy_flip + t_vec[1] * cy_flip
                t_bz_flip = t_vec[2]
                t_norm = np.sqrt(t_bx**2 + t_by**2 + t_bz**2) + 1e-9
                t_norm_flip = np.sqrt(t_bx_flip**2 + t_by_flip**2 + t_bz_flip**2) + 1e-9
                a_norm = np.sqrt(vx_body**2 + vy_body**2 + vz**2) + 1e-9
                align = (t_bx * vx_body + t_by * vy_body + t_bz * vz) / (t_norm * a_norm)
                align_flip = (t_bx_flip * vx_body + t_by_flip * vy_body + t_bz_flip * vz) / (t_norm_flip * a_norm)
                align_yflip = (t_bx * vx_body - t_by * vy_body + t_bz * vz) / (t_norm * a_norm)
                align_flip_yflip = (t_bx_flip * vx_body - t_by_flip * vy_body + t_bz_flip * vz) / (t_norm_flip * a_norm)
                print(
                    f"[DirCheck] yaw={yaw:.2f} | "
                    f"target_body=[{t_bx:.2f}, {t_by:.2f}, {t_bz:.2f}] | "
                    f"action_body=[{vx_body:.2f}, {vy_body:.2f}, {vz:.2f}] | "
                    f"align={align:.2f} | align_flip={align_flip:.2f} | "
                    f"align_yflip={align_yflip:.2f} | align_flip_yflip={align_flip_yflip:.2f}"
                )
        
            # ============================================================
          
            # ============================================================
          
          
            CMD_DURATION = 0.3  # 
            
          
            for name, vx, vy, vz, yr in cmd_list:
                try:
                  
                    with self._client_lock:
                        self.client.moveByVelocityAsync(
                            vx, vy, vz,
                            duration=CMD_DURATION,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yr),
                            vehicle_name=name
                        )
                except Exception as e:
                    if self.debug:
                        print(f"[Warning] moveByVelocity failed for {name}: {e}")

          
            
            # ============================================================
          
            # ============================================================
            if "target" in self.vehicle_names:
                self._update_target(positions)
            
            # ============================================================
          
            # ============================================================
            if self.step_count % max(self.nav_update_interval, 2) == 0:
                # update state only when needed
                if self.step_count % self.nav_update_interval == 0:
                    positions, velocities, yaws = self._get_agent_states()
                self._update_navigation(positions, velocities, yaws)
            else:
                self._update_target_tracking(positions)

            if self.debug_frontier and self.step_count % 10 == 0:
                goals = getattr(self, "_last_frontier_goals", None)
                if goals:
                    dists = []
                    for goal in goals.values():
                        if goal is None:
                            continue
                        dists.append(float(np.linalg.norm(goal - positions[0])))
                    if dists:
                        print(f"[Frontier-Debug] step={self.step_count} goals={len(dists)} "
                              f"min_dist={min(dists):.1f} max_dist={max(dists):.1f}")
            
            # ============================================================
          
            # ============================================================
            elapsed = time.perf_counter() - step_start_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            # elif self.debug and elapsed > self.dt * 1.5:
                # print(f"[Warning] Step took {elapsed*1000:.1f}ms, target={self.dt*1000:.1f}ms")
            
            # ============================================================
          
            # ============================================================
            obs = self._get_observations()
            
          
            positions, velocities, yaws = self._get_agent_states()
            min_dist = float('inf')
            for pos in positions:
                dist = np.linalg.norm(pos - self.target_pos)
                min_dist = min(min_dist, dist)
            
            captured = min_dist <= self.capture_radius
            
          
            collisions = [False] * self.num_agents
            for i, name in enumerate(self.agent_names):
                try:
                    with self._client_lock:
                        info = self.client.simGetCollisionInfo(vehicle_name=name)
                    collisions[i] = bool(info.has_collided)
                except Exception:
                    collisions[i] = False
            if self.team_collision_radius > 0.0 and self.num_agents > 1:
                rad_sq = self.team_collision_radius * self.team_collision_radius
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dz = positions[i, 2] - positions[j, 2]
                        if dx*dx + dy*dy + dz*dz <= rad_sq:
                            collisions[i] = True
                            collisions[j] = True

          
            reward_breakdown = None
            lidar_now = np.zeros((self.num_agents, 26), dtype=np.float32)
            for i in range(self.num_agents):
                lidar_now[i] = self._simulate_lidar(positions[i], yaws[i])
            min_dist_m = np.min(lidar_now, axis=1) * self.lidar_max_range

            # Mode-aware reward inputs (keep in sync with batch_compute_rewards_v2)
            target_mode_flags = self.target_mode_flag[None, :].astype(np.float64)

            if not hasattr(self, "prev_target_mode_flag"):
                self.prev_target_mode_flag = target_mode_flags[0].copy()
            if not hasattr(self, "prev_obs_target_pos"):
                self.prev_obs_target_pos = self.obs_target_pos.copy()
            is_switching_frame = (target_mode_flags != self.prev_target_mode_flag[None, ...])
            obs_target_jump_threshold = 15.0
            obs_target_changed = (
                np.linalg.norm(self.obs_target_pos - self.prev_obs_target_pos, axis=1) > obs_target_jump_threshold
            )
            is_potential_unstable = is_switching_frame | obs_target_changed[None, ...]

            if not hasattr(self, "prev_frontier_dist"):
                if self.current_goals is None or len(self.current_goals) == 0:
                    goals = np.repeat(self.target_pos[None, :], self.num_agents, axis=0)
                else:
                    goals = self.current_goals
                diffs = goals - positions
                self.prev_frontier_dist = np.linalg.norm(diffs, axis=1)

            if np.any(is_potential_unstable):
                for i in range(self.num_agents):
                    if not is_potential_unstable[0, i]:
                        continue
                    if target_mode_flags[0, i] > 0.5:
                        dx = positions[i, 0] - self.obs_target_pos[i, 0]
                        dy = positions[i, 1] - self.obs_target_pos[i, 1]
                        dz = positions[i, 2] - self.obs_target_pos[i, 2]
                        self.prev_target_dist[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
                    else:
                        goal = self.current_goals[i] if self.current_goals is not None else self.last_known_target_pos
                        fx = goal[0] - positions[i, 0]
                        fy = goal[1] - positions[i, 1]
                        fz = goal[2] - positions[i, 2]
                        self.prev_frontier_dist[i] = np.sqrt(fx * fx + fy * fy + fz * fz)

            if not hasattr(self, "position_history"):
                self.position_history = np.repeat(positions[:, None, :], 5, axis=1).astype(np.float64)
                self.history_idx = 0

            collision_impact_flags = np.zeros(self.num_agents, dtype=bool)
            for i in range(self.num_agents):
                if collisions[i]:
                    if self.collision_armed[i]:
                        collision_impact_flags[i] = True
                        self.collision_armed[i] = False
                    self.collision_rearm_count[i] = 0
                else:
                    if min_dist_m[i] > 0.5:
                        self.collision_rearm_count[i] += 1
                        if self.collision_rearm_count[i] >= 3:
                            self.collision_armed[i] = True
                            self.collision_rearm_count[i] = 0
                    else:
                        self.collision_rearm_count[i] = 0

            target_for_potential = self.obs_target_pos.copy()
            rewards, new_target_dist, new_frontier_dist, _, breakdown = batch_compute_rewards_v2(
                positions[None, ...],
                velocities[None, ...],
                self.target_pos[None, ...],
                self.target_vel[None, ...],
                target_for_potential[None, ...],
                self.path_lengths[None, ...],
                self.prev_target_dist[None, ...],
                self.prev_frontier_dist[None, ...],
                lidar_now[None, ...],
                self.guidance_vectors[None, ...],
                actions[None, ...],
                prev_actions_copy[None, ...],
                np.array(collisions, dtype=bool)[None, ...],
                np.ones_like(np.array(collisions, dtype=bool))[None, ...],
                collision_impact_flags[None, ...],
                shield_triggered[None, ...],
                self.exploration_deltas[None, ...],
                self.frontier_deltas[None, ...],
                self.reward_params,
                np.array([self.capture_radius], dtype=np.float64),
                self.lidar_max_range,
                self.frontier_reach_dist,
                self.lidar_safe_distance_m,
                self.search_speed_floor_mps,
                self.direction_gate_active_radius_m,
                target_mode_flags,
                is_potential_unstable,
                self.position_history[None, ...],
                self.current_goals[None, ...],
                self.world_min,
                self.world_max,
            )
            self.prev_target_dist = new_target_dist[0].copy()
            self.prev_frontier_dist = new_frontier_dist[0].copy()
            self.prev_target_mode_flag = target_mode_flags[0].copy()
            self.prev_obs_target_pos = self.obs_target_pos.copy()
            self.history_idx = (self.history_idx + 1) % 5
            self.position_history[:, self.history_idx, :] = positions
            vals = breakdown[0]
            reward_breakdown = {
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
          
            #     print(f"[Reward] {reward_breakdown}")
        
        info = {
            "min_dist": min_dist,
            "captured": captured,
            "collisions": collisions,
            "collision": bool(np.any(collisions)),
            "positions": positions,
            "paths": self.current_paths,
            "shield_triggered": shield_triggered,
            "reward_breakdown": reward_breakdown,
        }

        done = bool(captured or np.any(collisions))
        return obs, done, info
    
    def render_paths(self, duration=2.0):
        """ AirSim  A* """
      
        try:
            self.client.simFlushPersistentMarkers()
        except:
            pass
        
        colors = [
            [1.0, 0.0, 0.0, 0.8],  # Red
            [0.0, 1.0, 0.0, 0.8],  # Green
            [0.0, 0.0, 1.0, 0.8],  # Blue
            [1.0, 1.0, 0.0, 0.8],  # Yellow
        ]
        
        for i, path in enumerate(self.current_paths):
            if path is None or len(path) < 2:
                continue
            
          
            ned_points = []
            for p in path:
                ned_points.append(airsim.Vector3r(float(p[0]), float(p[1]), float(-p[2])))
            
            color = colors[i % len(colors)]
            try:
                self.client.simPlotLineStrip(ned_points, color_rgba=color, thickness=5.0, duration=duration, is_persistent=False)
            except Exception:
                pass
    
    def render_target(self, duration=5.0):
        """ AirSim """
        ned_target = airsim.Vector3r(
            float(self.target_pos[0]),
            float(self.target_pos[1]),
            float(-self.target_pos[2])
        )
        try:
            self.client.simPlotPoints([ned_target], color_rgba=[0.0, 1.0, 1.0, 1.0], size=20.0, duration=duration)
        except Exception:
            pass
    
    def render_obstacles(self, duration=300.0, max_points=10000000, z_slice=None):
        """Render obstacle voxels in AirSim.

        Args:
            duration: Render lifetime in seconds.
            max_points: Maximum number of rendered points.
            z_slice: Optional z-layer to render. `None` renders all layers.
        """
        if self.occupancy_grid is None:
            print("[Warning] No occupancy grid loaded")
            return
        
        print(f"[Obstacles] Grid shape: {self.occupancy_grid.shape}")
        print(f"[Obstacles] Origin: {self.origin}")
        print(f"[Obstacles] Voxel size: {self.voxel_size}m")
        
      
        obstacle_indices = np.argwhere(self.occupancy_grid == 1)
        total_obstacles = len(obstacle_indices)
        print(f"[Obstacles] Total obstacle voxels: {total_obstacles}")
        
        if total_obstacles == 0:
            print("[Warning] No obstacles found in grid!")
            return
        
      
        if z_slice is not None:
            obstacle_indices = obstacle_indices[obstacle_indices[:, 2] == z_slice]
            print(f"[Obstacles] Voxels at z={z_slice}: {len(obstacle_indices)}")
        
      
        if len(obstacle_indices) > max_points:
            indices = np.random.choice(len(obstacle_indices), max_points, replace=False)
            obstacle_indices = obstacle_indices[indices]
            print(f"[Obstacles] Sampled {max_points} points for rendering")
        
      
        ned_points = []
        for idx in obstacle_indices:
          
            world_pos = (np.array(idx, dtype=np.float64) + 0.5) * self.voxel_size + self.origin
            # World (z-up) -> NED (z-down)
            ned_points.append(airsim.Vector3r(
                float(world_pos[0]),
                float(world_pos[1]),
                float(-world_pos[2])
            ))
        
      
        try:
          
            batch_size = 500
            for i in range(0, len(ned_points), batch_size):
                batch = ned_points[i:i+batch_size]
                self.client.simPlotPoints(batch, color_rgba=[1.0, 0.3, 0.3, 0.7], size=8.0, duration=duration)
            print(f"[Obstacles] Rendered {len(ned_points)} obstacle points")
        except Exception as e:
            print(f"[Warning] Failed to render obstacles: {e}")
    
    def render_grid_bounds(self, duration=30.0):
        """"""
        if self.occupancy_grid is None:
            return
        
      
        grid_size = np.array(self.occupancy_grid.shape) * self.voxel_size
        min_corner = self.origin
        max_corner = self.origin + grid_size
        
        print(f"[Grid] World bounds: min={min_corner}, max={max_corner}")
        
      
        corners = [
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
        ]
        
      
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 
            (4, 5), (5, 6), (6, 7), (7, 4),  # 
            (0, 4), (1, 5), (2, 6), (3, 7),  # 
        ]
        
        try:
            for e in edges:
                p1 = corners[e[0]]
                p2 = corners[e[1]]
                ned_p1 = airsim.Vector3r(float(p1[0]), float(p1[1]), float(-p1[2]))
                ned_p2 = airsim.Vector3r(float(p2[0]), float(p2[1]), float(-p2[2]))
                self.client.simPlotLineStrip([ned_p1, ned_p2], color_rgba=[1.0, 1.0, 1.0, 0.5], thickness=3.0, duration=duration)
            print(f"[Grid] Rendered boundary box")
        except Exception as e:
            print(f"[Warning] Failed to render grid bounds: {e}")
    
    def render_frontier_goals(self, frontier_goals, duration=3.0):
        """"""
        if not frontier_goals:
            return
        
        ned_points = []
        for agent_idx, goal in frontier_goals.items():
            if goal is not None:
                ned_points.append(airsim.Vector3r(
                    float(goal[0]),
                    float(goal[1]),
                    float(-goal[2])
                ))
        
        if ned_points:
            try:
              
                self.client.simPlotPoints(ned_points, color_rgba=[1.0, 0.0, 0.0, 1.0], size=25.0, duration=duration)
            except Exception:
                pass
    
    def render_visited_voxels(self, duration=300.0, max_points=50000):
        """Render visited free voxels while excluding obstacles.

        Colors:
        - Yellow: confidence below `threshold`, so re-exploration is needed.
        - Green: confidence at or above `threshold`, so the region is well explored.
        """
        if not hasattr(self, 'voxel_map') or self.voxel_map.visits is None:
            return
        
        visits = self.voxel_map.visits
        threshold = self.confidence_threshold
        
      
        visited_mask = np.logical_and(visits > 0, self.occupancy_grid == 0)
        visited_indices = np.argwhere(visited_mask)
        
        if len(visited_indices) == 0:
            return
        
      
        low_conf_indices = []
        high_conf_indices = []
        
        for idx in visited_indices:
            conf = visits[idx[0], idx[1], idx[2]]
            if conf < threshold:
                low_conf_indices.append(idx)
            else:
                high_conf_indices.append(idx)
        
      
        if len(low_conf_indices) > max_points // 2:
            sample_idx = np.random.choice(len(low_conf_indices), max_points // 2, replace=False)
        if len(high_conf_indices) > max_points // 2:
            sample_idx = np.random.choice(len(high_conf_indices), max_points // 2, replace=False)
            high_conf_indices = [high_conf_indices[i] for i in sample_idx]
        
      
        if low_conf_indices:
            ned_points = []
            for idx in low_conf_indices:
                world_pos = (np.array(idx, dtype=np.float64) + 0.5) * self.voxel_size + self.origin
                ned_points.append(airsim.Vector3r(
                    float(world_pos[0]),
                    float(world_pos[1]),
                    float(-world_pos[2])
                ))
            try:
                batch_size = 500
                for i in range(0, len(ned_points), batch_size):
                    batch = ned_points[i:i+batch_size]
                    self.client.simPlotPoints(batch, color_rgba=[1.0, 1.0, 0.0, 0.6], size=6.0, duration=duration)
            except Exception:
                pass
        
      
        if high_conf_indices:
            ned_points = []
            for idx in high_conf_indices:
                world_pos = (np.array(idx, dtype=np.float64) + 0.5) * self.voxel_size + self.origin
                ned_points.append(airsim.Vector3r(
                    float(world_pos[0]),
                    float(world_pos[1]),
                    float(-world_pos[2])
                ))
            try:
                batch_size = 500
                for i in range(0, len(ned_points), batch_size):
                    batch = ned_points[i:i+batch_size]
                    self.client.simPlotPoints(batch, color_rgba=[0.0, 1.0, 0.0, 0.6], size=6.0, duration=duration)
            except Exception:
                pass
        
        return
    

    def _apply_boundary_shield(self, actions, positions):
        """"""
        triggered = np.zeros(self.num_agents, dtype=bool)
        boundary_margin = 10.0  # 
        
        for i in range(self.num_agents):
            pos = positions[i]
            action = actions[i]
            
          
            if pos[0] < self.world_min[0] + boundary_margin and action[0] < 0:
                actions[i, 0] = max(actions[i, 0], 2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near X-min boundary, forcing inward")
            elif pos[0] > self.world_max[0] - boundary_margin and action[0] > 0:
                actions[i, 0] = min(actions[i, 0], -2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near X-max boundary, forcing inward")
            
          
            if pos[1] < self.world_min[1] + boundary_margin and action[1] < 0:
                actions[i, 1] = max(actions[i, 1], 2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near Y-min boundary, forcing inward")
            elif pos[1] > self.world_max[1] - boundary_margin and action[1] > 0:
                actions[i, 1] = min(actions[i, 1], -2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near Y-max boundary, forcing inward")
            
          
            if pos[2] < self.world_min[2] + boundary_margin and action[2] < 0:
                actions[i, 2] = max(actions[i, 2], 2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near Z-min boundary, forcing upward")
            elif pos[2] > self.world_max[2] - boundary_margin and action[2] > 0:
                actions[i, 2] = min(actions[i, 2], -2.0)  # 
                triggered[i] = True
                if self.debug:
                    print(f"[Boundary-Shield] Agent {i} near Z-max boundary, forcing downward")
        
        return triggered
    def cleanup(self):
        """"""
      
        pass


def load_policy(policy_path, obs_dim, cfg, device):
    """"""
    payload = torch.load(policy_path, map_location=device, weights_only=False)

    if isinstance(payload, dict):
        #  checkpoint  state_dict
        if "policy_state_dict" in payload:
            state_dict = payload["policy_state_dict"]
        else:
            state_dict = payload
    else:
        raise ValueError(f"Unknown checkpoint format: {type(payload)}")

    # 
    input_dim = state_dict.get("model.base.0.weight", state_dict.get("base.0.weight"))
    ckpt_obs_dim = obs_dim
    if input_dim is not None:
        ckpt_obs_dim = input_dim.shape[1]
        if ckpt_obs_dim != obs_dim:
            print(f"[Warning] Obs dim mismatch: checkpoint={ckpt_obs_dim}, env={obs_dim}")

    policy = MAPPOPolicy3D(
        ckpt_obs_dim,
        action_dim=4,
        device=device,
        action_bounds=cfg.control.action_bounds,
        hidden_dim=cfg.model.hidden_dim if hasattr(cfg, "model") else 512,
        centralized_critic=str(getattr(cfg.experiment, "critic_mode", "local")).lower() == "ctde_joint_obs_plus_global",
        critic_obs_dim=(cfg.environment.num_agents * ckpt_obs_dim + 6)
        if str(getattr(cfg.experiment, "critic_mode", "local")).lower() == "ctde_joint_obs_plus_global"
        else None,
    )
    policy._expected_obs_dim = ckpt_obs_dim

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def _adapt_observation(obs, expected_dim: int):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] == expected_dim:
        return obs
    if obs.shape[-1] == 83 and expected_dim == 50:
        return AirSimEnv._project_full_obs_to_local50(obs)
    raise ValueError(
        f"Observation dim mismatch: got {obs.shape[-1]}, expected {expected_dim}. "
        "Please ensure AirSim eval config matches the training config/baseline."
    )


def _main_single():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
  
    config_path = resolve_config_path(args)
    cfg = load_config(config_path) if config_path else load_config()
    cfg = apply_baseline_overrides(cfg, args.baseline)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")
    print(f"[Config] Observation profile: {cfg.experiment.observation_profile}")
    
  
    stage_config = get_stage_config(cfg, args.stage)
    if args.stage and stage_config is None:
        print(f"[Error] Invalid stage: {args.stage}")
        return
    
  
    env = AirSimEnv(
        cfg,
        stage_config=stage_config,
        target_pos=args.target_pos,
        debug=args.debug,
        debug_frontier=args.debug_frontier,
        lidar_name=args.lidar_name,
        lidar_diagnostics=args.lidar_diagnostics,
        diag_interval=args.diag_interval,
    )
    
  
    if args.no_shield:
        env.shield_enabled = False
        print(f"[Config] Safety shield: DISABLED")
    else:
        print(f"[Config] Safety shield: enabled (min_dist={env.shield_min_dist}m)")
    
    if args.smoothing_alpha is not None:
        env.smoothing_alpha = args.smoothing_alpha
    print(f"[Config] Action smoothing alpha: {env.smoothing_alpha}")
    
  
    env.nav_update_interval = args.nav_interval
    print(f"[Config] Navigation update interval: {env.nav_update_interval} steps")
    
    try:
      
        obs = env.reset()
        obs_dim = len(obs[0])
        print(f"[Config] Obs dim: {obs_dim}, Num agents: {env.num_agents}")
        
      
        policy = load_policy(args.policy, obs_dim, cfg, device)
        print(f"[Policy] Loaded from {args.policy}")
        print(f"[Policy] Action std: {policy.get_current_std():.4f}")
        
        if args.follow_guidance:
            print(f"[Mode] Following A* guidance directly at {args.guidance_speed} m/s (policy ignored)")
        else:
            mode_str = "deterministic" if args.deterministic else "stochastic"
            print(f"[Mode] Using trained policy ({mode_str})")
        
      
        hidden = policy.init_hidden(num_envs=len(obs)).to(device)
        masks = torch.ones(len(obs), 1, device=device)
        
      
        env.render_target(duration=args.steps * env.dt + 10)
        env.render_paths(duration=3.0)
        
      
        if args.show_obstacles:
            env.render_obstacles(duration=6000.0, z_slice=args.obstacle_z)
            env.render_grid_bounds(duration=6000.0)
        
      
        print(f"\n{'='*60}")
        print(f"Starting episode with {args.steps} max steps")
        print(f"Target: {env.target_pos}")
        print(f"Capture radius: {env.capture_radius}m")
        print(f"Performance optimizations enabled:")
        print(f"  - Serial API calls with proper locking (AirSim thread-safe)")
        print(f"  - LiDAR caching (update every {env._lidar_update_interval} steps)")
        print(f"  - Navigation caching (update every {env._nav_cache_interval} steps)")
        print(f"  - State query caching (TTL: {env._state_cache_ttl*1000:.0f}ms)")
        print(f"  - Reduced debug output frequency")
        print(f"{'='*60}\n")
        
      
        step_times = deque(maxlen=50)  # 50
        
        for step in range(args.steps):
            step_start = time.perf_counter()
            
            if args.follow_guidance:
              
              
                positions, _, yaws = env._get_agent_states()
                actions_list = []
                
                for i in range(env.num_agents):
                    g = env.guidance_vectors[i]
                    yaw = yaws[i]
                    cy, sy = np.cos(yaw), np.sin(yaw)
                    
                    # World -> Body: R^T * v_world
                  
                    vx_body = (g[0] * cy + g[1] * sy) * args.guidance_speed
                    vy_body = (-g[0] * sy + g[1] * cy) * args.guidance_speed
                    vz_body = g[2] * args.guidance_speed
                    
                  
                    action = np.array([
                        vx_body,
                        vy_body,
                        vz_body,
                        0.0  # 
                    ], dtype=np.float64)
                    actions_list.append(action)
                    
                    if args.debug and i == 0 and step % 50 == 0:  # 
                        print(f"[Guidance] Agent {i}: g_world={g}, target_speed={args.guidance_speed}m/s, "
                              f"action_body={action[:3]}")
                
                actions_np = np.array(actions_list, dtype=np.float64)
            else:
              
                obs_adapted = _adapt_observation(obs, int(getattr(policy, "_expected_obs_dim", obs.shape[-1])))
                obs_tensor = torch.from_numpy(obs_adapted).to(device)
                with torch.no_grad():
                    if args.deterministic:
                        mean, _, _, hidden = policy.model(obs_tensor, hidden, masks)
                        actions = torch.tanh(mean) * policy._scale
                    else:
                        actions, _, _, hidden = policy.act(obs_tensor, hidden, masks)
                actions_np = actions.cpu().numpy().astype(np.float64)
            
          
            obs, done, info = env.step(actions_np)
            
          
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            
          
            if args.show_path and step % (args.path_interval * 2) == 0:  # 
                env.render_paths(duration=args.path_interval * env.dt * 2 + 1.0)
            
          
            if args.show_path and step % (args.path_interval * 2) == 0:
                if hasattr(env, '_last_frontier_goals') and env._last_frontier_goals:
                    env.render_frontier_goals(env._last_frontier_goals, duration=args.path_interval * env.dt * 2 + 1.0)
            
          
            # if step % 100 == 0 and step > 0:
                    # Visited voxel rendering disabled.
            
          
            if step % 50 == 0 or done:
                positions = info["positions"]
                collision_count = sum(info["collisions"])
                shield_count = sum(info["shield_triggered"]) if "shield_triggered" in info else 0
                
              
                if len(step_times) > 10:
                    avg_step_time = sum(step_times) / len(step_times)
                    freq_hz = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    freq_str = f"freq={freq_hz:.1f}Hz"
                else:
                    freq_str = "freq=calculating..."
                
                print(f"[Step {step:4d}] dist={info['min_dist']:6.2f}m | "
                      f"collisions={collision_count} | shield={shield_count} | "
                      f"{freq_str} | "
                      f"pos[0]=[{positions[0][0]:6.1f}, {positions[0][1]:6.1f}, {positions[0][2]:5.1f}]")
            
            if done:
                avg_freq = len(step_times) / sum(step_times) if sum(step_times) > 0 else 0
                print(f"\n{'='*60}")
                print(f"Episode finished at step {step+1}")
                print(f"Result: {'CAPTURED!' if info['captured'] else 'Collision/Timeout'}")
                print(f"Final distance to target: {info['min_dist']:.2f}m")
                print(f"Average frequency: {avg_freq:.1f}Hz")
                print(f"{'='*60}")
                break
        
        if not done:
            avg_freq = len(step_times) / sum(step_times) if sum(step_times) > 0 else 0
            print(f"\n{'='*60}")
            print(f"Episode reached max steps ({args.steps})")
            print(f"Final distance to target: {info['min_dist']:.2f}m")
            print(f"Average frequency: {avg_freq:.1f}Hz")
            print(f"{'='*60}")
    
    finally:
        env.cleanup()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = resolve_config_path(args)
    cfg = load_config(config_path) if config_path else load_config()
    cfg = apply_baseline_overrides(cfg, args.baseline)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")
    print(f"[Config] Observation profile: {cfg.experiment.observation_profile}")

    stage_config = get_stage_config(cfg, args.stage)
    if args.stage and stage_config is None:
        print(f"[Error] Invalid stage: {args.stage}")
        return

    env = AirSimEnv(
        cfg,
        stage_config=stage_config,
        target_pos=args.target_pos,
        debug=args.debug,
        debug_frontier=args.debug_frontier,
        lidar_name=args.lidar_name,
        lidar_diagnostics=args.lidar_diagnostics,
        diag_interval=args.diag_interval,
    )

    if args.no_shield:
        env.shield_enabled = False
        print("[Config] Safety shield: DISABLED")
    else:
        print(f"[Config] Safety shield: enabled (min_dist={env.shield_min_dist}m)")

    if args.smoothing_alpha is not None:
        env.smoothing_alpha = args.smoothing_alpha
    print(f"[Config] Action smoothing alpha: {env.smoothing_alpha}")

    env.nav_update_interval = args.nav_interval
    print(f"[Config] Navigation update interval: {env.nav_update_interval} steps")

    try:
        if args.target_pos is None:
            env.target_pos = None
        obs = env.reset()
        obs_dim = len(obs[0])
        print(f"[Config] Obs dim: {obs_dim}, Num agents: {env.num_agents}")

        policy = load_policy(args.policy, obs_dim, cfg, device)
        print(f"[Policy] Loaded from {args.policy}")
        print(f"[Policy] Action std: {policy.get_current_std():.4f}")

        if args.follow_guidance:
            print(f"[Mode] Following A* guidance directly at {args.guidance_speed} m/s (policy ignored)")
        else:
            mode_str = "deterministic" if args.deterministic else "stochastic"
            print(f"[Mode] Using trained policy ({mode_str})")

        results = []
        success = 0
        collision = 0
        timeout = 0
        total_episodes = max(1, int(args.episodes))

        for ep in range(total_episodes):
            if ep > 0:
                if args.target_pos is None:
                    env.target_pos = None
                obs = env.reset()

            hidden = policy.init_hidden(num_envs=len(obs)).to(device)
            masks = torch.ones(len(obs), 1, device=device)

            env.render_target(duration=args.steps * env.dt + 10)
            env.render_paths(duration=3.0)

            if args.show_obstacles and ep == 0:
                env.render_obstacles(duration=6000.0, z_slice=args.obstacle_z)
                env.render_grid_bounds(duration=6000.0)

            print(f"\n{'='*60}")
            print(f"[Episode {ep+1}/{total_episodes}] Starting with {args.steps} max steps")
            print(f"Target: {env.target_pos}")
            print(f"Capture radius: {env.capture_radius}m")
            print("Performance optimizations enabled:")
            print("  - Serial API calls with proper locking (AirSim thread-safe)")
            print(f"  - LiDAR caching (update every {env._lidar_update_interval} steps)")
            print(f"  - Navigation caching (update every {env._nav_cache_interval} steps)")
            print(f"  - State query caching (TTL: {env._state_cache_ttl*1000:.0f}ms)")
            print("  - Reduced debug output frequency")
            print(f"{'='*60}\n")

            step_times = deque(maxlen=50)
            done = False
            info = {}

            for step in range(args.steps):
                step_start = time.perf_counter()

                if args.follow_guidance:
                    positions, _, yaws = env._get_agent_states()
                    actions_list = []
                    for i in range(env.num_agents):
                        g = env.guidance_vectors[i]
                        yaw = yaws[i]
                        cy, sy = np.cos(yaw), np.sin(yaw)
                        vx_body = (g[0] * cy + g[1] * sy) * args.guidance_speed
                        vy_body = (-g[0] * sy + g[1] * cy) * args.guidance_speed
                        vz_body = g[2] * args.guidance_speed
                        action = np.array([vx_body, vy_body, vz_body, 0.0], dtype=np.float64)
                        actions_list.append(action)
                        if args.debug and i == 0 and step % 50 == 0:
                            print(f"[Guidance] Agent {i}: g_world={g}, target_speed={args.guidance_speed}m/s, "
                                  f"action_body={action[:3]}")
                    actions_np = np.array(actions_list, dtype=np.float64)
                else:
                    obs_adapted = _adapt_observation(obs, int(getattr(policy, "_expected_obs_dim", obs.shape[-1])))
                    obs_tensor = torch.from_numpy(obs_adapted).to(device)
                    with torch.no_grad():
                        if args.deterministic:
                            mean, _, _, hidden = policy.model(obs_tensor, hidden, masks)
                            actions = torch.tanh(mean) * policy._scale
                        else:
                            actions, _, _, hidden = policy.act(obs_tensor, hidden, masks)
                    actions_np = actions.cpu().numpy().astype(np.float64)

                obs, done, info = env.step(actions_np)

                step_time = time.perf_counter() - step_start
                step_times.append(step_time)

                if args.show_path and step % (args.path_interval * 2) == 0:
                    env.render_paths(duration=args.path_interval * env.dt * 2 + 1.0)

                if args.show_path and step % (args.path_interval * 2) == 0:
                    if hasattr(env, "_last_frontier_goals") and env._last_frontier_goals:
                        env.render_frontier_goals(
                            env._last_frontier_goals,
                            duration=args.path_interval * env.dt * 2 + 1.0,
                        )

                if step % 50 == 0 or done:
                    positions = info.get("positions")
                    has_positions = positions is not None and len(positions) > 0
                    collisions = info.get("collisions")
                    collision_count = int(np.sum(collisions)) if collisions is not None else 0
                    shield = info.get("shield_triggered", [])
                    if shield is None:
                        shield_count = 0
                    else:
                        shield_count = int(np.sum(shield))

                    if len(step_times) > 10:
                        avg_step_time = sum(step_times) / len(step_times)
                        freq_hz = 1.0 / avg_step_time if avg_step_time > 0 else 0
                        freq_str = f"freq={freq_hz:.1f}Hz"
                    else:
                        freq_str = "freq=calculating..."

                    if has_positions:
                        pos0 = positions[0]
                        pos_str = f"pos[0]=[{pos0[0]:6.1f}, {pos0[1]:6.1f}, {pos0[2]:5.1f}]"
                    else:
                        pos_str = "pos[0]=[n/a]"

                    print(f"[Step {step:4d}] dist={info.get('min_dist', 0.0):6.2f}m | "
                          f"collisions={collision_count} | shield={shield_count} | "
                          f"{freq_str} | {pos_str}")

                if done:
                    avg_freq = len(step_times) / sum(step_times) if sum(step_times) > 0 else 0
                    print(f"\n{'='*60}")
                    print(f"Episode finished at step {step+1}")
                    print(f"Result: {'CAPTURED!' if info.get('captured', False) else 'Collision/Timeout'}")
                    print(f"Final distance to target: {info.get('min_dist', 0.0):.2f}m")
                    print(f"Average frequency: {avg_freq:.1f}Hz")
                    print(f"{'='*60}")
                    break

            steps_taken = step + 1 if args.steps > 0 else 0
            if done:
                if info.get("captured", False):
                    reason = "success"
                    success += 1
                elif info.get("collision", False):
                    reason = "collision"
                    collision += 1
                else:
                    reason = "done"
            else:
                reason = "timeout"
                timeout += 1

                avg_freq = len(step_times) / sum(step_times) if sum(step_times) > 0 else 0
                print(f"\n{'='*60}")
                print(f"Episode reached max steps ({args.steps})")
                print(f"Final distance to target: {info.get('min_dist', 0.0):.2f}m")
                print(f"Average frequency: {avg_freq:.1f}Hz")
                print(f"{'='*60}")

            results.append({
                "episode": ep + 1,
                "steps": steps_taken,
                "done_reason": reason,
                "min_dist": float(info.get("min_dist", 0.0)),
            })

        total = len(results)
        if total > 0:
            print(f"[Eval] episodes={total} success_rate={success/total:.3f} "
                  f"collision_rate={collision/total:.3f} timeout_rate={timeout/total:.3f}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[Eval] Results written to {output_path}")

    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
