"""Configuration loader for MAPPO 3D training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class AirSimConfig:
    ip: str
    port: int
    api_timeout: float
    randomize_physics: bool
    disable_rendering: bool = True


@dataclass
class EnvironmentConfig:
    num_agents: int
    step_hz: int
    world_size_m: Sequence[float]
    voxel_size_m: float
    max_steps: int
    capture_radius_m: float = 5.0
    min_spawn_separation_m: float = 10.0
    confidence_decay: float = 0.995
    confidence_threshold: float = 0.5
    lkp_countdown_steps: int = 30
    lkp_breakout_dist_m: float = 2.0
    team_collision_radius_m: float = 2.5
    target_spawn_min_dist_m: float = 20.0
    target_spawn_max_dist_m: float = -1.0
    frame_stack: int = 4
    imu_accel_noise_std: float = 0.0
    imu_gyro_noise_std: float = 0.0
    lidar_update_interval: int = 1

    @property
    def step_seconds(self) -> float:
        return 1.0 / float(self.step_hz)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        sx, sy, sz = self.world_size_m
        cell = float(self.voxel_size_m)
        return (
            int(round(sx / cell)),
            int(round(sy / cell)),
            int(round(sz / cell)),
        )


@dataclass
class ControlConfig:
    smoothing_alpha: float
    ema_clip: float
    action_bounds: Dict[str, float]
    smoothness_penalty: float
    target_max_accel: float = 8.0
    action_delay_steps: int = 0
    action_delay_min_steps: int = 0
    action_delay_max_steps: int = 0


@dataclass
class ShieldConfig:
    min_distance_m: float


@dataclass
class PerceptionConfig:
    lidar_max_points: int
    bin_shape: Sequence[int]
    teammate_top_k: int
    obstacle_sector_floor: float
    lidar_sector_subrays: int = 5
    lidar_sector_spread_deg: float = 10.0
    lidar_dropout_prob: float = 0.0
    lidar_noise_std: float = 0.0
    target_obs_noise_std: float = 0.0


@dataclass
class NavigationConfig:
    lookahead_distance_m: float
    cache_refresh_steps: int
    occupancy_threshold: float
    frontier_fps_count: int
    frontier_cosine_margin: float
    frontier_vertical_weight: float
    frontier_lock_steps: int
    frontier_angle_min_deg: float
    frontier_sweep_trigger_m: float
    frontier_sweep_radius_m: float
    frontier_sweep_max_steps: int
    frontier_sweep_min_distance_m: float
    frontier_mask_path: Optional[str]
    heuristic_weight: float
    layer_z_bins: Sequence[Sequence[float]] = ((10.0, 33.0), (34.0, 67.0), (68.0, 110.0))
    explore_layer_radius_m: float = 50.0
    explore_sample_count: int = 300
    layer_dilate_radius_m: float = 6.0
    frontier_min_distance_m: float = 30.0
    frontier_unvisited_only: bool = True
    layer_preference_weight: float = 2.0
    layer_switch_cooldown_steps: int = 30
    layer_goal_blend: float = 0.4
    Offset_decay_distance: float = 30.0
    Explore_update_interval: int = 10
    nav_update_interval: int = 10
    astar_start_free_radius_vox: int = 2
    astar_goal_free_radius_vox: int = 2
    visibility_hysteresis_on_steps: int = 1
    visibility_hysteresis_off_steps: int = 2
    tactical_slot_radius_m: float = 16.0
    tactical_slot_radius_offset_m: float = 6.0
    tactical_slot_min_dist_to_target_m: float = 1.0
    tactical_slot_pullback_step_m: float = 0.5
    tactical_slot_max_pullback_steps: int = 96
    hybrid_predictor_stage_start: int = 2
    hybrid_predictor_gate_scale: float = 0.5
    hybrid_predictor_ema_beta: float = 0.95
    hybrid_predictor_init_error_m: float = 20.0


@dataclass
class RewardCoreDefaultsConfig:
    step_cost: float = -0.05
    progress_gain: float = 0.8
    exploration_gain: float = 0.2
    proximity_cost: float = -0.4
    collision_cost: float = -1.2
    direction_gain: float = 0.15
    control_cost: float = -0.25
    capture_gain: float = 1.5
    capture_quality_gain: float = 0.6


@dataclass
class RewardRuntimeConfig:
    frontier_reach_dist_m: float = 6.0
    lidar_safe_distance_m: float = 4.0
    potential_unstable_obs_jump_m: float = 15.0
    potential_unstable_path_jump_m: float = 10.0
    search_speed_floor_mps: float = 0.0
    direction_gate_active_radius_m: float = 80.0


@dataclass
class CurriculumStageConfig:
    name: str
    target_speed: float
    view_radius_m: float
    guidance_enabled: bool
    target_behavior: str
    collision_penalty: float | None
    randomize_physics: bool
    timesteps: int = 0
    capture_radius_m: Optional[float] = None
    promotion_success_rate: float = 1.0
    entropy_coef: float = 0.0
    entropy_min: float = 0.0
    entropy_decay_success_rate: float = 1.0
    max_std: float = 0.5
    min_std: float = 0.1
    view_speed_schedule: Optional[List[Dict[str, float]]] = None
    reward_core: Optional[Dict[str, float]] = None
    target_axis_limit: bool = False
    visibility_gate_min_ratio: float = 0.0
    visibility_gate_max_ratio: float = 0.0
    visibility_gate_open_prob: float = 0.0
    visibility_gate_timeout_steps: int = 0


@dataclass
class CurriculumConfig:
    success_window_fast: int
    success_window_slow: int
    spawn_distance_levels: Sequence[float]
    promotion_win_rate_slow: float
    promotion_collision_max: float
    promotion_required_streak_iters: int
    promotion_min_stage_ratio: float
    demotion_win_rate_fast: float
    anneal_win_rate_fast: float
    substage_promotion_win_rate_slow: float
    substage_promotion_encircle_rate_slow: float
    substage_promotion_collision_max: float
    substage_promotion_required_streak_iters: int
    substage_min_stage_ratio: float
    substage_min_env_steps: int
    substage_cooldown_iters: int
    stage5_early_stop_required_streak_iters: int
    collapse_ratio: float
    stages: List[CurriculumStageConfig]


@dataclass
class LoggingConfig:
    performance_debug: bool = False
    output_root: Path | None = None
    checkpoint_interval: int | None = None
    metrics_filename: str | None = None
    numba_warmup: bool = False


@dataclass
class ExperimentConfig:
    mode: str = "ours"
    observation_profile: str = "local50"
    guidance_backend: str = "astar"
    critic_mode: str = "local"
    eval_controller: str = "none"
    run_tag: str = ""


@dataclass
class TrainingConfig:
    timesteps: int
    actor_lr: float
    critic_lr: float
    gamma: float
    gae_lambda: float
    clip_range: float
    value_coef: float
    max_grad_norm: float
    rollout_length: int
    batch_size: int
    minibatch_size: int
    update_epochs: int
    parallel_envs: int
    bptt_trunc: int
    collision_rate_ema_alpha: float = 0.15
    spawn_promotion_collision_max: float = 0.10
    aux_min_dist_weight: float = 0.0
    aux_ttc_weight: float = 0.0
    aux_collision_prob_weight: float = 0.0
    aux_ttc_max_seconds: float = 5.0
    aux_collision_prob_horizon_s: float = 2.0
    aux_target_pred_weight: float = 0.05
    aux_stage_start_index: int = 2


@dataclass
class ThreeDConfig:
    air_sim: AirSimConfig
    environment: EnvironmentConfig
    control: ControlConfig
    shield: ShieldConfig
    perception: PerceptionConfig
    navigation: NavigationConfig
    reward_core_defaults: RewardCoreDefaultsConfig
    reward_runtime: RewardRuntimeConfig
    curriculum: CurriculumConfig
    experiment: ExperimentConfig
    logging: LoggingConfig
    training: TrainingConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _load_raw(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return _load_yaml(path)


def load_config(path: Path | str | None = None) -> ThreeDConfig:
    """Load config and validate critical constraints."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _load_raw(cfg_path)

    def _stage(item: dict) -> CurriculumStageConfig:
        return CurriculumStageConfig(
            name=str(item["name"]),
            target_speed=float(item["target_speed"]),
            view_radius_m=float(item["view_radius_m"]),
            guidance_enabled=bool(item.get("guidance_enabled", True)),
            target_behavior=str(item.get("target_behavior", "wander")),
            collision_penalty=item.get("collision_penalty"),
            randomize_physics=bool(item.get("randomize_physics", False)),
            timesteps=int(item.get("timesteps", 0)),
            capture_radius_m=item.get("capture_radius_m"),
            promotion_success_rate=float(item.get("promotion_success_rate", 1.0)),
            entropy_coef=float(item.get("entropy_coef", 0.0)),
            entropy_min=float(item.get("entropy_min", 0.0)),
            entropy_decay_success_rate=float(item.get("entropy_decay_success_rate", 1.0)),
            max_std=float(item.get("max_std", 0.5)),
            min_std=float(item.get("min_std", 0.1)),
            view_speed_schedule=item.get("view_speed_schedule"),
            reward_core=item.get("reward_core"),
            target_axis_limit=bool(item.get("target_axis_limit", False)),
            visibility_gate_min_ratio=float(item.get("visibility_gate_min_ratio", 0.0)),
            visibility_gate_max_ratio=float(item.get("visibility_gate_max_ratio", 0.0)),
            visibility_gate_open_prob=float(item.get("visibility_gate_open_prob", 0.0)),
            visibility_gate_timeout_steps=int(item.get("visibility_gate_timeout_steps", 0)),
        )

    curriculum_raw = raw["curriculum"]
    slow_window = int(curriculum_raw.get("success_window_slow", curriculum_raw.get("success_window", 200)))
    fast_window = int(curriculum_raw.get("success_window_fast", 50))
    spawn_levels = curriculum_raw.get("spawn_distance_levels", [50, 100, 150, 200, 300])
    curriculum = CurriculumConfig(
        success_window_fast=fast_window,
        success_window_slow=slow_window,
        spawn_distance_levels=[float(x) for x in spawn_levels],
        promotion_win_rate_slow=float(curriculum_raw.get("promotion_win_rate_slow", 0.75)),
        promotion_collision_max=float(curriculum_raw.get("promotion_collision_max", 0.10)),
        promotion_required_streak_iters=int(curriculum_raw.get("promotion_required_streak_iters", 20)),
        promotion_min_stage_ratio=float(curriculum_raw.get("promotion_min_stage_ratio", 0.30)),
        demotion_win_rate_fast=float(curriculum_raw.get("demotion_win_rate_fast", 0.15)),
        anneal_win_rate_fast=float(curriculum_raw.get("anneal_win_rate_fast", 0.40)),
        substage_promotion_win_rate_slow=float(
            curriculum_raw.get(
                "substage_promotion_win_rate_slow",
                curriculum_raw.get("promotion_win_rate_slow", 0.75),
            )
        ),
        substage_promotion_encircle_rate_slow=float(
            curriculum_raw.get("substage_promotion_encircle_rate_slow", 0.0)
        ),
        substage_promotion_collision_max=float(
            curriculum_raw.get("substage_promotion_collision_max", 0.10)
        ),
        substage_promotion_required_streak_iters=int(
            curriculum_raw.get("substage_promotion_required_streak_iters", 20)
        ),
        substage_min_stage_ratio=float(curriculum_raw.get("substage_min_stage_ratio", 0.30)),
        substage_min_env_steps=int(curriculum_raw.get("substage_min_env_steps", 0)),
        substage_cooldown_iters=int(curriculum_raw.get("substage_cooldown_iters", 20)),
        stage5_early_stop_required_streak_iters=int(
            curriculum_raw.get("stage5_early_stop_required_streak_iters", 20)
        ),
        collapse_ratio=float(curriculum_raw.get("collapse_ratio", 0.7)),
        stages=[_stage(item) for item in curriculum_raw["stages"]],
    )

    training_raw = raw.get("training")
    if training_raw is None:
        raise ValueError("training section missing in config.yaml")

    logging_raw = raw["logging"]
    if logging_raw.get("output_root"):
        logging_raw["output_root"] = Path(logging_raw["output_root"])

    cfg = ThreeDConfig(
        air_sim=AirSimConfig(**raw["air_sim"]),
        environment=EnvironmentConfig(**raw["environment"]),
        control=ControlConfig(**raw["control"]),
        shield=ShieldConfig(**raw["shield"]),
        perception=PerceptionConfig(**raw["perception"]),
        navigation=NavigationConfig(**raw["navigation"]),
        reward_core_defaults=RewardCoreDefaultsConfig(**raw["reward_core_defaults"]),
        reward_runtime=RewardRuntimeConfig(**raw.get("reward_runtime", {})),
        curriculum=curriculum,
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        logging=LoggingConfig(**logging_raw),
        training=TrainingConfig(**training_raw),
    )
    _validate_grid(cfg)
    return cfg


def _validate_grid(cfg: ThreeDConfig) -> None:
    """Validate grid shape consistency between environment config and computed values."""
    shape = cfg.environment.grid_shape
    voxel_size = cfg.environment.voxel_size_m
    expected = (
        int(round(cfg.environment.world_size_m[0] / voxel_size)),
        int(round(cfg.environment.world_size_m[1] / voxel_size)),
        int(round(cfg.environment.world_size_m[2] / voxel_size)),
    )
    if shape != expected:
        raise ValueError(
            f"Computed grid shape {shape} does not match expected {expected}. "
            "Check world_size_m and voxel_size_m."
        )
