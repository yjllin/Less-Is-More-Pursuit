"""MAPPO trainer for VectorizedMutualAStarEnvV2 (native batch interface)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.config import ThreeDConfig, CurriculumStageConfig
from src.controllers.mappo_policy import MAPPOPolicy3D, ActionNoiseConfig
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2


class RolloutBufferVec:
    """Rollout buffer for vectorized environment with shape (T, B*N, ...)."""
    
    def __init__(
        self,
        T: int,
        num_entities: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int,
        critic_obs_dim: Optional[int] = None,
    ):
        self.T = T
        self.num_entities = num_entities
        self.device = device
        self.hidden_dim = hidden_dim
        
        self.obs = torch.zeros(T, num_entities, obs_dim, device=device)
        self.critic_obs = (
            torch.zeros(T, num_entities, int(critic_obs_dim), device=device)
            if critic_obs_dim is not None
            else None
        )
        self.actions = torch.zeros(T, num_entities, action_dim, device=device)
        self.log_probs = torch.zeros(T, num_entities, device=device)
        self.values = torch.zeros(T, num_entities, device=device)
        self.rewards = torch.zeros(T, num_entities, device=device)
        self.dones = torch.zeros(T, num_entities, device=device)
        # mask used in TD bootstrap term (delta): 1 keeps next value, 0 truncates it
        self.bootstrap_masks = torch.ones(T, num_entities, device=device)
        # mask used in GAE trace recursion: 1 keeps trace across steps, 0 cuts trace
        self.trace_masks = torch.ones(T, num_entities, device=device)
        # Optional next-value overrides (e.g., terminal observation bootstrap for truncation)
        self.next_value_overrides = torch.zeros(T, num_entities, device=device)
        self.next_value_override_masks = torch.zeros(T, num_entities, device=device)
        self.masks = torch.ones(T, num_entities, device=device)
        # Split GRU states: (T+1, 2, 1, B, H)
        self.rnn_states = torch.zeros(T + 1, 2, 1, num_entities, hidden_dim, device=device)


class CurriculumScheduler:
    """Stage-based curriculum manager for 3D training."""

    def __init__(self, stages: List[CurriculumStageConfig]) -> None:
        self._stages = stages
        self._enabled = len(stages) > 0
        self._index = 0
        self._current_stage_start_step = 0
        self._promotion_pass_streak = 0
        self._promotion_required_streak = 20
        self._promotion_win_rate_threshold = 0.90
        self._promotion_collision_max = 0.10
        self._promotion_min_stage_ratio = 0.30
        
        # Build timestep thresholds from stage configs
        self._thresholds: List[int] = []
        stage_timesteps = [getattr(s, 'timesteps', 0) for s in stages]
        
        if any(ts > 0 for ts in stage_timesteps):
            total = 0
            for ts in stage_timesteps:
                total += ts
                self._thresholds.append(total)
            print(f"[Curriculum] Stage timesteps: {stage_timesteps}")
            print(f"[Curriculum] Thresholds: {self._thresholds}")
        else:
            self._thresholds = [float('inf')] * len(stages)

    def configure_promotion(
        self,
        *,
        win_rate_threshold: float,
        collision_max: float,
        required_streak: int,
        min_stage_ratio: float,
    ) -> None:
        self._promotion_win_rate_threshold = float(win_rate_threshold)
        self._promotion_collision_max = float(collision_max)
        self._promotion_required_streak = max(1, int(required_streak))
        self._promotion_min_stage_ratio = max(0.0, float(min_stage_ratio))

    @property
    def current_index(self) -> int:
        return self._index

    def current_stage(self) -> Optional[CurriculumStageConfig]:
        if not self._enabled or self._index >= len(self._stages):
            return None
        return self._stages[self._index]

    def maybe_advance(
        self,
        total_steps: int,
        win_rate: Optional[float] = None,
        collision_rate: Optional[float] = None,
        window_len: int = 0,
        min_window: int = 0,
    ) -> bool:
        if not self._enabled or self._index >= len(self._thresholds) - 1:
            return False
        stage = self._stages[self._index]
        stage_start = int(self._current_stage_start_step)
        stage_budget = max(0, int(getattr(stage, "timesteps", 0)))
        min_stage_steps = int(np.ceil(stage_budget * self._promotion_min_stage_ratio)) if stage_budget > 0 else 0
        enough_stage_steps = (total_steps - stage_start) >= min_stage_steps

        pass_now = False
        if (
            win_rate is not None
            and collision_rate is not None
            and (not min_window or window_len >= min_window)
        ):
            pass_now = (win_rate > self._promotion_win_rate_threshold) and (collision_rate < self._promotion_collision_max)

        if enough_stage_steps and pass_now:
            self._promotion_pass_streak += 1
        else:
            self._promotion_pass_streak = 0

        if self._promotion_pass_streak < self._promotion_required_streak:
            return False

        self._index += 1
        self._current_stage_start_step = int(total_steps)
        self._promotion_pass_streak = 0
        return True

    def set_stage(self, index: int) -> None:
        if 0 <= index < len(self._stages):
            self._index = index
            self._current_stage_start_step = 0
            self._promotion_pass_streak = 0


class MAPPOTrainerVec:
    """
    MAPPO Trainer optimized for VectorizedMutualAStarEnvV2.
    
    Key differences from MAPPOTrainer3D:
    - Single vectorized env instead of list of envs
    - Native batch operations without Python loops
    - Direct numpy<->torch conversion for efficiency
    """
    
    def __init__(
        self,
        config: ThreeDConfig,
        num_envs: int,
        device: Optional[torch.device] = None,
        init_policy: Optional[str] = None,
        start_stage_index: Optional[int] = None,
        start_spawn_level: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.num_envs = num_envs
        
        # Initialize curriculum
        self.curriculum = CurriculumScheduler(config.curriculum.stages)
        self.curriculum.configure_promotion(
            win_rate_threshold=float(getattr(config.curriculum, "promotion_win_rate_slow", 0.90)),
            collision_max=float(getattr(config.curriculum, "promotion_collision_max", 0.10)),
            required_streak=int(getattr(config.curriculum, "promotion_required_streak_iters", 20)),
            min_stage_ratio=float(getattr(config.curriculum, "promotion_min_stage_ratio", 0.30)),
        )
        if start_stage_index is not None:
            self.curriculum.set_stage(start_stage_index)
        self._active_stage = self.curriculum.current_stage()
        self._reset_stage_index = start_stage_index
        self._entropy_coef = 0.0
        self._entropy_coef_init = 0.0
        self._entropy_coef_min = 0.0
        self._entropy_decay_threshold = 1.0
        self._decay_lower = 0.4
        self._decay_upper = 0.9
        self._success_window_slow = int(getattr(self.cfg.curriculum, "success_window_slow", 500))
        self._success_window_fast = int(getattr(self.cfg.curriculum, "success_window_fast", 50))
        # Single win metric: clean capture rate over a fixed 150-episode window.
        self._clean_win_window = 150
        self._recent_clean_wins = deque(maxlen=self._clean_win_window)
        self._recent_capture_raw_slow = deque(maxlen=self._success_window_slow)
        self._recent_capture_raw_fast = deque(maxlen=self._success_window_fast)
        self._recent_clean_capture_slow = deque(maxlen=self._success_window_slow)
        self._recent_clean_capture_fast = deque(maxlen=self._success_window_fast)
        self._recent_encircle_capture_slow = deque(maxlen=self._success_window_slow)
        self._recent_encircle_capture_fast = deque(maxlen=self._success_window_fast)
        self._last_clean_win_rate = 0.0
        self._last_capture_rate_slow = 0.0
        self._last_capture_rate_fast = 0.0
        self._last_clean_capture_rate_slow = 0.0
        self._last_clean_capture_rate_fast = 0.0
        self._last_encircle_capture_rate_slow = 0.0
        self._last_encircle_capture_rate_fast = 0.0
        self._last_collision_rate = 0.0
        self._collision_rate_ema = None
        self._rollout_episode_collisions: list[float] = []
        self._prev_clean_win_rate = 0.0
        self._best_clean_win_rate = 0.0
        self._promotion_win_rate_slow = float(getattr(self.cfg.curriculum, "promotion_win_rate_slow", 0.75))
        self._anneal_win_rate_fast = float(getattr(self.cfg.curriculum, "anneal_win_rate_fast", 0.40))
        self._substage_promotion_win_rate_slow = float(
            getattr(self.cfg.curriculum, "substage_promotion_win_rate_slow", self._promotion_win_rate_slow)
        )
        self._substage_promotion_encircle_rate_slow = float(
            getattr(self.cfg.curriculum, "substage_promotion_encircle_rate_slow", 0.0)
        )
        self._substage_promotion_collision_max = float(
            getattr(self.cfg.curriculum, "substage_promotion_collision_max", 0.10)
        )
        self._substage_min_env_steps = int(getattr(self.cfg.curriculum, "substage_min_env_steps", 0))
        self._substage_cooldown_iters = max(0, int(getattr(self.cfg.curriculum, "substage_cooldown_iters", 20)))
        self._substage_cooldown = self._substage_cooldown_iters
        self._last_substage_advance_env_steps = 0
        self._substage_promotion_pass_streak = 0
        self._substage_promotion_required_streak = max(
            1, int(getattr(self.cfg.curriculum, "substage_promotion_required_streak_iters", 20))
        )
        self._substage_promotion_win_rate_slow = float(
            getattr(self.cfg.curriculum, "substage_promotion_win_rate_slow", 0.90)
        )
        self._substage_promotion_collision_max = float(
            getattr(self.cfg.curriculum, "substage_promotion_collision_max", 0.10)
        )
        self._substage_min_stage_ratio = max(
            0.0, float(getattr(self.cfg.curriculum, "substage_min_stage_ratio", 0.30))
        )
        self._collapse_ratio = float(getattr(self.cfg.curriculum, "collapse_ratio", 0.7))
        self._stage_view_speed_schedule: Optional[List[Dict[str, float]]] = None
        self._stage_view_speed_idx = 0
        self._symmetry_aug_prob = 0.5
        self._stage_entry_env_steps = 0
        self._stage5_early_stop_streak = 0
        self._stage5_early_stop_required_streak = max(
            1, int(getattr(self.cfg.curriculum, "stage5_early_stop_required_streak_iters", 20))
        )
        self._stage5_early_stop_win_rate_threshold = 0.90
        self._stage5_early_stop_min_stage_ratio = 0.30
        exp_cfg = getattr(self.cfg, "experiment", None)
        self._experiment_mode = str(getattr(exp_cfg, "mode", "ours"))
        self._observation_profile = str(getattr(exp_cfg, "observation_profile", "full83")).lower()
        self._guidance_backend = str(getattr(exp_cfg, "guidance_backend", "astar")).lower()
        self._critic_mode = str(getattr(exp_cfg, "critic_mode", "local")).lower()
        self._centralized_critic = self._critic_mode == "ctde_joint_obs_plus_global"

        # Create vectorized environment
        self.env = VectorizedMutualAStarEnvV2(num_envs=num_envs, cfg=config)
        if hasattr(self.env, "set_runtime_mode"):
            self.env.set_runtime_mode(training=True)
        self.num_agents = self.env.num_agents
        self.obs_dim = self.env.get_obs_dim()
        self.action_dim = 4
        self.num_entities = num_envs * self.num_agents  # Total agents across all envs
        self._episode_collisions = np.zeros((num_envs, self.num_agents), dtype=np.bool_)
        if start_spawn_level is not None:
            print("[Trainer] spawn_level mechanism is deprecated and ignored.")
        if self._observation_profile == "local50":
            # Local-observation baseline removes teammate/tactical topology blocks.
            self._symmetry_aug_prob = 0.0
        self._init_symmetry_indices()
        self._aux_min_dist_weight = float(getattr(self.cfg.training, "aux_min_dist_weight", 0.0))
        self._aux_ttc_weight_base = float(getattr(self.cfg.training, "aux_ttc_weight", 0.0))
        self._aux_collision_prob_weight_base = float(getattr(self.cfg.training, "aux_collision_prob_weight", 0.0))
        self._aux_target_pred_weight_base = float(getattr(self.cfg.training, "aux_target_pred_weight", 0.05))
        self._aux_stage_start_index = int(getattr(self.cfg.training, "aux_stage_start_index", 2))
        self._aux_ttc_weight = self._aux_ttc_weight_base
        self._aux_collision_prob_weight = self._aux_collision_prob_weight_base
        self._aux_target_pred_weight = self._aux_target_pred_weight_base
        self._aux_ttc_max_seconds = float(getattr(self.cfg.training, "aux_ttc_max_seconds", 5.0))
        self._aux_collision_prob_horizon_s = float(getattr(self.cfg.training, "aux_collision_prob_horizon_s", 2.0))
        self._update_aux_weights_for_stage(self.curriculum.current_index)
        self._critic_obs_dim = self.num_agents * self.obs_dim + 6 if self._centralized_critic else None
        
        # Initialize policy
        init_min_std, init_max_std = self._resolve_std_bounds(self._active_stage)
        action_noise = ActionNoiseConfig(initial_std=init_max_std, min_std=init_min_std, decay_rate=1e-6)
        self.policy = MAPPOPolicy3D(
            self.obs_dim,
            self.action_dim,
            self.device,
            action_noise=action_noise,
            hidden_dim=512,
            action_bounds=self.cfg.control.action_bounds,
            min_action_std=init_min_std,
            max_action_std=init_max_std,
            centralized_critic=self._centralized_critic,
            critic_obs_dim=self._critic_obs_dim,
        )
        
        if init_policy:
            ckpt = torch.load(init_policy, map_location=self.device)
            if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
                self.policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
            else:
                self.policy.load_state_dict(ckpt, strict=False)
            print(f"[Trainer] Loaded policy from {init_policy}")
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.training.actor_lr, eps=1e-5)

        # Stage-specific entropy setup
        self._reset_entropy_for_stage(self._active_stage)
        
        # Rollout buffer
        self.buffer = RolloutBufferVec(
            self.cfg.training.rollout_length,
            self.num_entities,
            self.obs_dim,
            self.action_dim,
            self.device,
            int(getattr(self.policy.model, "hidden_dim", 256)),
            critic_obs_dim=self._critic_obs_dim,
        )
        
        # RNN state: (2, 1, num_entities, hidden_dim)
        self.rnn_state = self.policy.init_hidden(self.num_entities)
        self._bootstrap_obs: Optional[torch.Tensor] = None
        self._bootstrap_critic_obs: Optional[torch.Tensor] = None
        self._bootstrap_masks: Optional[torch.Tensor] = None
        
        # Metrics tracking
        self._reward_sums: Dict[str, float] = {}
        self._reward_count = 0
        self._collision_terminations = 0
        self._team_target_min_dist_3d_sum = 0.0
        self._team_target_mean_abs_dz_sum = 0.0
        self._far_idle_ratio_sum = 0.0
        self._capture_contribution_sum = 0.0
        self._dist_to_target_p50_sum = 0.0
        self._dist_to_target_p90_sum = 0.0
        self._step_metric_count = 0
        self._last_action_std = 0.0
        
        # Logging
        self._init_logging()
        self.iteration = 0
        self._perf_debug = bool(getattr(self.cfg.logging, "performance_debug", False))
        self._perf_accum: Dict[str, float] = {}
        if self._perf_debug:
            self._perf_accum = {
                "act": 0.0,
                "env": 0.0,
                "buffer": 0.0,
                "env_pre": 0.0,
                "env_agents": 0.0,
                "env_collisions": 0.0,
                "env_targets": 0.0,
                "env_lidar": 0.0,
                "env_explore": 0.0,
                "env_nav": 0.0,
                "env_rewards": 0.0,
                "env_respawn": 0.0,
                "env_obs": 0.0,
                "nav_astar": 0.0,
                "nav_path_len": 0.0,
                "nav_frontier_envs": 0.0,
                "nav_visible_frac": 0.0,
            }
        self._log_ignored_config_warnings()

    def _log_ignored_config_warnings(self) -> None:
        """Log config fields that are currently no-ops in the vectorized training path."""
        msgs: list[str] = []
        if hasattr(self.cfg, "air_sim"):
            msgs.append("air_sim.* is not used by VectorizedMutualAStarEnvV2 training path.")
        if float(getattr(self.cfg.training, "critic_lr", self.cfg.training.actor_lr)) != float(self.cfg.training.actor_lr):
            msgs.append(
                f"training.critic_lr={self.cfg.training.critic_lr} is currently ignored; optimizer uses actor_lr={self.cfg.training.actor_lr}."
            )
        else:
            msgs.append("training.critic_lr is currently ignored; optimizer uses actor_lr only.")
        msgs.append(
            f"training.parallel_envs={getattr(self.cfg.training, 'parallel_envs', self.num_envs)} is informational here; actual num_envs={self.num_envs}."
        )
        if hasattr(self.cfg.training, "spawn_promotion_collision_max"):
            msgs.append("training.spawn_promotion_collision_max is currently unused in MAPPOTrainerVec.")
        if hasattr(self.cfg.curriculum, "demotion_win_rate_fast"):
            msgs.append("curriculum.demotion_win_rate_fast is currently unused (no demotion path implemented).")
        if any(float(getattr(s, "promotion_success_rate", 1.0)) != 1.0 for s in self.cfg.curriculum.stages):
            msgs.append("curriculum.stages[].promotion_success_rate is ignored; stage promotion uses global streak+collision thresholds.")
        if any(bool(getattr(s, "randomize_physics", False)) for s in self.cfg.curriculum.stages):
            msgs.append("curriculum.stages[].randomize_physics is currently ignored in vectorized env stage application.")
        msgs.append("perception.lidar_max_points / perception.bin_shape / perception.obstacle_sector_floor are not used in the current vectorized LiDAR path.")
        if msgs:
            print("[Config Warning] The following fields are currently no-ops / informational:", flush=True)
            for m in msgs:
                print(f"  - {m}", flush=True)

    def _init_logging(self) -> None:
        log_root = Path(self.cfg.logging.output_root)
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_root = log_root
        self._metrics_path = log_root / (self.cfg.logging.metrics_filename or "metrics.jsonl")
        
        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Unserializable type: {type(obj)}")
        
        with (log_root / "config.json").open("w", encoding="utf-8") as fp:
            json.dump(asdict(self.cfg), fp, ensure_ascii=False, indent=2, default=_serialize)

    def train(self) -> None:
        """Main training loop."""
        total_steps = self.cfg.training.timesteps
        rollout_len = self.cfg.training.rollout_length
        steps_per_iter = rollout_len * self.num_entities
        
        global_step = 0
        next_checkpoint_step = self.cfg.logging.checkpoint_interval or 100_000
        train_start = time.time()
        
        print("=" * 60)
        print(f"[Train] MAPPO-Vec | envs={self.num_envs} | agents={self.num_agents} | device={self.device}")
        print(f"  timesteps: {total_steps:,} | rollout: {rollout_len} | entities: {self.num_entities}")
        print(f"  logdir: {self.log_root}")
        print("=" * 60)
        print("[Train] Resetting environments...", flush=True)
        
        # Initial reset
        obs, _ = self.env.reset(stage_index=self._reset_stage_index)
        obs = self._to_flat_tensor(obs)  # (B, N, D) -> (B*N, D)
        masks = torch.ones(self.num_entities, 1, device=self.device)
        print("[Train] Environments ready.", flush=True)
        
        while global_step < total_steps:
            iter_start = time.time()
            self._collision_terminations = 0
            self._team_target_min_dist_3d_sum = 0.0
            self._team_target_mean_abs_dz_sum = 0.0
            self._far_idle_ratio_sum = 0.0
            self._capture_contribution_sum = 0.0
            self._dist_to_target_p50_sum = 0.0
            self._dist_to_target_p90_sum = 0.0
            self._step_metric_count = 0
            if self._perf_debug:
                self._perf_accum = {
                    "act": 0.0,
                    "env": 0.0,
                    "buffer": 0.0,
                    "env_pre": 0.0,
                    "env_agents": 0.0,
                    "env_collisions": 0.0,
                    "env_targets": 0.0,
                    "env_lidar": 0.0,
                    "env_explore": 0.0,
                    "env_nav": 0.0,
                    "env_rewards": 0.0,
                    "env_respawn": 0.0,
                    "env_obs": 0.0,
                    "nav_astar": 0.0,
                    "nav_path_len": 0.0,
                    "nav_frontier_envs": 0.0,
                    "nav_visible_frac": 0.0,
                }
            
            # Collect rollout
            obs, masks = self._collect_rollout(obs, masks)
            self._bootstrap_obs = obs.detach()
            self._bootstrap_critic_obs = self._build_critic_obs_from_env(obs).detach()
            self._bootstrap_masks = masks.detach()
            self._update_entropy_coef(global_step)
            
            # Update policy
            metrics = self._update()
            
            global_step += steps_per_iter
            self.iteration += 1
            stage5_early_stop_now = self._should_early_stop_stage5(global_step)
            
            # Build metrics
            iter_time = time.time() - iter_start
            sps = steps_per_iter / max(iter_time, 1e-6)
            
            metrics.update({
                "iteration": self.iteration,
                "env_steps": global_step,
                "sps": sps,
                "curriculum_stage": self._active_stage.name if self._active_stage else "none",
                "action_std": self._last_action_std,
                "collision_rate": self._last_collision_rate,
                "entropy_coef": self._entropy_coef,
                "clean_win": self._last_clean_win_rate,
                "capture_rate_slow": self._last_capture_rate_slow,
                "capture_rate_fast": self._last_capture_rate_fast,
                "mean_team_target_dist_3d": (
                    self._team_target_min_dist_3d_sum / max(float(self._step_metric_count), 1.0)
                ),
                "mean_abs_dz_to_target": (
                    self._team_target_mean_abs_dz_sum / max(float(self._step_metric_count), 1.0)
                ),
                "far_idle_ratio": (
                    self._far_idle_ratio_sum / max(float(self._step_metric_count), 1.0)
                ),
                "capture_contribution": (
                    self._capture_contribution_sum / max(float(self._step_metric_count), 1.0)
                ),
                "dist_to_target_p50": (
                    self._dist_to_target_p50_sum / max(float(self._step_metric_count), 1.0)
                ),
                "dist_to_target_p90": (
                    self._dist_to_target_p90_sum / max(float(self._step_metric_count), 1.0)
                ),
                "substage_index": int(self._stage_view_speed_idx + 1),
                "substage_cooldown": int(self._substage_cooldown),
                "stage5_early_stop_streak": int(self._stage5_early_stop_streak),
            })
            if self._perf_debug and self._perf_accum:
                metrics.update({
                    "perf_policy_act_s": self._perf_accum.get("act", 0.0),
                    "perf_env_step_s": self._perf_accum.get("env", 0.0),
                    "perf_buffer_s": self._perf_accum.get("buffer", 0.0),
                })
            
            # Add reward breakdown
            for key, value in self._reward_sums.items():
                if key == "reward_total":
                    metrics["reward_total_sum"] = value
                else:
                    metrics[f"reward_{key}_sum"] = value
            metrics["reward_samples"] = self._reward_count
            
            # Print progress (before clearing reward sums)
            reward_total = self._reward_sums.get("reward_total", 0)
            avg_reward = reward_total / max(self._reward_count, 1)
            if self._perf_debug and self._perf_accum:
                perf_act = self._perf_accum.get("act", 0.0)
                perf_env = self._perf_accum.get("env", 0.0)
                perf_buf = self._perf_accum.get("buffer", 0.0)
                perf_pre = self._perf_accum.get("env_pre", 0.0)
                perf_agents = self._perf_accum.get("env_agents", 0.0)
                perf_collisions = self._perf_accum.get("env_collisions", 0.0)
                perf_targets = self._perf_accum.get("env_targets", 0.0)
                perf_lidar = self._perf_accum.get("env_lidar", 0.0)
                perf_explore = self._perf_accum.get("env_explore", 0.0)
                perf_nav = self._perf_accum.get("env_nav", 0.0)
                perf_rewards = self._perf_accum.get("env_rewards", 0.0)
                perf_respawn = self._perf_accum.get("env_respawn", 0.0)
                perf_obs = self._perf_accum.get("env_obs", 0.0)
                nav_astar = self._perf_accum.get("nav_astar", 0.0)
                nav_path_len = self._perf_accum.get("nav_path_len", 0.0)
                nav_frontier_envs = self._perf_accum.get("nav_frontier_envs", 0.0)
                nav_visible_frac = self._perf_accum.get("nav_visible_frac", 0.0)
                rollout_len = float(self.cfg.training.rollout_length)
                nav_astar_per_step = nav_astar / max(rollout_len, 1.0)
                nav_frontier_per_step = nav_frontier_envs / max(rollout_len, 1.0)
                nav_visible_avg = nav_visible_frac / max(rollout_len, 1.0)
                nav_path_len_avg = nav_path_len / max(nav_astar, 1.0)
                print(f"[Iter {self.iteration}] steps={global_step:,} | sps={sps:.0f} | "
                      f"stage={metrics['curriculum_stage']} | reward_sum={reward_total:.0f} | avg_reward={avg_reward:.2f} | "
                      f"clean_win={self._last_clean_win_rate:.3f} | cap={self._last_capture_rate_slow:.3f} | "
                      f"collision_rate={self._last_collision_rate:.4f} | "
                      f"policy_loss={metrics['policy_loss']:.4f} | value_loss={metrics['value_loss']:.4f} | "
                      f"approx_kl={metrics.get('approx_kl', 0.0):.5f} | "
                      f"perf_act={perf_act:.2f}s | perf_env={perf_env:.2f}s | perf_buf={perf_buf:.2f}s | "
                      f"env_pre={perf_pre:.2f}s | env_agents={perf_agents:.2f}s | env_col={perf_collisions:.2f}s | "
                      f"env_targets={perf_targets:.2f}s | env_lidar={perf_lidar:.2f}s | env_explore={perf_explore:.2f}s | "
                      f"env_nav={perf_nav:.2f}s | env_rewards={perf_rewards:.2f}s | env_respawn={perf_respawn:.2f}s | "
                      f"env_obs={perf_obs:.2f}s | nav_astar/step={nav_astar_per_step:.2f} | "
                      f"nav_frontier/step={nav_frontier_per_step:.2f} | nav_visible={nav_visible_avg:.2f} | "
                      f"nav_path_len={nav_path_len_avg:.1f}")
            else:
                print(f"[Iter {self.iteration}] steps={global_step:,} | sps={sps:.0f} | "
                      f"stage={metrics['curriculum_stage']} | reward_sum={reward_total:.0f} | avg_reward={avg_reward:.2f} | "
                      f"clean_win={self._last_clean_win_rate:.3f} | cap={self._last_capture_rate_slow:.3f} | "
                      f"collision_rate={self._last_collision_rate:.4f} | "
                      f"policy_loss={metrics['policy_loss']:.4f} | value_loss={metrics['value_loss']:.4f} | "
                      f"approx_kl={metrics.get('approx_kl', 0.0):.5f}")
            
            # Log (this clears _reward_sums)
            self._log_metrics(metrics)
            
            # Checkpoint
            if global_step >= next_checkpoint_step:
                self._save_checkpoint(global_step)
                next_checkpoint_step += self.cfg.logging.checkpoint_interval or 100_000

            # Stage5 early stop: after minimum stage progress and sustained mastery.
            if stage5_early_stop_now:
                print(
                    f"[EarlyStop] Stage5 criteria met at iter={self.iteration}, step={global_step:,}. "
                    f"clean_win={self._last_clean_win_rate:.3f}, streak={self._stage5_early_stop_streak}",
                    flush=True,
                )
                break
            
            # Curriculum advancement
            if self.curriculum.maybe_advance(
                global_step,
                win_rate=self._last_clean_win_rate,
                collision_rate=self._last_collision_rate,
                window_len=len(self._recent_clean_wins),
                min_window=self._clean_win_window,
            ):
                obs, masks = self._on_stage_advance(global_step)
        
        # Final save
        self._save_checkpoint(global_step)
        total_time = time.time() - train_start
        print(f"\n[Train] Complete! Total time: {total_time/3600:.2f}h | Avg SPS: {global_step/total_time:.0f}")

    def _should_early_stop_stage5(self, global_step: int) -> bool:
        """Early-stop training once Stage5 is stably mastered."""
        stage = self._active_stage
        stage_name = str(getattr(stage, "name", "")).lower()
        if not stage_name.startswith("stage5"):
            self._stage5_early_stop_streak = 0
            return False
        stage_budget = max(0, int(getattr(stage, "timesteps", 0)))
        min_stage_steps = int(np.ceil(stage_budget * self._stage5_early_stop_min_stage_ratio)) if stage_budget > 0 else 0
        stage_steps = int(global_step) - int(self._stage_entry_env_steps)
        enough_steps = stage_steps >= min_stage_steps
        pass_now = float(self._last_clean_win_rate) >= self._stage5_early_stop_win_rate_threshold
        if enough_steps and pass_now:
            self._stage5_early_stop_streak += 1
        else:
            self._stage5_early_stop_streak = 0
        return self._stage5_early_stop_streak >= self._stage5_early_stop_required_streak

    def _to_flat_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert (B, N, D) numpy array to (B*N, D) tensor."""
        B, N, D = obs.shape
        flat = obs.reshape(B * N, D)
        return torch.from_numpy(flat.astype(np.float32)).to(self.device)

    def _build_critic_obs_from_batch_and_target(
        self,
        obs_flat: torch.Tensor,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> torch.Tensor:
        """Build per-entity CTDE critic input: joint obs + true target pos/vel."""
        if not self._centralized_critic:
            return obs_flat
        obs_bn = obs_flat.reshape(self.num_envs, self.num_agents, self.obs_dim)
        joint = obs_bn.reshape(self.num_envs, self.num_agents * self.obs_dim)
        target_state = np.concatenate(
            (
                np.asarray(target_pos, dtype=np.float32).reshape(self.num_envs, 3),
                np.asarray(target_vel, dtype=np.float32).reshape(self.num_envs, 3),
            ),
            axis=-1,
        )
        target_state_t = torch.from_numpy(target_state).to(self.device)
        critic_env = torch.cat((joint, target_state_t), dim=-1)
        return critic_env[:, None, :].expand(-1, self.num_agents, -1).reshape(self.num_entities, -1)

    def _build_critic_obs_from_env(self, obs_flat: torch.Tensor) -> torch.Tensor:
        if not self._centralized_critic:
            return obs_flat
        return self._build_critic_obs_from_batch_and_target(obs_flat, self.env.target_pos, self.env.target_vel)

    def _to_batch_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Convert (B*N, 4) tensor to (B, N, 4) numpy array."""
        actions_np = actions.detach().cpu().numpy()
        return actions_np.reshape(self.num_envs, self.num_agents, self.action_dim)

    def _to_batch_target_rel_prediction(self, pred_rel: torch.Tensor) -> np.ndarray:
        """Convert (B*N, 3) normalized target-rel prediction to (B, N, 3)."""
        pred_np = pred_rel.detach().cpu().numpy().astype(np.float64)
        pred_np = np.clip(pred_np, -1.0, 1.0)
        return pred_np.reshape(self.num_envs, self.num_agents, 3)

    def _extract_min_lidar(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Extract min lidar distance from the most recent stacked frame."""
        base_start = (self._frame_stack - 1) * self._obs_base_dim
        lidar = obs_flat[:, base_start:base_start + self._obs_lidar_dim]
        lidar = torch.clamp(lidar, 0.0, 1.0)
        return torch.min(lidar, dim=-1).values

    def _build_aux_targets(self, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build auxiliary targets: min_dist, ttc (norm), collision_prob."""
        base_start = (self._frame_stack - 1) * self._obs_base_dim
        lidar = obs_flat[:, base_start:base_start + self._obs_lidar_dim]
        lidar = torch.clamp(lidar, 0.0, 1.0)
        min_dist_norm = torch.min(lidar, dim=-1).values

        vel_start = base_start + self._obs_self_vel_start
        vel_norm = obs_flat[:, vel_start:vel_start + 3]
        speed = torch.norm(vel_norm, dim=-1)
        max_speed = float(getattr(self.env, "max_speed", 1.0))
        speed_mps = speed * max_speed

        lidar_max_range = float(getattr(self.env, "lidar_max_range", 1.0))
        dist_m = min_dist_norm * max(lidar_max_range, 1e-6)
        ttc_max = max(self._aux_ttc_max_seconds, 1e-3)
        ttc = dist_m / torch.clamp(speed_mps, min=1e-3)
        ttc = torch.clamp(ttc, max=ttc_max)
        ttc_norm = ttc / ttc_max

        horizon = max(self._aux_collision_prob_horizon_s, 1e-6)
        collision_prob = 1.0 - torch.clamp(ttc / horizon, 0.0, 1.0)
        return min_dist_norm, ttc_norm, collision_prob

    def _init_symmetry_indices(self) -> None:
        """Precompute observation indices for left-right mirroring."""
        top_k = int(self.cfg.perception.teammate_top_k)
        self._frame_stack = max(1, int(getattr(self.cfg.environment, "frame_stack", 1)))
        if self.obs_dim % self._frame_stack != 0:
            raise ValueError("Observation layout mismatch for symmetry augmentation.")
        self._obs_base_dim = self.obs_dim // self._frame_stack

        self._obs_lidar_dim = 26
        self._obs_target_rel_start = 26
        self._obs_target_vel_start = self._obs_target_rel_start + 3
        self._obs_self_vel_start = self._obs_target_vel_start + 3
        self._obs_imu_acc_start = self._obs_self_vel_start + 3
        self._obs_imu_gyro_start = self._obs_imu_acc_start + 3

        profile = getattr(self, "_observation_profile", "full83")
        if profile == "local50":
            self._obs_teammate_start = -1
            self._obs_teammate_dim = 0
            self._obs_guidance_start = self._obs_imu_gyro_start + 3
            self._obs_agent_id_start = self._obs_guidance_start + 3
            self._obs_target_mode_idx = self._obs_agent_id_start + self.num_agents
            self._obs_delay_idx = self._obs_target_mode_idx + 1
            self._obs_slot_id_start = -1
            self._obs_slot_id_dim = 0
            self._obs_slot_rel_start = -1
            self._obs_slot_rel_dim = 0
            self._obs_encircle_start = -1
            self._obs_encircle_dim = 0
            if self._obs_delay_idx + 1 != self._obs_base_dim:
                raise ValueError("local50 observation layout mismatch for symmetry augmentation.")
        else:
            self._obs_teammate_start = self._obs_imu_gyro_start + 3
            self._obs_teammate_dim = top_k * 8
            self._obs_guidance_start = self._obs_teammate_start + self._obs_teammate_dim
            self._obs_agent_id_start = self._obs_guidance_start + 3
            self._obs_target_mode_idx = self._obs_agent_id_start + self.num_agents
            self._obs_delay_idx = self._obs_target_mode_idx + 1
            self._obs_slot_id_start = self._obs_delay_idx + 1
            self._obs_slot_id_dim = self.num_agents
            self._obs_slot_rel_start = self._obs_slot_id_start + self._obs_slot_id_dim
            self._obs_slot_rel_dim = 3
            self._obs_encircle_start = self._obs_slot_rel_start + self._obs_slot_rel_dim
            self._obs_encircle_dim = 2
            if self._obs_encircle_start + self._obs_encircle_dim != self._obs_base_dim:
                raise ValueError("Observation layout mismatch for symmetry augmentation.")
        self._lidar_mirror_idx = self._build_lidar_mirror_index()

    def _build_lidar_mirror_index(self) -> torch.Tensor:
        """Build index mapping for mirroring lidar sectors across body Y."""
        dirs = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    dirs.append((dx, dy, dz))
        index_map = []
        for dx, dy, dz in dirs:
            mirrored = (dx, -dy, dz)
            try:
                index_map.append(dirs.index(mirrored))
            except ValueError as exc:
                raise ValueError("Failed to build lidar mirror mapping.") from exc
        return torch.tensor(index_map, dtype=torch.long, device=self.device)

    def _mirror_observations(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mirror observations across body Y for selected entities."""
        if not torch.any(mask):
            return obs
        obs_m = obs.clone()
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        for f in range(self._frame_stack):
            base = f * self._obs_base_dim
            # 1) Lidar sectors
            obs_m[idx, base:base + self._obs_lidar_dim] = obs_m.index_select(0, idx)[:, base:base + self._obs_lidar_dim][:, self._lidar_mirror_idx]
            # 2) Target relative position (body frame): flip Y
            obs_m[idx, base + self._obs_target_rel_start + 1] *= -1.0
            # 3) Target relative velocity (body frame): flip Y
            obs_m[idx, base + self._obs_target_vel_start + 1] *= -1.0
            # 4) Self velocity (body frame): flip Y
            obs_m[idx, base + self._obs_self_vel_start + 1] *= -1.0
            # 5) IMU linear acceleration (body frame): flip Y
            obs_m[idx, base + self._obs_imu_acc_start + 1] *= -1.0
            # 6) IMU angular velocity (body frame): flip Y and Z
            obs_m[idx, base + self._obs_imu_gyro_start + 1] *= -1.0
            obs_m[idx, base + self._obs_imu_gyro_start + 2] *= -1.0
            # 7) Teammate features: for each block of 8, flip dy and relative vy
            if self._obs_teammate_dim > 0:
                for k in range(0, self._obs_teammate_dim, 8):
                    obs_m[idx, base + self._obs_teammate_start + k + 1] *= -1.0
                    obs_m[idx, base + self._obs_teammate_start + k + 6] *= -1.0
            # 8) Guidance (body frame): flip Y
            obs_m[idx, base + self._obs_guidance_start + 1] *= -1.0
            # 9) Tactical slot relative vector (body frame): flip Y
            if self._obs_slot_rel_dim > 0:
                obs_m[idx, base + self._obs_slot_rel_start + 1] *= -1.0
        return obs_m

    def _mirror_actions(self, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mirror actions across body Y for selected entities."""
        if not torch.any(mask):
            return actions
        actions_m = actions.clone()
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        actions_m[idx, 1] *= -1.0  # vy
        actions_m[idx, 3] *= -1.0  # yaw_rate
        return actions_m

    def _collect_rollout(self, obs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect rollout data from vectorized environment."""
        rnn_state = self.rnn_state
        rollout_len = self.cfg.training.rollout_length
        reward_scale = 1.0
        self._rollout_episode_collisions.clear()
        rollout_std_sum = 0.0
        rollout_std_count = 0
        
        for t in range(rollout_len):
            # Get actions from policy
            act_start = time.perf_counter() if self._perf_debug else 0.0
            obs_policy = obs
            critic_obs_t = self._build_critic_obs_from_env(obs)
            mirror_mask = None
            if self._symmetry_aug_prob > 0.0:
                mirror_mask = (torch.rand(self.num_entities, device=self.device) < self._symmetry_aug_prob)
                if torch.any(mirror_mask):
                    obs_policy = self._mirror_observations(obs, mirror_mask)
            actions, logp, values, next_rnn, pred_target_rel = self.policy.act(
                obs_policy,
                rnn_state,
                masks,
                return_pred_target=True,
                critic_obs=critic_obs_t,
            )
            rollout_std_sum += self.policy.get_current_std()
            rollout_std_count += 1
            if self._perf_debug:
                self._perf_accum["act"] += time.perf_counter() - act_start
            
            # Step environment (convert to batch format)
            actions_env = actions
            if mirror_mask is not None and torch.any(mirror_mask):
                actions_env = self._mirror_actions(actions, mirror_mask)
                # Undo mirrored body-Y sign so env always receives canonical predictions.
                pred_target_rel = pred_target_rel.clone()
                pred_target_rel[mirror_mask, 1] *= -1.0
            self.env.set_nn_target_predictions(self._to_batch_target_rel_prediction(pred_target_rel))
            actions_batch = self._to_batch_actions(actions_env)
            env_start = time.perf_counter() if self._perf_debug else 0.0
            next_obs, rewards, dones, infos = self.env.step(actions_batch)
            if self._perf_debug:
                self._perf_accum["env"] += time.perf_counter() - env_start
                perf = getattr(self.env, "_perf_last", None)
                if perf:
                    self._perf_accum["env_pre"] += perf.get("pre", 0.0)
                    self._perf_accum["env_agents"] += perf.get("agents", 0.0)
                    self._perf_accum["env_collisions"] += perf.get("collisions", 0.0)
                    self._perf_accum["env_targets"] += perf.get("targets", 0.0)
                    self._perf_accum["env_lidar"] += perf.get("lidar", 0.0)
                    self._perf_accum["env_explore"] += perf.get("explore", 0.0)
                    self._perf_accum["env_nav"] += perf.get("nav", 0.0)
                    self._perf_accum["env_rewards"] += perf.get("rewards", 0.0)
                    self._perf_accum["env_respawn"] += perf.get("respawn", 0.0)
                    self._perf_accum["env_obs"] += perf.get("obs", 0.0)
                    self._perf_accum["nav_astar"] += perf.get("nav_astar", 0.0)
                    self._perf_accum["nav_path_len"] += perf.get("nav_path_len", 0.0)
                    self._perf_accum["nav_frontier_envs"] += perf.get("nav_frontier_envs", 0.0)
                    self._perf_accum["nav_visible_frac"] += perf.get("nav_visible_frac", 0.0)
            
            # Accumulate metrics
            self._accumulate_metrics(infos, dones)
            
            # Flatten rewards: (B, N) -> (B*N,)
            rewards_flat = rewards.reshape(-1) * reward_scale
            
            # Expand dones: (B,) -> (B*N,) - all agents in env share done flag
            dones_expanded = np.repeat(dones, self.num_agents)
            done_matrix = np.repeat(dones[:, None], self.num_agents, axis=1)

            # Build per-agent truncated-vs-terminated masks.
            # bootstrap_mask: whether TD delta should include gamma * V(next).
            # trace_mask: whether GAE trace can continue to next time index.
            terminated_matrix = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
            done_reason_list: List[str] = ["running"] * self.num_envs
            terminal_obs_by_env: List[Optional[np.ndarray]] = [None] * self.num_envs
            terminal_target_pos_by_env: List[Optional[np.ndarray]] = [None] * self.num_envs
            terminal_target_vel_by_env: List[Optional[np.ndarray]] = [None] * self.num_envs
            for b, info in enumerate(infos):
                reason = str(info.get("done_reason", "running"))
                done_reason_list[b] = reason
                term_mask = info.get("terminated_mask")
                if isinstance(term_mask, list) and len(term_mask) == self.num_agents:
                    terminated_matrix[b] = np.asarray(term_mask, dtype=np.float32)
                elif isinstance(term_mask, np.ndarray) and term_mask.shape[0] == self.num_agents:
                    terminated_matrix[b] = term_mask.astype(np.float32)
                term_obs = info.get("terminal_observation")
                if term_obs is not None:
                    terminal_obs_by_env[b] = np.asarray(term_obs, dtype=np.float32)
                term_tgt_pos = info.get("terminal_target_pos")
                if term_tgt_pos is not None:
                    terminal_target_pos_by_env[b] = np.asarray(term_tgt_pos, dtype=np.float32)
                term_tgt_vel = info.get("terminal_target_vel")
                if term_tgt_vel is not None:
                    terminal_target_vel_by_env[b] = np.asarray(term_tgt_vel, dtype=np.float32)

            bootstrap_matrix = np.ones((self.num_envs, self.num_agents), dtype=np.float32)
            if np.any(done_matrix):
                done_env_ids = np.where(dones)[0]
                for b in done_env_ids:
                    reason = done_reason_list[b]
                    if reason == "collision":
                        # True terminal only for colliders; teammates are truncated.
                        bootstrap_matrix[b] = 1.0 - terminated_matrix[b]
                    elif reason == "timeout":
                        # Time-limit truncation keeps value bootstrap.
                        bootstrap_matrix[b] = 1.0
                    else:
                        # Capture/other terminal events: stop bootstrap for all agents.
                        bootstrap_matrix[b] = 0.0
            trace_matrix = (~done_matrix).astype(np.float32)
            bootstrap_masks_flat = bootstrap_matrix.reshape(-1)
            trace_masks_flat = trace_matrix.reshape(-1)

            # Optional next-value override from terminal observations for truncated entities.
            next_value_override_flat = np.zeros((self.num_entities,), dtype=np.float32)
            next_value_override_mask_flat = np.zeros((self.num_entities,), dtype=np.float32)
            need_override = done_matrix & (bootstrap_matrix > 0.5)
            if np.any(need_override):
                override_indices: List[int] = []
                override_obs: List[np.ndarray] = []
                override_target_pos: List[np.ndarray] = []
                override_target_vel: List[np.ndarray] = []
                for b in np.where(dones)[0]:
                    term_obs_env = terminal_obs_by_env[b]
                    if term_obs_env is None:
                        continue
                    for n in range(self.num_agents):
                        if not need_override[b, n]:
                            continue
                        entity_idx = b * self.num_agents + n
                        override_indices.append(entity_idx)
                        override_obs.append(term_obs_env[n])
                        if self._centralized_critic:
                            tgt_pos_b = terminal_target_pos_by_env[b]
                            tgt_vel_b = terminal_target_vel_by_env[b]
                            if tgt_pos_b is None or tgt_vel_b is None:
                                # Fallback to current env state if terminal target snapshot is unavailable.
                                tgt_pos_b = self.env.target_pos[b].astype(np.float32, copy=False)
                                tgt_vel_b = self.env.target_vel[b].astype(np.float32, copy=False)
                            override_target_pos.append(np.asarray(tgt_pos_b, dtype=np.float32))
                            override_target_vel.append(np.asarray(tgt_vel_b, dtype=np.float32))
                if override_indices:
                    obs_override_t = torch.from_numpy(np.stack(override_obs, axis=0)).to(self.device)
                    idx_override_t = torch.tensor(override_indices, dtype=torch.long, device=self.device)
                    rnn_override = next_rnn[:, :, idx_override_t, :].detach()
                    mask_override = torch.ones(len(override_indices), 1, device=self.device)
                    critic_override_t = None
                    if self._centralized_critic:
                        obs_override_bn = np.zeros((self.num_envs, self.num_agents, self.obs_dim), dtype=np.float32)
                        target_pos_override_bn = np.asarray(self.env.target_pos, dtype=np.float32).copy()
                        target_vel_override_bn = np.asarray(self.env.target_vel, dtype=np.float32).copy()
                        for b in np.where(dones)[0]:
                            term_obs_env = terminal_obs_by_env[b]
                            if term_obs_env is not None and term_obs_env.shape == (self.num_agents, self.obs_dim):
                                obs_override_bn[b] = term_obs_env
                        for entity_idx, tgt_pos_row, tgt_vel_row in zip(override_indices, override_target_pos, override_target_vel):
                            b = int(entity_idx // self.num_agents)
                            target_pos_override_bn[b] = tgt_pos_row
                            target_vel_override_bn[b] = tgt_vel_row
                        obs_override_bn_t = torch.from_numpy(obs_override_bn).to(self.device).view(self.num_entities, self.obs_dim)
                        critic_full_t = self._build_critic_obs_from_batch_and_target(
                            obs_override_bn_t,
                            target_pos_override_bn,
                            target_vel_override_bn,
                        )
                        critic_override_t = critic_full_t.index_select(0, idx_override_t)
                    with torch.no_grad():
                        _, _, v_override, _ = self.policy.model(
                            obs_override_t,
                            rnn_override,
                            mask_override,
                            critic_obs=critic_override_t,
                        )
                    v_override_np = v_override.squeeze(-1).detach().cpu().numpy().astype(np.float32)
                    next_value_override_flat[np.asarray(override_indices, dtype=np.int64)] = v_override_np
                    next_value_override_mask_flat[np.asarray(override_indices, dtype=np.int64)] = 1.0
            
            # Store in buffer
            buf_start = time.perf_counter() if self._perf_debug else 0.0
            self.buffer.obs[t] = obs_policy
            if self.buffer.critic_obs is not None:
                self.buffer.critic_obs[t] = critic_obs_t.detach()
            self.buffer.actions[t] = actions.detach()
            self.buffer.log_probs[t] = logp.detach()
            self.buffer.values[t] = values.squeeze(-1).detach()
            self.buffer.rewards[t] = torch.from_numpy(rewards_flat.astype(np.float32)).to(self.device)
            self.buffer.dones[t] = torch.from_numpy(dones_expanded.astype(np.float32)).to(self.device)
            self.buffer.bootstrap_masks[t] = torch.from_numpy(bootstrap_masks_flat).to(self.device)
            self.buffer.trace_masks[t] = torch.from_numpy(trace_masks_flat).to(self.device)
            self.buffer.next_value_overrides[t] = torch.from_numpy(next_value_override_flat).to(self.device)
            self.buffer.next_value_override_masks[t] = torch.from_numpy(next_value_override_mask_flat).to(self.device)
            self.buffer.masks[t] = masks.squeeze(-1)
            self.buffer.rnn_states[t] = rnn_state
            
            # Prepare next step
            obs = self._to_flat_tensor(next_obs)
            done_tensor = torch.from_numpy(dones_expanded.astype(np.float32)).to(self.device).unsqueeze(-1)
            masks = 1.0 - done_tensor
            
            # Reset RNN for done agents
            masks_expanded = masks.transpose(0, 1).unsqueeze(-1)
            next_rnn = next_rnn * masks_expanded
            rnn_state = next_rnn.detach()
            if self._perf_debug:
                self._perf_accum["buffer"] += time.perf_counter() - buf_start
        
        self.buffer.rnn_states[rollout_len] = rnn_state
        self.rnn_state = rnn_state
        self._last_action_std = rollout_std_sum / max(float(rollout_std_count), 1.0)
        return obs, masks

    def _resolve_std_bounds(self, stage: Optional[CurriculumStageConfig]) -> Tuple[float, float]:
        """Resolve stage std bounds with safe defaults."""
        min_std = 0.1
        max_std = 0.5
        if stage is not None:
            min_std = float(getattr(stage, "min_std", min_std))
            max_std = float(getattr(stage, "max_std", max_std))
        min_std = max(min_std, 1e-8)
        max_std = max(max_std, min_std)
        return min_std, max_std

    def _reset_entropy_for_stage(self, stage: Optional[CurriculumStageConfig]) -> None:
        if stage is None:
            return
        self._entropy_coef_init = float(stage.entropy_coef)
        self._entropy_coef_min = float(stage.entropy_min)
        self._entropy_decay_threshold = float(stage.entropy_decay_success_rate)
        self._entropy_coef = self._entropy_coef_init
        self.policy.set_skip_entropy_mc(self._entropy_coef_init <= 0.0)
        # In near-converged stages, detach entropy gradient to prevent entropy farming.
        self.policy.set_entropy_gradient_detach(0.0 < self._entropy_coef_init <= 1e-3)
        min_std, max_std = self._resolve_std_bounds(stage)
        self.policy.configure_action_std(min_std=min_std, max_std=max_std)
        self._reset_performance_windows(reset_collision_metrics=True)
        self._substage_cooldown = self._substage_cooldown_iters
        self._init_stage_view_speed_schedule(stage)
        self._episode_collisions[:, :] = False

    def _reset_performance_windows(self, reset_collision_metrics: bool = False) -> None:
        self._recent_clean_wins.clear()
        self._recent_capture_raw_slow.clear()
        self._recent_capture_raw_fast.clear()
        self._recent_clean_capture_slow.clear()
        self._recent_clean_capture_fast.clear()
        self._recent_encircle_capture_slow.clear()
        self._recent_encircle_capture_fast.clear()
        self._last_clean_win_rate = 0.0
        self._last_capture_rate_slow = 0.0
        self._last_capture_rate_fast = 0.0
        self._last_clean_capture_rate_slow = 0.0
        self._last_clean_capture_rate_fast = 0.0
        self._last_encircle_capture_rate_slow = 0.0
        self._last_encircle_capture_rate_fast = 0.0
        self._prev_clean_win_rate = 0.0
        self._best_clean_win_rate = 0.0
        if reset_collision_metrics:
            self._last_collision_rate = 0.0
            self._collision_rate_ema = None
            self._rollout_episode_collisions.clear()

    def _init_stage_view_speed_schedule(self, stage: CurriculumStageConfig) -> None:
        schedule = getattr(stage, "view_speed_schedule", None)
        if not schedule:
            self._stage_view_speed_schedule = None
            self._stage_view_speed_idx = 0
            return
        cleaned: List[Dict[str, float]] = []
        for item in schedule:
            if not isinstance(item, dict):
                continue
            view_radius = item.get("view_radius_m")
            target_speed = item.get("target_speed")
            if view_radius is None or target_speed is None:
                continue
            cleaned.append({
                "view_radius_m": float(view_radius),
                "target_speed": float(target_speed),
            })
        if not cleaned:
            self._stage_view_speed_schedule = None
            self._stage_view_speed_idx = 0
            return
        self._stage_view_speed_schedule = cleaned
        self._stage_view_speed_idx = 0
        self._apply_stage_view_speed()

    def _apply_stage_view_speed(self) -> None:
        if not self._stage_view_speed_schedule:
            return
        stage_idx = self.curriculum.current_index
        if stage_idx < 0 or stage_idx >= len(self.cfg.curriculum.stages):
            return
        item = self._stage_view_speed_schedule[self._stage_view_speed_idx]
        view_radius = float(item["view_radius_m"])
        target_speed = float(item["target_speed"])
        if self._active_stage is not None:
            self._active_stage.view_radius_m = view_radius
            self._active_stage.target_speed = target_speed
        self.env.set_stage_params(stage_idx, target_speed=target_speed, view_radius=view_radius)

    def _compute_win_rate(self, results: deque, window: int) -> float:
        if window <= 0 or len(results) < window:
            return 0.0
        return float(np.mean(results))

    def _compute_survival_rate(self, results: deque, window: int) -> float:
        if window <= 0 or len(results) < window:
            return 0.0
        return float(np.mean(results))

    def _current_substage_min_env_steps(self) -> int:
        required = int(self._substage_min_env_steps)
        if not self._stage_view_speed_schedule:
            return required
        if self._active_stage is None:
            return required
        stage_budget = max(0, int(getattr(self._active_stage, "timesteps", 0)))
        if stage_budget <= 0:
            return required
        num_substages = max(1, len(self._stage_view_speed_schedule))
        ratio_steps = int(np.ceil((stage_budget / float(num_substages)) * self._substage_min_stage_ratio))
        return max(required, ratio_steps)

    def _update_entropy_coef(self, global_step: int) -> None:
        clean_win_rate = self._compute_win_rate(self._recent_clean_wins, self._clean_win_window)
        capture_rate_slow = self._compute_win_rate(self._recent_capture_raw_slow, self._success_window_slow)
        capture_rate_fast = self._compute_win_rate(self._recent_capture_raw_fast, self._success_window_fast)
        clean_capture_rate_slow = self._compute_win_rate(self._recent_clean_capture_slow, self._success_window_slow)
        clean_capture_rate_fast = self._compute_win_rate(self._recent_clean_capture_fast, self._success_window_fast)
        encircle_capture_rate_slow = self._compute_win_rate(self._recent_encircle_capture_slow, self._success_window_slow)
        encircle_capture_rate_fast = self._compute_win_rate(self._recent_encircle_capture_fast, self._success_window_fast)
        collision_rate = self._last_collision_rate
        if self._rollout_episode_collisions:
            batch_rate = float(np.mean(self._rollout_episode_collisions))
            alpha = float(getattr(self.cfg.training, "collision_rate_ema_alpha", 0.15))
            if self._collision_rate_ema is None:
                self._collision_rate_ema = batch_rate
            else:
                self._collision_rate_ema = alpha * batch_rate + (1.0 - alpha) * self._collision_rate_ema
            collision_rate = self._collision_rate_ema
        self._last_clean_win_rate = clean_win_rate
        self._last_capture_rate_slow = capture_rate_slow
        self._last_capture_rate_fast = capture_rate_fast
        self._last_clean_capture_rate_slow = clean_capture_rate_slow
        self._last_clean_capture_rate_fast = clean_capture_rate_fast
        self._last_encircle_capture_rate_slow = encircle_capture_rate_slow
        self._last_encircle_capture_rate_fast = encircle_capture_rate_fast
        self._last_collision_rate = collision_rate
        if self._substage_cooldown > 0:
            self._substage_cooldown -= 1

        # Stage sub-schedule control.
        can_change_level = self._substage_cooldown == 0
        enough_steps = (global_step - self._last_substage_advance_env_steps) >= self._current_substage_min_env_steps()
        collision_ok = collision_rate < self._substage_promotion_collision_max
        substage_pass_now = (
            can_change_level
            and enough_steps
            and clean_win_rate >= self._substage_promotion_win_rate_slow
            and collision_ok
        )
        if substage_pass_now:
            self._substage_promotion_pass_streak += 1
        else:
            self._substage_promotion_pass_streak = 0
        if (
            self._substage_promotion_pass_streak >= self._substage_promotion_required_streak
            and
            can_change_level
            and self._stage_view_speed_schedule
            and self._stage_view_speed_idx < len(self._stage_view_speed_schedule) - 1
            and enough_steps
        ):
            self._stage_view_speed_idx += 1
            self._apply_stage_view_speed()
            if self._entropy_coef < self._entropy_coef_init:
                self._entropy_coef = self._entropy_coef_init
            self._substage_cooldown = self._substage_cooldown_iters
            self._last_substage_advance_env_steps = int(global_step)
            self._substage_promotion_pass_streak = 0
            self._reset_performance_windows(reset_collision_metrics=False)

        # Annealing control
        threshold = self._entropy_decay_threshold
        trend_up = clean_win_rate > self._prev_clean_win_rate
        collapse = False
        base_entropy = self._entropy_coef
        if clean_win_rate > self._anneal_win_rate_fast and trend_up and not collapse and threshold < 0.95:
            lower = max(0.0, min(threshold, 0.95))
            upper = self._decay_upper
            denom = max(upper - lower, 1e-6)
            progress = (clean_win_rate - lower) / denom
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            base_entropy = self._entropy_coef_min + (self._entropy_coef_init - self._entropy_coef_min) * cosine_decay

        self._entropy_coef = max(self._entropy_coef_min, base_entropy)
        # Stage5 entropy gate:
        # Keep a small exploration floor before mastery, and only force entropy to zero
        # when clean-win (150-episode window) is >= 90%.
        stage_name = str(getattr(self._active_stage, "name", "")).lower()
        if stage_name.startswith("stage5"):
            stage5_mastery_threshold = 0.90
            if clean_win_rate >= stage5_mastery_threshold:
                self._entropy_coef = 0.0
            else:
                # Ensure stage5 retains minimal exploration when not yet mastered.
                stage5_floor = 1e-3
                if self._entropy_coef_init > 0.0:
                    stage5_floor = min(stage5_floor, self._entropy_coef_init)
                self._entropy_coef = max(self._entropy_coef, stage5_floor)
        self.policy.set_skip_entropy_mc(self._entropy_coef <= 0.0)
        self.policy.set_entropy_gradient_detach(0.0 < self._entropy_coef <= 1e-3)

        self._prev_clean_win_rate = clean_win_rate
        self._best_clean_win_rate = max(self._best_clean_win_rate, clean_win_rate)

    def _update_aux_weights_for_stage(self, stage_idx: int) -> None:
        """Enable stage-dependent auxiliary heads."""
        if stage_idx >= 1:
            self._aux_ttc_weight = self._aux_ttc_weight_base
            self._aux_collision_prob_weight = self._aux_collision_prob_weight_base
        else:
            self._aux_ttc_weight = 0.0
            self._aux_collision_prob_weight = 0.0
        if stage_idx >= self._aux_stage_start_index:
            self._aux_target_pred_weight = self._aux_target_pred_weight_base
        else:
            self._aux_target_pred_weight = 0.0

    def _accumulate_metrics(self, infos: List[Dict], dones: np.ndarray) -> None:
        """Accumulate reward breakdown and collision counts."""
        reward_keys = (
            "step_cost",
            "progress_gain",
            "exploration_gain",
            "proximity_cost",
            "collision_cost",
            "direction_gain",
            "control_cost",
            "capture_gain",
            "capture_quality_gain",
            "reward_total",
        )
        for i, info in enumerate(infos):
            # Reward breakdown
            rb_row = info.get("reward_breakdown_row")
            if isinstance(rb_row, np.ndarray) and rb_row.shape[0] >= 10:
                for idx, key in enumerate(reward_keys):
                    self._reward_sums[key] = self._reward_sums.get(key, 0.0) + float(rb_row[idx])
                self._reward_count += 1
            else:
                breakdowns = info.get("reward_breakdown", [])
                for item in breakdowns:
                    for key, value in item.items():
                        self._reward_sums[key] = self._reward_sums.get(key, 0.0) + float(value)
                    self._reward_count += 1

            self._team_target_min_dist_3d_sum += float(info.get("team_target_min_dist_3d", 0.0))
            self._team_target_mean_abs_dz_sum += float(info.get("team_target_mean_abs_dz", 0.0))
            self._far_idle_ratio_sum += float(info.get("far_idle_ratio", 0.0))
            self._capture_contribution_sum += float(info.get("capture_contribution", 0.0))
            self._dist_to_target_p50_sum += float(info.get("dist_to_target_p50", 0.0))
            self._dist_to_target_p90_sum += float(info.get("dist_to_target_p90", 0.0))
            self._step_metric_count += 1
            
            # Collision count
            if info.get("collision", False):
                self._collision_terminations += 1

            collisions = info.get("collisions")
            if isinstance(collisions, list) and len(collisions) == self.num_agents:
                self._episode_collisions[i] |= np.array(collisions, dtype=np.bool_)
            elif isinstance(collisions, np.ndarray) and collisions.shape[0] == self.num_agents:
                self._episode_collisions[i] |= collisions.astype(np.bool_, copy=False)

            if dones[i]:
                success = bool(info.get("captured", False))
                captured_raw = bool(info.get("captured_raw", success))
                clean_capture = bool(info.get("clean_capture", (captured_raw and not info.get("collision", False))))
                encircle_capture = captured_raw and clean_capture
                self._recent_clean_wins.append(1.0 if clean_capture else 0.0)
                self._recent_capture_raw_slow.append(1.0 if captured_raw else 0.0)
                self._recent_capture_raw_fast.append(1.0 if captured_raw else 0.0)
                self._recent_clean_capture_slow.append(1.0 if clean_capture else 0.0)
                self._recent_clean_capture_fast.append(1.0 if clean_capture else 0.0)
                self._recent_encircle_capture_slow.append(1.0 if encircle_capture else 0.0)
                self._recent_encircle_capture_fast.append(1.0 if encircle_capture else 0.0)
                collided = self._episode_collisions[i]
                collided_any = bool(np.any(collided))
                self._rollout_episode_collisions.append(1.0 if collided_any else 0.0)
                self._episode_collisions[i] = False

    def _compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        gamma = self.cfg.training.gamma
        gae_lambda = self.cfg.training.gae_lambda
        T = self.cfg.training.rollout_length
        
        advantages = torch.zeros_like(self.buffer.rewards)
        returns = torch.zeros_like(self.buffer.rewards)
        gae = torch.zeros(self.num_entities, device=self.device)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
            else:
                next_v = self.buffer.values[t + 1]
            override_mask = self.buffer.next_value_override_masks[t] > 0.5
            if torch.any(override_mask):
                next_v = torch.where(override_mask, self.buffer.next_value_overrides[t], next_v)

            bootstrap_mask = self.buffer.bootstrap_masks[t]
            trace_mask = self.buffer.trace_masks[t]
            delta = self.buffer.rewards[t] + gamma * next_v * bootstrap_mask - self.buffer.values[t]
            gae = delta + gamma * gae_lambda * trace_mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.buffer.values[t]
        
        return advantages, returns

    def _update(self) -> Dict[str, float]:
        """PPO update with truncated BPTT."""
        # Bootstrap value
        with torch.no_grad():
            if self._bootstrap_obs is not None and self._bootstrap_masks is not None:
                last_obs = self._bootstrap_obs
                last_critic_obs = self._bootstrap_critic_obs
                last_rnn = self.rnn_state.detach()
                last_mask = self._bootstrap_masks
            else:
                # Fallback for robustness in non-standard call flows.
                last_obs = self.buffer.obs[-1]
                last_critic_obs = self.buffer.critic_obs[-1] if self.buffer.critic_obs is not None else None
                last_rnn = self.buffer.rnn_states[-1]
                last_mask = torch.ones(self.num_entities, 1, device=self.device)
            _, _, next_value, _ = self.policy.model(last_obs, last_rnn, last_mask, critic_obs=last_critic_obs)
            next_value = next_value.squeeze(-1)
        
        # Compute advantages
        advantages, returns = self._compute_gae(next_value)
        self.policy.update_popart(returns)
        returns_norm = self.policy.normalize_values(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten for update
        T = self.cfg.training.rollout_length
        b_obs = self.buffer.obs.reshape(-1, self.obs_dim)
        b_actions = self.buffer.actions.reshape(-1, self.action_dim)
        b_log_probs = self.buffer.log_probs.reshape(-1)
        b_returns = returns_norm.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_log_probs_2d = self.buffer.log_probs
        b_returns_2d = returns_norm
        b_adv_2d = advantages
        b_masks = self.buffer.masks.reshape(T, self.num_entities, 1)
        b_critic_obs = self.buffer.critic_obs.reshape(T, self.num_entities, -1) if self.buffer.critic_obs is not None else None
        aux_min_w = self._aux_min_dist_weight
        aux_ttc_w = self._aux_ttc_weight
        aux_col_w = self._aux_collision_prob_weight
        aux_pred_w = self._aux_target_pred_weight
        use_aux = (aux_min_w > 0.0) or (aux_ttc_w > 0.0) or (aux_col_w > 0.0)
        use_pred_aux = aux_pred_w > 0.0
        min_dist_targets = None
        ttc_targets = None
        collision_prob_targets = None
        aux_min_losses = []
        aux_ttc_losses = []
        aux_col_losses = []
        aux_pred_losses = []
        if use_aux:
            min_dist_targets, ttc_targets, collision_prob_targets = self._build_aux_targets(b_obs)
            min_dist_targets_2d = min_dist_targets.reshape(T, self.num_entities)
            ttc_targets_2d = ttc_targets.reshape(T, self.num_entities)
            collision_prob_targets_2d = collision_prob_targets.reshape(T, self.num_entities)
        else:
            min_dist_targets_2d = None
            ttc_targets_2d = None
            collision_prob_targets_2d = None
        
        # Truncated BPTT
        trunc = max(1, self.cfg.training.bptt_trunc)
        num_update_epochs = max(1, int(getattr(self.cfg.training, "update_epochs", 1)))
        total_rollout_samples = T * self.num_entities
        cfg_batch_size = max(1, int(getattr(self.cfg.training, "batch_size", total_rollout_samples)))
        cfg_minibatch_size = max(1, int(getattr(self.cfg.training, "minibatch_size", total_rollout_samples)))
        effective_batch_size = min(total_rollout_samples, cfg_batch_size)
        effective_minibatch_size = min(effective_batch_size, cfg_minibatch_size)
        # Keep temporal structure intact by minibatching over entities.
        entity_mb_size = max(1, min(self.num_entities, effective_minibatch_size // trunc))
        policy_losses, value_losses, entropies = [], [], []
        approx_kls = []
        
        for _epoch in range(num_update_epochs):
            chunk_starts = list(range(0, T, trunc))
            np.random.shuffle(chunk_starts)
            entity_perm_np = np.random.permutation(self.num_entities).astype(np.int64)
            entity_groups_np = [entity_perm_np[i : i + entity_mb_size] for i in range(0, self.num_entities, entity_mb_size)]
            np.random.shuffle(entity_groups_np)
            samples_used_this_epoch = 0
            stop_epoch = False
            for start in chunk_starts:
                end = min(T, start + trunc)
                for entity_idx_np in entity_groups_np:
                    if entity_idx_np.size == 0:
                        continue
                    chunk_samples = (end - start) * int(entity_idx_np.size)
                    if samples_used_this_epoch >= effective_batch_size:
                        stop_epoch = True
                        break
                    samples_used_this_epoch += chunk_samples
                    entity_idx = torch.from_numpy(entity_idx_np).to(self.device, dtype=torch.long)
                    obs_seq = self.buffer.obs[start:end].index_select(1, entity_idx)
                    critic_obs_seq = (
                        b_critic_obs[start:end].index_select(1, entity_idx) if b_critic_obs is not None else None
                    )
                    act_seq = self.buffer.actions[start:end].index_select(1, entity_idx)
                    mask_seq = b_masks[start:end].index_select(1, entity_idx)
                    rnn = self.buffer.rnn_states[start][:, :, entity_idx, :].detach()
                
                    logps, ents, vals = [], [], []
                    aux_min_preds = []
                    aux_ttc_preds = []
                    aux_col_preds = []
                    aux_target_rel_preds = []
                    for t_idx in range(obs_seq.shape[0]):
                        if use_aux and use_pred_aux:
                            logp_t, ent_t, val_t, rnn, aux_pred_t, pred_rel_t = self.policy.evaluate_actions(
                                obs_seq[t_idx],
                                rnn,
                                mask_seq[t_idx],
                                act_seq[t_idx],
                                return_aux=True,
                                return_pred_target=True,
                                critic_obs=(critic_obs_seq[t_idx] if critic_obs_seq is not None else None),
                            )
                            min_pred_t, ttc_pred_t, col_pred_t = aux_pred_t
                            aux_min_preds.append(min_pred_t)
                            aux_ttc_preds.append(ttc_pred_t)
                            aux_col_preds.append(col_pred_t)
                            aux_target_rel_preds.append(pred_rel_t)
                        elif use_aux:
                            logp_t, ent_t, val_t, rnn, aux_pred_t = self.policy.evaluate_actions(
                                obs_seq[t_idx],
                                rnn,
                                mask_seq[t_idx],
                                act_seq[t_idx],
                                return_aux=True,
                                critic_obs=(critic_obs_seq[t_idx] if critic_obs_seq is not None else None),
                            )
                            min_pred_t, ttc_pred_t, col_pred_t = aux_pred_t
                            aux_min_preds.append(min_pred_t)
                            aux_ttc_preds.append(ttc_pred_t)
                            aux_col_preds.append(col_pred_t)
                        elif use_pred_aux:
                            logp_t, ent_t, val_t, rnn, pred_rel_t = self.policy.evaluate_actions(
                                obs_seq[t_idx],
                                rnn,
                                mask_seq[t_idx],
                                act_seq[t_idx],
                                return_pred_target=True,
                                critic_obs=(critic_obs_seq[t_idx] if critic_obs_seq is not None else None),
                            )
                            aux_target_rel_preds.append(pred_rel_t)
                        else:
                            logp_t, ent_t, val_t, rnn = self.policy.evaluate_actions(
                                obs_seq[t_idx],
                                rnn,
                                mask_seq[t_idx],
                                act_seq[t_idx],
                                critic_obs=(critic_obs_seq[t_idx] if critic_obs_seq is not None else None),
                            )
                        logps.append(logp_t)
                        ents.append(ent_t)
                        vals.append(val_t)
                    
                    seq_logp = torch.stack(logps, dim=0).reshape(-1)
                    seq_entropy = torch.stack(ents, dim=0).reshape(-1)
                    seq_value = torch.stack(vals, dim=0).reshape(-1)
                    seq_value_norm = self.policy.normalize_values(seq_value)
                    old_logp_seq = b_log_probs_2d[start:end].index_select(1, entity_idx).reshape(-1)
                    adv_seq = b_adv_2d[start:end].index_select(1, entity_idx).reshape(-1)
                    ret_seq = b_returns_2d[start:end].index_select(1, entity_idx).reshape(-1)
                    ratio = torch.exp(seq_logp - old_logp_seq)
                    log_ratio = seq_logp - old_logp_seq
                    surr1 = ratio * adv_seq
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.cfg.training.clip_range, 1.0 + self.cfg.training.clip_range
                    ) * adv_seq
                    # Standard PPO monitoring metric: approximate KL(new || old) on sampled actions.
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                    approx_kls.append(float(approx_kl.detach().item()))
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(seq_value_norm, ret_seq)
                    entropy_loss = -seq_entropy.mean()
                    entropy_val = -entropy_loss.item()
                    
                    loss = actor_loss + self.cfg.training.value_coef * value_loss + self._entropy_coef * entropy_loss
                    if use_aux and min_dist_targets_2d is not None:
                        if aux_min_w > 0.0:
                            seq_min = torch.stack(aux_min_preds, dim=0).reshape(-1)
                            min_target = min_dist_targets_2d[start:end].index_select(1, entity_idx).reshape(-1)
                            min_loss = F.mse_loss(seq_min, min_target)
                            loss = loss + aux_min_w * min_loss
                            aux_min_losses.append(min_loss.item())
                        if aux_ttc_w > 0.0:
                            seq_ttc = torch.stack(aux_ttc_preds, dim=0).reshape(-1)
                            ttc_target = ttc_targets_2d[start:end].index_select(1, entity_idx).reshape(-1)
                            ttc_loss = F.mse_loss(seq_ttc, ttc_target)
                            loss = loss + aux_ttc_w * ttc_loss
                            aux_ttc_losses.append(ttc_loss.item())
                        if aux_col_w > 0.0:
                            seq_col = torch.stack(aux_col_preds, dim=0).reshape(-1)
                            col_target = collision_prob_targets_2d[start:end].index_select(1, entity_idx).reshape(-1)
                            col_loss = F.binary_cross_entropy(seq_col, col_target)
                            loss = loss + aux_col_w * col_loss
                            aux_col_losses.append(col_loss.item())
                    if use_pred_aux and aux_target_rel_preds:
                        pred_seq = torch.stack(aux_target_rel_preds, dim=0)  # (L, B_sub, 3)
                        next_obs_seq = []
                        for t_idx in range(obs_seq.shape[0]):
                            g_t = start + t_idx
                            if g_t + 1 < T:
                                next_obs_seq.append(self.buffer.obs[g_t + 1].index_select(0, entity_idx))
                            else:
                                if self._bootstrap_obs is not None:
                                    next_obs_seq.append(self._bootstrap_obs.index_select(0, entity_idx))
                                else:
                                    next_obs_seq.append(self.buffer.obs[g_t].index_select(0, entity_idx))
                        next_obs_seq_t = torch.stack(next_obs_seq, dim=0)
                        base_start = (self._frame_stack - 1) * self._obs_base_dim
                        rel_start = base_start + self._obs_target_rel_start
                        mode_idx = base_start + self._obs_target_mode_idx
                        target_rel_next = next_obs_seq_t[:, :, rel_start:rel_start + 3].reshape(-1, 3)
                        target_mode_now = obs_seq[:, :, mode_idx].reshape(-1)
                        stage_mask_scalar = 1.0 if self.curriculum.current_index >= self._aux_stage_start_index else 0.0
                        train_mask = (target_mode_now > 0.5).float() * stage_mask_scalar
                        pred_flat = pred_seq.reshape(-1, 3)
                        per_sample = F.smooth_l1_loss(pred_flat, target_rel_next, reduction="none").mean(dim=-1)
                        denom = torch.clamp(train_mask.sum(), min=1.0)
                        pred_loss = torch.sum(per_sample * train_mask) / denom
                        if stage_mask_scalar > 0.0:
                            loss = loss + aux_pred_w * pred_loss
                        aux_pred_losses.append(pred_loss.item())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.training.max_grad_norm)
                    self.optimizer.step()
                    
                    policy_losses.append(actor_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy_val)
                if stop_epoch:
                    break
            if stop_epoch:
                pass
        
        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "ppo_update_epochs_effective": float(num_update_epochs),
            "ppo_batch_size_effective": float(effective_batch_size),
            "ppo_minibatch_size_effective": float(effective_minibatch_size),
            "ppo_entity_minibatch_size": float(entity_mb_size),
        }
        if aux_min_w > 0.0:
            metrics["aux_min_dist_loss"] = float(np.mean(aux_min_losses)) if aux_min_losses else 0.0
        if aux_ttc_w > 0.0:
            metrics["aux_ttc_loss"] = float(np.mean(aux_ttc_losses)) if aux_ttc_losses else 0.0
        if aux_col_w > 0.0:
            metrics["aux_collision_prob_loss"] = float(np.mean(aux_col_losses)) if aux_col_losses else 0.0
        if aux_pred_w > 0.0:
            metrics["aux_target_pred_loss"] = float(np.mean(aux_pred_losses)) if aux_pred_losses else 0.0
        return metrics

    def _on_stage_advance(self, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle curriculum stage transition."""
        old_stage = self._active_stage
        self._active_stage = self.curriculum.current_stage()
        self._update_aux_weights_for_stage(self.curriculum.current_index)
        
        print(f"\n[Curriculum] Stage transition: {old_stage.name if old_stage else 'none'} -> "
              f"{self._active_stage.name if self._active_stage else 'none'} at step {global_step:,}")
        
        self._reset_entropy_for_stage(self._active_stage)
        self._last_substage_advance_env_steps = int(global_step)
        self._stage_entry_env_steps = int(global_step)
        self._substage_promotion_pass_streak = 0
        self._stage5_early_stop_streak = 0
        
        # Reset environment with new stage
        stage_idx = self.curriculum.current_index
        obs, _ = self.env.reset(stage_index=stage_idx)
        obs = self._to_flat_tensor(obs)
        masks = torch.ones(self.num_entities, 1, device=self.device)
        
        # Reset RNN
        self.rnn_state = self.policy.init_hidden(self.num_entities)
        
        # Save checkpoint
        self._save_checkpoint(global_step)
        
        return obs, masks

    def _log_metrics(self, metrics: Dict) -> None:
        """Write metrics to JSONL file."""
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Clear accumulators
        self._reward_sums.clear()
        self._reward_count = 0

    def _save_checkpoint(self, global_step: int) -> None:
        """Save policy checkpoint."""
        ckpt_dir = self.log_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": global_step,
            "curriculum_stage": self._active_stage.name if self._active_stage else None,
            "curriculum_index": self.curriculum.current_index,
        }
        torch.save(payload, ckpt_dir / f"policy_{global_step}.pt")
        torch.save(self.policy.state_dict(), self.log_root / "policy.pt")
        print(f"[Checkpoint] Saved at step {global_step:,}")


__all__ = ["MAPPOTrainerVec", "CurriculumScheduler"]
