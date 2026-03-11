"""Policy/critic wrapper for 3D MAPPO using RMAPPOActorCritic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.policy.r_mappo import RMAPPOActorCritic


@dataclass
class ActionNoiseConfig:
    """Configuration for action noise decay during training."""
    initial_std: float = 0.6
    min_std: float = 0.2
    decay_rate: float = 1e-6  # std reduction per step


class MAPPOPolicy3D(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        action_noise: Optional[ActionNoiseConfig] = None,
        hidden_dim: int = 512,
        action_bounds: Optional[Dict[str, float]] = None,
        min_action_std: float = 0.1,
        max_action_std: float = 0.5,
        centralized_critic: bool = False,
        critic_obs_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = RMAPPOActorCritic(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            action_noise=action_noise,
            min_action_std=min_action_std,
            max_action_std=max_action_std,
            centralized_critic=centralized_critic,
            critic_obs_dim=critic_obs_dim,
        )
        self.device = device
        self.action_dim = action_dim
        # Per-dimension physical scaling read from the configured action bounds.
        bounds = action_bounds or {"vx": 8.0, "vy": 4.0, "vz": 3.0, "yaw_rate": 1.5}
        self._scale = torch.tensor(
            [
                float(bounds.get("vx", 8.0)),
                float(bounds.get("vy", 4.0)),
                float(bounds.get("vz", 3.0)),
                float(bounds.get("yaw_rate", 1.5)),
            ],
            dtype=torch.float32,
        ).to(device)
        self._log_scale_sum = torch.log(self._scale).sum()
        self._squash_eps = 1e-6
        
        # Action noise configuration
        self._action_noise = action_noise or ActionNoiseConfig()
        self._last_dist_std_mean = 0.0
        self._detach_entropy_grad = False
        self._skip_entropy_mc = False
        
        self.to(device)

    def _tanh_normal_log_prob(
        self,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        """Log-prob under scaled Tanh-Normal policy: action = tanh(u) * scale."""
        base_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        # Stable form:
        # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
        jacobian = (
            2.0 * (math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))
        ).sum(dim=-1)
        return base_log_prob - jacobian - self._log_scale_sum

    def _unsquash_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse map from scaled action to pre-tanh action."""
        squashed = torch.clamp(
            action / self._scale,
            -1.0 + self._squash_eps,
            1.0 - self._squash_eps,
        )
        pre_tanh = torch.atanh(squashed)
        return pre_tanh, squashed

    def init_hidden(self, num_envs: int) -> torch.Tensor:
        hidden_dim = int(getattr(self.model, "hidden_dim", 256))
        # Split GRU: [actor_state, critic_state] -> shape (2, 1, B, H)
        return torch.zeros(2, 1, num_envs, hidden_dim, device=self.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward-compatible mapping for legacy single-GRU checkpoints.
        if isinstance(state_dict, dict):
            remapped = dict(state_dict)
            has_actor = any(k.endswith("actor_gru.weight_ih_l0") for k in remapped)
            has_critic = any(k.endswith("critic_gru.weight_ih_l0") for k in remapped)
            has_legacy_gru = any(k.endswith("gru.weight_ih_l0") for k in remapped)
            has_legacy_norm = any(k.endswith("post_gru_norm.weight") for k in remapped)
            if has_legacy_gru and not (has_actor and has_critic):
                for k, v in list(remapped.items()):
                    if k.startswith("gru."):
                        remapped[k.replace("gru.", "actor_gru.")] = v
                        remapped[k.replace("gru.", "critic_gru.")] = v
                    elif ".gru." in k:
                        remapped[k.replace(".gru.", ".actor_gru.")] = v
                        remapped[k.replace(".gru.", ".critic_gru.")] = v
                # Remove legacy keys to avoid unexpected key warnings
                for k in list(remapped.keys()):
                    if k.startswith("gru.") or (".gru." in k and ".actor_gru." not in k and ".critic_gru." not in k):
                        remapped.pop(k, None)
            if has_legacy_norm:
                for k, v in list(remapped.items()):
                    if k.startswith("post_gru_norm."):
                        remapped[k.replace("post_gru_norm.", "actor_post_gru_norm.")] = v
                        remapped[k.replace("post_gru_norm.", "critic_post_gru_norm.")] = v
                    elif ".post_gru_norm." in k:
                        remapped[k.replace(".post_gru_norm.", ".actor_post_gru_norm.")] = v
                        remapped[k.replace(".post_gru_norm.", ".critic_post_gru_norm.")] = v
                for k in list(remapped.keys()):
                    if k.startswith("post_gru_norm.") or (
                        ".post_gru_norm." in k
                        and ".actor_post_gru_norm." not in k
                        and ".critic_post_gru_norm." not in k
                    ):
                        remapped.pop(k, None)
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def act(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        masks: torch.Tensor,
        return_aux: bool = False,
        return_pred_target: bool = False,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_pred = None
        target_rel_pred = None
        if return_aux or return_pred_target:
            (
                mean,
                log_std,
                value,
                next_state,
                min_dist_pred,
                ttc_pred,
                collision_prob_pred,
                target_rel_pred,
            ) = self.model.forward_with_aux(
                obs, rnn_state, masks, critic_obs=critic_obs
            )
            aux_pred = (min_dist_pred, ttc_pred, collision_prob_pred)
        else:
            mean, log_std, value, next_state = self.model(obs, rnn_state, masks, critic_obs=critic_obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        self._last_dist_std_mean = float(dist.stddev.mean().item())
        pre_tanh_action = dist.rsample()
      
        # print(f"scale:{self._scale}")
        squashed = torch.tanh(pre_tanh_action)
        action = squashed * self._scale
        log_prob = self._tanh_normal_log_prob(dist, pre_tanh_action)
        if return_aux and return_pred_target:
            return action, log_prob, value, next_state, aux_pred, target_rel_pred
        if return_aux:
            return action, log_prob, value, next_state, aux_pred
        if return_pred_target:
            return action, log_prob, value, next_state, target_rel_pred
        return action, log_prob, value, next_state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        return_aux: bool = False,
        return_pred_target: bool = False,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_pred = None
        target_rel_pred = None
        if return_aux or return_pred_target:
            (
                mean,
                log_std,
                value,
                next_state,
                min_dist_pred,
                ttc_pred,
                collision_prob_pred,
                target_rel_pred,
            ) = self.model.forward_with_aux(
                obs, rnn_state, masks, critic_obs=critic_obs
            )
            aux_pred = (min_dist_pred, ttc_pred, collision_prob_pred)
        else:
            mean, log_std, value, next_state = self.model(obs, rnn_state, masks, critic_obs=critic_obs)
        std = log_std.exp()
        dist = Normal(mean, std)
      
        pre_actions, _ = self._unsquash_action(actions)
        log_prob = self._tanh_normal_log_prob(dist, pre_actions)
        if self._skip_entropy_mc:
            # If entropy coefficient is zero, skip MC entropy sampling to save compute.
            entropy = torch.zeros_like(log_prob)
        else:
            # Tanh-Normal entropy is estimated via one-sample Monte Carlo.
            ent_pre = dist.rsample()
            entropy = -self._tanh_normal_log_prob(dist, ent_pre)
            if self._detach_entropy_grad:
                # Optional safeguard: disable entropy-gradient incentives in late stages.
                entropy = entropy.detach()
        if return_aux and return_pred_target:
            return log_prob, entropy, value, next_state, aux_pred, target_rel_pred
        if return_aux:
            return log_prob, entropy, value, next_state, aux_pred
        if return_pred_target:
            return log_prob, entropy, value, next_state, target_rel_pred
        return log_prob, entropy, value, next_state

    def update_action_std(self, elapsed_steps: int) -> None:
        """No-op: std is fully learnable and state-dependent."""
        return

    def set_entropy_gradient_detach(self, enabled: bool) -> None:
        """Enable/disable entropy-gradient detachment for MC entropy."""
        self._detach_entropy_grad = bool(enabled)

    def set_skip_entropy_mc(self, enabled: bool) -> None:
        """Enable/disable skipping Monte Carlo entropy computation."""
        self._skip_entropy_mc = bool(enabled)

    def set_std_bounds(self, min_std: float, max_std: float, reset_to_max: bool = False) -> None:
        """Update policy std clamp bounds in standard-deviation space."""
        min_std = max(float(min_std), 1e-8)
        max_std = max(float(max_std), min_std)
        log_std_min = math.log(min_std)
        log_std_max = math.log(max_std)
        self.model.log_std_min = log_std_min
        self.model.log_std_max = log_std_max
        with torch.no_grad():
            if reset_to_max:
                self.model.std_head.bias.fill_(log_std_max)
            self.model.std_head.bias.clamp_(min=log_std_min, max=log_std_max)

    def configure_action_std(self, min_std: float, max_std: float) -> None:
        """Apply stage std bounds and reset init std to the configured max."""
        min_std = max(float(min_std), 1e-8)
        max_std = max(float(max_std), min_std)
        self.set_std_bounds(min_std, max_std, reset_to_max=False)
        decay_rate = float(getattr(self._action_noise, "decay_rate", 1e-6))
        self.set_action_noise(
            ActionNoiseConfig(initial_std=max_std, min_std=min_std, decay_rate=decay_rate)
        )

    def set_action_noise(self, config: ActionNoiseConfig) -> None:
        """Switch to a new action-noise schedule and reset std to its initial value."""
        self._action_noise = config
        init_std = max(config.initial_std, config.min_std, 1e-8)
        log_std = math.log(init_std)
        log_std_min = float(getattr(self.model, "log_std_min", -3.0))
        log_std_max = float(getattr(self.model, "log_std_max", 0.0))
        log_std = min(max(log_std, log_std_min), log_std_max)
        with torch.no_grad():
            self.model.std_head.bias.fill_(float(log_std))

    def reset_noise(self, std: float) -> None:
        """Deprecated: std is state-dependent; no external reset."""
        return

    def set_max_std(self, max_std: float) -> None:
        """Set maximum allowable action std (clamps current std if needed)."""
        min_std = math.exp(float(getattr(self.model, "log_std_min", math.log(1e-8))))
        self.set_std_bounds(min_std=min_std, max_std=max_std, reset_to_max=False)

    def get_current_std(
        self,
        obs: Optional[torch.Tensor] = None,
        rnn_state: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> float:
        """Return action std from current policy distribution."""
        with torch.no_grad():
            if obs is not None and rnn_state is not None and masks is not None:
                mean, log_std, _, _ = self.model(obs, rnn_state, masks)
                dist = Normal(mean, log_std.exp())
                return float(dist.stddev.mean().item())
            if self._last_dist_std_mean > 0.0:
                return float(self._last_dist_std_mean)
            bias = self.model.std_head.bias
            log_std_min = float(getattr(self.model, "log_std_min", -3.0))
            log_std_max = float(getattr(self.model, "log_std_max", 0.0))
            clamped = torch.clamp(bias, min=log_std_min, max=log_std_max)
            return float(torch.exp(clamped).mean().item())

    def update_popart(self, returns: torch.Tensor) -> None:
        """Update PopArt running stats from unnormalized returns."""
        if hasattr(self.model, "update_popart"):
            self.model.update_popart(returns)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values using PopArt stats."""
        if hasattr(self.model, "normalize_values"):
            return self.model.normalize_values(values)
        return values


__all__ = ["MAPPOPolicy3D", "ActionNoiseConfig"]
