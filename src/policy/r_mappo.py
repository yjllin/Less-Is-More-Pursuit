"""
Recurrent Actor-Critic network for MAPPO with GRU backbone.
Optimized for High-Frequency Vectorized Environments.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

class RMAPPOActorCritic(nn.Module):
    """
    Optimized Recurrent Actor-Critic.
    Assumes input is always batched: (Batch, Obs_Dim).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 512,
        action_noise: Optional[Any] = None,
        min_action_std: float = 0.1,
        max_action_std: float = 0.5,
        centralized_critic: bool = False,
        critic_obs_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.centralized_critic = bool(centralized_critic)
        self.critic_obs_dim = int(critic_obs_dim) if critic_obs_dim is not None else int(obs_dim)
        critic_in_dim = self.critic_obs_dim if self.centralized_critic else int(obs_dim)

        # Backbone: Pre-activation style (Linear -> Norm -> Act) is stable for PPO
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.centralized_critic:
            self.critic_base = nn.Sequential(
                nn.Linear(critic_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.critic_base = self.base
        
        # Memory (separate GRUs for actor/critic)
        self.actor_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.critic_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor_post_gru_norm = nn.LayerNorm(hidden_dim)
        self.critic_post_gru_norm = nn.LayerNorm(hidden_dim)
        
        # Heads
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = PopArtLayer(hidden_dim, 1)
        self.min_dist_head = nn.Linear(hidden_dim, 1)
        self.ttc_head = nn.Linear(hidden_dim, 1)
        self.collision_prob_head = nn.Linear(hidden_dim, 1)
        # Predict next-step target relative position in agent body frame (normalized).
        self.target_rel_pred_head = nn.Linear(hidden_dim, 3)
        min_std = max(float(min_action_std), 1e-8)
        max_std = max(float(max_action_std), min_std)
        self.log_std_min = math.log(min_std)
        self.log_std_max = math.log(max_std)
        init_std = max_std
        if action_noise is not None:
            init_std = getattr(action_noise, "initial_std", max_std)

        self.apply(self._init_weights)
        # Initialize std_head bias in log-space and keep it within clamp range.
        init_log_std = math.log(max(float(init_std), 1e-8))
        init_log_std = min(max(init_log_std, self.log_std_min), self.log_std_max)
        with torch.no_grad():
            nn.init.constant_(self.std_head.bias, float(init_log_std))

    def _init_weights(self, module: nn.Module) -> None:
        # Orthogonal init is critical for PPO
        if isinstance(module, nn.Linear):
            # Default gain
            orthogonal_init(module, gain=math.sqrt(2))
            # Special gain for heads (0.01 makes initial policy near-zero / random)
            if module is self.actor_head or module is self.critic_head or module is self.std_head:
                orthogonal_init(module, gain=0.01)
            if module is self.min_dist_head:
                orthogonal_init(module, gain=1.0)
            if module is self.ttc_head or module is self.collision_prob_head:
                orthogonal_init(module, gain=1.0)
            if module is self.target_rel_pred_head:
                orthogonal_init(module, gain=1.0)
        elif isinstance(module, PopArtLayer):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=math.sqrt(2))
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def forward(
        self, 
        obs: torch.Tensor, 
        rnn_state: torch.Tensor, 
        masks: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast forward pass for Rollout (Single Step).
        Args:
            obs: (Batch, Obs_Dim)
            rnn_state: (2, 1, Batch, Hidden) for split actor/critic GRUs
            masks: (Batch, 1) - 1.0 if not done, 0.0 if done
        """
        actor_out, critic_out, next_state = self._forward_gru(obs, rnn_state, masks, critic_obs=critic_obs)

        # 3. Heads
        # Removed extra Tanh here to allow full dynamic range
        action_mean = self.actor_head(actor_out)
        action_log_std = torch.clamp(self.std_head(actor_out), self.log_std_min, self.log_std_max)
        value = self.critic_head(critic_out)
        
        return action_mean, action_log_std, value, next_state

    def forward_with_aux(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        masks: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass with auxiliary safety heads."""
        actor_out, critic_out, next_state = self._forward_gru(obs, rnn_state, masks, critic_obs=critic_obs)
        action_mean = self.actor_head(actor_out)
        action_log_std = torch.clamp(self.std_head(actor_out), self.log_std_min, self.log_std_max)
        value = self.critic_head(critic_out)
        min_dist_pred = torch.sigmoid(self.min_dist_head(critic_out))
        ttc_pred = torch.sigmoid(self.ttc_head(critic_out))
        collision_prob_pred = torch.sigmoid(self.collision_prob_head(critic_out))
        target_rel_pred = torch.tanh(self.target_rel_pred_head(critic_out))
        return (
            action_mean,
            action_log_std,
            value,
            next_state,
            min_dist_pred,
            ttc_pred,
            collision_prob_pred,
            target_rel_pred,
        )

    def _forward_gru(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        masks: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. MLP Encoder
        actor_x = self.base(obs)
        critic_in = critic_obs if critic_obs is not None else obs
        critic_x = self.critic_base(critic_in)

        # 2. GRU
        # Handle reset: if mask is 0, hidden state becomes 0
        mask = masks.view(1, -1, 1)
        if rnn_state.dim() == 4:
            if rnn_state.shape[0] != 2:
                raise ValueError("Expected rnn_state with shape (2, 1, B, H) for split GRUs.")
            actor_state = rnn_state[0] * mask
            critic_state = rnn_state[1] * mask
        elif rnn_state.dim() == 3:
            # Backward-compatible: share a single state for both GRUs
            actor_state = rnn_state * mask
            critic_state = rnn_state * mask
        else:
            raise ValueError("Invalid rnn_state shape for GRU.")

        # GRU expects (Batch, Seq_Len, Input_Size) -> (B, 1, H)
        actor_seq = actor_x.unsqueeze(1)
        critic_seq = critic_x.unsqueeze(1)
        actor_out, actor_next = self.actor_gru(actor_seq, actor_state)
        critic_out, critic_next = self.critic_gru(critic_seq, critic_state)

        # Flatten back to (Batch, Hidden)
        actor_out = self.actor_post_gru_norm(actor_out.squeeze(1))
        critic_out = self.critic_post_gru_norm(critic_out.squeeze(1))
        next_state = torch.stack((actor_next, critic_next), dim=0)
        return actor_out, critic_out, next_state

    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        rnn_state: torch.Tensor, 
        masks: torch.Tensor,
        action: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO Update.
        Used during training epoch to compute LogProb and Entropy.
        Note: This assumes 'obs' contains flattened sequences or single steps.
        For rigorous BPTT training, data should be handled in chunks.
        """
        mean, log_std, value, _ = self.forward(obs, rnn_state, masks, critic_obs=critic_obs)
        
        std = log_std.exp()
        dist = Normal(mean, std)
        
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
        
        return value, action_log_probs, dist_entropy

    def get_value(
        self,
        obs: torch.Tensor,
        rnn_state: torch.Tensor,
        masks: torch.Tensor,
        critic_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Helper to get just value (for GAE)."""
        _, critic_out, _ = self._forward_gru(obs, rnn_state, masks, critic_obs=critic_obs)
        return self.critic_head(critic_out)

    def update_popart(self, targets: torch.Tensor) -> None:
        self.critic_head.update(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.critic_head.normalize(values)


class PopArtLayer(nn.Module):
    """PopArt normalization for value targets with running stats."""

    def __init__(self, in_features: int, out_features: int = 1, beta: float = 0.0003, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.beta = float(beta)
        self.eps = float(eps)
        self.register_buffer("mu", torch.zeros(out_features))
        self.register_buffer("sigma", torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return y * self.sigma + self.mu

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mu) / self.sigma

    def update(self, targets: torch.Tensor) -> None:
        if targets.numel() == 0:
            return
        with torch.no_grad():
            flat = targets.view(-1, 1)
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0, unbiased=False)
            mu_old = self.mu.clone()
            sigma_old = self.sigma.clone()
            var_old = sigma_old * sigma_old
            mu_new = (1.0 - self.beta) * mu_old + self.beta * batch_mean
            var_new = (1.0 - self.beta) * var_old + self.beta * batch_var + (1.0 - self.beta) * self.beta * (batch_mean - mu_old) ** 2
            sigma_new = torch.sqrt(var_new + self.eps)

            self.weight.data.mul_(sigma_old / sigma_new)
            self.bias.data = (sigma_old * self.bias.data + mu_old - mu_new) / sigma_new

            self.mu.copy_(mu_new)
            self.sigma.copy_(sigma_new)

__all__ = ["RMAPPOActorCritic"]
