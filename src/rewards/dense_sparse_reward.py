"""Dense + sparse reward calculator with explicit components (3D aligned to 2D design)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RewardBreakdown:
    time_penalty: float
    collision_penalty: float
    dead_penalty: float
    alignment_reward: float
    lidar_penalty: float
    exploration_reward: float
    frontier_reward: float
    potential_reward: float
    standstill_penalty: float
    steering_penalty: float
    wrong_way_penalty: float
    capture_bonus: float
    tactical_reward: float
    encircle_reward: float
    smoothness_penalty: float
    safety_penalty: float
    shield_penalty: float

    @property
    def total(self) -> float:
        return (
            self.time_penalty
            + self.collision_penalty
            + self.dead_penalty
            + self.alignment_reward
            + self.lidar_penalty
            + self.exploration_reward
            + self.frontier_reward
            + self.potential_reward
            + self.standstill_penalty
            + self.steering_penalty
            + self.wrong_way_penalty
            + self.capture_bonus
            + self.tactical_reward
            + self.encircle_reward
            + self.smoothness_penalty
            + self.safety_penalty
            + self.shield_penalty
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "time_penalty": self.time_penalty,
            "collision_penalty": self.collision_penalty,
            "dead_penalty": self.dead_penalty,
            "alignment_reward": self.alignment_reward,
            "lidar_penalty": self.lidar_penalty,
            "exploration_reward": self.exploration_reward,
            "frontier_reward": self.frontier_reward,
            "potential_reward": self.potential_reward,
            "standstill_penalty": self.standstill_penalty,
            "steering_penalty": self.steering_penalty,
            "wrong_way_penalty": self.wrong_way_penalty,
            "capture_bonus": self.capture_bonus,
            "tactical_reward": self.tactical_reward,
            "encircle_reward": self.encircle_reward,
            "smoothness_penalty": self.smoothness_penalty,
            "safety_penalty": self.safety_penalty,
            "shield_penalty": self.shield_penalty,
            "total": self.total,
        }


__all__ = ["RewardBreakdown"]
