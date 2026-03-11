from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class AgentState:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    yaw: float
    prev_action: NDArray[np.float64]
    shield_triggered: int = 0


@dataclass
class TargetState:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]


@dataclass
class StageTracker:
    name: str
    target_speed: float
    view_radius: float
    guidance_enabled: bool
    target_behavior: str
    collision_penalty: Optional[float]
    randomize_physics: bool
    capture_radius_m: Optional[float] = None
    reward: Optional[dict] = None
