"""Action post-processing for quadrotor control."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def smooth_action(
    raw_action: NDArray[np.float64],
    prev_action: NDArray[np.float64],
    alpha: float,
    clip: float,
) -> NDArray[np.float64]:
    """
    Exponential moving average to suppress high-frequency oscillation.

    raw_action: [vx, vy, vz, yaw_rate] command.
    prev_action: previous command in the same space.
    alpha: smoothing coefficient in [0, 1], higher = smoother.
    clip: clamp magnitude to avoid drift after EMA.
    """

    blended = alpha * raw_action + (1.0 - alpha) * prev_action
    return np.clip(blended, -clip, clip)
