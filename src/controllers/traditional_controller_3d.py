"""Traditional 3D controller baseline (APF + PN/PD) for evaluation-only runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TraditionalControllerStats:
    mean_action_norm: float = 0.0
    mean_yaw_rate_cmd: float = 0.0
    apf_saturation_ratio: float = 0.0


class TraditionalController3D:
    """Classical controller baseline with RL-policy compatible `act` interface."""

    def __init__(self, env, cfg, device: torch.device, nav_gain: float = 1.8, rep_gain: float = 12.0) -> None:
        self.env = env
        self.cfg = cfg
        self.device = device
        bounds = cfg.control.action_bounds
        self._scale = np.array(
            [
                float(bounds.get("vx", 8.0)),
                float(bounds.get("vy", 4.0)),
                float(bounds.get("vz", 3.0)),
                float(bounds.get("yaw_rate", 0.8)),
            ],
            dtype=np.float64,
        )
        self._yaw_rate_max = float(self._scale[3])
        self._nav_gain = float(nav_gain)
        self._rep_gain = float(rep_gain)
        self._damp_gain = 0.25
        self._pn_gain = 2.0
        self._kp_gain = 0.10
        self._apf_radius_m = max(18.0, 3.0 * float(getattr(env, "voxel_size", 6.0)))
        self._boundary_margin = 10.0
        self._stats = TraditionalControllerStats()

    @property
    def stats(self) -> TraditionalControllerStats:
        return self._stats

    def init_hidden(self, num_envs: int) -> torch.Tensor:
        # Match policy hidden shape; unused by controller.
        return torch.zeros(2, 1, num_envs, 1, device=self.device)

    def eval(self) -> "TraditionalController3D":
        return self

    def step(self, obs, env_ctx=None) -> np.ndarray:
        env = env_ctx or self.env
        B, N = env.num_envs, env.num_agents
        actions = np.zeros((B, N, 4), dtype=np.float32)
        sat_count = 0
        action_norm_sum = 0.0
        yaw_rate_sum = 0.0

        for b in range(B):
            tgt_pos = env.target_pos[b]
            tgt_vel = env.target_vel[b]
            for n in range(N):
                pos = env.pos[b, n]
                vel = env.vel[b, n]
                yaw = float(env.yaw[b, n])

                f_att = self._pn_attraction(pos, vel, tgt_pos, tgt_vel)
                f_rep = self._apf_repulsion(env, pos)
                f_bnd = self._boundary_repulsion(env, pos)
                cmd_world = self._nav_gain * f_att + self._rep_gain * f_rep + f_bnd - self._damp_gain * vel

                # Convert to desired world velocity and clip to action-equivalent limits.
                vxw = float(np.clip(cmd_world[0], -self._scale[0], self._scale[0]))
                vyw = float(np.clip(cmd_world[1], -self._scale[0], self._scale[0]))
                vzw = float(np.clip(cmd_world[2], -self._scale[2], self._scale[2]))
                desired_world = np.array([vxw, vyw, vzw], dtype=np.float64)

                # World -> body for x/y; z remains world z == body z in this simplified model.
                cy = math.cos(yaw)
                sy = math.sin(yaw)
                vx_body = cy * desired_world[0] + sy * desired_world[1]
                vy_body = -sy * desired_world[0] + cy * desired_world[1]
                vz_body = desired_world[2]

                rel = tgt_pos - pos
                target_yaw = math.atan2(rel[1], rel[0]) if np.linalg.norm(rel[:2]) > 1e-6 else yaw
                yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
                yaw_rate = float(np.clip(1.4 * yaw_err, -self._yaw_rate_max, self._yaw_rate_max))

                action = np.array(
                    [
                        np.clip(vx_body, -self._scale[0], self._scale[0]),
                        np.clip(vy_body, -self._scale[1], self._scale[1]),
                        np.clip(vz_body, -self._scale[2], self._scale[2]),
                        yaw_rate,
                    ],
                    dtype=np.float32,
                )
                if np.any(np.abs(action - np.array([vx_body, vy_body, vz_body, yaw_rate], dtype=np.float32)) > 1e-5):
                    sat_count += 1
                action_norm_sum += float(np.linalg.norm(action[:3]))
                yaw_rate_sum += abs(float(action[3]))
                actions[b, n] = action

        total = max(B * N, 1)
        self._stats = TraditionalControllerStats(
            mean_action_norm=action_norm_sum / total,
            mean_yaw_rate_cmd=yaw_rate_sum / total,
            apf_saturation_ratio=float(sat_count) / float(total),
        )
        return actions

    def act(self, obs_tensor: torch.Tensor, rnn_state: torch.Tensor, masks: torch.Tensor, **kwargs):
        _ = obs_tensor, masks, kwargs
        actions_np = self.step(None, env_ctx=self.env)
        actions = torch.from_numpy(actions_np.reshape(self.env.num_envs * self.env.num_agents, 4)).to(self.device)
        logp = torch.zeros(actions.shape[0], device=self.device)
        value = torch.zeros(actions.shape[0], 1, device=self.device)
        return actions, logp, value, rnn_state

    def _pn_attraction(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        tgt_pos: np.ndarray,
        tgt_vel: np.ndarray,
    ) -> np.ndarray:
        rel = tgt_pos - pos
        dist = float(np.linalg.norm(rel))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float64)
        los = rel / dist
        v_rel = tgt_vel - vel
        # 3D PN-style lateral term + proportional closing term fallback.
        lateral = np.cross(np.cross(los, v_rel), los)
        attractive = self._pn_gain * lateral + self._kp_gain * rel
        return attractive.astype(np.float64, copy=False)

    def _boundary_repulsion(self, env, pos: np.ndarray) -> np.ndarray:
        f = np.zeros(3, dtype=np.float64)
        world_min = env.world_min
        world_max = env.world_max
        margin = self._boundary_margin
        for i in range(3):
            d_min = float(pos[i] - world_min[i])
            d_max = float(world_max[i] - pos[i])
            if d_min < margin:
                f[i] += (margin - d_min) / max(margin, 1e-6)
            if d_max < margin:
                f[i] -= (margin - d_max) / max(margin, 1e-6)
        return f

    def _apf_repulsion(self, env, pos: np.ndarray) -> np.ndarray:
        grid = getattr(env, "occupancy_grid", None)
        if grid is None:
            return np.zeros(3, dtype=np.float64)
        origin = np.asarray(env.origin, dtype=np.float64)
        voxel = float(env.voxel_size)
        radius = self._apf_radius_m
        r_vox = max(1, int(math.ceil(radius / max(voxel, 1e-6))))
        gshape = grid.shape
        center = np.floor((pos - origin) / voxel).astype(np.int64)
        rep = np.zeros(3, dtype=np.float64)
        for ix in range(max(0, center[0] - r_vox), min(gshape[0], center[0] + r_vox + 1)):
            for iy in range(max(0, center[1] - r_vox), min(gshape[1], center[1] + r_vox + 1)):
                for iz in range(max(0, center[2] - r_vox), min(gshape[2], center[2] + r_vox + 1)):
                    if grid[ix, iy, iz] <= 0:
                        continue
                    c = origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * voxel
                    dvec = pos - c
                    dist = float(np.linalg.norm(dvec))
                    if dist < 1e-6 or dist > radius:
                        continue
                    weight = (1.0 / max(dist * dist, 1e-6)) * ((radius - dist) / radius)
                    rep += (dvec / dist) * weight
        return rep

