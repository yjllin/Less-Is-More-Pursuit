"""Evaluate PID control following A* paths in the vectorized sim environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from src.config import load_config
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2


def _angle_wrap(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def _select_lookahead_point(path: list[np.ndarray], pos: np.ndarray, lookahead_dist: float) -> np.ndarray:
    if not path:
        return pos.copy()
    if len(path) == 1:
        return path[0].copy()
    accum = 0.0
    prev = path[0]
    for p in path[1:]:
        seg = p - prev
        seg_len = float(np.linalg.norm(seg))
        if accum + seg_len >= lookahead_dist:
            if seg_len > 1e-6:
                t = (lookahead_dist - accum) / seg_len
                return prev + seg * t
            return p.copy()
        accum += seg_len
        prev = p
    return path[-1].copy()


def _pid_astar_actions(
    env: VectorizedMutualAStarEnvV2,
    speed: float,
    kp_vel: float,
    kd_vel: float,
    kp_yaw: float,
    step_idx: int,
    lookahead_dist: float,
) -> np.ndarray:
    """Compute PID actions following A* paths for all agents in env0."""
    B, N = env.num_envs, env.num_agents
    actions = np.zeros((B, N, 4), dtype=np.float64)
    max_yaw_rate = float(env.cfg.control.action_bounds["yaw_rate"])
    max_vx = float(env.cfg.control.action_bounds["vx"])
    max_vy = float(env.cfg.control.action_bounds["vy"])
    max_vz = float(env.cfg.control.action_bounds["vz"])

    for n in range(N):
        pos = env.pos[0, n]
        vel = env.vel[0, n]
        yaw = float(env.yaw[0, n])
        target = env.target_pos[0]

        direction, world_path, _ = env.navigator.compute_direction(
            pos, target, current_step=step_idx, current_velocity=vel, cache_key=(0, n)
        )
        if world_path and len(world_path) >= 2:
            waypoint = _select_lookahead_point(world_path, pos, lookahead_dist)
        else:
            waypoint = target.copy()

        pos_err = waypoint - pos
        dist = float(np.linalg.norm(pos_err))
        if dist > 1e-6:
            desired_vel_world = pos_err / dist * speed
        else:
            desired_vel_world = np.zeros(3, dtype=np.float64)

        # Adaptive speed: slow down for sharp turns and short distances
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        desired_dir = desired_vel_world.copy()
        d_norm = float(np.linalg.norm(desired_dir[:2]))
        if d_norm > 1e-6:
            desired_dir[:2] /= d_norm
        turn_angle = float(np.arccos(np.clip(heading[0] * desired_dir[0] + heading[1] * desired_dir[1], -1.0, 1.0)))
        turn_scale = max(0.3, 1.0 - turn_angle / np.pi)
        dist_scale = max(0.3, min(1.0, dist / max(lookahead_dist, 1e-3)))
        speed_cmd = speed * turn_scale * dist_scale
        if speed_cmd < 0.5:
            speed_cmd = 0.5

        vel_err = desired_vel_world - vel
        cmd_world = desired_vel_world + kd_vel * vel_err

        cmd_speed = float(np.linalg.norm(cmd_world))
        if cmd_speed > speed_cmd and cmd_speed > 1e-6:
            cmd_world = cmd_world / cmd_speed * speed_cmd

        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cmd_body_x = cmd_world[0] * cy + cmd_world[1] * sy
        cmd_body_y = -cmd_world[0] * sy + cmd_world[1] * cy
        cmd_body_z = cmd_world[2]

        desired_yaw = float(np.arctan2(desired_vel_world[1], desired_vel_world[0]))
        yaw_err = _angle_wrap(desired_yaw - yaw)
        yaw_rate = float(np.clip(kp_yaw * yaw_err, -max_yaw_rate, max_yaw_rate))

        actions[0, n, 0] = np.clip(cmd_body_x, -max_vx, max_vx)
        actions[0, n, 1] = np.clip(cmd_body_y, -max_vy, max_vy)
        actions[0, n, 2] = np.clip(cmd_body_z, -max_vz, max_vz)
        actions[0, n, 3] = yaw_rate

    return actions


def run_pid_astar_eval(cfg_path: str | None, episodes: int) -> Dict[str, float]:
    cfg = load_config(cfg_path) if cfg_path else load_config()
    env = VectorizedMutualAStarEnvV2(num_envs=1, cfg=cfg)

    stage_index = 0  # stage1
    speed = min(float(cfg.control.action_bounds["vx"]), 6.0)
    lookahead_dist = 12.0

    kp_vel = 0.6
    kd_vel = 0.2
    kp_yaw = 2.5

    success = 0
    collisions = 0
    timeouts = 0
    collision_events = 0
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset(stage_index=stage_index)
        done = np.array([False], dtype=bool)
        step_idx = 0
        while not done[0]:
            actions = _pid_astar_actions(env, speed, kp_vel, kd_vel, kp_yaw, step_idx, lookahead_dist)
            obs, rewards, done, infos = env.step(actions)
            info = infos[0]
            if info.get("collision", False):
                collisions += 1
                collision_events += int(np.sum(info.get("collisions", [])))
            total_steps += 1
            step_idx += 1
        if info.get("captured", False):
            success += 1
        elif info.get("collision", False):
            collisions += 0  # already counted
        else:
            timeouts += 1

    return {
        "episodes": float(episodes),
        "success_rate": success / max(episodes, 1),
        "collision_rate": collisions / max(episodes, 1),
        "timeout_rate": timeouts / max(episodes, 1),
        "collision_events": float(collision_events),
        "avg_steps": total_steps / max(episodes, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PID A* follow evaluation (stage1).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes.")
    args = parser.parse_args()

    results = run_pid_astar_eval(args.config, args.episodes)
    print(f"[PID-A*] episodes={int(results['episodes'])} | "
          f"success_rate={results['success_rate']:.3f} | "
          f"collision_rate={results['collision_rate']:.3f} | "
          f"timeout_rate={results['timeout_rate']:.3f} | "
          f"collision_events={int(results['collision_events'])} | "
          f"avg_steps={results['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
