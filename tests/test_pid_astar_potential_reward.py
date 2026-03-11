"""
PIDA*


1. PIDAgentA*
2. (potential_reward)
3. 
4. Agent


- AgentA*
-  = potential_gamma * (prev_dist - curr_dist)
- 
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from src.config import load_config
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2


@dataclass
class EpisodeStats:
    """Episode"""
    total_steps: int = 0
    captured: bool = False
    collision_count: int = 0
    
  
    potential_rewards: List[float] = field(default_factory=list)
    total_potential: float = 0.0
    positive_potential_steps: int = 0
    negative_potential_steps: int = 0
    zero_potential_steps: int = 0
    
  
    initial_distance: float = 0.0
    final_distance: float = 0.0
    path_lengths: List[float] = field(default_factory=list)
    euclidean_distances: List[float] = field(default_factory=list)
    
  
    initial_path_length: float = 0.0
    total_path_reduction: float = 0.0
    
  
    total_reward: float = 0.0
    alignment_rewards: List[float] = field(default_factory=list)
    collision_penalties: List[float] = field(default_factory=list)
    time_penalties: List[float] = field(default_factory=list)


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

        # Adaptive speed
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


def compute_distance_to_target(env: VectorizedMutualAStarEnvV2, agent_idx: int = 0) -> float:
    """Agent"""
    pos = env.pos[0, agent_idx]
    target = env.target_pos[0]
    return float(np.linalg.norm(pos - target))


def run_potential_reward_test(
    cfg_path: str | None,
    episodes: int,
    stage_index: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """"""
    cfg = load_config(cfg_path) if cfg_path else load_config()
    env = VectorizedMutualAStarEnvV2(num_envs=1, cfg=cfg)

    speed = min(float(cfg.control.action_bounds["vx"]), 6.0)
    lookahead_dist = 12.0
    kp_vel = 0.6
    kd_vel = 0.2
    kp_yaw = 2.5

  
    stage_cfg = cfg.curriculum.stages[stage_index]
    stage_reward = getattr(stage_cfg, "reward", {})
    potential_gamma = stage_reward.get("potential_gamma", cfg.reward.potential_gamma)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" - Stage {stage_index}: {stage_cfg.name}")
        print(f"{'='*60}")
        print(f"potential_gamma = {potential_gamma}")
        print(f": {stage_cfg.target_behavior}")
        print(f": {stage_cfg.target_speed} m/s")
        print(f": {stage_cfg.view_radius_m} m")
        print(f"{'='*60}\n")

    all_stats: List[EpisodeStats] = []

    for ep in range(episodes):
        stats = EpisodeStats()
        obs, _ = env.reset(stage_index=stage_index)
        done = np.array([False], dtype=bool)
        step_idx = 0
        
      
        stats.initial_distance = compute_distance_to_target(env)
        stats.initial_path_length = float(env.path_lengths[0, 0])
        prev_path_length = stats.initial_path_length
        prev_target_dist = float(env.prev_target_dist[0, 0])
        
        if verbose and ep == 0:
            print(f"\n[Episode {ep+1} ]")
            print(f"  : {stats.initial_distance:.2f} m")
            print(f"  : {stats.initial_path_length:.2f} m")
            print(f"  prev_target_dist: {prev_target_dist:.2f} m")
        
        while not done[0]:
            actions = _pid_astar_actions(env, speed, kp_vel, kd_vel, kp_yaw, step_idx, lookahead_dist)
            obs, rewards, done, infos = env.step(actions)
            info = infos[0]
            
          
            curr_path_length = float(env.path_lengths[0, 0])
            curr_euclidean = compute_distance_to_target(env)
            curr_prev_target_dist = float(env.prev_target_dist[0, 0])
            
          
            path_reduction = prev_path_length - curr_path_length
            stats.total_path_reduction += path_reduction
            
          
            breakdown_list = info.get("reward_breakdown", [{}])
            breakdown = breakdown_list[0] if breakdown_list else {}
            potential_reward = breakdown.get("potential_reward", 0.0)
            alignment_reward = breakdown.get("alignment_reward", 0.0)
            collision_penalty = breakdown.get("collision_penalty", 0.0)
            time_penalty = breakdown.get("time_penalty", 0.0)
            
          
            if verbose and ep == 0 and step_idx < 20:
              
                # pot = gamma * (prev_dist - curr_dist)
              
                num_agents = env.num_agents
                per_agent_potential = potential_reward / num_agents if num_agents > 0 else 0
                print(f"  Step {step_idx:3d}: path_len={curr_path_length:7.2f}, "
                      f"euclidean={curr_euclidean:7.2f}, "
                      f"prev_dist={curr_prev_target_dist:7.2f}, "
                      f"total_pot={potential_reward:8.2f}, "
                      f"per_agent={per_agent_potential:8.2f}, "
                      f"path_delta={path_reduction:7.2f}")
            
          
            stats.potential_rewards.append(potential_reward)
            stats.total_potential += potential_reward
            stats.alignment_rewards.append(alignment_reward)
            stats.collision_penalties.append(collision_penalty)
            stats.time_penalties.append(time_penalty)
            stats.total_reward += float(rewards[0, 0])
            stats.path_lengths.append(curr_path_length)
            stats.euclidean_distances.append(curr_euclidean)
            
          
            prev_path_length = curr_path_length
            
            if potential_reward > 0.01:
                stats.positive_potential_steps += 1
            elif potential_reward < -0.01:
                stats.negative_potential_steps += 1
            else:
                stats.zero_potential_steps += 1
            
            if info.get("collision", False):
                stats.collision_count += 1
            
            stats.total_steps += 1
            step_idx += 1
        
      
        stats.final_distance = compute_distance_to_target(env)
        stats.captured = info.get("captured", False)
        all_stats.append(stats)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} - "
                  f"Captured: {stats.captured}, "
                  f"Steps: {stats.total_steps}, "
                  f"Total Potential: {stats.total_potential:.2f}, "
                  f"Positive/Negative/Zero: {stats.positive_potential_steps}/{stats.negative_potential_steps}/{stats.zero_potential_steps}")

  
    results = analyze_results(all_stats, potential_gamma, verbose, speed)
    return results


def analyze_results(all_stats: List[EpisodeStats], potential_gamma: float, verbose: bool, speed: float = 6.0) -> Dict[str, float]:
    """"""
    total_episodes = len(all_stats)
    captured_count = sum(1 for s in all_stats if s.captured)
    
  
    all_potentials = []
    for s in all_stats:
        all_potentials.extend(s.potential_rewards)
    
    total_positive_steps = sum(s.positive_potential_steps for s in all_stats)
    total_negative_steps = sum(s.negative_potential_steps for s in all_stats)
    total_zero_steps = sum(s.zero_potential_steps for s in all_stats)
    total_steps = sum(s.total_steps for s in all_stats)
    
    avg_total_potential = np.mean([s.total_potential for s in all_stats])
    avg_potential_per_step = np.mean(all_potentials) if all_potentials else 0.0
    std_potential_per_step = np.std(all_potentials) if all_potentials else 0.0
    
  
    avg_initial_dist = np.mean([s.initial_distance for s in all_stats])
    avg_final_dist = np.mean([s.final_distance for s in all_stats])
    avg_dist_reduction = avg_initial_dist - avg_final_dist
    
  
    avg_initial_path = np.mean([s.initial_path_length for s in all_stats])
    avg_path_reduction = np.mean([s.total_path_reduction for s in all_stats])
    
  
    theoretical_potential = potential_gamma * avg_path_reduction
    
  
    avg_collisions = np.mean([s.collision_count for s in all_stats])
    
    if verbose:
        print(f"\n{'='*60}")
        print("")
        print(f"{'='*60}")
        print(f"\n[]")
        print(f"  Episode: {total_episodes}")
        print(f"  : {captured_count}/{total_episodes} ({100*captured_count/total_episodes:.1f}%)")
        print(f"  : {total_steps/total_episodes:.1f}")
        print(f"  : {avg_collisions:.2f}")
        
        print(f"\n[]")
        print(f"  : {avg_initial_dist:.2f} m")
        print(f"  : {avg_final_dist:.2f} m")
        print(f"  : {avg_dist_reduction:.2f} m")
        print(f"  : {avg_initial_path:.2f} m")
        print(f"  : {avg_path_reduction:.2f} m")
        
        print(f"\n[]")
        print(f"  potential_gamma: {potential_gamma}")
        print(f"  : {theoretical_potential:.2f} (gamma * )")
        print(f"  : {avg_total_potential:.2f}")
        if abs(theoretical_potential) > 0.01:
            diff_pct = 100 * (avg_total_potential / theoretical_potential - 1)
            print(f"  : {avg_total_potential - theoretical_potential:.2f} ({diff_pct:.1f}% )")
        else:
            print(f"  : {avg_total_potential - theoretical_potential:.2f}")
        print(f"  : {avg_potential_per_step:.4f}  {std_potential_per_step:.4f}")
        
        print(f"\n[]")
        print(f"  : {total_positive_steps} ({100*total_positive_steps/total_steps:.1f}%)")
        print(f"  : {total_negative_steps} ({100*total_negative_steps/total_steps:.1f}%)")
        print(f"  : {total_zero_steps} ({100*total_zero_steps/total_steps:.1f}%)")
        
      
        if all_potentials:
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            print(f"\n[]")
            for p in percentiles:
                val = np.percentile(all_potentials, p)
                print(f"  {p}%: {val:.4f}")
        
      
        print(f"\n[]")
        
      
        first_step_potentials = [s.potential_rewards[0] if s.potential_rewards else 0 for s in all_stats]
        avg_first_step = np.mean(first_step_potentials)
        if abs(avg_first_step) > 50:
            print(f"   :  (={avg_first_step:.2f})")
            print("     : resetprev_target_diststeppath_lengthsA*")
            print("     : resetA*prev_target_dist")
        
        if avg_total_potential < 0.1 * theoretical_potential and theoretical_potential > 0:
            print("   : !")
            print("     :")
            print("     1. path_lengths")
            print("     2. ")
            print("     3. potential_gamma")
        elif total_zero_steps > 0.5 * total_steps:
            print("   : 50%!")
            print("     :")
            print("     1. is_potential_unstable")
            print("     2. ")
        else:
            print("   97%+")
            
      
        potentials_after_first = []
        for s in all_stats:
            if len(s.potential_rewards) > 1:
                potentials_after_first.extend(s.potential_rewards[1:])
        if potentials_after_first:
            avg_after_first = np.mean(potentials_after_first)
            print(f"\n[]")
            print(f"  : {avg_after_first:.4f}")
            print(f"  : {potential_gamma * speed / 10:.4f} (={speed:.1f}m/s, dt=0.1s)")
        
        print(f"\n{'='*60}\n")
    
    return {
        "episodes": float(total_episodes),
        "capture_rate": captured_count / max(total_episodes, 1),
        "avg_steps": total_steps / max(total_episodes, 1),
        "avg_collisions": avg_collisions,
        "avg_initial_dist": avg_initial_dist,
        "avg_final_dist": avg_final_dist,
        "avg_dist_reduction": avg_dist_reduction,
        "avg_initial_path": avg_initial_path,
        "avg_path_reduction": avg_path_reduction,
        "theoretical_potential": theoretical_potential,
        "actual_avg_potential": avg_total_potential,
        "potential_per_step": avg_potential_per_step,
        "positive_potential_ratio": total_positive_steps / max(total_steps, 1),
        "negative_potential_ratio": total_negative_steps / max(total_steps, 1),
        "zero_potential_ratio": total_zero_steps / max(total_steps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PIDA*")
    parser.add_argument("--config", type=str, default=None, help="")
    parser.add_argument("--episodes", type=int, default=20, help="Episode")
    parser.add_argument("--stage", type=int, default=0, help="Stage")
    parser.add_argument("--quiet", action="store_true", help="")
    args = parser.parse_args()

    results = run_potential_reward_test(
        args.config,
        args.episodes,
        stage_index=args.stage,
        verbose=not args.quiet,
    )
    
  
    print(f"[] capture_rate={results['capture_rate']:.2f} | "
          f"avg_potential={results['actual_avg_potential']:.2f} | "
          f"theoretical={results['theoretical_potential']:.2f} | "
          f"positive_ratio={results['positive_potential_ratio']:.2f}")


if __name__ == "__main__":
    main()
