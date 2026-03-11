"""
Stage3 



1.  - 
2.  - 
3.  - action smoothing
"""

from __future__ import annotations
import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from typing import Dict
import time

from src.config import load_config, ThreeDConfig
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2


class PolicyBehaviorDiagnostics:
    """"""
    
    def __init__(self, cfg: ThreeDConfig, num_envs: int = 8):
        self.cfg = cfg
        self.num_envs = num_envs
        self.env = None
        self.results: Dict[str, any] = {}

    def setup_env(self, stage_idx: int = 2) -> None:
        """"""
        print(f"[Setup] ...")
        self.env = VectorizedMutualAStarEnvV2(
            num_envs=self.num_envs,
            cfg=self.cfg,
        )
        self.stage_idx = stage_idx
        self.env.reset(stage_index=stage_idx)
        print(f"[Setup] Stage: {self.env.stages[stage_idx].name}")

    def test_reward_signal_analysis(self, num_steps: int = 500) -> Dict:
        """"""
        print("\n" + "="*60)
        print("[Test] ")
        print("="*60)
        
        obs, info = self.env.reset(stage_index=self.stage_idx)
        
        reward_sums = {
            "time_penalty": 0.0,
            "survival_bonus": 0.0,
            "collision_penalty": 0.0,
            "alignment_reward": 0.0,
            "potential_reward": 0.0,
            "lidar_penalty": 0.0,
            "capture_bonus": 0.0,
            "total": 0.0,
        }
        
        collision_count = 0
        capture_count = 0
        
        for step in range(num_steps):
          
            guidance = self.env.guidance_vectors.copy()
            actions = np.zeros((self.num_envs, self.env.num_agents, 4))
            
            for b in range(self.num_envs):
                for n in range(self.env.num_agents):
                    yaw = self.env.yaw[b, n]
                    cy, sy = np.cos(yaw), np.sin(yaw)
                    gx, gy, gz = guidance[b, n]
                    gnorm = np.sqrt(gx*gx + gy*gy + gz*gz)
                    if gnorm > 1e-6:
                        gx, gy, gz = gx/gnorm, gy/gnorm, gz/gnorm
                    actions[b, n, 0] = (gx * cy + gy * sy) * 5.0
                    actions[b, n, 1] = (-gx * sy + gy * cy) * 2.0
                    actions[b, n, 2] = gz * 2.0
            
            obs, rewards, dones, infos = self.env.step(actions)
            
          
            for info in infos:
                rb = info.get("reward_breakdown", {})
                for key in reward_sums:
                    if key in rb:
                        reward_sums[key] += rb[key]
                if info.get("collision", False):
                    collision_count += 1
                if info.get("captured", False):
                    capture_count += 1
            
            if np.any(dones):
                obs, info = self.env.reset(stage_index=self.stage_idx)
        
      
        total_agent_steps = num_steps * self.num_envs * self.env.num_agents
        
        print(f"\n   (: {total_agent_steps}):")
        print(f"  {'':<25} {'':>15} {'':>15}")
        print(f"  {'-'*55}")
        
        for key, value in reward_sums.items():
            avg = value / total_agent_steps
            print(f"  {key:<25} {value:>15.2f} {avg:>15.4f}")
        
        print(f"\n  : {collision_count}")
        print(f"  : {capture_count}")
        
      
        print(f"\n  :")
        if abs(reward_sums["collision_penalty"]) > 0:
            ratio = abs(reward_sums["capture_bonus"]) / abs(reward_sums["collision_penalty"])
            print(f"  / = {ratio:.2f}")
            if ratio > 10:
                print(f"  [] ")
        
        if abs(reward_sums["lidar_penalty"]) > 0:
            ratio = abs(reward_sums["potential_reward"]) / abs(reward_sums["lidar_penalty"])
            print(f"  /LiDAR = {ratio:.2f}")
        
        self.results["reward_analysis"] = reward_sums
        return reward_sums

    def test_action_smoothing_impact(self, num_steps: int = 200) -> Dict:
        """"""
        print("\n" + "="*60)
        print("[Test] ")
        print("="*60)
        
        obs, info = self.env.reset(stage_index=self.stage_idx)
        
      
        action_velocity_diffs = []
        response_delays = []
        
        for step in range(num_steps):
          
            guidance = self.env.guidance_vectors.copy()
            target_actions = np.zeros((self.num_envs, self.env.num_agents, 4))
            
            for b in range(self.num_envs):
                for n in range(self.env.num_agents):
                    yaw = self.env.yaw[b, n]
                    cy, sy = np.cos(yaw), np.sin(yaw)
                    gx, gy, gz = guidance[b, n]
                    gnorm = np.sqrt(gx*gx + gy*gy + gz*gz)
                    if gnorm > 1e-6:
                        gx, gy, gz = gx/gnorm, gy/gnorm, gz/gnorm
                    target_actions[b, n, 0] = (gx * cy + gy * sy) * 6.0
                    target_actions[b, n, 1] = (-gx * sy + gy * cy) * 2.0
                    target_actions[b, n, 2] = gz * 2.0
            
          
            vel_before = self.env.vel.copy()
            
            obs, rewards, dones, infos = self.env.step(target_actions)
            
          
            vel_after = self.env.vel.copy()
            
          
            for b in range(self.num_envs):
                for n in range(self.env.num_agents):
                    yaw = self.env.yaw[b, n]
                    cy, sy = np.cos(yaw), np.sin(yaw)
                    # Body to world
                    target_vx = target_actions[b, n, 0] * cy - target_actions[b, n, 1] * sy
                    target_vy = target_actions[b, n, 0] * sy + target_actions[b, n, 1] * cy
                    target_vz = target_actions[b, n, 2]
                    
                    actual_vx = vel_after[b, n, 0]
                    actual_vy = vel_after[b, n, 1]
                    actual_vz = vel_after[b, n, 2]
                    
                    diff = np.sqrt((target_vx - actual_vx)**2 + 
                                   (target_vy - actual_vy)**2 + 
                                   (target_vz - actual_vz)**2)
                    action_velocity_diffs.append(diff)
            
            if np.any(dones):
                obs, info = self.env.reset(stage_index=self.stage_idx)
        
        mean_diff = np.mean(action_velocity_diffs)
        max_diff = np.max(action_velocity_diffs)
        
        result = {
            "smoothing_alpha": self.cfg.control.smoothing_alpha,
            "mean_velocity_diff": float(mean_diff),
            "max_velocity_diff": float(max_diff),
            "max_speed": self.cfg.control.action_bounds["vx"],
            "response_ratio": float(mean_diff / self.cfg.control.action_bounds["vx"]),
        }
        
        print(f"   (alpha): {result['smoothing_alpha']}")
        print(f"  : {mean_diff:.2f} m/s")
        print(f"  : {max_diff:.2f} m/s")
        print(f"  : {result['response_ratio']*100:.1f}%")
        
        if result['response_ratio'] > 0.3:
            print(f"  [] >30%")
            print(f"  :  smoothing_alpha  {result['smoothing_alpha']}  0.2-0.3")
        
        self.results["smoothing_impact"] = result
        return result

    def test_aggressive_chase_collision(self, num_episodes: int = 5, max_steps: int = 500) -> Dict:
        """"""
        print("\n" + "="*60)
        print("[Test] ")
        print("="*60)
        
      
        speed_levels = [3.0, 5.0, 7.0, 8.0]  # 
        results_by_speed = {}
        
        for speed in speed_levels:
            total_steps = 0
            total_collisions = 0
            total_captures = 0
            
            for ep in range(num_episodes):
                obs, info = self.env.reset(stage_index=self.stage_idx)
                
                for step in range(max_steps):
                    actions = np.zeros((self.num_envs, self.env.num_agents, 4))
                    
                    for b in range(self.num_envs):
                        for n in range(self.env.num_agents):
                          
                            dx = self.env.target_pos[b, 0] - self.env.pos[b, n, 0]
                            dy = self.env.target_pos[b, 1] - self.env.pos[b, n, 1]
                            dz = self.env.target_pos[b, 2] - self.env.pos[b, n, 2]
                            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                            
                            if dist > 1e-6:
                                yaw = self.env.yaw[b, n]
                                cy, sy = np.cos(yaw), np.sin(yaw)
                              
                                vx_world = dx / dist * speed
                                vy_world = dy / dist * speed * 0.3
                                vz_world = dz / dist * speed * 0.3
                                
                              
                                actions[b, n, 0] = vx_world * cy + vy_world * sy
                                actions[b, n, 1] = -vx_world * sy + vy_world * cy
                                actions[b, n, 2] = vz_world
                    
                    obs, rewards, dones, infos = self.env.step(actions)
                    
                    for info in infos:
                        if info.get("collision", False):
                            total_collisions += 1
                        if info.get("captured", False):
                            total_captures += 1
                    
                    total_steps += self.num_envs
                    
                    if np.any(dones):
                        break
            
            collision_rate = total_collisions / total_steps if total_steps > 0 else 0
            capture_rate = total_captures / (num_episodes * self.num_envs)
            
            results_by_speed[speed] = {
                "collision_rate": collision_rate,
                "capture_rate": capture_rate,
                "total_steps": total_steps,
            }
            
            print(f"   {speed:.1f} m/s: ={collision_rate*100:.2f}%, ={capture_rate*100:.1f}%")
        
      
        print(f"\n  :")
        speeds = list(results_by_speed.keys())
        collision_rates = [results_by_speed[s]["collision_rate"] for s in speeds]
        
        if collision_rates[-1] > collision_rates[0] * 2:
            print(f"  [] ")
            print(f"  : ")
        
        self.results["aggressive_chase"] = results_by_speed
        return results_by_speed

    def test_lidar_reaction_time(self, num_trials: int = 100) -> Dict:
        """LiDAR"""
        print("\n" + "="*60)
        print("[Test] LiDAR")
        print("="*60)
        
        obs, info = self.env.reset(stage_index=self.stage_idx)
        
      
        shield_dist = self.cfg.shield.min_distance_m
        max_speed = self.cfg.control.action_bounds["vx"]
        dt = 1.0 / self.cfg.environment.step_hz
        
      
        reaction_steps = shield_dist / (max_speed * dt)
        
      
        alpha = self.cfg.control.smoothing_alpha
      
      
        # v_n = alpha^n * v_0
      
        if alpha > 0:
            steps_to_stop = int(np.ceil(np.log(0.1) / np.log(alpha)))
        else:
            steps_to_stop = 1
        
        stopping_distance = max_speed * dt * steps_to_stop * (1 + alpha) / 2
        
        result = {
            "shield_distance": shield_dist,
            "max_speed": max_speed,
            "dt": dt,
            "smoothing_alpha": alpha,
            "theoretical_reaction_steps": reaction_steps,
            "steps_to_stop": steps_to_stop,
            "stopping_distance": stopping_distance,
            "safety_margin": shield_dist - stopping_distance,
        }
        
        print(f"  Shield: {shield_dist:.1f}m")
        print(f"  : {max_speed:.1f}m/s")
        print(f"  : {dt:.3f}s")
        print(f"  : {alpha}")
        print(f"  : {reaction_steps:.1f}")
        print(f"  : {steps_to_stop}")
        print(f"  : {stopping_distance:.2f}m")
        print(f"  : {result['safety_margin']:.2f}m")
        
        if result['safety_margin'] < 0:
            print(f"  [] ! shield")
            print(f"  : shield {stopping_distance + 2:.1f}m smoothing_alpha")
        elif result['safety_margin'] < 1.0:
            print(f"  [] 1m")
        
        self.results["reaction_time"] = result
        return result

    def test_reward_config_analysis(self) -> Dict:
        """Stage3"""
        print("\n" + "="*60)
        print("[Test] Stage3 ")
        print("="*60)
        
        stage_cfg = self.cfg.curriculum.stages[self.stage_idx]
        stage_reward = stage_cfg.reward or {}
        
      
        collision_penalty = stage_reward.get("collision_penalty", self.cfg.reward.collision_penalty)
        capture_bonus = stage_reward.get("capture_bonus", self.cfg.reward.capture_bonus)
        potential_gamma = stage_reward.get("potential_gamma", self.cfg.reward.potential_gamma)
        lidar_alpha = stage_reward.get("lidar_alpha", self.cfg.reward.lidar_alpha)
        
        print(f"  Stage3 :")
        print(f"  - collision_penalty: {collision_penalty}")
        print(f"  - capture_bonus: {capture_bonus}")
        print(f"  - potential_gamma: {potential_gamma}")
        print(f"  - lidar_alpha: {lidar_alpha}")
        
      
        print(f"\n  :")
        
      
      
      
      
        avg_steps = 3000
        avg_collisions = 10
        avg_captures = 1
        
        total_collision_penalty = collision_penalty * avg_collisions
        total_capture_bonus = capture_bonus * avg_captures
        total_potential = potential_gamma * avg_steps * 0.1  # 0.1
        
        print(f"   (3000, 10, 1):")
        print(f"  - : {total_collision_penalty:.0f}")
        print(f"  - : {total_capture_bonus:.0f}")
        print(f"  - : {total_potential:.0f}")
        
      
        if abs(total_collision_penalty) < total_capture_bonus * 0.5:
            print(f"\n  [] ")
            print(f"  : ||/ = {abs(total_collision_penalty)/total_capture_bonus:.2f}")
            print(f"  :  collision_penalty  {-capture_bonus * 0.8:.0f}")
        
      
        lidar_safe_dist = stage_reward.get("lidar_safe_distance", self.cfg.reward.lidar_safe_distance)
        print(f"\n  LiDAR:")
        print(f"  - lidar_alpha: {lidar_alpha}")
        print(f"  - lidar_safe_distance: {lidar_safe_dist}m")
        
        if lidar_safe_dist < self.cfg.shield.min_distance_m * 2:
            print(f"  [] lidar_safe_distance ({lidar_safe_dist}m)  shield2 ({self.cfg.shield.min_distance_m * 2}m)")
        
        result = {
            "collision_penalty": collision_penalty,
            "capture_bonus": capture_bonus,
            "potential_gamma": potential_gamma,
            "lidar_alpha": lidar_alpha,
            "collision_capture_ratio": abs(collision_penalty) / capture_bonus if capture_bonus > 0 else 0,
        }
        
        self.results["reward_config"] = result
        return result

    def generate_recommendations(self) -> str:
        """"""
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        recommendations = []
        
      
        if "reaction_time" in self.results:
            rt = self.results["reaction_time"]
            if rt["safety_margin"] < 1.0:
                recommendations.append(
                    f"1.  shield.min_distance_m  {rt['shield_distance']}m  "
                    f"{rt['stopping_distance'] + 2:.1f}m"
                )
        
        if "smoothing_impact" in self.results:
            si = self.results["smoothing_impact"]
            if si["response_ratio"] > 0.3:
                recommendations.append(
                    f"2.  control.smoothing_alpha  {si['smoothing_alpha']}  0.25"
                )
        
        if "reward_config" in self.results:
            rc = self.results["reward_config"]
            if rc["collision_capture_ratio"] < 0.5:
                recommendations.append(
                    f"3.  collision_penalty  {-rc['capture_bonus'] * 0.8:.0f} "
                    f"(: {rc['collision_penalty']})"
                )
        
        if "aggressive_chase" in self.results:
            ac = self.results["aggressive_chase"]
            high_speed_collision = ac.get(8.0, {}).get("collision_rate", 0)
            low_speed_collision = ac.get(3.0, {}).get("collision_rate", 0)
            if high_speed_collision > low_speed_collision * 3:
                recommendations.append(
                    "4. : collision_penalty *= (1 + speed/max_speed)"
                )
        
      
        recommendations.extend([
            "5.  curriculum learning ",
            "6.  lidar_alpha ",
            "7.  stage3  entropy_coef ",
        ])
        
        for rec in recommendations:
            print(f"  {rec}")
        
        return "\n".join(recommendations)

    def run_all_tests(self) -> Dict:
        """"""
        print("="*60)
        print("Stage3 ")
        print("="*60)
        
        self.setup_env(stage_idx=2)
        
        self.test_reward_config_analysis()
        self.test_lidar_reaction_time()
        self.test_action_smoothing_impact(num_steps=100)
        self.test_reward_signal_analysis(num_steps=300)
        self.test_aggressive_chase_collision(num_episodes=3, max_steps=300)
        
        self.generate_recommendations()
        
        return self.results


def main():
    print("...")
    cfg = load_config()
    
    diagnostics = PolicyBehaviorDiagnostics(cfg, num_envs=8)
    results = diagnostics.run_all_tests()
    
    print("\n" + "="*60)
    print("!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
