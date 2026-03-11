"""
Stage3 

 stage3  20% 
 5% 


1. 
2.  safety shield 
3. 
4.  repulse  agent 
5. 
"""

from __future__ import annotations
import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from typing import Dict, List, Tuple
import time

from src.config import load_config, ThreeDConfig
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2
from src.environment.batch_kernels import (
    batch_check_collisions_v2,
    batch_apply_safety_shield,
    NUMBA_AVAILABLE,
)


class Stage3CollisionDiagnostics:
    """Stage3 """
    
    def __init__(self, cfg: ThreeDConfig, num_envs: int = 8):
        self.cfg = cfg
        self.num_envs = num_envs
        self.env = None
        self.results: Dict[str, any] = {}

    def setup_env(self, stage_idx: int = 2) -> None:
        """stage"""
        print(f"[Setup]  (num_envs={self.num_envs})...")
        self.env = VectorizedMutualAStarEnvV2(
            num_envs=self.num_envs,
            cfg=self.cfg,
        )
      
        self.stage_idx = stage_idx
        self.env.reset(stage_index=stage_idx)
        print(f"[Setup]  stage: {self.env.stages[stage_idx].name}")
        print(f"[Setup] Target speed: {self.env.stages[stage_idx].target_speed}")
        print(f"[Setup] View radius: {self.env.stages[stage_idx].view_radius}")
        print(f"[Setup] Target behavior: {self.env.stages[stage_idx].target_behavior}")
        
    def test_1_map_free_space_ratio(self) -> Dict:
        """1: """
        print("\n" + "="*60)
        print("[Test 1] ")
        print("="*60)
        
        occ = self.env.occupancy_grid
        total_voxels = occ.size
        occupied_voxels = np.sum(occ == 1)
        free_voxels = np.sum(occ == 0)
        
        free_ratio = free_voxels / total_voxels
        
      
        occ_astar = self.env.occupancy_grid_astar
        occupied_astar = np.sum(occ_astar == 1)
        free_astar = np.sum(occ_astar == 0)
        free_ratio_astar = free_astar / total_voxels
        
        result = {
            "total_voxels": int(total_voxels),
            "free_voxels_original": int(free_voxels),
            "free_ratio_original": float(free_ratio),
            "free_voxels_inflated": int(free_astar),
            "free_ratio_inflated": float(free_ratio_astar),
            "inflation_loss_ratio": float((free_voxels - free_astar) / free_voxels),
        }
        
        print(f"  : {free_ratio*100:.1f}%")
        print(f"  : {free_ratio_astar*100:.1f}%")
        print(f"  : {result['inflation_loss_ratio']*100:.1f}%")
        
      
        if free_ratio_astar < 0.3:
            print("  [] 30%")
        
        self.results["map_analysis"] = result
        return result

    def test_2_spawn_position_safety(self, num_trials: int = 100) -> Dict:
        """2: """
        print("\n" + "="*60)
        print("[Test 2] ")
        print("="*60)
        
        collision_at_spawn = 0
        near_obstacle_at_spawn = 0
        
        for trial in range(num_trials):
            obs, info = self.env.reset()
            
          
            collisions = batch_check_collisions_v2(
                self.env.pos,
                self.env.occupancy_grid,
                self.env.origin,
                self.env.voxel_size,
            )
            
            if np.any(collisions):
                collision_at_spawn += 1
            
          
            min_lidar = np.min(self.env.lidar_buffer)
            if min_lidar < 0.2:  # 20% of max range
                near_obstacle_at_spawn += 1
        
        result = {
            "num_trials": num_trials,
            "collision_at_spawn_count": collision_at_spawn,
            "collision_at_spawn_rate": collision_at_spawn / num_trials,
            "near_obstacle_at_spawn_count": near_obstacle_at_spawn,
            "near_obstacle_at_spawn_rate": near_obstacle_at_spawn / num_trials,
        }
        
        print(f"  : {collision_at_spawn}/{num_trials} ({result['collision_at_spawn_rate']*100:.1f}%)")
        print(f"  : {near_obstacle_at_spawn}/{num_trials} ({result['near_obstacle_at_spawn_rate']*100:.1f}%)")
        
        if result['collision_at_spawn_rate'] > 0.01:
            print("  [] !")
        
        self.results["spawn_safety"] = result
        return result

    def test_3_random_action_collision_rate(self, num_episodes: int = 10, max_steps: int = 500) -> Dict:
        """3: """
        print("\n" + "="*60)
        print("[Test 3] ")
        print("="*60)
        
        total_steps = 0
        total_collisions = 0
        episode_collision_rates = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            ep_steps = 0
            ep_collisions = 0
            
            for step in range(max_steps):
              
                actions = np.random.uniform(-1, 1, size=(self.num_envs, self.env.num_agents, 4))
                actions[:, :, 0] *= self.cfg.control.action_bounds["vx"]
                actions[:, :, 1] *= self.cfg.control.action_bounds["vy"]
                actions[:, :, 2] *= self.cfg.control.action_bounds["vz"]
                actions[:, :, 3] *= self.cfg.control.action_bounds["yaw_rate"]
                
                obs, rewards, dones, infos = self.env.step(actions)
                
              
                for info in infos:
                    if info.get("collision", False):
                        ep_collisions += 1
                
                ep_steps += self.num_envs
                
                if np.any(dones):
                    break
            
            total_steps += ep_steps
            total_collisions += ep_collisions
            ep_rate = ep_collisions / ep_steps if ep_steps > 0 else 0
            episode_collision_rates.append(ep_rate)
            
        result = {
            "num_episodes": num_episodes,
            "total_steps": total_steps,
            "total_collisions": total_collisions,
            "overall_collision_rate": total_collisions / total_steps if total_steps > 0 else 0,
            "mean_episode_collision_rate": np.mean(episode_collision_rates),
            "std_episode_collision_rate": np.std(episode_collision_rates),
        }
        
        print(f"  : {total_steps}")
        print(f"  : {total_collisions}")
        print(f"  : {result['overall_collision_rate']*100:.2f}%")
        print(f"  ()")
        
        self.results["random_baseline"] = result
        return result

    def test_4_guidance_following_collision_rate(self, num_episodes: int = 10, max_steps: int = 500) -> Dict:
        """4: """
        print("\n" + "="*60)
        print("[Test 4] ")
        print("="*60)
        
        total_steps = 0
        total_collisions = 0
        episode_collision_rates = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            ep_steps = 0
            ep_collisions = 0
            
            for step in range(max_steps):
              
                guidance = self.env.guidance_vectors.copy()  # (B, N, 3)
                
              
                actions = np.zeros((self.num_envs, self.env.num_agents, 4))
                
              
                for b in range(self.num_envs):
                    for n in range(self.env.num_agents):
                        yaw = self.env.yaw[b, n]
                        cy, sy = np.cos(yaw), np.sin(yaw)
                        gx, gy, gz = guidance[b, n]
                        
                      
                        vx_body = gx * cy + gy * sy
                        vy_body = -gx * sy + gy * cy
                        vz_body = gz
                        
                      
                        speed = np.sqrt(vx_body**2 + vy_body**2 + vz_body**2)
                        if speed > 1e-6:
                            scale = min(5.0, speed) / speed  # 5m/s
                            actions[b, n, 0] = vx_body * scale
                            actions[b, n, 1] = vy_body * scale
                            actions[b, n, 2] = vz_body * scale
                        
                      
                        target_yaw = np.arctan2(gy, gx)
                        yaw_error = target_yaw - yaw
                        while yaw_error > np.pi:
                            yaw_error -= 2 * np.pi
                        while yaw_error < -np.pi:
                            yaw_error += 2 * np.pi
                        actions[b, n, 3] = np.clip(yaw_error * 2.0, -0.8, 0.8)
                
                obs, rewards, dones, infos = self.env.step(actions)
                
              
                for info in infos:
                    if info.get("collision", False):
                        ep_collisions += 1
                
                ep_steps += self.num_envs
                
                if np.any(dones):
                    break
            
            total_steps += ep_steps
            total_collisions += ep_collisions
            ep_rate = ep_collisions / ep_steps if ep_steps > 0 else 0
            episode_collision_rates.append(ep_rate)
        
        result = {
            "num_episodes": num_episodes,
            "total_steps": total_steps,
            "total_collisions": total_collisions,
            "overall_collision_rate": total_collisions / total_steps if total_steps > 0 else 0,
            "mean_episode_collision_rate": np.mean(episode_collision_rates),
            "std_episode_collision_rate": np.std(episode_collision_rates),
        }
        
        print(f"  : {total_steps}")
        print(f"  : {total_collisions}")
        print(f"  : {result['overall_collision_rate']*100:.2f}%")
        
        if result['overall_collision_rate'] > 0.05:
            print("  [] >5%")
        else:
            print("  [] <5%")
        
        self.results["guidance_following"] = result
        return result

    def test_5_safety_shield_effectiveness(self, num_episodes: int = 5, max_steps: int = 300) -> Dict:
        """5: """
        print("\n" + "="*60)
        print("[Test 5] ")
        print("="*60)
        
        shield_triggers = 0
        collisions_after_shield = 0
        collisions_without_shield = 0
        total_steps = 0
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            
            for step in range(max_steps):
              
                actions = np.zeros((self.num_envs, self.env.num_agents, 4))
                
              
                lidar = self.env.lidar_buffer  # (B, N, 26)
                for b in range(self.num_envs):
                    for n in range(self.env.num_agents):
                        min_idx = np.argmin(lidar[b, n])
                      
                        actions[b, n, 0] = 6.0  # 
                
              
                prev_pos = self.env.pos.copy()
                
                obs, rewards, dones, infos = self.env.step(actions)
                
              
                for i, info in enumerate(infos):
                    if info.get("shield_triggered", 0) > 0:
                        shield_triggers += info["shield_triggered"]
                    if info.get("collision", False):
                      
                        if info.get("shield_triggered", 0) > 0:
                            collisions_after_shield += 1
                        else:
                            collisions_without_shield += 1
                
                total_steps += self.num_envs
                
                if np.any(dones):
                    break
        
        result = {
            "total_steps": total_steps,
            "shield_triggers": shield_triggers,
            "shield_trigger_rate": shield_triggers / total_steps if total_steps > 0 else 0,
            "collisions_after_shield": collisions_after_shield,
            "collisions_without_shield": collisions_without_shield,
            "shield_prevention_rate": 1 - (collisions_after_shield / max(shield_triggers, 1)),
        }
        
        print(f"  : {total_steps}")
        print(f"  Shield: {shield_triggers}")
        print(f"  Shield: {result['shield_trigger_rate']*100:.2f}%")
        print(f"  Shield: {collisions_after_shield}")
        print(f"  Shield: {result['shield_prevention_rate']*100:.1f}%")
        
        if result['shield_prevention_rate'] < 0.8:
            print("  [] Shield<80%")
        
        self.results["shield_effectiveness"] = result
        return result

    def test_6_target_repulse_trap_analysis(self, num_episodes: int = 10, max_steps: int = 500) -> Dict:
        """6: repulseagent"""
        print("\n" + "="*60)
        print("[Test 6] Repulse")
        print("="*60)
        
        target_near_obstacle_count = 0
        agent_collision_while_chasing = 0
        total_chase_steps = 0
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            
            for step in range(max_steps):
              
                target_pos = self.env.target_pos  # (B, 3)
                for b in range(self.num_envs):
                  
                    tp = target_pos[b]
                    ix = int((tp[0] - self.env.origin[0]) / self.env.voxel_size)
                    iy = int((tp[1] - self.env.origin[1]) / self.env.voxel_size)
                    iz = int((tp[2] - self.env.origin[2]) / self.env.voxel_size)
                    
                  
                    near_obstacle = False
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            for dz in range(-2, 3):
                                nx, ny, nz = ix + dx, iy + dy, iz + dz
                                if 0 <= nx < self.env.occupancy_grid.shape[0] and \
                                   0 <= ny < self.env.occupancy_grid.shape[1] and \
                                   0 <= nz < self.env.occupancy_grid.shape[2]:
                                    if self.env.occupancy_grid[nx, ny, nz] == 1:
                                        near_obstacle = True
                                        break
                            if near_obstacle:
                                break
                        if near_obstacle:
                            break
                    
                    if near_obstacle:
                        target_near_obstacle_count += 1
                
              
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
                            vx_body = (dx * cy + dy * sy) / dist * 6.0
                            vy_body = (-dx * sy + dy * cy) / dist * 2.0
                            vz_body = dz / dist * 2.0
                            
                            actions[b, n, 0] = vx_body
                            actions[b, n, 1] = vy_body
                            actions[b, n, 2] = vz_body
                
                obs, rewards, dones, infos = self.env.step(actions)
                
                for info in infos:
                    if info.get("collision", False):
                        agent_collision_while_chasing += 1
                
                total_chase_steps += self.num_envs
                
                if np.any(dones):
                    break
        
        result = {
            "total_chase_steps": total_chase_steps,
            "target_near_obstacle_count": target_near_obstacle_count,
            "target_near_obstacle_rate": target_near_obstacle_count / total_chase_steps if total_chase_steps > 0 else 0,
            "collision_while_chasing": agent_collision_while_chasing,
            "collision_rate_while_chasing": agent_collision_while_chasing / total_chase_steps if total_chase_steps > 0 else 0,
        }
        
        print(f"  : {total_chase_steps}")
        print(f"  : {target_near_obstacle_count}")
        print(f"  : {result['target_near_obstacle_rate']*100:.2f}%")
        print(f"  : {agent_collision_while_chasing}")
        print(f"  : {result['collision_rate_while_chasing']*100:.2f}%")
        
        if result['target_near_obstacle_rate'] > 0.3:
            print("  [] (>30%)repulseagent")
        
        self.results["repulse_trap"] = result
        return result

    def test_7_collision_detection_sensitivity(self) -> Dict:
        """7: """
        print("\n" + "="*60)
        print("[Test 7] ")
        print("="*60)
        
      
        occ = self.env.occupancy_grid
        origin = self.env.origin
        voxel_size = self.env.voxel_size
        
      
        free_indices = np.argwhere(occ == 0)
        
        if len(free_indices) == 0:
            print("  [] !")
            return {"error": "no_free_voxels"}
        
      
        num_samples = min(1000, len(free_indices))
        sample_indices = free_indices[np.random.choice(len(free_indices), num_samples, replace=False)]
        
        false_positive_count = 0
        edge_collision_count = 0
        
        for idx in sample_indices:
          
            center = origin + (idx + 0.5) * voxel_size
            
          
            pos = np.array([[[center[0], center[1], center[2]]]], dtype=np.float64)
            collision = batch_check_collisions_v2(pos, occ, origin, voxel_size)
            
            if collision[0, 0]:
                false_positive_count += 1
            
          
            for offset in [0.45, -0.45]:
                for axis in range(3):
                    edge_pos = center.copy()
                    edge_pos[axis] += offset * voxel_size
                    pos = np.array([[[edge_pos[0], edge_pos[1], edge_pos[2]]]], dtype=np.float64)
                    collision = batch_check_collisions_v2(pos, occ, origin, voxel_size)
                    if collision[0, 0]:
                        edge_collision_count += 1
        
        result = {
            "num_samples": num_samples,
            "false_positive_at_center": false_positive_count,
            "false_positive_rate": false_positive_count / num_samples,
            "edge_collision_count": edge_collision_count,
            "edge_collision_rate": edge_collision_count / (num_samples * 6),
            "voxel_size": float(voxel_size),
        }
        
        print(f"  : {num_samples}")
        print(f"  : {voxel_size}m")
        print(f"  : {false_positive_count} ({result['false_positive_rate']*100:.2f}%)")
        print(f"  : {edge_collision_count} ({result['edge_collision_rate']*100:.2f}%)")
        
        if result['false_positive_rate'] > 0.01:
            print("  [] !")
        
        self.results["detection_sensitivity"] = result
        return result

    def test_8_optimal_policy_lower_bound(self, num_episodes: int = 10, max_steps: int = 1000) -> Dict:
        """8: A*"""
        print("\n" + "="*60)
        print("[Test 8] A*")
        print("="*60)
        
        total_steps = 0
        total_collisions = 0
        captures = 0
        episode_stats = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            ep_steps = 0
            ep_collisions = 0
            ep_captures = 0
            
            for step in range(max_steps):
              
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
                        
                      
                        vx_body = (gx * cy + gy * sy) * 5.0
                        vy_body = (-gx * sy + gy * cy) * 2.0
                        vz_body = gz * 2.0
                        
                        actions[b, n, 0] = vx_body
                        actions[b, n, 1] = vy_body
                        actions[b, n, 2] = vz_body
                        
                      
                        target_yaw = np.arctan2(gy, gx)
                        yaw_error = target_yaw - yaw
                        while yaw_error > np.pi:
                            yaw_error -= 2 * np.pi
                        while yaw_error < -np.pi:
                            yaw_error += 2 * np.pi
                        actions[b, n, 3] = np.clip(yaw_error * 1.5, -0.8, 0.8)
                
                obs, rewards, dones, infos = self.env.step(actions)
                
                for info in infos:
                    if info.get("collision", False):
                        ep_collisions += 1
                    if info.get("captured", False):
                        ep_captures += 1
                
                ep_steps += self.num_envs
                
                if np.any(dones):
                    break
            
            total_steps += ep_steps
            total_collisions += ep_collisions
            captures += ep_captures
            
            episode_stats.append({
                "steps": ep_steps,
                "collisions": ep_collisions,
                "captures": ep_captures,
                "collision_rate": ep_collisions / ep_steps if ep_steps > 0 else 0,
            })
        
        result = {
            "num_episodes": num_episodes,
            "total_steps": total_steps,
            "total_collisions": total_collisions,
            "total_captures": captures,
            "overall_collision_rate": total_collisions / total_steps if total_steps > 0 else 0,
            "capture_rate": captures / (num_episodes * self.num_envs),
            "mean_collision_rate": np.mean([s["collision_rate"] for s in episode_stats]),
            "std_collision_rate": np.std([s["collision_rate"] for s in episode_stats]),
        }
        
        print(f"  : {total_steps}")
        print(f"  : {total_collisions}")
        print(f"  A*: {result['overall_collision_rate']*100:.2f}%")
        print(f"  : {result['capture_rate']*100:.1f}%")
        print(f"  : {result['std_collision_rate']*100:.2f}%")
        
        if result['overall_collision_rate'] < 0.05:
            print("  [] ! A*<5%")
        elif result['overall_collision_rate'] < 0.15:
            print("  [] ")
        else:
            print("  [] >15%")
        
        self.results["optimal_policy"] = result
        return result

    def test_9_lidar_coverage_analysis(self) -> Dict:
        """9: LiDAR"""
        print("\n" + "="*60)
        print("[Test 9] LiDAR")
        print("="*60)
        
        obs, info = self.env.reset()
        
      
        lidar_range = self.env.lidar_max_range
        num_rays = 26  # 26
        
      
        lidar = self.env.lidar_buffer  # (B, N, 26)
        
      
        mean_distances = np.mean(lidar, axis=(0, 1)) * lidar_range
        min_distances = np.min(lidar, axis=(0, 1)) * lidar_range
        
      
        blind_spots = np.sum(mean_distances > lidar_range * 0.95)
        
        result = {
            "lidar_range": float(lidar_range),
            "num_rays": num_rays,
            "mean_distance_per_ray": mean_distances.tolist(),
            "min_distance_per_ray": min_distances.tolist(),
            "blind_spot_count": int(blind_spots),
            "shield_min_distance": float(self.cfg.shield.min_distance_m),
        }
        
        print(f"  LiDAR: {lidar_range}m")
        print(f"  : {num_rays}")
        print(f"  Shield: {self.cfg.shield.min_distance_m}m")
        print(f"  : {blind_spots}")
        
      
        reaction_time = 0.1  # 100ms
        max_speed = self.cfg.control.action_bounds["vx"]
        min_safe_dist = max_speed * reaction_time * 2  # 
        
        if self.cfg.shield.min_distance_m < min_safe_dist:
            print(f"  [] Shield({self.cfg.shield.min_distance_m}m) < ({min_safe_dist:.1f}m)")
        
        self.results["lidar_analysis"] = result
        return result

    def generate_report(self) -> str:
        """"""
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        report = []
        report.append("# Stage3 \n")
        
      
        if "optimal_policy" in self.results:
            opt = self.results["optimal_policy"]
            if opt["overall_collision_rate"] < 0.05:
                report.append("## : \n")
                report.append(f"- A*: {opt['overall_collision_rate']*100:.2f}% (< 5%)\n")
                report.append("- : \n")
            else:
                report.append("## : \n")
                report.append(f"- A*: {opt['overall_collision_rate']*100:.2f}%\n")
        
      
        report.append("\n## \n")
        
        if "spawn_safety" in self.results:
            spawn = self.results["spawn_safety"]
            if spawn["collision_at_spawn_rate"] > 0.01:
                report.append(f"- [] : {spawn['collision_at_spawn_rate']*100:.1f}%\n")
        
        if "repulse_trap" in self.results:
            trap = self.results["repulse_trap"]
            if trap["target_near_obstacle_rate"] > 0.3:
                report.append(f"- [] : {trap['target_near_obstacle_rate']*100:.1f}%\n")
                report.append("  - : target repulse\n")
        
        if "shield_effectiveness" in self.results:
            shield = self.results["shield_effectiveness"]
            if shield["shield_prevention_rate"] < 0.8:
                report.append(f"- [] Shield: {shield['shield_prevention_rate']*100:.1f}%\n")
                report.append("  - : shield\n")
        
        if "map_analysis" in self.results:
            map_info = self.results["map_analysis"]
            if map_info["inflation_loss_ratio"] > 0.3:
                report.append(f"- [] : {map_info['inflation_loss_ratio']*100:.1f}%\n")
                report.append("  - : \n")
        
      
        report.append("\n## \n")
        report.append("1. A*<5%\n")
        report.append("2. agent\n")
        report.append("3. curriculum learning\n")
        report.append("4. action smoothing\n")
        
        report_str = "".join(report)
        print(report_str)
        
        return report_str

    def run_all_tests(self) -> Dict:
        """"""
        print("="*60)
        print("Stage3 ")
        print(": 20%")
        print(": 5%")
        print("="*60)
        
        self.setup_env(stage_idx=2)  # stage3
        
      
        self.test_1_map_free_space_ratio()
        self.test_2_spawn_position_safety(num_trials=50)
        self.test_7_collision_detection_sensitivity()
        self.test_9_lidar_coverage_analysis()
        self.test_3_random_action_collision_rate(num_episodes=5, max_steps=300)
        self.test_4_guidance_following_collision_rate(num_episodes=5, max_steps=300)
        self.test_5_safety_shield_effectiveness(num_episodes=3, max_steps=200)
        self.test_6_target_repulse_trap_analysis(num_episodes=5, max_steps=300)
        self.test_8_optimal_policy_lower_bound(num_episodes=5, max_steps=500)
        
      
        self.generate_report()
        
        return self.results


def main():
    """"""
    print("...")
    cfg = load_config()
    
    print(f"Numba: {NUMBA_AVAILABLE}")
    
  
    diagnostics = Stage3CollisionDiagnostics(cfg, num_envs=8)
    
  
    results = diagnostics.run_all_tests()
    
    print("\n" + "="*60)
    print("!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
