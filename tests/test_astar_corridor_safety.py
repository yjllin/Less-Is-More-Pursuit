"""
A* 

:
1. 
2.  A*  6m 
3.  6.57m

:
    python tests/test_astar_corridor_safety.py
"""

import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.navigation.astar3d import AStar3D


@dataclass
class CorridorTestConfig:
    """"""
  
    occupancy_path: str = "artifacts/level_occupancy.npy"
    
  
    corridor_radius_m: float = 6.0  # A* 
    
  
    braking_distance_m: float = 6.57  # 
    max_speed_m_s: float = 8.0
    
  
    lidar_max_range_m: float = 200.0  #  config.yaml stage1
    lidar_safe_distance_ratio: float = 4.0 / 200.0  # p_lidar_d / lidar_max_range
    
  
    num_test_paths: int = 50  # 
    sample_spacing_m: float = 1.0  # 
    
  
    inflate_min_vox: int = 0
    inflate_max_vox: int = 5


def load_occupancy_grid(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """"""
    npy_path = Path(path)
    if not npy_path.exists():
        raise FileNotFoundError(f": {path}")
    
    occupancy = np.load(npy_path).astype(np.int8)
    
  
    meta_path = npy_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        voxel_size = float(meta.get("voxel_size", 6.0))
    else:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        voxel_size = 6.0
    
    print(f"[] : {occupancy.shape}, : {voxel_size}m, : {origin}")
    occupied_count = np.sum(occupancy == 1)
    total_count = occupancy.size
    print(f"[] : {occupied_count}/{total_count} ({100*occupied_count/total_count:.2f}%)")
    
    return occupancy, origin, voxel_size


def inflate_occupancy_grid(occupancy: np.ndarray, radius_vox: int) -> np.ndarray:
    """
     _inflate_occupancy_grid 
    
    """
    if radius_vox <= 0:
        return occupancy.copy()
    
    gx, gy, gz = occupancy.shape
    inflated = occupancy.copy()
    
  
    offsets = []
    r2 = radius_vox * radius_vox
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    offsets.append((dx, dy, dz))
    
  
    occupied_indices = np.argwhere(occupancy == 1)
    for x, y, z in occupied_indices:
        for dx, dy, dz in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz:
                inflated[nx, ny, nz] = 1
    
    return inflated


def simulate_lidar_at_point(
    pos: np.ndarray,
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    max_range: float,
) -> np.ndarray:
    """
     26  batch_kernels._simulate_lidar_single 
     (26,)
    """
    gx, gy, gz = occupancy_grid.shape
    distances = np.ones(26, dtype=np.float64)
    step_size = min(voxel_size, 2.0)
    num_steps = int(max_range / step_size)
    
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                
                dir_len = math.sqrt(dx*dx + dy*dy + dz*dz)
                dir_x, dir_y, dir_z = dx/dir_len, dy/dir_len, dz/dir_len
                
                for step in range(1, num_steps + 1):
                    dist = step * step_size
                    px = pos[0] + dir_x * dist
                    py = pos[1] + dir_y * dist
                    pz = pos[2] + dir_z * dist
                    
                    ix = int(np.floor((px - origin[0]) / voxel_size))
                    iy = int(np.floor((py - origin[1]) / voxel_size))
                    iz = int(np.floor((pz - origin[2]) / voxel_size))
                    
                    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                        distances[idx] = dist / max_range
                        break
                    
                    if occupancy_grid[ix, iy, iz] == 1:
                        distances[idx] = dist / max_range
                        break
                
                idx += 1
    
    return distances


def check_collision_at_point(
    pos: np.ndarray,
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> bool:
    """"""
    gx, gy, gz = occupancy_grid.shape
    ix = int(np.floor((pos[0] - origin[0]) / voxel_size))
    iy = int(np.floor((pos[1] - origin[1]) / voxel_size))
    iz = int(np.floor((pos[2] - origin[2]) / voxel_size))
    
    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
        return True
    return occupancy_grid[ix, iy, iz] == 1


def generate_random_valid_point(
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    rng: np.random.Generator,
    z_range: Tuple[float, float] = (20.0, 90.0),
) -> Optional[np.ndarray]:
    """"""
    gx, gy, gz = occupancy_grid.shape
    world_max = origin + np.array([gx, gy, gz]) * voxel_size
    
    for _ in range(1000):
        pos = np.array([
            rng.uniform(origin[0] + 20, world_max[0] - 20),
            rng.uniform(origin[1] + 20, world_max[1] - 20),
            rng.uniform(z_range[0], z_range[1]),
        ])
        
        if not check_collision_at_point(pos, occupancy_grid, origin, voxel_size):
            return pos
    
    return None


def sample_corridor_points(
    path: List[np.ndarray],
    corridor_radius: float,
    sample_spacing: float,
    num_radial_samples: int = 8,
) -> List[np.ndarray]:
    """
    
    
    """
    if len(path) < 2:
        return path
    
    corridor_points = []
    
  
    accumulated_dist = 0.0
    prev_point = path[0]
    corridor_points.append(prev_point.copy())
    
    for i in range(1, len(path)):
        curr_point = path[i]
        segment = curr_point - prev_point
        segment_len = np.linalg.norm(segment)
        
        if segment_len < 1e-6:
            continue
        
        segment_dir = segment / segment_len
        
      
        dist_along = 0.0
        while dist_along < segment_len:
            point_on_path = prev_point + segment_dir * dist_along
            
          
            corridor_points.append(point_on_path.copy())
            
          
          
            if abs(segment_dir[2]) < 0.9:
                perp1 = np.cross(segment_dir, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(segment_dir, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(segment_dir, perp1)
            
            for j in range(num_radial_samples):
                angle = 2 * math.pi * j / num_radial_samples
                offset = corridor_radius * (math.cos(angle) * perp1 + math.sin(angle) * perp2)
                corridor_points.append(point_on_path + offset)
            
            dist_along += sample_spacing
        
        prev_point = curr_point
    
    return corridor_points


class CorridorSafetyTester:
    """A* """
    
    def __init__(self, config: CorridorTestConfig = None):
        self.config = config or CorridorTestConfig()
        self.occupancy_raw = None
        self.origin = None
        self.voxel_size = None
        self.rng = np.random.default_rng(42)
    
    def load_map(self):
        """"""
        self.occupancy_raw, self.origin, self.voxel_size = load_occupancy_grid(
            self.config.occupancy_path
        )
    
    def test_single_inflation(
        self,
        inflate_vox: int,
        verbose: bool = True,
    ) -> dict:
        """
        
        
        :
            results: 
        """
        cfg = self.config
        
      
        if verbose:
            print(f"\n[] : {inflate_vox} voxels ({inflate_vox * self.voxel_size:.1f}m)")
        
        occupancy_inflated = inflate_occupancy_grid(self.occupancy_raw, inflate_vox)
        
      
        navigator = AStar3D(
            grid_shape=occupancy_inflated.shape,
            voxel_size=self.voxel_size,
            cache_refresh_steps=2,
            lookahead_m=8.0,
            heuristic_weight=1.5,
            origin=self.origin,
        )
        navigator.update_grid(occupancy_inflated)
        
      
        total_corridor_points = 0
        collision_count = 0
        lidar_warning_count = 0
        min_lidar_distances = []
        
        successful_paths = 0
        
        for path_idx in range(cfg.num_test_paths):
          
            start = generate_random_valid_point(
                occupancy_inflated, self.origin, self.voxel_size, self.rng
            )
            goal = generate_random_valid_point(
                occupancy_inflated, self.origin, self.voxel_size, self.rng
            )
            
            if start is None or goal is None:
                continue
            
          
            if np.linalg.norm(goal - start) < 50.0:
                continue
            
          
            try:
                direction, world_path, path_length = navigator.compute_direction(
                    start, goal, current_step=0
                )
            except Exception:
                continue
            
            if len(world_path) < 2:
                continue
            
            successful_paths += 1
            
          
            corridor_points = sample_corridor_points(
                world_path,
                cfg.corridor_radius_m,
                cfg.sample_spacing_m,
            )
            
          
            for point in corridor_points:
                total_corridor_points += 1
                
              
                if check_collision_at_point(point, self.occupancy_raw, self.origin, self.voxel_size):
                    collision_count += 1
                    continue
                
              
                lidar = simulate_lidar_at_point(
                    point, self.occupancy_raw, self.origin, self.voxel_size, cfg.lidar_max_range_m
                )
                min_lidar = np.min(lidar)
                min_dist_m = min_lidar * cfg.lidar_max_range_m
                min_lidar_distances.append(min_dist_m)
                
              
                safe_dist_m = cfg.lidar_safe_distance_ratio * cfg.lidar_max_range_m
                if min_dist_m < safe_dist_m:
                    lidar_warning_count += 1
        
      
        collision_rate = collision_count / max(total_corridor_points, 1)
        lidar_warning_rate = lidar_warning_count / max(total_corridor_points, 1)
        
        results = {
            "inflate_vox": inflate_vox,
            "inflate_m": inflate_vox * self.voxel_size,
            "successful_paths": successful_paths,
            "total_corridor_points": total_corridor_points,
            "collision_count": collision_count,
            "collision_rate": collision_rate,
            "lidar_warning_count": lidar_warning_count,
            "lidar_warning_rate": lidar_warning_rate,
            "min_lidar_dist_mean": np.mean(min_lidar_distances) if min_lidar_distances else 0,
            "min_lidar_dist_min": np.min(min_lidar_distances) if min_lidar_distances else 0,
            "min_lidar_dist_p5": np.percentile(min_lidar_distances, 5) if min_lidar_distances else 0,
        }
        
        if verbose:
            print(f"  : {successful_paths}/{cfg.num_test_paths}")
            print(f"  : {total_corridor_points}")
            print(f"  : {100*collision_rate:.2f}% ({collision_count})")
            print(f"  : {100*lidar_warning_rate:.2f}% ({lidar_warning_count})")
            print(f"  : mean={results['min_lidar_dist_mean']:.2f}m, "
                  f"min={results['min_lidar_dist_min']:.2f}m, p5={results['min_lidar_dist_p5']:.2f}m")
        
        return results

    
    def find_minimum_safe_inflation(self) -> dict:
        """
        
        
        :
        1.  = 0%
        2. 
        """
        cfg = self.config
        
        print("\n" + "="*60)
        print("")
        print("="*60)
        print(f": {cfg.corridor_radius_m}m")
        print(f": {cfg.braking_distance_m}m")
        print(f": {self.voxel_size}m")
        
      
      
        safety_margin = cfg.braking_distance_m * 0.5  # 50% 
        theoretical_min_m = cfg.corridor_radius_m + safety_margin
        theoretical_min_vox = int(np.ceil(theoretical_min_m / self.voxel_size))
        
        print(f": {theoretical_min_m:.2f}m ({theoretical_min_vox} voxels)")
        
        all_results = []
        best_result = None
        
        for inflate_vox in range(cfg.inflate_min_vox, cfg.inflate_max_vox + 1):
            result = self.test_single_inflation(inflate_vox, verbose=True)
            all_results.append(result)
            
          
            is_safe = (
                result["collision_rate"] == 0.0 and
                result["min_lidar_dist_p5"] >= cfg.braking_distance_m
            )
            
            result["is_safe"] = is_safe
            
            if is_safe and best_result is None:
                best_result = result
                print(f"   !")
        
      
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        if best_result:
            print(f": {best_result['inflate_vox']} voxels ({best_result['inflate_m']:.1f}m)")
            print(f"  - : {100*best_result['collision_rate']:.2f}%")
            print(f"  - : {100*best_result['lidar_warning_rate']:.2f}%")
            print(f"  -  P5: {best_result['min_lidar_dist_p5']:.2f}m")
        else:
            print(": !")
            print(" inflate_max_vox  corridor_radius_m")
        
      
        print("\n:")
        print("-" * 80)
        print(f"{'(vox)':<10} {'(m)':<10} {'':<12} {'':<12} {'P5(m)':<12} {'':<6}")
        print("-" * 80)
        for r in all_results:
            safe_mark = "" if r.get("is_safe", False) else ""
            print(f"{r['inflate_vox']:<10} {r['inflate_m']:<10.1f} "
                  f"{100*r['collision_rate']:<12.2f} {100*r['lidar_warning_rate']:<12.2f} "
                  f"{r['min_lidar_dist_p5']:<12.2f} {safe_mark:<6}")
        
        return {
            "all_results": all_results,
            "best_result": best_result,
            "theoretical_min_vox": theoretical_min_vox,
        }
    
    def analyze_braking_safety(self) -> dict:
        """
        
        
        :
        1. Agent 
        2. 
        """
        cfg = self.config
        
        print("\n" + "="*60)
        print("")
        print("="*60)
        
      
        current_inflate_vox = 1  #  vectorized_env_v2.py: int(np.ceil(6.0 / voxel_size))
        current_inflate_m = current_inflate_vox * self.voxel_size
        
        print(f": {current_inflate_vox} voxels ({current_inflate_m:.1f}m)")
        print(f": {cfg.corridor_radius_m}m")
        print(f": {cfg.braking_distance_m}m")
        print(f": {cfg.max_speed_m_s}m/s")
        
      
      
        worst_case_clearance = current_inflate_m - cfg.corridor_radius_m
        
        print(f"\n:")
        print(f"  : {worst_case_clearance:.2f}m")
        print(f"  : {cfg.braking_distance_m:.2f}m")
        
        if worst_case_clearance >= cfg.braking_distance_m:
            print(f"   : ")
            is_safe = True
        else:
            deficit = cfg.braking_distance_m - worst_case_clearance
            print(f"   :  {deficit:.2f}m ")
            is_safe = False
        
      
        recommended_inflate_m = cfg.corridor_radius_m + cfg.braking_distance_m
        recommended_inflate_vox = int(np.ceil(recommended_inflate_m / self.voxel_size))
        
        print(f"\n:")
        print(f"  : {recommended_inflate_m:.2f}m ({recommended_inflate_vox} voxels)")
        
      
      
        filter_delay_distance = 0.25 * cfg.max_speed_m_s
        conservative_inflate_m = recommended_inflate_m + filter_delay_distance
        conservative_inflate_vox = int(np.ceil(conservative_inflate_m / self.voxel_size))
        
        print(f"   (): {conservative_inflate_m:.2f}m ({conservative_inflate_vox} voxels)")
        
        return {
            "current_inflate_vox": current_inflate_vox,
            "current_inflate_m": current_inflate_m,
            "worst_case_clearance": worst_case_clearance,
            "is_safe": is_safe,
            "recommended_inflate_vox": recommended_inflate_vox,
            "recommended_inflate_m": recommended_inflate_m,
            "conservative_inflate_vox": conservative_inflate_vox,
            "conservative_inflate_m": conservative_inflate_m,
        }


def run_full_analysis():
    """"""
    config = CorridorTestConfig(
        corridor_radius_m=6.0,
        braking_distance_m=6.57,
        num_test_paths=30,
        inflate_min_vox=0,
        inflate_max_vox=4,
    )
    
    tester = CorridorSafetyTester(config)
    tester.load_map()
    
  
    braking_results = tester.analyze_braking_safety()
    
  
    inflation_results = tester.find_minimum_safe_inflation()
    
  
    print("\n" + "="*60)
    print("")
    print("="*60)
    
    if inflation_results["best_result"]:
        best = inflation_results["best_result"]
        print(f"1.  A* : {best['inflate_vox']} voxels ({best['inflate_m']:.1f}m)")
    else:
        print(f"1. : {braking_results['conservative_inflate_vox']} voxels")
    
    print(f"2.  config.yaml  shield.min_distance_m = 4.0m")
    print(f"   : {config.braking_distance_m:.1f}m ()")
    
    print(f"3.  smoothing_alpha : 0.5-0.6")
    print(f"   ()")
    
    return {
        "braking_results": braking_results,
        "inflation_results": inflation_results,
    }


if __name__ == "__main__":
    results = run_full_analysis()
