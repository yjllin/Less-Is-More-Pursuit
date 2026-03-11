"""


"""

import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from src.config import load_config
from src.navigation import FrontierAllocator, VoxelMap3D


def create_realistic_voxel_map(cfg):
    """"""
    world_size = tuple(cfg.environment.world_size_m)
    voxel_size = cfg.environment.voxel_size_m
    
    voxel_map = VoxelMap3D(
        world_size_m=world_size,
        voxel_size_m=voxel_size,
        occupancy_threshold=cfg.navigation.occupancy_threshold,
        origin=(0.0, 0.0, 0.0)
    )
    voxel_map.visits = np.zeros(voxel_map.shape, dtype=np.float32)
    return voxel_map


def mark_explored_around_position(voxel_map, pos, radius_voxels=5):
    """"""
    idx = voxel_map.pos_to_idx(pos)
    for dx in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dz in range(-radius_voxels // 2, radius_voxels // 2 + 1):
                x, y, z = idx[0] + dx, idx[1] + dy, idx[2] + dz
                if 0 <= x < voxel_map.shape[0] and 0 <= y < voxel_map.shape[1] and 0 <= z < voxel_map.shape[2]:
                    voxel_map.visits[x, y, z] = 1.0


def test_multi_step_exploration():
    """
    
    Agent
    """
    print("\n" + "=" * 60)
    print(": ")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_realistic_voxel_map(cfg)
    
  
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
  
    occupancy[0:2, :, :] = 1
    occupancy[-2:, :, :] = 1
    occupancy[:, 0:2, :] = 1
    occupancy[:, -2:, :] = 1
    
  
    voxel_map.visits[occupancy == 1] = 1.0
    
    allowed_mask = (occupancy == 0).astype(np.int8)
    
    allocator = FrontierAllocator(
        fps_count=cfg.navigation.frontier_fps_count,
        cosine_margin=cfg.navigation.frontier_cosine_margin,
        vertical_weight=cfg.navigation.frontier_vertical_weight,
        lock_steps=cfg.navigation.frontier_lock_steps,
        angle_min_deg=cfg.navigation.frontier_angle_min_deg,
        sweep_trigger_m=cfg.navigation.frontier_sweep_trigger_m,
        sweep_radius_m=cfg.navigation.frontier_sweep_radius_m,
        sweep_max_steps=cfg.navigation.frontier_sweep_max_steps,
        confidence_threshold=cfg.environment.confidence_threshold,
        allowed_mask=allowed_mask,
    )
    allocator.set_occupancy_grid(occupancy)
    
  
    center = np.array([
        cfg.environment.world_size_m[0] / 2,
        cfg.environment.world_size_m[1] / 2,
        cfg.environment.world_size_m[2] / 3
    ])
    
    agent_positions = np.array([
        center + np.array([-20, -20, 0]),
        center + np.array([20, -20, 0]),
        center + np.array([20, 20, 0]),
        center + np.array([-20, 20, 0]),
    ], dtype=np.float64)
    
    agent_headings = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ], dtype=np.float64)
    agent_headings = agent_headings / np.linalg.norm(agent_headings, axis=1, keepdims=True)
    
  
    for pos in agent_positions:
        mark_explored_around_position(voxel_map, pos, radius_voxels=3)
    
    confidence_decay = cfg.environment.confidence_decay
    num_steps = 50
    min_dispersions = []
    
    print(f"\n {num_steps} ...")
    
    for step in range(num_steps):
      
        free_mask = occupancy == 0
        voxel_map.visits[free_mask] *= confidence_decay
        
      
        allocator._locks.clear()
        
      
        assignments = allocator.allocate(agent_positions, agent_headings, voxel_map, current_step=step)
        
      
        assigned_points = [assignments[i] for i in range(4) if assignments[i] is not None]
        
        if len(assigned_points) >= 2:
            min_dist = float('inf')
            for i in range(len(assigned_points)):
                for j in range(i + 1, len(assigned_points)):
                    dist = np.linalg.norm(assigned_points[i] - assigned_points[j])
                    min_dist = min(min_dist, dist)
            min_dispersions.append(min_dist)
            
          
            if min_dist < 10.0:
                print(f"  [] Step {step}:  ({min_dist:.1f}m)")
                for idx, pt in enumerate(assigned_points):
                    print(f"    Agent {idx}: {pt[:2]}")
        
      
        for idx in range(4):
            if assignments[idx] is not None:
                direction = assignments[idx] - agent_positions[idx]
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                  
                    move_dist = min(10.0, dist)
                    agent_positions[idx] += direction / dist * move_dist
                  
                    agent_headings[idx] = direction / dist
                  
                    mark_explored_around_position(voxel_map, agent_positions[idx], radius_voxels=3)
        
        if step % 10 == 0:
            explored_ratio = np.sum(voxel_map.visits > cfg.environment.confidence_threshold) / np.sum(allowed_mask)
            print(f"  Step {step}: ={explored_ratio:.1%}, ={min_dispersions[-1] if min_dispersions else 0:.1f}m")
    
  
    if min_dispersions:
        avg_dispersion = np.mean(min_dispersions)
        min_dispersion = np.min(min_dispersions)
        print(f"\n:")
        print(f"  : {avg_dispersion:.1f}m")
        print(f"  : {min_dispersion:.1f}m")
        
      
        assert avg_dispersion > 15.0, f"{avg_dispersion:.1f}m < 15m"
        print(f"\n ")
    else:
        print("")


def test_no_repeated_exploration():
    """
    
    """
    print("\n" + "=" * 60)
    print(": ")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_realistic_voxel_map(cfg)
    
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
    allowed_mask = (occupancy == 0).astype(np.int8)
    
  
    test_region = (10, 10, 5)
    voxel_map.visits[test_region[0]-2:test_region[0]+3, 
                    test_region[1]-2:test_region[1]+3,
                    test_region[2]-1:test_region[2]+2] = 1.0
    
    allocator = FrontierAllocator(
        fps_count=cfg.navigation.frontier_fps_count,
        cosine_margin=cfg.navigation.frontier_cosine_margin,
        vertical_weight=cfg.navigation.frontier_vertical_weight,
        lock_steps=cfg.navigation.frontier_lock_steps,
        angle_min_deg=cfg.navigation.frontier_angle_min_deg,
        sweep_trigger_m=cfg.navigation.frontier_sweep_trigger_m,
        sweep_radius_m=cfg.navigation.frontier_sweep_radius_m,
        sweep_max_steps=cfg.navigation.frontier_sweep_max_steps,
        confidence_threshold=cfg.environment.confidence_threshold,
        allowed_mask=allowed_mask,
    )
    allocator.set_occupancy_grid(occupancy)
    
    confidence_decay = cfg.environment.confidence_decay
    threshold = cfg.environment.confidence_threshold
    
  
    steps_until_frontier = 0
    for step in range(3000):
      
        free_mask = occupancy == 0
        voxel_map.visits[free_mask] *= confidence_decay
        
      
        region_confidence = voxel_map.visits[test_region]
        
        if region_confidence < threshold:
            steps_until_frontier = step
            break
    
    print(f" {steps_until_frontier} ")
    print(f"10Hz = {steps_until_frontier / 10:.1f} = {steps_until_frontier / 10 / 60:.1f}")
    
  
    assert steps_until_frontier >= 1000, f"{steps_until_frontier}"
    
    print(f"\n ")


if __name__ == "__main__":
    print("=" * 60)
    print("")
    print("=" * 60)
    
    try:
        test_no_repeated_exploration()
        test_multi_step_exploration()
        
        print("\n" + "=" * 60)
        print("")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
