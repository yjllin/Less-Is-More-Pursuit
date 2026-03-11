"""

AirSim
"""

import sys
import os
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pytest
from src.config import load_config
from src.navigation import FrontierAllocator, VoxelMap3D


def create_test_voxel_map(world_size=(100.0, 100.0, 50.0), voxel_size=2.0):
    """"""
    voxel_map = VoxelMap3D(
        world_size_m=world_size,
        voxel_size_m=voxel_size,
        occupancy_threshold=0.3,
        origin=(0.0, 0.0, 0.0)
    )
  
    voxel_map.visits = np.zeros(voxel_map.shape, dtype=np.float32)
    return voxel_map


def test_frontier_dispersion_at_episode_start():
    """
    EpisodeAgent
    
    EpisodeAgent
    Agent
    """
    print("\n" + "=" * 60)
    print(": Episode")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_test_voxel_map()
    
  
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
  
    cx, cy, cz = voxel_map.shape[0] // 2, voxel_map.shape[1] // 2, voxel_map.shape[2] // 2
    occupancy[cx-2:cx+2, cy-2:cy+2, cz-2:cz+2] = 1
    
  
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
    
  
    spawn_offset = 5.0
    center = np.array([50.0, 50.0, 25.0])
    agent_positions = np.array([
        center + np.array([-spawn_offset, -spawn_offset, 0]),
        center + np.array([spawn_offset, -spawn_offset, 0]),
        center + np.array([spawn_offset, spawn_offset, 0]),
        center + np.array([-spawn_offset, spawn_offset, 0]),
    ], dtype=np.float64)
    
  
    agent_headings = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ], dtype=np.float64)
  
    agent_headings = agent_headings / np.linalg.norm(agent_headings, axis=1, keepdims=True)
    
  
    for pos in agent_positions:
        idx = voxel_map.pos_to_idx(pos)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-2, 3):
                    x, y, z = idx[0] + dx, idx[1] + dy, idx[2] + dz
                    if 0 <= x < voxel_map.shape[0] and 0 <= y < voxel_map.shape[1] and 0 <= z < voxel_map.shape[2]:
                        voxel_map.visits[x, y, z] = 1.0
    
  
    assignments = allocator.allocate(agent_positions, agent_headings, voxel_map, current_step=0)
    
  
    assigned_points = []
    for idx, point in assignments.items():
        if point is not None:
            assigned_points.append(point)
            dist = np.linalg.norm(point - agent_positions[idx])
            print(f"Agent {idx}: ={agent_positions[idx][:2]}, ={point[:2]}, ={dist:.1f}m")
    
  
    min_inter_dist = float('inf')
    for i in range(len(assigned_points)):
        for j in range(i + 1, len(assigned_points)):
            dist = np.linalg.norm(assigned_points[i] - assigned_points[j])
            min_inter_dist = min(min_inter_dist, dist)
            print(f"  Agent {i} <-> Agent {j} : {dist:.1f}m")
    
    print(f"\n: {min_inter_dist:.1f}m")
    
  
    MIN_DISPERSION_THRESHOLD = 15.0
    assert min_inter_dist > MIN_DISPERSION_THRESHOLD, \
        f" {min_inter_dist:.1f}m < {MIN_DISPERSION_THRESHOLD}m"
    
    print(f"  ( > {MIN_DISPERSION_THRESHOLD}m)")


def test_voxel_confidence_decay():
    """
    
    
    
    
    """
    print("\n" + "=" * 60)
    print(": ")
    print("=" * 60)
    
    cfg = load_config()
    
    confidence_decay = cfg.environment.confidence_decay
    confidence_threshold = cfg.environment.confidence_threshold
    
  
    steps_to_threshold = int(np.ceil(np.log(confidence_threshold) / np.log(confidence_decay)))
    
    print(f":")
    print(f"  confidence_decay = {confidence_decay}")
    print(f"  confidence_threshold = {confidence_threshold}")
    print(f"   = {steps_to_threshold}")
    print(f"  10Hz = {steps_to_threshold / 10:.1f} = {steps_to_threshold / 10 / 60:.1f}")
    
  
    confidence = 1.0
    step = 0
    checkpoints = [100, 500, 1000, steps_to_threshold]
    
    print(f"\n:")
    for checkpoint in checkpoints:
        confidence_at_checkpoint = confidence_decay ** checkpoint
        print(f"  Step {checkpoint}: confidence = {confidence_at_checkpoint:.4f}")
    
  
    MIN_STEPS_TO_REEXPLORE = 1000  # 1000100
    assert steps_to_threshold > MIN_STEPS_TO_REEXPLORE, \
        f"{steps_to_threshold} > {MIN_STEPS_TO_REEXPLORE}"
    
    print(f"\n  ({steps_to_threshold} > {MIN_STEPS_TO_REEXPLORE})")


def test_frontier_detection_with_confidence():
    """
    
    
    
    """
    print("\n" + "=" * 60)
    print(": ")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_test_voxel_map(world_size=(50.0, 50.0, 20.0), voxel_size=2.0)
    
  
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
    
  
    threshold = cfg.environment.confidence_threshold
    print(f": {threshold}")
    
  
  
    voxel_map.visits[5:10, 5:10, 2:8] = 1.0
  
    voxel_map.visits[15:20, 5:10, 2:8] = threshold + 0.1
  
    voxel_map.visits[5:10, 15:20, 2:8] = threshold - 0.1
  
  
    
    allowed_mask = (occupancy == 0).astype(np.int8)
    
    allocator = FrontierAllocator(
        fps_count=30,
        cosine_margin=0.5,
        vertical_weight=1.6,
        lock_steps=100,
        angle_min_deg=60.0,
        sweep_trigger_m=15.0,
        sweep_radius_m=60.0,
        sweep_max_steps=50,
        confidence_threshold=threshold,
        allowed_mask=allowed_mask,
    )
    allocator.set_occupancy_grid(occupancy)
    
  
    frontiers = allocator.detect_frontiers(voxel_map, max_candidates=1000)
    
    print(f": {len(frontiers)}")
    
  
    region1_count = 0  # 
    region2_count = 0  # 
    region3_count = 0  # 
    region4_count = 0  # 
    
    for frontier in frontiers:
        idx = voxel_map.pos_to_idx(frontier)
        x, y, z = idx
        if 5 <= x < 10 and 5 <= y < 10 and 2 <= z < 8:
            region1_count += 1
        elif 15 <= x < 20 and 5 <= y < 10 and 2 <= z < 8:
            region2_count += 1
        elif 5 <= x < 10 and 15 <= y < 20 and 2 <= z < 8:
            region3_count += 1
        elif 15 <= x < 20 and 15 <= y < 20 and 2 <= z < 8:
            region4_count += 1
    
    print(f"\n:")
    print(f"  1 (=1.0, > {threshold}): {region1_count} ")
    print(f"  2 (={threshold + 0.1:.1f}, > {threshold}): {region2_count} ")
    print(f"  3 (={threshold - 0.1:.1f}, < {threshold}): {region3_count} ")
    print(f"  4 (=0.0, < {threshold}): {region4_count} ")
    
  
    assert region1_count == 0, f" {region1_count} "
    
  
    assert region2_count == 0, f" {region2_count} "
    
  
    assert region3_count > 0 or region4_count > 0, ""
    
    print(f"\n ")


def test_dispersion_cost_in_assignment():
    """
    
    """
    print("\n" + "=" * 60)
    print(": ")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_test_voxel_map()
    
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
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
    
  
    agent_positions = np.array([
        [50.0, 50.0, 25.0],
        [50.0, 50.0, 25.0],  # 
    ], dtype=np.float64)
    
    agent_headings = np.array([
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=np.float64)
    
  
    for pos in agent_positions:
        idx = voxel_map.pos_to_idx(pos)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                for dz in range(-3, 4):
                    x, y, z = idx[0] + dx, idx[1] + dy, idx[2] + dz
                    if 0 <= x < voxel_map.shape[0] and 0 <= y < voxel_map.shape[1] and 0 <= z < voxel_map.shape[2]:
                        voxel_map.visits[x, y, z] = 1.0
    
  
    assignments = allocator.allocate(agent_positions, agent_headings, voxel_map, current_step=0)
    
  
    targets = [assignments[i] for i in range(2) if assignments[i] is not None]
    
    if len(targets) == 2:
        dist_between_targets = np.linalg.norm(targets[0] - targets[1])
        print(f"Agent: {dist_between_targets:.1f}m")
        
      
        assert dist_between_targets > 10.0, \
            f"Agent {dist_between_targets:.1f}m"
        
        print(f"  > 10m")
    else:
        print(f" {len(targets)} ")


def test_lock_release_when_frontier_explored():
    """
    Agentlock
    
    
    1. Agent 0A
    2. Agent 1AAgent
    3. Agent 0lock
    """
    print("\n" + "=" * 60)
    print(": lock")
    print("=" * 60)
    
    cfg = load_config()
    voxel_map = create_test_voxel_map(world_size=(100.0, 100.0, 50.0), voxel_size=2.0)
    
    occupancy = np.zeros(voxel_map.shape, dtype=np.int8)
    allowed_mask = (occupancy == 0).astype(np.int8)
    
    allocator = FrontierAllocator(
        fps_count=30,
        cosine_margin=0.5,
        vertical_weight=1.6,
        lock_steps=100,  # lock
        angle_min_deg=60.0,
        sweep_trigger_m=15.0,
        sweep_radius_m=60.0,
        sweep_max_steps=50,
        confidence_threshold=cfg.environment.confidence_threshold,
        allowed_mask=allowed_mask,
    )
    allocator.set_occupancy_grid(occupancy)
    allocator._debug_enabled = True
    
  
    agent_positions = np.array([
        [50.0, 50.0, 25.0],
    ], dtype=np.float64)
    
    agent_headings = np.array([
        [1, 0, 0],
    ], dtype=np.float64)
    
  
    idx = voxel_map.pos_to_idx(agent_positions[0])
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-2, 3):
                x, y, z = idx[0] + dx, idx[1] + dy, idx[2] + dz
                if 0 <= x < voxel_map.shape[0] and 0 <= y < voxel_map.shape[1] and 0 <= z < voxel_map.shape[2]:
                    voxel_map.visits[x, y, z] = 1.0
    
  
    assignments = allocator.allocate(agent_positions, agent_headings, voxel_map, current_step=0)
    initial_target = assignments[0]
    print(f"Step 0: Agent 0  {initial_target[:2] if initial_target is not None else None}")
    
    assert initial_target is not None, ""
    
  
    assert 0 in allocator._locks, "Agent 0 "
    print(f"  Lock: {0 in allocator._locks}")
    
  
    target_idx = voxel_map.pos_to_idx(initial_target)
    print(f"   {initial_target[:2]} (: {target_idx})")
    
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-1, 2):
                x, y, z = target_idx[0] + dx, target_idx[1] + dy, target_idx[2] + dz
                if 0 <= x < voxel_map.shape[0] and 0 <= y < voxel_map.shape[1] and 0 <= z < voxel_map.shape[2]:
                    voxel_map.visits[x, y, z] = 1.0  # 
    
  
    assignments = allocator.allocate(agent_positions, agent_headings, voxel_map, current_step=1)
    new_target = assignments[0]
    print(f"Step 1: Agent 0  {new_target[:2] if new_target is not None else None}")
    
  
    if new_target is not None and initial_target is not None:
        dist = np.linalg.norm(new_target - initial_target)
        print(f"  : {dist:.1f}m")
        
      
        assert dist > 5.0, f" {dist:.1f}m"
        print(f" lock")
    else:
        print(f" - initial={initial_target}, new={new_target}")


if __name__ == "__main__":
    print("=" * 60)
    print("")
    print("=" * 60)
    
    try:
        test_voxel_confidence_decay()
        test_frontier_detection_with_confidence()
        test_frontier_dispersion_at_episode_start()
        test_dispersion_cost_in_assignment()
        test_lock_release_when_frontier_explored()
        
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
