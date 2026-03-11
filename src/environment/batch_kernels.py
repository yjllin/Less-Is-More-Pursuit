"""
High-Performance Numba Kernels for Vectorized Environment.
All kernels operate on batched tensors: (B, N, ...) or (B*N, ...).

Performance Target: 100+ SPS with 32 parallel environments.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange, float64, int32, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrapper(func): return func
        if args and callable(args[0]): return args[0]
        return wrapper
    prange = range


# =============================================================================
# Constants
# =============================================================================

PI = 3.141592653589793
TWO_PI = 6.283185307179586


# =============================================================================
# Physics Kernels
# =============================================================================

@njit(cache=True)
def _normalize_yaw(yaw: float) -> float:
    """Normalize yaw to [-pi, pi]."""
    while yaw > PI:
        yaw -= TWO_PI
    while yaw < -PI:
        yaw += TWO_PI
    return yaw


@njit(cache=True, parallel=True)
def batch_update_agents_v2(
    pos: np.ndarray,
    vel: np.ndarray,
    yaw: np.ndarray,
    actions: np.ndarray,       # (B, N, 4) in - [vx_body, vy_body, vz_body, yaw_rate]
    dt: float,
    max_speed: float,
    max_accel: float,
    max_yaw_rate: float,
    world_min: np.ndarray,
    world_max: np.ndarray,
    lin_acc_body: np.ndarray,  # (B, N, 3) out
    ang_vel_body: np.ndarray,  # (B, N, 3) out
    accel_noise: np.ndarray,   # (B, N, 3)
    gyro_noise: np.ndarray,    # (B, N, 3)
) -> np.ndarray:
    B, N, _ = pos.shape
    boundary_collisions = np.zeros((B, N), dtype=np.bool_)
    max_delta_v = max_accel * dt
    
    for b in prange(B):
        for n in range(N):
            # 1. Read current state.
            p = pos[b, n].copy()
            v = vel[b, n].copy()
            y = yaw[b, n]
            cy, sy = np.cos(y), np.sin(y)
            
            # 2. Read the body-frame velocity command.
            act_vx = actions[b, n, 0]
            act_vy = actions[b, n, 1]
            act_vz = actions[b, n, 2]
            target_yaw_rate = actions[b, n, 3]
            
            # 3. Rotate the commanded velocity into the world frame.
            # R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            # v_world = R * v_body
            target_vel_x = act_vx * cy - act_vy * sy
            target_vel_y = act_vx * sy + act_vy * cy
            target_vel_z = act_vz  # Yaw-only model; vertical motion is not coupled to pitch/roll.
            
            # 4. Clamp the commanded speed magnitude.
            tgt_speed = np.sqrt(target_vel_x**2 + target_vel_y**2 + target_vel_z**2)
            if tgt_speed > max_speed:
                scale = max_speed / tgt_speed
                target_vel_x *= scale
                target_vel_y *= scale
                target_vel_z *= scale
                
            # 5. Apply acceleration-bounded velocity tracking.
            err_x = target_vel_x - v[0]
            err_y = target_vel_y - v[1]
            err_z = target_vel_z - v[2]
            
            err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)
            if err_norm > max_delta_v and err_norm > 1e-9:
                scale = max_delta_v / err_norm
                err_x *= scale
                err_y *= scale
                err_z *= scale
                
            new_v = np.empty(3, dtype=np.float64)
            new_v[0] = v[0] + err_x
            new_v[1] = v[1] + err_y
            new_v[2] = v[2] + err_z

            # IMU linear acceleration in world frame (from velocity delta)
            acc_x = err_x / dt
            acc_y = err_y / dt
            acc_z = err_z / dt

            # Convert acceleration to body frame
            acc_body_x = acc_x * cy + acc_y * sy
            acc_body_y = -acc_x * sy + acc_y * cy
            acc_body_z = acc_z

            # Angular velocity in body frame (yaw-only model)
            ang_x = 0.0
            ang_y = 0.0
            ang_z = 0.0
            
            # Enforce the global speed limit after the acceleration update.
            new_spd = np.sqrt(new_v[0]**2 + new_v[1]**2 + new_v[2]**2)
            if new_spd > max_speed:
                scale = max_speed / new_spd
                new_v *= scale
            
            # 6. Integrate position.
            new_p = p + new_v * dt
            
            # 7. Integrate yaw with rate clamping.
            clamped_yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, target_yaw_rate))
            new_y = _normalize_yaw(y + clamped_yaw_rate * dt)
            ang_z = clamped_yaw_rate

            # Add IMU noise
            acc_body_x += accel_noise[b, n, 0]
            acc_body_y += accel_noise[b, n, 1]
            acc_body_z += accel_noise[b, n, 2]
            ang_x += gyro_noise[b, n, 0]
            ang_y += gyro_noise[b, n, 1]
            ang_z += gyro_noise[b, n, 2]

            lin_acc_body[b, n, 0] = acc_body_x
            lin_acc_body[b, n, 1] = acc_body_y
            lin_acc_body[b, n, 2] = acc_body_z
            ang_vel_body[b, n, 0] = ang_x
            ang_vel_body[b, n, 1] = ang_y
            ang_vel_body[b, n, 2] = ang_z
            
            # 8. Flag boundary violations; collision response happens later.
            boundary_col = False
            for k in range(3):
                if new_p[k] < world_min[k]:
                    boundary_col = True
                elif new_p[k] > world_max[k]:
                    boundary_col = True
            
            # Commit the updated state.
            pos[b, n] = new_p
            vel[b, n] = new_v
            yaw[b, n] = new_y
            boundary_collisions[b, n] = boundary_col
            
    return boundary_collisions


@njit(cache=True, parallel=True)
def batch_update_targets_v2(
    target_pos: np.ndarray,    # (B, 3) in/out
    target_vel: np.ndarray,    # (B, 3) in/out
    agent_pos: np.ndarray,     # (B, N, 3) in
    behaviors: np.ndarray,     # (B,) int - 0=static, 1=wander, 2=repulse
    target_speeds: np.ndarray, # (B,) float
    max_vx: float,
    max_vy: float,
    max_vz: float,
    dt: float,
    target_max_accel: float,
    world_min: np.ndarray,     # (3,)
    world_max: np.ndarray,     # (3,)
    rng_vals: np.ndarray,      # (B, 3) pre-generated random values
    occupancy_grid: np.ndarray,# (Gx, Gy, Gz) int8
    origin: np.ndarray,        # (3,)
    voxel_size: float,
    avoid_radius_vox: int,
) -> None:
    """
    Batch update all targets based on behavior.
    
    For repulse targets (behavior == 2), the desired motion combines:
    1. Agent repulsion.
    2. Boundary push-back.
    3. Obstacle repulsion from nearby occupied voxels.
    4. Weighted fusion into a preferred direction.
    5. A second-order velocity update with acceleration limits.
    """
    B = target_pos.shape[0]
    N = agent_pos.shape[1]
    
    # Behavior shaping constants for repulse mode.
    REPULSE_RADIUS = 50.0       # Agents inside this radius contribute repulsion.
    MIN_REPULSE_DIST = 10.0     # Soft lower bound to prevent singular repulsion.
    BOUNDARY_MARGIN = 12.0      # Start pushing back before reaching the boundary.
    BOUNDARY_HARD_MARGIN = 3.0  # Strong push-back very close to the boundary.
    VELOCITY_SMOOTHING = 0.4    # Legacy fallback smoothing when accel limit is disabled.
    OBSTACLE_WEIGHT = 1.8       # Relative weight of obstacle avoidance.
    BOUNDARY_WEIGHT = 1.0       # Relative weight of boundary avoidance.
    
    for b in prange(B):
        behavior = behaviors[b]
        speed = target_speeds[b]
        pos = target_pos[b].copy()
        v = target_vel[b].copy()
        
        gx, gy, gz = occupancy_grid.shape
        occ_min = origin
        occ_max = np.array([origin[0] + gx * voxel_size,
                            origin[1] + gy * voxel_size,
                            origin[2] + gz * voxel_size], dtype=np.float64)
        clamp_min = np.array([
            max(world_min[0], occ_min[0]),
            max(world_min[1], occ_min[1]),
            max(world_min[2], occ_min[2]),
        ], dtype=np.float64)
        clamp_max = np.array([
            min(world_max[0], occ_max[0]),
            min(world_max[1], occ_max[1]),
            min(world_max[2], occ_max[2]),
        ], dtype=np.float64)
        
        if behavior == 0:  # static
            target_v = np.zeros(3, dtype=np.float64)
        elif behavior == 1:  # wander
            jitter = rng_vals[b] * 0.2
            jitter[2] = 0.0
            target_v = v + jitter
        else:  # repulse (behavior == 2)
            # ============================================
            # 1. Aggregate repulsion from all nearby agents.
            # ============================================
            repulse_force = np.zeros(3, dtype=np.float64)
            total_weight = 0.0
            
            for n in range(N):
                # Vector from the agent to the target.
                dx = pos[0] - agent_pos[b, n, 0]
                dy = pos[1] - agent_pos[b, n, 1]
                dz = pos[2] - agent_pos[b, n, 2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if dist < REPULSE_RADIUS and dist > 1e-6:
                    # Stronger repulsion for close agents, softened by MIN_REPULSE_DIST.
                    effective_dist = max(dist, MIN_REPULSE_DIST)
                    weight = (REPULSE_RADIUS - dist) / (effective_dist * effective_dist)
                    
                    # Accumulate the weighted repulsion direction.
                    repulse_force[0] += (dx / dist) * weight
                    repulse_force[1] += (dy / dist) * weight
                    repulse_force[2] += (dz / dist) * weight
                    total_weight += weight
            
            # Normalize the combined repulsion vector.
            repulse_norm = np.sqrt(repulse_force[0]**2 + repulse_force[1]**2 + repulse_force[2]**2)
            if repulse_norm > 1e-6:
                repulse_force = repulse_force / repulse_norm
            else:
                # Fallback to a random direction if the force cancels out.
                repulse_force = rng_vals[b].copy()
                repulse_norm = np.sqrt(repulse_force[0]**2 + repulse_force[1]**2 + repulse_force[2]**2)
                if repulse_norm > 1e-6:
                    repulse_force = repulse_force / repulse_norm
            
            # ============================================
            # 2. Push away from world boundaries.
            # ============================================
            boundary_force = np.zeros(3, dtype=np.float64)
            
            for k in range(2):  # XY axes
                # Distance to the lower boundary.
                dist_to_min = pos[k] - clamp_min[k]
                if dist_to_min < BOUNDARY_MARGIN:
                    if dist_to_min < BOUNDARY_HARD_MARGIN:
                        # Strong emergency correction when extremely close.
                        boundary_force[k] += 3.0
                    else:
                        # Smooth push-back inside the warning margin.
                        strength = (BOUNDARY_MARGIN - dist_to_min) / BOUNDARY_MARGIN
                        boundary_force[k] += strength * BOUNDARY_WEIGHT
                
                # Distance to the upper boundary.
                dist_to_max = clamp_max[k] - pos[k]
                if dist_to_max < BOUNDARY_MARGIN:
                    if dist_to_max < BOUNDARY_HARD_MARGIN:
                        boundary_force[k] -= 3.0
                    else:
                        strength = (BOUNDARY_MARGIN - dist_to_max) / BOUNDARY_MARGIN
                        boundary_force[k] -= strength * BOUNDARY_WEIGHT
            
            # Use a tighter vertical margin to keep targets away from floor/ceiling.
            dist_to_z_min = pos[2] - clamp_min[2]
            dist_to_z_max = clamp_max[2] - pos[2]
            if dist_to_z_min < BOUNDARY_MARGIN * 0.5:
                boundary_force[2] += (BOUNDARY_MARGIN * 0.5 - dist_to_z_min) / (BOUNDARY_MARGIN * 0.5)
            if dist_to_z_max < BOUNDARY_MARGIN * 0.5:
                boundary_force[2] -= (BOUNDARY_MARGIN * 0.5 - dist_to_z_max) / (BOUNDARY_MARGIN * 0.5)
            
            # ============================================
            # 3. Push away from nearby occupied voxels.
            # ============================================
            obstacle_force = np.zeros(3, dtype=np.float64)
            
            if avoid_radius_vox > 0:
                pos_clamped = pos.copy()
                for k in range(3):
                    if pos_clamped[k] < clamp_min[k]:
                        pos_clamped[k] = clamp_min[k]
                    elif pos_clamped[k] > clamp_max[k]:
                        pos_clamped[k] = clamp_max[k]
                
                ix = int(np.floor((pos_clamped[0] - origin[0]) / voxel_size))
                iy = int(np.floor((pos_clamped[1] - origin[1]) / voxel_size))
                iz = int(np.floor((pos_clamped[2] - origin[2]) / voxel_size))
                
                radius_m = avoid_radius_vox * voxel_size
                
                for ddx in range(-avoid_radius_vox, avoid_radius_vox + 1):
                    nx = ix + ddx
                    if nx < 0 or nx >= gx:
                        continue
                    for ddy in range(-avoid_radius_vox, avoid_radius_vox + 1):
                        ny = iy + ddy
                        if ny < 0 or ny >= gy:
                            continue
                        for ddz in range(-avoid_radius_vox, avoid_radius_vox + 1):
                            nz = iz + ddz
                            if nz < 0 or nz >= gz:
                                continue
                            if occupancy_grid[nx, ny, nz] != 1:
                                continue
                            
                            ox = origin[0] + (nx + 0.5) * voxel_size
                            oy = origin[1] + (ny + 0.5) * voxel_size
                            oz = origin[2] + (nz + 0.5) * voxel_size
                            
                            vvx = pos[0] - ox
                            vvy = pos[1] - oy
                            vvz = pos[2] - oz
                            dist = np.sqrt(vvx*vvx + vvy*vvy + vvz*vvz)
                            
                            if dist < 1e-6 or dist > radius_m:
                                continue
                            
                            # Weight obstacle repulsion by normalized proximity.
                            w = ((radius_m - dist) / radius_m) ** 2
                            obstacle_force[0] += (vvx / dist) * w
                            obstacle_force[1] += (vvy / dist) * w
                            obstacle_force[2] += (vvz / dist) * w
                
                obs_norm = np.sqrt(obstacle_force[0]**2 + obstacle_force[1]**2 + obstacle_force[2]**2)
                if obs_norm > 1e-6:
                    obstacle_force = obstacle_force / obs_norm * OBSTACLE_WEIGHT
            
            # ============================================
            # 4. Fuse repulsion, boundary, and obstacle terms into a target direction.
            # ============================================
            target_dir = repulse_force.copy()
            
            target_dir[0] += boundary_force[0]
            target_dir[1] += boundary_force[1]
            target_dir[2] += boundary_force[2]
            
            target_dir[0] += obstacle_force[0]
            target_dir[1] += obstacle_force[1]
            target_dir[2] += obstacle_force[2]
            
            # Convert the fused direction into the desired target velocity.
            dir_norm = np.sqrt(target_dir[0]**2 + target_dir[1]**2 + target_dir[2]**2)
            if dir_norm > 1e-6:
                target_v = target_dir / dir_norm * speed
            else:
                target_v = np.zeros(3, dtype=np.float64)
            
        # ============================================
      
        # new_v = old_v + clamp(target_v - old_v, ||delta_v|| <= a_max * dt)
        # ============================================
        delta_vx = target_v[0] - v[0]
        delta_vy = target_v[1] - v[1]
        delta_vz = target_v[2] - v[2]
        max_delta_v = max(0.0, target_max_accel) * dt
        if max_delta_v > 0.0:
            req_norm = np.sqrt(delta_vx * delta_vx + delta_vy * delta_vy + delta_vz * delta_vz)
            if req_norm > max_delta_v:
                scale = max_delta_v / max(req_norm, 1e-8)
                delta_vx *= scale
                delta_vy *= scale
                delta_vz *= scale
        else:
            # Backward-compatible fallback if target_max_accel is disabled.
            delta_vx = (v[0] * VELOCITY_SMOOTHING + target_v[0] * (1.0 - VELOCITY_SMOOTHING)) - v[0]
            delta_vy = (v[1] * VELOCITY_SMOOTHING + target_v[1] * (1.0 - VELOCITY_SMOOTHING)) - v[1]
            delta_vz = (v[2] * VELOCITY_SMOOTHING + target_v[2] * (1.0 - VELOCITY_SMOOTHING)) - v[2]
        new_v = np.empty(3, dtype=np.float64)
        new_v[0] = v[0] + delta_vx
        new_v[1] = v[1] + delta_vy
        new_v[2] = v[2] + delta_vz
    
        # Enforce per-axis speed bounds.
        if max_vx > 0.0:
            new_v[0] = min(max(new_v[0], -max_vx), max_vx)
        if max_vy > 0.0:
            new_v[1] = min(max(new_v[1], -max_vy), max_vy)
        if max_vz > 0.0:
            new_v[2] = min(max(new_v[2], -max_vz), max_vz)

        # Enforce the overall target speed limit.
        v_speed = np.sqrt(new_v[0]**2 + new_v[1]**2 + new_v[2]**2)
        if v_speed > speed and v_speed > 1e-6:
            new_v = new_v / v_speed * speed
        
        # Integrate position.
        new_pos = pos + new_v * dt
        
        # Clamp to valid world bounds and reflect velocity if we hit the boundary.
        for i in range(3):
            if new_pos[i] < clamp_min[i]:
                new_pos[i] = clamp_min[i]
                # Bounce back softly instead of hard-stopping.
                if new_v[i] < 0:
                    new_v[i] = abs(new_v[i]) * 0.5
            elif new_pos[i] > clamp_max[i]:
                new_pos[i] = clamp_max[i]
                if new_v[i] > 0:
                    new_v[i] = -abs(new_v[i]) * 0.5

        # If we end inside an obstacle, project to the nearest free voxel center.
        ix = int(np.floor((new_pos[0] - origin[0]) / voxel_size))
        iy = int(np.floor((new_pos[1] - origin[1]) / voxel_size))
        iz = int(np.floor((new_pos[2] - origin[2]) / voxel_size))
        
        if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
            if occupancy_grid[ix, iy, iz] == 1:
                best_dist2 = 1e12
                best_center = new_pos.copy()
                radius = avoid_radius_vox + 2  # Expand the search slightly beyond the avoidance radius.
                
                for ddx in range(-radius, radius + 1):
                    nx = ix + ddx
                    if nx < 0 or nx >= gx:
                        continue
                    for ddy in range(-radius, radius + 1):
                        ny = iy + ddy
                        if ny < 0 or ny >= gy:
                            continue
                        for ddz in range(-radius, radius + 1):
                            nz = iz + ddz
                            if nz < 0 or nz >= gz:
                                continue
                            if occupancy_grid[nx, ny, nz] != 0:
                                continue
                            
                            cx = origin[0] + (nx + 0.5) * voxel_size
                            cy = origin[1] + (ny + 0.5) * voxel_size
                            cz = origin[2] + (nz + 0.5) * voxel_size
                            
                            dxp = new_pos[0] - cx
                            dyp = new_pos[1] - cy
                            dzp = new_pos[2] - cz
                            dist2 = dxp*dxp + dyp*dyp + dzp*dzp
                            
                            if dist2 < best_dist2:
                                best_dist2 = dist2
                                best_center[0] = cx
                                best_center[1] = cy
                                best_center[2] = cz
                
                if best_dist2 < 1e12:
                    new_pos = best_center
                    # Blend the current velocity with an escape velocity toward free space.
                    dvx = new_pos[0] - pos[0]
                    dvy = new_pos[1] - pos[1]
                    dvz = new_pos[2] - pos[2]
                    escape_v = np.array([dvx, dvy, dvz], dtype=np.float64) / max(dt, 1e-6)
                    escape_speed = np.sqrt(escape_v[0]**2 + escape_v[1]**2 + escape_v[2]**2)
                    if escape_speed > speed and escape_speed > 1e-6:
                        escape_v = escape_v / escape_speed * speed
                    # Favor the escape direction to keep the target moving out of collision.
                    new_v = new_v * 0.3 + escape_v * 0.7

        target_pos[b] = new_pos
        target_vel[b] = new_v


# =============================================================================
# Collision Detection Kernels
# =============================================================================

@njit(cache=True, parallel=True)
def batch_check_collisions_v2(
    pos: np.ndarray,           # (B, N, 3)
    occupancy_grid: np.ndarray,# (Gx, Gy, Gz) int8
    origin: np.ndarray,        # (3,)
    voxel_size: float,
) -> np.ndarray:               # (B, N) bool
    """Batch collision check for all agents."""
    B, N, _ = pos.shape
    gx, gy, gz = occupancy_grid.shape
    collisions = np.zeros((B, N), dtype=np.bool_)
    
    for b in prange(B):
        for n in range(N):
            p = pos[b, n]
            ix = int(np.floor((p[0] - origin[0]) / voxel_size))
            iy = int(np.floor((p[1] - origin[1]) / voxel_size))
            iz = int(np.floor((p[2] - origin[2]) / voxel_size))
            
            # Out of bounds = collision
            if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                collisions[b, n] = True
            elif occupancy_grid[ix, iy, iz] == 1:
                collisions[b, n] = True
    
    return collisions


@njit(cache=True, parallel=True)
def batch_check_team_collisions(
    pos: np.ndarray,           # (B, N, 3)
    radius: float,             # collision radius
    alive_mask: np.ndarray,    # (B, N) bool
) -> np.ndarray:               # (B, N) bool
    """
    Batch check team collisions using Numba parallel.
    Replaces Python nested loops for O(N^2) pairwise distance checks.
    
    Args:
        pos: Agent positions (B, N, 3)
        radius: Collision radius in meters
    
    Returns:
        team_collisions: (B, N) bool mask of agents in collision
    """
    B, N, _ = pos.shape
    team_collisions = np.zeros((B, N), dtype=np.bool_)
    rad_sq = radius * radius
    
    for b in prange(B):
        for i in range(N):
            if not alive_mask[b, i]:
                continue
            for j in range(i + 1, N):
                if not alive_mask[b, j]:
                    continue
                dx = pos[b, i, 0] - pos[b, j, 0]
                dy = pos[b, i, 1] - pos[b, j, 1]
                dz = pos[b, i, 2] - pos[b, j, 2]
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq <= rad_sq:
                    team_collisions[b, i] = True
                    team_collisions[b, j] = True
    
    return team_collisions


@njit(cache=True, parallel=True)
def batch_update_path_lengths(
    path_lengths: np.ndarray,  # (B, N) in/out - current path lengths
    pos: np.ndarray,           # (B, N, 3) - current positions
    prev_pos: np.ndarray,      # (B, N, 3) - previous positions
) -> None:
    """
    Update path lengths by subtracting actual displacement.
    Provides smooth potential signal between A* updates.
    
    Logic: new_length = max(0.0, old_length - step_dist)
    
    Args:
        path_lengths: Current estimated path lengths (modified in-place)
        pos: Current agent positions
        prev_pos: Previous agent positions
    """
    B, N, _ = pos.shape
    
    for b in prange(B):
        for n in range(N):
            # Calculate actual displacement this step
            dx = pos[b, n, 0] - prev_pos[b, n, 0]
            dy = pos[b, n, 1] - prev_pos[b, n, 1]
            dz = pos[b, n, 2] - prev_pos[b, n, 2]
            step_dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            
            # Subtract displacement from path length (clamp to 0)
            new_length = path_lengths[b, n] - step_dist
            if new_length < 0.0:
                new_length = 0.0
            path_lengths[b, n] = new_length


@njit(cache=True, parallel=True)
def batch_check_lkp_reached(
    pos: np.ndarray,           # (B, N, 3)
    lkp_pos: np.ndarray,       # (B, 3) - last known positions
    threshold: float,          # breakout distance threshold
) -> np.ndarray:               # (B,) bool - whether any agent reached LKP
    """
    Batch check if any agent in each env has reached the LKP.
    Replaces Python loop in _update_target_tracking.
    
    Args:
        pos: Agent positions (B, N, 3)
        lkp_pos: Last known target positions (B, 3)
        threshold: Distance threshold for LKP breakout
    
    Returns:
        lkp_reached: (B,) bool mask
    """
    B, N, _ = pos.shape
    lkp_reached = np.zeros(B, dtype=np.bool_)
    threshold_sq = threshold * threshold
    
    for b in prange(B):
        for n in range(N):
            dx = pos[b, n, 0] - lkp_pos[b, 0]
            dy = pos[b, n, 1] - lkp_pos[b, 1]
            dz = pos[b, n, 2] - lkp_pos[b, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < threshold_sq:
                lkp_reached[b] = True
                break  # Early exit for this env
    
    return lkp_reached


@njit(cache=True, parallel=True)
def batch_update_obs_targets(
    obs_target_pos: np.ndarray,    # (B, N, 3) out
    obs_target_vel: np.ndarray,    # (B, N, 3) out
    target_pos: np.ndarray,        # (B, 3)
    target_vel: np.ndarray,        # (B, 3)
    last_known_pos: np.ndarray,    # (B, 3)
    current_goals: np.ndarray,     # (B, N, 3)
    target_mode_flag: np.ndarray,  # (B, N)
    team_has_target: np.ndarray,   # (B,) bool
    needs_nav_update: np.ndarray,  # (B,) bool - skip envs that will do full nav
    guidance_enabled: bool,
) -> None:
    """
    Batch update observation targets for all agents.
    Replaces Python loops in _update_navigation_staggered.
    """
    B, N, _ = obs_target_pos.shape
    
    for b in prange(B):
        if needs_nav_update[b]:
            continue  # Skip envs that will do full nav update
        
        for n in range(N):
            if target_mode_flag[b, n] > 0.5:  # Tracking mode
                if team_has_target[b]:
                    obs_target_pos[b, n, 0] = target_pos[b, 0]
                    obs_target_pos[b, n, 1] = target_pos[b, 1]
                    obs_target_pos[b, n, 2] = target_pos[b, 2]
                    obs_target_vel[b, n, 0] = target_vel[b, 0]
                    obs_target_vel[b, n, 1] = target_vel[b, 1]
                    obs_target_vel[b, n, 2] = target_vel[b, 2]
                else:
                    obs_target_pos[b, n, 0] = last_known_pos[b, 0]
                    obs_target_pos[b, n, 1] = last_known_pos[b, 1]
                    obs_target_pos[b, n, 2] = last_known_pos[b, 2]
                    obs_target_vel[b, n, 0] = 0.0
                    obs_target_vel[b, n, 1] = 0.0
                    obs_target_vel[b, n, 2] = 0.0
            else:  # Searching mode
                if guidance_enabled:
                    obs_target_pos[b, n, 0] = current_goals[b, n, 0]
                    obs_target_pos[b, n, 1] = current_goals[b, n, 1]
                    obs_target_pos[b, n, 2] = current_goals[b, n, 2]
                else:
                    obs_target_pos[b, n, 0] = last_known_pos[b, 0]
                    obs_target_pos[b, n, 1] = last_known_pos[b, 1]
                    obs_target_pos[b, n, 2] = last_known_pos[b, 2]
                obs_target_vel[b, n, 0] = 0.0
                obs_target_vel[b, n, 1] = 0.0
                obs_target_vel[b, n, 2] = 0.0


@njit(cache=True, parallel=True)
def batch_rollback_collisions_v2(
    pos: np.ndarray,           # (B, N, 3) in/out
    vel: np.ndarray,           # (B, N, 3) in/out
    prev_pos: np.ndarray,      # (B, N, 3)
    collisions: np.ndarray,    # (B, N) bool
) -> None:
    """Rollback collided agents with nudge to prevent sticking."""
    B, N, _ = pos.shape
    for b in prange(B):
        for n in range(N):
            if collisions[b, n]:
              
                pos[b, n, 0] = prev_pos[b, n, 0]
                pos[b, n, 1] = prev_pos[b, n, 1]
                pos[b, n, 2] = prev_pos[b, n, 2]
                
              
                escape_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                vel_mag = np.sqrt(vel[b, n, 0]**2 + vel[b, n, 1]**2 + vel[b, n, 2]**2)
                
                if vel_mag > 1e-6:
                  
                    escape_dir[0] = -vel[b, n, 0] / vel_mag
                    escape_dir[1] = -vel[b, n, 1] / vel_mag
                    escape_dir[2] = -vel[b, n, 2] / vel_mag
                else:
                  
                    escape_dir[2] = 1.0
                
              
                nudge_distance = 0.05  # 5cm
                pos[b, n, 0] += escape_dir[0] * nudge_distance
                pos[b, n, 1] += escape_dir[1] * nudge_distance
                pos[b, n, 2] += escape_dir[2] * nudge_distance
                
              
                vel[b, n, 0] = 0.0
                vel[b, n, 1] = 0.0
                vel[b, n, 2] = 0.0


@njit(cache=True, parallel=True)
def batch_resolve_collisions_sliding(
    pos: np.ndarray,           # (B, N, 3) in/out (post-integrate)
    vel: np.ndarray,           # (B, N, 3) in/out
    prev_pos: np.ndarray,      # (B, N, 3) pre-integrate
    collisions: np.ndarray,    # (B, N) bool
    occupancy_grid: np.ndarray,# (Gx, Gy, Gz) int8
    origin: np.ndarray,        # (3,)
    voxel_size: float,
    world_min: np.ndarray,
    world_max: np.ndarray,
) -> None:
    """Resolve collisions via sliding projection (no teleport rollback)."""
    B, N, _ = pos.shape
    gx, gy, gz = occupancy_grid.shape
    step_size = max(0.1, voxel_size * 0.5)

    for b in prange(B):
        for n in range(N):
            if not collisions[b, n]:
                continue

            p0 = prev_pos[b, n]
            p1 = pos[b, n]
            v = vel[b, n]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            last_free = p0.copy()
            if dist > 1e-9:
                steps = int(np.ceil(dist / step_size))
                if steps < 1:
                    steps = 1
                for s in range(1, steps + 1):
                    t = float(s) / float(steps)
                    cx = p0[0] + dx * t
                    cy = p0[1] + dy * t
                    cz = p0[2] + dz * t

                    out_bounds = (cx < world_min[0] or cx > world_max[0] or
                                  cy < world_min[1] or cy > world_max[1] or
                                  cz < world_min[2] or cz > world_max[2])
                    if out_bounds:
                        break

                    ix = int(np.floor((cx - origin[0]) / voxel_size))
                    iy = int(np.floor((cy - origin[1]) / voxel_size))
                    iz = int(np.floor((cz - origin[2]) / voxel_size))
                    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                        break
                    if occupancy_grid[ix, iy, iz] == 1:
                        break

                    last_free[0] = cx
                    last_free[1] = cy
                    last_free[2] = cz

            # Approximate collision normal from axis penetration
            nx = 0.0
            ny = 0.0
            nz = 0.0
            if abs(dx) > 1e-9:
                cx = p0[0] + dx
                cy = p0[1]
                cz = p0[2]
                hit = False
                if cx < world_min[0] or cx > world_max[0]:
                    hit = True
                else:
                    ix = int(np.floor((cx - origin[0]) / voxel_size))
                    iy = int(np.floor((cy - origin[1]) / voxel_size))
                    iz = int(np.floor((cz - origin[2]) / voxel_size))
                    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                        hit = True
                    elif occupancy_grid[ix, iy, iz] == 1:
                        hit = True
                if hit:
                    nx = 1.0 if dx > 0.0 else -1.0

            if abs(dy) > 1e-9:
                cx = p0[0]
                cy = p0[1] + dy
                cz = p0[2]
                hit = False
                if cy < world_min[1] or cy > world_max[1]:
                    hit = True
                else:
                    ix = int(np.floor((cx - origin[0]) / voxel_size))
                    iy = int(np.floor((cy - origin[1]) / voxel_size))
                    iz = int(np.floor((cz - origin[2]) / voxel_size))
                    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                        hit = True
                    elif occupancy_grid[ix, iy, iz] == 1:
                        hit = True
                if hit:
                    ny = 1.0 if dy > 0.0 else -1.0

            if abs(dz) > 1e-9:
                cx = p0[0]
                cy = p0[1]
                cz = p0[2] + dz
                hit = False
                if cz < world_min[2] or cz > world_max[2]:
                    hit = True
                else:
                    ix = int(np.floor((cx - origin[0]) / voxel_size))
                    iy = int(np.floor((cy - origin[1]) / voxel_size))
                    iz = int(np.floor((cz - origin[2]) / voxel_size))
                    if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                        hit = True
                    elif occupancy_grid[ix, iy, iz] == 1:
                        hit = True
                if hit:
                    nz = 1.0 if dz > 0.0 else -1.0

            n_norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            if n_norm < 1e-6:
                if dist > 1e-9:
                    nx = dx / dist
                    ny = dy / dist
                    nz = dz / dist
                    n_norm = 1.0
            else:
                nx /= n_norm
                ny /= n_norm
                nz /= n_norm

            # Project velocity onto tangent plane
            if n_norm > 1e-6:
                proj = v[0] * nx + v[1] * ny + v[2] * nz
                if proj > 0.0:
                    v[0] -= proj * nx
                    v[1] -= proj * ny
                    v[2] -= proj * nz

            pos[b, n, 0] = last_free[0]
            pos[b, n, 1] = last_free[1]
            pos[b, n, 2] = last_free[2]
            vel[b, n, 0] = v[0]
            vel[b, n, 1] = v[1]
            vel[b, n, 2] = v[2]


# =============================================================================
# LiDAR Simulation Kernels
# =============================================================================

@njit(cache=True)
def _simulate_lidar_single(
    pos: np.ndarray,           # (3,)
    yaw: float,
    occupancy_grid: np.ndarray,# (Gx, Gy, Gz)
    origin: np.ndarray,        # (3,)
    voxel_size: float,
    max_range: float,
    sub_rays: int,
    spread_rad: float,
) -> np.ndarray:               # (26,) normalized distances
    """Simulate LiDAR for single agent - 26 directions with sector min-pooling."""
    gx, gy, gz = occupancy_grid.shape
    distances = np.ones(26, dtype=np.float64)
    # Use smaller step for near obstacles
    step_size = min(voxel_size, 2.0)
    num_steps = int(max_range / step_size)
    cone_cos = np.cos(spread_rad)
    cone_sin = np.sin(spread_rad)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Normalize direction (body frame)
                dir_len = np.sqrt(float(dx*dx + dy*dy + dz*dz))
                dir_x_body = dx / dir_len
                dir_y_body = dy / dir_len
                dir_z_body = dz / dir_len
                dir_x = dir_x_body * cy - dir_y_body * sy
                dir_y = dir_x_body * sy + dir_y_body * cy
                dir_z = dir_z_body

                # Build orthonormal basis around dir
                up_x = 0.0
                up_y = 0.0
                up_z = 1.0
                ux = dir_y_body * up_z - dir_z_body * up_y
                uy = dir_z_body * up_x - dir_x_body * up_z
                uz = dir_x_body * up_y - dir_y_body * up_x
                u_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
                if u_norm < 1e-6:
                    up_x = 1.0
                    up_y = 0.0
                    up_z = 0.0
                    ux = dir_y_body * up_z - dir_z_body * up_y
                    uy = dir_z_body * up_x - dir_x_body * up_z
                    uz = dir_x_body * up_y - dir_y_body * up_x
                    u_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
                inv_un = 1.0 / max(u_norm, 1e-6)
                ux *= inv_un
                uy *= inv_un
                uz *= inv_un
                # v = dir x u
                vx = dir_y_body * uz - dir_z_body * uy
                vy = dir_z_body * ux - dir_x_body * uz
                vz = dir_x_body * uy - dir_y_body * ux

                min_norm = 1.0
                rays = sub_rays if sub_rays > 0 else 1

                for s in range(rays):
                    if s == 0 or rays == 1:
                        sub_body_x = dir_x_body
                        sub_body_y = dir_y_body
                        sub_body_z = dir_z_body
                    else:
                        angle = TWO_PI * (s - 1) / max(1, rays - 1)
                        ca = np.cos(angle)
                        sa = np.sin(angle)
                        off_x = ca * ux + sa * vx
                        off_y = ca * uy + sa * vy
                        off_z = ca * uz + sa * vz
                        sub_body_x = dir_x_body * cone_cos + off_x * cone_sin
                        sub_body_y = dir_y_body * cone_cos + off_y * cone_sin
                        sub_body_z = dir_z_body * cone_cos + off_z * cone_sin

                    sub_dir_x = sub_body_x * cy - sub_body_y * sy
                    sub_dir_y = sub_body_x * sy + sub_body_y * cy
                    sub_dir_z = sub_body_z

                    for step in range(1, num_steps + 1):
                        dist = step * step_size
                        px = pos[0] + sub_dir_x * dist
                        py = pos[1] + sub_dir_y * dist
                        pz = pos[2] + sub_dir_z * dist

                        ix = int(np.floor((px - origin[0]) / voxel_size))
                        iy = int(np.floor((py - origin[1]) / voxel_size))
                        iz = int(np.floor((pz - origin[2]) / voxel_size))

                        if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
                            min_norm = min(min_norm, dist / max_range)
                            break

                        if occupancy_grid[ix, iy, iz] == 1:
                            min_norm = min(min_norm, dist / max_range)
                            break

                distances[idx] = min_norm
                idx += 1

    return distances


@njit(cache=True, parallel=True)
def batch_simulate_lidar_v2(
    pos: np.ndarray,           # (B, N, 3)
    yaw: np.ndarray,           # (B, N)
    occupancy_grid: np.ndarray,# (Gx, Gy, Gz)
    origin: np.ndarray,        # (3,)
    voxel_size: float,
    max_range: float,
    sub_rays: int,
    spread_deg: float,
) -> np.ndarray:               # (B, N, 26)
    """Batch LiDAR simulation for all agents."""
    B, N, _ = pos.shape
    lidar = np.zeros((B, N, 26), dtype=np.float64)
    spread_rad = spread_deg * PI / 180.0
    
    for b in prange(B):
        for n in range(N):
            lidar[b, n] = _simulate_lidar_single(
                pos[b, n], yaw[b, n], occupancy_grid, origin, voxel_size, max_range, sub_rays, spread_rad
            )
    
    return lidar


# =============================================================================
# Observation Building Kernels
# =============================================================================

@njit(cache=True, parallel=True)
def batch_build_teammate_features(
    pos: np.ndarray,           # (B, N, 3)
    vel: np.ndarray,           # (B, N, 3)
    yaw: np.ndarray,           # (B, N)
    top_k: int,
    view_radius: float,
) -> np.ndarray:               # (B, N, top_k * 8)
    """Build teammate features for all agents."""
    B, N, _ = pos.shape
    feats = np.zeros((B, N, top_k * 8), dtype=np.float64)
    
    for b in prange(B):
        for i in range(N):
            p_self = pos[b, i]
            y = yaw[b, i]
            cy = np.cos(y)
            sy = np.sin(y)
            
            # Collect distances to all other agents
            dists = np.empty(N - 1, dtype=np.float64)
            indices = np.empty(N - 1, dtype=np.int32)
            k = 0
            for j in range(N):
                if i == j:
                    continue
                dx = pos[b, j, 0] - p_self[0]
                dy = pos[b, j, 1] - p_self[1]
                dz = pos[b, j, 2] - p_self[2]
                dists[k] = np.sqrt(dx*dx + dy*dy + dz*dz)
                indices[k] = j
                k += 1
            
            # Simple selection sort for top-k (N is small)
            for k_idx in range(min(top_k, N - 1)):
                min_idx = k_idx
                for j in range(k_idx + 1, N - 1):
                    if dists[j] < dists[min_idx]:
                        min_idx = j
                # Swap
                dists[k_idx], dists[min_idx] = dists[min_idx], dists[k_idx]
                indices[k_idx], indices[min_idx] = indices[min_idx], indices[k_idx]
                
                # Compute features for this teammate
                j = indices[k_idx]
                w_dx = pos[b, j, 0] - p_self[0]
                w_dy = pos[b, j, 1] - p_self[1]
                w_dz = pos[b, j, 2] - p_self[2]
                
                # Transform to body frame
                b_dx = w_dx * cy + w_dy * sy
                b_dy = -w_dx * sy + w_dy * cy
                
                rvx = vel[b, j, 0] - vel[b, i, 0]
                rvy = vel[b, j, 1] - vel[b, i, 1]
                rvz = vel[b, j, 2] - vel[b, i, 2]
                b_rvx = rvx * cy + rvy * sy
                b_rvy = -rvx * sy + rvy * cy

                base = k_idx * 8
                feats[b, i, base + 0] = b_dx
                feats[b, i, base + 1] = b_dy
                feats[b, i, base + 2] = w_dz
                feats[b, i, base + 3] = dists[k_idx]
                feats[b, i, base + 4] = 1.0 if dists[k_idx] <= view_radius else 0.0
                feats[b, i, base + 5] = b_rvx
                feats[b, i, base + 6] = b_rvy
                feats[b, i, base + 7] = rvz
    
    return feats


@njit(cache=True, parallel=True)
def batch_visibility_mask(
    agent_pos: np.ndarray,     # (B, N, 3)
    target_pos: np.ndarray,    # (B, 3)
    target_vel: np.ndarray,    # (B, 3)
    view_radius: np.ndarray,   # (B,)
    force_visible: np.ndarray, # (B,) bool
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> tuple:                    # (B, N) bool mask, (B,) bool any_visible
    """Compute per-agent visibility mask with LOS and range.
    
    Returns:
        mask: (B, N) bool - per-agent visibility
        any_visible: (B,) bool - whether any agent in each env sees target
    """
    B, N, _ = agent_pos.shape
    mask = np.zeros((B, N), dtype=np.bool_)
    any_visible = np.zeros(B, dtype=np.bool_)
    for b in prange(B):
        if force_visible[b]:
            any_visible[b] = True
            for n in range(N):
                mask[b, n] = True
            continue
        for n in range(N):
            dx = target_pos[b, 0] - agent_pos[b, n, 0]
            dy = target_pos[b, 1] - agent_pos[b, n, 1]
            dz = target_pos[b, 2] - agent_pos[b, n, 2]
            if dx * dx + dy * dy + dz * dz <= view_radius[b] * view_radius[b]:
                if _line_of_sight(agent_pos[b, n], target_pos[b], occupancy_grid, origin, voxel_size):
                    mask[b, n] = True
                    any_visible[b] = True
    return mask, any_visible


@njit(cache=True)
def _line_of_sight(
    start: np.ndarray,
    end: np.ndarray,
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> boolean:
    """Return True if no occupied voxel blocks the line segment."""
    gx, gy, gz = occupancy_grid.shape
    x0 = int((start[0] - origin[0]) / voxel_size)
    y0 = int((start[1] - origin[1]) / voxel_size)
    z0 = int((start[2] - origin[2]) / voxel_size)
    x1 = int((end[0] - origin[0]) / voxel_size)
    y1 = int((end[1] - origin[1]) / voxel_size)
    z1 = int((end[2] - origin[2]) / voxel_size)

    if x0 < 0 or x0 >= gx or y0 < 0 or y0 >= gy or z0 < 0 or z0 >= gz:
        return False
    if x1 < 0 or x1 >= gx or y1 < 0 or y1 >= gy or z1 < 0 or z1 >= gz:
        return False
    if occupancy_grid[x0, y0, z0] == 1 or occupancy_grid[x1, y1, z1] == 1:
        return False

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            x0 += sx
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            if occupancy_grid[x0, y0, z0] == 1:
                return False
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            y0 += sy
            if p1 >= 0:
                x0 += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            if occupancy_grid[x0, y0, z0] == 1:
                return False
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            z0 += sz
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += sx
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            if occupancy_grid[x0, y0, z0] == 1:
                return False

    return True


@njit(cache=True, parallel=True)
def batch_build_observations_v2(
    # Agent state
    pos: np.ndarray,           # (B, N, 3)
    vel: np.ndarray,           # (B, N, 3)
    yaw: np.ndarray,           # (B, N)
    # Target state
    target_pos: np.ndarray,    # (B, 3) target position
    target_vel: np.ndarray,    # (B, 3) target velocity
    # Observation targets (per-agent, may be real target, frontier, or last known)
    obs_target_pos: np.ndarray, # (B, N, 3) per-agent observation target position
    obs_target_vel: np.ndarray, # (B, N, 3) per-agent observation target velocity
    target_mode_flag: np.ndarray, # (B, N) 1.0=tracking, 0.0=searching
    # Perception
    lidar: np.ndarray,         # (B, N, 26)
    guidance: np.ndarray,      # (B, N, 3)
    teammate_feats: np.ndarray,# (B, N, top_k * 8)
    # IMU
    linear_accel: np.ndarray,  # (B, N, 3)
    angular_vel: np.ndarray,   # (B, N, 3)
    # Control delay (per-env, normalized)
    delay_norm: np.ndarray,    # (B,)
    # Tactical slotting (from Hungarian assignment)
    slot_assignments: np.ndarray,  # (B, N) int, -1 means unassigned
    slot_positions: np.ndarray,    # (B, N, 3)
    # Config
    max_speed: float,
    map_size: float,
    num_agents: int,
    z_scale: float,
) -> np.ndarray:               # (B, N, obs_dim) float32
    """Build complete observation vectors for all agents.
    
    Uses per-agent observation targets which may be:
    - Real target position (when visible)
    - Frontier exploration point (when target not visible)
    - Last known target position (fallback)
    """
    B, N, _ = pos.shape
    top_k_feats = teammate_feats.shape[2]
    # +1 target_mode_flag, +1 delay_norm, +num_agents tactical slot-id one-hot, +3 slot relative vector
    obs_dim = 26 + 3 + 3 + 3 + 3 + 3 + top_k_feats + 3 + num_agents + 1 + 1 + num_agents + 3
    obs = np.zeros((B, N, obs_dim), dtype=np.float32)
    
    for b in prange(B):
        for n in range(N):
            y = yaw[b, n]
            cy = np.cos(y)
            sy = np.sin(y)
            
            idx = 0
            
            # 1. LiDAR sectors (26 dims) - already normalized
            for i in range(26):
                obs[b, n, idx] = lidar[b, n, i]
                idx += 1
            
            # 2. Target relative position in body frame (3 dims)
            # Use per-agent observation target (may be real target, frontier, or last known)
            t_dx = obs_target_pos[b, n, 0] - pos[b, n, 0]
            t_dy = obs_target_pos[b, n, 1] - pos[b, n, 1]
            t_dz = obs_target_pos[b, n, 2] - pos[b, n, 2]
            # World to body
            b_dx = t_dx * cy + t_dy * sy
            b_dy = -t_dx * sy + t_dy * cy
            # Normalize XY by map_size, Z uses amplified scale for stronger gradient
            # Z range (~100m) is much smaller than XY (~300m), so we amplify Z signal
            obs[b, n, idx] = max(-1.0, min(1.0, b_dx / map_size))
            obs[b, n, idx + 1] = max(-1.0, min(1.0, b_dy / map_size))
            # Amplify Z signal: use z_scale/3 to make Z differences more prominent
            # This helps the policy learn to actively adjust altitude
            z_den = z_scale if z_scale > 1e-6 else map_size
            z_norm_scale = z_den / 3.0  # Amplify Z signal 3x (e.g., 108m -> 36m normalization)
            obs[b, n, idx + 2] = max(-1.0, min(1.0, t_dz / z_norm_scale))
            idx += 3
            
            # 3. Target relative velocity in body frame (3 dims)
            # Relative target velocity is only meaningful when the target is being tracked.
            if target_mode_flag[b, n] > 0.5:  # tracking mode
                tv_dx = obs_target_vel[b, n, 0] - vel[b, n, 0]
                tv_dy = obs_target_vel[b, n, 1] - vel[b, n, 1]
                tv_dz = obs_target_vel[b, n, 2] - vel[b, n, 2]
                bv_dx = tv_dx * cy + tv_dy * sy
                bv_dy = -tv_dx * sy + tv_dy * cy
                obs[b, n, idx] = max(-1.0, min(1.0, bv_dx / max_speed))
                obs[b, n, idx + 1] = max(-1.0, min(1.0, bv_dy / max_speed))
                obs[b, n, idx + 2] = max(-1.0, min(1.0, tv_dz / max_speed))
            else:  # searching mode
                obs[b, n, idx] = 0.0
                obs[b, n, idx + 1] = 0.0
                obs[b, n, idx + 2] = 0.0
            idx += 3
            
            # 4. Self velocity in body frame (3 dims)
            sv_x = vel[b, n, 0] * cy + vel[b, n, 1] * sy
            sv_y = -vel[b, n, 0] * sy + vel[b, n, 1] * cy
            sv_z = vel[b, n, 2]
            obs[b, n, idx] = max(-1.0, min(1.0, sv_x / max_speed))
            obs[b, n, idx + 1] = max(-1.0, min(1.0, sv_y / max_speed))
            obs[b, n, idx + 2] = max(-1.0, min(1.0, sv_z / max_speed))
            idx += 3

            # 5. IMU linear acceleration in body frame (3 dims)
            obs[b, n, idx] = linear_accel[b, n, 0] / max_speed
            obs[b, n, idx + 1] = linear_accel[b, n, 1] / max_speed
            obs[b, n, idx + 2] = linear_accel[b, n, 2] / max_speed
            idx += 3

            # 6. IMU angular velocity in body frame (3 dims)
            obs[b, n, idx] = angular_vel[b, n, 0]
            obs[b, n, idx + 1] = angular_vel[b, n, 1]
            obs[b, n, idx + 2] = angular_vel[b, n, 2]
            idx += 3
            
            # 7. Teammate features (top_k * 8 dims) - normalize
            # Also amplify Z for teammate position/velocity features
            z_teammate_scale = z_den / 3.0  # Same amplification as target Z
            for i in range(top_k_feats):
                feat_idx = i % 8
                if feat_idx < 2:  # dx, dy
                    obs[b, n, idx] = teammate_feats[b, n, i] / map_size
                elif feat_idx == 2:  # dz - amplified
                    obs[b, n, idx] = teammate_feats[b, n, i] / z_teammate_scale
                elif feat_idx == 3:  # dist
                    obs[b, n, idx] = teammate_feats[b, n, i] / map_size
                elif feat_idx == 4:  # visible (0/1)
                    obs[b, n, idx] = teammate_feats[b, n, i]
                elif feat_idx in (5, 6):  # relative vx, vy in body frame
                    obs[b, n, idx] = teammate_feats[b, n, i] / max_speed
                else:  # relative vz in world frame
                    obs[b, n, idx] = teammate_feats[b, n, i] / max_speed
                idx += 1
            
            # 8. Guidance vector in body frame (3 dims)
            # Guidance is already a unit vector, but amplify Z component for emphasis
            g_x = guidance[b, n, 0]
            g_y = guidance[b, n, 1]
            g_z = guidance[b, n, 2]
            bg_x = g_x * cy + g_y * sy
            bg_y = -g_x * sy + g_y * cy
            obs[b, n, idx] = bg_x
            obs[b, n, idx + 1] = bg_y
            # Amplify guidance Z component to emphasize vertical movement
            obs[b, n, idx + 2] = g_z * 2.0  # 2x amplification for guidance Z
            idx += 3
            
            # 9. Agent ID one-hot (num_agents dims)
            for i in range(num_agents):
                obs[b, n, idx + i] = 1.0 if i == n else 0.0
            idx += num_agents
            
            # 10. Target mode flag (1 dim)
            obs[b, n, idx] = target_mode_flag[b, n]
            idx += 1

            # 11. Action delay (per-env, normalized) (1 dim)
            obs[b, n, idx] = delay_norm[b]
            idx += 1

            # 12. Tactical slot ID one-hot (num_agents dims)
            slot_id = int(slot_assignments[b, n])
            for i in range(num_agents):
                obs[b, n, idx + i] = 1.0 if slot_id == i else 0.0
            idx += num_agents

            # 13. Relative slot vector (body frame, non-normalized shape info, scaled only)
            # Keep distance magnitude by avoiding unit normalization.
            rel_sx = 0.0
            rel_sy = 0.0
            rel_sz = 0.0
            if slot_id >= 0 and slot_id < slot_positions.shape[1]:
                rel_sx = slot_positions[b, slot_id, 0] - pos[b, n, 0]
                rel_sy = slot_positions[b, slot_id, 1] - pos[b, n, 1]
                rel_sz = slot_positions[b, slot_id, 2] - pos[b, n, 2]
            # World -> body (same convention as target_rel features).
            body_sx = rel_sx * cy + rel_sy * sy
            body_sy = -rel_sx * sy + rel_sy * cy
            z_den = z_scale if z_scale > 1e-6 else map_size
            obs[b, n, idx] = body_sx / map_size
            obs[b, n, idx + 1] = body_sy / map_size
            obs[b, n, idx + 2] = rel_sz / z_den
            idx += 3
    
    return obs


# =============================================================================
# Reward Computation Kernels
# =============================================================================

@njit(cache=True, parallel=True)
def batch_compute_rewards_v2(
    pos: np.ndarray,
    vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    obs_target_pos: np.ndarray,
    path_lengths: np.ndarray,
    prev_target_dist: np.ndarray,
    prev_frontier_dist: np.ndarray,
    lidar: np.ndarray,
    guidance: np.ndarray,
    actions: np.ndarray,
    prev_actions: np.ndarray,
    collisions: np.ndarray,
    alive_mask: np.ndarray,
    collision_impact_flags: np.ndarray,
    shield_flags: np.ndarray,
    exploration_deltas: np.ndarray,
    frontier_deltas: np.ndarray,
    reward_core_coefs: np.ndarray,
    capture_radius: np.ndarray,
    lidar_max_range: float,
    frontier_reach_dist: float,
    lidar_safe_distance_m: float,
    search_speed_floor_mps: float,
    direction_gate_active_radius_m: float,
    target_mode_flag: np.ndarray,
    is_potential_unstable: np.ndarray,
    position_history: np.ndarray,
    frontier_goals: np.ndarray,
    world_min: np.ndarray,
    world_max: np.ndarray,
) -> tuple:
    """
    Compute 9-core reward with single-layer coefficients.

    Breakdown order:
    [0:step_cost, 1:progress_gain, 2:exploration_gain, 3:proximity_cost,
     4:collision_cost, 5:direction_gain, 6:control_cost, 7:capture_gain,
     8:capture_quality_gain, 9:reward_total]
    """
    B, N, _ = pos.shape
    rewards = np.zeros((B, N), dtype=np.float64)
    new_target_dist = np.zeros((B, N), dtype=np.float64)
    new_frontier_dist = np.zeros((B, N), dtype=np.float64)
    captured = np.zeros(B, dtype=np.bool_)
    breakdown = np.zeros((B, 10), dtype=np.float64)

    c_step = reward_core_coefs[0]
    c_progress = reward_core_coefs[1]
    c_explore = reward_core_coefs[2]
    c_proximity = reward_core_coefs[3]
    c_collision = reward_core_coefs[4]
    c_direction = reward_core_coefs[5]
    c_control = reward_core_coefs[6]
    c_capture = reward_core_coefs[7]
    c_capture_quality = reward_core_coefs[8]

    directions = np.zeros((26, 3), dtype=np.float64)
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                norm = np.sqrt(float(dx * dx + dy * dy + dz * dz))
                directions[idx, 0] = dx / norm
                directions[idx, 1] = dy / norm
                directions[idx, 2] = dz / norm
                idx += 1

    for b in prange(B):
        cap_r = capture_radius[b]
        any_captured = False
        b_step = 0.0
        b_progress = 0.0
        b_explore = 0.0
        b_proximity = 0.0
        b_collision = 0.0
        b_direction = 0.0
        b_control = 0.0
        b_capture = 0.0
        b_capture_quality = 0.0
        b_total = 0.0

        # Team uniformity signal for capture quality.
        u_sum_x = 0.0
        u_sum_y = 0.0
        u_sum_z = 0.0
        alive_count = 0.0
        for n in range(N):
            if not alive_mask[b, n]:
                continue
            dx_t = pos[b, n, 0] - target_pos[b, 0]
            dy_t = pos[b, n, 1] - target_pos[b, 1]
            dz_t = pos[b, n, 2] - target_pos[b, 2]
            d_t = np.sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t)
            if d_t > 1e-6:
                inv_d = 1.0 / d_t
                u_sum_x += dx_t * inv_d
                u_sum_y += dy_t * inv_d
                u_sum_z += dz_t * inv_d
            alive_count += 1.0
        denom = alive_count if alive_count > 1.0 else 1.0
        team_uniformity = np.sqrt(u_sum_x * u_sum_x + u_sum_y * u_sum_y + u_sum_z * u_sum_z) / denom
        if team_uniformity < 0.0:
            team_uniformity = 0.0
        elif team_uniformity > 1.0:
            team_uniformity = 1.0

        wall_count = 0.0
        wall_thr = 2.0
        if target_pos[b, 0] < world_min[0] + wall_thr:
            wall_count += 1.0
        if target_pos[b, 0] > world_max[0] - wall_thr:
            wall_count += 1.0
        if target_pos[b, 1] < world_min[1] + wall_thr:
            wall_count += 1.0
        if target_pos[b, 1] > world_max[1] - wall_thr:
            wall_count += 1.0
        if target_pos[b, 2] < world_min[2] + wall_thr:
            wall_count += 1.0
        if target_pos[b, 2] > world_max[2] - wall_thr:
            wall_count += 1.0
        wall_support = wall_count / 6.0

        for n in range(N):
            if not alive_mask[b, n]:
                new_target_dist[b, n] = prev_target_dist[b, n]
                new_frontier_dist[b, n] = prev_frontier_dist[b, n]
                continue

            dx = pos[b, n, 0] - target_pos[b, 0]
            dy = pos[b, n, 1] - target_pos[b, 1]
            dz = pos[b, n, 2] - target_pos[b, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= cap_r:
                any_captured = True

            dist_pot = path_lengths[b, n]
            if dist_pot <= 0.0:
                pdx = pos[b, n, 0] - obs_target_pos[b, n, 0]
                pdy = pos[b, n, 1] - obs_target_pos[b, n, 1]
                pdz = pos[b, n, 2] - obs_target_pos[b, n, 2]
                dist_pot = np.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
            new_target_dist[b, n] = dist_pot

            # 1) step_cost
            comp_step = c_step * 1.0

            # 2) progress_gain (signed delta, mode-aware)
            progress_term = 0.0
            current_frontier_dist = prev_frontier_dist[b, n]
            if is_potential_unstable[b, n]:
                new_frontier_dist[b, n] = prev_frontier_dist[b, n]
            elif target_mode_flag[b, n] > 0.5:
                progress_term = prev_target_dist[b, n] - dist_pot
                new_frontier_dist[b, n] = prev_frontier_dist[b, n]
            else:
                current_frontier_dist = path_lengths[b, n]
                if current_frontier_dist <= 0.0:
                    fx = frontier_goals[b, n, 0] - pos[b, n, 0]
                    fy = frontier_goals[b, n, 1] - pos[b, n, 1]
                    fz = frontier_goals[b, n, 2] - pos[b, n, 2]
                    current_frontier_dist = np.sqrt(fx * fx + fy * fy + fz * fz)
                new_frontier_dist[b, n] = current_frontier_dist
                progress_term = prev_frontier_dist[b, n] - current_frontier_dist
                if frontier_deltas[b, n] > 0.0:
                    progress_term = 0.0
            comp_progress = c_progress * progress_term

            # 3) exploration_gain (search only)
            explore_term = 0.0
            if target_mode_flag[b, n] <= 0.5:
                exp_gain = max(exploration_deltas[b, n], 0.0)
                frontier_cross = 0.0
                if frontier_reach_dist > 0.0:
                    prev_fd = prev_frontier_dist[b, n]
                    crossed = (prev_fd > frontier_reach_dist) and (current_frontier_dist <= frontier_reach_dist)
                    if crossed:
                        frontier_cross = 1.0
                explore_term = exp_gain + frontier_cross
            comp_explore = c_explore * explore_term

            # 4) proximity_cost (risk score)
            proximity_term = 0.0
            if lidar_safe_distance_m > 1e-6:
                min_lidar = 1.0
                min_idx = 0
                for i in range(26):
                    if lidar[b, n, i] < min_lidar:
                        min_lidar = lidar[b, n, i]
                        min_idx = i
                min_dist_m = min_lidar * lidar_max_range
                if min_dist_m < lidar_safe_distance_m:
                    obs_dir = directions[min_idx]
                    approach_speed = (
                        vel[b, n, 0] * obs_dir[0]
                        + vel[b, n, 1] * obs_dir[1]
                        + vel[b, n, 2] * obs_dir[2]
                    )
                    if approach_speed > 0.0:
                        base_risk = (lidar_safe_distance_m - min_dist_m) / lidar_safe_distance_m
                        speed = np.sqrt(
                            vel[b, n, 0] * vel[b, n, 0]
                            + vel[b, n, 1] * vel[b, n, 1]
                            + vel[b, n, 2] * vel[b, n, 2]
                        )
                        boost = 1.0
                        if speed >= 3.0 and min_dist_m < lidar_safe_distance_m * 0.5:
                            boost = 2.5
                        proximity_term = base_risk * boost * (1.0 + 0.25 * approach_speed)
            comp_proximity = c_proximity * proximity_term

            # 5) collision_cost
            collision_term = 0.0
            if collision_impact_flags[b, n]:
                collision_term += 1.0
            if collisions[b, n]:
                collision_term += 0.5
            if shield_flags[b, n]:
                collision_term += 0.35
            comp_collision = c_collision * collision_term

            # 6) direction_gain
            v_norm = np.sqrt(vel[b, n, 0] * vel[b, n, 0] + vel[b, n, 1] * vel[b, n, 1] + vel[b, n, 2] * vel[b, n, 2])
            g_norm = np.sqrt(guidance[b, n, 0] * guidance[b, n, 0] + guidance[b, n, 1] * guidance[b, n, 1] + guidance[b, n, 2] * guidance[b, n, 2])
            align_cos = 0.0
            if v_norm > 1e-6 and g_norm > 1e-6:
                align_cos = (
                    vel[b, n, 0] * guidance[b, n, 0]
                    + vel[b, n, 1] * guidance[b, n, 1]
                    + vel[b, n, 2] * guidance[b, n, 2]
                ) / (v_norm * g_norm)
            direction_term = max(0.0, align_cos)
            if align_cos < -0.2:
                direction_term -= 1.0
            # P1: distance-lock for direction gain/penalty.
            # No direction incentive outside active_radius.
            active_radius = direction_gate_active_radius_m
            if active_radius <= 0.0:
                dist_factor = 1.0
            else:
                inner_radius = active_radius * 0.5
                if dist >= active_radius:
                    dist_factor = 0.0
                elif dist <= inner_radius:
                    dist_factor = 1.0
                else:
                    denom = max(active_radius - inner_radius, 1e-6)
                    dist_factor = (active_radius - dist) / denom
            direction_term *= dist_factor
            comp_direction = c_direction * direction_term

            # 7) control_cost
            action_diff = 0.0
            for i in range(4):
                d_act = actions[b, n, i] - prev_actions[b, n, i]
                action_diff += d_act * d_act
            steer_jitter = abs(actions[b, n, 3] - prev_actions[b, n, 3])
            steer_term = steer_jitter / (1.0 + 0.5 * v_norm)
            control_term = 0.4 * action_diff + 0.6 * steer_term
            if v_norm < 0.4 and dist >= 10.0:
                control_term += 1.0
            if target_mode_flag[b, n] <= 0.5 and search_speed_floor_mps > 1e-6 and v_norm < search_speed_floor_mps:
                control_term += (search_speed_floor_mps - v_norm) / search_speed_floor_mps
            # Tracking anti-lazy penalty: far from target but low closing speed.
            if target_mode_flag[b, n] > 0.5 and dist >= max(2.5 * cap_r, 25.0):
                inv_dist = 1.0 / max(dist, 1e-6)
                ux = (target_pos[b, 0] - pos[b, n, 0]) * inv_dist
                uy = (target_pos[b, 1] - pos[b, n, 1]) * inv_dist
                uz = (target_pos[b, 2] - pos[b, n, 2]) * inv_dist
                rvx = vel[b, n, 0] - target_vel[b, 0]
                rvy = vel[b, n, 1] - target_vel[b, 1]
                rvz = vel[b, n, 2] - target_vel[b, 2]
                closing_speed = rvx * ux + rvy * uy + rvz * uz
                if closing_speed < 0.8:
                    speed_deficit = (0.8 - closing_speed) / 0.8
                    if speed_deficit > 1.5:
                        speed_deficit = 1.5
                    control_term += 1.0 * speed_deficit
            action_mag = np.sqrt(actions[b, n, 0] * actions[b, n, 0] + actions[b, n, 1] * actions[b, n, 1] + actions[b, n, 2] * actions[b, n, 2])
            if action_mag > 1.0:
                pos_current = pos[b, n]
                total_displacement = 0.0
                for h in range(5):
                    hist_pos = position_history[b, n, h]
                    ddx = pos_current[0] - hist_pos[0]
                    ddy = pos_current[1] - hist_pos[1]
                    ddz = pos_current[2] - hist_pos[2]
                    total_displacement += np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                if total_displacement < 0.3:
                    control_term += 1.5
            comp_control = c_control * control_term

            r = (
                comp_step
                + comp_progress
                + comp_explore
                + comp_proximity
                + comp_collision
                + comp_direction
                + comp_control
            )
            rewards[b, n] = r
            b_step += comp_step
            b_progress += comp_progress
            b_explore += comp_explore
            b_proximity += comp_proximity
            b_collision += comp_collision
            b_direction += comp_direction
            b_control += comp_control
            b_total += r

        if any_captured:
            quality_base = 1.0 - team_uniformity
            if quality_base < 0.0:
                quality_base = 0.0
            elif quality_base > 1.0:
                quality_base = 1.0
            quality_base = min(1.0, quality_base + 0.25 * wall_support)
            contrib_raw = np.zeros(N, dtype=np.float64)
            quality_raw = np.zeros(N, dtype=np.float64)
            alive_count_capture = 0.0
            effective_chasers = 0.0
            contrib_radius = max(3.0 * cap_r, 30.0)
            contribution_cutoff_dist = 60.0
            contrib_sum = 0.0
            for n in range(N):
                if not alive_mask[b, n]:
                    continue
                alive_count_capture += 1.0
                dx = pos[b, n, 0] - target_pos[b, 0]
                dy = pos[b, n, 1] - target_pos[b, 1]
                dz = pos[b, n, 2] - target_pos[b, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                prox_score = 0.0
                if contrib_radius > 1e-6:
                    prox_score = (contrib_radius - dist) / contrib_radius
                    if prox_score < 0.0:
                        prox_score = 0.0
                    elif prox_score > 1.0:
                        prox_score = 1.0
                rel_vx = vel[b, n, 0] - target_vel[b, 0]
                rel_vy = vel[b, n, 1] - target_vel[b, 1]
                rel_vz = vel[b, n, 2] - target_vel[b, 2]
                inv_dist = 1.0 / max(dist, 1e-6)
                ux = (target_pos[b, 0] - pos[b, n, 0]) * inv_dist
                uy = (target_pos[b, 1] - pos[b, n, 1]) * inv_dist
                uz = (target_pos[b, 2] - pos[b, n, 2]) * inv_dist
                closing = rel_vx * ux + rel_vy * uy + rel_vz * uz
                close_score = closing / 4.0
                if close_score < 0.0:
                    close_score = 0.0
                elif close_score > 1.0:
                    close_score = 1.0
                # P0: zero-contribution should yield zero capture share.
                raw = 0.70 * prox_score + 0.25 * close_score
                if dist <= cap_r:
                    raw += 0.30
                if dist > contribution_cutoff_dist:
                    raw = 0.0
                contrib_raw[n] = raw
                contrib_sum += raw
                if dist <= contrib_radius and closing > 0.5:
                    effective_chasers += 1.0

                intercept_quality = 0.0
                if dist < max(2.0 * cap_r, 25.0):
                    v_dot = (
                        vel[b, n, 0] * target_vel[b, 0]
                        + vel[b, n, 1] * target_vel[b, 1]
                        + vel[b, n, 2] * target_vel[b, 2]
                    )
                    if v_dot < 0.0:
                        intercept_quality = min((-v_dot) / 10.0, 1.0)
                dvx = vel[b, n, 0] - target_vel[b, 0]
                dvy = vel[b, n, 1] - target_vel[b, 1]
                dvz = vel[b, n, 2] - target_vel[b, 2]
                sync_quality = 1.0 / (1.0 + dvx * dvx + dvy * dvy + dvz * dvz)
                quality_raw[n] = 0.5 * quality_base + 0.3 * intercept_quality + 0.2 * sync_quality

            if alive_count_capture < 1.0:
                alive_count_capture = 1.0
            if contrib_sum <= 1e-8:
                contrib_sum = alive_count_capture
                for n in range(N):
                    if alive_mask[b, n]:
                        contrib_raw[n] = 1.0

            participation_scale = 1.0
            participation_ratio = effective_chasers / alive_count_capture
            if participation_ratio < 0.5:
                participation_scale = 0.6 + 0.4 * (participation_ratio / 0.5)
            total_capture_reward = c_capture * alive_count_capture * participation_scale
            for n in range(N):
                if not alive_mask[b, n]:
                    continue
                w = contrib_raw[n] / contrib_sum
                comp_capture = total_capture_reward * w
                # Strict contribution scaling: zero contribution => zero quality reward.
                w_scaled = w * alive_count_capture
                quality_scale = w_scaled
                comp_quality = c_capture_quality * quality_raw[n] * quality_scale * participation_scale
                rewards[b, n] += comp_capture + comp_quality
                b_capture += comp_capture
                b_capture_quality += comp_quality
                b_total += comp_capture + comp_quality

        breakdown[b, 0] = b_step
        breakdown[b, 1] = b_progress
        breakdown[b, 2] = b_explore
        breakdown[b, 3] = b_proximity
        breakdown[b, 4] = b_collision
        breakdown[b, 5] = b_direction
        breakdown[b, 6] = b_control
        breakdown[b, 7] = b_capture
        breakdown[b, 8] = b_capture_quality
        breakdown[b, 9] = b_total
        captured[b] = any_captured

    return rewards, new_target_dist, new_frontier_dist, captured, breakdown
# =============================================================================
# Spawn/Respawn Kernels
# =============================================================================

@njit(cache=True)
def _find_free_position_single(
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    spawn_min: np.ndarray,
    spawn_max: np.ndarray,
    exclude_positions: np.ndarray,  # (M, 3)
    min_separation: float,
    rng_vals: np.ndarray,           # (max_attempts, 3)
    max_attempts: int,
) -> np.ndarray:
    """Find a single free position."""
    gx, gy, gz = occupancy_grid.shape
    
    for attempt in range(max_attempts):
        # Generate random position
        pos = np.zeros(3, dtype=np.float64)
        pos[0] = spawn_min[0] + rng_vals[attempt, 0] * (spawn_max[0] - spawn_min[0])
        pos[1] = spawn_min[1] + rng_vals[attempt, 1] * (spawn_max[1] - spawn_min[1])
        pos[2] = spawn_min[2] + rng_vals[attempt, 2] * (spawn_max[2] - spawn_min[2])
        
        # Check collision with obstacles
        ix = int(np.floor((pos[0] - origin[0]) / voxel_size))
        iy = int(np.floor((pos[1] - origin[1]) / voxel_size))
        iz = int(np.floor((pos[2] - origin[2]) / voxel_size))
        
        if ix < 0 or ix >= gx or iy < 0 or iy >= gy or iz < 0 or iz >= gz:
            continue
        if occupancy_grid[ix, iy, iz] == 1:
            continue
        
        # Check separation from excluded positions
        too_close = False
        for i in range(exclude_positions.shape[0]):
            dx = pos[0] - exclude_positions[i, 0]
            dy = pos[1] - exclude_positions[i, 1]
            dz = pos[2] - exclude_positions[i, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < min_separation:
                too_close = True
                break
        
        if not too_close:
            return pos
    
    # Fallback: center of spawn area
    return np.array([
        (spawn_min[0] + spawn_max[0]) / 2,
        (spawn_min[1] + spawn_max[1]) / 2,
        (spawn_min[2] + spawn_max[2]) / 2,
    ], dtype=np.float64)


@njit(cache=True, parallel=True)
def batch_respawn_envs_v2(
    done_mask: np.ndarray,     # (B,) bool
    # State arrays (in/out)
    pos: np.ndarray,           # (B, N, 3)
    vel: np.ndarray,           # (B, N, 3)
    yaw: np.ndarray,           # (B, N)
    prev_actions: np.ndarray,  # (B, N, 4)
    target_pos: np.ndarray,    # (B, 3)
    target_vel: np.ndarray,    # (B, 3)
    steps: np.ndarray,         # (B,)
    prev_target_dist: np.ndarray, # (B, N)
    current_goals: np.ndarray, # (B, N, 3)
    # Map
    occupancy_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    # Spawn config
    spawn_min: np.ndarray,
    spawn_max: np.ndarray,
    min_separation: float,
    target_min_dist: float,
    target_max_dist: float,
    # Random values (pre-generated)
    rng_vals: np.ndarray,      # (B, N+1, max_attempts, 3)
    yaw_rng: np.ndarray,       # (B, N)
) -> None:
    """Batch respawn all done environments."""
    B = done_mask.shape[0]
    N = pos.shape[1]
    max_attempts = rng_vals.shape[2]
    
    for b in prange(B):
        if not done_mask[b]:
            continue
        
        # Reset step counter
        steps[b] = 0
        
        # Spawn agents first
        exclude = np.zeros((N, 3), dtype=np.float64)
        exclude_count = 0
        
        for n in range(N):
            agent_pos = _find_free_position_single(
                occupancy_grid, origin, voxel_size,
                spawn_min, spawn_max, exclude[:exclude_count],
                min_separation, rng_vals[b, n + 1], max_attempts
            )
            pos[b, n] = agent_pos
            vel[b, n, 0] = 0.0
            vel[b, n, 1] = 0.0
            vel[b, n, 2] = 0.0
            yaw[b, n] = yaw_rng[b, n] * 2 * PI - PI
            prev_actions[b, n, 0] = 0.0
            prev_actions[b, n, 1] = 0.0
            prev_actions[b, n, 2] = 0.0
            prev_actions[b, n, 3] = 0.0
            
            # Update exclude list
            exclude[exclude_count, 0] = agent_pos[0]
            exclude[exclude_count, 1] = agent_pos[1]
            exclude[exclude_count, 2] = agent_pos[2]
            exclude_count += 1
            
        # Spawn target with distance constraint to all agents (min distance only).
        tgt = np.zeros(3, dtype=np.float64)
        best_tgt = np.zeros(3, dtype=np.float64)
        best_min_dist = -1.0
        found = False
        for attempt in range(max_attempts):
            tgt[0] = spawn_min[0] + rng_vals[b, 0, attempt, 0] * (spawn_max[0] - spawn_min[0])
            tgt[1] = spawn_min[1] + rng_vals[b, 0, attempt, 1] * (spawn_max[1] - spawn_min[1])
            tgt[2] = spawn_min[2] + rng_vals[b, 0, attempt, 2] * (spawn_max[2] - spawn_min[2])

            ix = int(np.floor((tgt[0] - origin[0]) / voxel_size))
            iy = int(np.floor((tgt[1] - origin[1]) / voxel_size))
            iz = int(np.floor((tgt[2] - origin[2]) / voxel_size))
            if ix < 0 or ix >= occupancy_grid.shape[0] or iy < 0 or iy >= occupancy_grid.shape[1] or iz < 0 or iz >= occupancy_grid.shape[2]:
                continue
            if occupancy_grid[ix, iy, iz] == 1:
                continue

            min_dist = 1e9
            for n in range(N):
                dx = pos[b, n, 0] - tgt[0]
                dy = pos[b, n, 1] - tgt[1]
                dz = pos[b, n, 2] - tgt[2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < min_dist:
                    min_dist = dist

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_tgt[0] = tgt[0]
                best_tgt[1] = tgt[1]
                best_tgt[2] = tgt[2]

            if min_dist >= target_min_dist:
                found = True
                break

        if not found:
            if best_min_dist >= 0.0:
                tgt[0] = best_tgt[0]
                tgt[1] = best_tgt[1]
                tgt[2] = best_tgt[2]
            else:
                fallback = _find_free_position_single(
                    occupancy_grid, origin, voxel_size,
                    spawn_min, spawn_max, exclude[:exclude_count],
                    target_min_dist, rng_vals[b, 0], max_attempts
                )
                tgt[0] = fallback[0]
                tgt[1] = fallback[1]
                tgt[2] = fallback[2]

        target_pos[b] = tgt
        target_vel[b, 0] = 0.0
        target_vel[b, 1] = 0.0
        target_vel[b, 2] = 0.0

        for n in range(N):
            # Initial goal = target
            current_goals[b, n] = tgt

            # Initial distance
            dx = pos[b, n, 0] - tgt[0]
            dy = pos[b, n, 1] - tgt[1]
            dz = pos[b, n, 2] - tgt[2]
            prev_target_dist[b, n] = np.sqrt(dx*dx + dy*dy + dz*dz)


# =============================================================================
# Navigation/Guidance Kernels
# =============================================================================

@njit(cache=True, parallel=True)
def batch_compute_simple_guidance(
    agent_pos: np.ndarray,     # (B, N, 3)
    goals: np.ndarray,         # (B, N, 3)
) -> np.ndarray:               # (B, N, 3) unit vectors
    """
    Simple direct guidance: unit vector from agent to goal.
    Use when A* is too expensive or as fallback.
    """
    B, N, _ = agent_pos.shape
    guidance = np.zeros((B, N, 3), dtype=np.float64)
    
    for b in prange(B):
        for n in range(N):
            dx = goals[b, n, 0] - agent_pos[b, n, 0]
            dy = goals[b, n, 1] - agent_pos[b, n, 1]
            dz = goals[b, n, 2] - agent_pos[b, n, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist > 1e-6:
                guidance[b, n, 0] = dx / dist
                guidance[b, n, 1] = dy / dist
                guidance[b, n, 2] = dz / dist
    
    return guidance


@njit(cache=True, parallel=True)
def batch_select_goals(
    agent_pos: np.ndarray,     # (B, N, 3)
    target_pos: np.ndarray,    # (B, 3)
    frontier_goals: np.ndarray,# (B, N, 3) pre-allocated frontier goals
    visible_mask: np.ndarray,  # (B, N) bool
    guidance_enabled: np.ndarray, # (B,) bool
) -> np.ndarray:               # (B, N, 3) selected goals
    """
    Select goals based on target visibility mask.
    If visible and guidance enabled: use target.
    Otherwise: use frontier goal.
    """
    B, N, _ = agent_pos.shape
    goals = np.zeros((B, N, 3), dtype=np.float64)
    
    for b in prange(B):
        for n in range(N):
            if guidance_enabled[b] and visible_mask[b, n]:
                goals[b, n] = target_pos[b]
            else:
                goals[b, n] = frontier_goals[b, n]
    
    return goals


# =============================================================================
# Utility Kernels
# =============================================================================

@njit(cache=True, parallel=True)
def batch_compute_agent_centroids(
    pos: np.ndarray,           # (B, N, 3)
) -> np.ndarray:               # (B, 3)
    """Compute centroid of agents for each environment."""
    B, N, _ = pos.shape
    centroids = np.zeros((B, 3), dtype=np.float64)
    
    for b in prange(B):
        for n in range(N):
            centroids[b, 0] += pos[b, n, 0]
            centroids[b, 1] += pos[b, n, 1]
            centroids[b, 2] += pos[b, n, 2]
        centroids[b] /= N
    
    return centroids


@njit(cache=True, parallel=True)
def batch_apply_action_smoothing(
    actions: np.ndarray,       # (B, N, 4) in/out
    prev_actions: np.ndarray,  # (B, N, 4)
    alpha: float,
    deadband: float,
) -> None:
    """Apply EMA smoothing and deadband to actions."""
    B, N, _ = actions.shape
    
    for b in prange(B):
        for n in range(N):
            for i in range(4):
                # EMA smoothing
                actions[b, n, i] = alpha * actions[b, n, i] + (1.0 - alpha) * prev_actions[b, n, i]
                # Deadband
                if abs(actions[b, n, i]) < deadband:
                    actions[b, n, i] = 0.0

@njit(cache=True, parallel=True)
def batch_update_exploration_grid_optimized(
    # State
    pos: np.ndarray,            # (B, N, 3)
    lidar_dist: np.ndarray,     # (B, N, 26) Normalized distance [0, 1]
    # Maps (In/Out)
    batch_timestamps: np.ndarray,  # (B, Gx, Gy, Gz) int32: last visited step
    current_steps: np.ndarray,  # (B,) current step per env
    # Config
    origin: np.ndarray,         # (3,)
    voxel_size: float,
    lidar_max_range: float,
    steps_to_reexplore: int,    # steps until confidence falls below threshold
) -> np.ndarray:                # (B, N) int32: new_voxels_count
    """
    Batch update exploration grid using timestamps to avoid global decay.
    """
    B, N, _ = pos.shape
    Gx, Gy, Gz = batch_timestamps.shape[1:]
    new_counts = np.zeros((B, N), dtype=np.int32)

    # Pre-compute ray directions (same as lidar simulation)
    # This must match the order in simulate_lidar_26_sectors
    directions = np.zeros((26, 3), dtype=np.float32)
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                norm = np.sqrt(float(dx*dx + dy*dy + dz*dz))
                directions[idx, 0] = dx / norm
                directions[idx, 1] = dy / norm
                directions[idx, 2] = dz / norm
                idx += 1

    for b in prange(B):
        curr_step = current_steps[b]
        # Update exploration state via ray tracing
        for n in range(N):
            px, py, pz = pos[b, n]

            # For each ray
            for i in range(26):
                # Calculate endpoint based on lidar reading
                # Note: lidar_dist is normalized [0, 1]
                # If dist < 1.0, it hit an obstacle. We trace up to that hit.
                # If dist == 1.0, it's max range. We trace full range.

                dist_m = lidar_dist[b, n, i] * lidar_max_range

                dx = directions[i, 0]
                dy = directions[i, 1]
                dz = directions[i, 2]

                # Bresenham-like stepping
                # Step size = voxel size
                steps = int(dist_m / voxel_size)

                curr_x = px
                curr_y = py
                curr_z = pz

                for s in range(steps):
                    # Move
                    curr_x += dx * voxel_size
                    curr_y += dy * voxel_size
                    curr_z += dz * voxel_size

                    # To Grid Index
                    ix = int((curr_x - origin[0]) / voxel_size)
                    iy = int((curr_y - origin[1]) / voxel_size)
                    iz = int((curr_z - origin[2]) / voxel_size)

                    # Bounds Check
                    if 0 <= ix < Gx and 0 <= iy < Gy and 0 <= iz < Gz:
                        last_step = batch_timestamps[b, ix, iy, iz]
                        if last_step == 0 or (curr_step - last_step) > steps_to_reexplore:
                            batch_timestamps[b, ix, iy, iz] = curr_step
                            new_counts[b, n] += 1

    return new_counts

@njit(cache=True)
def _bresenham_2d_blocked(
    occ: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    x = x0
    y = y0

    while True:
        if occ[x, y] == 1:
            return True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return False

@njit(cache=True, parallel=True)
def batch_update_exploration_grid_2p5d(
    pos: np.ndarray,                 # (B, N, 3)
    batch_layer_timestamps: np.ndarray,  # (B, Gx, Gy, 3) int32
    current_steps: np.ndarray,       # (B,)
    origin: np.ndarray,              # (3,)
    voxel_size: float,
    layer_z_bounds: np.ndarray,      # (3, 2) world z ranges
    sample_offsets: np.ndarray,      # (M, 2) int32 offsets in grid cells
    static_layer_occ: np.ndarray,    # (Gx, Gy, 3) int8
    steps_to_reexplore: int,
) -> np.ndarray:                     # (B, N) int32 new counts
    B, N, _ = pos.shape
    Gx, Gy, _ = batch_layer_timestamps.shape[1:]
    new_counts = np.zeros((B, N), dtype=np.int32)

    for b in prange(B):
        curr_step = current_steps[b]
        for n in range(N):
            px = pos[b, n, 0]
            py = pos[b, n, 1]
            pz = pos[b, n, 2]

            # Determine layer index by z
            layer_idx = 0
            if pz >= layer_z_bounds[1, 0]:
                layer_idx = 1
            if pz >= layer_z_bounds[2, 0]:
                layer_idx = 2

            ix = int((px - origin[0]) / voxel_size)
            iy = int((py - origin[1]) / voxel_size)
            if ix < 0 or ix >= Gx or iy < 0 or iy >= Gy:
                continue

            occ_layer = static_layer_occ[:, :, layer_idx]
            ts_layer = batch_layer_timestamps[b, :, :, layer_idx]

            # Refresh cells that are unseen or have decayed below the re-exploration threshold.
            for k in range(sample_offsets.shape[0]):
                ox = sample_offsets[k, 0]
                oy = sample_offsets[k, 1]
                nx = ix + ox
                ny = iy + oy
                if nx < 0 or nx >= Gx or ny < 0 or ny >= Gy:
                    continue
                # Skip occupied cells in the sampled layer.
                if occ_layer[nx, ny] == 1:
                    continue

                last_step = ts_layer[nx, ny]
                if last_step == 0 or (curr_step - last_step) > steps_to_reexplore:
                    ts_layer[nx, ny] = curr_step
                    new_counts[b, n] += 1

    return new_counts

@njit(cache=True, parallel=True)
def batch_update_exploration_grid(
    pos: np.ndarray,
    lidar_dist: np.ndarray,
    batch_timestamps: np.ndarray,
    current_steps: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    lidar_max_range: float,
    steps_to_reexplore: int,
) -> np.ndarray:
    """Backward-compatible wrapper for optimized exploration update."""
    return batch_update_exploration_grid_optimized(
        pos,
        lidar_dist,
        batch_timestamps,
        current_steps,
        origin,
        voxel_size,
        lidar_max_range,
        steps_to_reexplore,
    )


@njit(cache=True, parallel=True)
def batch_apply_safety_shield(
    actions: np.ndarray,       # (B, N, 4) In/Out [vx, vy, vz, yaw_rate]
    lidar_dists: np.ndarray,   # (B, N, 26) Normalized distances [0, 1]
    min_dist_m: float,         # Safety threshold (e.g. 2.0m)
    lidar_max_range: float,    # Max lidar range (e.g. 50m)
    target_dist: np.ndarray,   # (B, N) Distance to target per agent
    capture_radius: np.ndarray,# (B,) Capture radius per env
) -> np.ndarray:               # (B, N) bool - Triggered flags
    """
    Apply Safety Shield: Modifies actions in-place to prevent collisions.
    Damping shield:
    - If within min_dist, limit inward velocity component.
    - No repulsive force; clamp inward component to 0 at contact.
    - Near target (within 2x capture radius): reduce threshold to allow capture
    """
    B, N, _ = actions.shape
    triggered = np.zeros((B, N), dtype=np.bool_)
    
    # Pre-compute ray directions (Must match Lidar simulation order!)
    # 26 directions: 3x3x3 grid minus center
    directions = np.zeros((26, 3), dtype=np.float64)
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0: continue
                norm = np.sqrt(float(dx*dx + dy*dy + dz*dz))
                directions[idx, 0] = dx / norm
                directions[idx, 1] = dy / norm
                directions[idx, 2] = dz / norm
                idx += 1

    for b in prange(B):
        cap_r = capture_radius[b]
        for n in range(N):
            # Adaptive threshold: reduce when close to target to allow capture
            dist_to_target = target_dist[b, n]
            if dist_to_target < cap_r * 2.5:
                # Near target: use reduced threshold (50% of normal)
                effective_min_dist = min_dist_m * 0.5
            else:
                effective_min_dist = min_dist_m
            
            # 1. Find nearest obstacle
            min_val = 1.0
            min_idx = -1
            
            for k in range(26):
                d = lidar_dists[b, n, k]
                if d < min_val:
                    min_val = d
                    min_idx = k
            
            # Check threshold
            current_dist_m = min_val * lidar_max_range
            
            if current_dist_m < effective_min_dist and min_idx != -1:
                triggered[b, n] = True
                
                # Get obstacle direction vector (Body Frame -> World Frame)
                # Note: Lidar is usually Body Frame. Actions are Body Frame.
                # So we can operate directly in Body Frame!
                obs_dir = directions[min_idx] # (3,)
                
                # Current command velocity
                cmd_vel = actions[b, n, :3]
                
                # 2. Limit velocity towards obstacle (Project & Damp)
                # proj = v . n
                proj = cmd_vel[0]*obs_dir[0] + cmd_vel[1]*obs_dir[1] + cmd_vel[2]*obs_dir[2]
                
                # If moving towards obstacle (proj > 0), damp it based on distance
                if proj > 0:
                    damp = current_dist_m / effective_min_dist
                    new_proj = proj * damp
                    delta = proj - new_proj
                    cmd_vel[0] -= delta * obs_dir[0]
                    cmd_vel[1] -= delta * obs_dir[1]
                    cmd_vel[2] -= delta * obs_dir[2]
                
                # Update action
                actions[b, n, 0] = cmd_vel[0]
                actions[b, n, 1] = cmd_vel[1]
                actions[b, n, 2] = cmd_vel[2]
                # Yaw rate usually doesn't need modification for simple shield
                
    return triggered


@njit(cache=True, parallel=True)
def batch_init_position_history(
    position_history: np.ndarray,  # (B, N, 5, 3)
    history_idx: np.ndarray,       # (B, N)
    current_pos: np.ndarray,       # (B, N, 3)
) -> None:
    """Initialize position history with current positions to fix stagnation detection."""
    B, N, _, _ = position_history.shape
    
    for b in prange(B):
        for n in range(N):
            # Fill all history slots with current position
            for h in range(5):
                position_history[b, n, h, 0] = current_pos[b, n, 0]
                position_history[b, n, h, 1] = current_pos[b, n, 1]
                position_history[b, n, h, 2] = current_pos[b, n, 2]
            # Reset history index
            history_idx[b, n] = 0


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Physics
    "batch_update_agents_v2",
    "batch_update_targets_v2",
    # Collision
    "batch_check_collisions_v2",
    "batch_check_team_collisions",
    "batch_rollback_collisions_v2",
    "batch_resolve_collisions_sliding",
    # LiDAR
    "batch_simulate_lidar_v2",
    # Observations
    "batch_build_teammate_features",
    "batch_build_observations_v2",
    # Rewards
    "batch_compute_rewards_v2",
    # Spawn
    "batch_respawn_envs_v2",
    # Navigation
    "batch_compute_simple_guidance",
    "batch_select_goals",
    "batch_visibility_mask",
    # Utility
    "batch_compute_agent_centroids",
    "batch_apply_action_smoothing",
    "batch_update_exploration_grid",
    "batch_apply_safety_shield",
    "batch_init_position_history",
    "batch_update_exploration_grid_optimized",
    "batch_update_exploration_grid_2p5d",
    "batch_update_path_lengths",
    "batch_check_lkp_reached",
    "batch_update_obs_targets",
]
