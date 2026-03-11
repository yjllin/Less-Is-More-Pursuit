import numpy as np

from src.environment.batch_kernels import batch_compute_rewards_v2


def _base_inputs(unstable: bool = False):
    B, N = 1, 1
    pos = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    vel = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    target_pos = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
    target_vel = np.zeros((B, 3), dtype=np.float64)
    obs_target_pos = target_pos[:, None, :].copy()
    path_lengths = np.array([[5.0]], dtype=np.float64)
    prev_target_dist = np.array([[8.0]], dtype=np.float64)
    prev_frontier_dist = np.array([[7.0]], dtype=np.float64)
    lidar = np.ones((B, N, 26), dtype=np.float64)
    guidance = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    actions = np.zeros((B, N, 4), dtype=np.float64)
    prev_actions = np.zeros((B, N, 4), dtype=np.float64)
    collisions = np.zeros((B, N), dtype=np.bool_)
    alive_mask = np.ones((B, N), dtype=np.bool_)
    collision_impact_flags = np.zeros((B, N), dtype=np.bool_)
    shield_flags = np.zeros((B, N), dtype=np.bool_)
    exploration_deltas = np.zeros((B, N), dtype=np.float64)
    frontier_deltas = np.zeros((B, N), dtype=np.float64)

    reward_core_coefs = np.array(
        [-0.1, 1.0, 0.5, -0.2, -2.0, 0.4, -0.3, 2.0, 1.0], dtype=np.float64
    )
    capture_radius = np.array([1.0], dtype=np.float64)
    lidar_max_range = 50.0
    frontier_reach_dist = 6.0
    lidar_safe_distance_m = 4.0
    search_speed_floor_mps = 1.0
    target_mode_flag = np.array([[1.0]], dtype=np.float32)
    is_potential_unstable = np.array([[unstable]], dtype=np.bool_)
    position_history = np.zeros((B, N, 5, 3), dtype=np.float64)
    frontier_goals = np.array([[[20.0, 0.0, 0.0]]], dtype=np.float64)
    world_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    world_max = np.array([100.0, 100.0, 50.0], dtype=np.float64)

    return (
        pos,
        vel,
        target_pos,
        target_vel,
        obs_target_pos,
        path_lengths,
        prev_target_dist,
        prev_frontier_dist,
        lidar,
        guidance,
        actions,
        prev_actions,
        collisions,
        alive_mask,
        collision_impact_flags,
        shield_flags,
        exploration_deltas,
        frontier_deltas,
        reward_core_coefs,
        capture_radius,
        lidar_max_range,
        frontier_reach_dist,
        lidar_safe_distance_m,
        search_speed_floor_mps,
        target_mode_flag,
        is_potential_unstable,
        position_history,
        frontier_goals,
        world_min,
        world_max,
    )


def test_reward_total_matches_breakdown_sum_and_is_finite():
    rewards, _, _, _, breakdown = batch_compute_rewards_v2(*_base_inputs(unstable=False))

    assert np.all(np.isfinite(rewards))
    assert np.all(np.isfinite(breakdown))
    assert breakdown.shape == (1, 10)
    np.testing.assert_allclose(breakdown[0, 9], np.sum(breakdown[0, :9]), rtol=1e-6, atol=1e-6)


def test_progress_gain_suppressed_on_unstable_steps():
    _, _, _, _, stable_breakdown = batch_compute_rewards_v2(*_base_inputs(unstable=False))
    _, _, _, _, unstable_breakdown = batch_compute_rewards_v2(*_base_inputs(unstable=True))

    # Index 1 is progress_gain in the new 9-core breakdown.
    assert stable_breakdown[0, 1] > 0.0
    assert abs(float(unstable_breakdown[0, 1])) < 1e-9


def test_collision_cost_sign_follows_negative_coefficient():
    args = list(_base_inputs(unstable=False))
    # collision_impact_flags, shield_flags, collisions
    args[14] = np.array([[True]], dtype=np.bool_)
    args[15] = np.array([[True]], dtype=np.bool_)
    args[12] = np.array([[True]], dtype=np.bool_)

    _, _, _, _, breakdown = batch_compute_rewards_v2(*tuple(args))
    # Index 4 is collision_cost.
    assert breakdown[0, 4] <= 0.0
