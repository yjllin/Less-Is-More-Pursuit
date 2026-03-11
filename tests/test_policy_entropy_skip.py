import pytest
import torch
from torch.distributions import Normal
from unittest.mock import patch

from src.controllers.mappo_policy import ActionNoiseConfig, MAPPOPolicy3D


def _build_policy(device: torch.device) -> MAPPOPolicy3D:
    return MAPPOPolicy3D(
        obs_dim=32,
        action_dim=4,
        device=device,
        action_noise=ActionNoiseConfig(initial_std=0.4, min_std=0.1, decay_rate=1e-6),
        hidden_dim=64,
        action_bounds={"vx": 8.0, "vy": 4.0, "vz": 3.0, "yaw_rate": 0.8},
        min_action_std=0.1,
        max_action_std=0.5,
    )


def test_evaluate_actions_skip_entropy_mc_returns_zero_and_avoids_sampling():
    device = torch.device("cpu")
    policy = _build_policy(device).eval()
    policy.set_skip_entropy_mc(True)

    batch = 8
    obs = torch.randn(batch, 32, device=device)
    rnn = policy.init_hidden(batch)
    masks = torch.ones(batch, 1, device=device)
    actions = torch.zeros(batch, 4, device=device)

    with patch.object(Normal, "rsample", side_effect=RuntimeError("rsample should not be called")):
        log_prob, entropy, value, _ = policy.evaluate_actions(obs, rnn, masks, actions)

    assert log_prob.shape == (batch,)
    assert value.shape == (batch, 1)
    assert torch.all(entropy == 0.0)


def test_evaluate_actions_entropy_mc_path_uses_sampling():
    device = torch.device("cpu")
    policy = _build_policy(device).eval()
    policy.set_skip_entropy_mc(False)

    batch = 8
    obs = torch.randn(batch, 32, device=device)
    rnn = policy.init_hidden(batch)
    masks = torch.ones(batch, 1, device=device)
    actions = torch.zeros(batch, 4, device=device)

    with patch.object(Normal, "rsample", side_effect=RuntimeError("rsample path exercised")):
        with pytest.raises(RuntimeError, match="rsample path exercised"):
            policy.evaluate_actions(obs, rnn, masks, actions)
