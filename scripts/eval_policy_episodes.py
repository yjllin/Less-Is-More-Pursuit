"""Evaluate a trained policy over fixed episodes with detailed diagnostics."""

from __future__ import annotations
import sys
import os

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import load_config
from src.controllers import MAPPOPolicy3D, TraditionalController3D
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2
from src.experiment_modes import BASELINE_CHOICES, get_baseline_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy for N episodes with diagnostics.")
    parser.add_argument("--policy", type=str, required=False, help="Path to policy checkpoint (.pt).")
    parser.add_argument(
        "--controller",
        type=str,
        default="none",
        choices=["none", "traditional_apf_pn"],
        help="Evaluation-only controller backend (traditional baseline).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=BASELINE_CHOICES,
        help="Apply unified baseline mode overrides to config.experiment.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml or config.json.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--steps", type=int, default=None, help="Max steps per episode (override config).")
    parser.add_argument("--stage-index", type=int, default=0, help="Curriculum stage index.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval-tag", type=str, default="", help="Optional tag written into summary output.")
    parser.add_argument("--output", type=str, default="eval_logs/episode_eval.jsonl", help="Output JSONL path.")
    return parser.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def apply_baseline_overrides(cfg, baseline: str | None):
    if not baseline:
        return cfg
    exp = cfg.experiment
    reward_runtime = cfg.reward_runtime
    override = get_baseline_override(baseline)
    assert override is not None
    if override["direction_gate_active_radius_m"] is not None:
        reward_runtime = replace(
            reward_runtime,
            direction_gate_active_radius_m=override["direction_gate_active_radius_m"],
        )
    return replace(
        cfg,
        reward_runtime=reward_runtime,
        experiment=replace(
            exp,
            mode=override["mode"],
            observation_profile=override["observation_profile"],
            guidance_backend=override["guidance_backend"],
            critic_mode=override["critic_mode"],
            eval_controller=override["eval_controller"],
        ),
    )


def build_eval_critic_obs(env: VectorizedMutualAStarEnvV2, obs: np.ndarray, device: torch.device, cfg) -> torch.Tensor | None:
    critic_mode = str(getattr(getattr(cfg, "experiment", None), "critic_mode", "local")).lower()
    if critic_mode != "ctde_joint_obs_plus_global":
        return None
    b, n, d = obs.shape
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
    joint = obs_t.reshape(b, n * d)
    target_state = np.concatenate((env.target_pos.astype(np.float32), env.target_vel.astype(np.float32)), axis=-1)
    target_t = torch.from_numpy(target_state).to(device)
    critic_env = torch.cat((joint, target_t), dim=-1)
    return critic_env[:, None, :].expand(-1, n, -1).reshape(b * n, -1)


def load_policy(policy_path: Path, obs_dim: int, cfg, device: torch.device) -> MAPPOPolicy3D:
    payload = torch.load(policy_path, map_location=device)
    state_dict = payload.get("policy_state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unrecognized checkpoint format at {policy_path}")
    weight = state_dict.get("model.base.0.weight", state_dict.get("base.0.weight"))
    ckpt_obs_dim = weight.shape[1] if weight is not None else obs_dim
    policy = MAPPOPolicy3D(
        ckpt_obs_dim,
        action_dim=4,
        device=device,
        action_bounds=cfg.control.action_bounds,
        hidden_dim=cfg.model.hidden_dim if hasattr(cfg, "model") else 512,
        centralized_critic=str(getattr(cfg.experiment, "critic_mode", "local")).lower() == "ctde_joint_obs_plus_global",
        critic_obs_dim=(cfg.environment.num_agents * ckpt_obs_dim + 6)
        if str(getattr(cfg.experiment, "critic_mode", "local")).lower() == "ctde_joint_obs_plus_global"
        else None,
    )
    policy._expected_obs_dim = ckpt_obs_dim
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def adapt_observation(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] != expected_dim:
        raise ValueError(
            f"Observation dim mismatch: got {obs.shape[-1]}, expected {expected_dim}. "
            "Please ensure eval config matches training config (frame_stack, obs layout)."
        )
    return obs


def init_episode_stats() -> Dict[str, float | int | Dict[str, float]]:
    return {
        "steps": 0,
        "min_dist": float("inf"),
        "dist_sum": 0.0,
        "visible_steps": 0,
        "first_visible_step": -1,
        "min_lidar": float("inf"),
        "collision_steps": 0,
        "collision_agent_events": 0,
        "shield_triggers": 0.0,
        "path_len_zero_steps": 0,
        "far_idle_sum": 0.0,
        "capture_contribution_sum": 0.0,
        "reward_sums": {},
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config) if args.config else load_config()
    cfg = apply_baseline_overrides(cfg, args.baseline)
    cfg = replace(cfg, training=replace(cfg.training, parallel_envs=args.num_envs))
    device = pick_device(args.device)

    env = VectorizedMutualAStarEnvV2(num_envs=args.num_envs, cfg=cfg)
    if args.steps is not None:
        env.max_steps = int(args.steps)
    obs, _ = env.reset(stage_index=args.stage_index, seed=args.seed)
    controller_name = args.controller
    if controller_name == "none" and str(getattr(cfg.experiment, "eval_controller", "none")) != "none":
        controller_name = str(cfg.experiment.eval_controller)
    policy = None
    controller = None
    if controller_name == "traditional_apf_pn":
        controller = TraditionalController3D(env=env, cfg=cfg, device=device)
    else:
        if not args.policy:
            raise ValueError("--policy is required unless --controller traditional_apf_pn is used.")
        policy = load_policy(Path(args.policy), obs.shape[-1], cfg, device)

    num_envs = env.num_envs
    num_agents = env.num_agents
    hidden = (policy or controller).init_hidden(num_envs=num_envs * num_agents)  # type: ignore[arg-type]
    masks = torch.ones(num_envs * num_agents, 1, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_env_stats = [init_episode_stats() for _ in range(num_envs)]
    episode_results: List[Dict[str, object]] = []
    total_done = 0
    eval_start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as fh:
        while total_done < args.episodes:
            pos = env.pos.copy()
            tgt = env.target_pos.copy()
            dists = np.linalg.norm(pos - tgt[:, None, :], axis=2)
            min_dist = np.min(dists, axis=1)
            mean_dist = np.mean(dists, axis=1)
            any_visible = getattr(env, "_any_visible", np.zeros(num_envs, dtype=bool))
            lidar = env.lidar_buffer
            if lidar is None:
                min_lidar = np.full(num_envs, np.inf, dtype=np.float64)
            else:
                min_lidar = np.min(lidar, axis=(1, 2)) * float(env.lidar_max_range)
            path_len_zero = np.sum(env.path_lengths <= 0.0, axis=1)

            for b in range(num_envs):
                stats = per_env_stats[b]
                stats["steps"] = int(stats["steps"]) + 1
                stats["min_dist"] = float(min(stats["min_dist"], float(min_dist[b])))
                stats["dist_sum"] = float(stats["dist_sum"]) + float(mean_dist[b])
                if bool(any_visible[b]):
                    stats["visible_steps"] = int(stats["visible_steps"]) + 1
                    if int(stats["first_visible_step"]) < 0:
                        stats["first_visible_step"] = int(stats["steps"])
                stats["min_lidar"] = float(min(stats["min_lidar"], float(min_lidar[b])))
                stats["path_len_zero_steps"] = int(stats["path_len_zero_steps"]) + int(path_len_zero[b])

            if controller is None:
                obs_adapted = adapt_observation(obs, int(getattr(policy, "_expected_obs_dim", obs.shape[-1])))
                flat_obs = obs_adapted.reshape(num_envs * num_agents, -1)
                obs_tensor = torch.from_numpy(flat_obs).to(device)
                critic_obs = build_eval_critic_obs(env, obs_adapted, device, cfg)
                with torch.no_grad():
                    if args.deterministic:
                        mean, _, _, hidden = policy.model(obs_tensor, hidden, masks, critic_obs=critic_obs)
                        actions = torch.tanh(mean) * policy._scale
                    else:
                        actions, _, _, hidden = policy.act(obs_tensor, hidden, masks, critic_obs=critic_obs)
                actions_np = actions.cpu().numpy().reshape(num_envs, num_agents, -1)
            else:
                with torch.no_grad():
                    actions, _, _, hidden = controller.act(torch.empty(0, device=device), hidden, masks)
                actions_np = actions.cpu().numpy().reshape(num_envs, num_agents, -1)

            obs, rewards, dones, infos = env.step(actions_np)

            for b in range(num_envs):
                stats = per_env_stats[b]
                info = infos[b]
                if info.get("collision", False):
                    stats["collision_steps"] = int(stats["collision_steps"]) + 1
                    collisions = info.get("collisions", [])
                    if isinstance(collisions, list):
                        stats["collision_agent_events"] = int(stats["collision_agent_events"]) + int(sum(collisions))
                metrics = info.get("metrics", {})
                stats["shield_triggers"] = float(stats["shield_triggers"]) + float(metrics.get("Shield_Trigger_Count", 0.0))
                stats["far_idle_sum"] = float(stats["far_idle_sum"]) + float(info.get("far_idle_ratio", 0.0))
                stats["capture_contribution_sum"] = float(stats["capture_contribution_sum"]) + float(info.get("capture_contribution", 0.0))
                breakdowns = info.get("reward_breakdown", [])
                if breakdowns:
                    for key, value in breakdowns[0].items():
                        rewards_sum = stats["reward_sums"].get(key, 0.0)
                        stats["reward_sums"][key] = float(rewards_sum) + float(value)

            if np.any(dones):
                masks = torch.ones(num_envs * num_agents, 1, device=device)
                for b in np.where(dones)[0]:
                    masks[b * num_agents : (b + 1) * num_agents] = 0.0

                for b in np.where(dones)[0]:
                    info = infos[b]
                    stats = per_env_stats[b]
                    steps = int(stats["steps"])
                    success = bool(info.get("success", info.get("captured", False)))
                    captured_raw = bool(info.get("captured_raw", info.get("captured", False)))
                    collision = bool(info.get("collision", False))
                    clean_capture = bool(info.get("clean_capture", (captured_raw and not collision)))
                    encircle_ok = bool(info.get("encircle_ok", False))
                    encircle_capture = bool(captured_raw and clean_capture and encircle_ok)
                    timeout = steps >= int(env.max_steps)
                    if success:
                        reason = "success"
                    elif collision:
                        reason = "collision"
                    elif timeout:
                        reason = "timeout"
                    else:
                        reason = "done"

                    result = {
                        "episode": total_done + 1,
                        "eval_tag": args.eval_tag,
                        "env_id": int(b),
                        "stage": env.stages[env.current_stage_idx].name,
                        "steps": steps,
                        "done_reason": reason,
                        "success": success,
                        "captured": success,
                        "captured_raw": captured_raw,
                        "clean_capture": clean_capture,
                        "encircle_ok": encircle_ok,
                        "encircle_capture": encircle_capture,
                        "collision": collision,
                        "timeout": timeout,
                        "min_dist": float(stats["min_dist"]),
                        "mean_dist": float(stats["dist_sum"]) / max(steps, 1),
                        "first_visible_step": int(stats["first_visible_step"]),
                        "visible_ratio": float(stats["visible_steps"]) / max(steps, 1),
                        "min_lidar": float(stats["min_lidar"]),
                        "collision_steps": int(stats["collision_steps"]),
                        "collision_agent_events": int(stats["collision_agent_events"]),
                        "shield_triggers": float(stats["shield_triggers"]),
                        "path_len_zero_steps": int(stats["path_len_zero_steps"]),
                        "far_idle_ratio": float(stats["far_idle_sum"]) / max(steps, 1),
                        "capture_contribution_mean": float(stats["capture_contribution_sum"]) / max(steps, 1),
                        "reward_sums": stats["reward_sums"],
                    }
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    episode_results.append(result)
                    total_done += 1
                    per_env_stats[b] = init_episode_stats()
                    if total_done >= args.episodes:
                        break

    # Summary
    success = sum(1 for r in episode_results if r["done_reason"] == "success")
    capture_raw = sum(1 for r in episode_results if r.get("captured_raw", False))
    clean_capture = sum(1 for r in episode_results if r.get("clean_capture", False))
    encircle_capture = sum(1 for r in episode_results if r.get("encircle_capture", False))
    collision = sum(1 for r in episode_results if r["done_reason"] == "collision")
    timeout = sum(1 for r in episode_results if r["done_reason"] == "timeout")
    avg_steps = float(np.mean([r["steps"] for r in episode_results])) if episode_results else 0.0
    avg_min_dist = float(np.mean([r["min_dist"] for r in episode_results])) if episode_results else 0.0
    avg_visible = float(np.mean([r["visible_ratio"] for r in episode_results])) if episode_results else 0.0
    avg_far_idle = float(np.mean([r.get("far_idle_ratio", 0.0) for r in episode_results])) if episode_results else 0.0
    eval_elapsed = max(time.perf_counter() - eval_start, 1e-6)
    eval_fps = float(len(episode_results) / eval_elapsed)

    print(f"[Eval] episodes={len(episode_results)} success_rate={success/len(episode_results):.3f} "
          f"collision_rate={collision/len(episode_results):.3f} timeout_rate={timeout/len(episode_results):.3f}")
    print(f"[Eval] capture_rate={capture_raw/len(episode_results):.3f} "
          f"clean_capture_rate={clean_capture/len(episode_results):.3f} "
          f"encircle_capture_rate={encircle_capture/len(episode_results):.3f}")
    print(f"[Eval] avg_steps={avg_steps:.1f} avg_min_dist={avg_min_dist:.2f} avg_visible_ratio={avg_visible:.2f} far_idle_ratio={avg_far_idle:.3f}")
    if controller is not None:
        stats = controller.stats
        print(
            f"[Eval] controller={controller_name} eval_fps={eval_fps:.3f} "
            f"mean_action_norm={stats.mean_action_norm:.3f} mean_yaw_rate_cmd={stats.mean_yaw_rate_cmd:.3f} "
            f"apf_saturation_ratio={stats.apf_saturation_ratio:.3f}"
        )
    print(f"[Eval] output={output_path}")


if __name__ == "__main__":
    main()
