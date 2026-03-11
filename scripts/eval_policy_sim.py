"""Evaluate a trained MAPPO policy in the vectorized simulation environment.

Logs min distance to target, capture status, and collisions.
"""

from __future__ import annotations
from collections import deque
from pathlib import Path
import sys
import os
import json

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.controllers import MAPPOPolicy3D, TraditionalController3D
from src.environment.vectorized_env_v2 import VectorizedMutualAStarEnvV2
from src.experiment_modes import BASELINE_CHOICES, get_baseline_override

HAS_MPL = False
plt = None
EXPLORE_TRAJECTORY_TAIL_STEPS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy in sim env.")
    parser.add_argument("--policy", type=str, required=False, help="Path to policy checkpoint (.pt).")
    parser.add_argument(
        "--controller",
        type=str,
        default="none",
        choices=["none", "traditional_apf_pn"],
        help="Evaluation controller backend (traditional baseline).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=BASELINE_CHOICES,
        help="Apply unified baseline overrides to config.experiment.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--stage-index", type=int, default=0, help="Curriculum stage index.")
    parser.add_argument("--spawn-level", type=int, default=None, help="Override spawn level (1-based).")
    parser.add_argument("--map-path", type=str, default=None, help="Override the occupancy map (.npy) used for evaluation.")
    parser.add_argument("--map-label", type=str, default=None, help="Optional label shown in the UI for the active map.")
    parser.add_argument("--steps", type=int, default=3000, help="Max steps per episode.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--explore-maps", action="store_true", help="Save layered exploration maps (3 images).")
    parser.add_argument("--explore-out", type=str, default="eval_explore", help="Output dir for exploration maps.")
    parser.add_argument("--explore-every", type=int, default=0, help="Save exploration maps every N steps (0=episode end).")
    parser.add_argument("--explore-live", action="store_true", help="Show live layered exploration maps.")
    parser.add_argument("--explore-live-every", type=int, default=5, help="Refresh live maps every N steps.")
    parser.add_argument("--explore-live-path", action="store_true", help="Overlay A* path projections in live maps.")
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="If set, record the live monitor visualization and save it as a .gif file.",
    )
    parser.add_argument("--gif-fps", type=int, default=10, help="Frame rate for the exported GIF.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="eval_logs/eval_log.txt", help="Write logs to file.")
    parser.add_argument("--std", type=float, default=None, help="Force action std (clamped to [0.05, 0.4]).")
    parser.add_argument(
        "--target-visible",
        type=str,
        default="auto",
        choices=["auto", "visible", "hidden"],
        help="Force target visibility: auto=normal LOS, visible=always visible, hidden=always invisible.",
    )
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
    policy.load_state_dict(state_dict, strict=False)
    policy._expected_obs_dim = ckpt_obs_dim
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


def _init_matplotlib(live: bool) -> bool:
    global HAS_MPL, plt
    try:
        import matplotlib
        if not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
        HAS_MPL = True
        return True
    except Exception:
        HAS_MPL = False
        plt = None
        return False


def _capture_live_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def _write_gif(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export GIF files. Install it with `pip install pillow`.") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _get_explore_layer_occupancy(env: VectorizedMutualAStarEnvV2, num_layers: int) -> np.ndarray | None:
    occ = getattr(env, "static_layer_occ", None)
    if occ is None:
        return None
    if occ.shape[2] < num_layers:
        return None
    return occ


def _apply_map_override(cfg, map_path: str, map_label: str | None = None):
    resolved = Path(map_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Map file not found: {resolved}")
    occupancy = np.load(resolved, mmap_mode="r")
    voxel_size = float(cfg.environment.voxel_size_m)
    meta = {}
    meta_path = resolved.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        voxel_size = float(meta.get("voxel_size", voxel_size))
    world_size_m = [float(dim) * voxel_size for dim in occupancy.shape]
    cfg = replace(
        cfg,
        navigation=replace(cfg.navigation, frontier_mask_path=str(resolved)),
        environment=replace(cfg.environment, world_size_m=world_size_m, voxel_size_m=voxel_size),
    )
    info = {
        "path": resolved,
        "label": map_label or str(meta.get("map_id") or resolved.stem),
    }
    return cfg, info


def _describe_active_map(env: VectorizedMutualAStarEnvV2, requested_info: dict[str, object] | None = None) -> dict[str, object]:
    path = Path(getattr(env.cfg.navigation, "frontier_mask_path", ""))
    label = str((requested_info or {}).get("label") or path.stem or "default_map")
    return {
        "path": path,
        "label": label,
        "shape": tuple(int(v) for v in env.occupancy_grid.shape),
        "voxel_size": float(env.voxel_size),
        "world_size": tuple(float(v) for v in (np.array(env.occupancy_grid.shape, dtype=np.float64) * env.voxel_size)),
    }


def _agent_mode_labels(env: VectorizedMutualAStarEnvV2, env_idx: int) -> list[str]:
    pos = env.pos[env_idx]
    tgt = env.target_pos[env_idx]
    dists = np.linalg.norm(pos - tgt, axis=1)
    capture_radius = float(env.env_capture_radius[env_idx]) if hasattr(env, "env_capture_radius") else float(env.cfg.environment.capture_radius_m)
    tracking_mask = env.target_mode_flag[env_idx] > 0.5
    labels: list[str] = []
    for agent_idx in range(pos.shape[0]):
        if dists[agent_idx] <= capture_radius:
            labels.append("capture")
        elif tracking_mask[agent_idx]:
            labels.append("chase")
        else:
            labels.append("search")
    return labels


def _apply_spawn_level(env: VectorizedMutualAStarEnvV2, cfg, level: int) -> None:
    levels = list(getattr(cfg.curriculum, "spawn_distance_levels", []))
    if not levels:
        raise ValueError("No spawn_distance_levels defined in curriculum config.")
    if level < 1 or level > len(levels):
        raise ValueError(f"spawn_level must be in [1, {len(levels)}], got {level}.")
    max_dist = float(levels[level - 1])
    min_dist = float(getattr(env, "target_spawn_min_dist", 20.0))
    env.set_target_spawn_range(min_dist, max_dist)
    print(f"[Config] Spawn level override: level={level} min_dist={min_dist:.1f} max_dist={max_dist:.1f}")


def _world_to_grid_xy(points: np.ndarray, origin: np.ndarray, voxel: np.ndarray | float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    origin_xy = np.asarray(origin, dtype=np.float64)[:2]
    voxel_arr = np.asarray(voxel, dtype=np.float64)
    if voxel_arr.ndim == 0:
        voxel_xy = np.full((2,), float(voxel_arr), dtype=np.float64)
    else:
        voxel_xy = voxel_arr[:2]
    return (pts[:, :2] - origin_xy) / voxel_xy


def _points_in_layer(
    points: np.ndarray,
    layer_idx: int,
    layer_bounds,
    layer_centers: np.ndarray | None = None,
) -> np.ndarray:
    if layer_centers is not None:
        assigned = np.argmin(np.abs(points[:, 2:3] - layer_centers[None, :]), axis=1)
        return assigned == layer_idx
    z_min, z_max = layer_bounds[layer_idx]
    return (points[:, 2] >= z_min) & (points[:, 2] <= z_max)


def _iter_layer_segments(
    points: np.ndarray,
    layer_idx: int,
    layer_bounds,
    layer_centers: np.ndarray | None = None,
):
    mask = _points_in_layer(points, layer_idx, layer_bounds, layer_centers)
    start = None
    for idx, is_in_layer in enumerate(mask):
        if is_in_layer and start is None:
            start = idx
        elif not is_in_layer and start is not None:
            if idx - start >= 2:
                yield start, idx, points[start:idx]
            start = None
    if start is not None and points.shape[0] - start >= 2:
        yield start, points.shape[0], points[start:]


def _draw_trajectory_with_arrow(ax, points: np.ndarray, color, label: str | None = None) -> None:
    if points.ndim != 2 or points.shape[0] < 2:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=2.0,
        alpha=0.95,
        label=label,
        zorder=4,
    )
    mid_idx = max(1, points.shape[0] // 2)
    delta = points[mid_idx] - points[mid_idx - 1]
    if np.allclose(delta, 0.0):
        return
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    ax.scatter(
        [points[mid_idx, 0]],
        [points[mid_idx, 1]],
        s=55,
        c=[color],
        marker=(3, 0, angle - 90.0),
        edgecolors="none",
        alpha=0.98,
        zorder=5,
    )


def _draw_layer_transition_marker(ax, point_xy: np.ndarray, color, went_up: bool) -> None:
    if went_up:
        ax.scatter(
            [point_xy[0]],
            [point_xy[1]],
            s=70,
            facecolors="none",
            edgecolors=[color],
            linewidths=1.8,
            zorder=7,
        )
        return
    ax.scatter(
        [point_xy[0]],
        [point_xy[1]],
        s=35,
        c=[color],
        edgecolors="white",
        linewidths=0.6,
        zorder=7,
    )


def _overlay_tail_trajectories(
    ax,
    env: VectorizedMutualAStarEnvV2,
    env_idx: int,
    layer_idx: int,
    layer_bounds,
    agent_history: np.ndarray | None,
    target_history: np.ndarray | None,
    tail_steps: int,
) -> None:
    if agent_history is None or target_history is None:
        return
    if agent_history.ndim != 4 or target_history.ndim != 3:
        return

    agent_tail = agent_history[-(tail_steps + 1) :, env_idx]
    target_tail = target_history[-(tail_steps + 1) :, env_idx]
    if agent_tail.shape[0] < 2 or target_tail.shape[0] < 2:
        return

    origin = env.origin
    voxel = env.voxel_size
    layer_centers = getattr(env, "layer_centers", None)
    cmap = plt.get_cmap("tab10")

    for agent_idx in range(agent_tail.shape[1]):
        color = cmap(agent_idx % 10)
        agent_points = agent_tail[:, agent_idx, :]
        if layer_centers is not None:
            assigned_layers = np.argmin(np.abs(agent_points[:, 2:3] - layer_centers[None, :]), axis=1)
        else:
            assigned_layers = np.full(agent_points.shape[0], -1, dtype=np.int64)
            for idx in range(len(layer_bounds)):
                assigned_layers[_points_in_layer(agent_points, idx, layer_bounds, None)] = idx
        for segment_idx, segment in enumerate(
            _iter_layer_segments(agent_points, layer_idx, layer_bounds, layer_centers)
        ):
            start_idx, end_idx, segment_points = segment
            grid_xy = _world_to_grid_xy(segment_points, origin, voxel)
            label = f"Agent {agent_idx}" if segment_idx == 0 else None
            _draw_trajectory_with_arrow(ax, grid_xy, color, label=label)
            if end_idx < agent_points.shape[0]:
                next_layer_idx = assigned_layers[end_idx]
                if next_layer_idx != layer_idx and next_layer_idx >= 0:
                    _draw_layer_transition_marker(
                        ax,
                        grid_xy[-1],
                        color,
                        went_up=next_layer_idx > layer_idx,
                    )

    if layer_centers is not None:
        target_assigned_layers = np.argmin(np.abs(target_tail[:, 2:3] - layer_centers[None, :]), axis=1)
    else:
        target_assigned_layers = np.full(target_tail.shape[0], -1, dtype=np.int64)
        for idx in range(len(layer_bounds)):
            target_assigned_layers[_points_in_layer(target_tail, idx, layer_bounds, None)] = idx

    for segment_idx, (start_idx, end_idx, segment) in enumerate(
        _iter_layer_segments(target_tail, layer_idx, layer_bounds, layer_centers)
    ):
        grid_xy = _world_to_grid_xy(segment, origin, voxel)
        label = "Target" if segment_idx == 0 else None
        _draw_trajectory_with_arrow(ax, grid_xy, "#00e5ff", label=label)
        if end_idx < target_tail.shape[0]:
            next_layer_idx = target_assigned_layers[end_idx]
            if next_layer_idx != layer_idx and next_layer_idx >= 0:
                _draw_layer_transition_marker(
                    ax,
                    grid_xy[-1],
                    "#00e5ff",
                    went_up=next_layer_idx > layer_idx,
                )


def _save_exploration_maps(
    env: VectorizedMutualAStarEnvV2,
    env_idx: int,
    out_dir: Path,
    tag: str,
    agent_history: np.ndarray | None = None,
    target_history: np.ndarray | None = None,
    captured: np.ndarray | None = None,
    tail_steps: int = EXPLORE_TRAJECTORY_TAIL_STEPS,
) -> None:
    from matplotlib.lines import Line2D

    layer_conf = env.get_layer_confidence(env_idx)
    occ = _get_explore_layer_occupancy(env, layer_conf.shape[2])
    origin = env.origin
    voxel = env.voxel_size
    layer_bounds = getattr(env, "layer_z_bounds", [(env.world_min[2], env.world_max[2])] * 3)
    layer_centers = getattr(env, "layer_centers", None)
    cmap = plt.get_cmap("tab10")
    if agent_history is not None and target_history is not None:
        pos = agent_history[-1, env_idx]
        tgt = target_history[-1, env_idx]
    else:
        pos = env.pos[env_idx]
        tgt = env.target_pos[env_idx]
    num_layers = layer_conf.shape[2]
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6))
    axes = np.atleast_1d(axes)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=cmap(agent_idx % 10),
            lw=2.0,
            marker="^",
            markersize=10,
            markerfacecolor=cmap(agent_idx % 10),
            markeredgecolor="white",
            label=f"Agent {agent_idx}",
        )
        for agent_idx in range(pos.shape[0])
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#00e5ff",
            lw=2.0,
            marker="*",
            markersize=14,
            markerfacecolor="#00e5ff",
            markeredgecolor="white",
            label="Target",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#ff3366",
            lw=0,
            marker="x",
            markersize=8,
            markeredgewidth=1.6,
            label="Captured target",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#606060",
            lw=0,
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=1.6,
            label="To upper layer",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#606060",
            lw=0,
            marker="o",
            markersize=6,
            markerfacecolor="#606060",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="To lower layer",
        )
    )

    for layer_idx, ax in enumerate(axes):
        ax.set_title(f"layer{layer_idx}")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.set_aspect("equal", "box")
        if occ is not None:
            ax.imshow(occ[:, :, layer_idx].T, origin="lower", cmap="Greys", alpha=0.35)
        ax.imshow(layer_conf[:, :, layer_idx].T, origin="lower", cmap="Blues", alpha=0.75, vmin=0.0, vmax=1.0)
        _overlay_tail_trajectories(
            ax=ax,
            env=env,
            env_idx=env_idx,
            layer_idx=layer_idx,
            layer_bounds=layer_bounds,
            agent_history=agent_history,
            target_history=target_history,
            tail_steps=tail_steps,
        )
        in_layer = _points_in_layer(pos, layer_idx, layer_bounds, layer_centers)
        if np.any(in_layer):
            for agent_idx in np.where(in_layer)[0]:
                idx = np.floor((pos[agent_idx] - origin) / voxel).astype(int)
                ax.scatter(
                    [idx[0]],
                    [idx[1]],
                    c=[cmap(agent_idx % 10)],
                    s=95,
                    marker="^",
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=6,
                )
        tgt_in_layer = _points_in_layer(tgt[None, :], layer_idx, layer_bounds, layer_centers)[0]
        if tgt_in_layer:
            tgt_idx = np.floor((tgt - origin) / voxel).astype(int)
            ax.scatter(
                [tgt_idx[0]],
                [tgt_idx[1]],
                c="#00e5ff",
                s=190,
                marker="*",
                edgecolors="white",
                linewidths=1.0,
                zorder=7,
            )
            if captured is not None and bool(captured[env_idx]):
                ax.scatter(
                    [tgt_idx[0]],
                    [tgt_idx[1]],
                    c="#ff3366",
                    s=90,
                    marker="x",
                    linewidths=1.8,
                    zorder=8,
                )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(legend_handles), max(2, num_layers + 1)),
        frameon=True,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.18)
    out_path = out_dir / f"{tag}_layers.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _init_live_explore_plot(
    env: VectorizedMutualAStarEnvV2,
    env_idx: int,
    num_agents: int,
    show_paths: bool = False,
    interactive: bool = True,
    map_info: dict[str, object] | None = None,
):
    if not HAS_MPL:
        return None
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    
  
    plt.style.use('dark_background')
    
  
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.3, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.06,
        right=0.94,
        top=0.81,
        bottom=0.08,
    )
    
  
    fig.suptitle(' Ours-Lite: Zero-Comm 3D Pursuit-Evasion', fontsize=14, fontweight='bold',
                 color='#00d4ff', y=0.978)
    stage_name = env.stages[env.current_stage_idx].name if hasattr(env, "stages") else "unknown"
    info_text = fig.text(
        0.06,
        0.952,
        "",
        ha="left",
        va="top",
        color="#cfd8dc",
        fontsize=9,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#334155", alpha=0.92),
    )
    mode_text = fig.text(
        0.94,
        0.952,
        "",
        ha="right",
        va="top",
        color="#f8f9fa",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#334155", alpha=0.92),
    )
    
  
    layer_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    layer_conf = env.get_layer_confidence(env_idx)
    occ = _get_explore_layer_occupancy(env, layer_conf.shape[2])
    layer_artists = []
    
    layer_names = ['Ground Layer (Z)', 'Mid Layer (Z)', 'High Layer (Z)']
    layer_colors = ['#16213e', '#1a1a2e', '#0f3460']
    agent_colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181',
                    '#aa96da', '#fcbad3', '#a8d8ea'][:num_agents]
    for layer_idx, ax in enumerate(layer_axes):
        ax.set_facecolor(layer_colors[layer_idx])
        ax.set_title(layer_names[layer_idx], fontsize=11, fontweight='bold',
                     color='#e94560', pad=6)
        ax.set_xlabel("Grid X", fontsize=9, color='#a0a0a0')
        ax.set_ylabel("Grid Y", fontsize=9, color='#a0a0a0')
        ax.set_aspect("equal", "box")
        ax.tick_params(colors='#808080', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(1.5)
        
        if occ is not None:
            ax.imshow(occ[:, :, layer_idx].T, origin="lower", cmap="Greys", alpha=0.4)
        im = ax.imshow(layer_conf[:, :, layer_idx].T, origin="lower", cmap="viridis", 
                       alpha=0.8, vmin=0.0, vmax=1.0)
        agents = ax.scatter([], [], c="#ff6b6b", s=60, label="Agents", 
                           edgecolors='white', linewidths=1.5, zorder=5)
        target = ax.scatter([], [], c="#00ff88", s=120, marker="*", label="Target",
                           edgecolors='white', linewidths=1, zorder=6)
        frontier = ax.scatter([], [], c="#ffd93d", s=15, marker=".", label="Frontier",
                             alpha=0.7, zorder=3)
        assigned = ax.scatter([], [], c="#c77dff", s=40, marker="D", label="Assigned",
                             edgecolors='white', linewidths=0.8, zorder=4)
        path_lines = []
        if show_paths:
            for i in range(num_agents):
                line, = ax.plot([], [], color=agent_colors[i], linewidth=1.5, alpha=0.8, zorder=2)
                path_lines.append(line)
        layer_artists.append((im, agents, target, frontier, assigned, path_lines))

    ax_vel = fig.add_subplot(gs[1, 0:2])
    ax_vel.set_facecolor('#16213e')
    ax_vel.set_title(" Agent Velocity", fontsize=11, fontweight='bold', 
                     color='#4ecdc4', pad=10)
    ax_vel.set_xlabel("Step", fontsize=9, color='#a0a0a0')
    ax_vel.set_ylabel("Speed (m/s)", fontsize=9, color='#a0a0a0')
    ax_vel.set_xlim(0, 100)
    ax_vel.set_ylim(0, 2)
    ax_vel.grid(True, alpha=0.2, color='#404040', linestyle='--')
    ax_vel.tick_params(colors='#808080', labelsize=8)
    for spine in ax_vel.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1.5)
    
    vel_lines = []
    for i in range(num_agents):
        line, = ax_vel.plot([], [], color=agent_colors[i], linewidth=2, 
                           label=f"Agent {i}", alpha=0.9)
        vel_lines.append(line)
    ax_vel.legend(loc="upper right", fontsize=8, facecolor='#1a1a2e', 
                  edgecolor='#404040', labelcolor='#e0e0e0')
    
  
    ax_dist = fig.add_subplot(gs[1, 2])
    ax_dist.set_facecolor('#16213e')
    ax_dist.set_title(" Distance to Target", fontsize=11, fontweight='bold', 
                      color='#ffe66d', pad=10)
    ax_dist.set_xlabel("Step", fontsize=9, color='#a0a0a0')
    ax_dist.set_ylabel("Distance (m)", fontsize=9, color='#a0a0a0')
    ax_dist.set_xlim(0, 100)
    ax_dist.set_ylim(0, 20)
    ax_dist.grid(True, alpha=0.2, color='#404040', linestyle='--')
    ax_dist.tick_params(colors='#808080', labelsize=8)
    for spine in ax_dist.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1.5)
    
    dist_lines = []
    for i in range(num_agents):
        line, = ax_dist.plot([], [], color=agent_colors[i], linewidth=2, 
                            label=f"Agent {i}", alpha=0.9)
        dist_lines.append(line)
    ax_dist.legend(loc="upper right", fontsize=8, facecolor='#1a1a2e', 
                   edgecolor='#404040', labelcolor='#e0e0e0')
    
  
    history = {
        "steps": [],
        "velocities": [[] for _ in range(num_agents)],
        "distances": [[] for _ in range(num_agents)],
    }
    
  
    handles, labels = layer_axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.908),
        facecolor='#1a1a2e',
        edgecolor='#404040',
        labelcolor='#e0e0e0',
        framealpha=0.9,
    )
    
    if map_info is None:
        map_info = _describe_active_map(env)
    map_line = (
        f"Stage: {stage_name}\n"
        f"Map: {map_info['label']}\n"
        f"Grid: {map_info['shape'][0]}x{map_info['shape'][1]}x{map_info['shape'][2]} | "
        f"Voxel: {map_info['voxel_size']:.2f} m"
    )
    info_text.set_text(map_line)
    modes = _agent_mode_labels(env, env_idx)
    mode_text.set_text("Modes\n" + "\n".join(f"Agent {i}: {mode}" for i, mode in enumerate(modes)))

    return fig, layer_axes, layer_artists, ax_vel, vel_lines, ax_dist, dist_lines, history, show_paths, info_text, mode_text, map_info


def _update_live_explore_plot(env: VectorizedMutualAStarEnvV2, env_idx: int, live_state, step: int, velocities: np.ndarray | None = None) -> None:
    if not HAS_MPL or live_state is None:
        return
    fig, layer_axes, layer_artists, ax_vel, vel_lines, ax_dist, dist_lines, history, show_paths, info_text, mode_text, map_info = live_state
    layer_conf = env.get_layer_confidence(env_idx)
    occ = _get_explore_layer_occupancy(env, layer_conf.shape[2])
    origin = env.origin
    voxel = env.voxel_size
    layer_bounds = getattr(env, "layer_z_bounds", [(env.world_min[2], env.world_max[2])] * 3)
    layer_centers = getattr(env, "layer_centers", None)
    pos = env.pos[env_idx]
    tgt = env.target_pos[env_idx]
    frontier_debug = env.get_frontier_debug(env_idx)
    stage_name = env.stages[env.current_stage_idx].name if hasattr(env, "stages") else "unknown"
    modes = _agent_mode_labels(env, env_idx)
    info_text.set_text(
        f"Stage: {stage_name}\n"
        f"Map: {map_info['label']}\n"
        f"Grid: {map_info['shape'][0]}x{map_info['shape'][1]}x{map_info['shape'][2]} | "
        f"Voxel: {map_info['voxel_size']:.2f} m | Step: {step}"
    )
    mode_text.set_text("Modes\n" + "\n".join(f"Agent {i}: {mode}" for i, mode in enumerate(modes)))
    
  
    agent_dists = np.linalg.norm(pos - tgt, axis=1)
    if velocities is not None:
        agent_speeds = np.linalg.norm(velocities, axis=1)
    else:
        agent_speeds = np.zeros(pos.shape[0])
    
  
    history["steps"].append(step)
    num_agents = len(vel_lines)
    for i in range(num_agents):
        history["velocities"][i].append(agent_speeds[i] if i < len(agent_speeds) else 0)
        history["distances"][i].append(agent_dists[i] if i < len(agent_dists) else 0)
    
  
    steps = history["steps"]
    for i, line in enumerate(vel_lines):
        line.set_data(steps, history["velocities"][i])
  
    if len(steps) > 0:
        ax_vel.set_xlim(0, max(100, steps[-1] + 10))
        max_vel = max(max(v) if v else 1 for v in history["velocities"])
        ax_vel.set_ylim(0, max(2, max_vel * 1.2))
    
  
    for i, line in enumerate(dist_lines):
        line.set_data(steps, history["distances"][i])
    if len(steps) > 0:
        ax_dist.set_xlim(0, max(100, steps[-1] + 10))
        max_dist = max(max(d) if d else 1 for d in history["distances"])
        ax_dist.set_ylim(0, max(20, max_dist * 1.2))
    
  
    paths = None
    if show_paths:
        paths = []
        goals = getattr(env, "current_goals", None)
        for i in range(pos.shape[0]):
            start = pos[i]
            if hasattr(env, "_snap_astar_start"):
                start = env._snap_astar_start(start)
            goal = goals[env_idx, i] if goals is not None else tgt
            if env.navigator is not None:
                _, path, _ = env.navigator.compute_direction(
                    start_position=start,
                    goal_position=goal,
                    current_step=int(env.steps[env_idx]),
                    current_velocity=env.vel[env_idx, i],
                    smooth=False,
                    cache_key=(int(env_idx), int(i)),
                )
                if path is None or len(path) < 2:
                    path = [start, goal]
            else:
                path = [start, goal]
            paths.append(np.asarray(path, dtype=np.float64))

    for layer_idx, ax in enumerate(layer_axes):
        im, agents, target, frontier, assigned, path_lines = layer_artists[layer_idx]
        im.set_data(layer_conf[:, :, layer_idx].T)
        z_min, z_max = layer_bounds[layer_idx]
        if layer_centers is not None:
            agent_layer_idx = np.argmin(np.abs(pos[:, 2:3] - layer_centers[None, :]), axis=1)
            in_layer = agent_layer_idx == layer_idx
            if np.any(in_layer):
                idx = np.floor((pos[in_layer] - origin) / voxel).astype(int)
                agents.set_offsets(idx[:, :2])
            else:
                agents.set_offsets(np.zeros((0, 2)))
            tgt_layer = int(np.argmin(np.abs(layer_centers - tgt[2])))
            if tgt_layer == layer_idx:
                tgt_idx = np.floor((tgt - origin) / voxel).astype(int)
                target.set_offsets(np.array([[tgt_idx[0], tgt_idx[1]]]))
            else:
                target.set_offsets(np.zeros((0, 2)))
        else:
            in_layer = (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max)
            if np.any(in_layer):
                idx = np.floor((pos[in_layer] - origin) / voxel).astype(int)
                agents.set_offsets(idx[:, :2])
            else:
                agents.set_offsets(np.zeros((0, 2)))
            if z_min <= tgt[2] <= z_max:
                tgt_idx = np.floor((tgt - origin) / voxel).astype(int)
                target.set_offsets(np.array([[tgt_idx[0], tgt_idx[1]]]))
            else:
                target.set_offsets(np.zeros((0, 2)))
        if frontier_debug and frontier_debug.get("candidates") is not None:
            candidates = frontier_debug["candidates"]
            if candidates.shape[1] >= 3:
                z = candidates[:, 2]
                mask = (z >= z_min) & (z <= z_max)
                if np.any(mask):
                    idx = np.floor((candidates[mask] - origin) / voxel).astype(int)
                    frontier.set_offsets(idx[:, :2])
                else:
                    frontier.set_offsets(np.zeros((0, 2)))
            else:
                frontier.set_offsets(np.zeros((0, 2)))
        else:
            frontier.set_offsets(np.zeros((0, 2)))
        if frontier_debug and frontier_debug.get("selected_goals") is not None:
            sel = frontier_debug["selected_goals"]
            if sel is not None and sel.size >= 3:
                z = sel[:, 2]
                mask = (z >= z_min) & (z <= z_max)
                if np.any(mask):
                    idx = np.floor((sel[mask] - origin) / voxel).astype(int)
                    assigned.set_offsets(idx[:, :2])
                else:
                    assigned.set_offsets(np.zeros((0, 2)))
            else:
                assigned.set_offsets(np.zeros((0, 2)))
        else:
            assigned.set_offsets(np.zeros((0, 2)))
        if show_paths and path_lines and paths is not None:
            for i, line in enumerate(path_lines):
                path_pts = paths[i]
                if path_pts.ndim != 2 or path_pts.shape[0] < 2:
                    line.set_data([], [])
                    continue
                mask = (path_pts[:, 2] >= z_min) & (path_pts[:, 2] <= z_max)
                if np.any(mask):
                    idx = np.floor((path_pts[mask] - origin) / voxel).astype(int)
                    line.set_data(idx[:, 0], idx[:, 1])
                else:
                    line.set_data([], [])
    fig.canvas.draw_idle()
    plt.pause(0.001)



def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gif_enabled = bool(args.gif)
    _init_matplotlib(args.explore_live or gif_enabled)
    cfg = load_config(args.config) if args.config else load_config()
    cfg = apply_baseline_overrides(cfg, args.baseline)
    requested_map_info = None
    if args.map_path:
        cfg, requested_map_info = _apply_map_override(cfg, args.map_path, args.map_label)
    cfg = replace(cfg, training=replace(cfg.training, parallel_envs=args.num_envs))
    device = pick_device(args.device)

    env = VectorizedMutualAStarEnvV2(num_envs=args.num_envs, cfg=cfg)
    active_map_info = _describe_active_map(env, requested_map_info)
    env.max_steps = args.steps
    if args.target_visible == "visible":
        env.visibility_override = 1
    elif args.target_visible == "hidden":
        env.visibility_override = -1
    else:
        env.visibility_override = 0
    if args.target_visible != "auto":
        print(f"[Config] Target visibility override: {args.target_visible}")
    print(
        f"[Config] Active map: {active_map_info['label']} "
        f"({active_map_info['shape'][0]}x{active_map_info['shape'][1]}x{active_map_info['shape'][2]}, "
        f"voxel={active_map_info['voxel_size']:.2f}m)"
    )
    if args.spawn_level is not None:
        _apply_spawn_level(env, cfg, args.spawn_level)
    obs, _ = env.reset(stage_index=args.stage_index, seed=args.seed)
    obs_dim = obs.shape[-1]
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
        policy = load_policy(Path(args.policy), obs_dim, cfg, device)
        if args.std is not None:
            policy.reset_noise(float(args.std))
            print(f"[Policy] Forced action std -> {policy.get_current_std():.4f}")
    expected_obs_dim = int(getattr(policy, "_expected_obs_dim", obs_dim)) if policy is not None else obs_dim

    explore_enabled = args.explore_maps and HAS_MPL
    explore_dir = Path(args.explore_out)
    if explore_enabled:
        explore_dir.mkdir(parents=True, exist_ok=True)
    elif args.explore_maps and not HAS_MPL:
        print("[Warning] matplotlib not available; disabling exploration maps.")

    live_state = None
    gif_frames: list[np.ndarray] = []
    if (args.explore_live or gif_enabled) and HAS_MPL:
        live_state = _init_live_explore_plot(
            env,
            0,
            env.num_agents,
            show_paths=args.explore_live_path,
            interactive=args.explore_live,
            map_info=active_map_info,
        )
    elif args.explore_live and not HAS_MPL:
        print("[Warning] matplotlib not available; disabling live exploration maps.")
    if gif_enabled and not HAS_MPL:
        print("[Warning] matplotlib not available; disabling GIF export.")
        gif_enabled = False

    num_entities = env.num_envs * env.num_agents
    hidden = (policy or controller).init_hidden(num_envs=num_entities)  # type: ignore[arg-type]
    masks = torch.ones(num_entities, 1, device=device)

    log_path = Path(args.output)
    if (log_path.exists() and log_path.is_dir()) or (not log_path.exists() and log_path.suffix == ""):
        log_path.mkdir(parents=True, exist_ok=True)
        log_path = log_path / "eval_log.txt"
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    print(f"[Log] Writing output to {log_path}")

    try:
        for ep in range(args.episodes):
            episode_seed = args.seed + ep
            if args.spawn_level is not None:
                _apply_spawn_level(env, cfg, args.spawn_level)
            obs, info = env.reset(stage_index=args.stage_index, seed=episode_seed)
            obs_adapted = adapt_observation(obs, expected_obs_dim)
            obs_tensor = torch.from_numpy(obs_adapted.reshape(num_entities, -1)).to(device)
            hidden = (policy or controller).init_hidden(num_envs=num_entities)  # type: ignore[arg-type]
            masks = torch.ones(num_entities, 1, device=device)
            ep_min_dist = np.full((env.num_envs,), np.inf, dtype=np.float64)
            ep_capture = np.zeros((env.num_envs,), dtype=bool)
            agent_history = deque(maxlen=EXPLORE_TRAJECTORY_TAIL_STEPS + 1)
            target_history = deque(maxlen=EXPLORE_TRAJECTORY_TAIL_STEPS + 1)
            agent_history.append(np.array(env.pos, copy=True))
            target_history.append(np.array(env.target_pos, copy=True))

            last_step = 0
            actions_np = None
            for step in range(1, args.steps + 1):
                with torch.no_grad():
                    if controller is not None:
                        actions, _, _, hidden = controller.act(obs_tensor, hidden, masks)
                    else:
                        critic_obs = build_eval_critic_obs(env, obs_adapted, device, cfg)
                        actions, _, _, hidden = policy.act(obs_tensor, hidden, masks, critic_obs=critic_obs)
                actions_np = actions.cpu().numpy().reshape(env.num_envs, env.num_agents, -1)
                obs, rewards, dones, infos = env.step(actions_np)
                # Enforce collision termination in evaluation loop.
                info_collision = np.array([bool(info.get("collision", False)) for info in infos], dtype=bool)
                if np.any(info_collision):
                    dones = np.logical_or(dones, info_collision)
                step_pos = np.array(env.pos, copy=True)
                step_tgt = np.array(env.target_pos, copy=True)
                for b, done in enumerate(dones):
                    if not bool(done):
                        continue
                    terminal_agent_pos = infos[b].get("terminal_agent_pos")
                    terminal_target_pos = infos[b].get("terminal_target_pos")
                    if terminal_agent_pos is not None:
                        step_pos[b] = np.asarray(terminal_agent_pos, dtype=np.float64)
                    if terminal_target_pos is not None:
                        step_tgt[b] = np.asarray(terminal_target_pos, dtype=np.float64)
                obs_adapted = adapt_observation(obs, expected_obs_dim)
                obs_tensor = torch.from_numpy(obs_adapted.reshape(num_entities, -1)).to(device)
                last_step = step
                agent_history.append(step_pos)
                target_history.append(step_tgt)

                if explore_enabled and args.explore_every > 0 and step % args.explore_every == 0:
                    _save_exploration_maps(
                        env,
                        0,
                        explore_dir,
                        f"seed{episode_seed}_step{step:04d}",
                        agent_history=np.stack(agent_history, axis=0),
                        target_history=np.stack(target_history, axis=0),
                        captured=ep_capture,
                    )
                if args.explore_live and args.explore_live_every > 0 and step % args.explore_live_every == 0:
                    velocities = actions_np[0, :, :3]  # (num_agents, 3)
                    _update_live_explore_plot(env, 0, live_state, step, velocities)
                elif gif_enabled and args.explore_live_every > 0 and step % args.explore_live_every == 0:
                    velocities = actions_np[0, :, :3]  # (num_agents, 3)
                    _update_live_explore_plot(env, 0, live_state, step, velocities)
                if gif_enabled and live_state is not None and args.explore_live_every > 0 and step % args.explore_live_every == 0:
                    gif_frames.append(_capture_live_frame(live_state[0]))

                # Track min distance to target (per env)
                pos = step_pos
                tgt = step_tgt
                for b in range(env.num_envs):
                    dists = np.linalg.norm(pos[b] - tgt[b], axis=1)
                    ep_min_dist[b] = min(ep_min_dist[b], float(np.min(dists)))
                    ep_capture[b] = ep_capture[b] or bool(infos[b].get("captured", False))

                if step % 10 == 0 or np.any(dones):
                    # Debug: visibility and policy action info
                    any_vis = env._any_visible  # (B,) bool
                    # Per-agent distances to target
                    dist_str = ""
                    for b in range(env.num_envs):
                        agent_dists = np.linalg.norm(pos[b] - tgt[b], axis=1)
                        dist_str += f"env{b}:{[f'{d:.2f}' for d in agent_dists]} "
                    act_str = ""
                    rel_str = ""
                    for b in range(env.num_envs):
                        act_str += f"env{b}:["
                        rel_str += f"env{b}:["
                        for n in range(env.num_agents):
                            vx, vy, vz, yaw = actions_np[b, n]
                            act_str += f"({vx:.2f},{vy:.2f},{vz:.2f},{yaw:.2f})"
                            # Body-frame lateral error (target on left => +b_dy)
                            dx = env.obs_target_pos[b, n, 0] - pos[b, n, 0]
                            dy = env.obs_target_pos[b, n, 1] - pos[b, n, 1]
                            cy = np.cos(env.yaw[b, n])
                            sy = np.sin(env.yaw[b, n])
                            b_dy = -dx * sy + dy * cy
                            rel_str += f"(b_dy={b_dy:.2f},vy={vy:.2f})"
                        act_str += "] "
                        rel_str += "] "
                    view_radius = float(env.env_view_radius[0]) if hasattr(env, "env_view_radius") else 0.0
                    stage_name = env.stages[env.current_stage_idx].name if hasattr(env, "stages") else "unknown"
                    line = (
                        f"[Ep {ep+1} Step {step:4d}] dists={dist_str}"
                        f"captured={ep_capture} collisions={infos[0].get('collisions')}"
                        f" stage={stage_name} view_radius={view_radius:.1f}"
                    )
                    print(line)
                    print(f"  visible={any_vis.tolist()} target={tgt.tolist()} pos={pos.tolist()} vel={act_str}")
                    print(f"  rel={rel_str}")
                    if log_fh:
                        log_fh.write(line + "\n")
                        log_fh.write(f"  visible={any_vis.tolist()} target={tgt.tolist()} pos={pos.tolist()} vel={act_str}\n")
                        log_fh.write(f"  rel={rel_str}\n")

                if np.any(dones):
                    break

            if explore_enabled and args.explore_every == 0:
                _save_exploration_maps(
                    env,
                    0,
                    explore_dir,
                    f"seed{episode_seed}_step{last_step:04d}",
                    agent_history=np.stack(agent_history, axis=0),
                    target_history=np.stack(target_history, axis=0),
                    captured=ep_capture,
                )
            if args.explore_live and args.explore_live_every == 0:
                velocities = actions_np[0, :, :3] if actions_np is not None else None
                _update_live_explore_plot(env, 0, live_state, last_step, velocities)
            elif gif_enabled and args.explore_live_every == 0:
                velocities = actions_np[0, :, :3] if actions_np is not None else None
                _update_live_explore_plot(env, 0, live_state, last_step, velocities)
            if gif_enabled and live_state is not None and args.explore_live_every == 0:
                gif_frames.append(_capture_live_frame(live_state[0]))
            elif gif_enabled and live_state is not None and last_step > 0 and (
                args.explore_live_every <= 0 or last_step % args.explore_live_every != 0
            ):
                velocities = actions_np[0, :, :3] if actions_np is not None else None
                _update_live_explore_plot(env, 0, live_state, last_step, velocities)
                gif_frames.append(_capture_live_frame(live_state[0]))

            # final_line = (
            #     f"[Ep {ep+1} Done] min_dist={ep_min_dist} "
            #     f"captured={ep_capture} steps={step}"
            # )
            # print(final_line)
            # if log_fh:
            #     log_fh.write(final_line + "\n")
    finally:
        if gif_enabled and gif_frames:
            gif_path = Path(args.gif)
            _write_gif(gif_frames, gif_path, args.gif_fps)
            print(f"[GIF] Saved visualization to {gif_path}")
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
