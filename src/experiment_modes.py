"""Shared baseline naming and override helpers."""

from __future__ import annotations

from typing import Any


ACTIVE_BASELINES = [
    "ours",
    "b1_traditional_apf_pn",
    "b2_euclidean_guidance",
    "b3_full_obs_ippo",
    "b4_ctde_mappo",
    "b5_local_obs_no_dist80_gate",
]

# Keep the legacy name accepted on CLI so older commands still work.
BASELINE_CHOICES = ACTIVE_BASELINES + ["b3_local_obs_ippo"]


def canonicalize_baseline_name(name: str | None) -> str | None:
    if name is None:
        return None
    raw = str(name).strip()
    if raw == "b3_local_obs_ippo":
        return "ours"
    return raw


def get_baseline_override(name: str | None) -> dict[str, Any] | None:
    canonical = canonicalize_baseline_name(name)
    if canonical is None:
        return None
    if canonical == "ours":
        return {
            "mode": "ours",
            "observation_profile": "local50",
            "guidance_backend": "astar",
            "critic_mode": "local",
            "eval_controller": "none",
            "direction_gate_active_radius_m": None,
        }
    if canonical == "b1_traditional_apf_pn":
        return {
            "mode": canonical,
            "observation_profile": "full83",
            "guidance_backend": "astar",
            "critic_mode": "local",
            "eval_controller": "traditional_apf_pn",
            "direction_gate_active_radius_m": None,
        }
    if canonical == "b2_euclidean_guidance":
        return {
            "mode": canonical,
            "observation_profile": "full83",
            "guidance_backend": "euclidean",
            "critic_mode": "local",
            "eval_controller": "none",
            "direction_gate_active_radius_m": None,
        }
    if canonical == "b3_full_obs_ippo":
        return {
            "mode": canonical,
            "observation_profile": "full83",
            "guidance_backend": "astar",
            "critic_mode": "local",
            "eval_controller": "none",
            "direction_gate_active_radius_m": None,
        }
    if canonical == "b4_ctde_mappo":
        return {
            "mode": canonical,
            "observation_profile": "full83",
            "guidance_backend": "astar",
            "critic_mode": "ctde_joint_obs_plus_global",
            "eval_controller": "none",
            "direction_gate_active_radius_m": None,
        }
    if canonical == "b5_local_obs_no_dist80_gate":
        return {
            "mode": canonical,
            "observation_profile": "local50",
            "guidance_backend": "astar",
            "critic_mode": "local",
            "eval_controller": "none",
            "direction_gate_active_radius_m": 0.0,
        }
    raise ValueError(f"Unsupported baseline: {name}")


def infer_canonical_baseline(
    *,
    mode: str | None,
    observation_profile: str | None,
    guidance_backend: str | None,
    critic_mode: str | None,
    eval_controller: str | None,
    direction_gate_active_radius_m: float | None = None,
) -> str:
    obs = str(observation_profile or "").strip().lower()
    guidance = str(guidance_backend or "").strip().lower()
    critic = str(critic_mode or "").strip().lower()
    controller = str(eval_controller or "").strip().lower()
    canonical_mode = canonicalize_baseline_name(mode) or ""
    gate = None if direction_gate_active_radius_m is None else float(direction_gate_active_radius_m)

    if controller == "traditional_apf_pn":
        return "b1_traditional_apf_pn"
    if obs == "full83" and guidance == "euclidean" and critic == "local":
        return "b2_euclidean_guidance"
    if obs == "local50" and guidance == "astar" and critic == "local":
        if gate is not None and abs(gate) <= 1e-9:
            return "b5_local_obs_no_dist80_gate"
        return "ours"
    if obs == "full83" and guidance == "astar" and critic == "ctde_joint_obs_plus_global":
        return "b4_ctde_mappo"
    if obs == "full83" and guidance == "astar" and critic == "local":
        return "b3_full_obs_ippo"
    if canonical_mode in ACTIVE_BASELINES:
        return canonical_mode
    return canonical_mode or "unknown"
