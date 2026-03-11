"""MAPPO 3D training entrypoint using VectorizedMutualAStarEnvV2."""

from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import sys
import os
import faulthandler

# Force single-threaded BLAS/OpenMP in-process to avoid oversubscription.
# Parallelism should come from vectorized env concurrency, not hidden math-library threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from src.config import load_config
from src.controllers import MAPPOTrainerVec
from src.experiment_modes import BASELINE_CHOICES, get_baseline_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO policy in 3D using vectorized environment.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: src/config.yaml).")
    parser.add_argument("--timesteps", type=int, help="Override training timesteps.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments in vectorized env (default: use config.training.parallel_envs).",
    )
    parser.add_argument("--device", type=str, default='cuda', help="torch device, e.g., cuda or cpu.")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=BASELINE_CHOICES,
        help="Baseline mode switch (unified experimental entrypoint).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed (recorded in config/run tag only).")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag suffix for output directory.")
    parser.add_argument(
        "--guidance-backend",
        type=str,
        default=None,
        choices=["astar", "euclidean"],
        help="Override experiment.guidance_backend.",
    )
    parser.add_argument(
        "--obs-profile",
        type=str,
        default=None,
        choices=["full83", "local50"],
        help="Override experiment.observation_profile.",
    )
    parser.add_argument(
        "--critic-mode",
        type=str,
        default=None,
        choices=["local", "ctde_joint_obs_plus_global"],
        help="Override experiment.critic_mode.",
    )
    parser.add_argument("--init-policy", type=str, default=None, help="Path to policy checkpoint to resume from.")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Curriculum stage name to start from (e.g., stage1_basic_hover).",
    )
    parser.add_argument(
        "--spawn-level",
        type=int,
        default=None,
        help="Override spawn level (1-based) for target spawn distance.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    cfg = load_config(args.config) if args.config else load_config()
    
    if args.timesteps is not None:
        cfg = replace(cfg, training=replace(cfg.training, timesteps=args.timesteps))
    
    # Update parallel_envs only when explicitly overridden on CLI.
    if args.num_envs is not None:
        cfg = replace(cfg, training=replace(cfg.training, parallel_envs=args.num_envs))

    # Unified baseline mode mapping
    exp = cfg.experiment
    direction_gate_radius_override = None
    if args.baseline:
        override = get_baseline_override(args.baseline)
        assert override is not None
        direction_gate_radius_override = override["direction_gate_active_radius_m"]
        exp = replace(
            exp,
            mode=override["mode"],
            observation_profile=override["observation_profile"],
            guidance_backend=override["guidance_backend"],
            critic_mode=override["critic_mode"],
            eval_controller=override["eval_controller"],
        )
    if args.guidance_backend:
        exp = replace(exp, guidance_backend=args.guidance_backend)
    if args.obs_profile:
        exp = replace(exp, observation_profile=args.obs_profile)
    if args.critic_mode:
        exp = replace(exp, critic_mode=args.critic_mode)
    if args.run_tag is not None:
        exp = replace(exp, run_tag=str(args.run_tag))
    cfg = replace(cfg, experiment=exp)
    if direction_gate_radius_override is not None:
        cfg = replace(
            cfg,
            reward_runtime=replace(
                cfg.reward_runtime,
                direction_gate_active_radius_m=float(direction_gate_radius_override),
            ),
        )

    # Timestamped log dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (args.run_tag if args.run_tag is not None else getattr(cfg.experiment, "run_tag", "")).strip()
    if run_tag:
        run_id = f"{run_id}_{run_tag}"
    log_root = cfg.logging.output_root / run_id
    cfg = replace(cfg, logging=replace(cfg.logging, output_root=log_root))
    return cfg


def resolve_stage_index(cfg, stage_name: str | None) -> int | None:
    if stage_name is None:
        return None
    for idx, stage in enumerate(cfg.curriculum.stages):
        if stage.name == stage_name:
            return idx
    raise ValueError(f"Stage '{stage_name}' not found in curriculum.")


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    if args.baseline == "b1_traditional_apf_pn":
        raise ValueError("Baseline b1_traditional_apf_pn is evaluation-only. Use eval_policy_episodes.py/eval_policy_sim.py with --controller traditional_apf_pn.")
    cfg = build_config(args)
    print("[Train] Config loaded, building trainer...", flush=True)
    num_envs = int(args.num_envs) if args.num_envs is not None else int(cfg.training.parallel_envs)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_stage_index = resolve_stage_index(cfg, args.stage)

    trainer = MAPPOTrainerVec(
        config=cfg,
        num_envs=num_envs,
        device=device,
        init_policy=args.init_policy,
        start_stage_index=start_stage_index,
        start_spawn_level=args.spawn_level,
    )
    print("[Train] Trainer initialized, starting train loop...", flush=True)

    trainer.train()


if __name__ == "__main__":
    main()
