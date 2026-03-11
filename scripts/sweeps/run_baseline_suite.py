"""Sequential runner for baseline training + Stage5 evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.experiment_modes import infer_canonical_baseline

DEFAULT_TRAIN_BASELINES = [
    "ours",
    "b1_traditional_apf_pn",
    "b2_euclidean_guidance",
    "b3_full_obs_ippo",
    "b4_ctde_mappo",
    "b5_local_obs_no_dist80_gate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline suite sequentially (train + Stage5 eval).")
    p.add_argument("--config", type=str, default="src/config.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-envs", type=int, default=None, help="Override train parallel envs; default uses config.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--baselines", type=str, nargs="+", default=DEFAULT_TRAIN_BASELINES)
    p.add_argument(
        "--only",
        type=str,
        default=None,
        choices=DEFAULT_TRAIN_BASELINES + ["b3_local_obs_ippo"],
        help="Only run one baseline across all seeds; overrides --baselines.",
    )
    p.add_argument("--episodes", type=int, default=500, help="Stage5 eval episodes per run.")
    p.add_argument("--stage-index", type=int, default=4, help="Stage5 index (0-based).")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _snapshot_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def _find_new_run(root: Path, before: set[Path], run_tag: str) -> Path:
    after = _snapshot_dirs(root)
    candidates = [p for p in after - before if run_tag in p.name]
    if candidates:
        return sorted(candidates, key=lambda p: p.name)[-1]
    # Fallback: latest timestamp-like or any latest dir.
    all_dirs = [p for p in after if p.is_dir()]
    if not all_dirs:
        raise FileNotFoundError(f"No run directory found under {root}")
    return sorted(all_dirs, key=lambda p: p.name)[-1]


def _find_policy_ckpt(run_dir: Path) -> Path:
    primary = run_dir / "policy.pt"
    if primary.exists():
        return primary
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        pts = sorted(ckpt_dir.glob("policy_*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _infer_run_baseline(run_dir: Path) -> str | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    import json

    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    exp = payload.get("experiment", {})
    reward_runtime = payload.get("reward_runtime", {})
    return infer_canonical_baseline(
        mode=exp.get("mode"),
        observation_profile=exp.get("observation_profile"),
        guidance_backend=exp.get("guidance_backend"),
        critic_mode=exp.get("critic_mode"),
        eval_controller=exp.get("eval_controller"),
        direction_gate_active_radius_m=reward_runtime.get("direction_gate_active_radius_m"),
    )


def _find_existing_run(root: Path, baseline: str, seed: int) -> Path:
    suffix = f"_s{seed}"
    candidates = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        if _infer_run_baseline(run_dir) != baseline:
            continue
        candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No existing run for baseline={baseline} seed={seed} under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def main() -> None:
    args = parse_args()
    baselines = [args.only] if args.only else list(args.baselines)
    baselines = ["ours" if b == "b3_local_obs_ippo" else b for b in baselines]
    cfg_path = Path(args.config)
    runs_root = ROOT / "runs" / "mappo_3d" / "default"
    for baseline in baselines:
        if baseline == "b1_traditional_apf_pn":
            for seed in args.seeds:
                if args.skip_eval:
                    continue
                out = ROOT / "runs" / "baselines" / baseline / f"eval_stage5_s{seed}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                _run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "eval_policy_episodes.py"),
                        "--controller",
                        "traditional_apf_pn",
                        "--baseline",
                        baseline,
                        "--config",
                        str(cfg_path),
                        "--stage-index",
                        str(args.stage_index),
                        "--episodes",
                        str(args.episodes),
                        "--seed",
                        str(seed),
                        "--output",
                        str(out),
                    ]
                )
            continue

        for seed in args.seeds:
            run_tag = f"{baseline}_s{seed}"
            run_dir: Path | None = None
            if not args.skip_train:
                before = _snapshot_dirs(runs_root)
                train_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "train_mappo_3d.py"),
                    "--config",
                    str(cfg_path),
                    "--baseline",
                    baseline,
                    "--seed",
                    str(seed),
                    "--device",
                    str(args.device),
                    "--run-tag",
                    run_tag,
                ]
                if args.num_envs is not None:
                    train_cmd.extend(["--num-envs", str(args.num_envs)])
                _run(train_cmd)
                run_dir = _find_new_run(runs_root, before, run_tag)
            else:
                run_dir = _find_existing_run(runs_root, baseline, seed)

            if args.skip_eval:
                continue
            assert run_dir is not None
            ckpt = _find_policy_ckpt(run_dir)
            eval_out = run_dir / "eval_stage5_episode_eval.jsonl"
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval_policy_episodes.py"),
                    "--policy",
                    str(ckpt),
                    "--baseline",
                    baseline,
                    "--config",
                    str(run_dir / "config.json"),
                    "--stage-index",
                    str(args.stage_index),
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    str(seed),
                    "--output",
                    str(eval_out),
                ]
            )


if __name__ == "__main__":
    main()
