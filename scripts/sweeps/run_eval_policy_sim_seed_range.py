"""Run scripts/eval_policy_sim.py across an inclusive seed range."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
EVAL_SCRIPT = ROOT / "scripts" / "eval_policy_sim.py"
DEFAULT_POLICY = ROOT / "runs" / "mappo_3d" / "default" / "20260301_083512_ours_s0" / "policy.pt"
DEFAULT_CONFIG = ROOT / "runs" / "mappo_3d" / "default" / "20260301_083512_ours_s0" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval_policy_sim.py for each seed in an inclusive range."
    )
    parser.add_argument("--seed-start", type=int, required=True, help="Inclusive start seed.")
    parser.add_argument("--seed-end", type=int, required=True, help="Inclusive end seed.")
    parser.add_argument("--policy", type=str, default=str(DEFAULT_POLICY), help="Policy checkpoint path.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Config path.")
    parser.add_argument("--stage-index", type=int, default=4, help="Stage index passed to eval_policy_sim.py.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch eval.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for per-seed logs and exploration maps. Default: eval_logs/eval_policy_sim_seed_range/<timestamp>.",
    )
    parser.add_argument(
        "--explore-maps",
        dest="explore_maps",
        action="store_true",
        default=True,
        help="Enable --explore-maps for each launched eval run (default: enabled).",
    )
    parser.add_argument(
        "--no-explore-maps",
        dest="explore_maps",
        action="store_false",
        help="Disable --explore-maps for each launched eval run.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one seed run fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to eval_policy_sim.py. Prefix with --, e.g. -- --steps 400 --device cuda",
    )
    return parser.parse_args()


def resolve_output_root(output_root: str | None) -> Path:
    if output_root:
        return Path(output_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROOT / "eval_logs" / "eval_policy_sim_seed_range" / timestamp).resolve()


def normalize_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def build_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(EVAL_SCRIPT),
        "--policy",
        str(Path(args.policy).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--stage-index",
        str(int(args.stage_index)),
        "--seed",
        str(int(seed)),
        "--output",
        str((run_dir / "eval_log.txt").resolve()),
    ]
    if args.explore_maps:
        cmd.extend(
            [
                "--explore-maps",
                "--explore-out",
                str((run_dir / "explore").resolve()),
            ]
        )
    cmd.extend(normalize_extra_args(list(args.extra_args)))
    return cmd


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError(f"seed-end must be >= seed-start, got {args.seed_start}..{args.seed_end}")

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    summary: list[dict[str, object]] = []

    for seed in seeds:
        run_dir = output_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, seed, run_dir)
        print(f"[Seed {seed}] {' '.join(cmd)}", flush=True)

        if args.dry_run:
            summary.append(
                {
                    "seed": int(seed),
                    "returncode": None,
                    "run_dir": str(run_dir),
                    "command": cmd,
                }
            )
            continue

        completed = subprocess.run(cmd, check=False, cwd=str(ROOT))
        summary.append(
            {
                "seed": int(seed),
                "returncode": int(completed.returncode),
                "run_dir": str(run_dir),
                "command": cmd,
            }
        )
        if completed.returncode != 0:
            print(f"[Seed {seed}] failed with returncode={completed.returncode}", flush=True)
            if args.stop_on_error:
                break

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
