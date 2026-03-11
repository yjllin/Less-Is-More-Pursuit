"""Evaluate Stage5 performance across pursuer yaw-rate limits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.experiment_modes import infer_canonical_baseline

EVAL_SCRIPT = ROOT / "scripts" / "eval_policy_episodes.py"
MODEL_SPECS = {
    "ours": {
        "display_name": "Ours",
        "baseline": "ours",
    },
    "euclidean": {
        "display_name": "Euclidean",
        "baseline": "b2_euclidean_guidance",
    },
    "apf": {
        "display_name": "APF",
        "baseline": "b1_traditional_apf_pn",
    },
}
DEFAULT_MODELS = ["ours", "euclidean", "apf"]
DEFAULT_YAW_LIMITS = ["0.2", "0.4", "0.6", "0.8"]
DEFAULT_SEEDS = [0, 1, 2]
SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage5 yaw-rate sweep for Ours, Euclidean, and APF.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_SPECS.keys()),
        help="Models/controllers to evaluate.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/mappo_3d/default",
        help="Run root used to discover latest policy.pt for learned baselines.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config override. Default: sibling config.json near policy, else src/config.yaml.",
    )
    parser.add_argument("--stage-index", type=int, default=4, help="Stage index for Stage5 (default: 4).")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per yaw-rate limit.")
    parser.add_argument(
        "--yaw-rates",
        type=str,
        nargs="+",
        default=DEFAULT_YAW_LIMITS,
        help="Yaw-rate caps in rad/s. Use 'unlimited' for a very large cap.",
    )
    parser.add_argument(
        "--unlimited-yaw-rate",
        type=float,
        default=100.0,
        help="Numeric yaw-rate cap used to approximate 'unlimited'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deprecated single-seed arg; ignored when --seeds is set.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Training/eval seeds to sweep.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto for evaluator.")
    parser.add_argument("--num-envs", type=int, default=12, help="Parallel env count in evaluator.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action in evaluator.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root. Default: runs/mappo_3d/yaw_rate_sweep/<timestamp>_stage5_yaw_rate.",
    )
    return parser.parse_args()


def infer_config_path(policy_path: Path | None, user_cfg: str | None) -> Path:
    if user_cfg:
        return Path(user_cfg).resolve()
    if policy_path is not None:
        sibling_cfg = policy_path.parent / "config.json"
        if sibling_cfg.exists():
            return sibling_cfg.resolve()
    return (ROOT / "src" / "config.yaml").resolve()


def load_raw_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_yaw_rate_limit(token: str, unlimited_value: float) -> tuple[str, float, float | None]:
    raw = str(token).strip().lower()
    if raw in {"unlimited", "inf", "infinite", "none"}:
        return "unlimited", float(unlimited_value), None
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"Yaw-rate limit must be positive, got: {token}")
    return f"{value:.1f}", value, value


def yaw_rate_tag(label: str) -> str:
    return label.replace(".", "p")


def aggregate_eval(eval_path: Path) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    if total == 0:
        return {
            "episodes": 0.0,
            "success_rate": 0.0,
            "clean_capture_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_steps": 0.0,
            "avg_min_dist": 0.0,
        }

    success = sum(1 for r in rows if bool(r.get("success", False)))
    clean_capture = sum(1 for r in rows if bool(r.get("clean_capture", False)))
    collision = sum(1 for r in rows if bool(r.get("collision", False)))
    timeout = sum(1 for r in rows if bool(r.get("timeout", False)))
    avg_steps = sum(float(r.get("steps", 0.0)) for r in rows) / float(total)
    avg_min_dist = sum(float(r.get("min_dist", 0.0)) for r in rows) / float(total)
    return {
        "episodes": float(total),
        "success_rate": float(success) / float(total),
        "clean_capture_rate": float(clean_capture) / float(total),
        "collision_rate": float(collision) / float(total),
        "timeout_rate": float(timeout) / float(total),
        "avg_steps": float(avg_steps),
        "avg_min_dist": float(avg_min_dist),
    }


def run_eval(
    *,
    policy_path: Path | None,
    eval_cfg_path: Path,
    out_jsonl: Path,
    stage_index: int,
    episodes: int,
    seed: int,
    device: str,
    num_envs: int,
    deterministic: bool,
    baseline: str,
    yaw_rate_label: str,
) -> None:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(eval_cfg_path),
        "--stage-index",
        str(stage_index),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--device",
        str(device),
        "--num-envs",
        str(num_envs),
        "--output",
        str(out_jsonl),
        "--eval-tag",
        f"stage{stage_index + 1}_yaw_rate_{yaw_rate_label}",
        "--baseline",
        baseline,
    ]
    if policy_path is not None:
        cmd.extend(["--policy", str(policy_path)])
    else:
        cmd.extend(["--controller", "traditional_apf_pn"])
    if deterministic:
        cmd.append("--deterministic")

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def discover_seeded_policy_runs(runs_root: Path) -> Dict[str, Dict[int, Path]]:
    discovered: Dict[str, Dict[int, Path]] = {}
    if not runs_root.exists():
        return discovered
    for run_dir in [p for p in runs_root.iterdir() if p.is_dir()]:
        cfg_path = run_dir / "config.json"
        policy_path = run_dir / "policy.pt"
        if not cfg_path.exists() or not policy_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exp = cfg.get("experiment", {})
        reward_runtime = cfg.get("reward_runtime", {})
        baseline = infer_canonical_baseline(
            mode=exp.get("mode"),
            observation_profile=exp.get("observation_profile"),
            guidance_backend=exp.get("guidance_backend"),
            critic_mode=exp.get("critic_mode"),
            eval_controller=exp.get("eval_controller"),
            direction_gate_active_radius_m=reward_runtime.get("direction_gate_active_radius_m"),
        )
        seed_match = SEED_SUFFIX_RE.search(run_dir.name)
        if seed_match is None:
            continue
        model_seed = int(seed_match.group(1))
        by_seed = discovered.setdefault(baseline, {})
        prev = by_seed.get(model_seed)
        if prev is None or run_dir.name > prev.name:
            by_seed[model_seed] = run_dir
    return discovered


def evaluate_one_model(
    *,
    model_key: str,
    policy_path: Path | None,
    base_cfg_path: Path,
    out_root: Path,
    stage_index: int,
    episodes: int,
    yaw_limits: list[tuple[str, float, float | None]],
    seed: int,
    device: str,
    num_envs: int,
    deterministic: bool,
    model_seed: int | None = None,
) -> List[Dict[str, Any]]:
    spec = MODEL_SPECS[model_key]
    raw_cfg = load_raw_config(base_cfg_path)
    stages = raw_cfg.get("curriculum", {}).get("stages", [])
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError(f"Invalid config: curriculum.stages is missing or empty in {base_cfg_path}")
    if stage_index < 0 or stage_index >= len(stages):
        raise IndexError(f"stage-index {stage_index} out of range [0, {len(stages) - 1}] for {base_cfg_path}")

    config_dir = out_root / "configs"
    logs_dir = out_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    print(f"[Info] model={spec['display_name']} baseline={spec['baseline']}")
    print(f"[Info] base_config={base_cfg_path}")
    print(f"[Info] output_dir={out_root}")
    print(f"[Info] stage_index={stage_index} stage_name={stages[stage_index].get('name', 'unknown')}")
    if model_seed is not None:
        print(f"[Info] model_seed={model_seed}")

    for label, effective_cap, reported_cap in yaw_limits:
        cfg = deepcopy(raw_cfg)
        control = dict(cfg.get("control", {}))
        action_bounds = dict(control.get("action_bounds", {}))
        action_bounds["yaw_rate"] = float(effective_cap)
        control["action_bounds"] = action_bounds
        cfg["control"] = control

        tag = yaw_rate_tag(label)
        eval_cfg_path = config_dir / f"config_stage{stage_index + 1}_yaw_rate_{tag}.json"
        eval_log_path = logs_dir / f"eval_stage{stage_index + 1}_yaw_rate_{tag}.jsonl"
        save_json(eval_cfg_path, cfg)

        run_eval(
            policy_path=policy_path,
            eval_cfg_path=eval_cfg_path,
            out_jsonl=eval_log_path,
            stage_index=int(stage_index),
            episodes=int(episodes),
            seed=int(seed),
            device=str(device),
            num_envs=int(num_envs),
            deterministic=bool(deterministic),
            baseline=str(spec["baseline"]),
            yaw_rate_label=label,
        )

        agg = aggregate_eval(eval_log_path)
        row = {
            "model": model_key,
            "display_name": str(spec["display_name"]),
            "baseline": str(spec["baseline"]),
            "model_seed": int(model_seed) if model_seed is not None else int(seed),
            "eval_seed": int(seed),
            "yaw_rate_label": str(label),
            "yaw_rate_limit_radps": reported_cap,
            "effective_yaw_rate_cap_radps": float(effective_cap),
            "stage_index": int(stage_index),
            "stage_name": str(stages[stage_index].get("name", "unknown")),
            "policy_path": str(policy_path) if policy_path is not None else "",
            "config_path": str(eval_cfg_path),
            "eval_log_path": str(eval_log_path),
            "seed": int(seed),
            "episodes_requested": int(episodes),
            **agg,
        }
        summary_rows.append(row)
        print(
            f"[Done] model={spec['display_name']} yaw_rate={label} "
            f"success_rate={row['success_rate']:.3f} clean_capture_rate={row['clean_capture_rate']:.3f} "
            f"collision_rate={row['collision_rate']:.3f}",
            flush=True,
        )

    summary_jsonl = out_root / "yaw_rate_sweep_summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as fp:
        for row in summary_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = out_root / "yaw_rate_sweep_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[Saved] {summary_jsonl}")
    print(f"[Saved] {summary_csv}")
    return summary_rows


def main() -> None:
    args = parse_args()
    yaw_limits = [parse_yaw_rate_limit(v, args.unlimited_yaw_rate) for v in args.yaw_rates]
    if not yaw_limits:
        raise ValueError("--yaw-rates must not be empty.")

    runs_root = Path(args.runs_root).resolve()
    discovered = discover_seeded_policy_runs(runs_root)

    if args.output_dir:
        root_out = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_out = (ROOT / "runs" / "mappo_3d" / "yaw_rate_sweep" / f"{ts}_stage5_yaw_rate").resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        baseline = str(spec["baseline"])
        if baseline == "b1_traditional_apf_pn":
            for eval_seed in args.seeds:
                policy_path = None
                base_cfg = infer_config_path(None, args.config)
                if not base_cfg.exists():
                    print(f"[Skip] model={model_key} missing config: {base_cfg}")
                    continue
                out_root = root_out / model_key / f"s{int(eval_seed)}"
                rows = evaluate_one_model(
                    model_key=model_key,
                    policy_path=policy_path,
                    base_cfg_path=base_cfg,
                    out_root=out_root,
                    stage_index=int(args.stage_index),
                    episodes=int(args.episodes),
                    yaw_limits=yaw_limits,
                    seed=int(eval_seed),
                    device=str(args.device),
                    num_envs=int(args.num_envs),
                    deterministic=bool(args.deterministic),
                    model_seed=int(eval_seed),
                )
                all_rows.extend(rows)
            continue

        seeded_runs = discovered.get(baseline, {})
        if not seeded_runs:
            print(f"[Skip] model={model_key} baseline={baseline} no discovered seeded policy.pt under {runs_root}")
            continue

        for model_seed in args.seeds:
            run_dir = seeded_runs.get(int(model_seed))
            if run_dir is None:
                print(f"[Skip] model={model_key} baseline={baseline} missing seed s{int(model_seed)}")
                continue
            policy_path = (run_dir / "policy.pt").resolve()
            base_cfg = infer_config_path(policy_path, args.config)

            if not base_cfg.exists():
                print(f"[Skip] model={model_key} seed=s{int(model_seed)} missing config: {base_cfg}")
                continue
            if not policy_path.exists():
                print(f"[Skip] model={model_key} seed=s{int(model_seed)} missing policy: {policy_path}")
                continue

            out_root = root_out / model_key / f"s{int(model_seed)}"
            rows = evaluate_one_model(
                model_key=model_key,
                policy_path=policy_path,
                base_cfg_path=base_cfg,
                out_root=out_root,
                stage_index=int(args.stage_index),
                episodes=int(args.episodes),
                yaw_limits=yaw_limits,
                seed=int(model_seed),
                device=str(args.device),
                num_envs=int(args.num_envs),
                deterministic=bool(args.deterministic),
                model_seed=int(model_seed),
            )
            all_rows.extend(rows)

    if all_rows:
        merged_jsonl = root_out / "full_yaw_rate_sweep_summary.jsonl"
        with merged_jsonl.open("w", encoding="utf-8") as fp:
            for row in all_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        merged_csv = root_out / "full_yaw_rate_sweep_summary.csv"
        with merged_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[Saved] {merged_jsonl}")
        print(f"[Saved] {merged_csv}")
    else:
        print("[Warn] yaw-rate sweep produced no evaluation rows.")


if __name__ == "__main__":
    main()
