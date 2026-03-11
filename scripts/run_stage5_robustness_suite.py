"""Run the Stage5 robustness evaluation suite in a fixed order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEED_SCRIPT = ROOT / "scripts" / "sweeps" / "eval_stage5_speed_sweep.py"
YAW_SCRIPT = ROOT / "scripts" / "sweeps" / "eval_stage5_yaw_rate_sweep.py"
NOISE_SCRIPT = ROOT / "scripts" / "sweeps" / "eval_stage5_observation_noise_sweep.py"
DELAY_SCRIPT = ROOT / "scripts" / "sweeps" / "eval_stage5_action_delay_sweep.py"
ROBUSTNESS_ROOT = ROOT / "runs" / "mappo_3d" / "robustness_suite"
MANIFEST_NAME = "suite_manifest.json"
STATUS_NAME = "suite_status.json"
DEFAULT_SPEED_BASELINES = ["ours", "b1_traditional_apf_pn", "b2_euclidean_guidance", "b3_full_obs_ippo"]
DEFAULT_SPEEDS = [7.0, 8.0, 8.5, 9.0, 9.5, 10.0]
DEFAULT_YAW_MODELS = ["ours", "euclidean", "apf"]
DEFAULT_YAW_RATES = ["0.2", "0.4", "0.6", "0.8"]
DEFAULT_NOISE_MODELS = ["ours", "full_obs", "euclidean"]
DEFAULT_NOISE_LEVELS = ["0.0", "0.05", "0.10", "0.20"]
DEFAULT_DELAY_MODELS = ["ours", "full_obs", "euclidean"]
DEFAULT_DELAY_STEPS = [0, 1, 2, 3]
DEFAULT_DELAY_EVAL_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_DELAY_SKIPPED_SEEDS = [0, 1, 2]
DEFAULT_DELAY_MODEL_SEED_MAP = {3: 0, 4: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage5 robustness sweeps in order: speed, yaw-rate, observation-noise, action-delay."
        )
    )
    parser.add_argument("--runs-root", type=str, default="runs/mappo_3d/default", help="Run root for policy discovery.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override passed to all sub-scripts.")
    parser.add_argument("--stage-index", type=int, default=4, help="Stage index for Stage5 (default: 4).")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per setting in each sweep.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Training/eval seeds to sweep.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto passed to all sub-scripts.")
    parser.add_argument("--num-envs", type=int, default=12, help="Parallel env count passed to all sub-scripts.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action in all sub-scripts.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional suite output root. Sub-scripts write into speed_sweep/yaw_rate_sweep/observation_noise_sweep/action_delay_sweep beneath it.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Existing robustness-suite directory (folder name or path). Detect missing items and only backfill them.",
    )
    parser.add_argument("--max-passes", type=int, default=8, help="Maximum detect-and-backfill passes.")
    parser.add_argument("--skip-speed", action="store_true", help="Skip the speed sweep.")
    parser.add_argument("--skip-yaw", action="store_true", help="Skip the yaw-rate sweep.")
    parser.add_argument("--skip-noise", action="store_true", help="Skip the observation-noise sweep.")
    parser.add_argument("--skip-delay", action="store_true", help="Skip the action-delay sweep.")
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> int:
    print(f"[Suite] start {name}", flush=True)
    print("[Suite] cmd:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if completed.returncode == 0:
        print(f"[Suite] done {name}", flush=True)
    else:
        print(f"[Suite] failed {name} returncode={completed.returncode}", flush=True)
    return int(completed.returncode)


def build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--runs-root",
        str(Path(args.runs_root)),
        "--stage-index",
        str(int(args.stage_index)),
        "--episodes",
        str(int(args.episodes)),
        "--device",
        str(args.device),
        "--num-envs",
        str(int(args.num_envs)),
    ]
    if args.config:
        common.extend(["--config", str(Path(args.config))])
    if args.deterministic:
        common.append("--deterministic")
    return common


def resolve_suite_root(output_root: str | None, resume_dir: str | None) -> Path:
    if resume_dir:
        candidate = Path(resume_dir)
        if candidate.exists():
            return candidate.resolve()
        fallback = ROBUSTNESS_ROOT / resume_dir
        return fallback.resolve()
    if output_root:
        return Path(output_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROBUSTNESS_ROOT / f"{ts}_stage5_robustness").resolve()


def manifest_path(suite_root: Path) -> Path:
    return suite_root / MANIFEST_NAME


def status_path(suite_root: Path) -> Path:
    return suite_root / STATUS_NAME


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest(args: argparse.Namespace, suite_root: Path) -> dict[str, Any]:
    enabled_steps = {
        "speed_sweep": not bool(args.skip_speed),
        "yaw_rate_sweep": not bool(args.skip_yaw),
        "observation_noise_sweep": not bool(args.skip_noise),
        "action_delay_sweep": not bool(args.skip_delay),
    }
    return {
        "suite_root": str(suite_root),
        "runs_root": str(Path(args.runs_root)),
        "config": str(Path(args.config)) if args.config else None,
        "stage_index": int(args.stage_index),
        "episodes": int(args.episodes),
        "seeds": [int(s) for s in args.seeds],
        "device": str(args.device),
        "num_envs": int(args.num_envs),
        "deterministic": bool(args.deterministic),
        "max_passes": int(args.max_passes),
        "enabled_steps": enabled_steps,
        "speed_sweep": {
            "subdir": "speed_sweep",
            "baselines": DEFAULT_SPEED_BASELINES,
            "speeds": DEFAULT_SPEEDS,
        },
        "yaw_rate_sweep": {
            "subdir": "yaw_rate_sweep",
            "models": DEFAULT_YAW_MODELS,
            "yaw_rates": DEFAULT_YAW_RATES,
        },
        "observation_noise_sweep": {
            "subdir": "observation_noise_sweep",
            "models": DEFAULT_NOISE_MODELS,
            "noise_levels": DEFAULT_NOISE_LEVELS,
        },
        "action_delay_sweep": {
            "subdir": "action_delay_sweep",
            "models": DEFAULT_DELAY_MODELS,
            "delay_steps": DEFAULT_DELAY_STEPS,
            "eval_seeds": DEFAULT_DELAY_EVAL_SEEDS,
            "skipped_eval_seeds": DEFAULT_DELAY_SKIPPED_SEEDS,
            "eval_to_model_seed": {str(k): int(v) for k, v in DEFAULT_DELAY_MODEL_SEED_MAP.items()},
        },
    }


def normalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    action_delay_cfg = manifest.setdefault("action_delay_sweep", {})
    action_delay_cfg.setdefault("subdir", "action_delay_sweep")
    action_delay_cfg.setdefault("models", list(DEFAULT_DELAY_MODELS))
    action_delay_cfg.setdefault("delay_steps", list(DEFAULT_DELAY_STEPS))
    action_delay_cfg.setdefault("eval_seeds", list(DEFAULT_DELAY_EVAL_SEEDS))
    action_delay_cfg.setdefault("skipped_eval_seeds", list(DEFAULT_DELAY_SKIPPED_SEEDS))
    action_delay_cfg.setdefault("eval_to_model_seed", {str(k): int(v) for k, v in DEFAULT_DELAY_MODEL_SEED_MAP.items()})
    return manifest


def ensure_manifest(args: argparse.Namespace, suite_root: Path) -> dict[str, Any]:
    path = manifest_path(suite_root)
    if path.exists():
        manifest = normalize_manifest(load_json(path))
        save_json(path, manifest)
        return manifest
    manifest = normalize_manifest(build_manifest(args, suite_root))
    save_json(path, manifest)
    return manifest


def make_dir_label(baseline: str) -> str:
    if baseline == "b1_traditional_apf_pn":
        return "APF+PN"
    return baseline


def has_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def speed_tag(speed: float) -> str:
    return f"{float(speed):.1f}".replace(".", "p")


def yaw_tag(value: str) -> str:
    return str(value).replace(".", "p")


def noise_tag(value: str) -> str:
    text = str(value)
    trimmed = text.rstrip("0").rstrip(".")
    return (trimmed or "0").replace(".", "p")


def delay_tag(delay_steps: int) -> str:
    return f"{int(delay_steps)}step"


def detect_missing(suite_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    missing: dict[str, Any] = {}
    enabled = manifest.get("enabled_steps", {})
    seeds = [int(s) for s in manifest.get("seeds", [])]

    if enabled.get("speed_sweep", False):
        step_cfg = manifest["speed_sweep"]
        step_root = suite_root / step_cfg["subdir"]
        speed_missing: dict[str, dict[str, list[float]]] = {}
        for baseline in step_cfg["baselines"]:
            dir_label = make_dir_label(str(baseline))
            for seed in seeds:
                missing_speeds: list[float] = []
                for speed in step_cfg["speeds"]:
                    log_path = step_root / dir_label / f"s{seed}" / "logs" / f"eval_stage5_speed_{speed_tag(speed)}.jsonl"
                    if not has_nonempty_file(log_path):
                        missing_speeds.append(float(speed))
                if missing_speeds:
                    speed_missing.setdefault(str(baseline), {})[str(seed)] = missing_speeds
        if speed_missing:
            missing["speed_sweep"] = speed_missing

    if enabled.get("yaw_rate_sweep", False):
        step_cfg = manifest["yaw_rate_sweep"]
        step_root = suite_root / step_cfg["subdir"]
        yaw_missing: dict[str, dict[str, list[str]]] = {}
        for model in step_cfg["models"]:
            for seed in seeds:
                missing_rates: list[str] = []
                for value in step_cfg["yaw_rates"]:
                    log_path = step_root / str(model) / f"s{seed}" / "logs" / f"eval_stage5_yaw_rate_{yaw_tag(str(value))}.jsonl"
                    if not has_nonempty_file(log_path):
                        missing_rates.append(str(value))
                if missing_rates:
                    yaw_missing.setdefault(str(model), {})[str(seed)] = missing_rates
        if yaw_missing:
            missing["yaw_rate_sweep"] = yaw_missing

    if enabled.get("observation_noise_sweep", False):
        step_cfg = manifest["observation_noise_sweep"]
        step_root = suite_root / step_cfg["subdir"]
        noise_missing: dict[str, dict[str, list[str]]] = {}
        for model in step_cfg["models"]:
            for seed in seeds:
                missing_levels: list[str] = []
                for value in step_cfg["noise_levels"]:
                    log_path = step_root / str(model) / f"s{seed}" / "logs" / f"eval_stage5_noise_{noise_tag(str(value))}.jsonl"
                    if not has_nonempty_file(log_path):
                        missing_levels.append(str(value))
                if missing_levels:
                    noise_missing.setdefault(str(model), {})[str(seed)] = missing_levels
        if noise_missing:
            missing["observation_noise_sweep"] = noise_missing

    if enabled.get("action_delay_sweep", False):
        step_cfg = manifest["action_delay_sweep"]
        step_root = suite_root / step_cfg["subdir"]
        eval_seeds = [int(s) for s in step_cfg.get("eval_seeds", seeds)]
        skipped_eval_seeds = {int(s) for s in step_cfg.get("skipped_eval_seeds", [])}
        delay_missing: dict[str, dict[str, list[int]]] = {}
        for model in step_cfg["models"]:
            for seed in eval_seeds:
                if seed in skipped_eval_seeds:
                    continue
                missing_delays: list[int] = []
                for value in step_cfg["delay_steps"]:
                    log_path = step_root / str(model) / f"s{seed}" / "logs" / f"eval_stage5_action_delay_{delay_tag(int(value))}.jsonl"
                    if not has_nonempty_file(log_path):
                        missing_delays.append(int(value))
                if missing_delays:
                    delay_missing.setdefault(str(model), {})[str(seed)] = missing_delays
        if delay_missing:
            missing["action_delay_sweep"] = delay_missing

    return missing


def summarize_missing(missing: dict[str, Any]) -> str:
    if not missing:
        return "none"
    parts: list[str] = []
    for step, spec in missing.items():
        combos = sum(len(seed_map) for seed_map in spec.values())
        parts.append(f"{step}:{combos}")
    return ", ".join(parts)


def build_retry_jobs(manifest: dict[str, Any], suite_root: Path, common: list[str], missing: dict[str, Any]) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    if "speed_sweep" in missing:
        step_root = suite_root / manifest["speed_sweep"]["subdir"]
        for baseline, seed_map in missing["speed_sweep"].items():
            for seed_str, speeds in seed_map.items():
                cmd = [
                    sys.executable,
                    str(SPEED_SCRIPT),
                    *common,
                    "--full",
                    "--baselines",
                    str(baseline),
                    "--seeds",
                    str(int(seed_str)),
                    "--speeds",
                    *[str(float(v)) for v in speeds],
                    "--output-dir",
                    str(step_root),
                ]
                jobs.append((f"speed_sweep:{baseline}:s{seed_str}", cmd))

    if "yaw_rate_sweep" in missing:
        step_root = suite_root / manifest["yaw_rate_sweep"]["subdir"]
        for model, seed_map in missing["yaw_rate_sweep"].items():
            for seed_str, yaw_rates in seed_map.items():
                cmd = [
                    sys.executable,
                    str(YAW_SCRIPT),
                    *common,
                    "--models",
                    str(model),
                    "--seeds",
                    str(int(seed_str)),
                    "--yaw-rates",
                    *[str(v) for v in yaw_rates],
                    "--output-dir",
                    str(step_root),
                ]
                jobs.append((f"yaw_rate_sweep:{model}:s{seed_str}", cmd))

    if "observation_noise_sweep" in missing:
        step_root = suite_root / manifest["observation_noise_sweep"]["subdir"]
        for model, seed_map in missing["observation_noise_sweep"].items():
            for seed_str, noise_levels in seed_map.items():
                cmd = [
                    sys.executable,
                    str(NOISE_SCRIPT),
                    *common,
                    "--models",
                    str(model),
                    "--seeds",
                    str(int(seed_str)),
                    "--noise-levels",
                    *[str(v) for v in noise_levels],
                    "--output-dir",
                    str(step_root),
                ]
                jobs.append((f"observation_noise_sweep:{model}:s{seed_str}", cmd))

    if "action_delay_sweep" in missing:
        step_root = suite_root / manifest["action_delay_sweep"]["subdir"]
        eval_to_model_seed = {
            int(eval_seed): int(model_seed)
            for eval_seed, model_seed in manifest["action_delay_sweep"].get("eval_to_model_seed", {}).items()
        }
        for model, seed_map in missing["action_delay_sweep"].items():
            for seed_str, delay_steps in seed_map.items():
                eval_seed = int(seed_str)
                cmd = [
                    sys.executable,
                    str(DELAY_SCRIPT),
                    *common,
                    "--models",
                    str(model),
                    "--seeds",
                    str(eval_seed),
                    "--delay-steps",
                    *[str(int(v)) for v in delay_steps],
                    "--output-dir",
                    str(step_root),
                ]
                model_seed = eval_to_model_seed.get(eval_seed)
                if model_seed is not None:
                    cmd.extend(["--model-seed-map", f"{eval_seed}:{model_seed}"])
                jobs.append((f"action_delay_sweep:{model}:s{seed_str}", cmd))
    return jobs


def save_status(suite_root: Path, payload: dict[str, Any]) -> None:
    save_json(status_path(suite_root), payload)


def main() -> None:
    args = parse_args()
    common = build_common_args(args)
    suite_root = resolve_suite_root(args.output_root, args.resume_dir)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(args, suite_root)

    enabled_steps = manifest.get("enabled_steps", {})
    if not any(bool(v) for v in enabled_steps.values()):
        raise ValueError("All suite steps are disabled in the manifest. Nothing to run.")

    print(f"[Suite] output_root={suite_root}", flush=True)
    save_status(
        suite_root,
        {
            "suite_root": str(suite_root),
            "mode": "resume" if args.resume_dir else "fresh",
            "pass": 0,
            "status": "running",
            "last_missing": {},
            "last_failures": [],
        },
    )

    previous_missing_signature: str | None = None
    final_missing: dict[str, Any] = {}
    for pass_idx in range(1, int(args.max_passes) + 1):
        missing = detect_missing(suite_root, manifest)
        final_missing = missing
        if not missing:
            save_status(
                suite_root,
                {
                    "suite_root": str(suite_root),
                    "mode": "resume" if args.resume_dir else "fresh",
                    "pass": pass_idx,
                    "status": "complete",
                    "last_missing": {},
                    "last_failures": [],
                },
            )
            print(f"[Suite] complete after pass {pass_idx - 1}", flush=True)
            return

        jobs = build_retry_jobs(manifest, suite_root, common, missing)
        if not jobs:
            break

        missing_signature = json.dumps(missing, sort_keys=True, ensure_ascii=False)
        if previous_missing_signature == missing_signature and pass_idx > 1:
            print("[Suite] missing set unchanged since last pass; continuing until max_passes.", flush=True)
        previous_missing_signature = missing_signature

        print(f"[Suite] pass={pass_idx} missing={summarize_missing(missing)}", flush=True)
        failures: list[dict[str, Any]] = []
        for name, cmd in jobs:
            returncode = run_step(name, cmd)
            if returncode != 0:
                failures.append({"job": name, "returncode": int(returncode), "cmd": cmd})

        save_status(
            suite_root,
            {
                "suite_root": str(suite_root),
                "mode": "resume" if args.resume_dir else "fresh",
                "pass": pass_idx,
                "status": "running",
                "last_missing": missing,
                "last_failures": failures,
            },
        )

    final_missing = detect_missing(suite_root, manifest)
    status = "complete" if not final_missing else "incomplete"
    save_status(
        suite_root,
        {
            "suite_root": str(suite_root),
            "mode": "resume" if args.resume_dir else "fresh",
            "pass": int(args.max_passes),
            "status": status,
            "last_missing": final_missing,
            "last_failures": [],
        },
    )
    if final_missing:
        print(f"[Suite] incomplete after max_passes={int(args.max_passes)} missing={summarize_missing(final_missing)}", flush=True)
        raise RuntimeError("Robustness suite remains incomplete after max_passes.")
    print("[Suite] complete", flush=True)


if __name__ == "__main__":
    main()
