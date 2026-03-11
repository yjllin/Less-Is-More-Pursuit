"""Generate random occupancy maps and evaluate the ours seed models on them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs" / "mappo_3d" / "default"
DEFAULT_OUTPUT_BASE = ROOT / "runs" / "mappo_3d" / "generalization_suite"
DEFAULT_SOURCE_MAP = ROOT / "artifacts" / "level_occupancy.npy"
DEFAULT_EVAL_SCRIPT = ROOT / "scripts" / "eval_policy_episodes.py"
DEFAULT_DENSITIES = [0.08, 0.12, 0.16, 0.20, 0.24]
DEFAULT_DIFFICULTIES = ["easy", "medium_easy", "medium", "medium_hard", "hard"]
EVAL_SUMMARY_FIELDS = [
    "model_seed",
    "run_name",
    "map_id",
    "difficulty",
    "target_density",
    "actual_density",
    "eval_seed",
    "episodes",
    "success_rate",
    "capture_rate",
    "clean_capture_rate",
    "encircle_capture_rate",
    "collision_rate",
    "timeout_rate",
    "avg_steps",
    "avg_min_dist",
    "avg_visible_ratio",
    "avg_far_idle_ratio",
    "avg_reward_total",
    "output_jsonl",
    "map_path",
    "config_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random maps with different obstacle densities and evaluate ours seed models on them."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Training run root.")
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP, help="Reference occupancy map (.npy).")
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=DEFAULT_DENSITIES,
        help="Target obstacle densities for the generated maps.",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=DEFAULT_DIFFICULTIES,
        help="Difficulty labels for generated maps. Must match the number of densities.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Model seeds to evaluate.")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per model-map evaluation.")
    parser.add_argument("--stage-index", type=int, default=4, help="Curriculum stage index.")
    parser.add_argument("--num-envs", type=int, default=12, help="Parallel environment count.")
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter.")
    parser.add_argument("--suite-seed", type=int, default=20260309, help="Master seed for map generation.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Suite output root. Default: runs/mappo_3d/generalization_suite/<timestamp>_random_maps.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--dry-run", action="store_true", help="Write maps/configs but skip subprocess evaluation.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one evaluation subprocess fails.",
    )
    return parser.parse_args()


def ensure_equal_lengths(densities: list[float], difficulties: list[str]) -> None:
    if len(densities) != len(difficulties):
        raise ValueError(
            f"--densities count ({len(densities)}) must equal --difficulties count ({len(difficulties)})."
        )


def resolve_output_root(output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_BASE / f"{stamp}_random_maps").resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference_map(source_map: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not source_map.exists():
        raise FileNotFoundError(f"Reference map not found: {source_map}")
    meta_path = source_map.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Reference map metadata not found: {meta_path}")
    occupancy = np.load(source_map).astype(np.int8)
    meta = load_json(meta_path)
    return occupancy, meta


def discover_ours_runs(runs_root: Path, model_seeds: list[int]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not runs_root.exists():
        raise FileNotFoundError(f"Run root not found: {runs_root}")

    candidates = [path for path in runs_root.iterdir() if path.is_dir() and "_ours_s" in path.name]
    candidates.sort(key=lambda item: item.name)

    for seed in model_seeds:
        pattern = re.compile(rf"_ours_s{seed}$")
        matches = [path for path in candidates if pattern.search(path.name)]
        if not matches:
            raise FileNotFoundError(f"Could not find ours run for seed {seed} in {runs_root}")
        run_dir = matches[-1]
        policy_path = run_dir / "policy.pt"
        config_path = run_dir / "config.json"
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing policy checkpoint: {policy_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        entries.append(
            {
                "seed": int(seed),
                "run_dir": run_dir.resolve(),
                "run_name": run_dir.name,
                "policy_path": policy_path.resolve(),
                "config_path": config_path.resolve(),
                "config": load_json(config_path),
            }
        )
    return entries


def validate_run_compatibility(run_entries: list[dict[str, Any]]) -> dict[str, Any]:
    first_cfg = run_entries[0]["config"]
    env_cfg = first_cfg["environment"]
    ref_world_size = tuple(float(x) for x in env_cfg["world_size_m"])
    ref_voxel_size = float(env_cfg["voxel_size_m"])

    for entry in run_entries[1:]:
        cfg = entry["config"]
        world_size = tuple(float(x) for x in cfg["environment"]["world_size_m"])
        voxel_size = float(cfg["environment"]["voxel_size_m"])
        if world_size != ref_world_size or voxel_size != ref_voxel_size:
            raise ValueError(
                "The selected ours runs do not share the same environment.world_size_m / voxel_size_m."
            )
    return first_cfg


TILE_SIZE_XY = 4
EDGE_N, EDGE_E, EDGE_S, EDGE_W = 0, 1, 2, 3
DIR_SPECS = [
    (0, 1, EDGE_N, EDGE_S),
    (1, 0, EDGE_E, EDGE_W),
    (0, -1, EDGE_S, EDGE_N),
    (-1, 0, EDGE_W, EDGE_E),
]


def build_reserved_mask(shape: tuple[int, int, int]) -> np.ndarray:
    gx, gy, gz = shape
    reserved = np.zeros(shape, dtype=bool)
    z_min = min(max(2, int(round(gz * 0.18))), gz - 3)
    z_max = max(z_min + 1, min(gz - 2, int(round(gz * 0.82))))

    plazas = [
        (gx // 2, gy // 2, 3, 3),
        (gx // 4, gy // 4, 2, 2),
        (gx // 4, (3 * gy) // 4, 2, 2),
        ((3 * gx) // 4, gy // 4, 2, 2),
        ((3 * gx) // 4, (3 * gy) // 4, 2, 2),
    ]
    for cx, cy, rx, ry in plazas:
        reserved[
            max(cx - rx, 0) : min(cx + rx + 1, gx),
            max(cy - ry, 0) : min(cy + ry + 1, gy),
            z_min:z_max,
        ] = True
    return reserved


def build_tile_library(target_density: float) -> list[dict[str, Any]]:
    dense = float(np.clip((target_density - 0.08) / max(0.24 - 0.08, 1e-6), 0.0, 1.0))
    room_weight = 4.0 - 2.4 * dense
    pillar_weight = 0.35 + 1.6 * dense
    corridor_weight = 1.1 + 0.4 * dense
    corner_weight = 0.9 + 0.3 * dense
    tee_weight = 0.45 + 0.8 * dense
    block_weight = 0.08 + 1.25 * dense

    return [
        {"name": "room", "edges": (1, 1, 1, 1), "feature": "none", "weight": room_weight},
        {"name": "pillar", "edges": (1, 1, 1, 1), "feature": "pillar", "weight": pillar_weight},
        {"name": "corridor_ns", "edges": (1, 0, 1, 0), "feature": "none", "weight": corridor_weight},
        {"name": "corridor_ew", "edges": (0, 1, 0, 1), "feature": "none", "weight": corridor_weight},
        {"name": "corner_ne", "edges": (1, 1, 0, 0), "feature": "none", "weight": corner_weight},
        {"name": "corner_es", "edges": (0, 1, 1, 0), "feature": "none", "weight": corner_weight},
        {"name": "corner_sw", "edges": (0, 0, 1, 1), "feature": "none", "weight": corner_weight},
        {"name": "corner_wn", "edges": (1, 0, 0, 1), "feature": "none", "weight": corner_weight},
        {"name": "tee_n", "edges": (1, 1, 1, 0), "feature": "none", "weight": tee_weight},
        {"name": "tee_e", "edges": (0, 1, 1, 1), "feature": "none", "weight": tee_weight},
        {"name": "tee_s", "edges": (1, 0, 1, 1), "feature": "none", "weight": tee_weight},
        {"name": "tee_w", "edges": (1, 1, 0, 1), "feature": "none", "weight": tee_weight},
        {"name": "block", "edges": (0, 0, 0, 0), "feature": "solid", "weight": block_weight},
    ]


def reserved_macro_mask(reserved_mask: np.ndarray) -> np.ndarray:
    gx, gy, gz = reserved_mask.shape
    if gx % TILE_SIZE_XY != 0 or gy % TILE_SIZE_XY != 0:
        raise ValueError("Grid shape must be divisible by TILE_SIZE_XY for WFC generation.")
    mx = gx // TILE_SIZE_XY
    my = gy // TILE_SIZE_XY
    reshaped = reserved_mask.reshape(mx, TILE_SIZE_XY, my, TILE_SIZE_XY, gz)
    return np.any(reshaped, axis=(1, 3, 4))


def weighted_tile_choice(rng: np.random.Generator, options: list[int], tiles: list[dict[str, Any]]) -> int:
    weights = np.array([max(float(tiles[idx]["weight"]), 1e-6) for idx in options], dtype=np.float64)
    probs = weights / np.sum(weights)
    picked = int(rng.choice(np.array(options, dtype=np.int64), p=probs))
    return picked


def build_initial_domains(mx: int, my: int, tiles: list[dict[str, Any]], open_hint: np.ndarray) -> list[list[set[int]]]:
    all_ids = set(range(len(tiles)))
    domains: list[list[set[int]]] = [[set(all_ids) for _ in range(my)] for _ in range(mx)]
    open_allowed = {idx for idx, tile in enumerate(tiles) if tile["name"] != "block"}
    room_like = {idx for idx, tile in enumerate(tiles) if tile["name"] in {"room", "pillar"}}

    center_x = mx // 2
    center_y = my // 2
    for x in range(mx):
        for y in range(my):
            domain = set(domains[x][y])
            if x == 0:
                domain = {idx for idx in domain if tiles[idx]["edges"][EDGE_W] == 0}
            if x == mx - 1:
                domain = {idx for idx in domain if tiles[idx]["edges"][EDGE_E] == 0}
            if y == 0:
                domain = {idx for idx in domain if tiles[idx]["edges"][EDGE_S] == 0}
            if y == my - 1:
                domain = {idx for idx in domain if tiles[idx]["edges"][EDGE_N] == 0}
            if open_hint[x, y]:
                domain &= open_allowed
            if abs(x - center_x) <= 1 and abs(y - center_y) <= 1:
                domain &= room_like
            if not domain:
                raise RuntimeError("Initial WFC domain became empty.")
            domains[x][y] = domain
    return domains


def propagate_domains(domains: list[list[set[int]]], tiles: list[dict[str, Any]], start_cells: list[tuple[int, int]]) -> bool:
    mx = len(domains)
    my = len(domains[0])
    queue = list(start_cells)
    while queue:
        x, y = queue.pop()
        source = domains[x][y]
        if not source:
            return False
        for dx, dy, edge_src, edge_dst in DIR_SPECS:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= mx or ny < 0 or ny >= my:
                continue
            neighbor = domains[nx][ny]
            allowed = {
                n_idx
                for n_idx in neighbor
                if any(tiles[s_idx]["edges"][edge_src] == tiles[n_idx]["edges"][edge_dst] for s_idx in source)
            }
            if not allowed:
                return False
            if allowed != neighbor:
                domains[nx][ny] = allowed
                queue.append((nx, ny))
    return True


def collapse_wfc_layout(
    rng: np.random.Generator,
    tiles: list[dict[str, Any]],
    macro_shape: tuple[int, int],
    open_hint: np.ndarray,
    max_restarts: int = 80,
) -> np.ndarray:
    mx, my = macro_shape
    for _ in range(max_restarts):
        domains = build_initial_domains(mx, my, tiles, open_hint)
        if not propagate_domains(domains, tiles, [(x, y) for x in range(mx) for y in range(my)]):
            continue

        while True:
            unresolved: list[tuple[int, int]] = [
                (x, y) for x in range(mx) for y in range(my) if len(domains[x][y]) > 1
            ]
            if not unresolved:
                grid = np.zeros((mx, my), dtype=np.int32)
                for x in range(mx):
                    for y in range(my):
                        grid[x, y] = int(next(iter(domains[x][y])))
                return grid

            entropies = np.array([len(domains[x][y]) for x, y in unresolved], dtype=np.int32)
            min_entropy = int(np.min(entropies))
            candidates = [cell for cell, entropy in zip(unresolved, entropies) if int(entropy) == min_entropy]
            x, y = candidates[int(rng.integers(0, len(candidates)))]
            chosen = weighted_tile_choice(rng, sorted(domains[x][y]), tiles)
            domains[x][y] = {chosen}
            if not propagate_domains(domains, tiles, [(x, y)]):
                break

    raise RuntimeError("WFC failed to produce a valid layout after repeated restarts.")


def macro_connectivity_ratio(tile_grid: np.ndarray, tiles: list[dict[str, Any]]) -> float:
    mx, my = tile_grid.shape
    open_cells = [(x, y) for x in range(mx) for y in range(my) if tiles[int(tile_grid[x, y])]["name"] != "block"]
    if not open_cells:
        return 0.0
    center = min(open_cells, key=lambda item: abs(item[0] - mx // 2) + abs(item[1] - my // 2))
    seen = {center}
    stack = [center]
    while stack:
        x, y = stack.pop()
        tile = tiles[int(tile_grid[x, y])]
        for dx, dy, edge_src, edge_dst in DIR_SPECS:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= mx or ny < 0 or ny >= my:
                continue
            neighbor = tiles[int(tile_grid[nx, ny])]
            if neighbor["name"] == "block":
                continue
            if tile["edges"][edge_src] != 1 or neighbor["edges"][edge_dst] != 1:
                continue
            cell = (nx, ny)
            if cell not in seen:
                seen.add(cell)
                stack.append(cell)
    return float(len(seen)) / float(len(open_cells))


def tile_footprint(tile: dict[str, Any]) -> np.ndarray:
    grid = np.zeros((TILE_SIZE_XY, TILE_SIZE_XY), dtype=np.uint8)
    if tile["name"] == "block":
        grid[:, :] = 2
        return grid

    north, east, south, west = tile["edges"]
    grid[0, :] = 1 if west == 0 else 0
    grid[-1, :] = 1 if east == 0 else 0
    grid[:, 0] = np.maximum(grid[:, 0], 1 if south == 0 else 0)
    grid[:, -1] = np.maximum(grid[:, -1], 1 if north == 0 else 0)

    grid[0, 0] = 1
    grid[0, -1] = 1
    grid[-1, 0] = 1
    grid[-1, -1] = 1

    if west == 1:
        grid[0, 1:-1] = 0
    if east == 1:
        grid[-1, 1:-1] = 0
    if south == 1:
        grid[1:-1, 0] = 0
    if north == 1:
        grid[1:-1, -1] = 0

    if tile["feature"] == "pillar":
        grid[1:-1, 1:-1] = 2
    return grid


def render_wfc_occupancy(
    tile_grid: np.ndarray,
    tiles: list[dict[str, Any]],
    target_density: float,
    generation_seed: int,
    shape: tuple[int, int, int],
    reserved_mask: np.ndarray,
) -> np.ndarray:
    gx, gy, gz = shape
    occupancy = np.zeros(shape, dtype=np.int8)
    rng = np.random.default_rng(generation_seed + 404)
    dense = float(np.clip((target_density - 0.08) / max(0.24 - 0.08, 1e-6), 0.0, 1.0))
    wall_base = int(round(np.interp(dense, [0.0, 1.0], [6.0, 12.0])))
    feature_base = int(round(np.interp(dense, [0.0, 1.0], [8.0, float(gz)])))
    hang_prob = float(np.interp(dense, [0.0, 1.0], [0.03, 0.22]))
    hang_depth = int(round(np.interp(dense, [0.0, 1.0], [2.0, 5.0])))

    for mx in range(tile_grid.shape[0]):
        for my in range(tile_grid.shape[1]):
            tile = tiles[int(tile_grid[mx, my])]
            footprint = tile_footprint(tile)
            xs = mx * TILE_SIZE_XY
            xe = xs + TILE_SIZE_XY
            ys = my * TILE_SIZE_XY
            ye = ys + TILE_SIZE_XY

            for lx in range(TILE_SIZE_XY):
                for ly in range(TILE_SIZE_XY):
                    cell_type = int(footprint[lx, ly])
                    if cell_type <= 0:
                        continue
                    x = xs + lx
                    y = ys + ly
                    if cell_type == 1:
                        height = int(np.clip(wall_base + rng.integers(-1, 2), 4, gz))
                    else:
                        height = int(np.clip(feature_base + rng.integers(-2, 3), 6, gz))
                    occupancy[x, y, :height] = 1

            if tile["name"] != "block" and rng.random() < hang_prob:
                hang_x0 = xs + 1
                hang_x1 = min(xs + TILE_SIZE_XY - 1, gx)
                hang_y0 = ys + 1
                hang_y1 = min(ys + TILE_SIZE_XY - 1, gy)
                if hang_x1 > hang_x0 and hang_y1 > hang_y0:
                    z0 = max(gz - hang_depth, 0)
                    occupancy[hang_x0:hang_x1, hang_y0:hang_y1, z0:gz] = 1

    occupancy[reserved_mask] = 0
    return occupancy


def tune_occupancy_density(
    occupancy: np.ndarray,
    target_density: float,
    generation_seed: int,
) -> np.ndarray:
    tuned = occupancy.copy().astype(np.int8)
    total_cells = int(tuned.size)
    target_cells = int(round(float(target_density) * total_cells))
    current_cells = int(np.count_nonzero(tuned))
    if current_cells == target_cells:
        return tuned

    rng = np.random.default_rng(generation_seed + 909)
    heights = np.sum(tuned > 0, axis=2).astype(np.int32)
    gz = tuned.shape[2]
    min_keep_height = int(np.clip(round(np.interp(target_density, [0.08, 0.24], [2.0, 4.0])), 1, max(gz - 1, 1)))

    if current_cells > target_cells:
        excess = current_cells - target_cells
        candidates = np.argwhere(heights > min_keep_height)
        if len(candidates) > 0:
            rng.shuffle(candidates)
            for idx, (x, y) in enumerate(candidates):
                if excess <= 0:
                    break
                col_height = int(heights[x, y])
                removable = max(col_height - min_keep_height, 0)
                if removable <= 0:
                    continue
                remaining = max(len(candidates) - idx, 1)
                cut = min(removable, max(1, int(math.ceil(excess / remaining))))
                tuned[x, y, col_height - cut : col_height] = 0
                heights[x, y] -= cut
                excess -= cut

        if excess > 0:
            candidates = np.argwhere(heights > 0)
            if len(candidates) > 0:
                rng.shuffle(candidates)
                for idx, (x, y) in enumerate(candidates):
                    if excess <= 0:
                        break
                    col_height = int(heights[x, y])
                    if col_height <= 0:
                        continue
                    remaining = max(len(candidates) - idx, 1)
                    cut = min(col_height, max(1, int(math.ceil(excess / remaining))))
                    tuned[x, y, col_height - cut : col_height] = 0
                    heights[x, y] -= cut
                    excess -= cut
    else:
        shortage = target_cells - current_cells
        candidates = np.argwhere((heights > 0) & (heights < gz))
        if len(candidates) > 0:
            rng.shuffle(candidates)
            for idx, (x, y) in enumerate(candidates):
                if shortage <= 0:
                    break
                col_height = int(heights[x, y])
                extendable = max(gz - col_height, 0)
                if extendable <= 0:
                    continue
                remaining = max(len(candidates) - idx, 1)
                add = min(extendable, max(1, int(math.ceil(shortage / remaining))))
                tuned[x, y, col_height : col_height + add] = 1
                heights[x, y] += add
                shortage -= add

    return tuned


def generate_random_map(
    *,
    shape: tuple[int, int, int],
    target_density: float,
    suite_seed: int,
    map_index: int,
    reserved_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_density = float(np.clip(target_density, 0.01, 0.40))
    gx, gy, _ = shape
    if gx % TILE_SIZE_XY != 0 or gy % TILE_SIZE_XY != 0:
        raise ValueError(f"Shape {shape} is incompatible with TILE_SIZE_XY={TILE_SIZE_XY}")

    mx = gx // TILE_SIZE_XY
    my = gy // TILE_SIZE_XY
    open_hint = reserved_macro_mask(reserved_mask)

    best_payload: tuple[np.ndarray, dict[str, Any]] | None = None
    best_error = float("inf")
    target_connectivity = float(np.interp(target_density, [0.08, 0.24], [0.80, 0.62]))

    for attempt in range(24):
        generation_seed = int(suite_seed + map_index * 9973 + attempt * 131)
        rng = np.random.default_rng(generation_seed)
        tiles = build_tile_library(target_density)
        tile_grid = collapse_wfc_layout(rng, tiles, (mx, my), open_hint)
        connectivity = macro_connectivity_ratio(tile_grid, tiles)
        occupancy = render_wfc_occupancy(
            tile_grid=tile_grid,
            tiles=tiles,
            target_density=target_density,
            generation_seed=generation_seed,
            shape=shape,
            reserved_mask=reserved_mask,
        )
        occupancy = tune_occupancy_density(
            occupancy=occupancy,
            target_density=target_density,
            generation_seed=generation_seed,
        )

        actual_density = float(np.count_nonzero(occupancy)) / float(occupancy.size)
        free_ratio = 1.0 - actual_density
        error = abs(actual_density - target_density) + max(target_connectivity - connectivity, 0.0) * 0.5
        metadata = {
            "generation_seed": generation_seed,
            "target_density": float(target_density),
            "actual_density": float(actual_density),
            "occupied_voxels": int(np.count_nonzero(occupancy)),
            "free_voxels": int(np.count_nonzero(occupancy == 0)),
            "grid_shape": [int(v) for v in shape],
            "generator": "wfc_2p5d",
            "tile_size_xy": TILE_SIZE_XY,
            "macro_grid_shape": [int(mx), int(my)],
            "macro_connectivity_ratio": float(connectivity),
            "attempt_index": int(attempt),
        }

        if free_ratio >= 0.45 and connectivity >= target_connectivity and abs(actual_density - target_density) <= 0.03:
            return occupancy, metadata
        if error < best_error:
            best_error = error
            best_payload = (occupancy, metadata)

    if best_payload is None:
        raise RuntimeError("WFC map generation failed to produce any valid candidate.")
    return best_payload


def write_map_artifacts(
    *,
    maps_dir: Path,
    map_id: str,
    difficulty: str,
    occupancy: np.ndarray,
    reference_meta: dict[str, Any],
    generation_meta: dict[str, Any],
) -> dict[str, Any]:
    map_path = maps_dir / f"{map_id}.npy"
    meta_path = maps_dir / f"{map_id}.json"
    np.save(map_path, occupancy.astype(np.int8))

    meta_payload = {
        "map_id": map_id,
        "difficulty": difficulty,
        "origin": reference_meta["origin"],
        "world_size": reference_meta["world_size"],
        "voxel_size": reference_meta["voxel_size"],
        "grid_shape": generation_meta["grid_shape"],
        "target_density": generation_meta["target_density"],
        "actual_density": generation_meta["actual_density"],
        "occupied_voxels": generation_meta["occupied_voxels"],
        "free_voxels": generation_meta["free_voxels"],
        "generation_seed": generation_meta["generation_seed"],
        "generator": generation_meta.get("generator", "unknown"),
        "tile_size_xy": generation_meta.get("tile_size_xy"),
        "macro_grid_shape": generation_meta.get("macro_grid_shape"),
        "macro_connectivity_ratio": generation_meta.get("macro_connectivity_ratio"),
        "attempt_index": generation_meta.get("attempt_index"),
        "source_map": str(DEFAULT_SOURCE_MAP.resolve()),
    }
    save_json(meta_path, meta_payload)
    return meta_payload


def create_eval_config(
    *,
    run_entry: dict[str, Any],
    map_path: Path,
    config_out: Path,
) -> Path:
    cfg = deepcopy(run_entry["config"])
    cfg.setdefault("navigation", {})
    cfg["navigation"]["frontier_mask_path"] = str(map_path.resolve())
    save_json(config_out, cfg)
    return config_out


def build_eval_command(
    *,
    python_exe: str,
    policy_path: Path,
    config_path: Path,
    output_path: Path,
    stage_index: int,
    episodes: int,
    num_envs: int,
    device: str,
    eval_seed: int,
    eval_tag: str,
    deterministic: bool,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(DEFAULT_EVAL_SCRIPT.resolve()),
        "--policy",
        str(policy_path.resolve()),
        "--config",
        str(config_path.resolve()),
        "--episodes",
        str(int(episodes)),
        "--stage-index",
        str(int(stage_index)),
        "--num-envs",
        str(int(num_envs)),
        "--device",
        str(device),
        "--seed",
        str(int(eval_seed)),
        "--eval-tag",
        str(eval_tag),
        "--output",
        str(output_path.resolve()),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return cmd


def summarize_eval_jsonl(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Evaluation output is empty: {path}")

    total = len(rows)
    reward_total = [float(row.get("reward_sums", {}).get("reward_total", 0.0)) for row in rows]
    return {
        "episodes": int(total),
        "success_rate": float(sum(bool(row.get("success", False)) for row in rows)) / total,
        "capture_rate": float(sum(bool(row.get("captured_raw", False)) for row in rows)) / total,
        "clean_capture_rate": float(sum(bool(row.get("clean_capture", False)) for row in rows)) / total,
        "encircle_capture_rate": float(sum(bool(row.get("encircle_capture", False)) for row in rows)) / total,
        "collision_rate": float(sum(bool(row.get("collision", False)) for row in rows)) / total,
        "timeout_rate": float(sum(bool(row.get("timeout", False)) for row in rows)) / total,
        "avg_steps": float(np.mean([float(row.get("steps", 0.0)) for row in rows])),
        "avg_min_dist": float(np.mean([float(row.get("min_dist", 0.0)) for row in rows])),
        "avg_visible_ratio": float(np.mean([float(row.get("visible_ratio", 0.0)) for row in rows])),
        "avg_far_idle_ratio": float(np.mean([float(row.get("far_idle_ratio", 0.0)) for row in rows])),
        "avg_reward_total": float(np.mean(reward_total)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary_rows.append(
            {
                key: group_key,
                "combos": len(group_rows),
                "episodes": int(sum(int(item["episodes"]) for item in group_rows)),
                "success_rate_mean": float(np.mean([float(item["success_rate"]) for item in group_rows])),
                "collision_rate_mean": float(np.mean([float(item["collision_rate"]) for item in group_rows])),
                "timeout_rate_mean": float(np.mean([float(item["timeout_rate"]) for item in group_rows])),
                "avg_steps_mean": float(np.mean([float(item["avg_steps"]) for item in group_rows])),
                "avg_reward_total_mean": float(np.mean([float(item["avg_reward_total"]) for item in group_rows])),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    densities = [float(value) for value in args.densities]
    difficulties = list(args.difficulties)
    ensure_equal_lengths(densities, difficulties)

    output_root = resolve_output_root(args.output_root)
    maps_dir = output_root / "maps"
    configs_dir = output_root / "configs"
    results_dir = output_root / "results"
    summaries_dir = output_root / "summaries"
    for path in (maps_dir, configs_dir, results_dir, summaries_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_entries = discover_ours_runs(args.runs_root.resolve(), [int(seed) for seed in args.seeds])
    base_cfg = validate_run_compatibility(run_entries)
    ref_occ, ref_meta = load_reference_map(args.source_map.resolve())
    if tuple(int(v) for v in ref_occ.shape) != tuple(int(v) for v in ref_meta["grid_shape"]):
        raise ValueError("Reference occupancy grid shape does not match its metadata.")

    env_voxel_size = float(base_cfg["environment"]["voxel_size_m"])
    ref_voxel_size = float(ref_meta["voxel_size"])
    env_grid_shape = tuple(
        int(round(float(v) / max(env_voxel_size, 1e-9))) for v in base_cfg["environment"]["world_size_m"]
    )
    ref_grid_shape = tuple(int(v) for v in ref_meta["grid_shape"])
    if env_grid_shape != ref_grid_shape or abs(env_voxel_size - ref_voxel_size) > 1e-9:
        raise ValueError(
            "Reference map metadata does not match the ours run environment grid shape / voxel size."
        )

    reserved_mask = build_reserved_mask(tuple(int(v) for v in ref_occ.shape))
    map_manifests: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for index, (difficulty, density) in enumerate(zip(difficulties, densities), start=1):
        map_id = f"map{index:02d}_{difficulty}"
        occupancy, generation_meta = generate_random_map(
            shape=tuple(int(v) for v in ref_occ.shape),
            target_density=float(density),
            suite_seed=int(args.suite_seed),
            map_index=index,
            reserved_mask=reserved_mask,
        )
        map_meta = write_map_artifacts(
            maps_dir=maps_dir,
            map_id=map_id,
            difficulty=difficulty,
            occupancy=occupancy,
            reference_meta=ref_meta,
            generation_meta=generation_meta,
        )
        map_manifests.append(map_meta)

        for run_entry in run_entries:
            model_seed = int(run_entry["seed"])
            eval_seed = int(args.suite_seed + index * 100 + model_seed)
            eval_tag = f"generalization_{map_id}_s{model_seed}"
            config_out = configs_dir / map_id / f"{run_entry['run_name']}.json"
            output_jsonl = results_dir / map_id / f"s{model_seed}" / "episode_eval.jsonl"
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            create_eval_config(run_entry=run_entry, map_path=maps_dir / f"{map_id}.npy", config_out=config_out)

            cmd = build_eval_command(
                python_exe=args.python,
                policy_path=Path(run_entry["policy_path"]),
                config_path=config_out,
                output_path=output_jsonl,
                stage_index=int(args.stage_index),
                episodes=int(args.episodes),
                num_envs=int(args.num_envs),
                device=str(args.device),
                eval_seed=eval_seed,
                eval_tag=eval_tag,
                deterministic=bool(args.deterministic),
            )
            commands.append(
                {
                    "map_id": map_id,
                    "model_seed": model_seed,
                    "command": cmd,
                }
            )

            if args.dry_run:
                continue

            completed = subprocess.run(cmd, check=False, cwd=str(ROOT))
            if completed.returncode != 0:
                if args.stop_on_error:
                    raise SystemExit(completed.returncode)
                combo_rows.append(
                    {
                        "model_seed": model_seed,
                        "run_name": run_entry["run_name"],
                        "map_id": map_id,
                        "difficulty": difficulty,
                        "target_density": map_meta["target_density"],
                        "actual_density": map_meta["actual_density"],
                        "eval_seed": eval_seed,
                        "episodes": 0,
                        "success_rate": np.nan,
                        "capture_rate": np.nan,
                        "clean_capture_rate": np.nan,
                        "encircle_capture_rate": np.nan,
                        "collision_rate": np.nan,
                        "timeout_rate": np.nan,
                        "avg_steps": np.nan,
                        "avg_min_dist": np.nan,
                        "avg_visible_ratio": np.nan,
                        "avg_far_idle_ratio": np.nan,
                        "avg_reward_total": np.nan,
                        "output_jsonl": str(output_jsonl.resolve()),
                        "map_path": str((maps_dir / f"{map_id}.npy").resolve()),
                        "config_path": str(config_out.resolve()),
                    }
                )
                continue

            eval_summary = summarize_eval_jsonl(output_jsonl)
            combo_rows.append(
                {
                    "model_seed": model_seed,
                    "run_name": run_entry["run_name"],
                    "map_id": map_id,
                    "difficulty": difficulty,
                    "target_density": map_meta["target_density"],
                    "actual_density": map_meta["actual_density"],
                    "eval_seed": eval_seed,
                    "episodes": int(eval_summary["episodes"]),
                    "success_rate": float(eval_summary["success_rate"]),
                    "capture_rate": float(eval_summary["capture_rate"]),
                    "clean_capture_rate": float(eval_summary["clean_capture_rate"]),
                    "encircle_capture_rate": float(eval_summary["encircle_capture_rate"]),
                    "collision_rate": float(eval_summary["collision_rate"]),
                    "timeout_rate": float(eval_summary["timeout_rate"]),
                    "avg_steps": float(eval_summary["avg_steps"]),
                    "avg_min_dist": float(eval_summary["avg_min_dist"]),
                    "avg_visible_ratio": float(eval_summary["avg_visible_ratio"]),
                    "avg_far_idle_ratio": float(eval_summary["avg_far_idle_ratio"]),
                    "avg_reward_total": float(eval_summary["avg_reward_total"]),
                    "output_jsonl": str(output_jsonl.resolve()),
                    "map_path": str((maps_dir / f"{map_id}.npy").resolve()),
                    "config_path": str(config_out.resolve()),
                }
            )

    manifest = {
        "suite_root": str(output_root),
        "runs_root": str(args.runs_root.resolve()),
        "source_map": str(args.source_map.resolve()),
        "stage_index": int(args.stage_index),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "device": str(args.device),
        "deterministic": bool(args.deterministic),
        "suite_seed": int(args.suite_seed),
        "model_seeds": [int(seed) for seed in args.seeds],
        "ours_runs": [
            {
                "seed": int(entry["seed"]),
                "run_name": entry["run_name"],
                "policy_path": str(Path(entry["policy_path"]).resolve()),
                "config_path": str(Path(entry["config_path"]).resolve()),
            }
            for entry in run_entries
        ],
        "maps": map_manifests,
        "commands": commands,
        "dry_run": bool(args.dry_run),
    }
    save_json(output_root / "suite_manifest.json", manifest)

    save_json(output_root / "summaries" / "combo_summary.json", {"rows": combo_rows})
    write_csv(output_root / "summaries" / "combo_summary.csv", combo_rows, EVAL_SUMMARY_FIELDS)

    if combo_rows and not args.dry_run:
        by_map = aggregate_by_key(combo_rows, "map_id")
        by_seed = aggregate_by_key(combo_rows, "model_seed")
        save_json(output_root / "summaries" / "summary_by_map.json", {"rows": by_map})
        save_json(output_root / "summaries" / "summary_by_seed.json", {"rows": by_seed})
        write_csv(
            output_root / "summaries" / "summary_by_map.csv",
            by_map,
            ["map_id", "combos", "episodes", "success_rate_mean", "collision_rate_mean", "timeout_rate_mean", "avg_steps_mean", "avg_reward_total_mean"],
        )
        write_csv(
            output_root / "summaries" / "summary_by_seed.csv",
            by_seed,
            ["model_seed", "combos", "episodes", "success_rate_mean", "collision_rate_mean", "timeout_rate_mean", "avg_steps_mean", "avg_reward_total_mean"],
        )

    print(f"[Generalization] suite_root={output_root}")
    print(f"[Generalization] maps={len(map_manifests)} model_seeds={len(run_entries)} episodes={int(args.episodes)}")
    if args.dry_run:
        print("[Generalization] dry-run mode: maps/configs/manifest were written, evaluations were skipped.")
    else:
        print(f"[Generalization] combo_summary={output_root / 'summaries' / 'combo_summary.csv'}")


if __name__ == "__main__":
    main()
