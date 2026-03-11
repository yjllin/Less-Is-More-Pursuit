"""Render random occupancy maps with PyVista."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None


ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
DEFAULT_SUITE_ROOT = ROOT / "runs" / "mappo_3d" / "generalization_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random occupancy maps using PyVista.")
    parser.add_argument(
        "--suite-root",
        type=str,
        default=None,
        help="Generalization suite root. Default: latest suite under runs/mappo_3d/generalization_suite.",
    )
    parser.add_argument(
        "--maps-dir",
        type=str,
        default=None,
        help="Direct maps directory override. Default: <suite-root>/maps.",
    )
    parser.add_argument(
        "--npy",
        type=str,
        default=None,
        help="Single occupancy map .npy path. If set, ignores --suite-root/--maps-dir map discovery.",
    )
    parser.add_argument(
        "--map-id",
        type=str,
        default=None,
        help="Single map id under maps dir, e.g. map03_medium.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=6,
        help="Maximum number of maps to render in multi-map mode.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Subsample occupied voxels for point overlay or mesh density reduction.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="voxels",
        choices=["voxels", "points"],
        help="Render occupied structure as voxel cells or as voxel-center points.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.70,
        help="Actor opacity for occupied voxels.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Point size when --style points.",
    )
    parser.add_argument(
        "--show-bounds",
        action="store_true",
        help="Show world bounds for each subplot.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show axes widget.",
    )
    parser.add_argument(
        "--link-views",
        action="store_true",
        help="Link camera across subplots in multi-map mode.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(1600, 900),
        metavar=("W", "H"),
        help="Render window size.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Use off-screen rendering. Useful with --screenshot.",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Optional screenshot output path (.png recommended).",
    )
    return parser.parse_args()


def resolve_latest_suite(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value).resolve()
    if not DEFAULT_SUITE_ROOT.exists():
        raise FileNotFoundError(f"Suite root not found: {DEFAULT_SUITE_ROOT}")
    candidates = [path for path in DEFAULT_SUITE_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No suite directories found under {DEFAULT_SUITE_ROOT}")
    return sorted(candidates, key=lambda item: item.name)[-1].resolve()


def load_map_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def discover_maps(args: argparse.Namespace) -> list[Path]:
    if args.npy:
        path = Path(args.npy).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Map not found: {path}")
        return [path]

    maps_dir = Path(args.maps_dir).resolve() if args.maps_dir else resolve_latest_suite(args.suite_root) / "maps"
    if not maps_dir.exists():
        raise FileNotFoundError(f"Maps dir not found: {maps_dir}")

    if args.map_id:
        path = maps_dir / f"{args.map_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Map id not found under {maps_dir}: {args.map_id}")
        return [path.resolve()]

    paths = sorted(maps_dir.glob("*.npy"), key=lambda item: item.name)
    if not paths:
        raise FileNotFoundError(f"No .npy maps found in {maps_dir}")
    return [path.resolve() for path in paths[: max(1, int(args.max_maps))]]


def occupied_points(occupancy: np.ndarray, origin: np.ndarray, voxel_size: float, downsample: int) -> np.ndarray:
    indices = np.argwhere(occupancy > 0)
    if downsample > 1 and len(indices) > 0:
        indices = indices[:: int(downsample)]
    return (indices.astype(np.float64) + 0.5) * float(voxel_size) + origin[None, :]


def build_voxel_mesh(occupancy: np.ndarray, origin: np.ndarray, voxel_size: float) -> "pv.DataSet":
    grid = pv.ImageData()
    shape = np.array(occupancy.shape, dtype=int)
    grid.dimensions = tuple((shape + 1).tolist())
    grid.origin = tuple(origin.tolist())
    grid.spacing = (float(voxel_size), float(voxel_size), float(voxel_size))
    grid.cell_data["occupancy"] = occupancy.astype(np.uint8).flatten(order="F")
    voxels = grid.threshold(0.5, scalars="occupancy", preference="cell")
    return voxels


def title_for_map(map_path: Path, meta: dict[str, Any], occupancy: np.ndarray) -> str:
    density = float(meta.get("actual_density", float((occupancy > 0).mean())))
    difficulty = str(meta.get("difficulty", map_path.stem))
    return f"{difficulty}\n{map_path.stem} | density={density:.3f}"


def subplot_shape(count: int) -> tuple[int, int]:
    cols = min(3, max(1, int(math.ceil(math.sqrt(count)))))
    rows = int(math.ceil(count / cols))
    return rows, cols


def add_map_actor(
    plotter: "pv.Plotter",
    occupancy: np.ndarray,
    meta: dict[str, Any],
    map_path: Path,
    args: argparse.Namespace,
) -> None:
    voxel_size = float(meta.get("voxel_size", 6.0))
    origin = np.array(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
    if args.style == "voxels":
        mesh = build_voxel_mesh(occupancy, origin, voxel_size)
        plotter.add_mesh(
            mesh,
            color="#D97706",
            opacity=float(args.opacity),
            show_edges=False,
            lighting=True,
            smooth_shading=False,
        )
    else:
        points = occupied_points(occupancy, origin, voxel_size, max(1, int(args.downsample)))
        cloud = pv.PolyData(points)
        plotter.add_mesh(
            cloud,
            color="#C2410C",
            opacity=float(args.opacity),
            point_size=float(args.point_size),
            render_points_as_spheres=True,
        )

    world_max = origin + np.array(occupancy.shape, dtype=np.float64) * voxel_size
    outline = pv.Box(bounds=(
        float(origin[0]), float(world_max[0]),
        float(origin[1]), float(world_max[1]),
        float(origin[2]), float(world_max[2]),
    ))
    plotter.add_mesh(outline, style="wireframe", color="#1F2937", line_width=1.0, opacity=0.4)
    plotter.add_text(title_for_map(map_path, meta, occupancy), font_size=10, position="upper_left")
    if args.show_bounds:
        plotter.show_bounds(grid="front", location="outer", ticks="outside", font_size=8)
    plotter.set_background("#F8FAFC")
    plotter.camera_position = "iso"


def main() -> None:
    if pv is None:
        raise SystemExit("pyvista package not installed. Install it with `pip install pyvista` first.")

    args = parse_args()
    map_paths = discover_maps(args)
    multi = len(map_paths) > 1
    rows, cols = subplot_shape(len(map_paths))
    plotter = pv.Plotter(
        shape=(rows, cols) if multi else None,
        off_screen=bool(args.off_screen),
        window_size=tuple(int(v) for v in args.window_size),
    )

    if args.show_axes:
        plotter.add_axes()

    for idx, map_path in enumerate(map_paths):
        occupancy = np.load(map_path).astype(np.uint8)
        meta = load_map_meta(map_path.with_suffix(".json"))
        if multi:
            r = idx // cols
            c = idx % cols
            plotter.subplot(r, c)
        add_map_actor(plotter, occupancy, meta, map_path, args)

    if multi and args.link_views:
        plotter.link_views()

    if args.screenshot:
        screenshot_path = Path(args.screenshot).resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=True)
        print(f"[PyVista] screenshot={screenshot_path}")
        return

    plotter.show(auto_close=True)


if __name__ == "__main__":
    main()
