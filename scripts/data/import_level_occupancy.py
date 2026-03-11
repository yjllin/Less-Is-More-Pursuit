"""Convert a UE level text export into a voxel occupancy grid."""

from __future__ import annotations
import sys
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    distance_transform_edt = None

from src.navigation import VoxelMap3D


def _parse_vec(line: str) -> np.ndarray:
    start = line.find("(")
    end = line.rfind(")")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failed to locate vector tuple in line: {line}")
    body = line[start + 1 : end]
    values: List[float] = []
    for part in body.split(","):
        if "=" not in part:
            continue
        _, raw = part.split("=", 1)
        try:
            values.append(float(raw))
        except ValueError as exc:  # pragma: no cover - malformed lines
            raise ValueError(f"Failed to parse number in line: {line}") from exc
    if len(values) != 3:
        raise ValueError(f"Failed to parse vector from line: {line}")
    return np.array(values, dtype=np.float64)


def _deg_to_rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


def _rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    pitch = _deg_to_rad(pitch_deg)
    yaw = _deg_to_rad(yaw_deg)
    roll = _deg_to_rad(roll_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


@dataclass
class StaticMeshActor:
    name: str
    mesh: str
    location_cm: np.ndarray
    scale: np.ndarray
    rotation_deg: np.ndarray

    def bbox_world(self) -> Tuple[np.ndarray, np.ndarray]:
        base_size_m = 1.0  # UE basic shapes span 100 units == 1 meter.
        center_m = self.location_cm * 0.01
        dims = self.scale * base_size_m
        half = 0.5 * dims
        rot = (
            _rotation_matrix(float(self.rotation_deg[0]), float(self.rotation_deg[1]), float(self.rotation_deg[2]))
            if np.any(self.rotation_deg)
            else np.eye(3)
        )
        corners: List[np.ndarray] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                    world = center_m + rot @ local
                    corners.append(world)
        stacked = np.vstack(corners)
        return np.min(stacked, axis=0), np.max(stacked, axis=0)


def _parse_level(path: Path) -> Tuple[List[StaticMeshActor], List[Tuple[np.ndarray, np.ndarray]]]:
    actors: List[StaticMeshActor] = []
    blocking_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    current: dict | None = None
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if line.startswith("Begin Actor"):
                current = {"name": line.split("Name=")[-1].split()[0]}
                if "Class=" in line:
                    part = line.split("Class=")[-1]
                    current["class"] = part.split()[0]
            elif line.startswith("End Actor"):
                if not current:
                    continue
                actor_class = current.get("class", "")
                if "StaticMeshActor" in actor_class and "mesh" in current and "scale" in current:
                    rotation = current.get("rotation", np.zeros(3, dtype=np.float64))
                    location = current.get("location", np.zeros(3, dtype=np.float64))
                    actors.append(
                        StaticMeshActor(
                            name=str(current.get("name", "Unknown")),
                            mesh=str(current["mesh"]),
                            location_cm=location,
                            scale=current["scale"],
                            rotation_deg=rotation,
                        )
                    )
                elif "BlockingVolume" in actor_class and "bbox" in current:
                    blocking_boxes.append(current["bbox"])
                current = None
            elif current is not None:
                if "StaticMesh=" in line:
                    current["mesh"] = line.split("StaticMesh=")[-1].strip().strip("'\"")
                elif line.startswith("RelativeLocation"):
                    current["location"] = _parse_vec(line)
                elif line.startswith("RelativeScale3D"):
                    current["scale"] = _parse_vec(line)
                elif line.startswith("RelativeRotation"):
                    current["rotation"] = _parse_vec(line)
                elif "ElemBox=" in line:
                    bbox = _parse_elem_box(line)
                    if bbox is not None:
                        current["bbox"] = bbox
    return actors, blocking_boxes


def _parse_elem_box(line: str) -> Tuple[np.ndarray, np.ndarray] | None:
    min_idx = line.find("Min=(")
    max_idx = line.find("Max=(")
    if min_idx == -1 or max_idx == -1:
        return None
    min_end = line.find(")", min_idx)
    max_end = line.find(")", max_idx)
    if min_end == -1 or max_end == -1:
        return None
    min_vec = _parse_vec(line[min_idx : min_end + 1])
    max_vec = _parse_vec(line[max_idx : max_end + 1])
    return min_vec * 0.01, max_vec * 0.01


def _compute_bounds(boxes: Sequence[Tuple[np.ndarray, np.ndarray]], margin: float) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.min([b[0] for b in boxes], axis=0)
    maxs = np.max([b[1] for b in boxes], axis=0)
    mins -= margin
    maxs += margin
    return mins, maxs


def _fill_voxels(
    voxel: VoxelMap3D,
    boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
    origin: np.ndarray,
    voxel_size: float,
) -> None:
    for box_min, box_max in boxes:
        local_min = box_min - origin
        local_max = box_max - origin
        idx_min = np.floor(local_min / voxel_size).astype(int)
        idx_max = np.ceil(local_max / voxel_size).astype(int)
        idx_min = np.clip(idx_min, 0, np.array(voxel.shape) - 1)
        idx_max = np.clip(idx_max, 0, np.array(voxel.shape))
        xs = slice(idx_min[0], idx_max[0])
        ys = slice(idx_min[1], idx_max[1])
        zs = slice(idx_min[2], idx_max[2])
        if xs.stop <= xs.start or ys.stop <= ys.start or zs.stop <= zs.start:
            continue
        voxel.hits[xs, ys, zs] = 1
        voxel.visits[xs, ys, zs] = 1


def _sphere_offsets(radius_cells: int, voxel_size: float, clearance: float) -> NDArray[np.int_]:
    coords: List[Tuple[int, int, int]] = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            for dz in range(-radius_cells, radius_cells + 1):
                dist = math.sqrt((dx * voxel_size) ** 2 + (dy * voxel_size) ** 2 + (dz * voxel_size) ** 2)
                if dist <= clearance:
                    coords.append((dx, dy, dz))
    return np.array(coords, dtype=int)


def _apply_clearance(grid: NDArray[np.int_], voxel_size: float, clearance: float) -> NDArray[np.int_]:
    if clearance <= 0.0:
        return grid
    expanded = grid.copy()
    occupied = expanded == 1
    occupied[0, :, :] = True
    occupied[-1, :, :] = True
    occupied[:, 0, :] = True
    occupied[:, -1, :] = True
    occupied[:, :, 0] = True
    occupied[:, :, -1] = True
    if distance_transform_edt is not None:
        free = np.logical_not(occupied)
        dist = distance_transform_edt(free, sampling=voxel_size)
        expanded[dist <= clearance] = 1
        return expanded
    radius_cells = int(math.ceil(clearance / voxel_size))
    if radius_cells <= 0:
        return expanded
    offsets = _sphere_offsets(radius_cells, voxel_size, clearance)
    max_idx = np.array(expanded.shape, dtype=int) - 1
    occ_idx = np.argwhere(occupied)
    for cell in occ_idx:
        base = cell.astype(int)
        for off in offsets:
            tgt = base + off
            if np.any(tgt < 0) or np.any(tgt > max_idx):
                continue
            expanded[tuple(tgt)] = 1
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UE level actors to a voxel occupancy grid.")
    parser.add_argument("--level", type=Path, default=Path("level_data.txt"), help="Path to the UE text export.")
    parser.add_argument("--voxel-size", type=float, default=6.0, help="Voxel resolution in meters.")
    parser.add_argument("--occupancy-threshold", type=float, default=0.3, help="Occupancy probability threshold.")
    parser.add_argument("--margin", type=float, default=4.0, help="Extra margin (meters) around the parsed geometry.")
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.0,
        help="Minimum clearance (meters) from geometry; cells within this distance are marked occupied.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="World-space origin for the voxel grid (default: 0 0 0).",
    )
    parser.add_argument(
        "--auto-origin",
        action="store_true",
        help="Auto-detect the origin from geometry bounds instead of using --origin.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "level_occupancy.npy", help="Output .npy file.")
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional JSON file to store origin/world_size metadata (defaults to <output>.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actors, extra_boxes = _parse_level(args.level)
    if not actors and not extra_boxes:
        raise SystemExit(f"No supported actors found in {args.level}")
    boxes = [actor.bbox_world() for actor in actors] + list(extra_boxes)
    if args.auto_origin:
        origin, world_max = _compute_bounds(boxes, args.margin)
    else:
        origin = np.array(args.origin, dtype=np.float64)
        mins = np.min([b[0] for b in boxes], axis=0) - args.margin
        if np.any(mins < origin):
            raise ValueError(
                f"Provided origin {origin} is outside geometry bounds; "
                f"minimum actor corner {mins} lies below the origin. "
                "Use --auto-origin or adjust your scene."
            )
        world_max = np.max([b[1] for b in boxes], axis=0) + args.margin
    world_size = world_max - origin
    voxel = VoxelMap3D(tuple(world_size), args.voxel_size, args.occupancy_threshold, origin=tuple(origin))
    _fill_voxels(voxel, boxes, origin, args.voxel_size)
    occupancy = voxel.occupancy_grid()
    if args.clearance > 0:
        occupancy = _apply_clearance(occupancy, args.voxel_size, args.clearance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    occupancy = occupancy.astype(np.int8)
    np.save(args.output, occupancy)
    print(f"Converted {len(actors)} actors into occupancy grid {occupancy.shape} saved to {args.output}")
    meta_path = args.meta_output if args.meta_output is not None else args.output.with_suffix(".json")
    meta = {
        "origin": origin.tolist(),
        "world_size": world_size.tolist(),
        "voxel_size": float(args.voxel_size),
        "grid_shape": list(occupancy.shape),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Wrote occupancy metadata to {meta_path}")


if __name__ == "__main__":
    main()
