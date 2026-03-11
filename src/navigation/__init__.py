from .astar3d import AStar3D
from .frontier_allocation import FrontierAllocator
from .voxel_map import VoxelMap3D
from .coordinates import (
    CoordinateFrame,
    world_to_grid,
    grid_to_world,
    world_to_body,
    body_to_world,
    normalize_angle,
    angle_difference,
    heading_from_velocity,
    distance_3d,
    distance_2d,
    interpolate_path,
)

# GPU-accelerated path guidance (v2)
try:
    from .gpu_path_guider_v2 import (
        FastBatchPathGuider,
        optimistic_downsample_3d,
        world_to_voxel_indices,
        CUDA_AVAILABLE as GPU_PATH_GUIDER_AVAILABLE,
    )
    from .gpu_bfs_planner import GPUPathField
except ImportError:  # pragma: no cover - optional dependency
    FastBatchPathGuider = None
    optimistic_downsample_3d = None
    world_to_voxel_indices = None
    GPUPathField = None
    GPU_PATH_GUIDER_AVAILABLE = False

__all__ = [
    "AStar3D",
    "FrontierAllocator",
    "VoxelMap3D",
    "CoordinateFrame",
    "world_to_grid",
    "grid_to_world",
    "world_to_body",
    "body_to_world",
    "normalize_angle",
    "angle_difference",
    "heading_from_velocity",
    "distance_3d",
    "distance_2d",
    "interpolate_path",
    # GPU Path Guider v2
    "FastBatchPathGuider",
    "optimistic_downsample_3d",
    "world_to_voxel_indices",
    "GPUPathField",
    "GPU_PATH_GUIDER_AVAILABLE",
]
