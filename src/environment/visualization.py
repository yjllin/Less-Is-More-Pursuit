"""3D visualization for multi-drone environment using matplotlib."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Agent colors for visualization
AGENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
TARGET_COLOR = '#e41a1c'
PATH_ALPHA = 0.6
FRONTIER_COLOR = '#7570b3'


class VoxelMapVisualizer:
    """3D visualization for voxel maps, agents, and paths."""

    def __init__(
        self,
        world_size: tuple[float, float, float],
        voxel_size: float,
        origin: Optional[NDArray[np.float64]] = None,
        figsize: tuple[int, int] = (14, 6),
        pause_time: float = 0.01,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")
        
        self.world_size = world_size
        self.voxel_size = voxel_size
        self.origin = origin if origin is not None else np.zeros(3)
        self._pause_time = pause_time
        
        # Create figure with 3D and 2D subplots
        self._fig = plt.figure(figsize=figsize)
        self._ax_3d = self._fig.add_subplot(121, projection='3d')
        self._ax_2d = self._fig.add_subplot(122)
        
        self._setup_axes()
        plt.ion()
        self._fig.canvas.draw()

    def _setup_axes(self) -> None:
        """Configure axis limits and labels."""
        # 3D axis
        self._ax_3d.set_xlabel('X (m)')
        self._ax_3d.set_ylabel('Y (m)')
        self._ax_3d.set_zlabel('Z (m)')
        self._ax_3d.set_title('3D View')
        
        # 2D top-down view
        self._ax_2d.set_xlabel('X (m)')
        self._ax_2d.set_ylabel('Y (m)')
        self._ax_2d.set_title('Top-Down View')
        self._ax_2d.set_aspect('equal')
        self._ax_2d.grid(True, alpha=0.3)

    def render(
        self,
        *,
        agent_positions: Sequence[NDArray[np.float64]],
        target_position: NDArray[np.float64],
        paths: Optional[Sequence[Sequence[NDArray[np.float64]]]] = None,
        frontier_candidates: Optional[NDArray[np.float64]] = None,
        frontier_targets: Optional[Sequence[Optional[NDArray[np.float64]]]] = None,
        occupancy_slice: Optional[NDArray[np.int_]] = None,
        slice_height: float = 0.0,
        agent_velocities: Optional[Sequence[NDArray[np.float64]]] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Render the current state of the environment.
        
        Args:
            agent_positions: List of agent 3D positions
            target_position: Target 3D position
            paths: Optional list of paths (one per agent)
            frontier_candidates: Optional array of frontier candidate positions
            frontier_targets: Optional list of assigned frontier targets per agent
            occupancy_slice: Optional 2D occupancy grid slice for top-down view
            slice_height: Height of the occupancy slice
            agent_velocities: Optional list of agent velocities
            step: Optional step number for title
            metrics: Optional dict of metrics to display
        """
        self._ax_3d.cla()
        self._ax_2d.cla()
        self._setup_axes()
        
        if step is not None:
            self._ax_3d.set_title(f'3D View (Step {step})')
        
        # Draw occupancy slice in 2D view
        if occupancy_slice is not None:
            self._draw_occupancy_2d(occupancy_slice, slice_height)
        
        # Draw frontier candidates
        if frontier_candidates is not None and len(frontier_candidates) > 0:
            self._draw_frontiers(frontier_candidates)
        
        # Draw paths
        if paths is not None:
            self._draw_paths(paths, agent_positions)
        
        # Draw frontier targets
        if frontier_targets is not None:
            self._draw_frontier_targets(frontier_targets, agent_positions)
        
        # Draw agents
        self._draw_agents(agent_positions, agent_velocities)
        
        # Draw target
        self._draw_target(target_position)
        
        # Draw metrics
        if metrics:
            self._draw_metrics(metrics)
        
        # Update display
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(self._pause_time)

    def _draw_agents(
        self,
        positions: Sequence[NDArray[np.float64]],
        velocities: Optional[Sequence[NDArray[np.float64]]] = None,
    ) -> None:
        """Draw agents as colored spheres with optional velocity arrows."""
        for i, pos in enumerate(positions):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            
            # 3D view
            self._ax_3d.scatter(
                pos[0], pos[1], pos[2],
                c=color, s=100, marker='o', label=f'Agent {i}',
                edgecolors='black', linewidths=1
            )
            
            # 2D view
            self._ax_2d.scatter(
                pos[0], pos[1],
                c=color, s=80, marker='o',
                edgecolors='black', linewidths=1
            )
            self._ax_2d.annotate(
                f'A{i}', (pos[0], pos[1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color=color
            )
            
            # Draw velocity arrows
            if velocities is not None and i < len(velocities):
                vel = velocities[i]
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 0.1:
                    scale = min(5.0, vel_norm)
                    vel_dir = vel / vel_norm * scale
                    
                    # 3D arrow
                    self._ax_3d.quiver(
                        pos[0], pos[1], pos[2],
                        vel_dir[0], vel_dir[1], vel_dir[2],
                        color=color, alpha=0.7, arrow_length_ratio=0.2
                    )
                    
                    # 2D arrow
                    self._ax_2d.arrow(
                        pos[0], pos[1], vel_dir[0], vel_dir[1],
                        head_width=0.5, head_length=0.3,
                        fc=color, ec=color, alpha=0.7
                    )

    def _draw_target(self, position: NDArray[np.float64]) -> None:
        """Draw target as a red star."""
        # 3D view
        self._ax_3d.scatter(
            position[0], position[1], position[2],
            c=TARGET_COLOR, s=200, marker='*', label='Target',
            edgecolors='black', linewidths=1
        )
        
        # 2D view
        self._ax_2d.scatter(
            position[0], position[1],
            c=TARGET_COLOR, s=150, marker='*',
            edgecolors='black', linewidths=1
        )
        self._ax_2d.annotate(
            'Target', (position[0], position[1]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, color=TARGET_COLOR
        )

    def _draw_paths(
        self,
        paths: Sequence[Sequence[NDArray[np.float64]]],
        agent_positions: Sequence[NDArray[np.float64]],
    ) -> None:
        """Draw A* paths for each agent."""
        for i, path in enumerate(paths):
            if not path or len(path) < 2:
                continue
            
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            
            # Extract coordinates
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            zs = [p[2] for p in path]
            
            # 3D path
            self._ax_3d.plot(
                xs, ys, zs,
                color=color, linestyle='--', linewidth=1.5,
                alpha=PATH_ALPHA
            )
            
            # 2D path
            self._ax_2d.plot(
                xs, ys,
                color=color, linestyle='--', linewidth=1.5,
                alpha=PATH_ALPHA
            )

    def _draw_frontiers(self, candidates: NDArray[np.float64]) -> None:
        """Draw frontier candidate points."""
        if candidates.shape[0] == 0:
            return
        
        # 3D view - small dots
        self._ax_3d.scatter(
            candidates[:, 0], candidates[:, 1], candidates[:, 2],
            c=FRONTIER_COLOR, s=10, alpha=0.3, marker='.'
        )
        
        # 2D view
        self._ax_2d.scatter(
            candidates[:, 0], candidates[:, 1],
            c=FRONTIER_COLOR, s=5, alpha=0.3, marker='.'
        )

    def _draw_frontier_targets(
        self,
        targets: Sequence[Optional[NDArray[np.float64]]],
        agent_positions: Sequence[NDArray[np.float64]],
    ) -> None:
        """Draw assigned frontier targets with lines to agents."""
        for i, target in enumerate(targets):
            if target is None:
                continue
            
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            pos = agent_positions[i]
            
            # 3D view - diamond marker
            self._ax_3d.scatter(
                target[0], target[1], target[2],
                c=color, s=80, marker='D', alpha=0.8,
                edgecolors='black', linewidths=1
            )
            
            # Line from agent to frontier target
            self._ax_3d.plot(
                [pos[0], target[0]], [pos[1], target[1]], [pos[2], target[2]],
                color=color, linestyle=':', linewidth=1, alpha=0.5
            )
            
            # 2D view
            self._ax_2d.scatter(
                target[0], target[1],
                c=color, s=60, marker='D', alpha=0.8,
                edgecolors='black', linewidths=1
            )
            self._ax_2d.plot(
                [pos[0], target[0]], [pos[1], target[1]],
                color=color, linestyle=':', linewidth=1, alpha=0.5
            )

    def _draw_occupancy_2d(
        self,
        occupancy: NDArray[np.int_],
        height: float,
    ) -> None:
        """Draw 2D occupancy grid slice."""
        # Create extent based on world size and origin
        extent = [
            self.origin[0],
            self.origin[0] + occupancy.shape[0] * self.voxel_size,
            self.origin[1],
            self.origin[1] + occupancy.shape[1] * self.voxel_size,
        ]
        
        self._ax_2d.imshow(
            occupancy.T,
            origin='lower',
            extent=extent,
            cmap='Greys',
            alpha=0.5,
            aspect='auto'
        )

    def _draw_metrics(self, metrics: Dict[str, float]) -> None:
        """Draw metrics text box."""
        text_lines = [f"{k}: {v:.2f}" for k, v in metrics.items()]
        text = '\n'.join(text_lines)
        
        self._ax_2d.text(
            0.02, 0.98, text,
            transform=self._ax_2d.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    def save_frame(self, path: str) -> None:
        """Save current frame to file."""
        self._fig.savefig(path, dpi=150, bbox_inches='tight')

    def close(self) -> None:
        """Close the figure."""
        plt.close(self._fig)


__all__ = ["VoxelMapVisualizer", "MATPLOTLIB_AVAILABLE"]
