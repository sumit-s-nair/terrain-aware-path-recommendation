"""
A* Path Comparison Tool for Terrain-Aware Path Recommendation.

Implements A* pathfinding with terrain-aware costs and compares
RL agent paths against A* and actual trail GPX files.

References:
    [1] Hart, P.E., Nilsson, N.J., Raphael, B. (1968). "A Formal Basis for the 
        Heuristic Determination of Minimum Cost Paths." IEEE Trans. Systems Science 
        and Cybernetics, 4(2), 100-107.
"""

from __future__ import annotations

import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import rasterio
from scipy.ndimage import zoom
import gpxpy
import gpxpy.gpx
from pyproj import Transformer, CRS


class TerrainAwarAStar:
    """
    A* pathfinding with terrain-aware costs.
    
    Cost function considers:
        - Distance traveled
        - Elevation gain (heavily penalized)
        - Terrain difficulty (from vegetation cost map)
        - Slope steepness
    
    This represents the "naive" approach that doesn't understand
    the cumulative energy cost that makes switchbacks valuable.
    """
    
    def __init__(
        self,
        elevation: np.ndarray,
        vegetation_cost: np.ndarray,
        cell_size: float,
        elevation_weight: float = 10.0,
        terrain_weight: float = 2.0,
        slope_weight: float = 1.0,
    ):
        self.elevation = elevation
        self.vegetation_cost = vegetation_cost
        self.cell_size = cell_size
        self.height, self.width = elevation.shape
        
        # Cost weights
        self.elevation_weight = elevation_weight
        self.terrain_weight = terrain_weight
        self.slope_weight = slope_weight
        
        # Precompute slope map
        self.slope = self._compute_slope()
    
    def _compute_slope(self) -> np.ndarray:
        """Compute slope in degrees from elevation."""
        dz_dy, dz_dx = np.gradient(self.elevation, self.cell_size)
        gradient = np.sqrt(dz_dx**2 + dz_dy**2)
        return np.degrees(np.arctan(gradient))
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors."""
        row, col = pos
        neighbors = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    # Check if passable (vegetation cost < 100 = not water)
                    if self.vegetation_cost[nr, nc] < 100:
                        neighbors.append((nr, nc))
        
        return neighbors
    
    def _movement_cost(
        self,
        current: Tuple[int, int],
        neighbor: Tuple[int, int]
    ) -> float:
        """
        Compute movement cost from current to neighbor.
        
        A* uses simple additive costs - it doesn't understand that
        cumulative steep climbing is exhausting, which is why it
        tends to take direct routes instead of switchbacks.
        """
        cr, cc = current
        nr, nc = neighbor
        
        # Base distance cost (diagonal = sqrt(2))
        is_diagonal = (cr != nr) and (cc != nc)
        distance_cost = self.cell_size * (1.414 if is_diagonal else 1.0)
        
        # Elevation gain cost (only penalize uphill)
        elev_current = self.elevation[cr, cc]
        elev_neighbor = self.elevation[nr, nc]
        elev_gain = max(0, elev_neighbor - elev_current)
        elevation_cost = elev_gain * self.elevation_weight
        
        # Terrain difficulty cost
        terrain_cost = self.vegetation_cost[nr, nc] * self.terrain_weight
        
        # Slope cost
        slope_cost = self.slope[nr, nc] * self.slope_weight
        
        return distance_cost + elevation_cost + terrain_cost + slope_cost
    
    def _heuristic(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """Euclidean distance heuristic."""
        dr = pos[0] - goal[0]
        dc = pos[1] - goal[1]
        return self.cell_size * np.sqrt(dr**2 + dc**2)
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Find shortest path using A*.
        
        Returns:
            path: List of (row, col) coordinates
            cost: Total path cost
        """
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[goal]
            
            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        # No path found
        return [], float('inf')


class PathComparator:
    """
    Compare paths from different sources:
        - RL Agent generated paths
        - A* computed paths
        - Actual hiking trail (GPX)
    """
    
    def __init__(self, processed_dir: str = "data/processed", raw_dir: str = "data/raw"):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self._load_data()
    
    def _load_data(self):
        """Load terrain data."""
        # Load DEM
        dem_path = self.raw_dir / "dem_st_helens.tif"
        with rasterio.open(dem_path) as src:
            self.elevation = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            
            # Calculate cell size in meters
            if src.crs.to_epsg() == 4326:
                latitude = 46.2
                self.cell_size = float(src.res[0]) * np.cos(np.radians(latitude)) * 111320
            else:
                self.cell_size = float(src.res[0])
        
        # Load vegetation cost
        veg_path = self.processed_dir / "vegetation_cost.tif"
        if veg_path.exists():
            with rasterio.open(veg_path) as src:
                self.vegetation_cost = src.read(1).astype(np.float32)
        else:
            self.vegetation_cost = np.ones_like(self.elevation) * 2.0
        
        # Load slope
        slope_path = self.processed_dir / "slope.tif"
        if slope_path.exists():
            with rasterio.open(slope_path) as src:
                self.slope = src.read(1).astype(np.float32)
        else:
            dz_dy, dz_dx = np.gradient(self.elevation, self.cell_size)
            gradient = np.sqrt(dz_dx**2 + dz_dy**2)
            self.slope = np.degrees(np.arctan(gradient))
        
        # Load trail coordinates
        trail_path = self.processed_dir / "trail_coordinates.npy"
        if trail_path.exists():
            self.trail_coords = np.load(trail_path)
        else:
            self.trail_coords = None
        
        # Initialize A* planner
        self.astar = TerrainAwarAStar(
            self.elevation,
            self.vegetation_cost,
            self.cell_size
        )
    
    def compute_path_metrics(self, path: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Compute comprehensive metrics for a path.
        
        Metrics:
            - total_distance: Path length in meters
            - total_elevation_gain: Cumulative uphill in meters
            - max_slope: Maximum slope encountered (degrees)
            - avg_slope: Average slope along path (degrees)
            - energy_cost: Estimated energy expenditure (Naismith-based)
            - risk_score: Combined safety metric
        """
        if len(path) < 2:
            return {"error": "Path too short"}
        
        total_distance = 0.0
        total_elevation_gain = 0.0
        slopes = []
        terrain_costs = []
        
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]
            
            # Distance
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            dist = self.cell_size * np.sqrt(dr**2 + dc**2)
            total_distance += dist
            
            # Elevation gain
            elev_prev = self.elevation[prev[0], prev[1]]
            elev_curr = self.elevation[curr[0], curr[1]]
            if elev_curr > elev_prev:
                total_elevation_gain += (elev_curr - elev_prev)
            
            # Slope
            slopes.append(self.slope[curr[0], curr[1]])
            
            # Terrain
            terrain_costs.append(self.vegetation_cost[curr[0], curr[1]])
        
        # Naismith's Rule energy: 1m vertical = 8m horizontal
        energy_cost = total_distance + (total_elevation_gain * 8.0)
        
        # Risk score (higher = more dangerous)
        max_slope = max(slopes)
        avg_slope = np.mean(slopes)
        avg_terrain = np.mean(terrain_costs)
        risk_score = (max_slope / 60) * 0.5 + (avg_terrain / 50) * 0.5
        
        return {
            "total_distance": total_distance,
            "total_elevation_gain": total_elevation_gain,
            "max_slope": float(max_slope),
            "avg_slope": float(avg_slope),
            "avg_terrain_cost": float(avg_terrain),
            "energy_cost": energy_cost,
            "risk_score": float(risk_score),
        }
    
    def compare_paths(
        self,
        rl_path: Optional[List[Tuple[int, int]]] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare paths from RL agent, A*, and actual trail.
        """
        results = {}
        
        # Determine start/goal from trail if not provided
        if self.trail_coords is not None and (start is None or goal is None):
            # Find summit (highest point)
            max_elev = -np.inf
            summit_idx = 0
            for i, point in enumerate(self.trail_coords):
                r, c = int(point[0]), int(point[1])
                if 0 <= r < self.elevation.shape[0] and 0 <= c < self.elevation.shape[1]:
                    elev = self.elevation[r, c]
                    if elev > max_elev:
                        max_elev = elev
                        summit_idx = i
            
            if start is None:
                start = (int(self.trail_coords[0, 0]), int(self.trail_coords[0, 1]))
            if goal is None:
                goal = (int(self.trail_coords[summit_idx, 0]), int(self.trail_coords[summit_idx, 1]))
        
        # Compute A* path
        print("Computing A* path...")
        astar_path, astar_cost = self.astar.find_path(start, goal)
        if astar_path:
            results["astar"] = self.compute_path_metrics(astar_path)
            results["astar"]["path"] = astar_path
        
        # Actual trail metrics
        if self.trail_coords is not None:
            trail_path = [(int(p[0]), int(p[1])) for p in self.trail_coords]
            results["actual_trail"] = self.compute_path_metrics(trail_path)
            results["actual_trail"]["path"] = trail_path
        
        # RL path metrics
        if rl_path:
            results["rl_agent"] = self.compute_path_metrics(rl_path)
            results["rl_agent"]["path"] = rl_path
        
        return results
    
    def visualize_comparison(
        self,
        results: Dict[str, Dict],
        output_path: str = "path_comparison.png"
    ):
        """Create visualization comparing different paths."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Map with paths
        ax1 = axes[0]
        
        # Show elevation as background
        elev_norm = (self.elevation - np.min(self.elevation)) / (np.max(self.elevation) - np.min(self.elevation))
        ax1.imshow(elev_norm, cmap='terrain', origin='upper')
        
        # Plot paths
        colors = {"astar": "red", "actual_trail": "blue", "rl_agent": "green"}
        labels = {"astar": "A* Path", "actual_trail": "Actual Trail", "rl_agent": "RL Agent"}
        
        for name, data in results.items():
            if "path" in data:
                path = data["path"]
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                ax1.plot(cols, rows, color=colors.get(name, "white"), 
                        linewidth=2, label=labels.get(name, name), alpha=0.8)
        
        ax1.legend(loc='upper right')
        ax1.set_title("Path Comparison on Terrain")
        ax1.set_xlabel("Column")
        ax1.set_ylabel("Row")
        
        # Right: Metrics comparison bar chart
        ax2 = axes[1]
        
        metrics_to_plot = ["total_distance", "total_elevation_gain", "max_slope", "energy_cost", "risk_score"]
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        for i, (name, data) in enumerate(results.items()):
            if "error" not in data:
                values = [data.get(m, 0) for m in metrics_to_plot]
                # Normalize for display
                max_vals = [5000, 500, 60, 10000, 1]  # Rough max values for each metric
                values_norm = [v / m for v, m in zip(values, max_vals)]
                ax2.bar(x + i * width, values_norm, width, label=labels.get(name, name), 
                       color=colors.get(name, "gray"))
        
        ax2.set_xlabel("Metric")
        ax2.set_ylabel("Normalized Value")
        ax2.set_title("Path Metrics Comparison")
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(["Distance", "Elev Gain", "Max Slope", "Energy", "Risk"])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()
    
    def print_comparison_table(self, results: Dict[str, Dict]):
        """Print formatted comparison table."""
        print("\n" + "=" * 80)
        print("PATH COMPARISON RESULTS")
        print("=" * 80)
        
        headers = ["Metric", "A*", "Actual Trail", "RL Agent"]
        metrics = [
            ("Distance (m)", "total_distance", ".0f"),
            ("Elevation Gain (m)", "total_elevation_gain", ".0f"),
            ("Max Slope (°)", "max_slope", ".1f"),
            ("Avg Slope (°)", "avg_slope", ".1f"),
            ("Energy Cost", "energy_cost", ".0f"),
            ("Risk Score", "risk_score", ".2f"),
        ]
        
        # Print header
        print(f"{'Metric':<25} | {'A*':>12} | {'Trail':>12} | {'RL Agent':>12}")
        print("-" * 70)
        
        for label, key, fmt in metrics:
            row = f"{label:<25} |"
            for source in ["astar", "actual_trail", "rl_agent"]:
                if source in results and key in results[source]:
                    val = results[source][key]
                    row += f" {val:{fmt}:>12} |"
                else:
                    row += f" {'N/A':>12} |"
            print(row)
        
        print("=" * 80)


def main():
    """Run path comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare hiking paths")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--metrics", action="store_true", help="Print metrics table")
    parser.add_argument("--rl-model", type=str, default=None, help="Path to RL model to evaluate")
    
    args = parser.parse_args()
    
    comparator = PathComparator()
    
    # Compare A* vs actual trail (RL agent path would be loaded from model)
    results = comparator.compare_paths()
    
    if args.metrics or not args.visualize:
        comparator.print_comparison_table(results)
    
    if args.visualize:
        comparator.visualize_comparison(results)


if __name__ == "__main__":
    main()
