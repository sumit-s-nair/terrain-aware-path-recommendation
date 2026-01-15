"""
PyBullet-based Terrain Environment for Hiking Path RL.

Physics-based hiking simulation with research-backed friction coefficients and energy models.

References:
    [1] Ziaei, M. et al. (2017). "Coefficient of friction, walking speed and cadence on slippery 
        and dry surfaces." Int. J. Occupational Safety and Ergonomics. DOI: 10.1080/10803548.2017.1398922
    [2] Scarf, P. A. (2007). "Route choice in mountain navigation, Naismith's Rule, and the 
        equivalence of distance and climb." J. Operational Research Society, 58(9), 1199-1205.
    [3] Naismith, W. W. (1892). Scottish Mountaineering Club Journal - Original Naismith's Rule.
    [4] Piazza et al. (2022). "Active Scree Slope Stability Investigation." MDPI Water, 14(16), 2569.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import rasterio
from scipy.ndimage import zoom
from typing import Optional, Tuple, Dict, Any

# PyBullet import with fallback
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: PyBullet not available. Using simplified physics mode.")


class PyBulletTerrainEnv(gym.Env):
    """
    Physics-based hiking environment using PyBullet simulation.
    
    Features:
        - Heightfield terrain from real DEM data
        - Research-backed friction coefficients [1]
        - Naismith's Rule energy model [2][3]
        - Sparse reward structure (goal-only)
        - Terrain-type based traversability
    
    The agent learns to find safe paths through environment physics alone,
    not reward shaping - encouraging natural switchback discovery.
    """
    
    metadata = {"render_modes": ["rgb_array", "human"]}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESEARCH-BACKED FRICTION COEFFICIENTS [1][4]
    # ═══════════════════════════════════════════════════════════════════════════
    # From: Ziaei et al. (2017) shoe friction testing and geotechnical literature
    TERRAIN_FRICTION = {
        "trail": 0.75,      # Established path - highest grip [1]
        "rock": 0.70,       # Dry solid rock - good grip [1]
        "grass": 0.55,      # Vegetation - moderate grip
        "forest": 0.50,     # Forest floor with debris
        "scree": 0.35,      # Loose rock - near angle of repose [4]
        "wet_rock": 0.40,   # Wet surface - reduced grip
        "water": 0.0,       # Impassable
    }
    
    # Energy multipliers per terrain type (effort to traverse)
    TERRAIN_ENERGY_MULTIPLIER = {
        "trail": 0.8,       # Easiest walking surface
        "rock": 1.2,        # Careful foot placement needed
        "grass": 1.0,       # Normal walking
        "forest": 1.3,      # Obstacles and uneven ground
        "scree": 2.5,       # Extremely exhausting [4]
        "water": 100.0,     # Impassable
    }
    
    # Map NLCD-like codes to terrain types (from vegetation_cost.tif)
    VEGETATION_TO_TERRAIN = {
        (0, 2.0): "trail",      # Very low cost = established path
        (2.0, 3.0): "grass",    # Low cost = grass/meadow
        (3.0, 5.0): "forest",   # Moderate = light forest
        (5.0, 15.0): "forest",  # Higher = dense forest
        (15.0, 50.0): "scree",  # High cost = difficult terrain
        (50.0, 100.0): "rock",  # Very high = exposed rock
        (100.0, float('inf')): "water",  # Impassable
    }
    
    # Naismith's Rule: 1m vertical = 8m horizontal [2][3]
    NAISMITH_RATIO = 8.0
    
    def __init__(
        self,
        processed_data_dir: str | Path = "data/processed",
        raw_data_dir: str | Path = "data/raw",
        max_steps: int = 10000,
        goal_distance_meters: float = 500.0,
        render_mode: Optional[str] = None,
        curriculum_level: int = 0,
        downsample_factor: int = 16,  # Reduce terrain resolution for PyBullet (16 = ~133x82 grid)
    ):
        super().__init__()
        
        self.processed_dir = Path(processed_data_dir)
        self.raw_dir = Path(raw_data_dir)
        self.max_steps = max_steps
        self.goal_distance_meters = goal_distance_meters
        self.render_mode = render_mode
        self.curriculum_level = curriculum_level
        self.downsample_factor = downsample_factor
        
        # Load terrain data
        self._load_terrain()
        
        # Initialize PyBullet
        self._setup_pybullet()
        
        # Agent state
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.stamina = 100.0
        self.health = 100.0
        self.step_count = 0
        self.trajectory = []
        self.goal = np.zeros(3, dtype=np.float32)
        
        # Progress tracking for exploration reward
        self.best_distance_achieved = float('inf')
        self.initial_distance = float('inf')
        
        # Heatmap accumulator for trajectory visualization
        self.trajectory_heatmap = None
        self.episode_heatmaps = []
        
        # Track curriculum successes
        self.curriculum_successes = 0
        self.curriculum_attempts = 0
        
        # Technique usage tracking
        self.technique_counts = {
            "walking": 0,
            "steep_hiking": 0,
            "climbing": 0,
            "rappelling": 0,
            "extreme": 0
        }
        
        # Observation and action spaces
        self._setup_spaces()
    
    def _load_terrain(self):
        """Load DEM and terrain classification data."""
        # Load elevation (DEM)
        dem_path = self.raw_dir / "dem_st_helens.tif"
        with rasterio.open(dem_path) as src:
            full_elevation = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            
            # Handle geographic coordinates
            if src.crs.to_epsg() == 4326:
                latitude = 46.2  # Mt. St. Helens
                lat_rad = np.radians(latitude)
                meters_per_degree = np.cos(lat_rad) * 111320
                self.cell_size = float(src.res[0]) * meters_per_degree
            else:
                self.cell_size = float(src.res[0])
        
        # Downsample for PyBullet performance
        self.elevation = zoom(full_elevation, 1.0 / self.downsample_factor, order=1)
        self.effective_cell_size = self.cell_size * self.downsample_factor
        self.map_h, self.map_w = self.elevation.shape
        
        print(f"Terrain loaded: {self.map_h}x{self.map_w} @ {self.effective_cell_size:.1f}m/cell")
        
        # Load vegetation cost for terrain classification
        veg_path = self.processed_dir / "vegetation_cost.tif"
        if veg_path.exists():
            with rasterio.open(veg_path) as src:
                veg_cost = src.read(1).astype(np.float32)
                self.vegetation_cost = zoom(veg_cost, 1.0 / self.downsample_factor, order=0)
        else:
            # Default to grass everywhere
            self.vegetation_cost = np.full(self.elevation.shape, 2.0, dtype=np.float32)
        
        # Load trail coordinates if available
        trail_path = self.processed_dir / "trail_coordinates.npy"
        if trail_path.exists():
            self.trail_coords = np.load(trail_path) / self.downsample_factor
        else:
            self.trail_coords = None
    
    def _setup_pybullet(self):
        """Initialize PyBullet simulation with terrain heightfield."""
        if not PYBULLET_AVAILABLE:
            self.physics_client = None
            return
        
        # Connect to PyBullet (direct mode for headless training)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create heightfield terrain
        # PyBullet expects row-major flattened array
        terrain_data = self.elevation.flatten().astype(np.float32)
        
        # Normalize heights to avoid numerical issues
        self.height_min = float(np.min(self.elevation))
        self.height_max = float(np.max(self.elevation))
        terrain_data_normalized = (terrain_data - self.height_min) / max(1.0, self.height_max - self.height_min)
        
        try:
            terrain_shape = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD,
                heightfieldData=terrain_data_normalized.tolist(),
                numHeightfieldRows=self.map_h,
                numHeightfieldColumns=self.map_w,
                meshScale=[self.effective_cell_size, self.effective_cell_size, self.height_max - self.height_min],
                heightfieldTextureScaling=1.0
            )
            
            # Center heightfield at origin
            self.terrain_id = p.createMultiBody(
                baseMass=0,  # Static terrain
                baseCollisionShapeIndex=terrain_shape,
                basePosition=[
                    self.map_w * self.effective_cell_size / 2,
                    self.map_h * self.effective_cell_size / 2,
                    (self.height_max + self.height_min) / 2
                ]
            )
            
            print("PyBullet heightfield terrain created successfully")
            
        except Exception as e:
            print(f"Failed to create PyBullet heightfield: {e}")
            print("Falling back to simplified physics mode")
            self.physics_client = None
            return
        
        # Create agent as a capsule (hiker)
        agent_radius = 0.3  # 30cm radius
        agent_height = 1.7  # 1.7m tall
        agent_mass = 70.0   # 70 kg hiker
        
        agent_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=agent_radius, height=agent_height)
        self.agent_id = p.createMultiBody(
            baseMass=agent_mass,
            baseCollisionShapeIndex=agent_shape,
            basePosition=[0, 0, 0]
        )
    
    def _setup_spaces(self):
        """Define observation and action spaces.
        
        Enhanced observation for better long-distance navigation.
        """
        # INCREASED view radius for better terrain awareness
        self.view_radius = 15  # 15 cells in each direction = 31x31 grid (~370m view at 12.3m/cell)
        view_size = 2 * self.view_radius + 1
        
        self.observation_space = spaces.Dict({
            # Local terrain heights (31x31 grid around agent)
            "terrain_heights": spaces.Box(
                low=-1000.0, high=5000.0,
                shape=(view_size, view_size),
                dtype=np.float32
            ),
            # Local terrain friction coefficients
            "terrain_friction": spaces.Box(
                low=0.0, high=1.0,
                shape=(view_size, view_size),
                dtype=np.float32
            ),
            # Local terrain slopes (NEW - helps agent see steep areas)
            "terrain_slopes": spaces.Box(
                low=0.0, high=90.0,
                shape=(view_size, view_size),
                dtype=np.float32
            ),
            # Agent physical state (expanded)
            "agent_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(12,),  # Added 2 more: progress_ratio, steps_remaining_ratio
                dtype=np.float32
            ),
            # Goal info: bearing + distance (NEW - critical for long distances!)
            "goal_info": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),  # [bearing_x, bearing_y, distance_normalized, elevation_diff]
                dtype=np.float32
            ),
        })
        
        # Continuous action: desired movement direction (2D normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32
        )
    
    def _get_terrain_type(self, row: int, col: int) -> str:
        """Get terrain type from vegetation cost map."""
        row = int(np.clip(row, 0, self.map_h - 1))
        col = int(np.clip(col, 0, self.map_w - 1))
        cost = self.vegetation_cost[row, col]
        
        for (low, high), terrain_type in self.VEGETATION_TO_TERRAIN.items():
            if low <= cost < high:
                return terrain_type
        return "grass"  # Default
    
    def _get_friction(self, row: int, col: int) -> float:
        """Get friction coefficient at position."""
        terrain_type = self._get_terrain_type(row, col)
        return self.TERRAIN_FRICTION.get(terrain_type, 0.5)
    
    def _get_slope(self, row: int, col: int) -> float:
        """Calculate local slope in degrees."""
        row = int(np.clip(row, 1, self.map_h - 2))
        col = int(np.clip(col, 1, self.map_w - 2))
        
        dz_dy = (self.elevation[row + 1, col] - self.elevation[row - 1, col]) / (2 * self.effective_cell_size)
        dz_dx = (self.elevation[row, col + 1] - self.elevation[row, col - 1]) / (2 * self.effective_cell_size)
        
        gradient = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_deg = np.degrees(np.arctan(gradient))
        return float(slope_deg)
    
    def _compute_stamina_drain(
        self,
        prev_pos: np.ndarray,
        new_pos: np.ndarray,
        terrain_type: str
    ) -> float:
        """
        Energy model based on Naismith's Rule [2][3].
        
        1 meter of vertical gain = 8 meters of horizontal walking in energy terms.
        """
        horizontal_dist = np.linalg.norm(new_pos[:2] - prev_pos[:2])
        vertical_gain = max(0, new_pos[2] - prev_pos[2])
        
        # Naismith's 8:1 equivalence for climbing
        equivalent_distance = horizontal_dist + (vertical_gain * self.NAISMITH_RATIO)
        
        # Base drain per equivalent meter
        base_drain = 0.002
        
        # Terrain multiplier (harder terrain = more exhausting)
        multiplier = self.TERRAIN_ENERGY_MULTIPLIER.get(terrain_type, 1.0)
        
        return equivalent_distance * base_drain * multiplier
    
    def _pos_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid coordinates."""
        col = int(pos[0] / self.effective_cell_size)
        row = int(pos[1] / self.effective_cell_size)
        return (
            int(np.clip(row, 0, self.map_h - 1)),
            int(np.clip(col, 0, self.map_w - 1))
        )
    
    def _grid_to_pos(self, row: int, col: int) -> np.ndarray:
        """Convert grid coordinates to world position."""
        x = col * self.effective_cell_size
        y = row * self.effective_cell_size
        z = self.elevation[row, col]
        return np.array([x, y, z], dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build observation dictionary with enhanced information."""
        row, col = self._pos_to_grid(self.agent_pos)
        view_size = 2 * self.view_radius + 1
        
        # Extract local terrain patches (heights, friction, AND slopes)
        terrain_heights = np.zeros((view_size, view_size), dtype=np.float32)
        terrain_friction = np.zeros_like(terrain_heights)
        terrain_slopes = np.zeros_like(terrain_heights)
        
        for dr in range(-self.view_radius, self.view_radius + 1):
            for dc in range(-self.view_radius, self.view_radius + 1):
                r = int(np.clip(row + dr, 0, self.map_h - 1))
                c = int(np.clip(col + dc, 0, self.map_w - 1))
                
                idx_r = dr + self.view_radius
                idx_c = dc + self.view_radius
                terrain_heights[idx_r, idx_c] = self.elevation[r, c]
                terrain_friction[idx_r, idx_c] = self._get_friction(r, c)
                terrain_slopes[idx_r, idx_c] = self._get_slope(r, c)
        
        # Agent state (expanded with progress info)
        current_friction = self._get_friction(row, col)
        current_slope = self._get_slope(row, col)
        
        # Progress ratio: how much of initial distance have we covered?
        progress_ratio = 1.0 - (self.best_distance_achieved / max(1.0, self.initial_distance))
        
        # Steps remaining ratio: how many steps left before timeout?
        steps_remaining_ratio = 1.0 - (self.step_count / max(1, self.max_steps))
        
        agent_state = np.array([
            self.agent_vel[0],
            self.agent_vel[1],
            self.agent_vel[2],
            self.stamina / 100.0,
            self.health / 100.0,
            0.0,  # roll (placeholder)
            0.0,  # pitch (placeholder)
            current_friction,
            current_slope / 60.0,  # Normalized
            self.agent_pos[2] / 3000.0,  # Normalized elevation
            progress_ratio,  # NEW: how much progress made
            steps_remaining_ratio,  # NEW: urgency signal
        ], dtype=np.float32)
        
        # Goal info: bearing + distance + elevation difference
        goal_direction = self.goal[:2] - self.agent_pos[:2]
        goal_dist = np.linalg.norm(goal_direction)
        
        if goal_dist > 0:
            goal_bearing = goal_direction / goal_dist
        else:
            goal_bearing = np.zeros(2, dtype=np.float32)
        
        # Normalize distance by curriculum goal distance for consistency across levels
        distance_normalized = goal_dist / max(1.0, self.goal_distance_meters)
        
        # Elevation difference (positive = uphill to goal)
        elevation_diff = (self.goal[2] - self.agent_pos[2]) / 1000.0  # Normalized by 1km
        
        goal_info = np.array([
            goal_bearing[0],
            goal_bearing[1],
            distance_normalized,  # CRITICAL: agent now knows how far!
            elevation_diff,  # Agent knows if goal is up or down
        ], dtype=np.float32)
        
        return {
            "terrain_heights": terrain_heights,
            "terrain_friction": terrain_friction,
            "terrain_slopes": terrain_slopes,
            "agent_state": agent_state,
            "goal_info": goal_info,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        self.curriculum_attempts += 1
        
        # FIXED GOAL = SUMMIT, Agent starts at curriculum distance and walks UPHILL
        if self.trail_coords is not None and len(self.trail_coords) >= 2:
            # Find summit (highest elevation point on trail) - THIS IS ALWAYS THE GOAL
            max_elevation = -np.inf
            summit_idx = 0
            
            for i, point in enumerate(self.trail_coords):
                r, c = int(point[0]), int(point[1])
                if 0 <= r < self.map_h and 0 <= c < self.map_w:
                    elev = self.elevation[r, c]
                    if elev > max_elevation:
                        max_elevation = elev
                        summit_idx = i
            
            # GOAL is ALWAYS the summit (fixed)
            summit_point = self.trail_coords[summit_idx]
            self.goal = self._grid_to_pos(int(summit_point[0]), int(summit_point[1]))
            
            # Agent spawns at curriculum distance FROM summit (in meters)
            target_distance_meters = self.goal_distance_meters
            cumulative_dist_meters = 0.0
            start_idx = 0  # Default to trailhead
            
            # Walk backwards from summit to find spawn point at target distance
            for i in range(summit_idx - 1, -1, -1):
                # Convert to world positions to get actual meters
                prev_pos = self._grid_to_pos(int(self.trail_coords[i + 1][0]), int(self.trail_coords[i + 1][1]))
                curr_pos = self._grid_to_pos(int(self.trail_coords[i][0]), int(self.trail_coords[i][1]))
                segment_dist = np.linalg.norm(curr_pos[:2] - prev_pos[:2])  # Distance in meters
                cumulative_dist_meters += segment_dist
                
                if cumulative_dist_meters >= target_distance_meters:
                    start_idx = i
                    break
            
            start_point = self.trail_coords[start_idx]
            self.agent_pos = self._grid_to_pos(int(start_point[0]), int(start_point[1]))
            
        else:
            # Fallback: random positions (should rarely happen)
            start_row = self.np_random.integers(self.map_h // 4, 3 * self.map_h // 4)
            start_col = self.np_random.integers(self.map_w // 4, 3 * self.map_w // 4)
            self.agent_pos = self._grid_to_pos(start_row, start_col)
            
            goal_row = self.np_random.integers(self.map_h // 4, 3 * self.map_h // 4)
            goal_col = self.np_random.integers(self.map_w // 4, 3 * self.map_w // 4)
            self.goal = self._grid_to_pos(goal_row, goal_col)
        
        # Reset agent state
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.stamina = 100.0
        self.health = 100.0
        self.step_count = 0
        self.trajectory = [self.agent_pos.copy()]
        
        # Reset technique tracking for new episode
        self.technique_counts = {
            "walking": 0,
            "steep_hiking": 0,
            "climbing": 0,
            "rappelling": 0,
            "extreme": 0
        }
        
        # Initialize progress tracking for this episode
        self.initial_distance = float(np.linalg.norm(self.agent_pos[:2] - self.goal[:2]))
        self.best_distance_achieved = self.initial_distance
        
        # Initialize trajectory heatmap for this episode
        self.trajectory_heatmap = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        start_row, start_col = self._pos_to_grid(self.agent_pos)
        self.trajectory_heatmap[start_row, start_col] = 1
        
        # Update PyBullet agent position if available
        if self.physics_client is not None:
            try:
                p.resetBasePositionAndOrientation(
                    self.agent_id,
                    self.agent_pos.tolist(),
                    [0, 0, 0, 1]
                )
                p.resetBaseVelocity(self.agent_id, [0, 0, 0], [0, 0, 0])
            except:
                pass
        
        obs = self._get_observation()
        info = {
            "start_pos": self.agent_pos.copy(),
            "goal_pos": self.goal.copy(),
            "initial_distance": self.initial_distance,
        }
        
        return obs, info
    
    def _apply_physics(self, action: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Apply physics-based movement with slope-dependent techniques.
        
        Returns:
            movement: 2D movement vector
            technique: Name of traversal technique used
            reward_multiplier: Multiplier for progress rewards (lower for harder techniques)
        
        Technique speed factors based on mountaineering research:
        - Normal walking: ~5 km/h horizontal, ~300 m/h vertical (Naismith's Rule)
        - Technical climbing: ~185 m/h vertical (23.4m in 7:36 min, outdoor 5c route)
        - Mountaineering estimate: 200-300 vertical m/h for moderate terrain
        """
        row, col = self._pos_to_grid(self.agent_pos)
        slope = self._get_slope(row, col)
        
        # Desired movement direction from action
        direction = np.array([action[0], action[1], 0], dtype=np.float32)
        dir_mag = np.linalg.norm(direction[:2])
        if dir_mag > 1.0:
            direction[:2] /= dir_mag
        
        # Determine if going uphill or downhill based on gradient
        dz_dy = (self.elevation[min(row + 1, self.map_h - 1), col] - 
                 self.elevation[max(row - 1, 0), col]) / (2 * self.effective_cell_size)
        dz_dx = (self.elevation[row, min(col + 1, self.map_w - 1)] - 
                 self.elevation[row, max(col - 1, 0)]) / (2 * self.effective_cell_size)
        gradient = np.array([dz_dx, dz_dy])
        
        # Dot product: positive = uphill, negative = downhill
        going_uphill = np.dot(direction[:2], gradient) > 0
        
        # Base walking speed (m/s) - average hiking pace
        base_speed = 1.5
        
        # Determine technique and speed factor based on slope
        # REWARD MULTIPLIERS INCREASED to encourage reaching steep summits
        if slope <= 30:
            # Normal walking - easy terrain
            technique = "walking"
            speed_factor = 1.0
            reward_multiplier = 1.0
        elif slope <= 45:
            # Steep hiking - challenging but no technical gear needed
            technique = "steep_hiking"
            speed_factor = 0.6
            reward_multiplier = 1.0  # Increased from 0.9 to encourage usage
        elif slope <= 60:
            # Technical terrain - climbing gear helpful
            technique = "climbing" if going_uphill else "rappelling"
            speed_factor = 0.3
            reward_multiplier = 0.95  # Increased from 0.8
        elif slope <= 75:
            # Very technical - requires climbing/rappelling gear
            technique = "climbing" if going_uphill else "rappelling"
            speed_factor = 0.15
            reward_multiplier = 0.9  # Increased from 0.6
        else:
            # Extreme terrain (75-90°) - barely traversable
            technique = "extreme"
            speed_factor = 0.05
            reward_multiplier = 0.4  # Kept low for safety
        
        # Track technique usage
        self.technique_counts[technique] += 1
        
        # Effective speed
        effective_speed = base_speed * speed_factor
        
        # Apply movement
        delta_time = 0.5  # Simulate 0.5 seconds per step
        movement = direction[:2] * effective_speed * delta_time
        
        return movement, technique, reward_multiplier
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        prev_pos = self.agent_pos.copy()
        
        # Get current terrain info
        row, col = self._pos_to_grid(self.agent_pos)
        terrain_type = self._get_terrain_type(row, col)
        
        # Apply physics-based movement
        movement, technique, reward_multiplier = self._apply_physics(action)
        
        # Update position
        self.agent_pos[0] += movement[0]
        self.agent_pos[1] += movement[1]
        
        # Clamp to map bounds
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, (self.map_w - 1) * self.effective_cell_size)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, (self.map_h - 1) * self.effective_cell_size)
        
        # Update elevation
        new_row, new_col = self._pos_to_grid(self.agent_pos)
        self.agent_pos[2] = self.elevation[new_row, new_col]
        
        # Update velocity (for observation)
        self.agent_vel = (self.agent_pos - prev_pos) / 0.5  # 0.5s timestep
        
        # Compute stamina drain (Naismith-based)
        new_terrain_type = self._get_terrain_type(new_row, new_col)
        # STAMINA DISABLED - agent doesn't get exhausted
        # stamina_drain = self._compute_stamina_drain(prev_pos, self.agent_pos, new_terrain_type)
        # self.stamina = max(0.0, self.stamina - stamina_drain)
        
        # Record trajectory
        self.trajectory.append(self.agent_pos.copy())
        
        # ═══════════════════════════════════════════════════════════════
        # REWARD FUNCTION with progress exploration bonus
        # ═══════════════════════════════════════════════════════════════
        goal_dist = np.linalg.norm(self.agent_pos[:2] - self.goal[:2])
        reached_goal = goal_dist < 5.0  # Within 5 meters
        
        reward = 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # PROGRESS REWARD (fires ONCE per new closest distance)
        # This prevents spiraling because agent can only earn this reward
        # by getting closer than ever before - can't farm by oscillating
        # ═══════════════════════════════════════════════════════════════
        if goal_dist < self.best_distance_achieved:
            # Progress reward scaled by technique difficulty
            # Walking gets full reward, climbing/rappelling gets less (encourages easier routes)
            progress = self.best_distance_achieved - goal_dist
            
            # PROXIMITY BONUS: Increase reward when close to goal
            # This helps agent push through steep summit terrain
            proximity_bonus = 1.0
            if goal_dist < 50:
                proximity_bonus = 2.0  # Double reward in last 50m
            if goal_dist < 25:
                proximity_bonus = 3.0  # Triple reward in last 25m
            
            reward += progress * 100.0 * reward_multiplier * proximity_bonus
            self.best_distance_achieved = goal_dist
        
        # Update heatmap with current position
        if self.trajectory_heatmap is not None:
            self.trajectory_heatmap[new_row, new_col] += 1
        
        if reached_goal:
            reward += 1000.0  # Reduced from 10000 to match progress rewards better
            self.curriculum_successes += 1
        # No health penalty - all terrain is traversable with appropriate technique
        
        # Check termination - only goal or timeout
        terminated = reached_goal
        truncated = self.step_count >= self.max_steps
        
        if truncated and not terminated:
            reward -= 100.0  # Reduced timeout penalty to match new scale
        
        # Build observation
        obs = self._get_observation()
        
        info = {
            "technique": technique,
            "technique_counts": self.technique_counts.copy(),
            "terrain_type": new_terrain_type,
            "stamina": self.stamina,
            "health": self.health,
            "goal_distance": goal_dist,
            "reached_goal": reached_goal,
            "position": self.agent_pos.copy(),
            "best_distance": self.best_distance_achieved,
            "progress_made": self.initial_distance - self.best_distance_achieved,
            "trajectory": self.trajectory.copy() if terminated or truncated else None,
            "heatmap": self.trajectory_heatmap.copy() if terminated or truncated else None,
        }
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "rgb_array":
            # Return terrain visualization with agent position
            vis = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
            
            # Elevation shading
            elev_norm = (self.elevation - self.height_min) / max(1.0, self.height_max - self.height_min)
            vis[:, :, 0] = (elev_norm * 200).astype(np.uint8)
            vis[:, :, 1] = (elev_norm * 200).astype(np.uint8)
            vis[:, :, 2] = (elev_norm * 200).astype(np.uint8)
            
            # Draw agent
            row, col = self._pos_to_grid(self.agent_pos)
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.map_h and 0 <= c < self.map_w:
                        vis[r, c] = [255, 0, 0]
            
            # Draw goal
            grow, gcol = self._pos_to_grid(self.goal)
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = grow + dr, gcol + dc
                    if 0 <= r < self.map_h and 0 <= c < self.map_w:
                        vis[r, c] = [0, 255, 0]
            
            return vis
        
        return None
    
    def save_heatmap(self, filepath: str, include_terrain: bool = True):
        """
        Save trajectory heatmap as an image.
        
        Args:
            filepath: Path to save the image (e.g., 'outputs/heatmap_ep100.png')
            include_terrain: Whether to overlay on terrain elevation
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        if self.trajectory_heatmap is None:
            print("No heatmap data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Background: terrain elevation
        if include_terrain:
            elev_norm = (self.elevation - self.height_min) / max(1.0, self.height_max - self.height_min)
            ax.imshow(elev_norm, cmap='terrain', origin='upper', alpha=0.7)
        
        # Overlay: trajectory heatmap (log scale for visibility)
        heatmap_vis = np.log1p(self.trajectory_heatmap)  # log(1 + x) for better visualization
        if heatmap_vis.max() > 0:
            # Custom colormap: transparent -> yellow -> red
            colors = [(0, 0, 0, 0), (1, 1, 0, 0.5), (1, 0, 0, 0.9)]
            cmap = LinearSegmentedColormap.from_list('trajectory', colors)
            ax.imshow(heatmap_vis, cmap=cmap, origin='upper')
        
        # Mark start position
        if len(self.trajectory) > 0:
            start = self.trajectory[0]
            sr, sc = self._pos_to_grid(start)
            ax.plot(sc, sr, 'go', markersize=15, markeredgecolor='white', 
                   markeredgewidth=2, label='Start')
        
        # Mark goal position
        gr, gc = self._pos_to_grid(self.goal)
        ax.plot(gc, gr, 'b*', markersize=20, markeredgecolor='white',
               markeredgewidth=2, label='Goal')
        
        # Mark current/final position
        cr, cc = self._pos_to_grid(self.agent_pos)
        ax.plot(cc, cr, 'r^', markersize=12, markeredgecolor='white',
               markeredgewidth=2, label='Final Position')
        
        # Add legend and title
        ax.legend(loc='upper right')
        
        progress = self.initial_distance - self.best_distance_achieved
        title = f"Episode Trajectory Heatmap\n"
        title += f"Progress: {progress:.1f}m / {self.initial_distance:.1f}m ({100*progress/max(1,self.initial_distance):.1f}%)"
        ax.set_title(title)
        
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to: {filepath}")
    
    def close(self):
        """Clean up PyBullet."""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        self.physics_client = None


# Curriculum levels - full trail is 14.6km with 1397m elevation gain
CURRICULUM_LEVELS = [
    {"goal_distance": 10,    "max_steps": 500,   "required_successes": 20},    # Level 0: 10m
    {"goal_distance": 15,    "max_steps": 600,   "required_successes": 25},    # Level 1: 15m
    {"goal_distance": 20,    "max_steps": 750,   "required_successes": 30},    # Level 2: 20m
    {"goal_distance": 25,    "max_steps": 900,   "required_successes": 35},    # Level 3: 25m
    {"goal_distance": 30,    "max_steps": 1000,  "required_successes": 40},    # Level 4: 30m
    {"goal_distance": 40,    "max_steps": 1250,  "required_successes": 50},    # Level 5: 40m
    {"goal_distance": 50,    "max_steps": 1500,  "required_successes": 60},    # Level 6: 50m
    {"goal_distance": 75,    "max_steps": 2000,  "required_successes": 80},    # Level 7: 75m
    {"goal_distance": 100,   "max_steps": 2500,  "required_successes": 100},   # Level 8: 100m
    {"goal_distance": 150,   "max_steps": 4500,  "required_successes": 120},   # Level 9: 150m (Increased steps)
    {"goal_distance": 200,   "max_steps": 5500,  "required_successes": 140},   # Level 10: 200m (NEW) (Increased steps)
    {"goal_distance": 250,   "max_steps": 6000,  "required_successes": 150},   # Level 11: 250m
    {"goal_distance": 350,   "max_steps": 8500,  "required_successes": 175},   # Level 12: 350m (NEW)
    {"goal_distance": 500,   "max_steps": 12000, "required_successes": 200},   # Level 13: 500m
    {"goal_distance": 1000,  "max_steps": 25000, "required_successes": 300},   # Level 14: 1km
    {"goal_distance": 2000,  "max_steps": 50000, "required_successes": 400},   # Level 15: 2km
    {"goal_distance": 5000,  "max_steps": 120000,"required_successes": 500},   # Level 16: 5km
    {"goal_distance": 10000, "max_steps": 250000,"required_successes": 750},   # Level 17: 10km
    {"goal_distance": 15000, "max_steps": 400000,"required_successes": 1000},  # Level 18: 14.6km (full trail)
]


def make_env(curriculum_level: int = 0):
    """Factory function to create environment."""
    config = CURRICULUM_LEVELS[min(curriculum_level, len(CURRICULUM_LEVELS) - 1)]
    
    return PyBulletTerrainEnv(
        goal_distance_meters=config["goal_distance"],
        max_steps=config["max_steps"],
        curriculum_level=curriculum_level,
    )


if __name__ == "__main__":
    # Test environment
    print("Testing PyBullet Terrain Environment...")
    
    env = make_env(curriculum_level=0)
    obs, info = env.reset()
    
    print(f"Start position: {info['start_pos']}")
    print(f"Goal position: {info['goal_pos']}")
    print(f"Initial distance: {info['initial_distance']:.1f}m")
    print(f"Observation shapes:")
    for key, val in obs.items():
        print(f"  {key}: {val.shape}")
    
    # Take a few random steps
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.1f}, stamina={info['stamina']:.1f}, dist={info['goal_distance']:.1f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}: {info['result']}")
            break
    
    print(f"Total reward: {total_reward:.1f}")
    env.close()
