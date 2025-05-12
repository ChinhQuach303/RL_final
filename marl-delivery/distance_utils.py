# utils/distance_utils.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distance_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistanceCalculator:
    def __init__(self, grid: np.ndarray):
        """
        Initialize distance calculator
        Args:
            grid: 2D numpy array representing the environment
        """
        try:
            logger.info("Initializing DistanceCalculator...")
            
            # Validate grid
            if not isinstance(grid, np.ndarray):
                raise TypeError(f"grid must be a numpy array, got {type(grid)}")
            if grid.ndim != 2:
                raise ValueError(f"grid must be 2D, got {grid.ndim}D")
            
            self.grid = grid
            self.rows, self.cols = grid.shape
            logger.info(f"Grid shape: {grid.shape}")
            
            # Initialize distance matrix cache
            self.distance_matrix = {}
            
            # Pre-compute distances for efficiency
            logger.info("Pre-computing distances...")
            self._compute_all_distances()
            logger.info("DistanceCalculator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DistanceCalculator: {str(e)}")
            raise

    def _compute_all_distances(self) -> None:
        """Pre-compute distances between all valid positions"""
        try:
            valid_positions = self._get_valid_positions()
            n_positions = len(valid_positions)
            logger.info(f"Computing distances for {n_positions} positions...")
            
            for i, pos1 in enumerate(valid_positions):
                for pos2 in valid_positions[i+1:]:
                    distance = self._compute_distance(pos1, pos2)
                    if distance < float('inf'):
                        self.distance_matrix[(pos1, pos2)] = distance
                        self.distance_matrix[(pos2, pos1)] = distance
            
            logger.info(f"Computed distances for {len(self.distance_matrix)//2} position pairs")
            
        except Exception as e:
            logger.error(f"Failed to compute distances: {str(e)}")
            raise

    def _get_valid_positions(self) -> List[Tuple[int, int]]:
        """Get list of valid positions (not obstacles)"""
        try:
            valid_positions = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.grid[i][j] != 1:  # 1 represents obstacle
                        valid_positions.append((i, j))
            return valid_positions
        except Exception as e:
            logger.error(f"Failed to get valid positions: {str(e)}")
            raise

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        try:
            i, j = pos
            return (0 <= i < self.rows and 
                    0 <= j < self.cols and 
                    self.grid[i][j] != 1)
        except Exception as e:
            logger.error(f"Failed to validate position {pos}: {str(e)}")
            return False

    def _compute_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Compute shortest path distance using BFS
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
        Returns:
            float: Shortest path distance or infinity if no path exists
        """
        try:
            if not self._is_valid_position(start) or not self._is_valid_position(goal):
                return float('inf')
            
            if start == goal:
                return 0.0
            
            # Initialize BFS
            queue = deque([(start, 0)])  # (position, distance)
            visited = {start}
            
            # Define possible moves (up, right, down, left)
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            while queue:
                (i, j), dist = queue.popleft()
                
                # Check all possible moves
                for di, dj in moves:
                    ni, nj = i + di, j + dj
                    next_pos = (ni, nj)
                    
                    if next_pos == goal:
                        return dist + 1
                    
                    if (self._is_valid_position(next_pos) and 
                        next_pos not in visited):
                        queue.append((next_pos, dist + 1))
                        visited.add(next_pos)
            
            return float('inf')  # No path found
            
        except Exception as e:
            logger.error(f"Failed to compute distance from {start} to {goal}: {str(e)}")
            return float('inf')

    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Get distance between two positions
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
        Returns:
            float: Distance between positions
        """
        try:
            # Check if positions are valid
            if not self._is_valid_position(pos1) or not self._is_valid_position(pos2):
                logger.warning(f"Invalid positions: {pos1} or {pos2}")
                return float('inf')
            
            # Check cache first
            if (pos1, pos2) in self.distance_matrix:
                return self.distance_matrix[(pos1, pos2)]
            if (pos2, pos1) in self.distance_matrix:
                return self.distance_matrix[(pos2, pos1)]
            
            # Compute distance if not in cache
            distance = self._compute_distance(pos1, pos2)
            
            # Cache the result
            if distance < float('inf'):
                self.distance_matrix[(pos1, pos2)] = distance
                self.distance_matrix[(pos2, pos1)] = distance
            
            return distance
            
        except Exception as e:
            logger.error(f"Failed to get distance between {pos1} and {pos2}: {str(e)}")
            return float('inf')

    def get_nearest_package(self, 
                          pos: Tuple[int, int], 
                          package_positions: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Get nearest package position and its distance
        Args:
            pos: Current position (row, col)
            package_positions: List of package positions
        Returns:
            Tuple of (nearest_package_position, distance)
        """
        try:
            if not package_positions:
                return None, float('inf')
            
            min_dist = float('inf')
            nearest_package = None
            
            for package_pos in package_positions:
                if not self._is_valid_position(package_pos):
                    continue
                    
                dist = self.get_distance(pos, package_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_package = package_pos
            
            return nearest_package, min_dist
            
        except Exception as e:
            logger.error(f"Failed to get nearest package from {pos}: {str(e)}")
            return None, float('inf')

    def get_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Get shortest path between two positions
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
        Returns:
            List of positions forming the shortest path, or None if no path exists
        """
        try:
            if not self._is_valid_position(start) or not self._is_valid_position(goal):
                return None
            
            if start == goal:
                return [start]
            
            # Initialize BFS
            queue = deque([(start, [start])])  # (position, path)
            visited = {start}
            
            # Define possible moves
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            while queue:
                (i, j), path = queue.popleft()
                
                for di, dj in moves:
                    ni, nj = i + di, j + dj
                    next_pos = (ni, nj)
                    
                    if next_pos == goal:
                        return path + [next_pos]
                    
                    if (self._is_valid_position(next_pos) and 
                        next_pos not in visited):
                        queue.append((next_pos, path + [next_pos]))
                        visited.add(next_pos)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get path from {start} to {goal}: {str(e)}")
            return None