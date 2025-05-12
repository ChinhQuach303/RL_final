# utils/state_utils.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from distance_utils import DistanceCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('state_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StateProcessor:
    def __init__(self, 
                 grid: np.ndarray,
                 n_robots: int,
                 n_packages: int):
        """
        Initialize state processor
        Args:
            grid: 2D numpy array representing the environment
            n_robots: Number of robots
            n_packages: Number of packages
        """
        try:
            logger.info("Initializing StateProcessor...")
            
            # Validate inputs
            if not isinstance(grid, np.ndarray):
                raise TypeError(f"grid must be a numpy array, got {type(grid)}")
            if grid.ndim != 2:
                raise ValueError(f"grid must be 2D, got {grid.ndim}D")
            if not isinstance(n_robots, int) or n_robots <= 0:
                raise ValueError(f"n_robots must be a positive integer, got {n_robots}")
            if not isinstance(n_packages, int) or n_packages <= 0:
                raise ValueError(f"n_packages must be a positive integer, got {n_packages}")
            
            self.grid = grid
            self.n_robots = n_robots
            self.n_packages = n_packages
            self.distance_calculator = DistanceCalculator(grid)
            
            # Calculate state size
            self.state_size = self._calculate_state_size()
            logger.info(f"State size: {self.state_size}")
            
            logger.info("StateProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize StateProcessor: {str(e)}")
            raise

    def _calculate_state_size(self) -> int:
        """Calculate the size of the state vector"""
        try:
            # Robot-specific information
            robot_features = 4  # position (2), has_package (1), nearest_package_dist (1)
            
            # Package information
            package_features = 3  # waiting packages, delivered packages, delivery progress
            
            # Global information
            global_features = 2  # robot distribution, delivery progress
            
            # Total features per agent
            return robot_features + package_features + global_features
            
        except Exception as e:
            logger.error(f"Failed to calculate state size: {str(e)}")
            raise

    def process_state(self, state: Dict) -> Dict[str, np.ndarray]:
        """
        Convert environment state to neural network input for each agent
        Args:
            state: Dictionary containing environment state
        Returns:
            Dictionary mapping agent IDs to their processed states
        """
        try:
            logger.debug("Processing state...")
            
            # Validate state dictionary
            required_keys = ['robot_positions', 'robot_packages', 'package_positions', 
                           'delivery_points', 'delivered_packages', 'grid']
            for key in required_keys:
                if key not in state:
                    raise KeyError(f"Missing required key in state: {key}")
            
            # Get basic state information
            robot_positions = state['robot_positions']
            robot_packages = state['robot_packages']
            package_positions = state['package_positions']
            delivery_points = state['delivery_points']
            delivered_packages = state['delivered_packages']
            
            # Calculate global metrics
            waiting_packages = sum(1 for pos in package_positions if pos is not None)
            delivery_progress = len(delivered_packages) / self.n_packages
            
            # Calculate robot distribution
            robot_positions_list = list(robot_positions.values())
            if len(robot_positions_list) > 1:
                distances = []
                for i in range(len(robot_positions_list)):
                    for j in range(i+1, len(robot_positions_list)):
                        dist = self.distance_calculator.get_distance(
                            robot_positions_list[i],
                            robot_positions_list[j]
                        )
                        if dist < float('inf'):
                            distances.append(dist)
                avg_robot_distance = np.mean(distances) if distances else 0
            else:
                avg_robot_distance = 0
            
            # Normalize robot distribution
            max_possible_distance = np.sqrt(self.grid.shape[0]**2 + self.grid.shape[1]**2)
            normalized_robot_distance = avg_robot_distance / max_possible_distance
            
            # Process state for each agent
            processed_states = {}
            for agent_id in robot_positions.keys():
                state_features = []
                
                # 1. Robot-specific information
                robot_pos = robot_positions[agent_id]
                robot_package = robot_packages[agent_id]
                
                # Normalized position
                state_features.extend([
                    robot_pos[0] / self.grid.shape[0],
                    robot_pos[1] / self.grid.shape[1]
                ])
                
                # Package carrying status
                state_features.append(1.0 if robot_package is not None else 0.0)
                
                # Distance to nearest package or delivery point
                if robot_package is None:
                    valid_package_positions = [pos for pos in package_positions if pos is not None]
                    if valid_package_positions:
                        nearest_package, min_dist = self.distance_calculator.get_nearest_package(
                            robot_pos,
                            valid_package_positions
                        )
                        state_features.append(min_dist / max_possible_distance)
                    else:
                        state_features.append(1.0)
                else:
                    delivery_point = delivery_points[robot_package]
                    dist = self.distance_calculator.get_distance(robot_pos, delivery_point)
                    state_features.append(dist / max_possible_distance)
                
                # 2. Package information
                state_features.extend([
                    waiting_packages / self.n_packages,
                    len(delivered_packages) / self.n_packages,
                    delivery_progress
                ])
                
                # 3. Global information
                state_features.extend([
                    normalized_robot_distance,
                    delivery_progress
                ])
                
                # Convert to numpy array
                processed_states[agent_id] = np.array(state_features, dtype=np.float32)
            
            logger.debug(f"Processed state for {len(processed_states)} agents")
            return processed_states
            
        except Exception as e:
            logger.error(f"Failed to process state: {str(e)}")
            raise

    def get_state_size(self) -> int:
        """Get the size of the state vector"""
        return self.state_size

    def get_observation_space(self) -> Dict:
        """Get observation space for each agent"""
        return {
            'state_size': self.state_size,
            'state_range': (0, 1),  # All features are normalized to [0,1]
            'state_dtype': np.float32
        }