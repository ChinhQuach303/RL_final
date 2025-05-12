# utils/reward_utils.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from distance_utils import DistanceCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reward_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, 
                 grid: np.ndarray,
                 max_time_steps: int,
                 move_cost: float = -0.1,
                 delivery_reward: float = 10.0,
                 timeout_penalty: float = -5.0,
                 collision_penalty: float = -2.0,
                 waiting_penalty: float = -0.05,
                 distance_reward_factor: float = 0.1):
        """
        Initialize reward calculator
        Args:
            grid: 2D numpy array representing the environment
            max_time_steps: Maximum time steps allowed
            move_cost: Cost for each movement action
            delivery_reward: Reward for successful delivery
            timeout_penalty: Penalty for timeout
            collision_penalty: Penalty for collision
            waiting_penalty: Penalty for waiting without package
            distance_reward_factor: Factor for distance-based rewards
        """
        try:
            logger.info("Initializing RewardCalculator...")
            
            # Validate inputs
            if not isinstance(grid, np.ndarray):
                raise TypeError(f"grid must be a numpy array, got {type(grid)}")
            if grid.ndim != 2:
                raise ValueError(f"grid must be 2D, got {grid.ndim}D")
            if not isinstance(max_time_steps, int) or max_time_steps <= 0:
                raise ValueError(f"max_time_steps must be a positive integer, got {max_time_steps}")
            
            self.grid = grid
            self.max_time_steps = max_time_steps
            self.distance_calculator = DistanceCalculator(grid)
            
            # Reward parameters
            self.move_cost = float(move_cost)
            self.delivery_reward = float(delivery_reward)
            self.timeout_penalty = float(timeout_penalty)
            self.collision_penalty = float(collision_penalty)
            self.waiting_penalty = float(waiting_penalty)
            self.distance_reward_factor = float(distance_reward_factor)
            
            logger.info("RewardCalculator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RewardCalculator: {str(e)}")
            raise

    def calculate_reward(self,
                        state: Dict,
                        next_state: Dict,
                        action: int,
                        agent_id: str,
                        time_step: int) -> float:
        """
        Calculate reward for an agent's action
        Args:
            state: Current state
            next_state: Next state after action
            action: Action taken (0: up, 1: right, 2: down, 3: left, 4: pickup, 5: deliver)
            agent_id: ID of the agent
            time_step: Current time step
        Returns:
            float: Calculated reward
        """
        try:
            logger.debug(f"Calculating reward for agent {agent_id} at time step {time_step}")
            
            # Initialize reward
            reward = 0.0
            
            # Validate inputs
            if agent_id not in state['robot_positions']:
                raise ValueError(f"Invalid agent_id: {agent_id}")
            if not isinstance(action, int) or action not in range(6):
                raise ValueError(f"Invalid action: {action}")
            if not isinstance(time_step, int) or time_step < 0:
                raise ValueError(f"Invalid time_step: {time_step}")
            
            # Get agent's current and next positions
            current_pos = state['robot_positions'][agent_id]
            next_pos = next_state['robot_positions'][agent_id]
            
            # Get agent's current and next package status
            current_package = state['robot_packages'][agent_id]
            next_package = next_state['robot_packages'][agent_id]
            
            # Check for successful delivery
            if current_package is not None and next_package is None:
                logger.debug(f"Agent {agent_id} delivered package {current_package}")
                reward += self.delivery_reward
                
                # Additional reward for on-time delivery
                if time_step <= self.max_time_steps:
                    reward += self.delivery_reward * 0.5
            
            # Check for collision
            if self._check_collision(next_pos, next_state['robot_positions']):
                logger.debug(f"Agent {agent_id} collided at position {next_pos}")
                reward += self.collision_penalty
            
            # Movement cost
            if action < 4:  # Movement actions
                reward += self.move_cost
                
                # Additional reward for moving towards package/delivery point
                if current_package is not None:
                    # Moving towards delivery point
                    delivery_point = next_state['delivery_points'][current_package]
                    current_dist = self.distance_calculator.get_distance(current_pos, delivery_point)
                    next_dist = self.distance_calculator.get_distance(next_pos, delivery_point)
                    if next_dist < current_dist:
                        reward += self.distance_reward_factor
                else:
                    # Moving towards nearest package
                    valid_package_positions = [pos for pos in next_state['package_positions'] 
                                            if pos is not None]
                    if valid_package_positions:
                        nearest_package, current_dist = self.distance_calculator.get_nearest_package(
                            current_pos,
                            valid_package_positions
                        )
                        _, next_dist = self.distance_calculator.get_nearest_package(
                            next_pos,
                            valid_package_positions
                        )
                        if next_dist < current_dist:
                            reward += self.distance_reward_factor
            
            # Penalty for waiting
            if action == 4 and current_package is None:  # Waiting without package
                reward += self.waiting_penalty
            
            # Timeout penalty
            if time_step >= self.max_time_steps:
                reward += self.timeout_penalty
            
            logger.debug(f"Calculated reward for agent {agent_id}: {reward}")
            return reward
            
        except Exception as e:
            logger.error(f"Failed to calculate reward: {str(e)}")
            raise

    def _check_collision(self, pos: Tuple[int, int], robot_positions: Dict[str, Tuple[int, int]]) -> bool:
        """Check if position collides with other robots"""
        try:
            return pos in robot_positions.values()
        except Exception as e:
            logger.error(f"Failed to check collision at position {pos}: {str(e)}")
            return True  # Assume collision if check fails