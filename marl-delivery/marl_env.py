# marl_env.py
import numpy as np
import torch
import logging
import os
from typing import Dict, List, Tuple, Optional
from env import Environment
from state_utils import StateProcessor
from reward_utils import RewardCalculator
from dqn_agent import DQNAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marl_env.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MARLEnvironment:
    def __init__(self,
                 map_file: str,
                 n_robots: int,
                 n_packages: int,
                 max_time_steps: int,
                 move_cost: float = -0.1,
                 delivery_reward: float = 10.0,
                 timeout_penalty: float = -5.0,
                 collision_penalty: float = -2.0,
                 waiting_penalty: float = -0.05,
                 distance_reward_factor: float = 0.1):
        """
        Initialize MARL environment
        Args:
            map_file: Path to map file
            n_robots: Number of robots
            n_packages: Number of packages
            max_time_steps: Maximum time steps per episode
            move_cost: Cost for each movement action
            delivery_reward: Reward for successful delivery
            timeout_penalty: Penalty for timeout
            collision_penalty: Penalty for collision
            waiting_penalty: Penalty for waiting without package
            distance_reward_factor: Factor for distance-based rewards
        """
        try:
            logger.info("Initializing MARL Environment...")
            
            # Validate input parameters
            if not isinstance(map_file, str):
                raise TypeError(f"map_file must be a string, got {type(map_file)}")
            if not os.path.exists(map_file):
                raise FileNotFoundError(f"Map file not found: {map_file}")
            if not isinstance(n_robots, int) or n_robots <= 0:
                raise ValueError(f"n_robots must be a positive integer, got {n_robots}")
            if not isinstance(n_packages, int) or n_packages <= 0:
                raise ValueError(f"n_packages must be a positive integer, got {n_packages}")
            if not isinstance(max_time_steps, int) or max_time_steps <= 0:
                raise ValueError(f"max_time_steps must be a positive integer, got {max_time_steps}")
            
            # Initialize base environment
            logger.info("Initializing base environment...")
            try:
                self.env = Environment(map_file)
            except Exception as e:
                logger.error(f"Failed to initialize base environment: {str(e)}")
                raise
            
            # Convert grid to numpy array
            logger.info("Converting grid to numpy array...")
            try:
                self.env.grid = np.array(self.env.grid, dtype=np.int32)
                logger.info(f"Grid shape: {self.env.grid.shape}")
            except Exception as e:
                logger.error(f"Failed to convert grid to numpy array: {str(e)}")
                raise
            
            # Store parameters
            self.n_robots = n_robots
            self.n_packages = n_packages
            self.max_time_steps = max_time_steps
            self.time_step = 0
            
            # Initialize state processor
            logger.info("Initializing state processor...")
            try:
                self.state_processor = StateProcessor(
                    grid=self.env.grid,
                    n_robots=n_robots,
                    n_packages=n_packages
                )
            except Exception as e:
                logger.error(f"Failed to initialize state processor: {str(e)}")
                raise
            
            # Initialize reward calculator
            logger.info("Initializing reward calculator...")
            try:
                self.reward_calculator = RewardCalculator(
                    grid=self.env.grid,
                    max_time_steps=max_time_steps,
                    move_cost=move_cost,
                    delivery_reward=delivery_reward,
                    timeout_penalty=timeout_penalty,
                    collision_penalty=collision_penalty,
                    waiting_penalty=waiting_penalty,
                    distance_reward_factor=distance_reward_factor
                )
            except Exception as e:
                logger.error(f"Failed to initialize reward calculator: {str(e)}")
                raise
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Initialize environment state
            logger.info("Resetting environment...")
            try:
                self._reset()  # Using private method for initialization
            except Exception as e:
                logger.error(f"Failed to reset environment: {str(e)}")
                raise
            
            logger.info("MARL Environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MARL Environment: {str(e)}")
            raise

    def _reset(self) -> None:
        """
        Private method to reset environment state
        This is called during initialization and by the public reset method
        """
        logger.info("Resetting environment state...")
        
        # Reset base environment
        try:
            self.env.reset()
        except Exception as e:
            logger.error(f"Failed to reset base environment: {str(e)}")
            raise
        
        self.time_step = 0
        
        # Initialize containers
        self.robot_positions = {}
        self.robot_packages = {}
        self.package_positions = []
        self.delivery_points = []
        self.delivered_packages = []
        
        # Get valid positions
        try:
            valid_positions = self._get_valid_positions()
            logger.info(f"Found {len(valid_positions)} valid positions")
        except Exception as e:
            logger.error(f"Failed to get valid positions: {str(e)}")
            raise
        
        # Validate number of positions
        required_positions = self.n_robots + 2 * self.n_packages
        if len(valid_positions) < required_positions:
            error_msg = f"Not enough valid positions. Need {required_positions}, got {len(valid_positions)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Place robots
        try:
            robot_positions = np.random.choice(len(valid_positions), self.n_robots, replace=False)
            for i in range(self.n_robots):
                agent_id = f'agent_{i}'
                self.robot_positions[agent_id] = valid_positions[robot_positions[i]]
                self.robot_packages[agent_id] = None
            logger.info(f"Placed {self.n_robots} robots")
        except Exception as e:
            logger.error(f"Failed to place robots: {str(e)}")
            raise
        
        # Place packages and delivery points
        try:
            remaining_positions = [pos for pos in valid_positions if pos not in self.robot_positions.values()]
            package_positions = np.random.choice(len(remaining_positions), self.n_packages, replace=False)
            delivery_positions = np.random.choice(len(remaining_positions), self.n_packages, replace=False)
            
            for i in range(self.n_packages):
                self.package_positions.append(remaining_positions[package_positions[i]])
                self.delivery_points.append(remaining_positions[delivery_positions[i]])
            logger.info(f"Placed {self.n_packages} packages and delivery points")
        except Exception as e:
            logger.error(f"Failed to place packages and delivery points: {str(e)}")
            raise

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Public method to reset environment
        Returns:
            Tuple of (processed_states, info)
        """
        logger.info("Resetting environment...")
        try:
            # Reset environment state
            self._reset()
            
            # Create initial state
            initial_state = self._create_state()
            
            # Process state for each agent
            processed_states = self.state_processor.process_state(initial_state)
            
            # Create info dictionary
            info = {
                'delivery_rate': 0.0,
                'on_time_rate': 0.0,
                'successful_deliveries': 0,
                'time_step': self.time_step
            }
            
            logger.info("Environment reset successfully")
            return processed_states, info
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {str(e)}")
            raise

    def _get_valid_positions(self) -> List[Tuple[int, int]]:
        """Get list of valid positions (not obstacles)"""
        try:
            valid_positions = []
            for i in range(self.env.grid.shape[0]):
                for j in range(self.env.grid.shape[1]):
                    if self.env.grid[i][j] != 1:  # 1 represents obstacle
                        valid_positions.append((i, j))
            return valid_positions
        except Exception as e:
            logger.error(f"Failed to get valid positions: {str(e)}")
            raise

    def _create_state(self) -> Dict:
        """Create state dictionary from current environment state"""
        try:
            return {
                'robot_positions': self.robot_positions.copy(),
                'robot_packages': self.robot_packages.copy(),
                'package_positions': self.package_positions.copy(),
                'delivery_points': self.delivery_points.copy(),
                'delivered_packages': self.delivered_packages.copy(),
                'grid': self.env.grid.copy()
            }
        except Exception as e:
            logger.error(f"Failed to create state: {str(e)}")
            raise

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)"""
        i, j = pos
        return (0 <= i < self.env.grid.shape[0] and 
                0 <= j < self.env.grid.shape[1] and 
                self.env.grid[i][j] != 1)
    
    def _get_next_position(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next position based on action"""
        i, j = pos
        if action == 0:    # Up
            return (i-1, j)
        elif action == 1:  # Right
            return (i, j+1)
        elif action == 2:  # Down
            return (i+1, j)
        elif action == 3:  # Left
            return (i, j-1)
        return pos  # Stay in place for invalid actions
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Take a step in the environment
        Args:
            actions: Dictionary of agent actions
        Returns:
            Tuple of (processed_next_states, rewards, done, info)
        """
        # Update time step
        self.time_step += 1
        
        # Get current state
        current_state = self._create_state()
        
        # Execute actions for each agent
        for agent_id, action in actions.items():
            if agent_id not in self.robot_positions:
                raise ValueError(f"Invalid agent_id: {agent_id}")
            
            robot_pos = self.robot_positions[agent_id]
            robot_package = self.robot_packages[agent_id]
            
            # Execute action
            if action < 4:  # Movement actions
                next_pos = self._get_next_position(robot_pos, action)
                if self._is_valid_position(next_pos):
                    self.robot_positions[agent_id] = next_pos
            elif action == 4:  # Pickup action
                if robot_package is None:  # Robot doesn't have package
                    for package_id, package_pos in enumerate(self.package_positions):
                        if (package_pos == robot_pos and 
                            package_id not in self.robot_packages.values() and 
                            package_id not in self.delivered_packages):
                            self.robot_packages[agent_id] = package_id
                            self.package_positions[package_id] = None
                            break
            elif action == 5:  # Deliver action
                if robot_package is not None:  # Robot has package
                    delivery_point = self.delivery_points[robot_package]
                    if robot_pos == delivery_point:
                        self.robot_packages[agent_id] = None
                        self.delivered_packages.append(robot_package)
        
        # Get next state and process it
        next_state = self._create_state()
        processed_next_states = self.state_processor.process_state(next_state)  # Process the state
        
        # Calculate rewards for each agent
        rewards = {}
        for agent_id in actions.keys():
            agent_reward = self.reward_calculator.calculate_reward(
                state=current_state,
                next_state=next_state,
                action=actions[agent_id],
                agent_id=agent_id,
                time_step=self.time_step
            )
            rewards[agent_id] = agent_reward
        
        # Check if episode is done
        done = (self.time_step >= self.max_time_steps or 
                len(self.delivered_packages) == self.n_packages)
        
        # Calculate delivery rate and on-time rate
        delivery_rate = len(self.delivered_packages) / self.n_packages
        on_time_rate = sum(1 for p in self.delivered_packages 
                          if p in self.delivered_packages[:self.time_step]) / self.n_packages
        
        # Create info dictionary
        info = {
            'delivery_rate': delivery_rate,
            'on_time_rate': on_time_rate,
            'successful_deliveries': len(self.delivered_packages),
            'time_step': self.time_step
        }
        
        return processed_next_states, rewards, done, info  # Return processed states
    
    def get_state(self) -> Dict:
        """Get current state of the environment"""
        return self._create_state()
    
    def get_observation_space(self) -> Dict:
        """Get observation space for each agent"""
        return self.state_processor.get_observation_space()
    
    def get_action_space(self) -> int:
        """Get action space size"""
        return 6  # 4 movement actions + pickup + deliver
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for all episodes"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_deliveries': self.successful_deliveries,
            'on_time_deliveries': self.on_time_deliveries,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_delivery_rate': np.mean(self.successful_deliveries) / self.n_packages 
                               if self.successful_deliveries else 0,
            'avg_on_time_rate': np.mean(self.on_time_deliveries) / 
                              np.mean(self.successful_deliveries) 
                              if self.successful_deliveries else 0
        }
    
    def render(self):
        """Render current state of environment"""
        self.env.render()

class MARLTrainer:
    def __init__(self,
                 env: MARLEnvironment,
                 agents: Dict[str, 'DQNAgent'],
                 n_episodes: int,
                 max_steps: int,
                 save_freq: int = 100,
                 eval_freq: int = 10,
                 eval_episodes: int = 5):
        """
        Trainer for MARL environment
        Args:
            env: MARL environment
            agents: Dictionary of DQN agents
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            save_freq: Frequency of saving models
            eval_freq: Frequency of evaluation
            eval_episodes: Number of episodes for evaluation
        """
        self.env = env
        self.agents = agents
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        
        # Training tracking
        self.training_rewards = []
        self.eval_rewards = []
        self.training_delivery_rates = []
        self.eval_delivery_rates = []
    
    def train(self):
        """Train agents in environment"""
        for episode in range(self.n_episodes):
            states, info = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                # Get actions from all agents
                actions = {}
                for agent_id, agent in self.agents.items():
                    actions[agent_id] = agent.act(states[agent_id])
                
                # Take step in environment using MARLEnvironment's step method
                next_states, rewards, done, info = self.env.step(actions)  # This is correct now
                
                # Update agents
                for agent_id, agent in self.agents.items():
                    agent.remember(
                        states[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_states[agent_id],
                        done
                    )
                    agent.replay()
                
                # Update states and reward
                states = next_states
                episode_reward += sum(rewards.values())
                
                if done:
                    break
            
            # Track training progress
            self.training_rewards.append(episode_reward)
            self.training_delivery_rates.append(
                info['successful_deliveries'] / self.env.n_packages
            )
            
            # Print episode summary
            print(f"\nEpisode {episode+1}/{self.n_episodes}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Steps: {step+1}")
            print(f"Delivery Rate: {info['delivery_rate']:.2%}")
            print(f"On-time Rate: {info['on_time_rate']:.2%}")
            
            # Evaluate periodically
            if (episode + 1) % self.eval_freq == 0:
                eval_reward, eval_delivery_rate = self.evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_delivery_rates.append(eval_delivery_rate)
                print(f"\nEvaluation after episode {episode+1}")
                print(f"Average Reward: {eval_reward:.2f}")
                print(f"Average Delivery Rate: {eval_delivery_rate:.2%}")
            
            # Save models periodically
            if (episode + 1) % self.save_freq == 0:
                self.save_models(f"checkpoints/episode_{episode+1}")
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate agents
        Returns:
            Tuple of (average reward, average delivery rate)
        """
        eval_rewards = []
        eval_delivery_rates = []
        
        for _ in range(self.eval_episodes):
            states, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                # Get actions from all agents (no exploration)
                actions = {}
                for agent_id, agent in self.agents.items():
                    actions[agent_id] = agent.act(states[agent_id], training=False)
                
                # Take step in environment using MARLEnvironment's step method
                next_states, rewards, done, info = self.env.step(actions)  # This is correct now
                
                # Update states and reward
                states = next_states
                episode_reward += sum(rewards.values())
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_delivery_rates.append(
                info['successful_deliveries'] / self.env.n_packages
            )
        
        return np.mean(eval_rewards), np.mean(eval_delivery_rates)
    
    def save_models(self, directory: str):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        for agent_id, agent in self.agents.items():
            agent.save(f"{directory}/{agent_id}.pt")
    
    def load_models(self, directory: str):
        """Load all agent models"""
        for agent_id, agent in self.agents.items():
            agent.load(f"{directory}/{agent_id}.pt")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'training_delivery_rates': self.training_delivery_rates,
            'eval_delivery_rates': self.eval_delivery_rates,
            'avg_training_reward': np.mean(self.training_rewards),
            'avg_eval_reward': np.mean(self.eval_rewards),
            'avg_training_delivery_rate': np.mean(self.training_delivery_rates),
            'avg_eval_delivery_rate': np.mean(self.eval_delivery_rates)
        }