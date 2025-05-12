# train_marl.py
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_env import MARLEnvironment, MARLTrainer
from dqn_agent import DQNAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self):
        # Environment parameters
        self.map_file = 'map1.txt'
        self.max_robots = 2
        self.max_packages = 3
        self.max_time_steps = 200
        
        # Reward parameters
        self.move_cost = -0.05
        self.delivery_reward = 20.0
        self.timeout_penalty = -2.0
        self.collision_penalty = -1.0
        self.waiting_penalty = -0.02
        self.distance_reward_factor = 0.2
        
        # Training parameters
        self.n_episodes = 1000
        self.max_steps = 200
        self.save_freq = 100
        self.eval_freq = 20
        self.eval_episodes = 10
        
        # Agent parameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.memory_size = 100000
        self.batch_size = 128
        self.target_update_freq = 1000
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        self.checkpoint_dir = 'checkpoints'
        self.plot_dir = 'plots'
        self.results_dir = 'results'
        
        # Create necessary directories
        for dir_path in [self.checkpoint_dir, self.plot_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from file"""
        config = cls()
        with open(filepath, 'r') as f:
            config.__dict__.update(json.load(f))
        return config

def analyze_map(map_file: str) -> Tuple[int, int]:
    """
    Analyze map to determine maximum number of robots and packages
    Args:
        map_file: Path to map file
    Returns:
        Tuple of (max_robots, max_packages)
    """
    try:
        # Read map file
        with open(map_file, 'r') as f:
            grid = [list(map(int, line.strip().split())) for line in f]
        grid = np.array(grid)
        
        # Count valid positions (non-obstacle positions)
        valid_positions = np.sum(grid == 0)  # Assuming 0 represents valid positions
        
        # Calculate maximum number of robots and packages
        max_total = max(1, valid_positions // 3)  # Ensure at least 1
        max_robots = max(1, max_total // 2)  # Ensure at least 1 robot
        max_packages = max(1, max_total // 2)  # Ensure at least 1 package
        
        logger.info(f"Map analysis:")
        logger.info(f"Grid size: {grid.shape}")
        logger.info(f"Valid positions: {valid_positions}")
        logger.info(f"Maximum robots: {max_robots}")
        logger.info(f"Maximum packages: {max_packages}")
        
        return int(max_robots), int(max_packages)
        
    except Exception as e:
        logger.error(f"Error analyzing map: {str(e)}")
        return 2, 3  # Default values if analysis fails

def create_agents(env: MARLEnvironment, config: TrainingConfig) -> Dict[str, DQNAgent]:
    """
    Create DQN agents for each robot
    Args:
        env: MARL environment
        config: Training configuration
    Returns:
        Dictionary mapping agent IDs to DQNAgent instances
    """
    agents = {}
    for i in range(env.n_robots):
        agent_id = f'agent_{i}'
        agents[agent_id] = DQNAgent(
            state_processor=env.state_processor,
            reward_calculator=env.reward_calculator,
            action_size=env.get_action_space(),
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            epsilon_decay=config.epsilon_decay,
            memory_size=config.memory_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            device=config.device
        )
    return agents

def plot_training_results(stats: Dict, config: TrainingConfig):
    """
    Plot and save training results
    Args:
        stats: Training statistics
        config: Training configuration
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(stats['training_rewards'], label='Training')
    plt.plot(stats['eval_rewards'], label='Evaluation')
    plt.title('Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot delivery rates
    plt.subplot(2, 2, 2)
    plt.plot(stats['training_delivery_rates'], label='Training')
    plt.plot(stats['eval_delivery_rates'], label='Evaluation')
    plt.title('Delivery Rates over Time')
    plt.xlabel('Episode')
    plt.ylabel('Delivery Rate')
    plt.legend()
    
    # Plot average rewards
    plt.subplot(2, 2, 3)
    plt.bar(['Training', 'Evaluation'], 
            [stats['avg_training_reward'], stats['avg_eval_reward']])
    plt.title('Average Rewards')
    plt.ylabel('Reward')
    
    # Plot average delivery rates
    plt.subplot(2, 2, 4)
    plt.bar(['Training', 'Evaluation'], 
            [stats['avg_training_delivery_rate'], stats['avg_eval_delivery_rate']])
    plt.title('Average Delivery Rates')
    plt.ylabel('Delivery Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, f'training_results_{timestamp}.png'))
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load or create configuration
    config = TrainingConfig()
    config_path = os.path.join(config.results_dir, 'training_config.json')
    if os.path.exists(config_path):
        config = TrainingConfig.load(config_path)
    else:
        config.save(config_path)
    
    # Analyze map
    max_robots, max_packages = analyze_map(config.map_file)
    
    # Use smaller numbers than maximum
    n_robots = min(config.max_robots, max_robots)
    n_packages = min(config.max_packages, max_packages)
    
    logger.info(f"\nUsing configuration:")
    logger.info(f"Number of robots: {n_robots}")
    logger.info(f"Number of packages: {n_packages}")
    
    try:
        # Initialize environment
        env = MARLEnvironment(
            map_file=config.map_file,
            n_robots=n_robots,
            n_packages=n_packages,
            max_time_steps=config.max_time_steps,
            move_cost=config.move_cost,
            delivery_reward=config.delivery_reward,
            timeout_penalty=config.timeout_penalty,
            collision_penalty=config.collision_penalty,
            waiting_penalty=config.waiting_penalty,
            distance_reward_factor=config.distance_reward_factor
        )
        
        # Create agents
        agents = create_agents(env, config)
        
        # Initialize trainer
        trainer = MARLTrainer(
            env=env,
            agents=agents,
            n_episodes=config.n_episodes,
            max_steps=config.max_steps,
            save_freq=config.save_freq,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes
        )
        
        # Train agents
        logger.info("\nStarting training...")
        trainer.train()
        
        # Save final models
        logger.info("\nSaving final models...")
        final_checkpoint_dir = os.path.join(config.checkpoint_dir, 'final')
        trainer.save_models(final_checkpoint_dir)
        
        # Get and save training statistics
        stats = trainer.get_training_stats()
        stats_path = os.path.join(config.results_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Plot results
        plot_training_results(stats, config)
        
        # Print final statistics
        logger.info("\nFinal Training Statistics:")
        logger.info(f"Average Training Reward: {stats['avg_training_reward']:.2f}")
        logger.info(f"Average Evaluation Reward: {stats['avg_eval_reward']:.2f}")
        logger.info(f"Average Training Delivery Rate: {stats['avg_training_delivery_rate']:.2%}")
        logger.info(f"Average Evaluation Delivery Rate: {stats['avg_eval_delivery_rate']:.2%}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Saving current models...")
        trainer.save_models(os.path.join(config.checkpoint_dir, 'interrupted'))
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        trainer.save_models(os.path.join(config.checkpoint_dir, 'error'))
        raise
    finally:
        # Clean up
        logger.info("\nCleaning up...")
        for agent in agents.values():
            agent.reset_noise()  # Reset noise in noisy layers

if __name__ == "__main__":
    main()