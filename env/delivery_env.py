import gym
from gym import spaces
import numpy as np
from env import Environment

class DeliveryEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for the package delivery problem.
    This class provides a standard Gym interface for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, map_file, max_time_steps=100, n_robots=5, n_packages=20,
                 move_cost=-0.01, delivery_reward=10., delay_reward=1., 
                 seed=2025):
        """
        Initialize the delivery environment.
        
        Args:
            map_file: Path to the map text file
            max_time_steps: Maximum number of time steps
            n_robots: Number of robots
            n_packages: Number of packages
            move_cost: Cost incurred when a robot moves
            delivery_reward: Reward for delivering a package on time
            delay_reward: Reward for delivering a package late
            seed: Random seed for reproducibility
        """
        super(DeliveryEnv, self).__init__()
        
        self.env = Environment(map_file, max_time_steps, n_robots, n_packages,
                             move_cost, delivery_reward, delay_reward, seed)
        self.n_robots = n_robots
        
        # Action space: each robot has 15 possible actions (5 moves * 3 package actions)
        self.action_space = spaces.MultiDiscrete([15] * self.n_robots)
        
        # Observation space: time_step + robot states + package states
        # Each robot: position (2) + carrying (1)
        # Each package: start (2) + target (2) + deadline (1)
        obs_dim = 1 + 3 * n_robots + 5 * n_packages
        self.observation_space = spaces.Box(
            low=0, 
            high=1000, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        state = self.env.reset()
        return self._encode_observation(state)

    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Args:
            actions: List of action indices for each robot
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        decoded_actions = self._decode_actions(actions)
        state, reward, done, info = self.env.step(decoded_actions)
        obs = self._encode_observation(state)
        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
        """
        self.env.render()

    def _encode_observation(self, state):
        """
        Encode the environment state into a numpy array.
        
        Args:
            state: Dictionary containing environment state
            
        Returns:
            np.ndarray: Encoded observation
        """
        obs = []

        # Add current time step
        obs.append(state['time_step'])

        # Encode robot states: (x, y, carrying)
        robots = state['robots']
        for i in range(self.n_robots):
            if i < len(robots):
                r = robots[i]
                obs.extend([r[0], r[1], r[2]])
            else:
                obs.extend([0, 0, 0])  # Padding for missing robots

        # Encode package states: (sx, sy, tx, ty, deadline)
        packages = state['packages']
        for i in range(self.env.n_packages):
            if i < len(packages):
                pkg = packages[i]
                obs.extend([pkg[1], pkg[2], pkg[3], pkg[4], pkg[6]])
            else:
                obs.extend([0, 0, 0, 0, 0])  # Padding for missing packages

        return np.array(obs, dtype=np.float32)

    def _decode_actions(self, actions):
        """
        Decode action indices into (move, package_action) tuples.
        
        Args:
            actions: List of action indices
            
        Returns:
            list: List of (move, package_action) tuples
        """
        move_map = ['S', 'L', 'R', 'U', 'D']
        pkg_map = ['0', '1', '2']
        decoded = []
        for a in actions:
            move = move_map[a // 3]
            pkg = pkg_map[a % 3]
            decoded.append((move, pkg))
        return decoded
