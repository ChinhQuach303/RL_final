import numpy as np
from abc import ABC, abstractmethod

class Agents(ABC):
    """
    Base class for all agent implementations in the package delivery environment.
    This class provides the basic interface that all agent implementations must follow.
    """
    def __init__(self):
        """
        Initialize the agent with empty state.
        """
        self.agents = []
        self.n_robots = 0
        self.state = None
        self.map = None
        self.is_init = False

    def init_agents(self, state):
        """
        Initialize the agent with the initial state of the environment.
        
        Args:
            state (dict): Initial state containing:
                - map: 2D grid representing the environment
                - robots: List of robot positions and states
                - packages: List of packages to be delivered
        """
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.is_init = True

    @abstractmethod
    def get_actions(self, state):
        """
        Get actions for all robots based on the current state.
        
        Args:
            state (dict): Current state of the environment
            
        Returns:
            list: List of tuples (move_action, package_action) for each robot
                - move_action: 'S' (stay), 'L' (left), 'R' (right), 'U' (up), 'D' (down)
                - package_action: '0' (no action), '1' (pickup), '2' (drop)
        """
        pass

    def train(self, *args, **kwargs):
        """
        Optional training method for agents that learn.
        Default implementation does nothing.
        """
        pass
