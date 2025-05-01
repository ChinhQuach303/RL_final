import numpy as np
from agent import Agents

class RandomAgent(Agents):
    """
    Random agent that selects actions randomly.
    This agent serves as a baseline for comparison.
    """
    def __init__(self):
        """
        Initialize the random agent.
        """
        super().__init__()

    def get_actions(self, state):
        """
        Get random actions for all robots.
        
        Args:
            state (dict): Current state of the environment
            
        Returns:
            list: List of random (move_action, package_action) tuples
        """
        if not self.is_init:
            self.init_agents(state)

        list_actions = ['S', 'L', 'R', 'U', 'D']
        actions = []
        for i in range(self.n_robots):
            move = np.random.randint(0, len(list_actions))
            pkg_act = np.random.randint(0, 3)
            actions.append((list_actions[move], str(pkg_act)))

        return actions

    def select_action(self, state):
        """Randomly select an action from the action space."""
        return self.action_space.sample()
    
    def train(self, *args, **kwargs):
        """Random agent doesn't need training."""
        pass 