import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agent import Agents

class DQN(nn.Module):
    """
    Deep Q-Network for learning action values.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent(Agents):
    """
    Deep Q-Learning agent for package delivery.
    Uses a neural network to learn optimal actions.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
    def get_actions(self, state):
        """
        Get actions for all robots using epsilon-greedy policy.
        
        Args:
            state (dict): Current state of the environment
            
        Returns:
            list: List of (move_action, package_action) tuples
        """
        if not self.is_init:
            self.init_agents(state)

        actions = []
        state_tensor = self._preprocess_state(state)
        
        for i in range(self.n_robots):
            if random.random() < self.epsilon:
                # Random action
                move = random.choice(['S', 'L', 'R', 'U', 'D'])
                pkg_act = str(random.randint(0, 2))
            else:
                # Greedy action
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action_idx = q_values.argmax().item()
                    move = ['S', 'L', 'R', 'U', 'D'][action_idx // 3]
                    pkg_act = str(action_idx % 3)
            
            actions.append((move, pkg_act))
            
        return actions
    
    def train(self, batch_size=32):
        """
        Train the DQN on a batch of experiences.
        
        Args:
            batch_size: Size of training batch
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _preprocess_state(self, state):
        """
        Preprocess state into tensor format.
        
        Args:
            state: Environment state
            
        Returns:
            torch.Tensor: Preprocessed state
        """
        # Convert state to tensor format
        # This is a placeholder - you'll need to implement proper state preprocessing
        state_tensor = torch.zeros(self.state_dim)
        return state_tensor.to(self.device) 