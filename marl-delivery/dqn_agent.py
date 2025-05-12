# agents/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Dict
import math
from state_utils import StateProcessor
from reward_utils import RewardCalculator

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network with Noisy Layers"""
    def __init__(self, input_size: int, output_size: int):
        super(DuelingDQNNetwork, self).__init__()
        
        # Feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        # Reset noise in value stream
        for module in self.value_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise in advantage stream
        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        if self.size < batch_size:
            return None, None, None, None, None, None

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.size

class DQNAgent:
    def __init__(self,
                 state_processor: StateProcessor,
                 reward_calculator: RewardCalculator,
                 action_size: int,
                 learning_rate: float = 0.0001,  # Reduced learning rate
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 100000,  # Increased memory size
                 batch_size: int = 128,      # Increased batch size
                 target_update_freq: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize DQN agent with advanced features
        """
        self.state_processor = state_processor
        self.reward_calculator = reward_calculator
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(memory_size)  # Using prioritized replay
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.target_update_freq = target_update_freq
        
        # Create main and target networks
        self.model = DuelingDQNNetwork(
            input_size=self.state_processor.get_state_size(),
            output_size=self.action_size
        ).to(device)
        
        self.target_model = DuelingDQNNetwork(
            input_size=self.state_processor.get_state_size(),
            output_size=self.action_size
        ).to(device)
        
        # Use Adam optimizer with gradient clipping
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()
        
        self.update_counter = 0
        self.training_step = 0

    def update_target_model(self):
        """Soft update target network"""
        tau = 0.005  # Soft update parameter
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using noisy network (no epsilon-greedy needed)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def reset_noise(self):
        """Reset noise in both main and target networks"""
        try:
            self.model.reset_noise()
            self.target_model.reset_noise()
        except Exception as e:
            logger.warning(f"Error resetting noise: {str(e)}")
            # Continue execution even if noise reset fails
            pass

    def replay(self):
        """Train on a batch of experiences with prioritized replay"""
        if len(self.memory) < self.batch_size:
            return None

        try:
            # Sample batch with priorities
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            if states is None:
                return None

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)

            # Double DQN: Use main network for action selection
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_model(next_states).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

            # Get current Q-values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Compute TD errors for priority update
            td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors + 1e-6)

            # Compute weighted loss
            loss = (weights.unsqueeze(1) * nn.MSELoss(reduction='none')(current_q_values, target_q_values)).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.update_target_model()

            # Reset noise in noisy layers
            self.reset_noise()

            self.training_step += 1
            return loss.item()

        except Exception as e:
            logger.error(f"Error in replay: {str(e)}")
            return None

    def save(self, filepath: str):
        """Save model weights and optimizer state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath: str):
        """Load model weights and optimizer state"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.update_target_model()

class MARLEnvironment:
    def __init__(self,
                 env,
                 n_agents: int,
                 state_processor: StateProcessor,
                 reward_calculator: RewardCalculator,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize MARL environment
        Args:
            env: Original environment
            n_agents: Number of agents
            state_processor: State processor
            reward_calculator: Reward calculator
            device: Device to run the networks on
        """
        self.env = env
        self.n_agents = n_agents
        self.state_processor = state_processor
        self.reward_calculator = reward_calculator
        self.device = device
        
        # Create agents
        self.agents = [
            DQNAgent(
                state_processor=state_processor,
                reward_calculator=reward_calculator,
                action_size=15,  # 5 moves * 3 package actions
                device=device
            )
            for _ in range(n_agents)
        ]
    
    def train(self, episodes: int, max_steps: int, save_freq: int = 100):
        """
        Train agents
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            save_freq: Frequency of saving models
        """
        for episode in range(episodes):
            state = self.env.reset()
            processed_state = self.state_processor.process_state(state)
            total_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Get actions from all agents
                actions = []
                for agent in self.agents:
                    action = agent.act(processed_state)
                    # Convert action index to (move_action, package_action)
                    move_idx = action // 3
                    pkg_idx = action % 3
                    move_action = ['S', 'L', 'R', 'U', 'D'][move_idx]
                    pkg_action = ['0', '1', '2'][pkg_idx]
                    actions.append((move_action, pkg_action))
                
                # Take actions in environment
                next_state, reward, done = self.env.step(actions)
                processed_next_state = self.state_processor.process_state(next_state)
                
                # Calculate individual rewards and update agents
                for i, agent in enumerate(self.agents):
                    agent_reward = self.reward_calculator.calculate_reward(
                        state, next_state, actions[i], i
                    )
                    agent.remember(
                        processed_state, actions[i], agent_reward,
                        processed_next_state, done
                    )
                    loss = agent.replay()
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Update state and total reward
                state = next_state
                processed_state = processed_next_state
                total_reward += reward
                
                if done:
                    break
            
            # Print episode summary
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"Episode: {episode+1}/{episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps: {step+1}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epsilon: {self.agents[0].epsilon:.3f}")
            print("-------------------")
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                self.save_agents(f"checkpoints/episode_{episode+1}")
    
    def save_agents(self, directory: str):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(f"{directory}/agent_{i}.pt")
    
    def load_agents(self, directory: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{directory}/agent_{i}.pt")
    
    def evaluate(self, episodes: int = 10, max_steps: int = 1000) -> float:
        """
        Evaluate agents without exploration
        Args:
            episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
        Returns:
            float: Average reward over evaluation episodes
        """
        total_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            processed_state = self.state_processor.process_state(state)
            episode_reward = 0
            
            for step in range(max_steps):
                # Get actions from all agents (no exploration)
                actions = []
                for agent in self.agents:
                    action = agent.act(processed_state, training=False)
                    move_idx = action // 3
                    pkg_idx = action % 3
                    move_action = ['S', 'L', 'R', 'U', 'D'][move_idx]
                    pkg_action = ['0', '1', '2'][pkg_idx]
                    actions.append((move_action, pkg_action))
                
                # Take actions in environment
                next_state, reward, done = self.env.step(actions)
                processed_next_state = self.state_processor.process_state(next_state)
                
                # Update state and reward
                state = next_state
                processed_state = processed_next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode+1}/{episodes}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Steps: {step+1}")
            print("-------------------")
        
        return np.mean(total_rewards)