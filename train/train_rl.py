import numpy as np
import torch
from env.delivery_env import DeliveryEnv
from agents.rl_agent import DQNAgent
import argparse
from utils.logger import Logger

def train(env, agent, num_episodes, batch_size=32, target_update=10):
    """Train the DQN agent."""
    logger = Logger('dqn_training')
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            agent.train(batch_size)
            
            state = next_state
            
        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Log progress
        logger.log({
            'episode': episode,
            'total_reward': total_reward,
            'epsilon': agent.epsilon
        })
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--map', type=str, default='maps/map1.txt',
                      help='path to the map file')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='batch size for training')
    parser.add_argument('--target-update', type=int, default=10,
                      help='episodes between target network updates')
    args = parser.parse_args()
    
    # Create environment and agent
    env = DeliveryEnv(args.map)
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_dim, env.action_space)
    
    # Train the agent
    train(env, agent, args.episodes, args.batch_size, args.target_update)
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'models/dqn_agent.pth')

if __name__ == '__main__':
    main() 