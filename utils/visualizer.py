import matplotlib.pyplot as plt
import numpy as np
import json

class Visualizer:
    """Utility for visualizing the environment and training metrics."""
    
    @staticmethod
    def plot_training_metrics(log_file):
        """Plot training metrics from a log file."""
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        episodes = [log['episode'] for log in logs]
        rewards = [log['total_reward'] for log in logs]
        epsilons = [log['epsilon'] for log in logs]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot total rewards
        ax1.plot(episodes, rewards, 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True)
        
        # Plot epsilon values
        ax2.plot(episodes, epsilons, 'r-')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate (Epsilon)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_metrics.png')
        plt.close()
    
    @staticmethod
    def visualize_episode(env, agent, max_steps=100):
        """Visualize an episode with the current agent."""
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Clear the console
            print('\033[2J')
            
            # Render the current state
            env.render()
            
            # Get action from agent
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            step += 1
            
            # Add a small delay for visualization
            plt.pause(0.5)
        
        print(f"Episode finished after {step} steps") 