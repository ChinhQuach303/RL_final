# Delivery Robot Environment

This project implements a delivery robot environment where agents need to learn to pick up and deliver packages efficiently. The environment follows the OpenAI Gym interface and includes multiple types of agents: random, greedy, and DQN (Deep Q-Network).

## Project Structure

```
project/
├── agents/
│   ├── random_agent.py    # Random action agent
│   ├── greedy_agent.py    # Greedy action selection agent
│   └── rl_agent.py        # DQN/PPO reinforcement learning agent
├── env/
│   ├── delivery_env.py    # Main environment (OpenAI Gym compatible)
│   ├── robot.py          # Robot class
│   └── package.py        # Package class
├── train/
│   ├── train_rl.py       # Training script for RL agents
│   └── evaluate.py       # Evaluation script
├── utils/
│   ├── logger.py         # Logging utilities
│   └── visualizer.py     # Visualization utilities
├── maps/
│   └── map1.txt         # Example map file
└── main.py              # Main script to run the environment
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running with Different Agents

```bash
# Run with random agent
python main.py --agent random

# Run with greedy agent
python main.py --agent greedy

# Run with trained DQN agent
python main.py --agent dqn --model-path models/dqn_agent.pth
```

### Training a DQN Agent

```bash
python train/train_rl.py --episodes 1000 --batch-size 32
```

### Map Format

Maps are text files where:
- '#' represents walls
- '.' represents empty spaces
- The robot and packages will be placed randomly in empty spaces

## Environment Details

- Action Space: 5 discrete actions (up, right, down, left, pickup/deliver)
- Observation Space: Robot position, package status, and positions
- Rewards:
  - -1 per step (time penalty)
  - +10 for picking up a package
  - +20 for delivering a package
  - +50 bonus for completing all deliveries

## Agents

1. **Random Agent**: Takes random actions from the action space
2. **Greedy Agent**: Takes actions that give the highest immediate reward
3. **DQN Agent**: Uses Deep Q-Learning to learn optimal actions

## Visualization

The environment includes visualization tools for:
- Real-time environment rendering
- Training metrics plotting
- Episode playback

## License

This project is licensed under the MIT License. 