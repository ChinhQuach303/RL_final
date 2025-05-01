import numpy as np
from agent import Agents

class GreedyAgent(Agents):
    """
    Greedy agent that selects actions based on immediate rewards.
    Uses BFS to find shortest paths to packages and delivery points.
    """
    def __init__(self):
        """
        Initialize the greedy agent.
        """
        super().__init__()
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []

    def init_agents(self, state):
        """
        Initialize the agent with the initial state.
        
        Args:
            state (dict): Initial state containing:
                - map: 2D grid representing the environment
                - robots: List of robot positions and states
                - packages: List of packages to be delivered
        """
        super().init_agents(state)
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def _run_bfs(self, start, goal):
        """
        Run Breadth-First Search to find the shortest path from start to goal.
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            
        Returns:
            tuple: (next_move, distance)
                - next_move: 'U', 'D', 'L', 'R', 'S' indicating the next move
                - distance: Distance to goal
        """
        n_rows = len(self.map)
        n_cols = len(self.map[0])

        queue = []
        visited = set()
        queue.append((goal, []))
        visited.add(goal)
        d = {}
        d[goal] = 0

        while queue:
            current, path = queue.pop(0)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if next_pos[0] < 0 or next_pos[0] >= n_rows or next_pos[1] < 0 or next_pos[1] >= n_cols:
                    continue
                if next_pos not in visited and self.map[next_pos[0]][next_pos[1]] == 0:
                    visited.add(next_pos)
                    d[next_pos] = d[current] + 1
                    queue.append((next_pos, path + [next_pos]))

        if start not in d:
            return 'S', float('inf')
        
        t = 0
        actions = ['U', 'D', 'L', 'R']
        current = start
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if next_pos in d:
                if d[next_pos] == d[current] - 1:
                    return actions[t], d[next_pos]
            t += 1
        return 'S', d[start]

    def _update_move_to_target(self, robot_id, target_package_id, phase='start'):
        """
        Update the movement action for a robot to reach its target.
        
        Args:
            robot_id: ID of the robot
            target_package_id: ID of the target package
            phase: 'start' for pickup or 'target' for delivery
            
        Returns:
            tuple: (move_action, package_action)
        """
        if phase == 'start':
            target_p = (self.packages[target_package_id][1], self.packages[target_package_id][2])
        else:
            target_p = (self.packages[target_package_id][3], self.packages[target_package_id][4])

        move, distance = self._run_bfs((self.robots[robot_id][0], self.robots[robot_id][1]), target_p)
        
        if distance == 0:
            pkg_act = '1' if phase == 'start' else '2'
        else:
            pkg_act = '0'

        return move, pkg_act

    def _update_inner_state(self, state):
        """
        Update the internal state of the agent based on the new environment state.
        
        Args:
            state (dict): New state of the environment
        """
        # Update robot positions and states
        for i in range(len(state['robots'])):
            prev = (self.robots[i][0], self.robots[i][1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            
            if prev[2] != 0:
                if self.robots[i][2] == 0:
                    self.robots_target[i] = 'free'
                else:
                    self.robots_target[i] = self.robots[i][2]
        
        # Update package positions and states
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def get_actions(self, state):
        """
        Get actions for all robots using a greedy strategy.
        
        Args:
            state (dict): Current state of the environment
            
        Returns:
            list: List of (move_action, package_action) tuples
        """
        if not self.is_init:
            self.init_agents(state)
        else:
            self._update_inner_state(state)

        actions = []
        
        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                # Robot is assigned to a package
                closest_package_id = self.robots_target[i]
                if self.robots[i][2] != 0:
                    # Move to delivery point
                    move, action = self._update_move_to_target(i, closest_package_id-1, 'target')
                else:
                    # Move to pickup point
                    move, action = self._update_move_to_target(i, closest_package_id-1)
                actions.append((move, action))
            else:
                # Find closest available package
                closest_package_id = None
                min_distance = float('inf')
                
                for j in range(len(self.packages)):
                    if not self.packages_free[j]:
                        continue

                    pkg = self.packages[j]                
                    d = abs(pkg[1]-self.robots[i][0]) + abs(pkg[2]-self.robots[i][1])
                    if d < min_distance:
                        min_distance = d
                        closest_package_id = pkg[0]

                if closest_package_id is not None:
                    self.packages_free[closest_package_id-1] = False
                    self.robots_target[i] = closest_package_id
                    move, action = self._update_move_to_target(i, closest_package_id-1)
                    actions.append((move, action))
                else:
                    actions.append(('S', '0'))

        return actions

    def train(self, *args, **kwargs):
        """Greedy agent doesn't need training."""
        pass 