import numpy as np

def run_bfs(map, start, goal):
    """
    Run Breadth-First Search to find the shortest path from start to goal.
    
    Args:
        map: 2D grid representing the environment
        start: Starting position (row, col)
        goal: Goal position (row, col)
        
    Returns:
        tuple: (next_move, distance)
            - next_move: 'U', 'D', 'L', 'R', 'S' indicating the next move
            - distance: Distance to goal
    """
    n_rows = len(map)
    n_cols = len(map[0])

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
            if next_pos not in visited and map[next_pos[0]][next_pos[1]] == 0:
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))

    if start not in d:
        return 'S', 100000
    
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

class GreedyAgents:
    """
    Greedy agent implementation that uses BFS to find shortest paths and makes
    locally optimal decisions for package delivery.
    """
    def __init__(self):
        """
        Initialize the greedy agent with empty state.
        """
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.is_init = False

    def init_agents(self, state):
        """
        Initialize the agent with the initial state.
        
        Args:
            state (dict): Initial state containing:
                - map: 2D grid representing the environment
                - robots: List of robot positions and states
                - packages: List of packages to be delivered
        """
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):
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
            distance = abs(self.packages[target_package_id][1]-self.robots[robot_id][0]) + \
            abs(self.packages[target_package_id][2]-self.robots[robot_id][1])
        else:
            distance = abs(self.packages[target_package_id][3]-self.robots[robot_id][0]) + \
            abs(self.packages[target_package_id][4]-self.robots[robot_id][1])

        move = 'S'
        pkg_act = '0'
        
        if distance >= 1:
            pkg = self.packages[target_package_id]
            target_p = (pkg[1], pkg[2]) if phase == 'start' else (pkg[3], pkg[4])
            move, distance = run_bfs(self.map, (self.robots[robot_id][0], self.robots[robot_id][1]), target_p)

            if distance == 0:
                pkg_act = '1' if phase == 'start' else '2'
        else:
            pkg_act = '1' if phase == 'start' else '2'

        return move, pkg_act
    
    def update_inner_state(self, state):
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
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])    

    def get_actions(self, state):
        """
        Get actions for all robots based on the current state using a greedy strategy.
        
        Args:
            state (dict): Current state of the environment
            
        Returns:
            list: List of tuples (move_action, package_action) for each robot
        """
        if not self.is_init:
            self.is_init = True
            self.update_inner_state(state)
        else:
            self.update_inner_state(state)

        actions = []
        
        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                # Robot is assigned to a package
                closest_package_id = self.robots_target[i]
                if self.robots[i][2] != 0:
                    # Move to delivery point
                    move, action = self.update_move_to_target(i, closest_package_id-1, 'target')
                else:
                    # Move to pickup point
                    move, action = self.update_move_to_target(i, closest_package_id-1)
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
                    move, action = self.update_move_to_target(i, closest_package_id-1)
                    actions.append((move, action))
                else:
                    actions.append(('S', '0'))

        return actions
