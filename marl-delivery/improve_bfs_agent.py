# improved_bfs_agent.py
import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set
import heapq

class ImprovedBFS:
    def __init__(self, map_grid: np.ndarray):
        """
        Initialize improved BFS with collision avoidance and deadline awareness
        Args:
            map_grid: 2D numpy array representing the environment
        """
        self.map = map_grid
        self.n_rows = len(map_grid)
        self.n_cols = len(map_grid[0])
        self.robot_positions = set()  # Current positions of all robots
        self.package_deadlines = {}   # Package deadlines for priority calculation
        
    def set_robot_positions(self, positions: List[Tuple[int, int]]):
        """Update current robot positions for collision avoidance"""
        self.robot_positions = set(positions)
    
    def set_package_deadlines(self, deadlines: Dict[int, int]):
        """Update package deadlines for priority calculation"""
        self.package_deadlines = deadlines
    
    def get_neighbors(self, pos: Tuple[int, int], 
                     avoid_positions: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions with collision avoidance
        Args:
            pos: Current position
            avoid_positions: Set of positions to avoid (other robots)
        Returns:
            List of valid neighboring positions
        """
        if avoid_positions is None:
            avoid_positions = self.robot_positions
        
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            next_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= next_pos[0] < self.n_rows and 
                0 <= next_pos[1] < self.n_cols and 
                self.map[next_pos[0]][next_pos[1]] == 0 and
                next_pos not in avoid_positions):
                neighbors.append(next_pos)
        return neighbors
    
    def calculate_priority(self, package_id: int, current_time: int) -> float:
        """
        Calculate priority score for a package based on deadline and distance
        Returns higher priority for packages with:
        - Closer deadlines
        - Shorter distances
        - Higher urgency (less time remaining)
        """
        if package_id not in self.package_deadlines:
            return float('inf')
        
        deadline = self.package_deadlines[package_id]
        time_remaining = deadline - current_time
        
        # Higher priority for packages with less time remaining
        # But avoid negative priorities for expired deadlines
        return max(0, time_remaining)
    
    def find_path(self, 
                  start: Tuple[int, int], 
                  goal: Tuple[int, int],
                  current_time: int,
                  package_id: int = None,
                  other_robot_paths: Dict[int, List[Tuple[int, int]]] = None) -> Tuple[str, int, List[Tuple[int, int]]]:
        """
        Find path using improved BFS with:
        - Collision avoidance
        - Deadline awareness
        - Path optimization
        Args:
            start: Start position
            goal: Goal position
            current_time: Current time step
            package_id: ID of the package being delivered
            other_robot_paths: Dictionary of other robots' planned paths
        Returns:
            Tuple of (action, distance, path)
        """
        if other_robot_paths is None:
            other_robot_paths = {}
        
        # Priority queue for A* like behavior
        queue = []
        # (priority, distance, position, path, time_step)
        heapq.heappush(queue, (0, 0, start, [start], 0))
        
        visited = {start: 0}  # position -> distance
        best_path = None
        best_distance = float('inf')
        
        while queue:
            priority, dist, pos, path, time_step = heapq.heappop(queue)
            
            if pos == goal:
                if dist < best_distance:
                    best_distance = dist
                    best_path = path
                continue
            
            if dist > visited.get(pos, float('inf')):
                continue
            
            # Get positions to avoid (other robots' positions at this time step)
            avoid_positions = set()
            for robot_id, robot_path in other_robot_paths.items():
                if time_step < len(robot_path):
                    avoid_positions.add(robot_path[time_step])
            
            for next_pos in self.get_neighbors(pos, avoid_positions):
                new_dist = dist + 1
                
                # Calculate priority based on:
                # 1. Distance to goal
                # 2. Package deadline (if applicable)
                # 3. Collision risk
                priority = new_dist
                if package_id is not None:
                    priority += 1.0 / (1.0 + self.calculate_priority(package_id, current_time + time_step))
                
                # Add collision risk penalty
                collision_risk = sum(1 for p in avoid_positions 
                                   if abs(next_pos[0] - p[0]) + abs(next_pos[1] - p[1]) <= 1)
                priority += collision_risk * 0.5
                
                if next_pos not in visited or new_dist < visited[next_pos]:
                    visited[next_pos] = new_dist
                    heapq.heappush(queue, (priority, new_dist, next_pos, 
                                         path + [next_pos], time_step + 1))
        
        if best_path is None:
            return 'S', float('inf'), []
        
        # Convert path to action
        if len(best_path) < 2:
            return 'S', 0, best_path
        
        # Get the first move
        current = best_path[0]
        next_pos = best_path[1]
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx == -1:
            action = 'U'
        elif dx == 1:
            action = 'D'
        elif dy == -1:
            action = 'L'
        elif dy == 1:
            action = 'R'
        else:
            action = 'S'
        
        return action, best_distance, best_path

class ImprovedGreedyAgent:
    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.is_init = False
        self.bfs = None
        self.current_time = 0
        self.robot_paths = {}  # Store planned paths for each robot
        
    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.bfs = ImprovedBFS(self.map)
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]) 
                        for p in state['packages']]  # Added deadline
        self.packages_free = [True] * len(self.packages)
        
        # Initialize package deadlines
        deadlines = {p[0]: p[6] for p in self.packages}
        self.bfs.set_package_deadlines(deadlines)
    
    def update_move_to_target(self, robot_id: int, target_package_id: int, 
                            phase: str = 'start') -> Tuple[str, str]:
        """
        Update robot movement with improved path finding
        Args:
            robot_id: ID of the robot
            target_package_id: ID of the target package
            phase: 'start' for pickup or 'target' for delivery
        Returns:
            Tuple of (move_action, package_action)
        """
        robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        pkg = self.packages[target_package_id]
        
        # Set target position based on phase
        if phase == 'start':
            target_pos = (pkg[1], pkg[2])
        else:
            target_pos = (pkg[3], pkg[4])
        
        # Update robot positions for collision avoidance
        robot_positions = [(r[0], r[1]) for r in self.robots]
        self.bfs.set_robot_positions(robot_positions)
        
        # Get other robots' planned paths
        other_paths = {i: path for i, path in self.robot_paths.items() 
                      if i != robot_id}
        
        # Find path with collision avoidance
        move, distance, path = self.bfs.find_path(
            robot_pos, target_pos, self.current_time,
            pkg[0], other_paths
        )
        
        # Update planned path for this robot
        self.robot_paths[robot_id] = path
        
        # Determine package action
        pkg_act = '0'
        if distance == 0:
            if phase == 'start':
                pkg_act = '1'  # Pickup
            else:
                pkg_act = '2'  # Drop
        
        return move, pkg_act
    
    def update_inner_state(self, state):
        """Update internal state with new environment state"""
        self.current_time = state['time_step']
        
        # Update robot positions and states
        for i in range(len(state['robots'])):
            prev = (self.robots[i][0], self.robots[i][1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            
            if prev[2] != 0:
                if self.robots[i][2] == 0:
                    # Robot has dropped the package
                    self.robots_target[i] = 'free'
                    if i in self.robot_paths:
                        del self.robot_paths[i]
                else:
                    self.robots_target[i] = self.robots[i][2]
        
        # Update package information
        new_packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]) 
                       for p in state['packages']]
        self.packages.extend(new_packages)
        self.packages_free.extend([True] * len(new_packages))
        
        # Update deadlines
        deadlines = {p[0]: p[6] for p in self.packages}
        self.bfs.set_package_deadlines(deadlines)
    
    def get_actions(self, state):
        """Get actions for all robots using improved strategy"""
        if not self.is_init:
            self.is_init = True
            self.update_inner_state(state)
        else:
            self.update_inner_state(state)
        
        actions = []
        
        # First, update all robot positions for collision avoidance
        robot_positions = [(r[0], r[1]) for r in self.robots]
        self.bfs.set_robot_positions(robot_positions)
        
        # Assign packages to robots
        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                # Robot is already assigned to a package
                package_id = self.robots_target[i]
                if self.robots[i][2] != 0:
                    # Move to delivery point
                    move, action = self.update_move_to_target(i, package_id-1, 'target')
                else:
                    # Move to pickup point
                    move, action = self.update_move_to_target(i, package_id-1, 'start')
                actions.append((move, action))
            else:
                # Find best package to pick up
                best_package = None
                best_score = float('inf')
                
                for j, pkg in enumerate(self.packages):
                    if not self.packages_free[j]:
                        continue
                    
                    # Calculate package score based on:
                    # 1. Distance to robot
                    # 2. Time remaining until deadline
                    # 3. Distance to delivery point
                    robot_pos = (self.robots[i][0], self.robots[i][1])
                    pickup_pos = (pkg[1], pkg[2])
                    delivery_pos = (pkg[3], pkg[4])
                    
                    # Get distances
                    _, pickup_dist, _ = self.bfs.find_path(robot_pos, pickup_pos, 
                                                         self.current_time, pkg[0])
                    _, delivery_dist, _ = self.bfs.find_path(pickup_pos, delivery_pos,
                                                           self.current_time, pkg[0])
                    
                    # Calculate time pressure
                    time_remaining = pkg[6] - self.current_time
                    if time_remaining <= 0:
                        continue  # Skip expired packages
                    
                    # Calculate score (lower is better)
                    score = (pickup_dist + delivery_dist) / (1.0 + time_remaining)
                    
                    if score < best_score:
                        best_score = score
                        best_package = (j, pkg[0])
                
                if best_package is not None:
                    pkg_idx, pkg_id = best_package
                    self.packages_free[pkg_idx] = False
                    self.robots_target[i] = pkg_id
                    move, action = self.update_move_to_target(i, pkg_idx, 'start')
                    actions.append((move, action))
                else:
                    actions.append(('S', '0'))
        
        return actions