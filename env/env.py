import numpy as np

class Robot: 
    """
    Class representing a robot in the environment.
    """
    def __init__(self, position): 
        self.position = position  # Current position (row, col)
        self.carrying = 0  # ID of package being carried (0 if not carrying)

class Package: 
    """
    Class representing a package to be delivered.
    """
    def __init__(self, start, start_time, target, deadline, package_id): 
        self.start = start  # Starting position (row, col)
        self.start_time = start_time  # Time when package becomes available
        self.target = target  # Target position (row, col)
        self.deadline = deadline  # Deadline for delivery
        self.package_id = package_id  # Unique ID of the package
        self.status = 'None'  # Status: 'waiting', 'in_transit', 'delivered'

class Environment: 
    """
    Environment class for the package delivery problem.
    Manages the grid map, robots, packages, and simulation state.
    """
    def __init__(self, map_file, max_time_steps=100, n_robots=5, n_packages=20,
             move_cost=-0.01, delivery_reward=10., delay_reward=1., 
             seed=2025): 
        """
        Initialize the simulation environment.
        
        Args:
            map_file: Path to the map text file
            max_time_steps: Maximum number of time steps
            n_robots: Number of robots
            n_packages: Number of packages
            move_cost: Cost incurred when a robot moves
            delivery_reward: Reward for delivering a package on time
            delay_reward: Reward for delivering a package late
            seed: Random seed for reproducibility
        """
        self.map_file = map_file
        self.grid = self.load_map()
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0]) if self.grid else 0 
        self.move_cost = move_cost 
        self.delivery_reward = delivery_reward 
        self.delay_reward = delay_reward
        self.t = 0 
        self.robots = []  # List of Robot objects
        self.packages = []  # List of Package objects
        self.total_reward = 0

        self.n_robots = n_robots
        self.max_time_steps = max_time_steps
        self.n_packages = n_packages

        self.rng = np.random.RandomState(seed)
        self.reset()
        self.done = False
        self.state = None

    def load_map(self):
        """
        Load the map from file.
        Returns a 2D grid where 0 indicates free cell and 1 indicates obstacle.
        """
        grid = []
        with open(self.map_file, 'r') as f:
            for line in f:
                row = [int(x) for x in line.strip().split(' ')]
                grid.append(row)
        return grid
    
    def is_free_cell(self, position):
        """
        Check if a cell is free (not an obstacle and within bounds).
        
        Args:
            position: (row, col) position to check
            
        Returns:
            bool: True if cell is free, False otherwise
        """
        r, c = position
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r][c] == 0

    def add_robot(self, position):
        """
        Add a robot at the given position if the cell is free.
        
        Args:
            position: (row, col) position to place robot
            
        Raises:
            ValueError: If position is invalid
        """
        if self.is_free_cell(position):
            robot = Robot(position)
            self.robots.append(robot)
        else:
            raise ValueError("Invalid robot position: must be on a free cell")

    def reset(self):
        """
        Reset the environment to initial state.
        Places robots and packages at random free positions.
        
        Returns:
            dict: Initial state of the environment
        """
        self.t = 0
        self.robots = []
        self.packages = []
        self.total_reward = 0
        self.done = False
        self.state = None

        # Place robots at random free positions
        tmp_grid = np.array(self.grid)
        for i in range(self.n_robots):
            position, tmp_grid = self.get_random_free_cell(tmp_grid)
            self.add_robot(position)
        
        # Generate packages
        N = self.n_rows
        list_packages = []
        for i in range(self.n_packages):
            # Random start and target positions
            start = self.get_random_free_cell_p()
            while True:
                target = self.get_random_free_cell_p()
                if start != target:
                    break
            
            # Set deadline and start time
            deadline = self.t + self.rng.randint(N/2, 3*N)
            if i <= min(self.n_robots, 20):
                start_time = 0
            else:
                start_time = self.rng.randint(1, self.max_time_steps)
            list_packages.append((start_time, start, target, deadline))

        # Sort packages by start time and create Package objects
        list_packages.sort(key=lambda x: x[0])
        for i in range(self.n_packages):
            start_time, start, target, deadline = list_packages[i]
            package_id = i+1
            self.packages.append(Package(start, start_time, target, deadline, package_id))

        return self.get_state()
    
    def get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            dict: State containing:
                - time_step: Current time step
                - map: Grid map
                - robots: List of robot positions and states
                - packages: List of packages that are available at current time
        """
        selected_packages = []
        for i in range(len(self.packages)):
            if self.packages[i].start_time == self.t:
                selected_packages.append(self.packages[i])
                self.packages[i].status = 'waiting'

        state = {
            'time_step': self.t,
            'map': self.grid,
            'robots': [(robot.position[0] + 1, robot.position[1] + 1,
                        robot.carrying) for robot in self.robots],
            'packages': [(package.package_id, package.start[0] + 1, package.start[1] + 1, 
                          package.target[0] + 1, package.target[1] + 1, package.start_time, package.deadline) 
                         for package in selected_packages]
        }
        return state

    def get_random_free_cell_p(self):
        """
        Get a random free cell position.
        
        Returns:
            tuple: (row, col) position of a free cell
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) \
                      if self.grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        return free_cells[i]

    def get_random_free_cell(self, new_grid):
        """
        Get a random free cell and mark it as occupied.
        
        Args:
            new_grid: Grid to mark cells as occupied
            
        Returns:
            tuple: (position, updated_grid)
        """
        free_cells = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) \
                      if new_grid[i][j] == 0]
        i = self.rng.randint(0, len(free_cells))
        new_grid[free_cells[i][0]][free_cells[i][1]] = 1
        return free_cells[i], new_grid

    def step(self, actions):
        """
        Advance the simulation by one time step.
        
        Args:
            actions: List of (move_action, package_action) for each robot
            
        Returns:
            tuple: (state, reward, done, info)
                - state: New state after actions
                - reward: Reward for this step
                - done: Whether episode is finished
                - info: Additional information
        """
        r = 0
        if len(actions) != len(self.robots):
            raise ValueError("Number of actions must match number of robots")

        # Process robot movements
        proposed_positions = []
        old_pos = {}
        next_pos = {}
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            new_pos = self.compute_new_position(robot.position, move)
            if not self.valid_position(new_pos):
                new_pos = robot.position
            proposed_positions.append(new_pos)
            old_pos[robot.position] = i
            next_pos[new_pos] = i

        # Resolve movement conflicts
        moved_robots = [0 for _ in range(len(self.robots))]
        computed_moved = [0 for _ in range(len(self.robots))]
        final_positions = [None] * len(self.robots)
        occupied = {}
        
        while True:
            updated = False
            for i in range(len(self.robots)):
                if computed_moved[i] != 0: 
                    continue

                pos = self.robots[i].position
                new_pos = proposed_positions[i]
                can_move = False
                
                if new_pos not in old_pos:
                    can_move = True
                else:
                    j = old_pos[new_pos]
                    if (j != i) and (computed_moved[j] == 0):
                        continue
                    can_move = True

                if can_move:
                    if new_pos not in occupied:
                        occupied[new_pos] = i
                        final_positions[i] = new_pos
                        computed_moved[i] = 1
                        moved_robots[i] = 1
                        updated = True
                    else:
                        new_pos = pos
                        occupied[new_pos] = i
                        final_positions[i] = pos
                        computed_moved[i] = 1
                        moved_robots[i] = 0
                        updated = True

                if updated:
                    break

            if not updated:
                break

        for i in range(len(self.robots)):
            if computed_moved[i] == 0:
                final_positions[i] = self.robots[i].position 
        
        # Update robot positions and apply movement cost
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            if move in ['L', 'R', 'U', 'D'] and final_positions[i] != robot.position:
                r += self.move_cost
            robot.position = final_positions[i]

        # Process package actions
        for i, robot in enumerate(self.robots):
            move, pkg_act = actions[i]
            
            # Pick up package
            if pkg_act == '1':
                if robot.carrying == 0:
                    for j in range(len(self.packages)):
                        if (self.packages[j].status == 'waiting' and 
                            self.packages[j].start == robot.position and 
                            self.packages[j].start_time <= self.t):
                            package_id = self.packages[j].package_id
                            robot.carrying = package_id
                            self.packages[j].status = 'in_transit'
                            break

            # Drop package
            elif pkg_act == '2':
                if robot.carrying != 0:
                    package_id = robot.carrying
                    target = self.packages[package_id - 1].target
                    if robot.position == target:
                        pkg = self.packages[package_id - 1]
                        pkg.status = 'delivered'
                        if self.t <= pkg.deadline:
                            r += self.delivery_reward
                        else:
                            r += self.delay_reward
                        robot.carrying = 0
        
        # Increment time step
        self.t += 1
        self.total_reward += r

        done = False
        infos = {}
        if self.check_terminate():
            done = True
            infos['total_reward'] = self.total_reward
            infos['total_time_steps'] = self.t

        return self.get_state(), r, done, infos
    
    def check_terminate(self):
        """
        Check if the episode should terminate.
        
        Returns:
            bool: True if episode should end
        """
        if self.t == self.max_time_steps:
            return True
        
        for p in self.packages:
            if p.status != 'delivered':
                return False
            
        return True

    def compute_new_position(self, position, move):
        """
        Compute new position based on current position and move.
        
        Args:
            position: Current (row, col) position
            move: Move action ('S', 'L', 'R', 'U', 'D')
            
        Returns:
            tuple: New (row, col) position
        """
        r, c = position
        if move == 'S':
            return (r, c)
        elif move == 'L':
            return (r, c - 1)
        elif move == 'R':
            return (r, c + 1)
        elif move == 'U':
            return (r - 1, c)
        elif move == 'D':
            return (r + 1, c)
        else:
            return (r, c)

    def valid_position(self, pos):
        """
        Check if a position is valid (within bounds and not an obstacle).
        
        Args:
            pos: (row, col) position to check
            
        Returns:
            bool: True if position is valid
        """
        r, c = pos
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        if self.grid[r][c] == 1:
            return False
        return True

    def render(self):
        """
        Render the current state of the environment.
        Shows obstacles (1), free cells (0), and robots (R0, R1, etc.)
        """
        grid_copy = [row[:] for row in self.grid]
        for i, robot in enumerate(self.robots):
            r, c = robot.position
            grid_copy[r][c] = 'R%i'%i
        for row in grid_copy:
            print('\t'.join(str(cell) for cell in row))

if __name__=="__main__":
    env = Environment('map.txt', 10, 2, 5)
    state = env.reset()
    print("Initial State:", state)
    print("Initial State:")
    env.render()

    # Agents
    # Initialize agents
    from greedyagent import GreedyAgents as Agents
    agents = Agents()   # You should define a default parameters here
    agents.init_agents(state) # You have a change to init states which can be used or not. Depend on your choice
    print("Agents initialized.")
    
    # Example actions for robots
    list_actions = ['S', 'L', 'R', 'U', 'D']
    n_robots = len(state['robots'])
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state) 
        state, reward, done, infos = env.step(actions)
    
        print("\nState after step:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Infos: {infos}")
        print("Total Reward:", env.total_reward)
        print("Time step:", env.t)
        print("Packages:", state['packages'])
        print("Robots:", state['robots'])

        # For debug purpose
        t += 1
        if t == 100:
            break
    