class Robot:
    """
    Class representing a robot in the environment.
    A robot can move around and carry one package at a time.
    """
    def __init__(self, position):
        """
        Initialize a robot.
        
        Args:
            position: Tuple (row, col) representing the robot's position
        """
        self.position = position  # Current position (row, col)
        self.carrying = 0  # ID of package being carried (0 if not carrying)
    
    def move_to(self, new_position):
        """
        Move the robot to a new position.
        
        Args:
            new_position: Tuple (row, col) representing the new position
        """
        self.position = new_position
    
    def pick_up_package(self, package_id):
        """
        Pick up a package.
        
        Args:
            package_id: ID of the package to pick up
            
        Returns:
            bool: True if package was picked up successfully
        """
        if self.carrying == 0:
            self.carrying = package_id
            return True
        return False
    
    def deliver_package(self):
        """
        Deliver the currently held package.
        
        Returns:
            bool: True if package was delivered successfully
        """
        if self.carrying != 0:
            self.carrying = 0
            return True
        return False 