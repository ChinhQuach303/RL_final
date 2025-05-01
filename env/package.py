class Package:
    """
    Class representing a package to be delivered.
    A package has a start position, target position, and delivery deadline.
    """
    def __init__(self, start, start_time, target, deadline, package_id):
        """
        Initialize a package.
        
        Args:
            start: Tuple (row, col) representing the start position
            start_time: Time step when the package becomes available
            target: Tuple (row, col) representing the target position
            deadline: Time step by which the package should be delivered
            package_id: Unique ID of the package
        """
        self.start = start  # Starting position (row, col)
        self.start_time = start_time  # Time when package becomes available
        self.target = target  # Target position (row, col)
        self.deadline = deadline  # Deadline for delivery
        self.package_id = package_id  # Unique ID of the package
        self.status = 'None'  # Status: 'waiting', 'in_transit', 'delivered' 