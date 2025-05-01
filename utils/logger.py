import json
import os
from datetime import datetime

class Logger:
    """Simple logger that saves metrics to a JSON file."""
    
    def __init__(self, name):
        self.name = name
        self.logs = []
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'{name}_{timestamp}.json')
    
    def log(self, metrics):
        """Log a dictionary of metrics."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.logs.append(metrics)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_logs(self):
        """Return all logged metrics."""
        return self.logs 