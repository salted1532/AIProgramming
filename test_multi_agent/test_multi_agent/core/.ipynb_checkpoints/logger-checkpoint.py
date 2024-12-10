#logger.py

import logging
from datetime import datetime
from typing import Any, Dict

class Logger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('medical_ai_agents.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_input(self, task_type: str, input_data: Any):
        self.logger.info(f"Task: {task_type} - Input received at {datetime.now()}")

    def log_output(self, task_type: str, result: Dict[str, Any], validation: Dict[str, bool]):
        self.logger.info(f"Task: {task_type} - Output generated at {datetime.now()}")
        self.logger.info(f"Validation result: {validation['is_valid']}")

    def log_error(self, task_type: str, error_message: str):
        self.logger.error(f"Task: {task_type} - Error: {error_message}") 