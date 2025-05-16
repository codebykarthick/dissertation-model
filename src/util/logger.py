import logging
import os
from datetime import datetime

LOG_PATH = "logs"
log_file = os.path.join(
    LOG_PATH, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
os.makedirs(LOG_PATH, exist_ok=True)
logger = None


def setup_logger() -> logging.Logger:
    """Creates an instance of logger to be used across the project for uniformity. Writes to stdout and files

    Returns:
        logging.Logger: Logger to be used for logging information.
    """
    # Logger
    global logger
    if logger is None:
        logger = logging.getLogger("DissertationLogger")
        logger.setLevel(logging.INFO)

        # Formatting
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

    return logger
