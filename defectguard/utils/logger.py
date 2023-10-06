import os
import logging

# Create a folder named 'logs' if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger instance
logger = logging.getLogger('my_logger')

# Define a file handler to log to a file in the 'logs' folder
log_file_path = os.path.join('logs', 'logs.log')
file_handler = logging.FileHandler(log_file_path)

# Optionally, you can set the logging level for the file handler
file_handler.setLevel(logging.DEBUG)

# Define a formatter for the file handler (if needed)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

## Log some messages
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')