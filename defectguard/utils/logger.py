import os
from icecream import ic
from datetime import datetime

# Create a folder named 'logs' if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Define a file to log IceCream output
log_file_path = os.path.join('logs', 'logs.log')

# Replace logging configuration with IceCream configuration
# ic.configureOutput(prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ', outputFunction=open(log_file_path, 'a').write)
ic.configureOutput(prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ')