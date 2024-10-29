import os
import logging
import logging.config
from datetime import datetime

current_datetime = datetime.now()

# Get the directory of the current module
module_dir = os.path.dirname(robyn.__file__)

# Define the path to the logging configuration file
logging_conf_path = os.path.join(module_dir, "common/config/logging.conf")

# Create a default configuration if the file is missing or fails to load
if not os.path.exists(logging_conf_path):
    # Create a basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
else:
    # Attempt to load the configuration file
    try:
        logging.config.fileConfig(
            logging_conf_path,
            defaults={"asctime": current_datetime.strftime("%Y-%m-%d_%H-%M-%S")},
            disable_existing_loggers=False,
        )
    except KeyError:
        # Handle KeyError by using basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )