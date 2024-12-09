# import os
# import logging
# import logging.config
# from datetime import datetime
# import robyn

# current_datetime = datetime.now()

# # Get the directory of the current module
# module_dir = os.path.dirname(robyn.__file__)

# log_directory = "/tmp/robynpy/logs"
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory)

# # Define the path to the logging configuration file
# logging_conf_path = os.path.join(module_dir, "common/config/logging.conf")

# # Create a default configuration if the file is missing or fails to load
# if not os.path.exists(logging_conf_path):
#     # Create a basic configuration
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     )
# else:
#     # Attempt to load the configuration file
#     try:
#         logging.config.fileConfig(
#             logging_conf_path,
#             defaults={"asctime": current_datetime.strftime("%Y-%m-%d_%H-%M-%S")},
#             disable_existing_loggers=False,
#         )
#     except KeyError:
#         # Handle KeyError by using basic configuration
#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         )

import os
import logging
from datetime import datetime
import robyn

current_datetime = datetime.now()

# Get the directory of the current module
module_dir = os.path.dirname(robyn.__file__)

# Set up basic logging configuration to log to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# If you have any specific loggers you want to configure, you can do so here
logger = logging.getLogger(__name__)
logger.info("Logging is set up to console only.")
