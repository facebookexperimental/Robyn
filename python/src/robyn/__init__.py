import datetime
import logging.config
import os

# Get the current datetime
current_datetime = datetime.datetime.now()

logging_conf_path = os.path.join(
    os.path.dirname(__file__), "common/config/logging.conf"
)

# Load the logging configuration file
logging.config.fileConfig(
    logging_conf_path,
    defaults={"asctime": current_datetime.strftime("%Y-%m-%d_%H-%M-%S")},
)
