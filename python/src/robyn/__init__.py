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
