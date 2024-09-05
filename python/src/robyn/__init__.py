import logging.config
import datetime

# Get the current datetime
current_datetime = datetime.datetime.now()

logging.config.fileConfig('logging.conf', defaults={'asctime': current_datetime.strftime('%Y-%m-%d_%H-%M-%S')})