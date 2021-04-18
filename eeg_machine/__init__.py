# Built-in/Generic Imports
import logging
import sys
import logging.handlers
import os

# Own modules
from src.util import file_utils

PKG_LOGGER = logging.getLogger(__name__)


def setup_logging(name, timestamp, level=logging.DEBUG, log_path=r'./log'):
    """
    Sets up the logger for the classification.
    :param name: Logger instance name
    :param timestamp: The timestamp to apply to the file
    :param level: Logging level from logging
    :param log_path: Logging path
    :return: None
    """

    msg_format = '%(asctime)s [%(levelname)8s] %(message)s (%(name)s - %(filename)s:%(lineno)s)'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=msg_format, datefmt=date_format)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    PKG_LOGGER.addHandler(console_handler)
    PKG_LOGGER.setLevel(logging.DEBUG)
    PKG_LOGGER.propagate = False
    PKG_LOGGER.info('finished logging setup!')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print("Logging to {}/ named: {}".format(log_path, name))

    log_file = file_utils.generate_filename(name, '.log', timestamp=timestamp)

    # log_file_hist = file_utils.generate_filename(name, '_hist.log',
    #                                              components=file_components,
    #                                              optional_components=optional_file_components,
    #                                              timestamp=timestamp)

    formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(process)d] [%(levelname)s] '
                                  '([%(filename)s::%(funcName)s) :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Handlers
    fh_path = os.path.join(log_path, log_file)

    file_handler = logging.FileHandler(fh_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # fh_hist_path = os.path.join(log_dir, log_file_hist)
    # file_handler_hist = logging.handlers.RotatingFileHandler(fh_hist_path, 'a', 512 * 1024, backupCount=20)
    # file_handler_hist.setLevel(logging.DEBUG)
    # file_handler_hist.setFormatter(formatter)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(level)

    # Logger
    PKG_LOGGER.addHandler(file_handler)
    # my_log.addHandler(file_handler_hist)
    PKG_LOGGER.addHandler(std_handler)
    PKG_LOGGER.propagate = False
    PKG_LOGGER.handlers = []
    PKG_LOGGER.setLevel(level)


def print_imports_versions():
    """
    Prints on logger the information about the version of all the imported modules

    :param logger: Logging object to be used
    :return:
    """
    for name, module in sorted(sys.modules.items()):
        if hasattr(module, '__version__'):
            PKG_LOGGER.info('{0} :: {1}'.format(name, module.__version__))