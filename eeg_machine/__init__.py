# Built-in/Generic Imports
import logging
import sys
import logging.handlers
import os

# Own modules
from eeg_machine.util import file_utils

EEG_LOGGER = logging.getLogger(__name__)


def setup_logging(name, timestamp, level=logging.DEBUG, log_path=r'./../logs'):
    """
    Sets up the logger for the classification.
    :param name: Logger instance name
    :param timestamp: The timestamp to apply to the file
    :param level: Logging level from logging
    :param log_path: Logging path
    :return: None
    """

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print("Logging to {}/ named: {}".format(log_path, name))

    log_file = file_utils.generate_filename(name, '.log', timestamp=timestamp)

    # log_file_hist = file_utils.generate_filename(name, '_hist.log',
    #                                              components=file_components,
    #                                              optional_components=optional_file_components,
    #                                              timestamp=timestamp)

    # formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(process)d] [%(levelname)s] '
    #                              '([%(filename)s|%(name)s::%(funcName)s) :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(process)d] [%(levelname)s] '
                                  '[%(name)s::%(funcName)s()] :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

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
    EEG_LOGGER.addHandler(file_handler)
    # my_log.addHandler(file_handler_hist)
    EEG_LOGGER.addHandler(std_handler)
    EEG_LOGGER.propagate = False
    EEG_LOGGER.setLevel(level)

    EEG_LOGGER.info('Logging configured at project level!')


def print_imports_versions():
    """
    Prints on logger the information about the version of all the imported modules

    :return: None
    """
    for name, module in sorted(sys.modules.items()):
        if hasattr(module, '__version__'):
            EEG_LOGGER.info('{0} :: {1}'.format(name, module.__version__))