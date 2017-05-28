"""
------------------------------------------------------------------------------------------------------
Script:    Module to configure logging
------------------------------------------------------------------------------------------------------
"""

import sys
import logging
import logging.handlers
import os

from . import file_utils


def setup_logging(name, timestamp, file_components, optional_file_components, args):
    """
    Sets up the logger for the classification.
    :param name: Logger insance name
    :param timestamp: The timestamp to apply to the file
    :param file_components: mandatory file name dict
    :param optional_file_components: optional file name parts
    :param args: a dictionary with the arguments which are used by the classifier. This dict will be modified,
                 removing items which shouldn't be sent to the classification function.
    :return: None
    """

    log_dir = args['LOG_PATH']
    del args['LOG_PATH']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("Logging to {}/ named: {}".format(log_dir, name))

    log_file = file_utils.generate_filename('eeg-machine', '.log',
                                            components=file_components,
                                            optional_components=optional_file_components,
                                            timestamp=timestamp)

    log_file_hist = file_utils.generate_filename('eeg-machine', '_hist.log',
                                                 components=file_components,
                                                 optional_components=optional_file_components,
                                                 timestamp=timestamp)

    # Logger
    my_log = logging.getLogger(name)
    my_log.propagate = False
    my_log.handlers = []
    my_log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(process)d] [%(levelname)s] '
                                  '([%(filename)s::%(funcName)s) :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Handlers
    fh_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(fh_path)
    file_handler.setFormatter(formatter)
    """
    fh_hist_path = os.path.join(log_dir, log_file_hist)
    file_handler_hist = logging.handlers.RotatingFileHandler(fh_hist_path, 'a', 512 * 1024, backupCount=20)
    file_handler_hist.setLevel(logging.DEBUG)
    file_handler_hist.setFormatter(formatter)
    """
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)

    my_log.addHandler(file_handler)
    # my_log.addHandler(file_handler_hist)
    my_log.addHandler(std_handler)


def print_imports_versions(logger):
    """
    Prints on logger the information about the version of all the imported modules

    :param logger: Logging object to be used
    :return:
    """
    for name, module in sorted(sys.modules.items()):
        if hasattr(module, '__version__'):
            logger.info('{0} :: {1}'.format(name, module.__version__))
