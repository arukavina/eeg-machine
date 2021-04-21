#!/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------------------
Script:    System __init__ file. Sets up logging
------------------------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import logging
import sys
import logging.handlers
import os

# Own modules
from eeg_machine.util import file_utils

eeg_logger = logging.getLogger(__name__)


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
    log_file_hist = file_utils.generate_filename(name, '_hist.log', timestamp=timestamp)

    formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(process)d] [%(levelname)s] '
                                  '[%(name)s::%(funcName)s()] :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Handlers

    # Simple
    fh_path = os.path.join(log_path, log_file)
    file_handler = logging.FileHandler(fh_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Rotating
    fh_hist_path = os.path.join(log_path, log_file_hist)
    file_handler_hist = logging.handlers.RotatingFileHandler(fh_hist_path, 'a', 512 * 1024, backupCount=20)
    file_handler_hist.setFormatter(formatter)
    file_handler_hist.setLevel(logging.DEBUG)

    # Streaming
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(level)

    # Logger
    eeg_logger.addHandler(file_handler)
    eeg_logger.addHandler(file_handler_hist)
    eeg_logger.addHandler(std_handler)
    eeg_logger.propagate = False
    eeg_logger.setLevel(level)

    eeg_logger.info('Logging configured at project level!')
