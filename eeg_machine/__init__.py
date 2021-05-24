#!/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------------------
Script:    System __init__ file. Sets up logging
------------------------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import logging
import logging.handlers
from logging.handlers import QueueHandler, QueueListener
import sys
import os
import time
import datetime
import random
import multiprocessing

# Own modules
from eeg_machine.util import file_utils

eeg_logger = logging.getLogger(__name__)


def f(i):
    time.sleep(random.uniform(.01, .05))
    eeg_logger.info('function called with in worker thread.')
    eeg_logger.debug('function called with in worker thread.')
    time.sleep(random.uniform(.01, .05))
    return i


def worker_init(q):
    # All records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


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

    formatter = logging.Formatter('%(asctime)s [%(threadName)s-%(processName)s] [%(levelname)s] '
                                  '[%(name)s::%(funcName)s()] :: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Handlers

    # Simple
    fh_path = os.path.join(log_path, log_file)
    file_handler = logging.FileHandler(fh_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Rotating
    fh_hist_path = os.path.join(log_path, log_file_hist)
    file_handler_hist = logging.handlers.RotatingFileHandler(fh_hist_path, 'a',  100 * 512 * 1024, backupCount=20)
    file_handler_hist.setFormatter(formatter)
    file_handler_hist.setLevel(logging.DEBUG)

    # Streaming
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(level)

    # Logger Config
    eeg_logger.propagate = False
    eeg_logger.setLevel(logging.DEBUG)

    # Adding Handlers
    eeg_logger.addHandler(file_handler)
    eeg_logger.addHandler(file_handler_hist)
    eeg_logger.addHandler(std_handler)

    q = multiprocessing.Queue()
    # Variable ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, std_handler, file_handler, file_handler_hist)
    ql.start()

    # Add the handler to the logger so records from this process are handled
    eeg_logger.addHandler(std_handler)

    eeg_logger.info('Logging configured at project level!')

    return q, ql


def main():

    ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    q, ql = setup_logging('eeg-init-test',
                          timestamp=ts,
                          level=logging.INFO)

    eeg_logger.info('Main thread info. Should appear everywhere')
    eeg_logger.debug('Main thread debug. Should only appear in log files')
    pool = multiprocessing.Pool(4, worker_init, [q])
    for _ in pool.map(f, range(10)):
        pass
    pool.close()
    pool.join()
    ql.stop()


if __name__ == '__main__':
    main()
