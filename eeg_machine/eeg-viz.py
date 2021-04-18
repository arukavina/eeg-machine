#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running visualizations of segments and basic statistics.
"""

# Libs
import datetime
import logging

# Own
import eeg-machine
from src.classification import classification_pipeline
from src.features import hills, wavelets, cross_correlate
from src.util import log_utils as lu
from src.util import settings_util as su
from src.util import file_utils as fu

__author__ = 'Andrei Rukavina'
__copyright__ = '2021, EGG-Machine'
__credits__ = ['Andrei Rukavina']
__license__ = 'GPL'
__version__ = '0.3.0'
__email__ = 'rukavina.andrei@gmail.com'
__status__ = 'dev'

# TODO: Finish this from Viz adaptation


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the SETTINGS.json to use for the training")

    args = vars(parser.parse_args())
    if args is None:
        print("Using default root SETTINGS.json location")
        args['settings'] = r'SETTINGS.json'

    settings = su.get_settings(args['settings'])

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    file_components = [settings['FEATURE_TYPE'],
                       'svm',
                       'frame_length_{}'.format(12)]
    optional_file_components = {'standardized': True}

    # Setup logging stuff, this removes 'log_dir' from the dictionary
    lu.setup_logging(src.get_logger_name(), timestamp, file_components, optional_file_components, settings['LOG_LEVEL'],
                     settings['LOG_PATH'])
    eeg_logger = logging.getLogger(src.get_logger_name())

    fh_args_dict = dict([
        ('train_path', settings['TRAIN_DATA_PATH']),
        ('test_path', settings['TEST_DATA_PATH']),
        ('cat_column', settings['CAT_COLUMN']),
        ('class_labels', settings['CLASS_LABELS']),
        ('logger', eeg_logger)
    ])

    try:
        fh = fu.FileHelper(**fh_args_dict)
    except AttributeError:
        raise AttributeError('Attribute error when trying to instantiate class. Check __init__ or __doc__')
    except Exception as e:
        raise AttributeError('Something else is really wrong: {}'.format(e))

    eeg_logger.info("Viz")


if __name__ == '__main__':
    main()
