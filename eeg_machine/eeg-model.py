#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running model training.
"""

# Libs
import datetime
import logging

# Own
import src
from src import classification_pipeline
from src import hills, wavelets, cross_correlate
from src import log_utils as lu
from src.util import settings_util as su
from src.util import file_utils as fu

__author__ = 'Andrei Rukavina'
__copyright__ = '2021, EGG-Machine'
__credits__ = ['Andrei Rukavina']
__license__ = 'GPL'
__version__ = '0.3.0'
__email__ = 'rukavina.andrei@gmail.com'
__status__ = 'dev'


def train_model(settings, fh):
    """
    Trains a SVM classifier using the features selected by the  *settings* dictionary.
    When fitted, the model will automatically be used to assign scores and a submission file will be generated in the
    folder given by 'SUBMISSION_PATH' in the *settings* dictionary.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :param fh: File Handler
    :return: None. The model will be pickled to a file in the corresponding subject feature folder. A submission file
             will be written to the folder given by 'SUBMISSION_PATH' in the settings dictionary.
    """
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    eeg_logger = logging.getLogger(src.get_logger_name())
    eeg_logger.info("Starting training with the following arguments: {}".format(settings))

    classification_pipeline.run_batch_classification(feature_folders=[settings['FEATURE_PATH']],
                                                     timestamp=timestamp,
                                                     scores_file=settings['SUBMISSION_PATH'],
                                                     frame_length=12,
                                                     feature_type=settings['FEATURE_TYPE'],
                                                     processes=settings['WORKERS'],
                                                     do_standardize=True,
                                                     no_crossvalidation=True,
                                                     rebuild_model=True,
                                                     method='svm',
                                                     model_params={'C': 500, 'gamma': 0},
                                                     file_handler=fh)


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

    eeg_logger.info("Training model")
    train_model(settings, fh)


if __name__ == '__main__':
    main()
