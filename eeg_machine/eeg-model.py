#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running model training.
"""

# Libs
import datetime
import logging

# Own
from eeg_machine import setup_logging
from eeg_machine.classification import classification_pipeline
from eeg_machine.util import settings_util as su
from eeg_machine.util import file_utils as fu

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

    :param settings: A dictionary with settings. Usually created from the json file 'FEATURE_SETTINGS.json' in the project root
                     directory.
    :param fh: File Handler
    :return: None. The model will be pickled to a file in the corresponding subject feature folder. A submission file
             will be written to the folder given by 'SUBMISSION_PATH' in the settings dictionary.
    """
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    train_segment_paths = settings['TRAIN_DATA_PATH']
    score_path = settings['SCORE_PATH']

    feature_type = settings['FEATURE_TYPE']
    frame_length = settings['FEATURE_SETTINGS']['FRAME_LENGTH']

    # Model Params as *kwargs
    model_method = settings['MODEL']
    model_params = settings['MODEL_PARAMS']

    # Training
    do_standardize = settings['DO_STANDARDIZE']
    no_crossvalidation = settings['NO_CROSSVALIDATION']
    rebuild_model = settings['REBUILD_MODEL']

    workers = settings['WORKERS']

    eeg_logger = logging.getLogger('eeg_machine.main')
    eeg_logger.info("Starting training with the following arguments: {}".format(settings))

    classification_pipeline.run_batch_classification(feature_folders=[train_segment_paths],
                                                     timestamp=timestamp,
                                                     scores_file=score_path,
                                                     frame_length=frame_length,
                                                     feature_type=feature_type,
                                                     processes=workers,
                                                     do_standardize=do_standardize,
                                                     no_crossvalidation=no_crossvalidation,
                                                     rebuild_model=rebuild_model,
                                                     method=model_method,
                                                     model_params=model_params,
                                                     file_handler=fh)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the FEATURE_SETTINGS.json to use for the training")

    args = vars(parser.parse_args())
    if args is None:
        print("Using default root FEATURE_SETTINGS.json location")
        args = {'settings': r'FEATURE_SETTINGS.json'}

    settings = su.get_settings(args['settings'])

    # For log file
    file_components = [settings['FEATURE_TYPE'],
                       'svm',
                       'frame_length_{}'.format(12)]
    optional_file_components = {'standardized': True}

    # Setup logging stuff, this removes 'log_dir' from the dictionary
    # lu.setup_logging(src.get_logger_name(), timestamp, file_components, optional_file_components, settings['LOG_LEVEL'], settings['LOG_PATH'])
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    setup_logging('eeg-model', timestamp, settings['LOG_LEVEL'], settings['LOG_PATH'])
    eeg_logger = logging.getLogger('eeg_machine.main')

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
