#!/bin/env python

"""
------------------------------------------------------------------------------------------------------
Script:    Module for running the feature extraction and model training.
Author:    Andrei Rukavina <arukavina@analyticar.com>
------------------------------------------------------------------------------------------------------
"""

# Future
from __future__ import absolute_import
from __future__ import print_function

import datetime
import logging

import src
from src.classification import classification_pipeline
from src.features import hills_features, wavelets, cross_correlate
from src.util import log_utils as lu
from src.util import settings_util as su
from src.util import file_utils as fu


def extract_features(settings):
    """
    Extract features based on the dictionary *settings*. The type of features extracted depends on the key
    'FEATURE_TYPE' and should be either 'xcorr', 'wavelets' or 'hills'.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :return: None. The features will be saved as csv files to the directory given by the key 'FEATURE_PATH' in the
             settings dictionary.
    """
    output_dir = settings['FEATURE_PATH']
    workers = settings['WORKERS']
    window_size = settings['FEATURE_SETTINGS']['WINDOW_LENGTH']
    frame_length = settings['FEATURE_SETTINGS']['FEATURE_WINDOWS']
    train_segment_paths = settings['TRAIN_DATA_PATH']

    eeg_logger = logging.getLogger(src.get_logger_name())
    eeg_logger.info("Starting extract_features with the following arguments: {}".format(settings))

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

    if settings['FEATURE_TYPE'] == 'hills':
        hills_features.extract_features(segment_paths=train_segment_paths,
                                        output_dir=output_dir,
                                        workers=workers,
                                        window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'],
                                        file_handler=fh,
                                        feature_length_seconds=window_size*frame_length)

    elif settings['FEATURE_TYPE'] == 'xcorr':
        cross_correlate.extract_features(segment_paths=train_segment_paths,
                                         output_dir=output_dir,
                                         workers=workers,
                                         window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'],
                                         file_handler=fh)

    elif settings['FEATURE_TYPE'] == 'wavelets':
        wavelets.extract_features(segment_paths=train_segment_paths,
                                  output_dir=output_dir,
                                  workers=workers,
                                  window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'],
                                  file_handler=fh,
                                  feature_length_seconds=window_size*frame_length)


def train_model(settings):
    """
    Trains a SVM classifier using the features selected by the  *settings* dictionary.
    When fitted, the model will automatically be used to assign scores and a submission file will be generated in the
    folder given by 'SUBMISSION_PATH' in the *settings* dictionary.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :return: None. The model will be pickled to a file in the corresponding subject feature folder. A submission file
             will be written to the folder given by 'SUBMISSION_PATH' in the settings dictionary.
    """
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    eeg_logger = logging.getLogger(src.get_logger_name())
    eeg_logger.info("Starting training with the following arguments: {}".format(settings))

    classification_pipeline.run_batch_classification(feature_folders=[settings['FEATURE_PATH']],
                                                     timestamp=timestamp,
                                                     submission_file=settings['SUBMISSION_PATH'],
                                                     frame_length=12,
                                                     feature_type=settings['FEATURE_TYPE'],
                                                     processes=settings['WORKERS'],
                                                     do_standardize=True,
                                                     no_crossvalidation=True,
                                                     rebuild_model=True,
                                                     method='svm',
                                                     model_params={'C': 500, 'gamma': 0})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the SETTINGS.json to use for the training")

    args = vars(parser.parse_args())
    if args is None:
        print("Using default root SETTINGS.json location")
        args = r'SETTINGS.json'

    settings = su.get_settings(args['settings'])

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    file_components = [settings['FEATURE_TYPE'],
                       'svm',
                       'frame_length_{}'.format(12)]
    optional_file_components = {'standardized': True}

    # Setup logging stuff, this removes 'log_dir' from the dictionary
    lu.setup_logging(src.get_logger_name(), timestamp, file_components, optional_file_components, settings)
    eeg_logger = logging.getLogger(src.get_logger_name())

    eeg_logger.info("Extracting Features")
    extract_features(settings)

    eeg_logger.info("Training model")
    train_model(settings)


if __name__ == '__main__':
    main()
