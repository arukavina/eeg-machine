#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running the feature extraction and model training.
"""

# Built-in/Generic Imports
import datetime
import logging

# Own
import src
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


def extract_features(settings, fh):
    """
    Extract features based on the dictionary *settings*. The type of features extracted depends on the key
    'FEATURE_TYPE' and should be either 'xcorr', 'wavelets' or 'hills'.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :param fh: File Handler
    :return: None. The features will be saved as csv files to the directory given by the key 'FEATURE_PATH' in the
             settings dictionary.
    """
    train_segment_paths = settings['TRAIN_DATA_PATH']
    output_path = settings['FEATURE_PATH']
    stats_path = settings['STATS_PATH']

    old_segment_format = settings['MATLAB_SEGMENT_FORMAT']
    only_missing_files = settings['ONLY_MISSING_FILES']
    sample_size = settings['SAMPLE_SIZE']

    feature_types = settings['FEATURE_TYPE']

    window_size = settings['FEATURE_SETTINGS']['WINDOW_SIZE']
    frame_length = settings['FEATURE_SETTINGS']['FRAME_LENGTH']
    resample_frequency = settings['FEATURE_SETTINGS']['RESAMPLE_FREQUENCY']
    normalize_signal = settings['FEATURE_SETTINGS']['NORMALIZE_SIGNAL']

    workers = settings['WORKERS']

    eeg_logger = logging.getLogger(src.get_logger_name())
    eeg_logger.info("Starting extract_features with the following arguments:")
    for s, v in settings.items():
        eeg_logger.info("\t{} = {}".format(s, v))

    if 'hills' in feature_types:
        hills.extract_features(segment_paths=train_segment_paths,
                               output_dir=output_path,
                               workers=workers,
                               sample_size=sample_size,
                               matlab_segment_format=old_segment_format,
                               resample_frequency=resample_frequency,
                               normalize_signal=normalize_signal,
                               stats_directory=stats_path,
                               only_missing_files=only_missing_files,
                               file_handler=fh,
                               feature_length_seconds=window_size * frame_length,
                               window_size=window_size)

    elif 'xcorr' in settings['FEATURE_TYPE'] or 'cross correlations' in settings['FEATURE_TYPE']:
        cross_correlate.extract_features(segment_paths=train_segment_paths,
                                         output_dir=output_path,
                                         workers=workers,
                                         window_size=settings['FEATURE_SETTINGS']['WINDOW_SIZE'],
                                         file_handler=fh)

    elif 'wavelets' in settings['FEATURE_TYPE']:
        wavelets.extract_features(segment_paths=train_segment_paths,
                                  output_dir=output_path,
                                  workers=workers,
                                  window_size=settings['FEATURE_SETTINGS']['WINDOW_SIZE'],
                                  file_handler=fh,
                                  feature_length_seconds=window_size * frame_length)

    elif 'combined' in settings['FEATURE_TYPE']:  # Must go first
        hills.extract_features(segment_paths=train_segment_paths,
                               output_dir=output_path,
                               workers=workers,
                               sample_size=sample_size,
                               matlab_segment_format=old_segment_format,
                               resample_frequency=resample_frequency,
                               normalize_signal=normalize_signal,
                               stats_directory=stats_path,
                               only_missing_files=only_missing_files,
                               file_handler=fh,
                               feature_length_seconds=window_size * frame_length,
                               window_size=window_size)

        cross_correlate.extract_features(segment_paths=train_segment_paths,
                                         output_dir=output_path,
                                         workers=workers,
                                         window_size=settings['FEATURE_SETTINGS']['WINDOW_SIZE'],
                                         file_handler=fh)

        wavelets.extract_features(segment_paths=train_segment_paths,
                                  output_dir=output_path,
                                  workers=workers,
                                  window_size=settings['FEATURE_SETTINGS']['WINDOW_SIZE'],
                                  file_handler=fh,
                                  feature_length_seconds=window_size * frame_length)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the SETTINGS.json to use for the training")

    args = vars(parser.parse_args())
    if args is None:
        print("Using default root SETTINGS.json location")
        args = {'settings': r'SETTINGS.json'}

    settings = su.get_settings(args['settings'])

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    lu.setup_logging(__name__, timestamp, settings['LOG_LEVEL'], settings['LOG_PATH'])
    eeg_logger = logging.getLogger(__name__)

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

    eeg_logger.info("Extracting Features")
    extract_features(settings, fh)


if __name__ == '__main__':
    main()
