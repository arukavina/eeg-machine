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

# Libs
import datetime
import logging
import src

# Own
from src.util import log_utils as lu
from src.util import file_utils as fu
from src.features import hills, wavelets, cross_correlate


if __name__ == '__main__':

    import argparse

    __logger__ = 'eeg-features'

    parser = argparse.ArgumentParser(description="Calculates features.")

    parser.add_argument("segments",
                        help="The files to process. This can either be the path to a matlab file holding"
                             " the segment or a directory holding such files.", nargs='+')
    parser.add_argument("--csv-directory", help="Directory to write the csv files to, if omitted, the files will be"
                                                " written to the same directory as the segment")
    parser.add_argument("--window-size", help="What length in seconds the epochs should be.", type=float,
                        default=5.0)
    parser.add_argument("--feature-length",
                        help="The length of the feature vectors in seconds, will be produced by "
                             "concatenating the phase lock values from the windows.", type=float,
                        default=60.0)
    parser.add_argument("--workers",
                        help="The number of worker processes used for calculating the cross-correlations.",
                        type=int, default=1)
    parser.add_argument("--resample-frequency", help="The frequency to resample to.",
                        type=float,
                        dest='resample_frequency')
    parser.add_argument("--normalize-signal",
                        help="Setting this flag will normalize the channels based on the subject median and MAD.",
                        default=False,
                        action='store_true',
                        dest='normalize_signal')
    parser.add_argument("--stats-directory",
                        help="Directory where to find the stats of files. Center must be calculated in advance",
                        default=False,
                        dest='stats_directory')
    parser.add_argument("--log-path", help="Log directory.", default='logs/', dest='LOG_PATH')
    args = parser.parse_args()

    if args is None:
        print("Using default values for all args")

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    # TODO: FIx this crap
    file_components = ['hills',
                       'frame_length_{}'.format(12)]

    optional_file_components = {'normalize_signal': args.normalize_signal,
                                'resample_frequency': args.resample_frequency}

    # Setup logging stuff, this removes 'log_dir' from the dictionary
    lu.setup_logging(__logger__, timestamp, file_components, optional_file_components, vars(args))
    feature_logger = logging.getLogger(src.get_logger_name())

    feature_logger.info("Extracting Features")

    feature_logger.info(args.segments)

    fh_args_dict = dict([
        ('train_path', args.segments),
        ('test_path', ''),
        ('cat_column', ''),
        ('class_labels', ''),
        ('logger', feature_logger)
    ])

    try:
        fh = fu.FileHelper(**fh_args_dict)
    except AttributeError:
        raise AttributeError('Attribute error when trying to instantiate class. Check __init__ or __doc__')
    except Exception as e:
        raise AttributeError('Something else is really wrong: {}'.format(e))

    hills.extract_features(args.segments,
                           args.csv_directory,
                           workers=args.workers,
                           resample_frequency=args.resample_frequency,
                           file_handler=fh,
                           normalize_signal=args.normalize_signal,
                           stats_directory=args.stats_directory,
                           feature_length_seconds=args.feature_length,
                           window_size=args.window_size)
