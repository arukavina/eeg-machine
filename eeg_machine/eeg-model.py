#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running model training.
"""

# Built-in/Generic Imports
import datetime
import logging

# Libs
import numpy as np

# Own
from eeg_machine import setup_logging
from eeg_machine.classification import classification_pipeline


__author__ = 'Andrei Rukavina'
__copyright__ = '2021, EGG-Machine'
__credits__ = ['Andrei Rukavina']
__license__ = 'GPL'
__version__ = '0.3.0'
__email__ = 'rukavina.andrei@gmail.com'
__status__ = 'dev'


def fix_model_params(model_params_string):
    return eval(model_params_string)


def main():
    args_dict = get_cli_args()

    if args_dict['random_state'] is not None:
        np.random.seed(args_dict['random_state'])

    if args_dict['model_params'] is not None:
        args_dict['model_params'] = fix_model_params(args_dict['model_params'])

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    setup_logging('eeg-model', timestamp, args_dict['log_level'], args_dict['log_path'])
    eeg_logger = logging.getLogger('eeg_machine.main')

    eeg_logger.info("Starting training with the following arguments: {}".format(args_dict))
    classification_pipeline.run_batch_classification(timestamp=timestamp,  **args_dict)


def get_cli_args():
    """
    Returns the command line arguments.
    :return: A dictionary with the command line argument keys
    """
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")

    parser.add_argument("feature_folders",
                        help=("The folders containing the features. Multiple"
                              " paths can be specified. The given path will "
                              "be checked if it's a feature root folder, which"
                              " means it's a folder containing the canonical"
                              " subject directories. If that is the case,"
                              " it will be expanded into those subject folders."
                              " If it doesn't contain any canonical subject"
                              " folder, the argument is assumed to contain"
                              " feature files."),
                        nargs='+')
    parser.add_argument("-t", "--feature-type",
                        help=("The type of the features for the classification."
                              " 'cross-correlations' and 'xcorr' are synonyms."
                              "If the method is 'combined', the name of the "
                              "folder wil be used to decide which feature loader"
                              " to use. The folder must have the string "
                              "'wavelet' in it for the wavelet features and "
                              "the string 'corr' in it for cross correlation"
                              " features."),
                        choices=["wavelets",
                                 "cross-correlations",
                                 "xcorr",
                                 "hills",
                                 "combined"],
                        required=True,
                        dest='feature_type')
    parser.add_argument("--rebuild-data",
                        action='store_true',
                        help="Should the dataframes be re-read from the csv feature files",
                        dest='rebuild_data')
    parser.add_argument("--training-ratio",
                        type=float,
                        default=0.8,
                        help="What ratio of the data should be used for training",
                        dest='training_ratio')
    parser.add_argument("--rebuild-model",
                        action='store_true',
                        help="Should the model be rebuild, or should a cached version (if available) be used.",
                        dest='rebuild_model')
    parser.add_argument("--no-downsample",
                        action='store_false',
                        default=True,
                        help="Disable downsampling of the majority class",
                        dest='do_downsample')
    parser.add_argument("--downsample-ratio",
                        default=2.0,
                        type=float,
                        help="The ratio of majority class to minority class after downsampling.",
                        dest='downsample_ratio')
    parser.add_argument("--standardize",
                        action='store_true',
                        help="Standardize the variables",
                        dest='do_standardize',
                        default=False)
    parser.add_argument("--pca",
                        action='store_true',
                        help="Perform dimensionality reduction on the data using PCA",
                        dest='do_pca',
                        default=False)
    parser.add_argument("--no-refit",
                        action='store_false',
                        default=True,
                        help="Don't refit the selected model with the held-out data used to produce accuracy scores",
                        dest='do_refit')
    parser.add_argument("--no-segment-split",
                        action='store_false',
                        help="Disable splitting data by segment.",
                        dest='do_segment_split',
                        default=True)
    parser.add_argument("--method",
                        help="What method to use for learning",
                        dest='method',
                        choices=['logistic',
                                 'svm',
                                 'sgd',
                                 'random-forest',
                                 'nearest-centroid',
                                 'knn',
                                 'bagging'],
                        default='logistic')
    parser.add_argument("--processes",
                        help="How many processes should be used for parallelized work.",
                        dest='processes',
                        default=4,
                        type=int)
    parser.add_argument("--csv-directory",
                        help="Which directory the classification CSV files should be written to.",
                        dest='csv_directory')
    parser.add_argument("--submission-file",
                        help="""If this argument is supplied, a score file
                        with the scores for the the test segments will be produced""",
                        dest='submission_file')
    parser.add_argument("--frame-length",
                        help="The size in windows each frame (feature vector) should be.",
                        dest='frame_length', default=12, type=int)
    parser.add_argument("--sliding-frames",
                        help=("If enabled, frames for the training-data will be"
                              " extracted by overlapping windows, greatly "
                              "increasing the number of frames."),
                        dest='sliding_frames',
                        default=False,
                        action='store_true')
    parser.add_argument("--log-dir",
                        help="Directory for writing classification log files.",
                        default='../../classification_logs',
                        dest='log_dir')
    parser.add_argument("--no-cv",
                        help="Turn off cross-validation and grid search.",
                        default=False,
                        action='store_true',
                        dest='no_crossvalidation')
    parser.add_argument("--cv-verbosity",
                        help=("The verbosity level of the Cross-Validation grid"
                              " search. The higher, the more verbose the grid"
                              " search is. 0 disables output."),
                        default=1,
                        type=int,
                        choices=[0, 1, 2],
                        dest='cv_verbosity')
    parser.add_argument("--model-params", "--p",
                        help=("Allows setting model parameters for the method"
                              " used. This should be a string with a sics_seizure_prediction"
                              " expression containing a similar data structure"
                              " to the grid_param argument to the cross "
                              "validation grid search, but the values doesn't have to be sequences. "
                              "It will be used instead of the default grid_params."),
                        dest='model_params')
    parser.add_argument("--random-state",
                        help=("Give a start seed for random stuff. Ensures repeatability between runs. "
                              "If set to 'None', The random functions won't be seeded (using default initialization)"),
                        dest='random_state',
                        default='2806')
    args_dict = vars(parser.parse_args())

    # Since we use 'None' to turn of constant seeding, we use eval here instead of just parsing the argument as an int
    args_dict['random_state'] = eval(args_dict['random_state'])
    return args_dict


if __name__ == '__main__':
    main()
