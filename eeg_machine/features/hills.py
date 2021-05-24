#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for Extracting Hills features.
Creates a feature dictionary from a Segment object, according to the provided transformation function.
GPL
"""

# Built-in/Generic Imports
import logging
import sys
import datetime

# Libs
import mne
from itertools import chain

# Own modules
from eeg_machine import setup_logging
from eeg_machine.util import file_utils as fu
from eeg_machine.features import feature_extractor
from eeg_machine.features import wavelets
from eeg_machine.features.transforms import FFTWithTimeFreqCorrelation as FFT_TF_xcorr

mne.set_log_level(verbose='WARNING')
eeg_logger = logging.getLogger(__name__)


def extract_hills_features_for_segment(segment, transformation=None, feature_length_seconds=60, window_size=5):
    """
    Creates a feature dictionary from a Segment object, according to the provided
    transformation function.
    :param segment: A Segment object containing the EEG segment from which we want
        to extract the features
    :param transformation: A class that should implement apply(data), which takes
        an ndarray (n_channels x n_samples) and returns a 1d ndarray of features.
    :param feature_length_seconds: The number of seconds each frame should consist
        of, should be exactly divisible by window_size.
    :param window_size: The length of a window in seconds.
    :return: A dict of features, where each keys are the frames indexes in the segment
        and the values are a List of doubles containing all the feature values
        for that frame.
        Ex. For a 10 min segment with feature_length_seconds=60 (sec) we should
        get 10 frames. The length of the lists then depends on the window_size,
        number of channels and number of frequency bands we are examining.
    """
    if transformation is None:
        transformation = FFT_TF_xcorr(1, 48, 400, 'usf')

    eeg_logger.info("Using extraction function: HILLS using {} transformation".format(transformation))

    # Here we define how many windows we will have to concatenate
    # in order to create the features we want
    windows_in_frame = int(feature_length_seconds / window_size)
    iters = int(segment.get_duration() / feature_length_seconds)

    # TODO: Do I still want this?
    segment.raw_plot(2)

    # Create Epochs object according to defined window size
    epochs = wavelets.epochs_from_segment(segment, window_size)

    feature_list = []
    # Create a list of features
    for epoch in epochs:
        feature_list.append(transformation.apply(epoch).tolist())

    # eeg_logger.info("Epochs: {} of {} features each: ".format(len(feature_list), len(feature_list[0])))

    feature_dict = {}
    # Slice the features to frames
    for i in range(iters):
        window_features = feature_list[i*windows_in_frame:(i+1)*windows_in_frame]
        feature_dict[i] = list(chain.from_iterable(window_features))

    if len(feature_dict) != iters:
        sys.stderr.write("WARNING: Wrong number of features created, expected"
                         " %d, got %d instead." % (iters, len(feature_dict)))

    for index, feature in sorted(feature_dict.items()):
        eeg_logger.info("Features per Iter: {}".format(len(feature)))

        break

    return feature_dict


def get_transform(transformation=None, **kwargs):
    eeg_logger.debug("Obtaining transform")
    if transformation is None:
        return FFT_TF_xcorr(1, 48, 400, 'usf')
    else:
        transformation(**kwargs)


def extract_features(segment_paths,
                     output_dir,
                     workers=1,
                     sample_size=None,
                     matlab_segment_format=True,
                     resample_frequency=None,
                     normalize_signal=False,
                     stats_glob='/Users/arukavina/Documents/EEG/Statistics/*.csv',
                     only_missing_files=True,
                     file_handler=None,
                     feature_length_seconds=60,
                     window_size=5):
    """
    Performs feature extraction of the segment files found in *segment_paths*. The features are written to csv
    files in *output_dir*. See :py:function`feature_extractor.extract` for more info.
    :param segment_paths:
    :param output_dir:
    :param workers:
    :param sample_size:
    :param matlab_segment_format:
    :param resample_frequency:
    :param stats_glob: Directory where to find the stats of files. Center must be calculated in advance
    :param normalize_signal:
    :param only_missing_files:
    :param file_handler: fh instance
    :param feature_length_seconds:
    :param window_size:
    :return:
    """
    eeg_logger.info("Starting Hills Extractor")

    feature_extractor.extract(segment_paths,
                              extractor_function=extract_hills_features_for_segment,  # This is defined above
                              # Arguments for feature_extractor.extract
                              output_dir=output_dir,
                              workers=workers,
                              sample_size=sample_size,
                              matlab_segment_format=matlab_segment_format,
                              resample_frequency=resample_frequency,
                              stats_directory=stats_glob,
                              normalize_signal=normalize_signal,
                              only_missing_files=only_missing_files,
                              file_handler=file_handler,
                              # Worker function kwargs:
                              feature_length_seconds=feature_length_seconds,
                              window_size=window_size)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculates the SPLV phase lock between pairwise channels.")

    parser.add_argument("segments",
                        help=("The files to process. This can either be the path to a matlab file holding "
                              "the segment or a directory holding such files."),
                        nargs='+',
                        metavar="SEGMENT_FILE")
    parser.add_argument("--csv-directory",
                        help=("Directory to write the csv files to, if omitted, the files will be written to the "
                              "same directory as the segment"))
    parser.add_argument("--matlab-segment-format",
                        help="Should the segment object be loaded with the old segment format.",
                        action='store_true',
                        dest='matlab_segment_format',
                        default=False)
    parser.add_argument("--only-missing-files",
                        help="Should process only missing segments",
                        action='store_true',
                        dest='only_missing_files',
                        default=False)
    parser.add_argument("--window-size",
                        help="What length in seconds the epochs should be.",
                        type=float,
                        default=5.0)
    parser.add_argument("--frame-length",
                        help="The length of the feature vectors in seconds, will be produced by concatenating the "
                             "phase lock values from the windows.",
                        type=float,
                        default=6.0)
    parser.add_argument("--workers",
                        help="The number of worker processes used for calculating the cross-correlations.",
                        type=int,
                        default=1)
    parser.add_argument("--resample-frequency",
                        help="The frequency to resample to.",
                        type=float,
                        dest='resample_frequency')
    parser.add_argument("--sample-size",
                        help="Optionally sample this many samples from the input files.",
                        type=float,
                        dest='sample_size')
    parser.add_argument("--stats-glob",
                        help="Statistics CSV directory to which read from.",
                        dest='stats_glob',
                        default='../../data/segment_statistics')
    parser.add_argument("--normalize-signal",
                        help="Whether to normalize the signal before performing the feature extraction",
                        action='store_true',
                        dest='normalize_signal',
                        default=False)
    parser.add_argument("--log-dir",
                        help="Directory for writing classification log files.",
                        default='./../../logs',
                        dest='log_path')
    parser.add_argument("--log-level",
                        help="Logging module verbosity level:"
                             "CRITICAL = 50"
                             "ERROR = 40"
                             "WARNING = 30"
                             "INFO = 20"
                             "DEBUG = 10"
                             "NOTSET = 0",
                        default=20,
                        choices=[50, 40, 30, 20, 10, 0],
                        type=int,
                        dest='log_level')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    setup_logging('eeg-hills-features', timestamp, args.log_level, args.log_path)

    global eeg_logger
    eeg_logger = logging.getLogger('eeg_machine.features.hills')

    eeg_logger.info("Calculating Hills features")

    fh_args_dict = dict([
        ('train_path', args.segments)
    ])

    try:
        fh = fu.FileHelper(**fh_args_dict)
    except AttributeError:
        raise AttributeError('Attribute error when trying to instantiate class. Check __init__ or __doc__')
    except Exception as e:
        raise AttributeError('Something else is really wrong: {}'.format(e))

    extract_features(segment_paths=args.segments,
                     output_dir=args.csv_directory,
                     workers=args.workers,
                     sample_size=args.sample_size,
                     matlab_segment_format=args.matlab_segment_format,
                     resample_frequency=args.resample_frequency,
                     normalize_signal=args.normalize_signal,
                     stats_glob=args.stats_glob,
                     only_missing_files=args.only_missing_files,
                     file_handler=fh,
                     feature_length_seconds=args.frame_length * args.window_size,
                     window_size=args.window_size)


if __name__ == '__main__':
    main()
