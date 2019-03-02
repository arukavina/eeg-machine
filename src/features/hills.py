from __future__ import absolute_import

# Generic
import logging
import sys

# Libs
import mne
from itertools import chain

# Own
import src
from src.features import feature_extractor
from src.features import wavelets
from src.features.transforms import FFTWithTimeFreqCorrelation as FFT_TF_xcorr

mne.set_log_level(verbose='WARNING')
eeg_logger = logging.getLogger(src.get_logger_name())


def extract_features_for_segment(segment, transformation=None, feature_length_seconds=60, window_size=5):
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
    eeg_logger.info("Using extraction function: {}".format('extract_features_for_segment'))

    if transformation is None:
        transformation = FFT_TF_xcorr(1, 48, 400, 'usf')

    # Here we define how many windows we will have to concatenate
    # in order to create the features we want
    windows_in_frame = int(feature_length_seconds / window_size)
    iters = int(segment.get_duration() / feature_length_seconds)

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
    eeg_logger.dubug("Starting")
    if transformation is None:
        return FFT_TF_xcorr(1, 48, 400, 'usf')
    else:
        transformation(**kwargs)


def extract_features(segment_paths,
                     output_dir,
                     workers=1,
                     sample_size=None,
                     old_segment_format=True,
                     resample_frequency=None,
                     normalize_signal=False,
                     stats_directory='/Users/arukavina/Documents/EEG/Statistics/*.csv',
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
    :param old_segment_format:
    :param resample_frequency:
    :param stats_directory: Directory where to find the stats of files. Center must be calculated in advance
    :param normalize_signal:
    :param only_missing_files:
    :param file_handler: fh instance
    :param feature_length_seconds:
    :param window_size:
    :return:
    """
    eeg_logger.info("Starting Hills Extractor")

    feature_extractor.extract(segment_paths,
                              extract_features_for_segment,
                              # Arguments for feature_extractor.extract
                              output_dir=output_dir,
                              workers=workers,
                              sample_size=sample_size,
                              matlab_segment_format=old_segment_format,
                              resample_frequency=resample_frequency,
                              stats_directory=stats_directory,
                              normalize_signal=normalize_signal,
                              only_missing_files=only_missing_files,
                              file_handler=file_handler,
                              # Worker function kwargs:
                              feature_length_seconds=feature_length_seconds,
                              window_size=window_size)

