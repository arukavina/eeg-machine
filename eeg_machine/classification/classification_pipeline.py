#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running the classification pipeline in python
GPL
"""

# Built-in/Generic Imports
import datetime
import glob
import logging
import os
import os.path
import pickle

# Own modules
from eeg_machine.util import file_utils
from eeg_machine.util.file_utils import FileHelper
from eeg_machine.classification import submissions
from eeg_machine.classification import seizure_modeling
from eeg_machine.datasets import dataset, features_combined, correlation_convertion, wavelet_classification

eeg_logger = logging.getLogger(__name__)


def run_batch_classification(feature_folders,
                             timestamp,
                             file_components=None,
                             optional_file_components=None,
                             scores_file=None,
                             frame_length=1,
                             sliding_frames=False,
                             rebuild_data=False,
                             feature_type='hills',
                             processes=1,
                             csv_directory=None,
                             do_downsample=True,
                             downsample_ratio=2.0,
                             do_standardize=False,
                             do_segment_split=True,
                             do_pca=False,
                             random_state=None,
                             **kwargs):
    """
    Runs the batch classification on the feature folders.

    :param feature_folders: Should be a list of folders containing feature files or folders containing the canonical
    subject folders {'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'}.
    If the folder contains any of the subject folders, it will be replaced by them in the list of feature folders.
    :param timestamp: The provided timestamp for this classification
    :param file_components: A list of strings which should be used as parts of the filename of any file generated during
                            the classification.
    :param optional_file_components: A dictionary of strings to booleans of filename parts which should be included only
                                     if the value is True.
    :param scores_file: If this argument is a path, the classification scores will be written to a csv file with
    that path.
    :param frame_length: The length of each frame, in number of windows
    :param sliding_frames: If True, sliding windows will be used
    :param rebuild_data: If True, the cache files will be rebuilt from the csv data.
    :param feature_type: A string describing the type of features to use. If  'wavelets' is supplied, the feature files
    will be loaded as wavelets. If 'cross-correlations' or 'xcorr' is supplied, the features will be loaded as
    cross-correlation features. If 'combined' is supplied, the path of the feature folders will be used to determine
    which features it contains, and the results will be combined column-wise into longer feature vectors.
    :param processes: The number of processes to use in the grid search
    :param csv_directory: Where to place the resulting classification files
    :param do_downsample: If true downsampling of the interictal features will be performed
    :param downsample_ratio: The ratio of interictal to preictal samples after downsampling.
    :param do_standardize: If True, the features will be standardized before classification
    :param do_segment_split: If True, the cross-validation will be performed by splitting on whole segments
    :param do_pca: If True, the the PCA feature reduction will be performed on the data.
    :param random_state: Seed
    :param kwargs: Incoming kwargs that will be passed on to the classifiers
    :return: None. Creates the output classification and submission files in ../submissions.
    """

    all_scores = []
    for feature_dict in load_features(feature_folders,
                                      feature_type=feature_type,
                                      frame_length=frame_length,
                                      sliding_frames=sliding_frames,
                                      rebuild_data=rebuild_data,
                                      processes=processes,
                                      do_downsample=do_downsample,
                                      downsample_ratio=downsample_ratio,
                                      do_standardize=do_standardize,
                                      do_segment_split=do_segment_split,
                                      do_pca=do_pca,
                                      random_state=random_state):
        kwargs.update(feature_dict)  # Adds the content of feature dict to the keywords for run_classification
        segment_scores = run_classification(processes=processes,
                                            csv_directory=csv_directory,
                                            file_components=file_components,
                                            optional_file_components=optional_file_components,
                                            random_state=random_state,
                                            **kwargs)
        score_dict = segment_scores.to_dict()['Preictal']
        all_scores.append(score_dict)

    if scores_file is None or os.path.isdir(scores_file):
        # We need to generate a new score filename
        if file_components is None:
            file_components = [feature_type,
                               kwargs['method'],
                               'frame_length_{}'.format(frame_length)]
        if optional_file_components is None:
            optional_file_components = dict(standardized=do_standardize,
                                            sliding_frames=sliding_frames)

        filename = file_utils.generate_filename('submission',
                                                '.csv',
                                                file_components,
                                                optional_file_components,
                                                timestamp=timestamp)
        # Let the scores path take precedence if it's a folder
        if scores_file is not None:
            scores_file = os.path.join(scores_file, filename)
        elif csv_directory is not None:
            scores_file = os.path.join(csv_directory, filename)
        else:
            scores_file = os.path.join('..', '..', 'scores', filename)

    eeg_logger.info("Saving scores to {}".format(scores_file))
    with open(scores_file, 'w') as fp:
        submissions.write_scores(all_scores, output=fp, do_normalize=True)


def load_features(feature_folders,
                  feature_type='cross-correlations',
                  frame_length=1,
                  sliding_frames=False,
                  rebuild_data=False,
                  processes=1,
                  do_downsample=False,
                  downsample_ratio=2.0,
                  do_standardize=False,
                  do_segment_split=True,
                  do_pca=False,
                  random_state=None):
    """
    Loads the features from the list of paths *feature_folder*. Returns an iterator of dictionaries, where each
    dictionary has the keys 'subject_folder', 'interictal_data', ''preictal_data' and 'unlabeled_data'.

    :param feature_folders: A list of paths to folders containing features. The features in these folders will be
    combined into three data frames.
    :param feature_type: A string describing the type of features to use. If  'wavelets' is supplied, the feature files
    will be loaded as wavelets. If 'cross-correlations' or 'xcorr' is supplied, the features will be loaded as
    cross-correlation features. If 'combined' is supplied, the path of the feature folders will be used to determine
    which features it contains, and the results will be combined column-wise into longer feature vectors.
    :param frame_length: The desired frame length in windows of the features.
    :param sliding_frames: If True, the training feature frames will be generated by a sliding window, greatly
    increasing the number of generated frames.
    :param rebuild_data: If True, will rebuild the feature data cache files.
    :param processes: Number of processes to use in the loading of the data
    :param do_downsample:
    :param downsample_ratio:
    :param do_standardize:
    :param do_segment_split:
    :param do_pca:
    :param random_state:
    :return: A generator object which gives a dictionary with features for every call to next. The dictionary contains
    the keys 'subject_folder', 'interictal_data', 'preictal_data' and 'unlabeled_data'.
    """

    feature_folders = sorted(FileHelper.expand_folders(feature_folders))

    if feature_type in ['wavelets', 'hills', 'cross-correlations', 'xcorr']:

        if feature_type in ['wavelets', 'hills']:
            feature_module = wavelet_classification
        else:
            feature_module = correlation_convertion

        for feature_folder in feature_folders:
            interictal, preictal, unlabeled = feature_module.load_data_frames(feature_folder,
                                                                              rebuild_data=rebuild_data,
                                                                              processes=processes,
                                                                              frame_length=frame_length,
                                                                              sliding_frames=sliding_frames)
            interictal, preictal, unlabeled = preprocess_features(interictal, preictal, unlabeled,
                                                                  do_downsample=do_downsample,
                                                                  downsample_ratio=downsample_ratio,
                                                                  do_standardize=do_standardize,
                                                                  do_segment_split=do_segment_split,
                                                                  do_pca=do_pca,
                                                                  random_state=random_state)
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=feature_folder)

    elif feature_type == 'combined':
        combined_folders = FileHelper.group_folders(feature_folders)
        for subject, combo_folders in combined_folders.items():
            # We create an output folder which is based on the subject name
            subject_folder = os.path.join('..', '..', 'data', 'combined', subject)
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)

            dataframes = dataset.load_data_frames(combo_folders,
                                                  load_function=features_combined.load,
                                                  find_features_function=file_utils.find_grouped_feature_files,
                                                  rebuild_data=rebuild_data,
                                                  processes=processes,
                                                  frame_length=frame_length,
                                                  sliding_frames=sliding_frames,
                                                  output_folder=subject_folder)
            interictal, preictal, unlabeled = dataframes
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=subject_folder)
    else:
        raise NotImplementedError("No feature loading method implemented for feature type {}".format(feature_type))


def preprocess_features(interictal,
                        preictal,
                        test,
                        do_downsample=False,
                        downsample_ratio=2.0,
                        do_standardize=False,
                        do_segment_split=False,
                        do_pca=False,
                        random_state=None):
    """
    Performs pre-processing on the features.

    This can include downsampling the interictal data, standardizing the data and performing PCA feature reduction.
    :param interictal: A DataFrame with the interictal training data.
    :param preictal: A DataFrame with the preictal training data.
    :param test: A DataFrame with the unlabeled test data.
    :param do_downsample: If True, the majority class (the interictal data) will be downsampled.
    :param downsample_ratio: The ratio of interictal/preictal class size after downsampling. Set to 1.0 to make the
                             classes equal size.
    :param do_standardize: If True, the data will be centered to 0 mean and scaled to std. deviation 1.
    :param do_segment_split: If True, all data manipulation (such as downsampling) will be on a segment basis.
                             If False, the data will be manipulated at the observation level.
    :param do_pca: If True, the dataset will be projected on it's principal components.
    :param random_state: If not None, this constant will be used to seed the random number generator.
    :return: A triple of DataFrames (interictal, preictal, test)
    """
    eeg_logger.info("Preprocessing features")
    if do_downsample:
        interictal = dataset.downsample(interictal,
                                        len(preictal) * downsample_ratio,
                                        do_segment_split=do_segment_split,
                                        random_state=random_state)
    if do_standardize:
        eeg_logger.info("Standardizing variables.")
        interictal, preictal, test = dataset.scale([interictal, preictal, test])
        eeg_logger.info("Shapes after standardization:")
        eeg_logger.info("Interictal: {}".format(interictal.shape))
        eeg_logger.info("Preictal: {}".format(preictal.shape))
        eeg_logger.info("Unlabeled: {}".format(test.shape))

    if do_pca:
        eeg_logger.info("Performing SVD decomposition of features")
        interictal, preictal, test = dataset.pca_transform([interictal, preictal, test])
        eeg_logger.info("Shapes after SVD decomposition:")
        eeg_logger.info("Interictal: {}".format(interictal.shape))
        eeg_logger.info("Preictal: {}".format(preictal.shape))
        eeg_logger.info("Unlabeled: {}".format(test.shape))

    return interictal, preictal, test


def run_classification(interictal_data,
                       preictal_data,
                       unlabeled_data,
                       subject_folder,
                       training_ratio=.8,
                       file_components=None,
                       optional_file_components=None,
                       model_file=None,
                       rebuild_model=False,
                       method="logistic",
                       do_segment_split=False,
                       processes=4,
                       csv_directory=None,
                       do_refit=True,
                       cv_verbosity=2,
                       model_params=None,
                       random_state=None,
                       no_crossvalidation=False):
    """
    Trains a model for a single subject and returns the classification scores for the unlabeled data of that subject.

    :param interictal_data: A DataFrame with the interictal training data.
    :param preictal_data: A DataFrame with the preictal training data.
    :param unlabeled_data: A DataFrame with the unlabeled test data.
    :param subject_folder: The folder holding the features for the subject, this is where the model file and subject
                           scores are saved if csv_directory is not given.
    :param training_ratio: The ratio of training data to use for the training set during cross validation.
    :param file_components: A list of strings which should be included in any filename generated.
    :param optional_file_components: A dictionary of strings as keys and booleans as values for filename components
                                     which should be included in generated filenames if the value is true.
    :param model_file: If given, the model will be unpickled from the given path and used for classification.
    :param rebuild_model: If given, a new model will always be fitted regardless of whether *model_file* is given.
    :param method: The model to use for classification.
    :param do_segment_split: If True, all data manipulation (e.g. training/test split) will be on a per segment basis.
                             If False, the data will be manipulated on a per row basis, ignoring segments.
    :param processes: The number of parallel processes to use for cross validation.
    :param csv_directory: If given, the subject classification scores will be written to this folder. If None, the
                          scores are saved to the subject folder.
    :param do_refit: If True, the model will be refit using all training data after cross validation has selected the
                     parameters. If False, the model will only be fit with *training_ratio* of the training data.
    :param cv_verbosity: The verbosity level for the parameter grid search. 0 is silent, 2 is maximum verbosity.
    :param model_params: A dictionary with keywords to use as arguments for the model. If cross validation is used,
                         the values need to be lists. If cross validation is not used, the values needs to be the type
                         expected by the model.
    :param random_state: An optional constant to seed the random number generator with.
    :param no_crossvalidation: If True, no cross validation will be performed. If this is the case, *model_params* must
                               be given.
    :return: A dictionary of scores
    """
    eeg_logger.info("Running classification on folder {}".format(subject_folder))

    model = None
    if model_file is None and not rebuild_model:
        model = get_latest_model(subject_folder, method)
        if model is None:
            rebuild_model = True

    timestamp = None
    if rebuild_model:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model = seizure_modeling.train_model(interictal_data,
                                             preictal_data,
                                             method=method,
                                             do_segment_split=do_segment_split,
                                             training_ratio=training_ratio,
                                             processes=processes,
                                             cv_verbosity=cv_verbosity,
                                             model_params=model_params,
                                             random_state=random_state,
                                             no_crossvalidation=no_crossvalidation)
        if model_file is None:
            # Create a new filename based on the model method and the date
            if file_components is None:
                model_basename = "model_{}_{}.pickle".format(method, timestamp)
            else:
                model_basename = file_utils.generate_filename('model',
                                                              '.pickle',
                                                              components=file_components,
                                                              optional_components=optional_file_components,
                                                              timestamp=timestamp)
            model_file = os.path.join(subject_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    # If we don't use cross-validation we shouldn't refit
    if do_refit and not no_crossvalidation:
        eeg_logger.info("Refitting model with held-out data.")
        model = seizure_modeling.refit_model(interictal_data, preictal_data, model)

    if csv_directory is None:
        csv_directory = subject_folder
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    scores = write_scores(csv_directory, unlabeled_data, model, file_components=file_components,
                          optional_file_components=optional_file_components, timestamp=timestamp)

    eeg_logger.info("Finished with classification on folder {}".format(subject_folder))

    return scores


def write_scores(csv_directory, test_data, model, file_components=None, optional_file_components=None, timestamp=None):
    """
    Writes the model prediction scores for the segments of *test_data* to a csv file.

    :param csv_directory: The directory to where the classification scores will be written.
    :param test_data: The dataframe holding the test data
    :param model: The model to use for predicting the preictal probability of the test data.
    :param file_components: A list of strings which should be included in any filename generated.
    :param optional_file_components: A dictionary of strings as keys and booleans as values for filename components
                                     which should be included in generated filenames if the value is true.
    :param timestamp: If this argument is provided, it will be used as part of the generated filename.
                      Otherwise the current time will be used as a time stamp for the file.
    :return: A dataframe containing the predicted segment preictal probabilities. The probabilities are also written
    to the *csv_directory*.
    """

    if timestamp is None:
        timestamp = datetime.datetime.now().replace(microsecond=0)

    segment_scores = seizure_modeling.assign_segment_scores(test_data, model)
    if file_components is None:
        score_file = "classification_{}.csv".format(timestamp)
    else:
        score_file = file_utils.generate_filename('classification', '.csv', components=file_components,
                                                  optional_components=optional_file_components, timestamp=timestamp)
    score_path = os.path.join(csv_directory, score_file)
    eeg_logger.info("Writing classification scores to {}.".format(score_path))
    segment_scores.to_csv(score_path, index_label='file')
    return segment_scores


def get_latest_model(feature_folder, method, model_pattern="model*{method}*.pickle"):
    """
    Retrieve the latest cached model for specified feature folder
    """
    model_glob = os.path.join(feature_folder, model_pattern.format(method=method))
    files = glob.glob(model_glob)
    times = [(os.path.getctime(model_file), model_file)
             for model_file in files]
    if times:
        _, latest_model = max(times)
        eeg_logger.info("Latest model is:", latest_model)
        with open(latest_model, 'rb') as fp:
            eeg_logger.info("Loading classifier from {}.".format(latest_model))
            model = pickle.load(fp, encoding='bytes')
            return model
    else:
        return None
