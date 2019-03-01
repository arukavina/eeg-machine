"""
------------------------------------------------------------------------------------------------------
Script:    Module holding some utilities for doing file related stuff
------------------------------------------------------------------------------------------------------
"""

from __future__ import absolute_import

# General
import os.path
import re
import pandas as pd
import json
import glob
import numpy as np

from collections import defaultdict
from glob import glob

TEST_SEGMENT_NAMES_FILE = 'data/test_segment_names.json'


class FileSystemException(RuntimeError):
    def __init__(self, arg):
        self.arg = arg


class FileHelper:
    def __init__(self,
                 train_path,
                 test_path,
                 cat_column,
                 class_labels,
                 logger):

        self.logger = logger
        self.train_path = train_path
        self.test_path = test_path
        self.cat_column = cat_column
        self.class_labels = class_labels

    def get_training_files(self, sample_size=None, balance=True, random_state=None):
        self.logger.debug("args: sample_size: {} - balance: {} - random_state: {}".format(sample_size, balance,
                                                                                          random_state))
        train_df = pd.DataFrame([{'path': c_path,
                                  'image_name': os.path.basename(c_path),
                                  self.cat_column: os.path.basename(os.path.dirname(c_path))} for c_path in
                                 glob(self.train_path)])

        train_df = train_df[1:]
        if sample_size is not None:
            if balance:
                idx = self.balanced_subsample(train_df[self.cat_column], size=sample_size, random_state=random_state)
                train_df = train_df.loc[idx, ]
            else:
                train_df = train_df.sample(sample_size, random_state=random_state)

        return train_df

    def get_testing_files(self, sample_size=None, random_state=None):
        self.logger.debug("args: sample_size: {} - random_state: {}".format(sample_size, random_state))
        test_df = pd.DataFrame([{'path': c_path,
                                 'image_name': os.path.basename(c_path)}
                                for c_path in glob(self.test_path)])

        test_df = test_df[1:]
        if sample_size is not None:
            test_df = test_df.sample(sample_size, random_state=random_state)
        return test_df

    @staticmethod
    def get_subject(string):
        """
        Extracts the subject string from the given string. The string must contain a substring matching a canonical
        folder name.

        :param string: The string to match subject name in.
        :return: The subject name if it is exists in the string. None if no match is found.
        """
        subject_pattern = r".*(Patient_[12]|Dog_[1-5]).*"
        subject = re.match(subject_pattern, string)
        if subject is not None:
            return subject.group(1)
        else:
            return None

    @staticmethod
    def expand_paths(file_names, recursive=True):
        """
        Goes through the list of *filenames* and expands any directory to the files included in that directory.

        :param file_names: A list of paths to expand. If any path is a directory, it will be replaced in the list with
        the contents of the directory.
        :param recursive: If True, any directory in an expanded directory will also be expanded.
        :return: A list of files.
        """

        new_files = []
        for fn in file_names:
            if os.path.isdir(fn):
                if recursive:
                    # We recurse over all files contained in the directory and add them to the list of files
                    for dir_path, _, sub_file_names in os.walk(fn):
                        new_files.extend([os.path.join(dir_path, filename)
                                          for filename in sub_file_names])
                else:
                    # No recursion, we just do a listfile on the files of any directoy in filenames
                    for subfile in os.listdir(fn):
                        if os.path.isfile(subfile):
                            new_files.append(os.path.join(fn, subfile))
            elif os.path.isfile(fn):
                new_files.append(fn)
        return new_files

    def expand_folders(self, feature_folders, canonical_folders=None):
        """
        Goes through the list of *feature_folders* and replaces any directory which contains the canonical subject
        folders with the path to those folders. Folders not containing any canonical feature folder is left as is.
        :param feature_folders: The list of folders to expand.
        :param canonical_folders: The list of canonical folder names to look for in the feature folders.
        :return: A list of paths where any folder in the original feature folders list containing canonical folders has
                 been replaced with the path to those folders.
        """
        self.logger.debug("args: feature_folders: {} - canonical_folders: {}".format(feature_folders, canonical_folders))

        if canonical_folders is None:
            canonical_folders = self.class_labels

        canonical_folders = set(canonical_folders)
        new_folders = []
        for folder in feature_folders:
            subfolders = set([sf for sf in os.listdir(folder) if os.path.isdir(os.path.join(folder, sf))])

            common_folders = subfolders & canonical_folders
            if common_folders:
                new_folders.extend([os.path.join(folder, common)
                                    for common
                                    in common_folders])
            else:
                new_folders.append(folder)
        return new_folders

    @staticmethod
    def group_folders(feature_folders):
        """
        Groups the feature folder per subject. Returns a dictionary with subject to
        feature folders lists. If the subject of any of the folders can't be found,
        it will be excluded from the grouping.
        :param feature_folders: The list of folders paths to group.
        :return: A dictionary of subject to folder list mappings.
        """

        grouped_folders = defaultdict(list)
        for feature_folder in feature_folders:
            subject = FileHelper.get_subject(feature_folder)
            if subject is not None:
                grouped_folders[subject].append(feature_folder)
        return grouped_folders

    @staticmethod
    def balanced_subsample(y, random_state, size=None):

        subsample = []
        np.random.RandomState(random_state)

        if size is None:
            n_smp = y.value_counts().min()
        else:
            n_smp = int(size / len(y.value_counts().index))

        for label in y.value_counts().index:
            samples = y[y == label].index.values
            index_range = range(samples.shape[0])
            indexes = np.random.choice(index_range, size=n_smp, replace=False)
            subsample += samples[indexes].tolist()

        return subsample

    @staticmethod
    def generate_test_segment_names(name_file=TEST_SEGMENT_NAMES_FILE):
        """
        Generates a json file containing all the canonical test file names. The file is saved to the path denoted by
        the module constant TESTSEGMENT_NAMES_FILE'

        :param name_file: A path to a file containing segment names.
        :return: A set of all canonical test segment names.
        """
        import subprocess
        filenames = subprocess.check_output(['find', '../../data/', '-name', '*test*.mat']).split()
        decoded = [os.path.basename(name.decode()) for name in filenames]
        matched_names = [re.match(r'([DP][a-z]*_[1-5]_[a-z]*_segment_[0-9]{4}).*', name) for name in decoded]
        only_matches = [match.group(1) for match in matched_names if match]
        only_matches.sort()
        formatted_names = ["{}.mat".format(name) for name in only_matches]
        with open(name_file, 'w') as fp:
            json.dump(formatted_names, fp, indent=4, separators=(',', ': '))
        return set(formatted_names)

    def load_test_segment_names(self, name_file=TEST_SEGMENT_NAMES_FILE):
        """Loads the test segment names as a set of names.
        :param name_file: The file with names to load.
        :return: A set of test segment names. """
        self.logger.debug("args: name_file: {}".format(name_file))

        try:
            with open(name_file, 'r') as fp:
                names = json.load(fp)
                return set(names)
        except FileSystemException:
            return self.generate_test_segment_names(name_file)


def create_path(p):
    """
    Creates paths if it doesn't exit
    :param p:
    :return:
    """
    if isinstance(p, list):
        for k in p:
            if not os.path.exists(k):
                try:
                    os.makedirs(k)
                except OSError as e:
                    print('Can\'t create path {}: {}'.format(k, e))
    else:
        if not os.path.exists(p):
            try:
                os.makedirs(p)
            except OSError as e:
                print('Can\'t create path {}: {}'.format(p, e))


def generate_filename(prefix, suffix, components, optional_components=None, sep='-', timestamp=None):
    """
    Generates a filename given the arguments.

    :param prefix: A string giving the prefix of the generated filename.
    :param suffix: A string giving the suffix of the generated filename, usually includes the file extension.
    :param components: A list of strings which should be included in the filename.
    :param optional_components: A dictionary of string keys to boolean values. If a value is True, that string key will
                                be included in the filename.
    :param sep: The separator to use between the prefix, components, optional components and timestamp.
    :param timestamp: A timestamp to include as the last name component before the suffix.
    :return: A string with the given file name parts joined together.
    """

    name_components = [prefix]
    if timestamp is not None:
        name_components.append(timestamp)

    name_components = name_components + components

    if optional_components is not None:
        for optional_component, value in optional_components.items():
            if value:
                name_components.append("_".join((optional_component, str(value))))

    return sep.join(name_components) + suffix


def find_grouped_feature_files(feature_folders, class_name, file_pattern="*segment*.csv"):
    """
    Collects multiple feature files from *feature_folders* matching *class_name* and *file_pattern* and groups them
    based on the original segment name. The difference from *find_feature_files* is that this version can find multiple
    features for every original segment.

    :param feature_folders: A list of feature folder to search for files in.
    :param class_name: The class name of files to find, usually one of {'interictal', 'preictal', 'test'}.
    :param file_pattern: A unix shell style glob pattern to match the files in the feature folder.
    :return: A list of dictionaries, each dictionary corresponding to one original segment. The dictionaries has the
             keys 'segment' and 'files'. 'segment' is the original segment name and 'files' is a list of paths
             the feature files for this segment.
    """

    segments = defaultdict(list)
    for feature_folder in feature_folders:
        # First we locate the files with the canonical segment they
        # are derived from, using the usual find_feature_files
        feature_file_dicts = find_feature_files(feature_folder, class_name, file_pattern=file_pattern)

        # feature_file_dicts is a list of dictionaries, containing a
        # segment name key and a files key, we group this into our
        # segments dictionary
        for feature_file_dict in feature_file_dicts:
            segment = feature_file_dict['segment']
            filename = feature_file_dict['files']
            segments[segment].append(filename)

    return [dict(segment=segment, files=files)
            for segment, files
            in sorted(segments.items())]


def find_feature_files(feature_folder, class_name, file_pattern="*segment*.csv"):
    """
    Collects the files from *feature_folder* matching *class_name* and *file_pattern*.
    :param feature_folder: The folder to search for files in.
    :param class_name: The class name of files to find, usually one of {'interictal', 'preictal', 'test'}.
    :param file_pattern: A unix shell style glob pattern to match the files in the feature folder.
    :return: A list of dictionaries, each dictionary corresponding to one original segment. The dictionaries has the
             keys 'segment' and 'files'. 'segment' is the original segment name and 'files' is the path to the feature
             file for this segment.
    """

    full_pattern = "*{}*{}".format(class_name, file_pattern)
    glob_pattern = os.path.join(feature_folder, full_pattern)
    files = glob.glob(glob_pattern)
    return [{'segment': get_segment_name(filename), 'files': filename}
            for filename in sorted(files)]


def get_segment_name(name):
    """
    Returns the name of the original .mat segment file the name is based on, using a regular expression.

    :param name: A string to canonicalize.
    :return: Either the canonical name of the given string, or if no match is found the original string *name*.
    """

    pattern = r"([DP][a-z]*_[1-5]_[a-z]*_segment_[0-9]{4}).*"
    basename = os.path.basename(name)  # Remove any directories from the name
    match = re.match(pattern, basename)
    if match is None:
        return name
    else:
        return match.group(1) + '.mat'
