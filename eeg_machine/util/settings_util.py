#!/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------------------
Script:    Module to process settings.json
------------------------------------------------------------------------------------------------------
"""

# Generics
import json
import os.path

# Own modules
from eeg_machine import file_utils as fu

REQUIRED_KEYS = (
    "TRAIN_DATA_PATH",
    "TEST_DATA_PATH",
    "MODEL_PATH",
    "SUBMISSION_PATH",
    "LOG_PATH",
    "FEATURE_PATH",
    "PIPELINE_FILE",
    "RANDOM_STATE",
    "N_JOBS")


def create_paths_if_necessary(settings):
    for key, setting in settings.items():
        if 'path' in key.lower():
            fu.create_path(setting)


def get_file_components(fixed_settings):
    """
    Creates a list of required keys + values. it's based on REQUIRED_KEYS
    :param fixed_settings: Setting json processed file
    :return: List of required keys + values
    """
    file_components = []
    for key, setting in fixed_settings.items():
        if 'path' not in key.lower() and key in REQUIRED_KEYS:
            if isinstance(setting, dict):
                file_components.append(":".join(get_file_components(setting)))
            else:
                file_components.append(":".join((str(key).lower(), str(setting))))

    return file_components


def get_optional_file_components(fixed_settings):
    """
    Creates a dict of non-required keys + values. it's based on REQUIRED_KEYS
    :param fixed_settings: Setting json processed file
    :return: Dict of non-required keys + values
    """
    optional_file_components = {}
    for key, setting in fixed_settings.items():
        if 'path' not in key.lower() and key not in REQUIRED_KEYS:
            if isinstance(setting, dict):
                optional_file_components.update(get_optional_file_components(setting))
            else:
                optional_file_components[key.lower()] = str(setting)

    return optional_file_components


def fix_settings(settings, root_dir):
    """
    Goes through the settings dictionary and makes sure the paths are correct.
    :param settings: A dictionary with settings, usually obtained from SETTINGS.json in the root directory.
    :param root_dir: The root path to which any path should be relative.
    :return: A settings dictionary where all the paths are fixed to be relative to the supplied root directory.
    """
    fixed_settings = {}
    for key, setting in settings.items():
        if 'path' in key.lower():
            if isinstance(setting, str):
                setting = os.path.join(root_dir, setting)
            elif isinstance(setting, list):
                setting = [os.path.join(root_dir, path) for path in setting]
        fixed_settings[key] = setting
        # print("k {} - s {}".format(key, setting))
    return fixed_settings


def get_settings(settings_path):
    """
    Reads the given json settings file and makes sure the path in it are correct.
    Creates path variables if necessary
    :param settings_path: The path to the json file holding the settings.
    :return: A dictionary with settings.
    """
    with open(settings_path) as settings_fp:
        settings = json.load(settings_fp)
    root_dir = os.path.dirname(settings_path)
    fixed_settings = fix_settings(settings, root_dir)

    create_paths_if_necessary(fixed_settings)

    return fixed_settings
