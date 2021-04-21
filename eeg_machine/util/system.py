#!/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------------------
Script:    Module to process settings.json
------------------------------------------------------------------------------------------------------
"""

# Generics
import json
import os
import logging
import platform
import sys

eeg_logger = logging.getLogger(__name__)


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
    return fixed_settings


def get_settings(settings_path):
    """
    Reads the given json settings file and makes sure the path in it are correct.
    :param settings_path: The path to the json file holding the settings.
    :return: A dictionary with settings.
    """
    with open(settings_path) as settings_fp:
        settings = json.load(settings_fp)
    root_dir = os.path.dirname(settings_path)
    return fix_settings(settings, root_dir)


def log_system_information():
    """
    Prints all necessary system information
    :return:
    """

    eeg_logger.info(platform.machine())
    eeg_logger.info(platform.version())
    eeg_logger.info(platform.platform())
    eeg_logger.info(platform.uname())
    eeg_logger.info(platform.system())
    eeg_logger.info(platform.processor())


def log_requirements():
    """
    Prints on logger the information about the version of all the imported modules

    :return: None
    """
    for name, module in sorted(sys.modules.items()):
        if hasattr(module, '__version__'):
            eeg_logger.debug('{0} :: {1}'.format(name, module.__version__))
