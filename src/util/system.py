import json
import platform
import logging
import logging.handlers
import os
import datetime
import sys

import src
from src.util import file_utils

print(platform.machine())
print(platform.version())
print(platform.platform())
print(platform.uname())
print(platform.system())
print(platform.processor())


'''
x86_64
Darwin Kernel Version 20.3.0: Thu Jan 21 00:07:06 PST 2021; root:xnu-7195.81.3~1/RELEASE_X86_64
macOS-10.16-x86_64-i386-64bit
uname_result(system='Darwin', node='ar-ws-01', release='20.3.0', version='Darwin Kernel Version 20.3.0: Thu Jan 21 00:07:06 PST 2021; root:xnu-7195.81.3~1/RELEASE_X86_64', machine='x86_64')
Darwin
i386
'''


def setup_logging(name, timestamp, file_components, optional_file_components, args):
    """
    Sets up the logger for the classification.
    :param name: Logger insance name
    :param timestamp: The timestamp to apply to the file
    :param file_components: mandatory file name dict
    :param optional_file_components: optional file name parts
    :param args: a dictionary with the arguments which are used by the classifier. This dict will be modified,
                 removing items which shouldn't be sent to the classification function.
    :return: None
    """

    log_dir = args['LOG_PATH']
    del args['LOG_PATH']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("Logging to {}/ named: {}".format(log_dir, name))

    log_file = file_utils.generate_filename('clog', '.log',
                                            components=file_components,
                                            optional_components=optional_file_components,
                                            timestamp=timestamp)

    log_file_hist = file_utils.generate_filename('clog', '_hist.log',
                                                 components=file_components,
                                                 optional_components=optional_file_components,
                                                 timestamp=timestamp)

    # Logger
    my_log = logging.getLogger(name)
    my_log.propagate = False
    my_log.handlers = []
    my_log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)s] (%(funcName)s) :: '
                                  '%(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # Handlers
    fh_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(fh_path)
    file_handler.setFormatter(formatter)

    fh_hist_path = os.path.join(log_dir, log_file_hist)
    file_handler_hist = logging.handlers.RotatingFileHandler(fh_hist_path, 'a', 512 * 1024, backupCount=20)
    file_handler_hist.setLevel(logging.DEBUG)
    file_handler_hist.setFormatter(formatter)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)

    my_log.addHandler(file_handler)
    my_log.addHandler(file_handler_hist)
    my_log.addHandler(std_handler)


def fix_settings(settings, root_dir):
    """
    Goes through the settings dictionary and makes sure the paths are correct.
    :param settings: A dictionary with settings, usually obtained from SETTINGS.json in the root directory.
    :param root_dir: The root path to which any path should be relative.
    :return: A settings dictionary where all the paths are fixed to be relative to the supplied root directory.
    """
    fixed_settings = dict()
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
    fixed_settings = fix_settings(settings, root_dir)
    return fixed_settings


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the SETTINGS.json to use for the training")

    args = r'../../SETTINGS.json'  # parser.parse_args()
    settings = get_settings(args)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    file_components = [settings['FEATURE_TYPE'],
                       'svm',
                       'frame_length_{}'.format(12)]
    optional_file_components = {'standardized': True}

    # Setup logging stuff, this removes 'log_dir' from the dictionary
    setup_logging(src.get_logger_name(), timestamp, file_components, optional_file_components, settings)
    eeg_logger = logging.getLogger(src.get_logger_name())
    print(eeg_logger)

    eeg_logger.info("Extracting Features")


if __name__ == '__main__':
    main()
