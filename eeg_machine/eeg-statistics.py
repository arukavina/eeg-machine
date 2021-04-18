#!/bin/env python
# -*- coding: utf-8 -*-

"""
Module for running basic segment statistics.
"""

# Libs
import datetime
import logging

# Own
from src import setup_logging
from src.features import basic_segment_statistics as bss

__author__ = 'Andrei Rukavina'
__copyright__ = '2021, EGG-Machine'
__credits__ = ['Andrei Rukavina']
__license__ = 'GPL'
__version__ = '0.3.0'
__email__ = 'rukavina.andrei@gmail.com'
__status__ = 'dev'


def main():
    import argparse

    parser = argparse.ArgumentParser(description="""Script for generating statistics about the segments""")

    parser.add_argument("feature_folder",
                        help="""The folder containing the features to be analyzed""")
    parser.add_argument("--glob-suffix",
                        help=("Unix-style glob patterns to select which files to run the analysis over."
                              "This suffix will be appended to the class label (eg 'interictal') so should not be "
                              "expressed here. "
                              "Be sure to encase the pattern in \" so it won't be expanded by the shell."),
                        dest='glob_suffix', default='*')
    parser.add_argument("--csv-directory",
                        help="Which directory the statistics CSV file be written to.",
                        dest='csv_directory',
                        default='../../data/segment_statistics')
    parser.add_argument("--metrics",
                        nargs='+',
                        help="A selection of statistics to collect",
                        choices=bss.get_default_metrics().keys(),
                        dest='subset')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    setup_logging(__name__, timestamp)
    eeg_logger = logging.getLogger(__name__)

    eeg_logger.info("Calculating segment statistics")
    bss.calculate_statistics(args.feature_folder, args.csv_directory, args.glob_suffix, args.subset)


if __name__ == '__main__':
    main()
