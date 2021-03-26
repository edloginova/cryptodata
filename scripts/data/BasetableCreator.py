#!/usr/bin/env python
"""
Creates a basetable using aggregated text features from TextFeatureExtractor, price and Google trends data.
Saves versions with preserved NaN values, dropped NaNs, filled-in (with 0) and interpolated.
"""

import os
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from helper_functions import parse_topic_filename, generate_topic_filename

__author__ = "Ekaterina Loginova, Guus van Heijningen"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"


class BasetableCreator:
    """ This class is constructed to merge the gathered data into the final basetable """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.dump_folder = os.path.join(".", "data")
        if not os.path.exists(self.dump_folder):
            os.makedirs(self.dump_folder)
        pass

    def create_basetable(self, filename: str, text_features_filename: str,
                         google_trends_filename: str, price_data_filename: str, selected_coins: List[str],
                         save: bool = False):
        """ Use the data features extracted from the comments in combination with the financial data from CryptoCompare
        and the trends data from Google Trends to create the final base table for prediction analysis

        :param filename: filepath to save the basetable to
        :param text_features_filename: filepath to the file containing the extracted comment features
        :param google_trends_filename: filepath to the file containing the Google Trends data
        :param price_data_filename: filepath to the file containing the CryptoCompare financial data
        :param selected_coins: the coins to be used in further analysis to filter the gathered data on
        :param save: whether to save the final basetable or not
        """
        filename = os.path.join(self.dump_folder, filename)
        text_features_filename = os.path.join(self.dump_folder, text_features_filename)
        google_trends_filename = os.path.join(self.dump_folder, google_trends_filename)
        price_data_filename = os.path.join(self.dump_folder, price_data_filename)

        price_data = pickle.load(open(price_data_filename, 'rb'))
        if self.verbose:
            print('Loaded price features. Time range:')
            price_data_time = price_data.time.apply(datetime.fromtimestamp)
            print(len(price_data), min(price_data_time), max(price_data_time))
        price_data['return'] = price_data.groupby('coin').close.pct_change()
        price_data['date'] = pd.to_datetime(price_data['time'], unit='s')
        price_data['spread'] = price_data['high'] - price_data['low']
        price_data['direction'] = price_data['return'].apply(self.directional_return)
        basetable = price_data[['date', 'coin', 'direction', 'return', 'close', 'volumefrom', 'volumeto', 'spread']]
        basetable = basetable.loc[basetable.coin.isin(selected_coins)]
        if self.verbose:
            print('Added price features. Dataset size:', len(basetable))

        trend_data = pickle.load(open(google_trends_filename, 'rb'))
        if self.verbose:
            print('Loaded trend features. Time range:')
            trend_data_time = trend_data.date.apply(pd.Timestamp)
            print(len(trend_data), min(trend_data_time), max(trend_data_time))
        #         trend_data = trend_data.rename(columns={'index': 'date', 'Name': 'coin'})
        trend_data = trend_data.rename(columns={'Name': 'coin'})
        trend_data['date'] = pd.to_datetime(trend_data['date'])
        trend_data.loc[trend_data.coin == 'XRP', 'CoinName'] = 'Ripple'
        basetable = basetable.merge(trend_data[['date', 'coin', 'CoinName', 'interest']], on=['coin', 'date'])
        if self.verbose:
            print('Added trend features. Dataset size:', len(basetable))

        text_data = pickle.load(open(text_features_filename, 'rb'))
        if self.verbose:
            print('Loaded text features. Time range:')
            text_data_time = text_data.date.apply(pd.Timestamp)
            print(len(text_data), min(text_data_time), max(text_data_time))
        timestamp_column = ''
        if 'comment_time' in text_data.columns:
            timestamp_column = 'comment_time'
        elif 'published_on' in text_data.columns:
            timestamp_column = 'published_on'
        elif 'date' in text_data.columns:
            timestamp_column = 'date'
        elif 'created_utc' in text_data.columns:
            timestamp_column = 'created_utc'
        elif 'post_time' in text_data.columns:
            timestamp_column = 'post_time'
        text_data.rename({timestamp_column: 'date'}, inplace=True, axis=1)
        basetable = basetable.merge(text_data, on=['date', 'CoinName'], how='left').set_index('date')
        # Calculate the ratio of total positive divided by the total of negative comments
        basetable['text_polarity_tot_posneg'] = basetable['text_polarity_tot_pos'] / basetable['text_polarity_tot_neg']
        # Calculate the ratio of the sum of positive scores dived by the sum of negative scores
        basetable['text_polarity_posneg'] = basetable['text_polarity_positive'] / basetable['text_polarity_negative']
        basetable['text_subjectivity_tot_posneg'] = basetable['text_subjectivity_tot_pos'] / basetable[
            'text_subjectivity_tot_neg']
        basetable['text_subjectivity_posneg'] = basetable['text_subjectivity_positive'] / basetable[
            'text_subjectivity_negative']
        if self.verbose:
            print('Added text features. Dataset size:', len(basetable))

        basetable = basetable.replace(np.inf, 0)
        basetable_interpolated = basetable.interpolate()
        lb = len(basetable)
        basetable_nonan = basetable.dropna()
        basetable_fillnan = basetable.fillna(0)
        print('Dropped NaN: {} -> {}'.format(lb, len(basetable)))
        print(min(basetable.index), max(basetable.index))
        if self.verbose:
            print('Filled in missing values by interpolation.')
        if save:
            basetable.to_pickle(filename)
            basetable_nonan.to_pickle(filename.replace('.p', '_nonan.p'))
            basetable_fillnan.to_pickle(filename.replace('.p', '_fillnan.p'))
            basetable_interpolated.to_pickle(filename.replace('.p', '_interpolated.p'))
            if self.verbose:
                print('Saved pickle file with basetable.')
        return basetable

    @staticmethod
    def directional_return(ret):
        if ret > 0:
            return True
        return False


# python BasetableCreator.py -d ../../data/processed/news/
# python BasetableCreator.py -d ../../data/processed/bitcointalk/date_no_structure/ -f aggregated_JST_text_features.p -o JST_new_basetable.p
# python BasetableCreator.py -d ../../data/processed/news/ -f aggregated_JST_text_features.p -o JST_new_basetable.p
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="input_dir",
                           help="The path to the directory with input files.", default='')
    argparser.add_argument("-o", "--output_filename", dest="output_filename", default='',
                           help='The path for the output pickle file(s) of this program with the basetable.')
    argparser.add_argument("-f", "--filename", dest="text_features_filename",
                           help="The path to the pickle file with daily aggregated text features.",
                           default='aggregated_text_features.p')
    argparser.add_argument("-ft", "--trends_filename", dest="trends_data_filename",
                           help="The path to the pickle file with daily google trends features.",
                           default='google_trends_data.p')
    argparser.add_argument("-fp", "--price_filename", dest="price_data_filename",
                           help="The path to the pickle file with daily price features", default='price_data.p')
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')
    argparser.add_argument("-s", "--save", dest="save", type=bool, default=True,
                           help='Whether to save the final output dataframe as a pickle file.')

    args = argparser.parse_args()
    args = vars(args)
    verbose = args['verbose']
    MyBC = BasetableCreator(verbose=True)
    if args['input_dir'][-1] != '/':
        args['input_dir'] = args['input_dir'] + '/'
    MyBC.dump_folder = args['input_dir']
    topic_mode, num_topics, num_sentiments = parse_topic_filename(args['text_features_filename'])
    args['output_filename'] = generate_topic_filename(topic_mode, num_topics, num_sentiments,
                                                      current_filename=args['output_filename'])
    basetable = MyBC.create_basetable(args['output_filename'], text_features_filename=args['text_features_filename'],
                                      google_trends_filename=args['trends_data_filename'],
                                      price_data_filename=args['price_data_filename'],
                                      selected_coins=['BTC', 'ETH', 'XRP', 'LTC', 'NEO', 'IOTA'], save=args['save'])
