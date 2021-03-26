#!/usr/bin/env python
"""
Extract sentiment (polarity + objectivity) features, part of speech tags, subjects, adds topic features from TopicExtractor.
Optionally aggregates these features daily to use in BasetableCreator.
"""

import multiprocessing
import os
import pickle
import re
import sys
import time
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from typing import List
import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions import parse_topic_filename, slack_message

__author__ = "Ekaterina Loginova, Guus van Heijningen"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()
NUM_CORES = multiprocessing.cpu_count()
SLACK_USER = 'ekaterina.loginova'


def parallelize_dataframe(df, func, n_cores=NUM_CORES):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


class TextFeatureExtractor:

    def __init__(self, verbose: bool = False, dump_folder: str = "../../data/interim/experiments/"):
        self.verbose = verbose
        self.dump_folder = dump_folder
        if not os.path.exists(self.dump_folder):
            os.makedirs(self.dump_folder)

    def extract_comment_features(self, coinlist_filename: str,
                                 texts_filename: str, parents_filename: str = 'comment_id_data.p',
                                 nlp_analyser: str = 'nltk', thread_structure: str = 'post+comment',
                                 topic_filename: str = '', aggregate: bool = False,
                                 save: bool = False, save_intermediate: bool = False, output_filename: str = '',
                                 small_data: bool = False, features: str = 'all', num_data: int = 200,
                                 topic_mode: str = '',
                                 drop_nosubj=False, parallelize=True):
        """ Extract subjects, sentiment and other features from the Reddit data. This function does a large amount of
        processing to extract the subject, assign a sentiment score and aggregate the features into the required
        granularity.
        :param coinlist_filename: the path to the file with the list of coin names
        :param texts_filename: the path to the file with comments' texts
        :param nlp_analyser: whether to use NLTK or SpaCy for POS-tagging and tokenization
        :param thread_structure: whether to look at the thread parent message to determine subjects for comments in this thread
        :param aggregate: whether to aggregate by coin and day
        :param save: whether to save the final dataframe
        :param output_filename: the path to the file with the final output (extracted text features per comment)
        :param small_data: whether to use a small subset of the data (useful to speed up debugging)
        :param features: which text features to extract: polarity, subjectivity, sentiment (=polarity+subjectivity), topic, all, none
        :param topic_mode: whether LDA/JST/TSLDA are used in extracted topics; can be inferred from topic_filename.
        :param topic_filename: the path to the file with topic distribution per comment.
        :param parents_filename: the path to the file with  the information about parent-comment id (to use thread structure for subject identification)
        """
        self.drop_nosubj = drop_nosubj
        self.thread_structure = thread_structure
        if len(parents_filename) > 0:
            parents_filename = os.path.join(self.dump_folder, parents_filename)
        else:
            self.thread_structure = 'comment'
        # infer topic mode from the topic filename if needed
        if topic_filename != '' and topic_mode == '':
            topic_mode, num_topics, num_sentiments = parse_topic_filename(topic_filename)
            topic_filename = os.path.join(self.dump_folder, topic_filename)
        print('topic', topic_filename, topic_mode, num_topics)
        add_output_filename = topic_mode
        if num_topics != '':
            add_output_filename += '_t' + num_topics
        if num_sentiments != '':
            add_output_filename += '_s' + num_sentiments
        output_filename = add_output_filename + output_filename

        if aggregate:
            aggr_output_filename = 'aggregated_' + output_filename
        basic_output_filename = 'basic_' + output_filename
        meta_output_filename = 'meta_' + output_filename
        output_filename = os.path.join(self.dump_folder, output_filename)
        basic_output_filename = os.path.join(self.dump_folder, basic_output_filename)
        meta_output_filename = os.path.join(self.dump_folder, meta_output_filename)
        if aggregate:
            aggr_output_filename = os.path.join(self.dump_folder, aggr_output_filename)
        if '../data/' not in texts_filename:
            texts_filename = os.path.join(self.dump_folder, texts_filename)
        coinlist_filename = os.path.join(self.dump_folder, coinlist_filename)

        text_data = self.load_data(parents_filename, texts_filename, num_data, small_data)
        print(text_data.head())
        self.text_column = ''
        if 'text' in text_data.columns:
            self.text_column = 'text'
        elif 'body' in text_data.columns:
            self.text_column = 'body'

        start_time = time.time()

        try:
            my_file = Path(basic_output_filename)
            if my_file.exists():
                text_data = pickle.load(open(basic_output_filename, 'rb'))
                print("Loaded sentiment and POS tag text features.")
            else:
                if parallelize:
                    text_data = parallelize_dataframe(text_data, self.calculate_sentiment_features, n_cores=4)
                    text_data = parallelize_dataframe(text_data, self.calculate_basic_features, n_cores=4)
                else:
                    text_data = self.calculate_sentiment_features(text_data)
                    text_data = self.calculate_basic_features(text_data)
                    if self.verbose:
                        slack_message('Extracted sentiment and POS features. (Folder: {})'.format(self.dump_folder), SLACK_USER)
                if save and save_intermediate:
                    try:
                        pickle.dump(text_data, open(basic_output_filename, 'wb'))
                    except Exception as e:
                        print(e)
                        joblib.dump(text_data, open(basic_output_filename, 'wb'))
        except Exception as e:
            slack_message('Error {} during extracting sentiment and POS features. (Folder: {})'.format(e, self.dump_folder),
                          SLACK_USER)
        print(text_data.head())
        try:
            my_file = Path(meta_output_filename)
            if my_file.exists():
                text_data = pickle.load(open(meta_output_filename, 'rb'))
                print("Loaded topic and coin subject text features.")
            else:
                text_data = self.extract_coin_subjects(text_data, coinlist_filename, nlp_analyser)
                print(text_data.head())
                if save and save_intermediate:
                    try:
                        pickle.dump(text_data, open(meta_output_filename, 'wb'))
                    except Exception as e:
                        print(e)
                        joblib.dump(text_data, open(meta_output_filename, 'wb'))
                if 'id' in text_data.columns:
                    text_data.rename({'id': 'comment_id'}, inplace=True, axis=1)
                if 'comment_id' not in text_data.columns:
                    text_data['comment_id'] = text_data.index
                if self.drop_nosubj:
                    text_data = text_data.loc[text_data['subject_coinname'].notnull()]
                text_data['num_subjects'] = text_data['subject_coinname'].apply(len)
                if not self.drop_nosubj:
                    text_data = text_data.reindex(text_data.index.repeat(text_data.num_subjects.replace(0, 1)))
                else:
                    text_data = text_data.reindex(text_data.index.repeat(text_data.num_subjects))
                    if self.verbose:
                        print('Dropped texts without subjects', len(text_data))
                text_data['subject'] = (text_data.groupby(level=0)['subject_coinname']
                                        .transform(lambda x: [x.iloc[0][i] for i in range(len(x))]))
                text_data = text_data.loc[-text_data[['comment_id', 'subject']].duplicated()]

                if self.verbose:
                    print('Removed duplicates (by comment_id + subject).', len(text_data))
                if topic_mode != '':
                    text_data = self.add_topic_features(text_data, features, topic_filename, topic_mode)
                    print(text_data.head())
                if self.verbose:
                    slack_message('Extracted coin and topic features. (Folder: {})'.format(self.dump_folder), SLACK_USER)
                if save and save_intermediate:
                    try:
                        pickle.dump(text_data, open(meta_output_filename, 'wb'))
                    except Exception as e:
                        print(e)
                        joblib.dump(text_data, open(meta_output_filename, 'wb'))
        except Exception as e:
            slack_message('Error {} during extracting coin and topic features. (Folder: {})'.format(e, self.dump_folder),
                          SLACK_USER)
            return 0
        if save:
            if (aggregate and save_intermediate) or (not aggregate):
                try:
                    pickle.dump(text_data, open(output_filename, 'wb'))
                except Exception as e:
                    print(e)
                    print('Failed to save the file, trying to parallelize.')
                    joblib.dump(text_data, open(output_filename, 'wb'))
            if aggregate:
                my_file = Path(aggr_output_filename)
                if my_file.exists():
                    text_data = pickle.load(open(aggr_output_filename, 'rb'))
                    print("Loaded aggregated text features.")
                else:
                    text_data = self.aggregate_features(text_data)
                    try:
                        pickle.dump(text_data, open(aggr_output_filename, 'wb'))
                    except Exception as e:
                        print(e)
                        print('Failed to save the file, trying to parallelize.')
                        joblib.dump(text_data, open(aggr_output_filename, 'wb'))
        elapsed_time = time.time() - start_time
        print('Finished text extractor. Time:', elapsed_time)
        if self.verbose:
            slack_message('Finished text extractor. Time: {} (Folder: {})'.format(elapsed_time, self.dump_folder), SLACK_USER)
        return text_data

    def load_data(self, parents_filename, texts_filename, num_data, small_data):

        if self.thread_structure == 'post+comment':
            post_data = pickle.load(open(parents_filename, 'rb'))
            if small_data:
                post_data = post_data.iloc[:num_data]
                print("Small data:", len(post_data))
            # post_data = post_data.rename(columns={'id': 'post_id', 'score': 'post_score', 'body': 'parent_body' ,'date': 'parent_date'})
            post_data = post_data.rename(
                columns={'id': 'parent_id', 'score': 'post_score', 'title': 'parent_body', 'date': 'parent_date'})
            post_data.parent_id = post_data.parent_id.apply(lambda x: x.split('_')[-1])
            post_data[['parent_body']] = post_data[['parent_body']].fillna(value='')
            # post_data = post_data.reset_index(drop=True)
            post_data = post_data[~(post_data.comment_ids.apply(len) == 0)]
            post_data.drop(['comment_ids', 'num_comments'], axis=1, inplace=True)
            # post_data = post_data.loc[post_data.index.repeat(post_data['num_comments'])]
            # post_data = post_data.assign(comment_id=(
            #     post_data.groupby(level=0)['comment_ids'].transform(
            #         lambda x: [int(x.iloc[0][ind]) for ind in range(len(x))])))
            if 'parent_date' in post_data.columns:
                post_data['post_time'] = post_data.parent_date
            else:
                post_data['post_time'] = post_data.created_utc.apply(datetime.fromtimestamp)
            if self.verbose:
                print("Loaded post data.", len(post_data))

            comment_data = pickle.load(open(texts_filename, 'rb'))
            if small_data:
                comment_data = comment_data[comment_data.parent_id.isin(post_data.parent_id)]
                # comment_data = comment_data[comment_data.id.isin(post_data.comment_id)]
            comment_data = comment_data.rename(columns={'id': 'comment_id', 'score': 'comment_score'})
            if 'date' in comment_data.columns:
                comment_data['comment_time'] = comment_data.date
            else:
                comment_data['comment_time'] = comment_data.created_utc.apply(datetime.fromtimestamp)
            comment_data.parent_id = comment_data.parent_id.apply(lambda x: x.split('_')[-1])

            if self.verbose:
                print("Loaded comment data.", len(comment_data))

            # text_data = comment_data.merge(post_data, on='comment_id')
            text_data = comment_data.merge(post_data, on='parent_id')
            text_data = text_data.dropna(subset=['body'])
            text_data = text_data.reset_index(drop=True)
            # Free up memory space
            del post_data
            del comment_data
        else:
            text_data = pickle.load(open(texts_filename, 'rb'))
            if small_data:
                text_data = text_data.iloc[:num_data]
                print("Small data:", len(text_data))

        if self.verbose:
            print("Created/loaded text data dataframe.", len(text_data))

        if self.thread_structure == 'post+comment':
            text_data = text_data.drop_duplicates(['comment_id'])
        else:
            text_data = text_data.drop_duplicates(['id'])

        print('Removed duplicates', len(text_data))

        return text_data

    def add_topic_features(self, text_data, features='all', topic_filename='', topic_mode='LDA'):
        """Merge by comment id with per-text topic distributions."""
        if 'topic' in features or 'all' in features:
            if topic_filename != '':
                if topic_mode in ['JST', 'TSLDA']:
                    comments_topics = pickle.load(open(topic_filename, 'rb'))
                    comments_topics = comments_topics[
                        ['id'] + [x for x in comments_topics.columns if 'sentiment_scores' in x or 'topic_scores' in x]]
                    if 'id' in comments_topics.columns:
                        comments_topics.rename({'id': 'comment_id'}, inplace=True, axis=1)
                    text_data = text_data.merge(comments_topics, on=['comment_id'])
                else:
                    comments_topics = pickle.load(open(topic_filename, 'rb'))
                    if 'id' in comments_topics.columns:
                        comments_topics.rename({'id': 'comment_id'}, inplace=True, axis=1)
                    if 'comment_id' not in comments_topics.columns:
                        comments_topics['comment_id'] = comments_topics.index
                    if 'topic' in comments_topics.columns and 'topic0' not in comments_topics.columns:
                        if self.verbose:
                            print('\tSplitting topic column into several columns.')
                        num_topics = len(comments_topics.topic.iloc[0])
                        comments_topics['topic'] = comments_topics['topic'].apply(lambda x: {y[0]: y[1] for y in x})
                        for i in range(0, num_topics):
                            comments_topics['topic' + str(i)] = [x[i] if i in x.keys() else 0 for x in
                                                                 comments_topics['topic'].values]
                    if 'topic' in comments_topics.columns:
                        comments_topics.drop(['topic'], axis=1, inplace=True)
                    topic_columns = [x for x in comments_topics.columns if 'topic' in x]
                    comments_topics = comments_topics[['comment_id'] + topic_columns]
                    comments_topics = comments_topics[comments_topics.comment_id.isin(text_data.comment_id)]
                    text_data = text_data.merge(comments_topics, on=['comment_id'])
        return text_data

    def calculate_basic_features(self, text_data, nlp_analyser='nltk'):
        """POS-tag and tokenize texts."""
        if nlp_analyser == 'spacy':
            text_data['comment_nlp'] = text_data[self.text_column].apply(nlp)
            text_data['comment_tagged'] = text_data['comment_nlp'].apply(
                lambda x: [(token.text, token.pos_) for token in x])

            del text_data[self.text_column], text_data['comment_nlp']

            if self.thread_structure == 'post+comment':
                text_data['post_nlp'] = text_data['parent_body'].apply(nlp)
                text_data['post_tagged'] = text_data['post_nlp'].apply(
                    lambda x: [(token.text, token.pos_) for token in x])

                del text_data['parent_body'], text_data['post_nlp']
        elif nlp_analyser == 'nltk':
            text_data['comment_words'] = text_data[self.text_column].apply(word_tokenize)
            text_data['comment_tagged'] = text_data['comment_words'].apply(nltk.pos_tag)

            del text_data[self.text_column]
            del text_data['comment_words']

            if self.thread_structure == 'post+comment':
                text_data['post_words'] = text_data['parent_body'].apply(word_tokenize)
                text_data['post_tagged'] = text_data['post_words'].apply(nltk.pos_tag)

                del text_data['parent_body']
                del text_data['post_words']

        if self.verbose:
            print("Tokenized and POS-tagged.", len(text_data))
        return text_data

    def calculate_sentiment_features(self, text_data, features='all'):
        """Calculate sentiment (polarity and subjectivity) scores."""
        sid = SentimentIntensityAnalyzer()
        if 'polarity' in features or 'sentiment' in features or features == 'all':
            text_data['text_polarity'] = text_data[self.text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        if 'subjectivity' in features or 'sentiment' in features or features == 'all':
            text_data['text_subjectivity'] = text_data[self.text_column].apply(
                lambda x: TextBlob(x).sentiment.subjectivity)
        if self.verbose:
            print("Calculated sentiment features.", len(text_data))
        return text_data

    def aggregate_features(self, comment_features):
        """Aggregate extracted text features (daily)."""
        timestamp_column = ''
        if 'comment_time' in comment_features.columns:
            timestamp_column = 'comment_time'
        elif 'published_on' in comment_features.columns:
            timestamp_column = 'published_on'
        elif 'date' in comment_features.columns:
            timestamp_column = 'date'
        elif 'created_utc' in comment_features.columns:
            timestamp_column = 'created_utc'
        elif 'post_time' in comment_features.columns:
            timestamp_column = 'post_time'
        if timestamp_column == 'created_utc':
            comment_features[timestamp_column] = comment_features[timestamp_column].apply(datetime.fromtimestamp).apply(
                pd.Timestamp)
        else:
            comment_features[timestamp_column] = comment_features[timestamp_column].apply(pd.Timestamp)
        month_day = comment_features[timestamp_column].map(
            lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day))
        aggfunc_dict = {'text_polarity': {'tot_pos': lambda x: self.tot_pos(x, 0),
                                          'tot_neg': lambda x: self.tot_neg(x, -0),
                                          'positive': lambda x: self.positive(x, 0),
                                          'negative': lambda x: self.negative(x, -0),
                                          'sum': np.mean, 'len': len},
                        'text_subjectivity': {'tot_pos': lambda x: self.tot_pos(x, 0.5),
                                              'tot_neg': lambda x: self.tot_neg(x, 0.5),
                                              'positive': lambda x: self.positive(x, 0.5),
                                              'negative': lambda x: self.negative(x, 0.5),
                                              'sum': np.mean}}
        topic_columns = [x for x in comment_features.columns if 'topic' in x]
        absa_columns = [x for x in comment_features.columns if 'sentiment_scores' in x]
        aggfunc_dict.update({x: [np.mean] for x in topic_columns})
        aggfunc_dict.update({x: [np.mean] for x in absa_columns})

        comment_features = pd.pivot_table(comment_features,
                                          values=['text_polarity', 'text_subjectivity'] + topic_columns + absa_columns,
                                          index=['subject', month_day],
                                          aggfunc=aggfunc_dict,
                                          fill_value=0)
        comment_features.columns = ['_'.join(col) for col in comment_features.columns]
        comment_features = comment_features.reset_index()
        comment_features = comment_features.rename({'text_polarity_len': 'len'}, axis=1)
        comment_features = comment_features.rename({'subject': 'CoinName'}, axis=1)
        comment_features = comment_features.rename({timestamp_column: 'date'}, axis=1)
        comment_features['date'] = pd.to_datetime(comment_features.date)
        return comment_features

    def extract_coin_subjects(self, text_data, coinlist_filename, nlp_analyser):

        coinlist = pickle.load(open(coinlist_filename, 'rb'))
        # Fetch the list of coins and create lists with all their names and tickersymbols
        coin_names = coinlist.CoinName.values
        lower_coin_names = coinlist.CoinName.str.lower().values
        names = coinlist.Name.values

        coin_names_split, coin_names_split_lower, splitted = [], [], []
        for coin in coin_names:
            if ' ' in coin:
                splitted += [coin]
                if nlp_analyser == 'spacy':
                    doc = nlp(coin)
                    tagged = (doc[0].text, doc[0].pos_)
                elif nlp_analyser == 'nltk':
                    tagged = nltk.pos_tag(nltk.word_tokenize(coin))
                for word in tagged:
                    if word[1].startswith('NN') and (word[0].isalpha() or word[0].isnumeric()) and word[0] not in names:
                        coin_names_split += [word[0]]
                        coin_names_split_lower += [word[0].lower()]

        # Put names that occur multiple times on a blacklist so duplicates cannot occur
        count_words = Counter(coin_names_split)
        coin_blacklist = []
        for word in count_words.most_common():
            if word[1] > 1:
                coin_blacklist += [word[0]]

        # Remove special characters from coin abbreviations before identifying names within posts and comments
        for i, name in enumerate(names):
            if '*' in name:
                names[i] = re.sub(r'\W+', '', name)

        if self.verbose:
            print("Loaded and pre-processed coin names.")

        if self.thread_structure == 'post+comment':
            post_coins = [{'CoinName': set(),
                           'Name': set(),
                           'LowerCoinName': set(),
                           'CoinNamesSplit': set(),
                           'LowerLowerCoinName': set()} for _ in range(len(text_data['post_tagged']))]
            for i, post in enumerate(text_data['post_tagged']):
                for word in post:
                    if word[0] in coin_names:
                        post_coins[i]['CoinName'].add(word[0])
                    if word[0] in lower_coin_names:
                        post_coins[i]['LowerCoinName'].add(word[0])
                    if word[0] in names:
                        post_coins[i]['Name'].add(word[0])
                    if nlp_analyser == 'spacy':
                        if word[0] in coin_names_split and word[1] in set(['NOUN', 'PROPN']) and word[
                            0] not in coin_blacklist:
                            post_coins[i]['CoinNamesSplit'].add(word[0])
                    elif nlp_analyser == 'nltk':
                        if word[0] in coin_names_split and word[1].startswith('NN') and word[0] not in coin_blacklist:
                            post_coins[i]['CoinNamesSplit'].add(word[0])

            text_data['PostCoinMatches'] = post_coins

            if self.verbose:
                print("Extracted coin matches for posts.", len(text_data))

        comment_coins = [{'CoinName': set(),
                          'Name': set(),
                          'LowerCoinName': set(),
                          'CoinNamesSplit': set(),
                          'LowerCoinNamesSplit': set()} for _ in range(len(text_data['comment_tagged']))]
        for i, comment in enumerate(text_data['comment_tagged']):
            for word in comment:
                if word[0] in coin_names:
                    comment_coins[i]['CoinName'].add(word[0])
                if word[0] in lower_coin_names:
                    comment_coins[i]['LowerCoinName'].add(word[0])
                if word[0] in names:
                    comment_coins[i]['Name'].add(word[0])
                if nlp_analyser == 'spacy':
                    if word[0] in coin_names_split and word[1] in set(['NOUN', 'PROPN']) and word[
                        0] not in coin_blacklist:
                        comment_coins[i]['CoinNamesSplit'].add(word[0])
                    if (word[0] in coin_names_split_lower and word[1] in set(['NOUN', 'PROPN'])
                            and word[0].capitalize() not in coin_blacklist):
                        comment_coins[i]['LowerCoinNamesSplit'].add(word[0])
                elif nlp_analyser == 'nltk':
                    if word[0] in coin_names_split and word[1].startswith('NN') and word[0] not in coin_blacklist:
                        comment_coins[i]['CoinNamesSplit'].add(word[0])
                    if (word[0] in coin_names_split_lower and word[1].startswith('NN')
                            and word[0].capitalize() not in coin_blacklist):
                        comment_coins[i]['LowerCoinNamesSplit'].add(word[0])

        text_data['CommentCoinMatches'] = comment_coins
        if self.verbose:
            print("Extracted coin matches for comments.", len(text_data))

        del text_data['comment_tagged']
        if self.thread_structure == 'post+comment':
            del text_data['post_tagged']

        if self.thread_structure == 'post+comment':
            text_data['subject'] = text_data[['PostCoinMatches',
                                              'CommentCoinMatches',
                                              'parent_id'
                                              ]].apply(lambda x: self.subject_finder(x, text_data), axis=1)

            del text_data['PostCoinMatches']
        else:
            text_data['subject'] = text_data[['CommentCoinMatches']].apply(lambda x: self.subject_finder(x, text_data),
                                                                           axis=1)

        del text_data['CommentCoinMatches']
        text_data['subject_coinname'] = text_data['subject'].apply(lambda x: self.subject(x, coinlist, splitted))
        text_data.subject_coinname = [x if len(x) > 0 else ['NoName'] for x in text_data.subject_coinname.values]
        if self.verbose:
            print("Extracted subjects.", len(text_data))

        return text_data

    def subject_finder(self, row, df):
        """ This function uses the tree structure in the Reddit threads in order to assign a subject to the specific
        comment. If the comment itself does not seem to discuss a subject by itself, it first checks if any subject is
        identified in the comments that are placed above concerning comment. If yes, the subject discussed in the
        closest parent comment is assigned to the concerning comment. If there are also no subjects observed in the
        parent comments, the post title is used to assign a subject to the comment.

        :param row: the particular row that contains a comment
        :param df: the full data frame to identify parent comments
        Author: Guus van Heijningen
        """
        output = []
        if row['CommentCoinMatches']['CoinName']:
            output = ('CoinName', row['CommentCoinMatches']['CoinName'])
        elif row['CommentCoinMatches']['CoinNamesSplit']:
            output = ('CoinNamesSplit', row['CommentCoinMatches']['CoinNamesSplit'])
        elif row['CommentCoinMatches']['Name']:
            output = ('Name', row['CommentCoinMatches']['Name'])

        if self.thread_structure == 'post+comment':
            while str(row['parent_id']).startswith('d') and output is None:
                parent = str(row['parent_id'])
                row = df.loc[df['comment_id'] == parent].to_dict('records')[0]
                if row['CommentCoinMatches']['CoinName']:
                    output = ('CoinName', row['CommentCoinMatches']['CoinName'])
                elif row['CommentCoinMatches']['CoinNamesSplit']:
                    output = ('CoinNamesSplit', row['CommentCoinMatches']['CoinNamesSplit'])
                elif row['CommentCoinMatches']['Name']:
                    output = ('Name', row['CommentCoinMatches']['Name'])

            if output is None and row['PostCoinMatches']:
                if row['PostCoinMatches']['CoinName']:
                    output = ('CoinName', row['PostCoinMatches']['CoinName'])
                elif row['PostCoinMatches']['CoinNamesSplit']:
                    output = ('CoinNamesSplit', row['PostCoinMatches']['CoinNamesSplit'])
                elif row['PostCoinMatches']['Name']:
                    output = ('Name', row['PostCoinMatches']['Name'])

        return output

    @staticmethod
    def subject_counter(df, top_n):
        return df.value_counts()[:top_n]

    @staticmethod
    def subject(subj, coinlist: pd.DataFrame, splitted: List[str]):
        output = []
        if len(subj) == 0:
            return []
        if subj[0] == 'CoinName':
            for sub in subj[1]:
                output += [sub]
        elif subj[0] == 'Name':
            for sub in subj[1]:
                output += [coinlist.loc[coinlist['Name'] == sub]['CoinName'].iloc[0]]
        elif subj[0] == 'CoinNamesSplit':
            for sub in subj[1]:
                output += [split for split in splitted if sub in split.split()]
        elif subj[0] == 'LowerCoinName':
            for sub in subj[1]:
                output += [sub.capitalize()]
        return output

    @staticmethod
    def tot_pos(column, threshold=0.5):
        total = 0
        for value in column:
            if value > threshold:
                total += 1
        return total

    @staticmethod
    def positive(column, threshold=0.5):
        total = 0
        for value in column:
            if value > threshold:
                total += value
        return total

    @staticmethod
    def tot_neg(column, threshold=0.5):
        total = 0
        for value in column:
            if value < threshold:
                total += 1
        return total

    @staticmethod
    def negative(column, threshold=0.5):
        total = 0
        for value in column:
            if value < threshold:
                total += abs(value)
        return total


# python TextFeatureExtractor.py -d ../../data/processed/news/
# python TextFeatureExtractor.py -d ../../data/processed/bitcointalk/date_structure/ -ts post+comment -topicf unique_comments_5topics.p
# python TextFeatureExtractor.py -d ../../data/processed/bitcointalk/date_no_structure/ -topicf unique_comments_JST.p -sd 1 -of JST_text_features
# python TextFeatureExtractor.py -d ../../data/processed/bitcointalk/date_no_structure/ -topicf unique_comments_TSLDA.p -sd 1 -of TSLDA_text_features
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="dir",
                           help='The path to the directory with input files.')
    argparser.add_argument("-cf", "--coinlist_filename", dest="coinlist_filename", default='coinlist.p',
                           help='The path to the pickle file with the list of coin names (format specified by Guus). Usually it is the same coinlist.p across all data sources.')
    argparser.add_argument("-tf", "--texts_filename", dest="texts_filename", default='unique_comments.p',
                           help='The path to the pickle file with comments (with their text and dates).')
    argparser.add_argument("-of", "--output_filename", dest="output_filename", default='text_features.p',
                           help='The path for the output pickle file(s) of this program with the text features.')
    argparser.add_argument("-topicf", "--topic_filename", dest="topic_filename", default='unique_comments_5topics.p',
                           help='The path to the pickle file with topic distributions per comment.')
    argparser.add_argument("-sd", "--small_data", dest="small_data", type=bool, default=False,
                           help='Whether to use only a small subset of the data. Helpful for debugging.')
    argparser.add_argument("-n", "--num_data", dest="num_data", type=int, default=200,
                           help='How many texts to take for a small subset of the data. Helpful for debugging.')
    argparser.add_argument("-fe", "--features", dest="features", default='all',
                           help='Whether to use only a small subset of the data. Helpful for debugging.')
    argparser.add_argument("-tm", "--topic_mode", dest="topic_mode", default='',
                           help='Whether LDA or ABSA models are used. Will be inferred from topic_filename argument if left empty.')
    argparser.add_argument("-ts", "--thread_structure", dest="thread_structure", default='comment',
                           help='Whether to use parent thread when extracting subjects. Requires comment_id_data.p.')
    argparser.add_argument("-dns", "--drop_nosubj", dest="drop_nosubj", action='store_true',
                           help='Whether to drop the comments with no subjects.')
    argparser.add_argument("-a", "--aggr", dest="aggregate", type=bool, default=True,
                           help='Whether to agrgegate text features daily or not.')
    argparser.add_argument("-s", "--save", dest="save", type=bool, default=True,
                           help='Whether to save the final output dataframe as a pickle file.')
    argparser.add_argument("-si", "--save_intermediate", dest="save_intermediate", type=bool, default=True,
                           help='Whether to save intermediate files. Helpful for debugging and post-changes to preprocessing.')
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false',
                           help='Whether to print the intermediate log messages.')
    argparser.add_argument("-p", "--parallelize", dest="parallelize",
                           help="Whether to parallelize or not.", action='store_false')
    argparser.add_argument("-su", "--slack_user", dest="slack_user", default='ekaterina.loginova')
    if len(sys.argv) == 1:
        argparser.print_help(sys.stderr)
        sys.exit(1)
    args = argparser.parse_args()
    args = vars(args)
    print(args)
    SLACK_USER = args['slack_user']
    verbose = args['verbose']
    if args['dir'][-1] != '/':
        args['dir'] = args['dir'] + '/'
    MyFE = TextFeatureExtractor(verbose=verbose, dump_folder=args['dir'])
    start_time = time.time()
    comment_features = MyFE.extract_comment_features(coinlist_filename=args['coinlist_filename'],
                                                     texts_filename=args['texts_filename'],
                                                     thread_structure=args['thread_structure'],
                                                     aggregate=args['aggregate'],
                                                     save=args['save'],
                                                     save_intermediate=args['save_intermediate'],
                                                     output_filename=args['output_filename'],
                                                     topic_filename=args['topic_filename'],
                                                     small_data=args['small_data'],
                                                     features=args['features'],
                                                     num_data=args['num_data'],
                                                     topic_mode=args['topic_mode'],
                                                     drop_nosubj=args['drop_nosubj'],
                                                     parallelize=args['parallelize'])
    elapsed_time = time.time() - start_time
    if verbose:
        print(elapsed_time)