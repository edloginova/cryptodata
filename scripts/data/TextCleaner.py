#!/usr/bin/env python
"""
Removes stopwords, punctuation. Converts emojis, URLs, currency symbols, numbers in special tokens. Expands contractions.
For ABSA mode, additionally splits in sentences for further use in ProbModelFormatter.
"""


import pickle
import re
import string
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from emoji import UNICODE_EMOJI
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

__author__ = "Ekaterina Loginova"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"

nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "ner"])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
nltk.download('stopwords', quiet=True)
en_stop = set(nltk.corpus.stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()


# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def remove_criteria(tk):
    return tk.is_stop or tk.is_punct or tk.is_space or tk.is_bracket or tk.is_quote or tk.text in en_stop or tk.pos_ in set(
        ['DET', 'ADP'])


def transform_token(token):
    new_token = token.text.lower()
    if token.like_url:
        new_token = '#url#'
    elif token.orth_.startswith('@'):
        new_token = '#screen_name#'
    elif token.like_num:
        new_token = '#num#'
    elif token.is_currency:
        new_token = '#currency#'
    elif token.text in UNICODE_EMOJI:
        new_token = '#emoji#'
    else:  # expand contractions
        if token.text == 've':
            new_token = 'have'
        elif token.text == 'm':
            new_token = 'am'
        elif token.text == 'nt':
            new_token = 'not'
    return (new_token, token.pos_)


def clean_tokens(tokens):
    tokens = [token for token in tokens if not remove_criteria(token)]  # remove stopwords, punctuation, spaces
    tokens = [transform_token(token) for token in tokens]  # replace urls, emojis and numbers
    tokens = [(get_lemma(token[0]), token[1]) for token in tokens]  # lemmatize + lowercase
    return tokens


def clean_text(text, absa=False):
    if not absa:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ',
                      text.replace(u'\xa0', u' ').replace('\r', '').replace('\n', ' ').replace('  ', ' ')).strip()
    doc = nlp(text)
    if absa:
        clean_sentences_tokens = []
        for sent in doc.sents:
            new_tokens = clean_tokens(sent)
            if len(new_tokens) > 0:
                clean_sentences_tokens.append(new_tokens)
        return clean_sentences_tokens
    else:
        new_tokens = clean_tokens(doc)
        new_tokens = [x[0] for x in new_tokens]
        return new_tokens


def clean_df_absa(df):
    text_column = ''
    if 'body' in df.columns:
        text_column = 'body'
    elif 'text' in df.columns:
        text_column = 'text'
    df['clean_doc_absa'] = df[text_column].apply(lambda x: clean_text(x, True))
    return df


def clean_df(df):
    text_column = ''
    if 'body' in df.columns:
        text_column = 'body'
    elif 'text' in df.columns:
        text_column = 'text'
    df['clean_doc'] = df[text_column].apply(lambda x: clean_text(x, False))
    return df


# python TextCleaner.py -d ../../data/processed/news/ -f unique_comments.p
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="data_folder",
                           help="The path to the data folder containing the pickle files.", default='')
    argparser.add_argument("-f", "--filename", dest="texts_filename",
                           help="The path to the pickle file with texts.", default='comments.p')
    argparser.add_argument("-m", "--model", dest="model_type",
                           help="Cleaning procedure for either JST or TSLDA or LDA or all.", default='all')
    argparser.add_argument("-s", "--size", dest="data_size",
                           help="How many texts in the dataset to use.", type=int, default=-1)
    argparser.add_argument("-p", "--parallelize", dest="parallelize",
                           help="Whether to parallelize or not.", action='store_false')
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')

    args = argparser.parse_args()
    args = vars(args)
    if args['data_folder'][-1] != '/':
        args['data_folder'] = args['data_folder'] + '/'

    start_time = time.time()
    output_filename = args['data_folder'] + args['texts_filename'].replace('.p', '') + '_text-cleaner_' + str(
        args['model_type']) + '.p'
    my_file = Path(output_filename)
    if my_file.exists():
        texts = pickle.load(open(output_filename, 'rb'))
        print('Loaded from existing file.')
    else:
        texts = pickle.load(open(args['data_folder'] + args['texts_filename'], 'rb'))
        if args['data_size'] != -1:
            texts = texts.iloc[:args['data_size']]
        if args['verbose']:
            print('Loaded the texts file. Number of texts:', len(texts))

        old_len = len(texts)
        text_column = ''
        if 'body' in texts.columns:
            text_column = 'body'
        elif 'text' in texts.columns:
            text_column = 'text'
        if text_column == '':
            raise ValueError('Could not find the text column.')
        texts = texts.dropna(subset=[text_column])
        print('Dropped NAN: {} texts -> {} texts'.format(old_len, len(texts)))
        if args['model_type'] in ['LDA', 'all']:
            if args['parallelize']:
                texts = parallelize_dataframe(texts, clean_df, n_cores=4)
            else:
                texts['clean_doc'] = texts[text_column].apply(clean_text)
            print("\t#remaining texts (lda):",
                  len([x for x in texts['clean_doc'].values if type(x) == list and len(x) > 0]))
        if args['model_type'] != 'LDA':
            if args['parallelize']:
                texts = parallelize_dataframe(texts, clean_df_absa, n_cores=4)
            else:
                texts['clean_doc_absa'] = texts[text_column].apply(lambda x: clean_text(x, True))
            print("\t#remaining texts (absa):",
                  len([x for x in texts['clean_doc_absa'].values if type(x) == list and len(x) > 0]))
        elapsed_time = time.time() - start_time
        if args['verbose']:
            print('Finished SpaCy preprocessing and cleaning. Time:', elapsed_time)

        texts.head()
        if len(set(texts.index)) != len(texts.index):
            texts.reset_index(inplace=True)

        pickle.dump(texts, open(output_filename, 'wb'))
    if args['model_type'] != 'LDA':
        if args['model_type'] == 'all':
            output_filename = args['data_folder'] + args['texts_filename'].replace('.p', '') + '_text-cleaner_' + str(
                args['model_type']) + '_omitted_texts.p'
            omitted_texts = [idx for idx, row in texts.iterrows() if len(row['clean_doc']) == 0]
            pickle.dump(omitted_texts, open(output_filename, 'wb'))
        output_filename = args['data_folder'] + args['texts_filename'].replace('.p', '') + '_text-cleaner_' + str(
        args['model_type']) + '_omitted_texts_absa.p'
        omitted_texts = [idx for idx, row in texts.iterrows() if len(row['clean_doc_absa']) == 0]
    else:
        output_filename = args['data_folder'] + args['texts_filename'].replace('.p', '') + '_text-cleaner_' + str(
        args['model_type']) + '_omitted_texts.p'
        omitted_texts = [idx for idx, row in texts.iterrows() if len(row['clean_doc']) == 0]
    if args['verbose']:
        print('{} texts omitted because no words are left after cleaning'.format(len(omitted_texts)))
    pickle.dump(omitted_texts, open(output_filename, 'wb'))
