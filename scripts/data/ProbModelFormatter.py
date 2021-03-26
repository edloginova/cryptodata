#!/usr/bin/env python
"""
Saves the dataframe with clean texts in an ABSA-model compatible format (with .dat files as output).
"""

import os
import pickle
from argparse import ArgumentParser

import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NLP_FOLDER = 'nlp/'
nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "ner"])
analyzer = SentimentIntensityAnalyzer()

__author__ = "Ekaterina Loginova"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"


def combine_datasets(input_filenames, output_filename, model='TSLDA', restart_sentences=True):
    content = []
    for filename in input_filenames:
        tfile = open(filename, 'r', encoding='utf-8')
        current_content = tfile.readlines()
        current_content = [line for line in current_content if len(line.strip()) > 0]
        content.extend(current_content)

    tmp_filename = 'tmp.dat'
    with open(tmp_filename, 'w', encoding='utf-8') as tmp_file:
        for line in content:
            tmp_file.write(line)

    restore_numeration(tmp_filename, output_filename, model=model, restart_sentences=restart_sentences)
    os.remove(tmp_filename)


def restore_numeration(input_filename, output_filename, model='TSLDA', restart_sentences=True):
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        content = input_file.readlines()
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        iter_cntr = 0
        restart_str_cntr = 0
        for line in content:
            if model == 'TSLDA':
                doc_cntr, str_cntr, *categories = line.split(' ')
                if iter_cntr == 0:
                    prev_doc_cntr = doc_cntr
                    iter_cntr = 1
                if prev_doc_cntr != doc_cntr:
                    doc_cntr = 'd' + str(int(prev_doc_cntr[1:]) + 1)
                    restart_str_cntr = 0
                else:
                    restart_str_cntr += 1
                prev_doc_cntr = doc_cntr
                if restart_sentences:
                    str_cntr = 's' + str(restart_str_cntr)
                new_line = ' '.join([doc_cntr, str_cntr] + categories)
            else:
                doc_cntr, *categories = line.split(' ')
                if iter_cntr == 0:
                    prev_doc_cntr = doc_cntr
                    iter_cntr = 1
                if prev_doc_cntr != doc_cntr:
                    doc_cntr = 'd' + str(int(prev_doc_cntr[1:]) + 1)
                prev_doc_cntr = doc_cntr
                new_line = ' '.join([doc_cntr] + categories)
            output_file.write(new_line)


def read_crypto(filename='', prefix='', model='TSLDA', source='tokens'):
    if filename == '':
        filename = 'data/' + model.upper() + '/' + prefix + source + '.dat'
    with open(filename, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()
    content_lines = [line for line in content_lines if len(line.strip()) > 0]
    data = {}
    for idx, line in enumerate(content_lines):
        if model == 'TSLDA':
            doc_idx, sent_idx, *tokens = line.split()
            doc_idx = int(doc_idx[1:])
            sent_idx = int(sent_idx[1:])
            text = ' '.join(tokens)
            data[idx] = {'doc_idx': doc_idx, 'sent_idx': sent_idx, source: text}
        else:
            doc_idx, *tokens = line.split()
            doc_idx = int(doc_idx[1:])
            text = ' '.join(tokens)
            data[idx] = {'doc_idx': doc_idx, source: text}
    data = pd.DataFrame.from_dict(data, orient='index')
    return data


def save_crypto(input_dir, data, prefix='', model='TSLDA', source='tokens'):
    filename = input_dir + model.upper() + '_' + prefix + source + '.dat'
    with open(filename, 'w', encoding='utf-8') as tfile:
        for idx, row in data.iterrows():
            content_index = 'd' + str(row['doc_idx'])
            if model == 'TSLDA':
                content_index = content_index + ' s' + str(row['sent_idx'])
            tfile.write(' '.join([content_index, row[source], '\n']))
    return 'Saved ' + str(len(data)) + ' texts succesfully to ' + filename


def is_category_1(pos, lenient=True):
    if lenient:
        return pos in set(['PROPN', 'NOUN', 'VERB'])
    else:
        return pos in set(['PROPN', 'NOUN'])


def is_category_2(sentiment_score, pos, lenient=True):
    if lenient:
        return sentiment_score['compound'] != 0 or pos in set(['ADJ', 'ADV'])
    else:
        return sentiment_score['compound'] != 0


def extract_categories(tokens, verbose=False, lenient=True):
    category_labels = [-1] * len(tokens)
    for i, word in enumerate(tokens):
        if verbose:
            print(i, word)
        if is_category_1(word[1], lenient):
            category_labels[i] = 1
        else:
            sentiment_score = analyzer.polarity_scores(word[0])
            if is_category_2(sentiment_score, word[1], lenient):
                category_labels[i] = 2
            else:
                category_labels[i] = 0
    return category_labels


# python ProbModelFormatter.py -d ../../data/processed/news/ -f unique_comments.p -m TSLDA
# python ProbModelFormatter.py -d ../../data/processed/reddit/long/ -f unique_comments.p -m TSLDA -s 10
# python ProbModelFormatter.py -d ../../data/processed/bitcointalk/date_no_structure -f unique_comments.p -m TSLDA
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--dir", dest="input_dir",
                           help="The path to the data folder containing the pickle files.", default='')
    argparser.add_argument("-f", "--filename", dest="texts_filename",
                           help="The path to the pickle file with texts.", default='comments.p')
    argparser.add_argument("-m", "--model", dest="model_type",
                           help="Formatting for either JST or TSLDA or all", default='all')
    argparser.add_argument("-s", "--size", dest="data_size",
                           help="How many texts in the dataset to use", type=int, default=-1)
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')

    args = argparser.parse_args()
    args = vars(args)
    verbose = args['verbose']
    if args['input_dir'][-1] != '/':
        args['input_dir'] = args['input_dir'] + '/'
    input_filename = args['input_dir'] + args['texts_filename'].replace('.p', '') + '_text-cleaner_' + 'all' + '.p'
    try:
        texts = pickle.load(open(input_filename, 'rb'))
    except Exception as e:
        if verbose:
            print(e)
            print('Failed to load clean texts. Attempting to load raw texts...')
        try:
            input_filename = args['input_dir'] + args['texts_filename']
            texts = pickle.load(open(input_filename, 'rb'))
            if verbose:
                print('Loaded raw texts.')
        except Exception as e:
            if verbose:
                print(e)
                raise ValueError('Failed to load raw texts.')
    if args['data_size'] != -1:
        texts = texts.iloc[:args['data_size']]
    if verbose:
        print('Loaded the texts file. Number of texts:', len(texts))

    data = {}
    cntr = 0
    for i, doc in enumerate(texts['clean_doc_absa'].values):
        for j, sent in enumerate(doc):
            data[cntr] = {'doc': sent, 'doc_idx': str(i), 'sent_idx': str(j)}
            cntr += 1
    data = pd.DataFrame.from_dict(data, orient='index')
    if verbose:
        print('Converted to dataframe.')

    data['categories'] = data['doc'].apply(extract_categories)
    data['categories'] = data['categories'].apply(lambda x: ' '.join([str(y) for y in x]))
    if verbose:
        print('Extracted and formatted categories.')
    data['tokens'] = data['doc'].apply(lambda x: ' '.join([y[0] for y in x]))
    if verbose:
        print('Formatted tokens.')
    if args['model_type'] == 'all':
        args['model_type'] = ['JST', 'TSLDA']
    else:
        args['model_type'] = [args['model_type']]
    for model_type in args['model_type']:
        save_crypto(args['input_dir'], data, prefix='', model=model_type, source='tokens')
        save_crypto(args['input_dir'], data, prefix='', model=model_type, source='categories')
