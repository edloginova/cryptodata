#!/usr/bin/env python
"""
Extracts LDA topics for a given dataframe with texts.
"""

import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import gensim
import nltk
import numpy as np
import spacy
from gensim import corpora

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
en_stop = set(nltk.corpus.stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
NLP_FOLDER = '../../models/topic/'
DATA_FOLDER = '../../data/processed/'

__author__ = "Ekaterina Loginova"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"


def predict_topic(text, lda_model, dictionary):
    if type(text) == list and len(text) > 0:
        doc = dictionary.doc2bow(text)
        return lda_model.get_document_topics(doc)
    else:
        return np.nan


def train_lda(text_data, verbose, save_intermediate, num_topics, model_name):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    if save_intermediate:
        pickle.dump(corpus, open(NLP_FOLDER + model_name + '_corpus.pkl', 'wb'))
        dictionary.save(NLP_FOLDER + model_name + '_dictionary.gensim')
    if verbose:
        print('Started training.')
    start_time = time.time()
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    elapsed_time = time.time() - start_time
    if verbose:
        print('Finished training. Time:', elapsed_time)
    if save_intermediate:
        lda_model.save(NLP_FOLDER + model_name + str(num_topics) + '.gensim')
    if verbose:
        print("Saved dictionary, corpus and LDA model.")
    topics = lda_model.print_topics(num_words=4)
    if verbose:
        for topic in topics:
            print(topic)
    return lda_model, dictionary


def predict_topics(comments, ldamodel, dictionary, separate_topic_columns, save_final, output_filename, verbose):
    comments['topic'] = comments['clean_text'].apply(lambda x: predict_topic(x, ldamodel, dictionary))
    if verbose:
        print("Predicted topics.")
    if verbose:
        print(comments.head())
    if separate_topic_columns:
        comments.dropna(subset=['topic'], inplace=True, axis=0)
        comments['topic'] = comments['topic'].apply(lambda x: {y[0]: y[1] for y in x})
        for i in range(0, ldamodel.num_topics):
            comments['topic' + str(i)] = [x[i] if i in x.keys() else 0 for x in comments['topic'].values]
        if verbose:
            print("Separated topics into column features.")
            print(comments.head())
    if save_final:
        pickle.dump(comments, open(output_filename, 'wb'))
    if verbose:
        print("Saved texts with extracted topics.")
    return comments


def load_data(input_filename, verbose):
    comments = pickle.load(open(input_filename, 'rb'))
    my_file = Path(input_filename.replace('.p', '') + '_text-cleaner_all.p')
    if my_file.exists():
        text_data = pickle.load(open(input_filename.replace('.p', '') + '_text-cleaner_all.p', 'rb'))
        comments['clean_text'] = text_data['clean_doc']
        if verbose:
            print("Loaded the cleaned texts.")
    else:
        my_file = Path(input_filename)
        if my_file.exists():
            text_data = pickle.load(open(input_filename, 'rb'))
            comments['text'] = text_data['doc']
            if verbose:
                print("Loaded the raw texts.")
    if verbose:
        print(comments.head())
    return comments


# python TopicExtractor.py -d ../../data/processed/news/ -f unique_comments.p -m news_new -n 5
# python TopicExtractor.py -d ../../data/processed/reddit/long/ -f unique_comments.p -m reddit_long_new -n 5
# python TopicExtractor.py -d ../../data/processed/bitcointalk/date_structure/ -f unique_comments.p -m bitcointalk_structure_new -n 5
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-t", "--total", dest="total", type=bool, default=False,
                           help='Whether to combine text data from all data sources for training the model. RELATIVE PATHS ARE HARDCODED.')
    argparser.add_argument("-d", "--input_dir", dest="input_dir",
                           help='The path to the directory with input files.')
    argparser.add_argument("-f", "--input_filename", dest="input_filename",
                           help='The path to the pickle file with comments (with their text and dates).')
    argparser.add_argument("-m", "--model_name", dest="model_name", default='',
                           help="The name of the model (for saving files).")
    argparser.add_argument("-n", "--num", dest="topic_num", type=int,
                           help="The number of topics to extract.", default=10)
    argparser.add_argument("-si", "--save-intermediate", dest="save_intermediate", type=bool, default=True,
                           help='Whether to save intermediate files.')
    argparser.add_argument("-sf", "--save-final", dest="save_final", type=bool, default=True,
                           help='Whether to save the final output dataframe as a pickle file.')
    argparser.add_argument("-sc", "--sep_columns", dest="sep_columns", action='store_false',
                           help='Whether to store topic assignments as separate columns (e.g., for each text we will have columns topic_0, topic_1 ... with corresponding probabilities.')
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')

    args = argparser.parse_args()
    args = vars(args)
    verbose = args['verbose']
    separate_topic_columns = args['sep_columns']
    if args['input_dir'][-1] != '/':
        args['input_dir'] = args['input_dir'] + '/'
    if args['total']:
        save_intermediate = True
        save_final = True
        datasets = {'news': DATA_FOLDER + 'news/unique_comments.p',
                    'bitcointalk': DATA_FOLDER + 'bitcointalk/date_no_structure/unique_comments.p',
                    'reddit_long': DATA_FOLDER + 'reddit/long/unique_comments.p',
                    'reddit_original': DATA_FOLDER + 'reddit/original/unique_comments.p', }
        for model_name, input_filename in datasets.items():
            for num_topics in [3, 5, 10]:
                if verbose:
                    print(model_name, '---', input_filename, '---', num_topics)
                output_filename = input_filename.replace('.p', '') + '_' + str(num_topics) + 'topics.p'
                comments, text_data = load_data(input_filename, verbose)
                ldamodel, dictionary = train_lda(text_data, verbose, save_intermediate, num_topics, model_name)
                comments = predict_topics(comments, ldamodel, dictionary, separate_topic_columns, save_final,
                                          output_filename, verbose)
    else:
        model_name = args['model_name']
        if model_name == '':
            model_name = args['input_dir'].split('/')[-2]
        num_topics = args['topic_num']
        save_intermediate = args['save_intermediate']
        save_final = args['save_final']
        input_filename = args['input_filename']
        DATA_FOLDER = args['input_dir']
        NLP_FOLDER = args['input_dir'] + '../../../models/topic/'
        if args['input_dir'] not in input_filename:
            input_filename = args['input_dir'] + input_filename
        output_filename = input_filename.replace('.p', '') + '_' + str(num_topics) + 'topics.p'
        comments = load_data(input_filename, verbose)
        doc_column = list({'clean_text', 'text'}.intersection(set(comments.columns)))[0]
        ldamodel, dictionary = train_lda(comments[doc_column], verbose, save_intermediate, num_topics, model_name)
        comments = predict_topics(comments, ldamodel, dictionary, separate_topic_columns, save_final, output_filename,
                                  verbose)
