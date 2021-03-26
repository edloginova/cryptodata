"""
Add ABDA (TSLDA or JST) topics to the existing file with text features by merging two dataframes on comment id.
"""

import pickle
from argparse import ArgumentParser

from TextFeatureExtractor import *

# python add_absa_topics.py -d ../../data/processed/bitcointalk/date_structure/ -m TSLDA -ntopics 2 -nsentiLabs 2
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="dir",
                        help="Directory with original files", default='')
    parser.add_argument("-f", "--filename", dest="file",
                        help="Original files", default='basic_text_features.p')
    parser.add_argument("-m", "--method", dest="method",
                        help="ABSA method", default='TSLDA')
    parser.add_argument("-ntopics", dest="ntopics",
                        help="The number of topics", type=int, default=2)
    parser.add_argument("-nsentiLabs", dest="nsentiLabs",
                        help="The number of sentiments", type=int, default=2)

    args = parser.parse_args()
    args = vars(args)
    data = pickle.load(open(args['dir'] + args['file'], 'rb'))

    if 'bitcointalk' in args['dir']:
        dataset = 'bitcointalk/date_structure/'
    elif 'reddit' in args['dir']:
        dataset = 'reddit/long/'
    elif 'news' in args['dir']:
        dataset = 'news'
    sentiment_features = pickle.load(open(
        args['dir'] + '/' + args['method'] + '_t' + str(args['ntopics']) + '_s' + str(
            args['nsentiLabs']) + '_sentiments.p', 'rb'))
    topic_features = pickle.load(open(
        args['dir'] + '/' + args['method'] + '_t' + str(args['ntopics']) + '_s' + str(
            args['nsentiLabs']) + '_topics.p', 'rb'))

    sentiment_features.drop('sentiment', axis=1, inplace=True)
    topic_features.drop('topic', axis=1, inplace=True)

    for x in sentiment_features.columns:
        if 'sentiment' in x:
            data['sentiment_scores_' + str(x[-1])] = sentiment_features[x]
    for x in topic_features.columns:
        if 'topic' in x:
            data['topic_scores_' + str(x[-1])] = topic_features[x]

    pickle.dump(data, open(args['dir'] + args['file'].replace('.p', '').replace('basic_', '') + '_' + '_'.join(
        [args['method'], 't' + str(args['ntopics']), 's' + str(args['nsentiLabs'])]) + '.p', 'wb'))

    verbose = True
    MyFE = TextFeatureExtractor(verbose=verbose, dump_folder=args['dir'])
    aggregated_data = MyFE.aggregate_features(data)
    pickle.dump(aggregated_data, open(
        args['dir'] + 'aggregated_' + args['file'].replace('.p', '').replace('basic_', '') + '_' + '_'.join(
            [args['method'], 't' + str(args['ntopics']), 's' + str(args['nsentiLabs'])]) + '.p', 'wb'))
