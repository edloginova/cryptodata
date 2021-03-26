import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def read_file(textfile):
    file = open(textfile, "r")
    lines = file.readlines()
    lines = [l.strip() for l in lines]
    file.close()
    return lines


# python tslda_result2dataframe.py -r E:/results/TSLDA_bitcointalk_ntopic3_nsenti3/

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--results", dest="results_folder",
                        help="Results folder", default='')
    parser.add_argument("-f", "--filename", dest="tassign_filename",
                        help="Assignment filename", default='final.tassign')
    parser.add_argument("-ntopics", dest="ntopics",
                        help="The number of topics", type=int, default=0)
    parser.add_argument("-nsentiLabs", dest="nsentiLabs",
                        help="The number of sentiments", type=int, default=0)
    parser.add_argument("-o", "--output", dest="output_file",
                        help="The main output filename", default="")

    args = parser.parse_args()
    args = vars(args)
    nsentiLabs = args["nsentiLabs"]
    ntopics = args["ntopics"]

    if ntopics == 0:
        for part in args["results_folder"].replace('/', '').split('_'):
            if 'ntopic' in part:
                ntopics = int(part[-1])
                args["ntopics"] = int(part[-1])
    if nsentiLabs == 0:
        for part in args["results_folder"].replace('/', '').split('_'):
            if 'nsenti' in part:
                nsentiLabs = int(part[-1])
                args["nsentiLabs"] = int(part[-1])
    print('# topics:', ntopics, '#nsentiLabs:', nsentiLabs)

    tassign_file = os.path.join(args["results_folder"], args["tassign_filename"])
    if 'bitcointalk' in args['results_folder']:
        dataset = 'bitcointalk/date_structure/'
    elif 'reddit' in args['results_folder']:
        dataset = 'reddit/long/'
    elif 'news' in args['results_folder']:
        dataset = 'news/'
    if args["output_file"] == '':
        args["output_file"] = '../../data/processed/' + dataset + 'TSLDA_t' + str(args['ntopics']) + '_s' + str(
            args['nsentiLabs'])
    output_file_sent = args["output_file"] + "_sentiments.p"
    output_file_topic = args["output_file"] + "_topics.p"

    lines = read_file(tassign_file)
    ndocs = int(lines[-2].split(":")[0][1:]) + 1
    sentiment_assignments = np.zeros((ndocs, nsentiLabs), dtype=float)
    topic_assignments = np.zeros((ndocs, ntopics), dtype=float)

    for i in range(ndocs):
        assign = lines[2 * i].split(":")  # lines[2*i]

        doc_idx = int(assign[0][1:])
        topic = int(assign[2])
        sentiment = int(assign[3])

        sentiment_assignments[doc_idx, sentiment] += 1
        topic_assignments[doc_idx, topic] += 1

    sentiment_assignments = pd.DataFrame(data=sentiment_assignments,
                                         columns=["sentiment" + str(i) for i in range(nsentiLabs)])
    topic_assignments = pd.DataFrame(data=topic_assignments,
                                     columns=["topic" + str(i) for i in range(ntopics)])

    sentiment_normalized = sentiment_assignments.div(sentiment_assignments.sum(axis=1), axis=0)
    topic_normalized = topic_assignments.div(topic_assignments.sum(axis=1), axis=0)

    sentiment_info = pd.DataFrame({"id": range(ndocs),
                                   "sentiment": sentiment_normalized.apply(
                                       lambda x: {i: x[i] for i in range(nsentiLabs)}, axis=1)})
    topic_info = pd.DataFrame({"id": range(ndocs),
                               "topic": topic_normalized.apply(lambda x: {i: x[i] for i in range(ntopics)}, axis=1)})

    sentiment_normalized = pd.concat([sentiment_info, sentiment_normalized], axis=1)
    topic_normalized = pd.concat([topic_info, topic_normalized], axis=1)

    file = open(output_file_sent, mode="wb")
    pickle.dump(obj=sentiment_normalized, file=file)
    file.close()

    file = open(output_file_topic, mode="wb")
    pickle.dump(obj=topic_normalized, file=file)
    file.close()
