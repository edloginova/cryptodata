from credentials import *
from slack import WebClient

__author__ = "Ekaterina Loginova"


def slack_message(message, slack_user):
    client = WebClient(SLACK_API_KEY)
    response = client.chat_postMessage(channel='@' + slack_user, text=message)
    return response


def parse_topic_filename(topic_filename):
    if topic_filename.endswith('5topics.p'):
        return 'LDA', '5', '0'
    topic_mode = ''
    num_topics = ''
    num_sentiments = ''
    if 'LDA' in topic_filename:
        topic_mode = 'LDA'
    elif 'TSLDA' in topic_filename:
        topic_mode = 'TSLDA'
    elif 'JST' in topic_filename:
        topic_mode = 'JST'
    for part in topic_filename.split('_'):
        if 't' in part and len(part) == 2:
            num_topics = part[1]
        if 's' in part and len(part) == 2:
            num_sentiments = part[1]
    return topic_mode, num_topics, num_sentiments


def generate_topic_filename(topic_mode, num_topics, num_sentiments, current_filename=''):
    if current_filename == '':
        current_filename = topic_mode
        if num_topics != '':
            current_filename += '_t' + num_topics
        if num_sentiments != '':
            current_filename += '_s' + num_sentiments
        current_filename += 'basetable.p'
    return current_filename


def determine_data_source(basetable_filename):
    data_source = ''
    if 'reddit' in basetable_filename:
        if 'long' in basetable_filename:
            data_source = 'reddit-long'
        else:
            data_source = 'reddit'
    elif 'news' in basetable_filename:
        data_source = 'news'
    elif 'bitcointalk' in basetable_filename:
        data_source = 'bitcointalk'
    if 'JST' in basetable_filename or 'TSLDA' in basetable_filename:
        if 'JST' in basetable_filename:
            data_source += "_jst"
        if 'TSLDA' in basetable_filename:
            data_source += "_tslda"
        for part in basetable_filename.split('_'):
            if len(part) == 2:
                if 't' in part:
                    num_topics = part[1]
                    data_source += '_t' + str(num_topics)
                if 's' in part:
                    num_senti = part[1]
                    data_source += '_s' + str(num_senti)
    else:
        if data_source != '':
            data_source += "_"
        data_source += 'lda_t5'
    if 'fillnan' in basetable_filename:
        if data_source != '':
            data_source += "_"
        data_source += 'fillnan'

    if 'nosubjectdrop' in basetable_filename:
        if data_source != '':
            data_source += "_"
        data_source += 'nsd'
    return data_source
