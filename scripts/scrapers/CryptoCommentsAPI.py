# Import identifiers for connecting with Reddit
import os
import re
import sys
import unicodedata
from datetime import datetime

import pandas as pd
import praw

module_path = os.path.abspath(os.path.join('../identifiers'))
if module_path not in sys.path:
    sys.path.append(module_path)
from Identifiers import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD

__author__ = "Guus van Heijningen"


class CryptoCommentsFetcher():
    def __init__(self, user=REDDIT_USERNAME, pw=REDDIT_PASSWORD, client_id=REDDIT_CLIENT_ID,
                 client_secret=REDDIT_CLIENT_SECRET):
        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, username=user, password=pw,
                                  user_agent='sentiment analysis for dissertation by van Heijningen, UGent')

    def comment_fetcher(self, subreddit, start_unix, end_unix, save=False):
        # Create the dict for your df
        CryptoComments = {'Title': [],
                          'PostScore': [],
                          'NumComments': [],
                          'PostTime': [],
                          'Flair': [],
                          'ParentID': [],
                          'CommentID': [],
                          'CommentTime': [],
                          'CommentScore': [],
                          'CommentText': []
                          }

        subreddit = self.reddit.subreddit(subreddit)
        submissions = subreddit.top('year')  # submissions(start_unix, end_unix)#.top('day')

        for submission in submissions:
            #    if not submission.stickied:
            submission.comments.replace_more(limit=None)  # 32 is max possible

            for comment in submission.comments.list():
                CryptoComments['Title'].append(submission.title)
                CryptoComments['PostScore'].append(submission.score)
                CryptoComments['NumComments'].append(submission.num_comments)
                posttime = datetime.utcfromtimestamp(submission.created_utc)
                CryptoComments['PostTime'].append(posttime)
                CryptoComments['Flair'].append(submission.link_flair_text)
                CryptoComments['ParentID'].append(comment.parent())
                CryptoComments['CommentID'].append(comment.id)
                commenttime = datetime.utcfromtimestamp(comment.created_utc)
                CryptoComments['CommentTime'].append(commenttime)
                CryptoComments['CommentScore'].append(comment.score)
                text = re.sub("\s\s+", ' ', unicodedata.normalize('NFKD', comment.body)).replace('\n', ' ')
                CryptoComments['CommentText'].append(text)

        comments_df = pd.DataFrame(CryptoComments)

        if save:
            save_path = os.path.abspath(os.path.join('../data'))
            file = '{}/Reddit_{}_{}_{}.csv'.format(save_path, subreddit,
                                                   datetime.fromtimestamp(start_unix).strftime('%m%d%y'),
                                                   datetime.fromtimestamp(end_unix).strftime('%m%d%y'))
            comments_df.to_csv(file, index=False, encoding='utf-8')

        # Put all retrieved data into pandas dataframe
        return comments_df
