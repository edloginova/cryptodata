import json
import os
import pickle
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime, date
from random import randint
from time import sleep

import credentials
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from slack import WebClient

SCROLL_PAUSE_TIME = 5
SCRAPER_DIR = '../../data/raw/'
today = date.today()
SCRAPE_DATE = today.strftime("%b-%d-%Y")
MIN_SLEEP = 2
MAX_SLEEP = 5
MAX_BITCOINTALK_ITER = 49960
SLACK_USER = ''


def slack_message(message):
    client = WebClient(credentials.SLACK_API_KEY)
    response = client.chat_postMessage(channel='@' + SLACK_USER, text=message)
    return response


def convert_datetime(x, scrape_date=-1):
    if scrape_date == -1:
        today = date.today()
        scrape_date = today
    scrape_date = scrape_date.strftime("%B %d, %Y, ")
    if ': ' in x:
        x = x.split(': ')[1].split(' by')[0]
    if 'Today at ' in x:
        x = x.replace('Today at ', scrape_date)
    return datetime.strptime(x, '%B %d, %Y, %I:%M:%S %p')


def scrape_cryptocompare_forum(verbose=True, max_count=200):
    # source: https://stackoverflow.com/questions/20986631/how-can-i-scroll-a-web-page-using-selenium-webdriver-in-python
    link = 'https://www.cryptocompare.com/coins/btc/forum/USD'
    driver = webdriver.Firefox()
    driver.get(link)
    el = driver.find_element_by_link_text("Login")
    el.click()
    email = driver.find_element_by_name("email")
    password = driver.find_element_by_name("password")
    sleep(randint(MIN_SLEEP, MAX_SLEEP))
    email.send_keys(credentials.USERNAME)
    sleep(randint(MIN_SLEEP, MAX_SLEEP))
    password.send_keys(credentials.PASSWORD)
    driver.find_element_by_class_name("btn-login").click()
    if verbose:
        print("Successfully logged in CryptoCompare.")

    last_height = driver.execute_script("return document.body.scrollHeight")
    all_forum_elems = set()
    iterations = 0
    data = {}
    while iterations < max_count:
        iterations = iterations + 1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(SCROLL_PAUSE_TIME)
        forum_elems = driver.find_elements_by_class_name("item-post")
        forum_elems = list(set(forum_elems) - all_forum_elems)
        all_forum_elems.update(forum_elems)
        for elem in forum_elems:
            post_id = elem.get_attribute('id')
            username = elem.find_element_by_class_name("item-username").text
            ago = elem.find_element_by_class_name("item-ago").get_attribute('title')
            body = elem.find_element_by_class_name("content-body").text
            upvotes = elem.find_elements_by_class_name('counter-agree')
            if len(upvotes) > 0:
                upvotes = upvotes[0].text
            else:
                upvotes = 0
            downvotes = elem.find_elements_by_class_name('counter-disagree')
            if len(downvotes) > 0:
                downvotes = downvotes[0].text
            else:
                downvotes = 0
            replies = []
            reply_button = elem.find_elements_by_class_name('btn-replies')[0]
            if 'replies' in reply_button.text:
                driver.execute_script("arguments[0].scrollIntoView();", reply_button)
                reply_button.send_keys('\n')
                replies_elements = elem.find_elements_by_class_name("item-replies")[0].find_elements_by_class_name(
                    "post-reply")
                replies = [None] * len(replies_elements)
                for i in range(0, len(replies_elements)):
                    reply = replies_elements[i]
                    reply_username = reply.find_element_by_class_name('username').text
                    reply_date = reply.find_element_by_class_name('reply-date').get_attribute('title')
                    reply_body = reply.find_element_by_css_selector('p').text
                    reply_upvotes = reply.find_elements_by_class_name('counter-agree')
                    if len(reply_upvotes) > 0:
                        reply_upvotes = reply_upvotes[0].text
                    else:
                        reply_upvotes = 0
                    reply_downvotes = reply.find_elements_by_class_name('counter-disagree')
                    if len(reply_downvotes) > 0:
                        reply_downvotes = reply_downvotes[0].text
                    else:
                        reply_downvotes = 0
                    replies[i] = {'username': reply_username, 'date': reply_date, 'body': reply_body,
                                  'upvotes': reply_upvotes, 'downvotes': reply_downvotes}
            data[post_id] = {'username': username, 'date': ago, 'text': body,
                             'upvotes': upvotes, 'downvotes': downvotes, 'replies': replies}

        if verbose:
            sys.stdout.write(
                '{} iterations, {} posts: {}, {}, {}, {}, {}, {} \r'.format(iterations, len(data), username, ago,
                                                                            len(body), upvotes, downvotes,
                                                                            len(replies)))
            sys.stdout.flush()

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        sleep(randint(2, 5))
    data = pd.DataFrame.from_dict(data, orient='index')
    pickle.dump(data, open(SCRAPER_DIR + "cryptocompare_forum/cryptocompare_btc_forum_" + SCRAPE_DATE + ".p", "wb"))
    return data


def scrape_bitcointalk(verbose=True, last_i=0, postfix='', max_subforum_page=49960, altcoins=False, max_urls=-1):

    if altcoins:
        PATH = 'bitcointalk/altcoins/'
    else:
        PATH = 'bitcointalk/'

    with open(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p', 'rb') as f:
        filtered_thread_links = pickle.load(f)
    with open(SCRAPER_DIR + PATH + 'raw_html.p', 'rb') as f:
        raw_html_texts = pickle.load(f)

    # extract comments from threads
    comment_texts = {}
    cntr_texts = 0
    errors = []
    if last_i > 0:
        postfix = postfix + '_itgt' + str(last_i)
    if max_urls == -1:
        max_urls = len(filtered_thread_links)
    else:
        max_urls = last_i + max_urls
    if verbose:
        print('Scraping urls from {} to {}'.format(last_i, max_urls))
    for i in range(last_i, max_urls):
        if verbose:
            sys.stdout.write('{} / {}, errors: {}, comments: {}, texts: {}, \r'.format(i, len(filtered_thread_links),
                                                                                       len(errors), len(comment_texts),
                                                                                       cntr_texts))
            # sys.stdout.flush()
        url = filtered_thread_links[i]
        comments = []
        thread_date = -1
        if i % 10 == 0:
            with open(SCRAPER_DIR + PATH + 'bitcointalk_dictionary_tmp' + SCRAPE_DATE + '.p', 'wb') as f:
                pickle.dump(comment_texts, f)
        try:
            raw_html = raw_html_texts[url][0]
            html = BeautifulSoup(raw_html, 'html.parser')
            posts = html.findAll('td', attrs={'class': 'td_headerandpost'})
            for x in posts:
                date = x.findAll('div', attrs={'class': 'smalltext'})[0]
                if thread_date == -1:
                    thread_date = date.text
                comment = x.findAll('div', attrs={'class': 'post'})[0]
                comments.append((date.text, comment.text))
            cntr_texts += len(comments)
            page_numbers = html.findAll('a', attrs={'class': 'navPages'})
            if len(page_numbers) > 0:
                max_page_num = max([int(y.attrs['href'].split('.')[-1]) for y in page_numbers])
                for j, page_num in enumerate(range(0, max_page_num + 20, 20)):
                    page_url = filtered_thread_links[i].split('.0')[0] + '.' + str(page_num)
                    raw_html = raw_html_texts[url][1:][j]
                    html = BeautifulSoup(raw_html, 'html.parser')
                    posts = html.findAll('td', attrs={'class': 'td_headerandpost'})
                    for x in posts:
                        date = x.findAll('div', attrs={'class': 'smalltext'})[0]
                        comment = x.findAll('div', attrs={'class': 'post'})[0]
                        comments.append((date.text, comment.text))
                    cntr_texts += len(comments)
            comment_texts[(url, thread_date)] = comments
        except Exception as e:
            print(e)
            errors.append(i)
            if verbose:
                message = "ERROR during scraping bitcointalk forum. Scrape date: {}, error: {} on URL {} (# {}).".format(
                    SCRAPE_DATE, e, url, i)
                slack_message(message)
            with open(SCRAPER_DIR + PATH + 'last_successful_link' + SCRAPE_DATE + '.p', 'wb') as f:
                pickle.dump(last_i, f)
            break
        last_i = i
    with open(SCRAPER_DIR + PATH + 'last_successful_link' + SCRAPE_DATE + '.p', 'wb') as f:
        pickle.dump(last_i, f)
    with open(SCRAPER_DIR + PATH + 'bitcointalk_dictionary_' + SCRAPE_DATE + postfix + '.p', 'wb') as f:
        pickle.dump(comment_texts, f)

    comments_dataframe = {}
    for key, val in comment_texts.items():
        for x in val:
            if 'AM' in x[0] or 'PM' in x[0]:
                comments_dataframe[len(comments_dataframe)] = {'date': x[0], 'body': x[1]}
    comments_dataframe = pd.DataFrame.from_dict(comments_dataframe, orient='index')
    if len(comments_dataframe) > 0:
        with open(SCRAPER_DIR + PATH + 'bitcointalk_dataframe_' + SCRAPE_DATE + postfix + '.p', 'wb') as f:
            pickle.dump(comments_dataframe, f)
        if altcoins:
            PATH = '../../data/processed/bitcointalk/altcoins/'
        else:
            PATH = '../../data/processed/bitcointalk/'
        with open(PATH + 'comments_no_thread_structure.p', 'wb') as f:
            pickle.dump(comments_dataframe, f)
        clean_comments = {}
        for key, separate_comments in comment_texts.items():
            clean_comment = []
            for comment in separate_comments:
                if not comment[0].isnumeric():
                    clean_comment.append(comment)
            clean_comments[key] = clean_comment
        post_counter = 0
        posts_df = {}
        comment_cntr = 0
        for key, val in clean_comments.items():
            if len(val) > 0:
                comment_ids = []
                for _ in val[1:]:
                    comment_ids = comment_ids + [comment_cntr]
                    comment_cntr += 1
                posts_df[post_counter] = {'id': post_counter, 'date': key[1], 'post_url': key[0],
                                          'comment_ids': comment_ids, 'title': val[0][1], 'num_comments': len(comment_ids)}
                post_counter += 1
        posts_df = pd.DataFrame.from_dict(posts_df, orient='index')
        posts_df['date'] = posts_df['date'].apply(convert_datetime)
        post_counter = 0
        comments_df = {}
        comment_cntr = 0
        for key, val in clean_comments.items():
            for comment in val[1:]:
                comment_cntr += 1
                comments_df[comment_cntr] = {'parent_id': post_counter, 'date': comment[0], 'body': comment[1],
                                             'id': comment_cntr}
            post_counter += 1
        comments_df = pd.DataFrame.from_dict(comments_df, orient='index')
        comments_df['date'] = comments_df['date'].apply(convert_datetime)
        pickle.dump(comments_df, open(PATH + 'comments.p', 'wb'))
        pickle.dump(posts_df,
                    open(PATH + 'comment_id_data.p', 'wb'))
        if verbose:
            message = "Finished scraping bitcointalk forum. Scrape date: {}, number of texts: {}.".format(SCRAPE_DATE,
                                                                                                          len(comments_dataframe))
            slack_message(message)


def scrape_cryptocompare_news(save_sources=False, max_count=5000, coin='BTC'):
    key = credentials.SCRAPER_KEY

    if save_sources:
        link = 'https://min-api.cryptocompare.com/data/news/feeds' + '?api_key=' + key
        raw_html = requests.get(link)
        btc_news_sources = json.loads(raw_html.text)
        btc_news_sources = pd.DataFrame.from_records(btc_news_sources)
        with open(SCRAPER_DIR + 'news/{}_news_sources.p'.format(coin), 'wb') as f:
            pickle.dump(btc_news_sources, f)

    datasets = []
    link = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={}'.format(coin) + '&api_key=' + key
    raw_html = requests.get(link)
    more_data = json.loads(raw_html.text)
    more_data = pd.DataFrame.from_records(more_data["Data"])
    datasets.append(more_data)
    recent_timestamp = str(min(more_data['published_on']))
    count = 0
    output_filename = SCRAPER_DIR + 'news/'
    os.makedirs(output_filename, exist_ok=True)
    try:
        if max_count == -1:
            keep_scraping = True
            while keep_scraping:
                count = count + 1
                link = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={}'.format(coin) + '&lTs=' + recent_timestamp + '&api_key=' + key
                raw_html = requests.get(link)
                more_data = json.loads(raw_html.text)
                more_data = pd.DataFrame.from_records(more_data["Data"])
                datasets.append(more_data)
                recent_timestamp = min(more_data['published_on'])
                print(count, more_data.head(), recent_timestamp)
                if min(more_data['published_on']) <= 1438898400:
                    keep_scraping = False
                recent_timestamp = str(recent_timestamp)
                sleep(10)
            full_data = pd.concat(datasets)
            with open(output_filename + 'cryptocompare_news_texts_{}_'.format(coin) + SCRAPE_DATE + '.p', 'wb') as f:
                pickle.dump(full_data, f)
        else:
            while count < max_count:
                count = count + 1
                link = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={}'.format(coin) + '&lTs=' + recent_timestamp + '&api_key=' + key
                raw_html = requests.get(link)
                more_data = json.loads(raw_html.text)
                more_data = pd.DataFrame.from_records(more_data["Data"])
                datasets.append(more_data)
                recent_timestamp = str(min(more_data['published_on']))
                sleep(10)
            full_data = pd.concat(datasets)
            with open(output_filename + 'cryptocompare_news_texts_{}_'.format(coin) + SCRAPE_DATE + '.p', 'wb') as f:
                pickle.dump(full_data, f)

        with open(output_filename + 'cryptocompare_news_texts_{}_'.format(coin) + SCRAPE_DATE + '.p', 'wb') as f:
            pickle.dump(full_data, f)
        if verbose:
            slack_message(
                "Finished scraping cryptocompare news. Scrape date: {}, iterations: {}.".format(SCRAPE_DATE, count))
    except Exception as e:
        print(e)
        print(more_data)

def get_last_successful_link(cur_dir, dir_path):
    dir_path = cur_dir + dir_path
    os.chdir(dir_path)
    files = filter(os.path.isfile, os.listdir(dir_path))
    files = [os.path.join(dir_path, f) for f in files]
    files = [x for x in files if 'last_successful_link' in x]
    files.sort(key=lambda x: os.path.getmtime(x))
    if len(files) > 0:
        last_successful_link = pickle.load(open(files[0], 'rb'))
    else:
        print('No last_successful_link found, setting to 0')
        last_successful_link = 0
    os.chdir(cur_dir)
    return last_successful_link


# python WebScraper.py -s bitcointalk -i 5
# python WebScraper.py -s cryptocompare-forum -i 5
# python WebScraper.py -s cryptocompare-news -i 10
if __name__ == "__main__":
    
    warnings.filterwarnings(action="ignore", message="unclosed",
                            category=ResourceWarning)
    warnings.filterwarnings(action='once')

    argparser = ArgumentParser()
    argparser.add_argument("-s", "--source", dest="source")
    argparser.add_argument("-p", "--postfix", dest="postfix", default='')
    argparser.add_argument("-i", "--iter", dest="iter", type=int)
    argparser.add_argument("-l", "--last_i", dest="last_i", type=int, default=0)
    argparser.add_argument("-m", "--max_urls", dest="max_urls", type=int, default=-1)
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')
    argparser.add_argument("-ls", "--last_success", dest="last_success", action='store_true')
    argparser.add_argument("-su", "--slack_user", dest="slack_user", default='ekaterina.loginova')
    argparser.add_argument("-c", "--coin", dest="coin", default='btc')

    args = argparser.parse_args()
    args = vars(args)
    SLACK_USER = args['slack_user']
    verbose = args['verbose']
    if not verbose:
        print('Running in silent mode.')
    if 'bitcointalk' in args['source']:
        if args['last_success']:
            dir_path = '/../../data/raw/bitcointalk/'
            if 'altcoin' in args['source']:
                dir_path += 'altcoins/'
            args['last_i'] = get_last_successful_link(os.getcwd(), dir_path)
        if args['iter'] is None:
            iternum = MAX_BITCOINTALK_ITER
        else:
            iternum = args['iter']
        scrape_bitcointalk(verbose=verbose, max_subforum_page=iternum, last_i=args['last_i'], postfix=args['postfix'],
                           altcoins='altcoin' in args['source'],max_urls=args['max_urls'])
    elif args['source'] == 'cryptocompare-forum':
        scrape_cryptocompare_forum(verbose=verbose, max_count=args['iter'])
    elif args['source'] == 'cryptocompare-news':
        scrape_cryptocompare_news(max_count=args['iter'], coin=args['coin'])
    else:
        print('Unknown source.')
