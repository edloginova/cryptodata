import pickle
import warnings
from datetime import date
from random import randint
from time import sleep
import os
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import re
import sys
import credentials
from pathlib import Path
import requests
from slack import WebClient
warnings.filterwarnings("ignore")

SCROLL_PAUSE_TIME = 5
SCRAPER_DIR = '../../data/raw/'
today = date.today()
SCRAPE_DATE = today.strftime("%b-%d-%Y")
MIN_SLEEP = 2
MAX_SLEEP = 5
MAX_BITCOINTALK_ITER = 30320
SLACK_USER = ''


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


def slack_message(message):
    client = WebClient(credentials.SLACK_API_KEY)
    response = client.chat_postMessage(channel='@' + SLACK_USER, text=message)
    return response


def extract_thread_links_bitcointalk(proxies, certificate, altcoins=False, verbose=True, max_subforum_page=-1):
    if altcoins:
        PATH = 'bitcointalk/altcoins/'
        BOARD_NUM = 67
    else:
        PATH = 'bitcointalk/'
        BOARD_NUM = '1'
    if verbose:
        print('BITCOINTALK. Max # threads:', max_subforum_page)

    # extract/load links to forum threads
    my_file = Path(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p')
    print(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p')
    if my_file.exists():
        with open(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p', 'rb') as f:
            filtered_thread_links = pickle.load(f)
            if verbose:
                print("Loaded thread links from BitcoinTalk forum from a pre-saved file.")
    else:
        filtered_thread_links = []
        if verbose:
            print("Scraping thread links from BitcoinTalk forum.")
        if max_subforum_page == -1:
            if altcoins:
                range_subforum_pages = range(4480, 59940, 20)
            else:
                range_subforum_pages = range(1600, 26620, 20)
        else:
            max_subforum_page = max(max_subforum_page, MAX_BITCOINTALK_ITER)
            range_subforum_pages = range(0, max_subforum_page, 20)
        for subforum_page in range_subforum_pages:
            if verbose:
                sys.stdout.write('{} / {} \r'.format(subforum_page, max_subforum_page))
                sys.stdout.flush()
            url = 'https://bitcointalk.org/index.php?board=' + str(BOARD_NUM) + '.' + str(subforum_page)
            raw_html = requests.get(url, proxies=proxies, verify=certificate)
            html = BeautifulSoup(raw_html.text, 'html.parser')
            thread_links = html.findAll('a', attrs={'href': re.compile(".?topic=\d+\.0$")})
            for link in thread_links:
                if len(link.text) > 1:
                    filtered_thread_links.append(link.attrs['href'])
            sleep(randint(2, 10))
            if subforum_page % 60 == 0:
                with open(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p',
                          'wb') as f:
                    pickle.dump(filtered_thread_links, f)
        filtered_thread_links = list(set(filtered_thread_links))
        filtered_thread_links = sorted(filtered_thread_links, key=lambda x: int(x.split('.')[-2].split('=')[-1]),
                                       reverse=True)
        if verbose:
            print('Extracted thread links:', len(filtered_thread_links))
        with open(SCRAPER_DIR + PATH + 'filtered_thread_links' + str(max_subforum_page) + '.p',
                  'wb') as f:
            pickle.dump(filtered_thread_links, f)
    return filtered_thread_links


def extract_raw_html_bitcointalk(filtered_thread_links, proxies, certificate, last_i, dir_path):
    if os.path.exists(dir_path + str(last_i) + 'raw_html.p'):
        with open(dir_path + 'raw_html' + str(last_i) + '.p', 'rb') as f:
            threads = pickle.load(f)
    else:
        threads = {}
    comment_counter = 0
    try:
        for i in range(last_i + len(threads), len(filtered_thread_links)):
            sys.stdout.write(str(i))
            sys.stdout.flush()
            url = filtered_thread_links[i]
            raw_html = requests.get(url, proxies=proxies, verify=certificate)
            html = BeautifulSoup(raw_html.text, 'html.parser')
            page_numbers = html.findAll('a', attrs={'class': 'navPages'})
            comments = [raw_html.text]
            if len(page_numbers) > 0:
                max_page_num = max([int(y.attrs['href'].split('.')[-1]) for y in page_numbers])
                for page_num in range(0, max_page_num + 20, 20):
                    page_url = filtered_thread_links[i].split('.0')[0] + '.' + str(page_num)
                    raw_html = requests.get(page_url, proxies=proxies, verify=certificate)
                    comments.append(raw_html.text)
                    comment_counter += 1
                    sleep(randint(2, 10))
            threads[url] = comments
            sleep(randint(2, 10))
            if i % 10 == 0:
                with open(dir_path + 'raw_html' + str(last_i) + '.p', 'wb') as f:
                    pickle.dump(threads, f)
        with open(dir_path + 'raw_html' + str(last_i) + '.p', 'wb') as f:
            pickle.dump(threads, f)
    except Exception as e:
        slack_message('Caught error on home pc scraper, iteration {}, error {}'.format(i, e))
    slack_message(
        'Finished saving raw html threads for bitcointalk. Saved {} threads with {} comments.'.format(len(threads),
                                                                                                    comment_counter))
    return threads


if __name__ == "__main__":
    warnings.filterwarnings(action='once')
    warnings.filterwarnings(action="ignore", message="unclosed",
                            category=ResourceWarning)

    argparser = ArgumentParser()
    argparser.add_argument("-i", "--max_subforum_page", dest="max_subforum_page", type=int, default=-1)
    argparser.add_argument("-l", "--last_i", dest="last_i", type=int, default=0)
    argparser.add_argument("-m", "--max_urls", dest="max_urls", type=int, default=-1)
    argparser.add_argument("-v", "--verbose", dest="verbose", action='store_false')
    argparser.add_argument("-a", "--altcoins", dest="altcoins", action='store_true')
    argparser.add_argument("-ls", "--last_success", dest="last_success", action='store_true')
    argparser.add_argument("-su", "--slack_user", dest="slack_user", default='ekaterina.loginova')

    args = argparser.parse_args()
    args = vars(args)
    SLACK_USER = args['slack_user']
    verbose = args['verbose']
    proxy_host = "proxy.crawlera.com"
    proxy_port = "8010"
    proxy_auth = credentials.PROXY_AUTH  # Make sure to include ':' at the end
    proxies = {"https": "https://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port),
               "http": "http://{}@{}:{}/".format(proxy_auth, proxy_host, proxy_port)}
    certificate = credentials.CERTIFICATE_PATH
    if args['max_subforum_page'] is None:
        max_subforum_page = MAX_BITCOINTALK_ITER
    else:
        max_subforum_page = args['max_subforum_page']

    dir_path = '../../data/raw/bitcointalk/'
    if args['altcoins']:
        dir_path += 'altcoins/'
    if args['last_success']:
        args['last_i'] = get_last_successful_link(os.getcwd(), dir_path)
    filtered_thread_links = extract_thread_links_bitcointalk(altcoins=args['altcoins'], verbose=verbose,
                                                             max_subforum_page=max_subforum_page,
                                                             proxies=proxies,
                                                             certificate=certificate)
    texts = extract_raw_html_bitcointalk(filtered_thread_links, proxies, certificate, last_i=args['last_i'],
                                         dir_path=dir_path)
