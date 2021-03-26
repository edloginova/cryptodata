import os
import pickle
import time

import pandas as pd
from pytrends.request import TrendReq

__author__ = "Guus van Heijningen"


class GoogleTrendsFetcher():
    def __init__(self):

        # req = requests.get('https://www.cryptocompare.com/api/data/coinlist/').json()
        # info = req['Data']
        # CCcoinlist = pd.DataFrame(info).transpose()
        #
        # CMCreq = requests.get('https://api.coinmarketcap.com/v2/ticker/?limit=100').json()
        # CMCreq2 = requests.get('https://api.coinmarketcap.com/v2/ticker/?start=101&limit=2').json()
        # CMCinfo = CMCreq['data']
        # CMCinfo2 = CMCreq2['data']
        # CMCinfo.update(CMCinfo2)
        # CMCcoins = pd.DataFrame(CMCinfo).transpose()[['name', 'symbol']]
        # CMCcoins.iloc[29,1] = 'IOT'

        self.coinlist = pickle.load(open('../data/coinlist50.p', 'rb'))
        # CMCcoins.merge(CCcoinlist, left_on='symbol', right_on='Name')[['CoinName', 'Name']].head(100)
        self.CryptoTrend = TrendReq()

    def crypto_trends_fetcher(self, start, end, batch_size=500, sleep=3, missing=False, save=False):
        coins = list(self.coinlist['CoinName'])
        if missing:
            coins = self.missingcoins.copy()

        missingcoins = []
        batch_start = 0
        batch_end = min(len(coins), batch_size)
        AllCoinTrends = pd.DataFrame()

        while batch_start < len(coins):
            CoinTrends = pd.DataFrame()
            for coin in coins[batch_start: batch_end]:
                try:
                    self.CryptoTrend.build_payload(kw_list=[coin], timeframe='{} {}'.format(start, end))
                    interest_coin = self.CryptoTrend.interest_over_time()
                    interest_coin.rename(columns={coin: 'interest'}, inplace=True)
                    interest_coin.insert(0, 'CoinName', coin)
                    CoinTrends = CoinTrends.append(interest_coin)
                except Exception as e:
                    print('unable to fetch {} trends data'.format(coin))
                    print(e)
                    missingcoins += [coin]
                    pass
                time.sleep(sleep)  # Request at most 1 time per second to avoid request limits

            AllCoinTrends = AllCoinTrends.append(CoinTrends)

            batch_start += batch_size
            batch_end = min(len(coins), batch_start + batch_size)
            time.sleep(60)

        self.missingcoins = missingcoins
        AllCoinTrends = AllCoinTrends.reset_index().merge(self.coinlist[['CoinName', 'Name']], on=['CoinName'])

        if save:
            save_path = os.path.abspath(os.path.join('../data'))
            file = '{}/CryptoTrends_{}_{}.csv'.format(save_path, start, end)
            AllCoinTrends.to_csv(file, encoding='utf-8')

        return AllCoinTrends

    # # Request the trends data for each list
    # trends1 = CoinInterest(coins1, start, end)
    # trends2 = CoinInterest(coins2, start, end)
    # trends3 = CoinInterest(coins3, start, end)
    # missing = list(set(coins1)-set(trends1.coin))
    # missing += list(set(coins2)-set(trends2.coin))
    # missing += list(set(coins3)-set(trends3.coin))
    # trendsMiss = CoinInterest(missing, start, end)
    # missing2 = list(set(missing)-set(trendsMiss.coin))
    # trendsMiss2 = CoinInterest(missing2, start, end)
    # missing3 = list(set(missing2)-set(trendsMiss2.coin))

    # # Merge data together and save
    # AllCoinTrends = pd.DataFrame()
    # AllCoinTrends = AllCoinTrends.append([trends1, trends2, trends3, trendsMiss, trendsMiss2])
    # AllCoinTrends = AllCoinTrends.reset_index().merge(coinList[['CoinName', 'Name']], left_on=['coin'], right_on=['CoinName']).set_index('index')
    # del AllCoinTrends['coin']

    # # Not all coins seem to be covered by the algorithm, could be due to the coins with star behind their Name??
    # AllCoinTrends.to_csv('CoinTrends.csv')
