import os
import time
from datetime import datetime

import pandas as pd
import requests

__author__ = "Guus van Heijningen"

class CryptoValuesFetcher():
    def __init__(self):
        req = requests.get('https://www.cryptocompare.com/api/data/coinlist/').json()
        info = req['Data']
        CCcoinlist = pd.DataFrame(info).transpose()

        CMCreq = requests.get('https://api.coinmarketcap.com/v2/ticker/?limit=51').json()
        # CMCreq2 = requests.get('https://api.coinmarketcap.com/v2/ticker/?start=101&limit=2').json()
        CMCinfo = CMCreq['data']
        # CMCinfo2 = CMCreq2['data']
        # CMCinfo.update(CMCinfo2)
        CMCcoins = pd.DataFrame(CMCinfo).transpose()[['name', 'symbol']]
        CMCcoins.iloc[29, 1] = 'IOT'

        self.coinlist = CMCcoins.merge(CCcoinlist, left_on='symbol', right_on='Name')[['CoinName', 'Name']]
        save_path = os.path.abspath(os.path.join('../data'))
        self.coinlist.to_pickle('{}/coinlist50.p'.format(save_path))

    def getHistDay(self, end, days, sleep=1, save=False):
        self.end = end
        self.days = days
        missingcoins = []
        fullPriceData = pd.DataFrame()
        for coin in self.coinlist['Name']:
            try:
                req = requests.get(
                    'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym=USD&limit={}&toTs={}'.format(coin,
                                                                                                               days,
                                                                                                               end)).json()
                info = req['Data']
                priceData = pd.DataFrame(info)
                priceData.insert(0, 'coin', coin)
                fullPriceData = fullPriceData.append(priceData, ignore_index=True)
            except Exception as e:
                print('unable to fetch {} price data').format(coin)
                print(e)
                missingcoins += [coin]
                pass
            time.sleep(sleep)

        if save:
            save_path = os.path.abspath(os.path.join('../data'))
            file = '{}/PriceList_{}_{}.csv'.format(save_path, datetime.fromtimestamp(end).strftime('%m%d%y'), days)
            fullPriceData.to_csv(file, index=False, encoding='utf-8')

        return fullPriceData
