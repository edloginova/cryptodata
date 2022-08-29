# Cryptocurrency Dataset
The dataset covers comments on multiple cryptocurrencies as well as financial indicators and search trends from 7 August 2015 to 7 April 2019. We release it under the CC BY-SA 4.0 licence.

Full corpus in .csv and pickle format per source can be found on [Google Drive](https://drive.google.com/drive/folders/1a4d1oojHuYm5Ft0vGTHTgSPam6q4DbnU?usp=sharing).

## Data sources
* financial data from CryptoCompare: the daily opening and closing price, high-low and volume.
* search queries frequency from Google Trends
* textual data from forums and news: CryptoCompare, Reddit and Bitcointalk: 78 902, 2 635 046 and 1 643 705 texts, respectively.

## Format
Textual data is provided in .json files. Each source has fields text, post_id, and date. Bitcointalk and Reddit also have parent_id field which aligns comments with the first thread messages. For price and trends data, associated coin name and date are provided.

## Method
Seach queries frequency from Google Trends was collected using Pytrends. Reddit textual data is fetched through Pushshift.io. For Cryptocompare, we scraped news using the official API of the aggregator. For Bitcointak, a custom Selenium scraper was implemented.
We only include BTC, ETH, LTC, XPR, and XMR - the top five cryptocurrencies based on market capitalisation (accordng to coinmarketcap.com using an official API. Coins that were re-branded (had their name changed) or did not have sufficient coverage during the research time period were excluded.
You can read more about the data collection process and possible feature engineering schemes in our paper "Forecasting Directional Bitcoin Price Returns using Aspect-based Sentiment Analysis on Online Communities Data".

## Task
The goal is binary classification of directional returns, which are calculated using closing prices. An upward movement in the closing price corresponds to the positive class and no movement or a downward movement is the negative class. If you would like your results to be added to the leaderboard, send us an email.

## Team
We are a part of Data Mining laboratory of Faculty of Economics at Ghent University, Belgium. This project was carried out by Wai Kit Tsang and Ekaterina Loginova under the supervision of Prof Dr Dries Benoit based on Guus van Heijningen's master thesis.
