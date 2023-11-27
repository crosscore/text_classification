import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import datetime
import re

# 今日の日付を取得し、指定されたフォーマットに変換
today_date = datetime.datetime.now().strftime('%Y%m%d')
# ディレクトリ内のファイル名を取得
files = os.listdir('../csv/yahoo_news/concat/')
# ファイル名から日付とバージョンを抽出するための正規表現
pattern = re.compile(rf'{today_date}_v(\d+)\.csv$')
# 最新のバージョンを探す
latest_version = 0
latest_file = ''
for file in files:
    print(file)
    match = pattern.search(file)
    print(match)
    if match:
        version = int(match.group(1))
        print(version)
        if version > latest_version:
            latest_version = version
            latest_file = file
# バージョンが見つからない場合は初期バージョンを設定
if latest_version == 0:
    latest_file = f'yahoo_news_concat_{today_date}_v1.csv'
# 新しいバージョン番号を設定
next_version = latest_version + 1
output_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{next_version}.csv'
print(output_file)