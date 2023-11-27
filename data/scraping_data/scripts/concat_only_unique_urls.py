#csv_concat_and_to_only_unique_urls.py
import pandas as pd
import os
import datetime
import re
import time

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
    match = pattern.search(file)
    if match:
        version = int(match.group(1))
        if version > latest_version:
            latest_version = version
            latest_file = file
# バージョンが見つからない場合は初期バージョンを設定
if latest_version == 0:
    latest_file = f'yahoo_news_concat_{today_date}_v1.csv'
# 新しいバージョン番号を設定
next_version = latest_version + 1
input_old = os.path.join('../csv/yahoo_news/concat/', latest_file)
input_new = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{next_version}.csv'
output_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{next_version}.csv'
print(f'input_old: {input_old}')
print(f'input_new: {input_new}')
print(f'output_file: {output_file}')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_original = pd.read_csv(input_old)
print(f'before: {df_original["url"].nunique()}')

df_new = pd.read_csv(input_new)
df_concat = pd.concat([df_original, df_new])
df_concat.drop_duplicates(subset=['url'], inplace=True) # 'url'列の値が重複する場合削除
print(f"after : {df_concat['url'].nunique()}")
print(df_concat['category'].value_counts())

df_concat.to_csv(output_file, index=False)
