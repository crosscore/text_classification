import pandas as pd
import os
import datetime
import re
import time

# 今日の日付と前日の日付を取得
today_date = datetime.datetime.now().strftime('%Y%m%d')
yesterday_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')

# ディレクトリ内のファイル名を取得
files = os.listdir('../csv/yahoo_news/concat/')

# ファイル名から日付とバージョンを抽出するための正規表現
pattern_yesterday = re.compile(rf'{yesterday_date}_v(\d+)\.csv$')
pattern_today = re.compile(rf'{today_date}_v(\d+)\.csv$')

# 前日の最新のバージョンを探す
latest_version_yesterday = 0
latest_file_yesterday = ''
latest_version_today = 0
for file in files:
    match_yesterday = pattern_yesterday.search(file)
    match_today = pattern_today.search(file)
    if match_yesterday:
        version = int(match_yesterday.group(1))
        if version > latest_version_yesterday:
            latest_version_yesterday = version
            latest_file_yesterday = file
    elif match_today:
        version = int(match_today.group(1))
        if version > latest_version_today:
            latest_version_today = version

# 本日のファイルが存在するかチェックし、バージョンを設定
if latest_version_today > 0:
    version = latest_version_today + 1
else:
    version = 1

# 本日のファイルが存在するかチェックし、バージョンを設定
if latest_version_today > 0:
    version = latest_version_today + 1
    # 本日の2度目以降の実行の場合のファイル名設定
    input_new = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{version}.csv'
    output_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{version}.csv'
    # 本日の2度目以降の実行の場合の前日のファイル名設定
    input_old = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{latest_version_today}.csv'
else:
    # 本日初めての実行の場合のファイル名設定
    input_new = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v1.csv'
    output_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v1.csv'
    # 本日初めての実行の場合の前日のファイル名設定
    input_old = os.path.join('../csv/yahoo_news/concat/', latest_file_yesterday) if latest_file_yesterday else None


print(f'input_old: {input_old}')
print(f'input_new: {input_new}')
print(f'output_file: {output_file}')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 前日のファイルが存在すれば読み込む、存在しなければ空のDataFrameを作成
if input_old and os.path.exists(input_old):
    df_original = pd.read_csv(input_old)
else:
    df_original = pd.DataFrame()

print(f'before: {df_original["url"].nunique()}')

# 本日のデータの読み込み
df_new = pd.read_csv(input_new)

# データの結合
df_concat = pd.concat([df_original, df_new])

# 'url'列の値が重複する場合削除
df_concat.drop_duplicates(subset=['url'], inplace=True)

print(f"after : {df_concat['url'].nunique()}")
print(df_concat['category'].value_counts())

# 結合したデータを保存
df_concat.to_csv(output_file, index=False)
