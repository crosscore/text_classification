#remove_text_outliers.py
import pandas as pd
import os
import numpy as np
import datetime
import re

def remove_outliers(df, column):
    lengths = df[column].str.len()
    mean_len = lengths.mean()
    std_len = lengths.std()
    upper_threshold = mean_len + 3 * std_len
    lower_threshold = mean_len - 3 * std_len
    return df[(lengths <= upper_threshold) & (lower_threshold <= lengths)]

def find_latest_csv():
    now = datetime.datetime.now()
    today_date = now.strftime('%Y%m%d')
    yesterday_date = (now - datetime.timedelta(days=1)).strftime('%Y%m%d')

    files = os.listdir('../csv/yahoo_news/concat/')
    pattern = re.compile(rf'({yesterday_date}|{today_date})_v(\d+)\.csv$')

    latest_file = ''
    latest_version = 0
    for file in files:
        match = pattern.search(file)
        if match:
            version = int(match.group(2))
            if version > latest_version:
                latest_version = version
                latest_file = file
    return os.path.join('../csv/yahoo_news/concat/', latest_file) if latest_file else None

print("====== exec remove_text_outliers.py ======")
latest_csv = find_latest_csv()
print(f"latest_csv: {latest_csv}")
if latest_csv and os.path.exists(latest_csv):
    df = pd.read_csv(latest_csv)
    print(f"before: {df['title'].nunique()}")
    # 異常値の削除
    df = remove_outliers(df, 'title')
    df = remove_outliers(df, 'content')

    # 処理後のデータを保存
    df.to_csv(latest_csv, index=False)
else:
    print("最新のCSVファイルが見つかりませんでした。")
print(f"after: {df['title'].nunique()}")
print('---------')
print(f"df['category'].value_counts(dropna=False):\n{df['category'].value_counts(dropna=False)}")
print('---------')