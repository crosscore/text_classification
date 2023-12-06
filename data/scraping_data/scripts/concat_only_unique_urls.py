#concat_only_unique_urls.py
import pandas as pd
import os
import datetime
import re

category_list = ['国内', '国際', '経済', 'エンタメ', 'スポーツ', 'IT', '科学', 'ライフ', '地域']

now = datetime.datetime.now()
today_date = now.strftime('%Y%m%d')
yesterday_date = (now - datetime.timedelta(days=1)).strftime('%Y%m%d')

files = os.listdir('../csv/yahoo_news/concat/')
pattern_yesterday = re.compile(rf'{yesterday_date}_v(\d+)\.csv$')
pattern_today = re.compile(rf'{today_date}_v(\d+)\.csv$')

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

if latest_version_today > 0:
    version = latest_version_today + 1
else:
    version = 1

new_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{version}.csv'
output_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{version}.csv'
original_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{latest_version_today}.csv' if latest_version_today > 0 else os.path.join('../csv/yahoo_news/concat/', latest_file_yesterday) if latest_file_yesterday else None
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f'input_old: {original_file}')
print(f'input_new: {new_file}')
print(f'output_file: {output_file}')

if original_file and os.path.exists(original_file):
    df_original = pd.read_csv(original_file)
    # 個別の重複削除（url, title, content）
    df_original.drop_duplicates(subset=['url', 'title', 'content'], inplace=True)
else:
    df_original = pd.DataFrame()

print(f"befor : {df_original['url'].nunique()}")

# 新規ファイルの読み込み
df_new = pd.read_csv(new_file)
# 個別の重複削除（url, title, content）
df_new.drop_duplicates(subset=['url', 'title', 'content'], inplace=True)

# データの結合
df_concat = pd.concat([df_original, df_new])

# 結合後の重複削除（url, title, content）
df_concat.drop_duplicates(subset=['url', 'title', 'content'], inplace=True)

# 個別の列に対する重複削除
df_concat.drop_duplicates(subset=['title'], inplace=True)
df_concat.drop_duplicates(subset=['content'], inplace=True)

# NaN値の削除（title, content）
df_concat.dropna(subset=['title', 'content'], inplace=True)

# カテゴリの処理
df_concat['category'] = pd.Categorical(df_concat['category'], categories=category_list, ordered=True)
df_concat.sort_values(by='category', inplace=True)

# ファイルへの出力
df_concat.to_csv(output_file, index=False)
print(f"after : {df_concat['url'].nunique()}")

# データの確認
df = df_concat.copy()[~df_concat['url'].str.contains('/pickup/')]
print('---------')
print(f'df.isnull().sum():\n{df.isnull().sum()}')
print('---------')
print(f'df.info():\n{df.info()}')
print('---------')
print(f'df.describe():\n{df.describe()}')
print('---------')
print(f"df['category'].unique():\n{df['category'].unique()}")
print('---------')
print(f"df['category'].value_counts(dropna=False):\n{df['category'].value_counts(dropna=False)}")
print('---------')
for column in df.columns:
    print(f"df['{column}'].duplicated().sum(): {df[column].duplicated().sum()}")
print(f"after : {df_concat['url'].nunique()}")
