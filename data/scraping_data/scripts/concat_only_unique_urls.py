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

daily_files = os.listdir('../csv/yahoo_news/daily/')
daily_pattern = re.compile(rf'yahoo_news_articles_({today_date})_v(\d+)\.csv$')

latest_version = 0
latest_daily_file = ''
for file in daily_files:
    match = daily_pattern.search(file)
    if match:
        version = int(match.group(2))
        if version > latest_version:
            latest_version = version
            latest_daily_file = file

# Get the date of the last 7 days
recent_dates = [(now - datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]

# Find the latest version of the file for each date
latest_files = {}
for file in files:
     for date in recent_dates:
         pattern = re.compile(rf'{date}_v(\d+)\.csv$')
         match = pattern.search(file)
         if match:
             version = int(match.group(1))
             if date not in latest_files or version > latest_files[date][1]:
                 latest_files[date] = (file, version)

# select latest file
latest_file = None
if latest_files:
    latest_file = max(latest_files.values(), key=lambda x: x[1])[0]

concat_files = os.listdir('../csv/yahoo_news/concat/')
concat_pattern = re.compile(r'_v(\d+)\.csv$')

max_version = 0
for file in concat_files:
    match = concat_pattern.search(file)
    if match:
        version = int(match.group(1))
        if version > max_version:
            max_version = version

original_file = os.path.join('../csv/yahoo_news/concat/', latest_file) if latest_file else None
original_file_pattern = re.compile(rf'yahoo_news_concat_(\d{8})_v(\d+)\.csv$')
original_file_date = None
if original_file:
    match = original_file_pattern.search(original_file)
    if match:
        original_file_date = match.group(1)
if original_file_date is None or original_file_date < today_date:
    new_version = 1
else:
    new_version = max_version + 1

new_file = os.path.join('../csv/yahoo_news/daily/', latest_daily_file) if latest_daily_file else None
output_file = f'../csv/yahoo_news/concat/yahoo_news_concat_{today_date}_v{new_version}.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("============ exec concat_only_unique_urls.py ============")
print(f'input_old: {original_file}')
print(f'input_new: {new_file}')
print(f'output_file: {output_file}')

if original_file and os.path.exists(original_file):
    df_original = pd.read_csv(original_file)
else:
    df_original = pd.DataFrame()
print(f"df_original['category'].value_counts(dropna=False):\n{df_original['category'].value_counts(dropna=False)}")

before_num = df_original['url'].nunique()
df_new = pd.read_csv(new_file)
df_concat = pd.concat([df_original, df_new])

df_concat.drop_duplicates(subset=['title'], inplace=True)
df_concat.drop_duplicates(subset=['content'], inplace=True)
df_concat.drop_duplicates(subset=['url'], inplace=True)
df_concat.dropna()

# Leave only str lines (deals with float-only lines)
df_concat = df_concat[(df_concat['title'].apply(lambda x: isinstance(x, str))) & (df_concat['content'].apply(lambda x: isinstance(x, str)))]

# Processing categories
df_concat['category'] = pd.Categorical(df_concat['category'], categories=category_list, ordered=True)
df_concat.sort_values(by='category', inplace=True)
df_concat.to_csv(output_file, index=False)

# Check the data
#df = df_concat.copy()[~df_concat['url'].str.contains('/pickup/')]
df = df_concat.copy()
print('---------')
print(f'df.isnull().sum():\n{df.isnull().sum()}')
print('---------')
print(f'df.describe():\n{df.describe()}')
print('---------')
print(f"df['category'].unique():\n{df['category'].unique()}")
print('---------')
print(f"df['category'].value_counts(dropna=False):\n{df['category'].value_counts(dropna=False)}")
print('---------')
for column in df.columns:
    print(f"df['{column}'].duplicated().sum(): {df[column].duplicated().sum()}")
print('---------')
#Delete all .csv files in '../csv/yahoo_news/daily/' folder
for file in os.listdir('../csv/yahoo_news/daily/'):
    os.remove(os.path.join('../csv/yahoo_news/daily/', file))
    print(f"remove: {file}")
print('---------')
print(f"befor : {before_num}")
print(f"after : {df_concat['url'].nunique()}")