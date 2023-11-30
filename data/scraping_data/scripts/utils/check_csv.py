import pandas as pd
import glob
import re

file_list = glob.glob('../../csv/yahoo_news/concat/*_v*.csv')
# バージョン番号に基づいて最新のファイルを選択
latest_file = max(file_list, key=lambda x: int(re.search(r'_v(\d+).csv', x).group(1)))
print(latest_file)
df = pd.read_csv(latest_file, encoding='utf-8')

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
print(f"df['category'].isnull().sum(): \n{df['category'].isnull().sum()}")