import pandas as pd

csv_file = '../../csv/yahoo_news/concat/yahoo_news_concat_1126_v2.csv'

df = pd.read_csv(csv_file, encoding='utf-8')
print(df.head())

#dfのヘッダーリストの空のデータの数の合計を出力
print("df.isnull().sum()")
print(df.isnull().sum())