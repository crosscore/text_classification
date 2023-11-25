import pandas as pd

file_name = '../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1124_v5.csv'
df = pd.read_csv(file_name)
print(df.head())
print('---')
print(df.shape)
print('---')
print(df.columns)
print('---')
print(df.info())
print('---')
print(df['category'].value_counts())
