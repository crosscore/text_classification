#csv_concat_and_to_only_unique_urls.py
import pandas as pd
import os

input_old = '../csv/yahoo_news/concat/yahoo_news_concat_1126_v3.csv'
input_new = '../csv/yahoo_news/daily/yahoo_news_articles_1126_v4.csv'

output_path = '../csv/yahoo_news/concat/yahoo_news_concat_1126_v4.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df_original = pd.read_csv(input_old)
print(df_original['url'].nunique())

df_new = pd.read_csv(input_new)
df_concat = pd.concat([df_original, df_new])
df_concat.drop_duplicates(subset=['url'], inplace=True)
print(df_concat['url'].nunique())
print(df_concat['category'].value_counts())

df_concat.to_csv(output_path, index=False)