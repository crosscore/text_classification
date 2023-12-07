import pandas as pd
import glob

files = glob.glob("../../csv/yahoo_news/concat/*.csv")
df = pd.read_csv(files[0])

print(df['category'].value_counts())

#dfの'url'列の重複している行の数を出力
print(df['url'].duplicated().sum())

#dfの'url'の重複を削除する
df = df.drop_duplicates(subset='url')
print(df['url'].duplicated().sum())

print(df['category'].value_counts())

df.to_csv('../../csv/yahoo_news/concat/yahoo_news_concat_20231207_v8.csv', index=False)