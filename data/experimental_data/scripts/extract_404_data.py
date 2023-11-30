import pandas as pd

csv_file = '../csv/add_category/device_with_category.csv'
df = pd.read_csv(csv_file, encoding='utf-8', dtype={'user': str})

df = df[df['category'] == '404_not_found']
print(df.info())

df.to_csv('../csv/add_category/404_not_found.csv', index=False, encoding='utf-8')