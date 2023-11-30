import pandas as pd

csv_file = '../csv/add_category/404_not_found.csv'

df = pd.read_csv(csv_file, encoding='utf-8')

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