import pandas as pd
import glob

csv_files = glob.glob('../csv/man_power/*.csv')
print(f'csv_files: {csv_files}')
df = pd.read_csv(csv_files[0], encoding='utf-8')

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
print('---------')
print(f"df[df['category'].isnull()]:\n{df[df['category'].isnull()]}")
