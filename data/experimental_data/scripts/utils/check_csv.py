import pandas as pd
import glob

csv_files = glob.glob('../../csv/original/*.csv')
#csv_files = glob.glob('../csv/original/all_404_not_found_v3.csv')
print(f'csv_files: {csv_files[0]}')
df = pd.read_csv(csv_files[0], encoding='utf-8', dtype={'user': str})
print('---------')
print(f'df.isnull().sum():\n{df.isnull().sum()}')
print('---------')
print(f'df.info():\n{df.info()}')


print('---------')
print('---------')

print(f'csv_files: {csv_files[1]}')
df = pd.read_csv(csv_files[1], encoding='utf-8', dtype={'user': str})
print('---------')
print(f'df.isnull().sum():\n{df.isnull().sum()}')
print('---------')
print(f'df.info():\n{df.info()}')