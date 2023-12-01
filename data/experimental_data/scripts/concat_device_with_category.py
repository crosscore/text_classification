import pandas as pd

df1 = pd.read_csv('../csv/add_category/device_with_category.csv')
df2 = pd.read_csv('../csv/add_category/device_with_category_v2.csv')

# 'url'をインデックスに設定
df1 = df1.set_index('url')
df2 = df2.set_index('url')

# df1をdf2のデータで更新
df1.update(df2['category'])

# インデックスをリセット
df = df1.reset_index()
    
df.to_csv('../csv/add_category/device_with_category_v3.csv', index=False)

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