import pandas as pd

device_original = pd.read_csv('../../../csv/original/device_original_formatted.csv', dtype={'user':str})
device_with_category = pd.read_csv('../../../csv/add_category/device_with_category_v3.csv', dtype={'user':str})

# 'category' 列を device_original に追加
device_original['category'] = ''

# 'url' 列の値が一致する行の 'category' 値を更新
for i, row in device_original.iterrows():
    url = row['url']
    category = device_with_category[device_with_category['url'] == url]['category']
    if not category.empty:
        device_original.at[i, 'category'] = category.iloc[0]

device_original.to_csv('../../../csv/complete/device_original_with_category.csv', index=False)
print(device_original)
