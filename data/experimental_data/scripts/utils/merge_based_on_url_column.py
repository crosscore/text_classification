import pandas as pd

all_user_plus = pd.read_csv('../../csv/original/all_user_plus_json_to_url.csv', dtype={'user':str})
device_with_category = pd.read_csv('../../csv/add_category/device_with_category_v3.csv', dtype={'user':str})

print(all_user_plus)
print(device_with_category)

# 'url'列を基にしてマージする
merged_data = all_user_plus.merge(device_with_category[['url', 'category']], on='url', how='left')

merged_data.to_csv('../../csv/complete/all_user_plus_category.csv', index=False)
