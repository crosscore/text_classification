import pandas as pd
import os

categories = ['国内', '国際', '経済', 'エンタメ', 'スポーツ', 'IT・科学', 'ライフ', '地域']
new_category_dict = {
    'ニュース': ['国内', '国際', '経済'],
    'スポーツ': ['スポーツ'],
    'その他': ['エンタメ', 'IT・科学', 'ライフ', '地域']
}

output_path = './csv/change_category/yahoo_news_change_1116.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

original_df = pd.read_csv('./csv/concat/yahoo_news_concat_1116_v3.csv')

#original_dfの'category'列の文字列が、new_category_dictのvalueの中に存在する場合は
for category in categories:
    for key, value in new_category_dict.items():
        if category in value:
            #'category'列の値をnew_category_dictのkeyに変更
            original_df.loc[original_df['category'] == category, 'category'] = key
            break
print(original_df)
original_df.to_csv(output_path, index=False)
print(original_df['category'].unique())
