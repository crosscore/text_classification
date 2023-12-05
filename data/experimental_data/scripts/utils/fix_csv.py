import pandas as pd

df = pd.read_csv("../../csv/man_power/all_404_not_found_v3_man.csv", dtype={'user': str})
#'user'列を文字列に変換し、4桁で0埋めする

df['user'] = df['user'].astype(str).str.zfill(4)
print(df)

df.to_csv("../../csv/man_power/all_404_not_found_v3_man2.csv", index=False)