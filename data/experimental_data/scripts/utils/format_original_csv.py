import pandas as pd

df = pd.read_csv("../../csv/original/device_original.csv", dtype={'user': str})
print(df)

# 各列に含まれる'+09:00'を取り除く
df['start_viewing_date'] = df['start_viewing_date'].str.replace('+09:00', '')
df['stop_viewing_date'] = df['stop_viewing_date'].str.replace('+09:00', '')
df['eliminate_date'] = df['eliminate_date'].str.replace('+09:00', '')
df['base_date'] = df['base_date'].str.replace('+09:00', '')

#user,action,device_id,url,start_viewing_date,stop_viewing_date,
df = df[['user', 'action', 'device_id', 'start_viewing_date', 'stop_viewing_date', 'url']]

df = df[df['action'] == 'view']
df = df.dropna()

df.to_csv("../../csv/original/device_original_formatted.csv", index=False)