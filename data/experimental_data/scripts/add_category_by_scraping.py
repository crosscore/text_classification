#add_category_by_scraping.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import threading
import os

category_dict = {
    'domestic': '国内',
    'world': '国際',
    'business': '経済',
    'entertainment': 'エンタメ',
    'sports': 'スポーツ',
    'it': 'IT',
    'science': '科学',
    'life': 'ライフ',
    'local': '地域',
}

def get_category_from_archive(url, max_retries=6, wait_seconds=15):
    retries = 0
    time.sleep(3)
    while retries < max_retries:
        try:
            archive_url = f'https://web.archive.org/web/{url}'
            response = requests.get(archive_url, timeout=18)
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            for script in scripts:
                if 'currentCategory' in script.text:
                    match = re.search(r'"currentCategory":"(\w+)"', script.text)
                    if match:
                        category_key = match.group(1)
                        if category_key in category_dict:
                            print(f'category: {category_key}')
                            # "headline" キーの値を取得
                            headline_match = re.search(r'"headline":"([^"]+)"', script.text)
                            if headline_match:
                                title = headline_match.group(1)
                                print(f'title: {title}')
                            else:
                                title = 'タイトル不明'
                            return category_dict[category_key]
                        else:
                            print(f"category_dictに存在しないkey({category_key})です。")
                            return None
            return None  # スクリプト内でカテゴリが見つからなかった場合
        except requests.RequestException as e:
            print(f"Error processing URL {url}: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")
            time.sleep(wait_seconds)
    return "retry_limit_exceeded"  # 最大リトライ回数を超えた場合

def listen_for_exit_command():
    global exit_command_issued
    input("Press Enter to stop the process... ")
    exit_command_issued = True

start = time.time()
output_dir = "../csv/add_category"
os.makedirs(output_dir, exist_ok=True)
output_file_complete = "../csv/add_category/device_with_category.csv"
output_file_partial = "../csv/add_category/device_with_category_partial.csv"

exit_command_issued = False
exit_listener = threading.Thread(target=listen_for_exit_command)
exit_listener.start()

df = pd.read_csv('../csv/original/device_original.csv', dtype={'user': str})
# 'category' 列が存在しない場合は、空の列を作成
if 'category' not in df.columns:
    df['category'] = None
print(df)

retry_limit_exceeded = False
exit_command_detected = False
for index, row in df.iterrows():
    if exit_command_issued:
        print("Exit command issued. Saving partial data...")
        df.to_csv(output_file_partial, index=False)
        exit_command_detected = True
        break
    # 既にカテゴリが割り当てられている行はスキップ
    if pd.notna(row['category']):
        continue
    category = get_category_from_archive(row['url'])
    if category == "retry_limit_exceeded":
        retry_limit_exceeded = True
        print("Retry limit exceeded. Saving partial data...")
        break
    df.at[index, 'category'] = category

if not exit_command_detected: #Enterが押された場合completeの保存はスキップするため
    if retry_limit_exceeded:
        df.to_csv(output_file_partial, index=False)
    else:
        print("All data processed successfully. Saving data...")
        df.to_csv(output_file_complete, index=False)

end = time.time()
print(f"Elapsed time: {end - start} seconds.")
