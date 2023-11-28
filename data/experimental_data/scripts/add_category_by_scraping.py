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

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def get_category_from_archive(url, max_retries=5, wait_seconds=12, max_wait_seconds=60):
    retries = 0
    time.sleep(6)
    while retries < max_retries:
        try:
            archive_url = f'https://web.archive.org/web/{url}'
            response = requests.get(archive_url, timeout=18)
            response.raise_for_status() #HTTPエラーをチェック
            if response.status_code == 200:
                print("Request successful. Status code: 200")

                # Save the HTML file only if status code is 200
                html_dir = '../html/archive_files/'
                os.makedirs(html_dir, exist_ok=True)
                file_name = sanitize_filename(url.split("/")[-1]) + '.html'
                with open(f'{html_dir}{url.split("/")[-1]}.html', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f'Saved HTML file: {file_name}')
                
                # HTMLの内容を解析してカテゴリを見つける
                soup = BeautifulSoup(response.content, 'html.parser')
                scripts = soup.find_all('script')
                for script in scripts:
                    if 'currentCategory' in script.text:
                        print('Discover currentCategory.')
                        match = re.search(r'"currentCategory":"(\w+)"', script.text)
                        if match:
                            print(f'match.group(1): {match.group(1)}')
                            category_key = match.group(1)
                            if category_key in category_dict:
                                print(f'category_key: {category_key}')
                                headline_match = re.search(r'"headline":"([^"]+)"', script.text)
                                if headline_match:
                                    title = headline_match.group(1)
                                    print(f'headline_match.group(1): {title}')
                                else:
                                    title = 'None'
                                return category_dict[category_key]
                            else:
                                print(f"category_dictに存在しないkey({category_key})です。")
                                return None
                else:
                    print(f"Request failed with status code: {response.status_code}")
                    retries += 1
                    time.sleep(wait_seconds)
                    wait_seconds = min(wait_seconds * 2, max_wait_seconds)  # 待機時間を指数的に増やすが最大値は超えない
                    continue  # Retry the request

        except requests.exceptions.Timeout as e:
            print(f"Timeout error with URL {url}: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")
            time.sleep(wait_seconds)
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error with URL {url}: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")
            time.sleep(wait_seconds)
        except requests.exceptions.HTTPError as e:
            # 404エラーを特定して処理
            if e.response.status_code == 404:
                print(f"404 Not Found error for URL {url}: {e}")
                return "not_found"
            else:
                print(f"HTTP error with URL {url}: {e}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(wait_seconds)
                continue
        except requests.exceptions.TooManyRedirects as e:
            print(f"Too many redirects for URL {url}: {e}")
            return "too_many_redirects"
        except requests.RequestException as e:
            print(f"Error processing URL {url}: {e}")
            return "request_exception"
    
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
    print("Adding 'None' to the category column")
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
    category = None
    while category is None:
        category = get_category_from_archive(row['url'])
        if category == "retry_limit_exceeded" or category == "request_exception":
            print("Retry limit exceeded. | RequestException occured. Saving partial data...")
            retry_limit_exceeded = True
            break
    if category is not None:
        df.at[index, 'category'] = category
        print(f"Row {index+1}/{len(df)}: Added category '{category}'")

if not exit_command_detected: #Enterが押された場合completeの保存はスキップするため
    if retry_limit_exceeded:
        df.to_csv(output_file_partial, index=False)
    else:
        print("All data processed successfully. Saving data...")
        df.to_csv(output_file_complete, index=False)

end = time.time()
print(f"Elapsed time: {end - start} seconds.")
