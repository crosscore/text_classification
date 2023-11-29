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

def sanitize_filename(url):
    filename = re.sub(r'[\/:*?"<>|]', '_', url)
    return filename + '.html'

def file_exists(file_path):
    return os.path.exists(file_path)

def parse_html_for_category(html_content):
    print("--- parse_html_for_category ---")
    soup = BeautifulSoup(html_content, 'html.parser')
    scripts = soup.find_all('script')
    category_found = False
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
                    category_found = True
                    return category_dict[category_key]
                else:
                    print(f"category_dictに存在しないkey({category_key})です。")
            continue
        elif 'id_type' in script.text:
            category_info = re.search(r'"id_type":"(\w+)"', script.text)
            if category_info:
                category_key = category_info.group(1)
                if category_key in category_dict:
                    category_found = True
                    return category_dict[category_key]
            continue
    if not category_found:
        metatags = soup.find_all('meta')
        for meta in metatags:
            if 'name' in meta.attrs and meta.attrs['name'] == 'description':
                content = meta.attrs.get('content', '').lower()  # 小文字に変換
                if 'スポーツ' in content:  # 'スポーツ'が含まれているかチェック
                    print("Discover categories among meta tags: return 'スポーツ'")
                    return 'スポーツ'
        print("######### Category not found in HTML content. #########")
        return "category_not_found"

def get_category_from_archive(url, max_retries=3, wait_seconds=12, max_wait_seconds=60):
    time.sleep(9)
    print(f"--------------- Analyzing {url} ---------------")
    retries = 0
    html_dir = '../html/archive_files/'
    file_name = sanitize_filename(url)
    file_path = f'{html_dir}{file_name}'
    print(file_path)
    html_content = ''
    if file_exists(file_path):
        print(f'File already exists: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        while retries < max_retries:
            try:
                archive_url = f'https://web.archive.org/web/{url}'
                print(f"Trying requests.get({archive_url})")
                response = requests.get(archive_url, timeout=18)
                response.raise_for_status()
                print("Request successful. Status code: 200")
                # 200ステータスコードの場合のみHTMLを保存
                os.makedirs(html_dir, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    print(f'open: {file_path}')
                    f.write(response.text)
                print(f'Saved HTML file: {file_name}')
                html_content = response.text
                break
            except requests.exceptions.Timeout as e:
                print(f"Timeout with URL {url}: {e}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(wait_seconds)
            except requests.exceptions.ConnectionError as e:
                print(f"ConnectionError with URL {url}: {e}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(wait_seconds)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"404 Not Found error for URL {url}: {e}")
                    return "404_not_found"
                elif e.response.status_code == 403:
                    print(f"403 Forbidden error for URL {url}: {e}")
                    return "403_forbidden"
                else:
                    print(f"HTTP error with URL {url}: {e}")
                    retries += 1
                    print(f"Retrying... ({retries}/{max_retries})")
                    time.sleep(wait_seconds)
                    continue
            except requests.exceptions.TooManyRedirects as e:
                print(f"Too many redirects for URL {url}: {e}")
                return "too_many_redirects"
            except requests.exceptions.RequestException as e:
                print(f"Error with URL {url}: {e}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2, max_wait_seconds)  # 待機時間を倍にして、最大値に制限を設ける
    # HTMLの内容を解析してカテゴリを見つける
    if html_content:
        return parse_html_for_category(html_content)
    print("Retry_limit_exceeded.")
    return "retry_limit_exceeded"

def listen_for_exit_command():
    global exit_command_issued
    global exit_thread
    while not exit_thread:
        input("Press Enter to stop the process... ")
        exit_command_issued = True
        break


start = time.time()
exit_command_issued = False
exit_thread = False

output_dir = "../csv/add_category"
os.makedirs(output_dir, exist_ok=True)
output_file_complete = "../csv/add_category/device_with_category.csv"
output_file_partial = "../csv/add_category/device_with_category_partial.csv"

# "device_with_category_partial.csv"の存否により処理を分岐
if os.path.exists(output_file_partial):
    print(f"Loading partial data from {output_file_partial}")
    df = pd.read_csv(output_file_partial, dtype={'user': str})
else:
    print("Loading original data")
    df = pd.read_csv('../csv/original/device_original.csv', dtype={'user': str})
    # 'category' 列が存在しない場合は、空の列を作成
    if 'category' not in df.columns:
        df['category'] = None

exit_command_issued = False
exit_listener = threading.Thread(target=listen_for_exit_command)
exit_listener.start()

retry_limit_exceeded = False
exit_command_detected = False #スレッド終了フラグ

error_urls = []
categories = []
for index, row in df.iterrows():
    if exit_command_issued:
        print("Exit command issued. Saving partial data...")
        # 全行にカテゴリを割り当てるまで、現在のcategoriesの長さをチェック
        while len(categories) < len(df):
            categories.append(None) #未割り当ての行にはNoneを追加
        df['category'] = categories  # 現在までの結果を保存
        df.to_csv(output_file_partial, index=False)
        print(f"{output_file_partial} was saved: ")
        exit_command_detected = True
        break
    # 既にカテゴリが割り当てられている行はスキップ
    if pd.notna(row['category']):
        categories.append(row['category'])
        continue
    # get_category_from_archive関数を呼び出してカテゴリを取得
    category = get_category_from_archive(row['url'])
    if category in ["retry_limit_exceeded", "request_exception", "category_not_found"]:
        print("Error occurred. URL will be reprocessed later.")
        categories.append(None)  # エラー時はNoneを追加
        error_urls.append((row['url'], row['title']))
        continue
    # 取得したカテゴリをリストに追加
    categories.append(category)
# 'category' 列を更新（もしくは追加）
df['category'] = categories

# 処理が完了した場合、完全なデータとエラーURLのデータを保存
if not exit_command_detected:
    if retry_limit_exceeded:
        df.to_csv(output_file_partial, index=False)
    else:
        print("All data processed successfully. Saving data...")
        df.to_csv(output_file_complete, index=False)

# error_urlsが空でなければ、それをCSVファイルとして保存
if error_urls:
    error_df = pd.DataFrame(error_urls, columns=['url', 'title'])
    error_output_file = "../csv/add_category/error_urls.csv"
    error_df.to_csv(error_output_file, index=False)
    print(f"Error URLs saved to {error_output_file}")

exit_thread = True
exit_listener.join()
end = time.time()
print(f"Elapsed time: {end - start} seconds.")
