#scraping_from_yahoo_news.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import os
import datetime
import re

# 今日の日付を取得し、指定されたフォーマットに変換
today_date = datetime.datetime.now().strftime('%Y%m%d')
files = os.listdir('../csv/yahoo_news/concat/')
# ファイル名から日付とバージョンを抽出するための正規表現
pattern = re.compile(rf'{today_date}_v(\d+)\.csv$')
# 最新のバージョンを探す
latest_version = 0
for file in files:
    match = pattern.search(file)
    if match:
        version = int(match.group(1)) #group(0): 正規表現に一致した全体の文字列,　group(1): 1番目のグループの文字列
        if version > latest_version:
            latest_version = version
# 今日の日付のファイルが存在しない場合、v1として設定
if latest_version == 0:
    print("First scraping today.")
    output_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v1.csv'
else:
    # 新たなバージョン番号を設定
    next_version = latest_version + 1
    output_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{next_version}.csv'
print(output_file)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
skipped_articles = 0
# urls = {
#     '国内': 'https://news.yahoo.co.jp/topics/domestic',
#     '国際': 'https://news.yahoo.co.jp/topics/world',
#     '経済': 'https://news.yahoo.co.jp/topics/business',
#     'エンタメ': 'https://news.yahoo.co.jp/topics/entertainment',
#     'スポーツ': 'https://news.yahoo.co.jp/topics/sports',
#     'IT': 'https://news.yahoo.co.jp/topics/it',
#     '科学': 'https://news.yahoo.co.jp/topics/science',
#     'ライフ': 'https://news.yahoo.co.jp/categories/life',
#     '地域': 'https://news.yahoo.co.jp/topics/local'
# }
urls = {
    'ライフ': 'https://news.yahoo.co.jp/categories/life'
}
add_url_list = ['?page=1', '?page=2', '?page=3', '?page=4', '?page=5', '?page=6']
MAX_RETRIES = 6
RETRY_INTERVAL = 12 # seconds

def scrape_news(category, url):
    for i in range(MAX_RETRIES):
        try:
            print(f'scrape_news... category: {category}, url: {url}')
            response = requests.get(url, timeout=(12, 18))
            print(response.status_code)
            soup = BeautifulSoup(response.content, 'html.parser')
            if "指定されたURLは存在しませんでした。" in str(soup):
                print(f"Error page detected, skipping remaining page for ({category}) url:{url}")
                return []
            articles = []
            for item in soup.find_all("a", class_="newsFeed_item_link"):
                title = item.find("div", class_="newsFeed_item_title").get_text(strip=True)
                link = item['href']
                print(f"category: {category}, title: {title}")
                print(f"link: {link}")
                articles.append((category, title, link))
            time.sleep(1)
            return articles
        except:
            print(f"Failed, retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
    print("Failed scrape_news after max retries.")
    return []

def scrape_article_content(link,  MAX_RETRIES = 6):
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            print(f"scrape_article_content... url: {link}")
            response = requests.get(link, timeout=(12, 18))
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraph = soup.find('p', class_=lambda x: x and 'highLightSearchTarget' in x)
            if paragraph:
                return paragraph.get_text(strip=True)
            else:
                return ""
        except requests.RequestException as e:
            attempts += 1
            print(f"Error occurred: {e}, retrying {attempts}/{MAX_RETRIES}")
            time.sleep(6)
    print(f"Failed to scrape_article_content after {MAX_RETRIES} attempts, skipping URL: {link}")
    skipped_articles += 1
    return ""

def scrape_dynamic_news2(url, max_clicks=30):
    options = Options()
    options.headless = True  # ブラウザをGUIなしで起動
    options.add_argument("--disable-logging")  # ログ出力を無効にする
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    articles = []
    for _ in range(max_clicks):
        try:
            more_button = driver.find_element(By.CLASS_NAME, 'FeedLoadMoreButton__ReadMoreButton-kbeqvq')
            driver.execute_script("arguments[0].click();", more_button)
            time.sleep(5)
        except NoSuchElementException:
            print("No more 'もっと見る' button found.")
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    for item in soup.find_all("a", class_="newsFeed_item_link"):
        title = item.find("div", class_="newsFeed_item_title").get_text(strip=True)
        link = item['href']
        articles.append((title, link))
    return articles

def scrape_dynamic_news(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    articles = []
    # すべての関連するクラス名を含むボタンを取得
    button_classes = ['sc-glMzqP.bVVtQV', 'TabItemText__Button-mjrpV.sAzpp', 'TabItemText__Button-mjrpV.cqquRy']
    for class_name in button_classes:
        buttons = driver.find_elements(By.CLASS_NAME, class_name)
        for button in buttons:
            try:
                print(f"Clicking button: {button.text}")
                time.sleep(3)
                button.click()
                time.sleep(2)
                new_page = driver.current_url
                articles.append(new_page)
            except Exception as e:
                print(f"Error occurred while clicking or loading page: {e}")
    driver.quit()
    return articles

print('Start scraping.')
start = time.time()
all_articles = []
for category, url in urls.items():
    if category == 'ライフ':
        print(f'category: {category}, url: {url}')
        dynamic_articles = scrape_dynamic_news(url)
        for title, link in dynamic_articles:
            all_articles.append((category, title, link))
        print(f'Found {len(dynamic_articles)} dynamic articles.')
    else:
        for add_url in add_url_list:
            url_with_page = url + add_url
            print(f'category: {category}, url: {url_with_page}')
            articles = scrape_news(category, url_with_page)
            if not articles:
                break
            print(f'Found {len(articles)} articles.')
            all_articles.extend(articles)
print(f'Complete. Found {len(all_articles)} articles.')
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['category', 'title', 'url', 'content'])
    for article in all_articles:
        category, title, link = article
        print(f'Scraping ({category}) {title} ...')
        content = scrape_article_content(link)
        print(f'Found {len(content)} characters.')
        writer.writerow([category, title, link, content])
        time.sleep(0.9)
print('End scraping.')
print(f'Number of skipped articles: {skipped_articles}')
end = time.time()
print(f'Time elapsed: {end - start} seconds.')
