#scraping_from_yahoo_news_v2.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import datetime
import re

today_date = datetime.datetime.now().strftime('%Y%m%d')
concat_dir = '../csv/yahoo_news/concat/'

if os.path.exists(concat_dir) and os.listdir(concat_dir):
    files = os.listdir(concat_dir)
    pattern = re.compile(rf'{today_date}_v(\d+)\.csv$')
    latest_version = 0
    for file in files:
        match = pattern.search(file)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
    next_version = latest_version + 1
    output_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v{next_version}.csv'
else:
    print("First scraping today or directory does not exist or is empty.")
    output_file = f'../csv/yahoo_news/daily/yahoo_news_articles_{today_date}_v1.csv'
print(output_file)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

base_url = 'https://news.yahoo.co.jp/rss/categories/'
categories = {
    '国内': 'domestic',
    '国際': 'world',
    '経済': 'business',
    'エンタメ': 'entertainment',
    'スポーツ': 'sports',
    'IT': 'it',
    '科学': 'science',
    'ライフ': 'life',
    '地域': 'local'
}
urls = {category: f'{base_url}{categories[category]}.xml' for category in categories}

skipped_articles = 0
MAX_RETRIES = 6
RETRY_INTERVAL = 12 # seconds

def scrape_news(category, url):
    for i in range(MAX_RETRIES):
        try:
            print(f'scrape_news... {category}: {url}')
            response = requests.get(url, timeout=(12, 18))
            print(response.status_code)
            soup = BeautifulSoup(response.content, 'html.parser')
            if "指定されたURLは存在しませんでした。" in str(soup):
                print(f"Error page detected, skipping remaining page for ({category}) url:{url}")
                return []
            articles = []
            for item in soup.find_all("item"):
                title = item.find("title").get_text(strip=True)
                link_tag = item.find("link")
                link = link_tag.next_sibling.strip() if link_tag else ""  # <link> タグの次のテキストノードを取得
                print(f"{category}: {title}")
                print(f"{link}")
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

print('Start scraping.')
start = time.time()
all_articles = []
for category, url in urls.items():
    print(f'{category}: {url}')
    articles = scrape_news(category, url)
    if not articles:
        break
    print(f'Found {len(articles)} articles.')
    all_articles.extend(articles)
print(f'Complete. Found {len(all_articles)} articles.')

all_articles_data = []
for article in all_articles:
    category, title, link = article
    print(f'Scraping ({category}) {title} ...')
    content = scrape_article_content(link)
    print(f'Found {len(content)} characters.')
    all_articles_data.append([category, title, link, content])
    time.sleep(0.9)
df = pd.DataFrame(all_articles_data, columns=['category', 'title', 'url', 'content'])
df.to_csv(output_file, index=False, encoding='utf-8')

print('End scraping.')
print(f'Number of skipped articles: {skipped_articles}')
end = time.time()
print(f'Time elapsed: {end - start} seconds.')
