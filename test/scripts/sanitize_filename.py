import re

def sanitize_filename(url):
    # ファイル名に使えない文字を置き換える
    filename = re.sub(r'[\/:*?"<>|]', '_', url)
    return filename + '.html'

url = 'https://news.yahoo.co.jp/byline/sugieyuji/20221126-00325579'
filename = sanitize_filename(url)
print(filename)