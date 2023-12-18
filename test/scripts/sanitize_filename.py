import re

def sanitize_filename(url):
    # Replace characters that cannot be used in file names
    filename = re.sub(r'[\/:*?"<>|]', '_', url)
    return filename + '.html'

url = 'https://news.yahoo.co.jp/byline/sugieyuji/20221126-00325579'
filename = sanitize_filename(url)
print(filename)