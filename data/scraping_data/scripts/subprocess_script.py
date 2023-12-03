import subprocess

script_file1 = "./scraping_from_yahoo_news.py"
script_file2 = "./concat_only_unique_urls.py"

subprocess.run(["caffeinate", "-i", "python", script_file1])
subprocess.run(["caffeinate", "-i", "python", script_file2])