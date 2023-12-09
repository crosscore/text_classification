#subprocess_script.py
import subprocess

script_file1 = "./scraping_from_yahoo_news.py"
script_file2 = "./concat_only_unique_urls.py"
script_file3 = "./remove_text_outliers.py"

subprocess.run(["python", script_file1])
subprocess.run(["python", script_file2])
subprocess.run(["python", script_file3])