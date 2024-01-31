#subprocess_script.py
import subprocess

script_file0 = "./unzip_zip_file.py"
script_file1 = "./scraping_from_yahoo_news.py"
script_file2 = "./concat_only_unique_urls.py"
script_file3 = "./remove_text_outliers.py"
script_file4 = "./compress_csv_to_zip.py"

subprocess.run(["python3", script_file0])
subprocess.run(["python3", script_file1])
subprocess.run(["python3", script_file2])
subprocess.run(["python3", script_file3])
subprocess.run(["python3", script_file4])