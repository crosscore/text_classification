#livedoor_news_to_csv.py
import os
import csv

def extract_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        title = lines[2].strip() # 3rd line is title
        content = ''.join(lines[3:]).strip() # Content from the 4th line onwards
    return title, content

def check_empty_entries(root_dir):
    empty_titles = 0
    empty_contents = 0
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if os.path.isdir(category_dir):
            for file in os.listdir(category_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(category_dir, file)
                    title, content = extract_data(file_path)
                    if not title:
                        empty_titles += 1
                    if not content:
                        empty_contents += 1
    return empty_titles, empty_contents

def create_csv(root_dir, output_file):
    empty_titles, empty_contents = check_empty_entries(root_dir)
    print(f"空のタイトルの数: {empty_titles}")
    print(f"空のコンテンツの数: {empty_contents}")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['category', 'title', 'content'])

        for category in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                for file in os.listdir(category_dir):
                    if file.endswith('.txt'):
                        file_path = os.path.join(category_dir, file)
                        title, content = extract_data(file_path)
                        writer.writerow([category, title, content])

root_dir = '../../../livedoor_news/text'
output_file = '../../data/livedoor_news_corpus/csv/livedoor_news_data.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
create_csv(root_dir, output_file)
