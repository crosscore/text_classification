import pandas as pd
import glob
import zipfile
import os

file_path = '../../data/scraping_data/csv/yahoo_news/concat/*.zip'
zip_file_list = glob.glob(file_path)

for zip_file in zip_file_list:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the list of files in the zip file
        zip_files = zip_ref.namelist()
        # unzip to temporary directory
        zip_ref.extractall("temp_dir")

        for file in zip_files:
            if file.endswith('.csv'):
                df = pd.read_csv(f"temp_dir/{file}")
                print(df.head())
                print('---')
                print(df.shape)
                print('---')
                print(df.columns)
                print('---')
                print(df.info())
                print('---')
                print(df['category'].value_counts())

        # delete temporary directory
        for file in zip_files:
            os.remove(f"temp_dir/{file}")
        os.rmdir("temp_dir")

        # Compression processing
        with zipfile.ZipFile(f"{zip_file[:-4]}_analyzed.zip", 'w') as zipf:
            for file in zip_files:
                if file.endswith('.csv'):
                    zipf.write(f"temp_dir/{file}", arcname=file)
