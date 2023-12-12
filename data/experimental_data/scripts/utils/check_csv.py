import pandas as pd
import numpy as np

df = pd.read_csv('../../csv/original/all_device.csv', encoding='utf-8', dtype={'user': str})
print(df)

print(f"df['url'].nunique():\n{df['url'].nunique()}")