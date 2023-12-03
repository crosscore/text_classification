import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import glob
import time

def clean_text(text):
    text = text.strip()
    return text

start = time.time()
PRE_TRAINED = '../versions/v102/'
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED)

# テストデータの読み込み
test_data_path = 'path/to/test.csv'
test_df = pd.read_csv(test_data_path)
test_df['text'] = test_df['title'] + '。' + test_df['content']
test_df['text'] = test_df['text'].apply(clean_text)

# 訓練データからラベルエンコーダのマッピングを作成
train_data_path = glob.glob('../../../data/scraping_data/csv/yahoo_news/concat/*.csv')
df = pd.read_csv(train_data_path[0])
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

# ラベルマッピングの逆変換用辞書
label_mapping_inv = {v: k for k, v in label_mapping.items()}

# テキストをトークナイズ
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

test_dataset = Dataset.from_dict({'text': test_df['text'].tolist()})
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict(dataset):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = {k: v[i].unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.argmax(-1).item())
    return predictions

predictions = predict(tokenized_test_dataset)

# 予測されたラベルをカテゴリ名に逆変換
test_df['predicted_category'] = [label_mapping_inv[label] for label in predictions]
test_df.to_csv('path/to/predicted_test.csv', index=False)
end = time.time()
print(f"Elapsed time: {end - start} seconds")