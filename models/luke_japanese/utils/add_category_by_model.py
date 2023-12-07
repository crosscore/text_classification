import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import glob
import time

def clean_text(text):
    text = text.strip()
    return text

start = time.time()
PRE_TRAINED = '../versions/v101/' #0.918
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED)

# テストデータの読み込み
test_data_path = '../../../data/experimental_data/csv/complete/device_original_with_category.csv'
test_df = pd.read_csv(test_data_path, dtype={'user': str})
print(test_df.head())
test_df['text'] = test_df['title'].apply(clean_text) + '。' + test_df['content'].apply(clean_text)
print(test_df.head())

# 訓練データからラベルエンコーダのマッピングを作成
train_data_path = glob.glob('../../../data/scraping_data/csv/yahoo_news/concat/*.csv')
df = pd.read_csv(train_data_path[0], dtype={'user': str})
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
#{'IT': 0, 'エンタメ': 1, 'スポーツ': 2, 'ライフ': 3, '国内': 4, '国際': 5, '地域': 6, '科学': 7, '経済': 8}
print(label_mapping)

# ラベルマッピングの逆変換用辞書
label_mapping_inv = {v: k for k, v in label_mapping.items()}

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
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            # トークン化されたデータを取得
            inputs = dataset[i]
            # データをテンソルに変換し、適切なデバイスに配置
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.argmax(-1).item())
    return predictions

predictions = predict(tokenized_test_dataset)

# 予測されたラベルをカテゴリ名に逆変換
test_df['predicted_category'] = [label_mapping_inv[label] for label in predictions]
test_df.to_csv('../../../data/experimental_data/csv/complete/device_original_with_category_plus_predict.csv', index=False)
end = time.time()
print(f"Elapsed time: {end - start} seconds")