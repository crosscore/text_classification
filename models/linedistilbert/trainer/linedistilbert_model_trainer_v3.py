#linedistilbert_model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertJapaneseTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import re
import time
import os
import torch

def clean_text_for_bert(text):
    text = text.strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def livedoor_news_to_df(root_dir):
    data = []
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if os.path.isdir(category_dir):
            for file in os.listdir(category_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(category_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        title = lines[2].strip()
                        content = ''.join(lines[3:]).strip()
                        if title and content:
                            data.append([category, title, content])
    return pd.DataFrame(data, columns=['category', 'title', 'content'])

start = time.time()
livedoor_news_dir = '../../../../livedoor_news/text'
df = livedoor_news_to_df(livedoor_news_dir)
#df = pd.read_csv('../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1125_v2.csv')
df['text'] = df['title'] + '。' + df['content']
df['text'] = df['text'].apply(clean_text_for_bert)
print(df)
print(df['text'])

# LabelEncoderを使用してラベルをエンコード
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(label_mapping)

# 訓練データとテストデータに分割
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# データセットの作成
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})

# トークナイザーのロード
PRE_TRAINED = 'line-corporation/line-distilbert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED)

# トークナイズ関数の定義
def tokenize_function(examples):
    tokenized_inputs =tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    # 必要なキーのみを保持する
    tokenized_inputs = {key: tokenized_inputs[key] for key in ['input_ids', 'attention_mask']}
    return tokenized_inputs

# データセットのトークナイズ
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# モデルの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(le.classes_)).to(device)

# 訓練設定
training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss", # どのmetricを改善の基準とするか
    greater_is_better=False, # metricが小さいほど良い場合はFalse
    remove_unused_columns=True
)
# トレーナーの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()
output_dir = '../versions/v3/'
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
end = time.time()
print(f'Elapsed time={end-start}')