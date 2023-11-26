#linedistilbert_model_trainer_v100.py
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

    # logits がタプルの場合、最初の要素を使用
    if isinstance(logits, tuple):
        logits = logits[0]

    # logits の型と形状を確認
    print(type(logits))
    if isinstance(logits, np.ndarray):
        print(logits.shape)
    elif isinstance(logits, tuple):
        print("logits is a tuple, check its contents.")

    #二値分類の場合、logitsを[0, 1]に変換
    if logits.ndim == 1 or logits.shape[1] == 1:
        predictions = np.where(logits < 0.5, 0, 1)
    #多クラス分類の場合、argmaxを用いてラベルを取得
    else:
        predictions = np.argmax(logits, axis=-1)

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
# livedoor_news_dir = '../../../../livedoor_news/text'
# df = livedoor_news_to_df(livedoor_news_dir)
df = pd.read_csv('../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1126_v4.csv')
df['text'] = df['title'] + '。' + df['content']
df['text'] = df['text'].apply(clean_text_for_bert)
print(df)
print(df['text'])

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # 訓練データとテストデータのサイズを一時的に減らす
# train_df = train_df.sample(frac=0.01, random_state=42)
# test_df = test_df.sample(frac=0.01, random_state=42)

# データセットの作成
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})

# トークナイザーのロード
PRE_TRAINED = 'line-corporation/line-distilbert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED)

def tokenize_function(examples):
    tokenized_inputs =tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    tokenized_inputs = {key: tokenized_inputs[key] for key in ['input_ids', 'attention_mask']}
    return tokenized_inputs

# データセットのトークナイズ
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(le.classes_)).to(device)

training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500, # 学習率の下がりを待つステップ数
    lr_scheduler_type="linear", # 学習率の下がりを選択する方法
    learning_rate=3e-5,#2e-5: 85.5%, 6 epoch
    load_best_model_at_end=True,
    metric_for_best_model="loss", # どのmetricを改善の基準とするか
    greater_is_better=False, # metricが小さいほど良い場合はFalse
    remove_unused_columns=True,
    report_to='tensorboard',
)

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
output_dir = '../versions/v101/'
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
end = time.time()
print(f'Elapsed time={end-start}')