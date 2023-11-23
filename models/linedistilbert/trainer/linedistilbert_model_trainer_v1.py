#linedistilbert_model_trainer.py
import pandas as pd
import MeCab
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertJapaneseTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import re
import time
import os
import sys

mecab = MeCab.Tagger('-d "C:/Program Files/MeCab/dic/ipadic" -u "C:/Program Files/MeCab/dic/NEologd/NEologd.dic"')

def extract_nouns_adjs(text):
    node = mecab.parseToNode(text)
    words = []
    while node:
        if node.feature.split(',')[0] in ['名詞', '形容詞']:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    return text

start = time.time()

df = pd.read_csv('./csv/concat/yahoo_news_concat_1121.csv')

df['text'] = df['title'] + ' ' + df['content']
#df['text'] = df['text'].apply(clean_tag)
df['text'] = df['text'].apply(extract_nouns_adjs)
print(df['text'])

# ラベルのマッピング
unique_categories = df['category'].unique()
label_mapping = {category: idx for idx, category in enumerate(unique_categories)}
print(label_mapping)
df['label'] = df['category'].map(label_mapping)

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
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# データセットのトークナイズ
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# モデルの設定
model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(unique_categories))

# 訓練設定
training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    remove_unused_columns=True
)
# トレーナーの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('./model/linedistilbert/nouns')
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])

end = time.time()
print(f'Elapsed time={end-start}')