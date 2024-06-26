# luke_ja_base_lite.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
# from transformers import MLukeTokenizer, LukeForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import re
import time
import os
import torch
import glob
import random

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def clean_text(text):
    text = text.strip()
    return text

"""
compute_metrics関数の引数eval_predは、モデルの予測結果（logits）と正解ラベル（labels）のタプル。

一般的に、シーケンス分類タスクにおけるモデルの出力（logits）は、以下の2つの形式のいずれかを取る：
1. (batch_size, num_classes) の形状を持つ2次元テンソル：各サンプルについて、各クラスに属する確率または確信度を表す。この場合、予測値は各サンプルについて最大の確率または確信度を持つクラスのインデックスとなる。

2. (batch_size, 1) の形状を持つ2次元テンソルまたは (batch_size,) の形状を持つ1次元テンソル：各サンプルについて、特定のクラスに属する確率や確信度を表す。この場合、予測値は確率や確信度がしきい値（ここでは0.5）を超えるかどうかによって決定される。
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Calculate predicted values based on the shape of logits
    if logits.ndim == 1 or logits.shape[1] == 1:
        predictions = np.where(logits < 0.5, 0, 1)
    else:
        predictions = np.argmax(logits, axis=-1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

start = time.time()
read_file = glob.glob(
    "../../../../scraping-data/data/csv/yahoo_news/backup/*.csv"
)
print(f"read_file[0]: {read_file[0]}")
df = pd.read_csv(read_file[0], dtype={'user': str})

# Remove lines containing the string '/pickup/' in the url string
df = df[~df["url"].str.contains("/pickup/")]
df = df.groupby("category").tail(1000).reset_index(drop=True)

df['text'] = df['title'].apply(clean_text) + '。' + df['content'].apply(clean_text)

min_category_num = df['category'].value_counts().min()
df = df.groupby('category').apply(lambda x: x.sample(min(len(x), min_category_num), random_state=SEED)).reset_index(drop=True)
print(df['category'].value_counts(dropna=False))
print(df['text'])

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})
PRE_TRAINED = 'studio-ousia/luke-japanese-base-lite'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED, trust_remote_code=True)

"""
examples['text'] :データセットの各サンプルのテキストデータ。
padding=True: バッチ内の各サンプルの長さを統一するためにパディング"[PAD]"を行う。ex. [1,1,1,0,0]
truncation=True: テキストの長さがmax_lengthを超える場合、テキストを切り捨て。
max_length=512: トークン化後のテキストの最大長を指定。モデルの入力サイズが固定される。
"""
def tokenize_function(examples):
    tokenized_inputs =tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    tokenized_inputs = {key: tokenized_inputs[key] for key in ['input_ids', 'attention_mask']}
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(le.classes_)).to(device)

print(f"Class of tokenizer used: {tokenizer.__class__.__name__}")
print(f"Class of model used: {model.__class__.__name__}")

"""
metric_for_best_model: 最良のモデルを選択するための評価指標。loss: モデルの予測値と正解値の差を定量化
greater_is_better: loss(損失)は小さいほど良いモデルと考えられるためFalse
"""
training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #warmup_steps=500,
    #lr_scheduler_type="linear",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
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

output_dir = '../versions/lite/v101/'
os.makedirs(output_dir, exist_ok=True)

trainer.save_model(output_dir)
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])

end = time.time()
print(f'Elapsed time={end-start}')
