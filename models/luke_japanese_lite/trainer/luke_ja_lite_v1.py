# luke_ja_base_v1.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import MLukeTokenizer, LukeForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import re
import time

def clean_text_for_bert(text):
    # 連続するスペース、タブ、改行をスペースに置換
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logitsがタプルの場合、その中身を確認
    if isinstance(logits, tuple):
        print("Logits is a tuple, contents:", logits)
        # 予測スコアが含まれている部分を選択
        logits = logits[0]

    print("Logits shape:", logits.shape)
    try:
        predictions = np.argmax(logits, axis=-1)
    except Exception as e:
        print("Error in argmax:", e)
        print("Logits:", logits)
        raise
    return {'accuracy': accuracy_score(labels, predictions)}

start = time.time()
df = pd.read_csv('../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1123_v3.csv')

df['text'] = df['title'] + '。' + df['content']
#df['text'] = df['text'].apply(clean_text_for_bert)
print(df)
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
PRE_TRAINED = 'studio-ousia/luke-japanese-base-lite'
tokenizer = MLukeTokenizer.from_pretrained(PRE_TRAINED)

# トークナイズ関数の定義
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    # 必要なキーのみを保持する
    tokenized_inputs = {key: tokenized_inputs[key] for key in ['input_ids', 'attention_mask']}
    return tokenized_inputs

# データセットのトークナイズ
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# モデルの設定
model = LukeForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(unique_categories))

# 訓練設定
training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
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

print('### Start training ###')
trainer.train()
output_dir = '../versions/v1/'
trainer.save_model(output_dir)
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
end = time.time()
print(f'Elapsed time={end-start}')
