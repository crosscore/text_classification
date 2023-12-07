#linedistilbert_model_trainer_v100.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
#from transformers import BertJapaneseTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import re
import time
import os
import torch
import glob

def clean_text(text):
    text = text.strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    print(type(logits))
    if isinstance(logits, np.ndarray):
        print(logits.shape)
    elif isinstance(logits, tuple):
        print("logits is a tuple, check its contents.")
    if logits.ndim == 1 or logits.shape[1] == 1:
        predictions = np.where(logits < 0.5, 0, 1)
    else:
        predictions = np.argmax(logits, axis=-1)
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

start = time.time()
read_file = glob.glob('../../../data/scraping_data/csv/yahoo_news/concat/*.csv')
df = pd.read_csv(read_file[0])
df['text'] = df['title'].apply(clean_text) + '。' + df['content'].apply(clean_text)
#df['url']に'/pickup/'が含まれる行を削除
df = df[~df['url'].str.contains('/pickup/')]
print(f"'url'列に'/pickup/'の含まれる行数: {df[df['url'].str.contains('/pickup/')]}")
print(df['category'].value_counts(dropna=False))
print(df['text'])

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})
PRE_TRAINED = 'line-corporation/line-distilbert-base-japanese'
#tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED, trust_remote_code=True)

def tokenize_function(examples):
    tokenized_inputs =tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    tokenized_inputs = {key: tokenized_inputs[key] for key in ['input_ids', 'attention_mask']}
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(le.classes_)).to(device)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED, num_labels=len(le.classes_)).to(device)

print(f"Class of tokenizer used: {tokenizer.__class__.__name__}")
print(f"Class of model used: {model.__class__.__name__}")

training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
output_dir = '../versions/v102/'
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
print('The model has been saved.')

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
end = time.time()
print(f'Elapsed time={end-start}')