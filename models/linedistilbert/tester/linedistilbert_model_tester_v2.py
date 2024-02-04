import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertJapaneseTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
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

df = pd.read_csv('../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1126_v2.csv')
df['text'] = df['title'] + '。' + df['content']
df['text'] = df['text'].apply(clean_text_for_bert)

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
print(df)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})

PRE_TRAINED = 'line-corporation/line-distilbert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

model_path = '../versions/v3/'
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(le.classes_))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

training_args = TrainingArguments(
    output_dir='./result',
    per_device_eval_batch_size=64
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
