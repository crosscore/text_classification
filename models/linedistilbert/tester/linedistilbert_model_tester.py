#linedistilbert_model_test.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertJapaneseTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction
from datasets import Dataset
import time

# Calculate correct answer rate
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0].argmax(-1) if isinstance(p.predictions, tuple) else p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

start = time.time()
df = pd.read_csv('../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1122_v2.csv')
df['text'] = df['title'] + ' ' + df['content']

# Label mapping
unique_categories = df['category'].unique()
label_mapping = {category: idx for idx, category in enumerate(unique_categories)}
print(label_mapping)
df['label'] = df['category'].map(label_mapping)

# Prepare test data
_, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_dataset = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})
print(test_dataset)

# Load tokenizer
PRE_TRAINED = 'line-corporation/line-distillbert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_TRAINED, trust_remote_code=True)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Tokenized test dataset
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# load model
model = DistilBertForSequenceClassification.from_pretrained('../versions/v2/', num_labels=len(unique_categories))

# Trainer settings
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Evaluate with test data and output correct answer rate
test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result['eval_accuracy'])
end = time.time()
print(f"Time elapsed: {end - start} seconds.")
