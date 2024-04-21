import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import re
import time
import os
import torch
import glob
import random
import optuna
from torch.utils.tensorboard import SummaryWriter

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def clean_text(text):
    text = text.strip()
    return text


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim == 1 or logits.shape[1] == 1:
        predictions = np.where(logits < 0.5, 0, 1)
    else:
        predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 10)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32, 64]
    )
    warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)

    training_args = TrainingArguments(
        output_dir="./result",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    return eval_metrics["eval_loss"]


start = time.time()
read_file = glob.glob("../../../../scraping-data/data/csv/yahoo_news/backup/*.csv")
print(f"read_file[0]: {read_file[0]}")
df = pd.read_csv(read_file[0], dtype={"user": str})

df = df[~df["url"].str.contains("/pickup/")]
df = df.groupby("category").tail(1000).reset_index(drop=True)

df["text"] = df["title"].apply(clean_text) + "。" + df["content"].apply(clean_text)

min_category_num = df["category"].value_counts().min()
df = (
    df.groupby("category")
    .apply(lambda x: x.sample(min(len(x), min_category_num), random_state=SEED))
    .reset_index(drop=True)
)
print(df["category"].value_counts(dropna=False))
print(df["text"])

le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(label_mapping)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict(
    {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()}
)
test_dataset = Dataset.from_dict(
    {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}
)

PRE_TRAINED = "studio-ousia/luke-japanese-base-lite"
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED, trust_remote_code=True)


def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], padding=True, truncation=True, max_length=512
    )
    tokenized_inputs = {
        key: tokenized_inputs[key] for key in ["input_ids", "attention_mask"]
    }
    return tokenized_inputs


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    PRE_TRAINED, num_labels=len(le.classes_)
).to(device)

print(f"Class of tokenizer used: {tokenizer.__class__.__name__}")
print(f"Class of model used: {model.__class__.__name__}")

writer = SummaryWriter()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    writer.add_text(f"Best/{key}", str(value))

output_dir = "../versions/lite/v101/"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("The model has been saved.")

test_result = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print("Accuracy:", test_result["eval_accuracy"])

for key, value in test_result.items():
    writer.add_scalar(f"Test/{key}", value)

writer.close()

end = time.time()
print(f"Elapsed time={end-start}")
