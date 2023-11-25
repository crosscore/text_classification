#bert_livedoor_news_llc.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import random
import time
import glob
import os

start = time.time()

def set_seed(seed_value=369):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# 乱数シードの設定
set_seed(369)

# デバイスの設定
device = get_device()
print(f"Using {device}...")

# カテゴリー数と内容を確認
livedoor_news_path = "../../../livedoor_news/text/"
files_folders = [name for name in os.listdir(livedoor_news_path)]
print(files_folders)
categories = [name for name in os.listdir(
    livedoor_news_path) if os.path.isdir(livedoor_news_path + name)]
print("カテゴリー数:", len(categories))
print(categories)

# 本文を取得する前処理関数を定義
def extract_main_txt(file_name):
    with open(file_name, encoding='utf-8') as text_file:
        text = text_file.readlines()[3:]
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        text = ''.join(text)
        text = text.translate(str.maketrans(
            {'\n': '', '\t': '', '\r': '', '\u3000': ''}))
        return text

# リストに前処理した本文と、カテゴリーのラベルを追加していく
list_text = []
list_label = []
for cat in categories:
    text_files = glob.glob(os.path.join(livedoor_news_path, cat, "*.txt"))
    # 前処理extract_main_txtを実施して本文を取得
    body = [extract_main_txt(text_file) for text_file in text_files]
    label = [cat] * len(body)  # body数分だけカテゴリー名のラベルのリストを作成
    list_text.extend(body)  # appendが要素を追加するのに対して、extendはリストごと追加する
    list_label.extend(label)

df = pd.DataFrame({'text': list_text, 'label': list_label})
print(df)
print(df.shape)

# カテゴリーの辞書を作成
dic_id2cat = dict(zip(list(range(len(categories))), categories))
dic_cat2id = dict(zip(categories, list(range(len(categories)))))
print(dic_id2cat)
print(dic_cat2id)

# DataFrameにカテゴリーindexの列を作成
df["label_index"] = df["label"].map(dic_cat2id)

# カスタム Dataset クラスの定義
class LivedoorDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, label_dict):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dict = label_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx, 0]  # テキストデータ
        label_str = self.dataframe.iloc[idx, 1]  # ラベル（文字列）
        label = self.label_dict[label_str]  # ラベルを整数に変換
        # トークナイザーを使用してテキストをトークン化
        encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# トークナイザーの準備
tokenizer = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    mecab_kwargs={"mecab_dic": "ipadic", "mecab_option": None}
)

# データセットの分割
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=369)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=369)
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# データセットの作成
max_length = 512
train_dataset = LivedoorDataset(train_df, tokenizer, max_length, dic_cat2id)
val_dataset = LivedoorDataset(val_df, tokenizer, max_length, dic_cat2id)
test_dataset = LivedoorDataset(test_df, tokenizer, max_length, dic_cat2id)

# DataLoaderの使用
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデルの定義
model = BertForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    num_labels=len(categories)  # カテゴリの数に基づいて出力層を設定
)
model.to(device)

# オプティマイザと損失関数の設定
optimizer = AdamW(model.parameters(), lr=2e-5)#2e-5:93%

# 訓練関数の定義
def train_model(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

# 評価関数の定義
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()
            preds = outputs[1].argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 各種スコアの計算
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    return total_loss / len(dataloader), accuracy, precision, recall, f1

# 検証データセットの準備（訓練データの一部を検証用に分割）
val_df = train_df.sample(frac=0.1, random_state=369)  #10%を検証データセットとする
train_df = train_df.drop(val_df.index)
val_dataset = LivedoorDataset(val_df, tokenizer, max_length, dic_cat2id)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 早期停止クラスの定義
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        patience: 性能が改善されないエポックの最大数
        min_delta: 改善と見なされる最小の変化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=1, min_delta=0.01)

# 訓練プロセス（早期停止を組み込む）
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    train_loss = train_model(model, train_loader, optimizer)
    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader)
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1 Score: {val_f1}")
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("Early stopping")
        break

# 評価プロセス
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")

end = time.time()
print(f"Elapsed time: {end -start}")