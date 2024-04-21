"""
TF-IDFを用いてテキストデータをベクトル化し、ナイーブベイズ分類器を用いて分類モデルを学習。

TF-IDFは、単語の出現頻度と文書内の単語の重要度を考慮してテキストをベクトル変換する手法。
"""

import pandas as pd
import glob
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import loguniform
import time

input_path = glob.glob("../../../../scraping-data/data/csv/yahoo_news/backup/*.csv")
df = pd.read_csv(input_path[0])
df = df.groupby("category").tail(1000).reset_index(drop=True)

start = time.time()
le = LabelEncoder()

# Convert 'category' column to numerical labels
encoded_labels = le.fit_transform(df["category"])
print(encoded_labels)

# Display the mapping between labels and their corresponding numbers
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Label mapping:", label_mapping)

texts = df["title"] + " " + df["content"]
print(df["category"].value_counts())

# Split the dataset into training and test sets
TEST_SIZE = 0.2
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=TEST_SIZE, random_state=42, stratify=encoded_labels
)
print(f"test_size={TEST_SIZE}")

# Create a TfidfVectorizer object
tfidf = TfidfVectorizer()

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(tfidf, MultinomialNB())

# Define the hyperparameter search space
parameters = {
    "multinomialnb__alpha": loguniform(1e-2, 1e2),
    "tfidfvectorizer__min_df": [1, 2, 3],
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
}

"""
CV"Cross-Validation"（交差検証): 機械学習モデルの性能評価と過学習防止に使用。
データを訓練用とテスト用に分割し、訓練用データでモデルを学習させ、テスト用データでモデルの性能を評価する。
単純にデータを一度だけ分割して評価すると、データの分割方法によって評価結果が大きく変わる可能性があるため、交差検証では、データを複数回に分けて分割し、それぞれの分割で訓練とテストを行い、その結果を平均化することで、より信頼性の高い評価を行う。

代表的な交差検証の手法は以下の通り。
1. k-fold交差検証（k-fold CV）: データをk個に分割し、k回の訓練とテストを行う。
2. StratifiedKFold交差検証: クラスの比率を維持しながらk-fold交差検証を行う。
3. Leave-one-out交差検証（LOOCV）: データ数をnとすると、n回の訓練とテストを行う。各回で1つのデータをテスト用に、残りのn-1個のデータを訓練用に使用。

`RandomizedSearchCV(cv=5)`と指定した場合、5-fold交差検証が使用され、データを5つに分割し、5回の訓練とテストを行い、その平均スコアを算出する。これを指定された回数（`n_iter`）だけ繰り返し、最適なハイパーパラメータを求める。

一般的には、RandomizedSearchCVで大まかな最適領域を見つけ、その後その領域内でGridSearchCVを行う
"""
grid = RandomizedSearchCV(
    pipeline, param_distributions=parameters, scoring="accuracy", cv=5, n_iter=50
)
grid.fit(train_texts, train_labels)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Build the model with the best hyperparameters
tfidf = TfidfVectorizer(
    min_df=grid.best_params_["tfidfvectorizer__min_df"],
    ngram_range=grid.best_params_["tfidfvectorizer__ngram_range"],
)

"""
多項分布ナイーブベイズ（Multinomial Naive Bayes)
学習フェーズで、各クラスにおける単語の出現頻度を数え、クラス条件付き確率を計算する。分類フェーズでは、新しい文書に含まれる単語の出現頻度を考慮して、各クラスに属する確率を計算し、最も確率の高いクラスを予測結果とする
"""
nb = MultinomialNB(alpha=grid.best_params_["multinomialnb__alpha"])

model = make_pipeline(tfidf, nb)

# Train the model using the best hyperparameters
model.fit(train_texts, train_labels)

# Evaluate the model on the test set
predicted = model.predict(test_texts)
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted)}")

# Save the trained model to a file
output_path = "./model/tfidf/yahoo_news_tfidf_naive_bayes_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import joblib

joblib.dump(model, output_path)

end = time.time()
print(f"Elapsed time={end-start}")
