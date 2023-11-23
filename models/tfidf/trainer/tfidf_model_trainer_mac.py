import pandas as pd
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import Parallel, delayed
import time

input_path = '../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1123.csv'
nlp = spacy.load('ja_core_news_sm', disable=['ner', 'lemmatizer'])
df = pd.read_csv(input_path)

def extract_nouns_adjs(text):
    words = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ']:
            words.append(token.text)
    return " ".join(words)

start = time.time()

# ラベルエンコーダーの初期化
le = LabelEncoder()

# 'category' 列を数値に変換
encoded_labels = le.fit_transform(df['category'])
print(encoded_labels)

# ラベルの対応関係を表示
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Label mapping:", label_mapping)

# 並列処理を行わずに実行
texts = df['title'] + " " + df['content']
print(texts)
texts = texts.apply(extract_nouns_adjs)
end1 = time.time()
print(texts)
print(f'time={end1 - start}')

# データセットをトレーニングセットとテストセットに分割
TEST_SIZE = 0.2
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=TEST_SIZE, random_state=42, stratify=encoded_labels)
print(f'test_size={TEST_SIZE}')

#n-gramを除去。1-gramと2-gramの特徴量を生成。マイナーパラメータmin_dfで最小ドキュメント頻度を設定
tfidf = TfidfVectorizer()

# pipelineにGridSearchCVを追加してハイパーパラメータチューニング
pipeline = make_pipeline(tfidf, MultinomialNB())

parameters = {
    'multinomialnb__alpha': [0.005],
    'tfidfvectorizer__min_df': [1],
    'tfidfvectorizer__ngram_range': [(1, 4)]
}

grid = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', cv=5)
grid.fit(train_texts, train_labels)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

#ベストパラメータでmodelを構築
tfidf = TfidfVectorizer(min_df=grid.best_params_['tfidfvectorizer__min_df'], ngram_range=grid.best_params_['tfidfvectorizer__ngram_range'])
nb = MultinomialNB(alpha=grid.best_params_['multinomialnb__alpha'])
model = make_pipeline(tfidf, nb)

# モデルのトレーニング
model.fit(train_texts, train_labels)

# モデルの評価
predicted = model.predict(test_texts)
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted)}")

# モデルの保存
output_path = '../versions/mac/yahoo_news_tfidf_model.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import joblib
joblib.dump(model, output_path)

end2 = time.time()
print(f'Elapsed time={end2 - start}')
