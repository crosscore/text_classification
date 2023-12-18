#tfidf_model_trainer_v2.py
import pandas as pd
import os
import re
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import time

input_path = '../../../data/scraping_data/csv/yahoo_news/concat/yahoo_news_concat_1123.csv'
print(input_path)
mecab = MeCab.Tagger('-d "C:/Program Files/MeCab/dic/ipadic" -u "C:/Program Files/MeCab/dic/NEologd/NEologd.dic"')
df = pd.read_csv(input_path)

def extract_nouns_adjs(text):
    node = mecab.parseToNode(text)
    words = []
    while node:
        if node.feature.split(',')[0] in ['名詞', '形容詞']:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

def text_normalization(text):
    text = re.sub(r'[０-９]', lambda x: chr(ord(x.group(0)) - 0xFEE0), text)
    # Remove unnecessary symbols
    text = re.sub(r'[^\w\s]', '', text)
    return text

def replace_digits(text):
    return re.sub(r'\d+', '0', text)

start = time.time()
le = LabelEncoder()

# convert 'category' column to number
encoded_labels = le.fit_transform(df['category'])
print(encoded_labels)

# Show label correspondence
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Label mapping:", label_mapping)

texts = df['title'] + " " + df['content']
Convert #texts to a string of nouns and adjectives only
texts = texts.apply(extract_nouns_adjs)

# Split the dataset into training set and test set
TEST_SIZE = 0.1
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=TEST_SIZE, random_state=42, stratify=encoded_labels)
print(f'test_size={TEST_SIZE}')

#Remove n-gram. Generate 1-gram and 2-gram features. Set minimum document frequency with minor parameter min_df
tfidf = TfidfVectorizer()

# Add GridSearchCV to the pipeline and tune hyperparameters
pipeline = make_pipeline(tfidf, MultinomialNB())

parameters = {
    'multinomialnb__alpha': [0.005],
    'tfidfvectorizer__min_df': [1],
    'tfidfvectorizer__ngram_range': [(1, 4)]
}

default_parameters = {
    'multinomialnb__alpha': [1.0],
    'tfidfvectorizer__min_df': [1],
    'tfidfvectorizer__ngram_range': [(1, 1)]
}

grid = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', cv=5)
grid.fit(train_texts, train_labels)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

#Build model with best parameters
tfidf = TfidfVectorizer(min_df=grid.best_params_['tfidfvectorizer__min_df'], ngram_range=grid.best_params_['tfidfvectorizer__ngram_range'])
nb = MultinomialNB(alpha=grid.best_params_['multinomialnb__alpha'])
model = make_pipeline(tfidf, nb)

# Train the model
model.fit(train_texts, train_labels)

# Evaluate the model
predicted = model.predict(test_texts)
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted)}")

output_path = '../versions/v2/yahoo_news_tfidf_model.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import joblib
joblib.dump(model, output_path)

end = time.time()
print(f'Elapsed time={end-start}')
