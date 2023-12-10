#tfidf_model_trainer_v1.py
import pandas as pd
import os
import re
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

mecab = MeCab.Tagger('-d "C:/Program Files/MeCab/dic/ipadic" -u "C:/Program Files/MeCab/dic/NEologd/NEologd.dic"')
df = pd.read_csv('./csv/concat/yahoo_news_concat_1116.csv')

def extract_nouns_adjs(text):
    node = mecab.parseToNode(text)
    words = []
    while node:
        if node.feature.split(',')[0] in ['名詞', '形容詞']:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

# def extract_nouns_adjs(text):
#   stopwords = ['こと', 'もの', 'いう', 'とても', 'ある']
#   node = mecab.parseToNode(text)
#   words = []
#   while node:
#     if node.feature.split(',')[0] in ['名詞', '形容詞']:
#       if node.surface not in stopwords:
#         words.append(node.surface)
#     node = node.next
#   return " ".join(words)

def replace_digits(text):
    return re.sub(r'\d+', '', text)

print(df['category'].value_counts())

le = LabelEncoder()

# convert 'category' column to number
encoded_labels = le.fit_transform(df['category'])

texts = df['title'] + " " + df['content']

texts = list(map(extract_nouns_adjs, texts))

#Replace numbers in texts with 0
#texts = list(map(replace_digits, texts))
#print(texts)

# Split the dataset into training set and test set
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

#Remove n-gram. Generate 1-gram and 2-gram features. Set minimum document frequency with minor parameter min_df
tfidf = TfidfVectorizer(ngram_range=(1, 8), min_df=1)

# Create a pipeline of TF-IDF Vectorizer and Naive Bayes classifier
model = make_pipeline(tfidf, MultinomialNB())
#model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(train_texts, train_labels)

# Evaluate the model
predicted = model.predict(test_texts)
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted)}")

output_path = './model/tfidf/yahoo_news_tfidf_naive_bayes_model_v1.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import joblib
joblib.dump(model, output_path)
