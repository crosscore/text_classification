import pandas as pd
import glob
import os
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import time

input_path = glob.glob("../../../../scraping-data/data/csv/yahoo_news/backup/*.csv")
df = pd.read_csv(input_path[0])
df = df.groupby("category").tail(2000).reset_index(drop=True)
nlp = spacy.load("ja_ginza_electra")

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
    "multinomialnb__alpha": [0.1],
    "tfidfvectorizer__min_df": [1],
    "tfidfvectorizer__ngram_range": [(1, 2)],
}

# Perform grid search to find the best hyperparameters
grid = GridSearchCV(pipeline, param_grid=parameters, scoring="accuracy", cv=5)
grid.fit(train_texts, train_labels)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Build the model with the best hyperparameters
tfidf = TfidfVectorizer(
    min_df=grid.best_params_["tfidfvectorizer__min_df"],
    ngram_range=grid.best_params_["tfidfvectorizer__ngram_range"],
)
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
