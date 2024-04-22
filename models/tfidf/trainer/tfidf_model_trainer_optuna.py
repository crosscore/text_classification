import pandas as pd
import glob
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import optuna

input_path = glob.glob("../../../../scraping-data/data/csv/yahoo_news/backup/*.csv")
df = pd.read_csv(input_path[0])
df = df.groupby("category").tail(2000).reset_index(drop=True)

start = time.time()
le = LabelEncoder()
encoded_labels = le.fit_transform(df["category"])
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

texts = df["title"] + "。 " + df["content"]

TEST_SIZE = 0.2
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=TEST_SIZE, random_state=42, stratify=encoded_labels
)


def objective(trial):
    # Define the hyperparameter search space
    min_df = trial.suggest_int("min_df", 1, 5)
    ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
    alpha = trial.suggest_loguniform("alpha", 1e-2, 1e2)

    # Create a TfidfVectorizer object with the sampled hyperparameters
    tfidf = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

    # Create a pipeline with TfidfVectorizer and MultinomialNB
    model = make_pipeline(tfidf, MultinomialNB(alpha=alpha))

    # Train the model and compute the accuracy score
    model.fit(train_texts, train_labels)
    predicted = model.predict(test_texts)
    accuracy = metrics.accuracy_score(test_labels, predicted)

    return accuracy


# Create an Optuna study object
study = optuna.create_study(direction="maximize")

# Optimize the hyperparameters using Optuna
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)

# Train the model with the best hyperparameters
best_params = study.best_params
tfidf = TfidfVectorizer(
    min_df=best_params["min_df"], ngram_range=best_params["ngram_range"]
)
nb = MultinomialNB(alpha=best_params["alpha"])
model = make_pipeline(tfidf, nb)
model.fit(train_texts, train_labels)

# Evaluate the model on the test set
predicted = model.predict(test_texts)
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted)}")

# Save the trained model to a file
output_path = "./model/tfidf/yahoo_news_tfidf_naive_bayes_model_optuna.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import joblib

joblib.dump(model, output_path)

end = time.time()
print(f"Elapsed time={end-start}")
