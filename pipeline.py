import pandas as pd
import re
import kagglehub
import os
import contractions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_kaggle():
    # Download dataset and read
    path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
    subdir = os.path.join(path, "News _dataset")
    
    fake_df = pd.read_csv(os.path.join(subdir, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(subdir, "True.csv"))
    
    # Add labels
    fake_df["label"] = 0   # 0 = Fake
    true_df["label"] = 1   # 1 = True
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    return df.copy()

def clean_text(text):
    text = text.lower()  # lowercase everything
    text = contractions.fix(text)  # expand contractions (ie couldn't --> could not)
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

def clean_kaggle(df):
    df = df.copy()
    # Clean text
    df["text"] = df["text"].apply(clean_text)
    
    # Deduplicate
    df = df.drop_duplicates(subset=["text"])
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def train_logreg(X_train, y_train, max_features=5000, ngram_range=(1, 2), max_iter=1000):
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer


def eval_on_dataset(model, vectorizer, X_test, y_test):
    # Vectorize test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, report