import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re

def preprocess_reviews(df):
    # Basic cleaning, can be extended
    df['clean_text'] = df['Review Text'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
    return df

def extract_features(df):
    tfidf = TfidfVectorizer(max_features=100)
    X = tfidf.fit_transform(df['clean_text']).toarray()
    return X, tfidf

def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf 