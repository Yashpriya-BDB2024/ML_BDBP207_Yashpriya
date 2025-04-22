### Implement Naive Bayes classifier for spam detection using scikit-learn library - use the dataset from https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset/data

"""
Different types of NB classifiers in Scikit-learn:

1. MultinomialNB: Best for longer text, SMS, articles, spam messages; uses how many times each word appears; best if using TF-IDF and when repeated words matter in your dataset.
2. BernoulliNB: Used when feature values are binary for short binary texts, email headers, etc.
3. GaussianNB: For non-text data with continuous features like sensor readings, biometric data, etc.
4. ComplementNB: Works well for imbalanced text classification (e.g. - many more hams than spams), often better than MultinomialNB.
5. CategoricalNB: For features that are categorical but not text.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def perform_eda(df):
    eda_results = {}
    eda_results['head'] = df.head()
    eda_results['missing_values'] = df.isnull().sum()
    df['message_length'] = df['v2'].apply(len)   # v2 is the text column (email message) in the dataset.
    eda_results['length_stats'] = df['message_length'].describe()   # Message length statistics
    eda_results['class_distribution'] = df['v1'].value_counts()   # v1 is the target column in the dataset.
    return df, eda_results

def preprocess_data(df, use_tfidf=True):
    # Label Encoding (bec. model needs numeric targets) -
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['v1'])   # v1 is the target column in the dataset.
    X = df['v2']
    return X, y

def split_and_vectorize(X, y, test_size=0.3, random_state=42):   # Split the dataset into 70% train set and 30% test set.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Feature Encoding -
    # One Hot Encoding is more useful for categorical tabular data, so will not be used here.
    # CountVectorizer: Converts text to a matrix of word counts. It is standard for NB.
    # TfidfVectorizer: Like count, but down-weights common words. It works well with NB.
    # Word2Vec: It gives dense vectors per word, not a single vector for the whole message, then we need to average them. Since, NB expects sparse matrix of features, so it's not suitable here.
    vectorizer = TfidfVectorizer(stop_words='english')   # to remove common words that don't carry significant meaning like 'the', 'and', etc.
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(X_train, y_train):   # Train the Naive Bayes model.
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    return accuracy, report

def main(filepath):
    df = load_data(filepath)
    df, _ = perform_eda(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, vectorizer = split_and_vectorize(X, y)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    print("Model: MultinomialNB with TF-IDF Vectorizer")
    print(f'Accuracy: {accuracy*100:.2f}%\n')
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main('spam_sms.csv')
