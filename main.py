import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
data = pd.read_csv("Insert url for training data labeled fake_or_real_news.csv")
data["FAKE"] = data["label"].map({"REAL": 0, "FAKE": 1})
data = data.drop("label", axis=1)

# Clean text data
import re
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    return text

data['text'] = data['text'].apply(clean_text)


X, y = data['text'], data['FAKE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2), max_features=10000, min_df=5)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


clf = LinearSVC(C=0.5, class_weight='balanced')
clf.fit(X_train_vectorized, y_train)

y_pred = clf.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# mytext.txt is an example file given with a real article, you can predict your own by inserting .txt url here
with open("mytext.txt", "r") as f:
    text = f.read()
cleaned_text = clean_text(text)
vectorized_text = vectorizer.transform([cleaned_text])
print(clf.predict(vectorized_text))