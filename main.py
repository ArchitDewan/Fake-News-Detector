import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC   
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Insert csv file URL")
# Changes all the given data "FAKE" and "REAL" to binary values
data["FAKE"] = data["label"].map({"REAL": 0, "FAKE": 1})
print(data)

data = data.drop("label", axis =1)
X,y = data['text'],data['FAKE']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df = 0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)


with open("mytext.txt", "r") as f:
    text = f.read()
vectorized_text = vectorizer.transform([text])
print(clf.predict(vectorized_text))


#TF-IDF is used to calculate the most useful terms compared to other articles