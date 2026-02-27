import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import nltk
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data.csv")

# Setup tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
data['text'] = data['text'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data['text'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Prediction function
def predict(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Cyberbullying Detected"
    else:
        return "Normal Text"