# --------------------> Importing the required libraries --------------------------------

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

# --------------------> Instanciating the stemmer ---------------------------------------

wordnet_lemmatizer = WordNetLemmatizer()

# --------------------> Setting stopwords -----------------------------------------------

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# --------------------> Getting the positive reviews -----------------------------------

positive_review = BeautifulSoup(open('positive.review', 'r').read(), 'lxml')
positive_review = positive_review.find_all('review_text')

# --------------------> Getting the negative reviews ------------------------------------

negative_review = BeautifulSoup(open('negative.review', 'r').read(), 'lxml')
negative_review = negative_review.find_all('review_text')

# --------------------> Defining custom tokenizer ------------------------------------

def my_tokenizer(s: str):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

# --------------------> Making the Bag of words ---------------------------------------
    
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_review:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)     # Making list positive tokenized reviews
    
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1   # Updating the Bag of word model

for review in negative_review:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)      # Making list negative tokenized reviews
    
    for token in tokens: 
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1      # Updating the Bag of word model

# --------------------> Vectorizing the tokens ---------------------------------------

def tokens_to_vectorize(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
        
    x = x / x.sum()
    x[-1] = label
    return x

N = len(negative_tokenized) + len(positive_tokenized)

data = np.zeros((N, len(word_index_map) + 1))

i = 0

for tokens in positive_tokenized:
    xy = tokens_to_vectorize(tokens, 1)
    data[i, :] = xy
    i += 1
    
for tokens in negative_tokenized:
    xy = tokens_to_vectorize(tokens, 0)
    data[i, :] = xy
    i += 1

# ------------------------> Splitting the vectorized matrix ---------------------------
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ------------------------> Running the classification -------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

print('Classification Rate is', model.score(X_test, y_test))

# ------------------------> Checking the weights -------------------------------------

threshold = 0.5
for word, index in word_index_map.items():
    weights = model.coef_[0][index]
    
    if weights > threshold or weights < -threshold:
        print(word, weights)
