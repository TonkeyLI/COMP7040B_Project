import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class Package_model:
    def load_dataset(filename, cols):
        dataset = pd.read_csv(filename, encoding='latin-1')
        dataset.columns = cols
        return dataset

    def get_feature_vector(train_fit):
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector

    def train_test(self):
        # Load dataset
        dataset = load_dataset("./training.csv", ['target', 'text'])
        # Same tf vector will be used for Testing sentiments on unseen trending data
        tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
        #print(tf_vector)
        X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
        #print(X)
        y = np.array(dataset.iloc[:, 0]).ravel()
        #print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # Training Naive Bayes model
        NB_model = MultinomialNB()
        NB_model.fit(X_train, y_train)
        y_predict_nb = NB_model.predict(X_test)
        print("Accuracy using sklearn.naive_bayes: " + accuracy_score(y_test, y_predict_nb))

        # Training Logistics Regression model
        LR_model = LogisticRegression(solver='lbfgs')
        LR_model.fit(X_train, y_train)
        y_predict_lr = LR_model.predict(X_test)
        print("Accuracy using sklearn.linear_model: " + accuracy_score(y_test, y_predict_lr))