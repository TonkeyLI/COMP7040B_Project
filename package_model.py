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
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_tweet_text(self, tweet):
        tweet.lower()
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#','', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in self.stop_words]
        
        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
        
        return " ".join(filtered_words)
    def load_dataset(self, filename, cols):
        dataset = pd.read_csv(filename, encoding='latin-1')
        dataset.columns = cols
        return dataset

    def get_feature_vector(self, train_fit):
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector

    def train_test(self):
        # Load dataset
        dataset = self.load_dataset("train_data/training.csv", ['label', 'tweet'])
        #Preprocess data
        dataset.text = dataset['tweet'].apply(self.preprocess_tweet_text)
        print("data processed")
        # Split dataset into Train, Test

        # Same tf vector will be used for Testing sentiments on unseen trending data
        tf_vector = self.get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
        print("data to tf-idf processed")
        X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
        #print(X)
        y = np.array(dataset.iloc[:, 0]).ravel()
        #print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
        print("data spliited")

        # Training Naive Bayes model
        NB_model = MultinomialNB()
        NB_model.fit(X_train, y_train)
        y_predict_nb = NB_model.predict(X_test)
        print(accuracy_score(y_test, y_predict_nb))

        #Training Logistics Regression model
        LR_model = LogisticRegression(solver='lbfgs')
        LR_model.fit(X_train, y_train)
        y_predict_lr = LR_model.predict(X_test)
        print(accuracy_score(y_test, y_predict_lr))