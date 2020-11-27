# -*- coding: utf-8 -*-

import numpy as np

class MultinomialNaiveBayesNH:

    def __init__(self):
        self.trained = False
        self.n_words = 0
        self.prior = 0 # num of tweets of class c / total num of tweets
        self.count = 0 # total num of unique words of a class 
        self.occur_count = 0 #shape(num of unique words, num of class) num of occurrences of a word in class c
        self.NW = 0 # set of negation words
        self.word_index = 0 # {"word": index} a dic shows the indexes of words 
        self.test_data = 0

    def train(self, x, y, feat_dict):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        print ("x_count = %d, word_count = %d, num_classes = %d, y_count=%d" % (n_docs, n_words, n_classes, y.shape[0]))

        prior = np.zeros(n_classes) 
        count = np.zeros(n_classes)
        occur_count = np.zeros((n_words, n_classes))

        prior[0] = np.count_nonzero(y==0) / (float)(n_docs)   # negative class (y=0) = positive review
        prior[1] = np.count_nonzero(y==1) / (float)(n_docs) # positive class (y=1) = negative review

        for j in range(n_words):
            for i in range(n_docs):
                if x[i][j] > 0:
                    count[y[i][0]] += 1
                    occur_count[j][y[i][0]] += x[i][j]
        
        #load negation words
        lines = []
        with open("negative-words.txt", 'r') as f:
            lines = f.readlines()
            lines = [line[0:len(line)-2] for line in lines]

        self.trained = True
        self.n_words = n_words
        self.prior = prior
        self.count = count
        self.occur_count = occur_count
        self.NW = lines
        self.word_index = feat_dict

    def test(self, tweets):
        prediction = []
        for t in tweets:
            p_pos = self.calculate(t, 0)
            p_neg = self.calculate(t, 1)
            if p_pos >= p_neg:
                prediction.append(0)
            else:
                prediction.append(1)
        return prediction
    
    def calculate(self, tweet, c):
        p = 0
        for i in range(len(tweet)):
            if tweet[i] in self.word_index:
                f = self.occur_count[self.word_index[tweet[i]]][c]
                if i > 0 and tweet[i-1] in self.NW:
                    p -= (float(f)+1)/(float(self.count[c])+self.n_words)
                else:
                    p += (float(f)+1)/(float(self.count[c])+self.n_words)
        return p * self.prior(c)

    def evaluate(self, truth, predicted):
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total
            
