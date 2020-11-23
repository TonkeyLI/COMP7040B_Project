# -*- coding: utf-8 -*-

############### COMP7404 Assignment 3 ##################
'''
This program builds a naive bayes (NB) classifier to 
classify between positive and negative reviews on books.

The algorithm consists of these steps: 
(1) Initializing the prior and likelihood
(2) Computing the probability of prior = prob( doc | class )
(3) Computing the probability of likelihood = prob( word | class )
(4) Predicting values for training and testing data, and evaluating performance

The program also compares NB with two other algorithms, 
Support Vector Machine (SVM) and Neural Network (NN), which
are to be implemented using the scikit-learn package.

'''
# ACKNOWLEDGMENT (Type your full name and date here)
#
# My submission for this assignment is entirely my own original work done 
# exclusively for this assignment. I have not made and will not make my 
# solutions to assignment, homework, quizzes or exams available to anyone else.
# These include both solutions created by me and any official solutions provided 
# by the course instructor or TA. I have not and will not engage in any 
# activities that dishonestly improve my results or dishonestly improve/hurt
# the results of others.
#
# Your full name: __________
# Your student ID: __________
# Date: __________ 

import numpy as np

class MultinomialNaiveBayes:

    def __init__(self):
        self.trained = False
        self.prior = 0 # num of tweets of class c / total num of tweets
        self.count = 0 # total num of unique words of a class 
        self.occur_count = 0 #shape(num of unique words, num of class) num of occurrences of a word in class c
        self.NW = 0 # set of negation words
        self.word_index = 0 # {"word": index} a dic shows the indexes of words 

    def train(self, x, y, feat_dic):
        # to do 
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

        prior[0] = np.count_nonzero(y==0) / n_docs   # negative class (y=0) = positive review
        prior[1] = np.count_nonzero(y==1) / n_docs   # positive class (y=1) = negative review

        for j in range(n_words):
            contain_words = np.zeros(n_classes)
            for i in range(n_docs):
                if x[i][j] > 0:
                    contain_words[y[i][0]] = 1
                    occur_count[j][y[i][0]] += x[i][j]
            count += contain_words   
        
        self.trained = True
        self.prior = prior
        self.count = count
        self.occur_count = occur_count
        self.NW = 
        self.word_index = feat_dic

    def test(self, x, w):
        prediction = []
        for t in x:
            p_pos = calculate(t, 0)
            p_neg = calculate(t, 1)
            if p_pos >= p_neg:
                prediction.append(0)
            else:
                prediction.append(1)
        return prediction
    
    def calculate(self, tweet, c):
        p = 0
        for i in range(len(tweet)):
            f = self.occur_count[self.word_index[tweet[i]]]
            if i > 0 and tweet[i-1] in self.NW:
                p -= f/self.count[c]
            else:
                p += f/self.count[c]
        return p

    def evaluate(self, truth, predicted):
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total

