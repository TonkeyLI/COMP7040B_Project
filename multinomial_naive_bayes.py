# -*- coding: utf-8 -*-

import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1
        
    def train(self, x, y):
        n_docs, n_words = x.shape
        classes = np.unique(y)
        n_classes = np.unique(y).shape[0]
        
        prior = np.array([ 0.0 for i in range(n_classes)])
        likelihood = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])
        #likelihood = np.array([[ 0.0 for i in range(n_classes)] for j in range(n_words)])
        
        print ("x_count = %d, word_count = %d, num_classes = %d, y_count=%d" % (
            n_docs, n_words, n_classes, y.shape[0]))

        prior[0] = np.count_nonzero(y==0) / float(n_docs)   
        prior[1] = np.count_nonzero(y==1) / float(n_docs)   

        # n = np.zeros(n_classes)     
        # nk = np.zeros((n_words, n_classes))    
        # for j in range(n_words):
        #     for i in range(n_docs):
        #         nk[j][y[i]] += x[i][j]
        #         n[y[i]] += x[i][j]
        # for j in range(n_words):
        #     for m in range(n_classes):  
        #         ## update likelihood for each class (approx. 1 line)
        #         likelihood[j][m] = (float)(nk[j][m]+1)/(float)(n[m]+n_words)
        frequency = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])
        for i in range(n_docs):
            if y[i] == 0:
                frequency[0] += x[i]
            elif y[i] == 1:
                frequency[1] += x[i]

        for j in range(n_classes):
            likelihood[j] = (frequency[j] + 1) / (np.sum(frequency[j]) + n_words + 1)

        likelihood = likelihood.T

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
