# -*- coding: utf-8 -*-
import math
import numpy as np
import json

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
        self.likelihood = 0

    def train(self, x, y, feat_dict):
        n_docs, n_words = x.shape
        classes = np.unique(y)
        n_classes = np.unique(y).shape[0]

        print ("x_count = %d, word_count = %d, num_classes = %d, y_count=%d" % (n_docs, n_words, n_classes, y.shape[0]))
        
        prior = [0, 0]
        prior[0] = np.count_nonzero(y==0) / float(n_docs)   
        prior[1] = np.count_nonzero(y==1) / float(n_docs) 

        occur_count = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])
        #likelihood = np.array([[ 0.0 for i in range(n_words)] for j in range(n_classes)])
        for i in range(n_docs):
            if y[i] == 0:
                occur_count[0] += x[i]
            elif y[i] == 1:
                occur_count[1] += x[i]

        count = [0, 0]
        for j in range(n_classes):
            count[j] = np.sum(occur_count[j])

        # for j in range(n_classes):
        #     likelihood[j] = (occur_count[j]+1) / float(count[j])
        
        occur_count = occur_count.T
        #likelihood = likelihood.T

        #load negation words
        lines = []
        with open("train_data/negation_words.txt", 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        self.trained = True
        self.n_words = n_words
        self.prior = prior
        self.count = count
        self.occur_count = occur_count
        self.NW = lines
        self.word_index = feat_dict
        #self.likelihood = likelihood

    def test(self, tweets):
        prediction = []
        for t in tweets:
            p_pos = self.calculate(t, 0)   
            p_neg = self.calculate(t, 1)
            if p_pos > p_neg:
                prediction.append(0)
            else:
                prediction.append(1)
        return prediction
    
    # def calculate(self, tweet, c):
    #     p_total=[]
    #     p = 0
    #     for i in range(len(tweet)):
    #         if tweet[i] in self.word_index:
    #             if i == 0:
    #                 p = self.likelihood[self.word_index[tweet[i]]][c]
    #             elif i > 0 and tweet[i-1] in self.NW:
    #                 p = 1 - self.likelihood[self.word_index[tweet[i]]][c]
    #                #p = self.likelihood[self.word_index[tweet[i]]][1-c]
    #                 #f = self.occur_count[self.word_index[tweet[i]]][1-c]
    #                 #p = (float(f)+1)/(float(self.count[1-c])+self.n_words)
    #                 #p = float(f)/self.count[1-c]
    #             else:
    #                 p = self.likelihood[self.word_index[tweet[i]]][c]
    #                 #f = self.occur_count[self.word_index[tweet[i]]][c]
    #                 #p = (float(f)+1)/(float(self.count[c])+self.n_words)
    #                 #p = float(f)/self.count[c]
    #             p_total.append(p)
    #         # else:
    #         #     if i > 0 and tweet[i-1] in self.NW:
    #         #         p = 1.0/self.count[1-c]
    #         #     else:
    #         #         p = 1.0/self.count[c]
    #         #     p_total.append(math.log(p))
    #     p = 1
    #     for i in p_total:
    #         p = p*i
    #     return p * self.prior[c]
    
    def calculate(self, tweet, c):
        p_total = math.log(self.prior[c])
        p = 0
        for i in range(len(tweet)):
            word_count = self.find_word_count(tweet[i], c)
            if tweet[i] in self.NW:
                continue
            if i > 0 and tweet[i-1] in self.NW:
                p = float(word_count)/self.count[c]
                p_total -= math.log(p)
            else:
                p = float(word_count)/self.count[c]
                p_total += math.log(p)
        return p_total 

    def find_word_count(self, word, c):
        result = 0
        if word in self.word_index:
            result = self.occur_count[self.word_index[word]][c]
        return result+1

    def evaluate(self, truth, predicted):
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total

    def save_model(self):
        model = {
            'n_words': self.n_words,
            'prior': self.prior,
            'count': self.count,
            'word_index': self.word_index,
            'NW': self.NW
        }

        #save model except occur_count
        with open('model/model.json', 'w') as f:
            json.dump(model, f)
        #save occur_count
        np. savetxt('model/model.csv', self.occur_count, delimiter=',')
            
