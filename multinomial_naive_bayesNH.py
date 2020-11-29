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
        self.prositive_words = 0
        self.negative_words = 0

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
        negation_words = []
        with open("train_data/negation_words.txt", 'r') as f:
            negation_words = f.readlines()
            negation_words = [line.strip() for line in negation_words]
        negative_words = []
        with open("train_data/negative_words.txt", 'r') as f:
            negative_words = f.readlines()
            negative_words = [line.strip() for line in negative_words]
        positive_words = []
        with open("train_data/positive_words.txt", 'r') as f:
            positive_words = f.readlines()
            positive_words = [line.strip() for line in positive_words]

        # self.weight = [1.0 for i in range(n_words)]
        # for key in feat_dict.keys():
        #     if key in positive_words or key in negative_words:
        #         self.weight[feat_dict[key]] *= 100

        # self.learning_rate = 0.5
        # print(self.weight)
        self.trained = True
        self.n_words = n_words
        self.prior = prior
        self.count = count
        self.occur_count = occur_count
        self.NW = negation_words
        self.word_index = feat_dict
        self.positive_words = positive_words
        self.negative_words = negative_words
        #self.likelihood = likelihood

    def test_train(self, tweets, truth):
        prediction = []
        #support_count = 0
        for i in range(len(tweets)):
            result = 0
            p_pos = self.calculate(tweets[i], 0)   
            p_neg = self.calculate(tweets[i], 1)
            if p_pos > p_neg:
                result = 0
            else:
                result = 1

            if result == truth[i]:
                self.increse_weight(tweets[i])
            else:
                self.decrese_weight(tweets[i])
            #prediction.append(result)
        #print("support count:", support_count)
        #return prediction
        # for i in self.weight:
        #     print(i)

    def test(self, tweets):
        prediction = []
        for i in range(len(tweets)):
            result = 0
            p_pos = self.calculate(tweets[i], 0)   
            p_neg = self.calculate(tweets[i], 1)
            if p_pos > p_neg:
                result = 0
            else:
                result = 1
            prediction.append(result)
        return prediction

    def test_predict(self, tweets):
        prediction = []
        for i in range(len(tweets)):
            result = 0
            p_pos = self.calculate_predict(tweets[i], 0)   
            p_neg = self.calculate_predict(tweets[i], 1)
            if p_pos > p_neg:
                result = 0
            else:
                result = 1
            prediction.append(result)
        return prediction

    def increse_weight(self, tweet):
        #print("increase!!!!!!!!!!!!!!!!!!!!!!")
        for t in tweet:
            if t in self.word_index:
                self.weight[self.word_index[t]] += self.learning_rate

    def decrese_weight(self, tweet):
        #print("decrease!!!!!!!!!!!!!!!!!!!!!!")
        for t in tweet:
            if t in self.word_index:
                self.weight[self.word_index[t]] -= self.learning_rate    
                if self.weight[self.word_index[t]] < 0:
                    self.weight[self.word_index[t]] = self.learning_rate  
    
    #multiply
    # def calculate(self, tweet, c):
    #     p_total = []
    #     p = 0
    #     for i in range(len(tweet)):
    #         word_count = self.find_word_count(tweet[i], c)
    #         # if tweet[i] in self.NW:
    #         #     continue
    #         if i > 0 and tweet[i-1] in self.NW:
    #             p = float(word_count)/self.count[1-c]
    #             if tweet[i] in self.word_index:
    #                 p *= self.weight[self.word_index[tweet[i]]]
    #             p_total.append(p)
    #         else:
    #             p = float(word_count)/self.count[c]
    #             if tweet[i] in self.word_index:
    #                 p *= self.weight[self.word_index[tweet[i]]]
    #             p_total.append(p)
    #     p = 1
    #     for i in p_total:
    #         p *= i
    #     return p*self.prior[c] 
    #log
    def calculate(self, tweet, c):
        p_total = math.log(self.prior[c])
        p = 0
        #hasNW = False
        for i in range(len(tweet)):
            word_count = self.find_word_count(tweet[i], c)
            if i > 0 and tweet[i-1] in self.NW:
                p = float(word_count)/self.count[1-c]
                p_total += math.log(p)
                # hasNW = False
            else:
                p = float(word_count)/self.count[c]
                p_total += math.log(p)
            # if tweet[i] in self.NW:
            #     hasNW = True
        return p_total 

    def calculate_predict(self, tweet, c):
        p_total=[]
        p = 0
        for i in range(len(tweet)):
            if tweet[i] in self.word_index:
                if i > 0 and tweet[i-1] in self.NW:
                    f = self.occur_count[self.word_index[tweet[i]]][1-c]
                    #p = (float(f)+1)/(float(self.count[1-c])+self.n_words)
                    p = float(f)/self.count[1-c]
                else:
                    f = self.occur_count[self.word_index[tweet[i]]][c]
                    #p = (float(f)+1)/(float(self.count[c])+self.n_words)
                    p = float(f)/self.count[c]
                p_total.append(p)
        p = 1
        for i in p_total:
            p = p*i
        return p * self.prior[c]

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
            
