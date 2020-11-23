#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':

    dataset = SentimentCorpus() # to do
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y, dataset.feat_dict)
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    print("\n=======================================================\n")
    print("+++ Naive Bayes +++")
    print("Accuracy on training data = %f \n Accuracy on testing data = %f" % (eval_train, eval_test))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset.test_y,predict_test))
    print(classification_report(dataset.test_y,predict_test))
     

