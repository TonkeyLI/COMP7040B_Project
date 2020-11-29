#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes
from multinomial_naive_bayesNH import MultinomialNaiveBayesNH
from sklearn.metrics import classification_report, confusion_matrix
from package_model import Package_model

if __name__ == '__main__':

    print("Train using package")
    p  = Package_model()
    p.train_test()

    print("load dataset")
    dataset = SentimentCorpus() # to do

    print("\n=======================================================\n")
    print("+++ Naive Bayes +++")
    nb = MultinomialNaiveBayes()
    print("Model created")
    params = nb.train(dataset.train_X, dataset.train_y)
    print("train completed")
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    print("Evaluate train completed")
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    print("Accuracy on training data = %f \nAccuracy on testing data = %f" % (eval_train, eval_test))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset.test_y,predict_test))
    print(classification_report(dataset.test_y,predict_test))

    print("\n=======================================================\n")
    print("+++ Naive Bayes with negation handling +++")
    nbNH = MultinomialNaiveBayesNH()
    print("Model created")
    nbNH.train(dataset.train_X, dataset.train_y, dataset.feat_dict)
    print("train completed")
    nbNH.save_model()
    print("model saved")
    predict_train = nbNH.test(dataset.train_X_NH)
    eval_train = nbNH.evaluate(predict_train, dataset.train_y)
    print("Evaluate train completed")
    
    predict_test = nbNH.test(dataset.test_X_NH)
    eval_test = nbNH.evaluate(predict_test, dataset.test_y)
    print("Accuracy on training data = %f \nAccuracy on testing data = %f" % (eval_train, eval_test))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset.test_y,predict_test))
    print(classification_report(dataset.test_y,predict_test))

    
     

