# COMP7404B_Project
Course Project of Machine Learning 

Group Members (Alphabetically): Chan Wai Kei Wikie, Cheng Pengjie, Law Tsz Kit, Li Jiazhou, Xu Siyu

Package need to be install:
Numpy
Pandas
json
matplotlib
sklearn.metrics
math
csv


File Description:

1.main.py
The start point of this project. It will train models and do testing using two algorithms(Naive bayes as well as Naive bayes with negation handling).

2.sentiment_reader.py
Load the dataset, generate word dictionaries and split data.

3.multinomial_naive_bayes.py
Generate the model of naive bayes classifier. Calcualte prior and likelihood of words based on naive bayes algorithm.

4.multinomial_naive_bayesNH.py
Generate the model of the naive bayes classifier with negation words handling. Used to do training, testing, evaluation and saving the model.

5.linear_classifier.py
A base class offering functions of tesing and evaluation.

6.package_model.py
Generate models using sklearn library as a reference. 

7.Predict.py
This python file is used to predict the tweets. The result is either 0 or 1. 0 means positive sentiment and 1 means negative sentiment. It will print a graph to show the prediction result.

8.tweet_improve.py
This python file is used to analyze the data that were grabbed from twitter. It will collect the useful data and change the data into correct form. The code will generate two new files. Those files contrain either only positive sentiment or negative sentiment.

9.California_Tweets.py


How to run:

1.main.py(To train and save the model)

python ./main.py

2.Predict.py(Load model, do testing and visulize result)

python ./predict.py

3.tweet_improve.py

python ./twweet_improve.py


