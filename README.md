# COMP7404B Project: Group 5
## Course Project of Machine Learning 
## [Twitter Sentiment Analysis on COVID-19 in California](https://github.com/TonkeyLI/COMP7404B_Project)

### Group members (Alphabetically): 
* Chan Wai Kei Wikie 
* Cheng Pengjie 
* Law Tsz Kit 
* Li Jiazhou 
* Xu Siyu

### Package need to be installed:
* Numpy
* Pandas
* matplotlib
* sklearn.metrics

### Description of the project
Twitter, widely used in the US, is a popular social media for public to express their opinions and it has been used in several machine learning research to investigate common people's preferences. 
In our project, we would like to find out if there were any relationships between residents' sentiments in California and the number of COVID-19 infections during a two-month period starting from 01/04/2020.
Throughout the project, we implemented our naives bayes probability model based on a research paper, [Twitter Sentiment Analysis Using a Modified Naïve Bayes Algorithm](https://link.springer.com/chapter/10.1007/978-3-319-67220-5_16), and made further improvements by considering negation words and weights.

### Project workflow
1. Data collection
2. Training the model by data from [Sentiment140 ](https://www.kaggle.com/kazanova/sentiment140)
3. Testing the model
4. Tuning the accuracy through various implementations
5. Plotting the sentiments and infection population
6. Intepretation

## Files description:

1.main.py

The start point of this project. It will train models and do testing using two algorithms (Naive Bayes, as well as Naive Bayes with negation handling).

2.sentiment_reader.py

Load the dataset, generate word dictionaries and split data.

3.multinomial_naive_bayes.py

Generate the model of Naive Bayes classifier. Calcualte prior and likelihood of words based on Naive Bayes algorithm.

4.multinomial_naive_bayesNH.py

Generate the model of the naive bayes classifier with negation words handling. It is used to do training, testing, evaluation and saving the model.

5.linear_classifier.py

A base class offering functions of tesing and evaluation.

6.package_model.py

Generate models using scikit-learn library as a reference. 

7.predict.py

This python file is used to predict the tweets. The result is either 0 or 1. 0 means positive sentiment and 1 means negative sentiment. It will print a graph to show the prediction result.

8.tweet_improve.py

This python file is used to analyze the data that were grabbed from twitter. It will collect the useful data and change the data into correct form. The code will generate two new files. Those files contain either only positive sentiments or negative sentiments.

9.California_Tweets.py
The file does the webscraping by using Twitter's public API to fetch tweets within the designated period.## 

### How to run:

1.main.py (To train and save the model)

> python ./main.py

2.Predict.py(Load model, do testing and visulize result)

> python ./predict.py

3.tweet_improve.py

> python ./twweet_improve.py

### Results
[![Result](https://iili.io/FwNvBS.md.png)](https://freeimage.host/i/FwNvBS)

[![Comparison](https://iili.io/FwNg2e.md.png)](https://freeimage.host/i/FwNg2e)

### Conclusions
* Potential positive relationship between the negative sentiments and COVID-19 infections.
* Simple Naives Bayes Model can be further improved by including negation words and weightings.

