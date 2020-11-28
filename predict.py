import numpy as np
import pandas as pd
import json
from multinomial_naive_bayesNH import MultinomialNaiveBayesNH

time_internal=1

def load_model(file_path_json,file_path_csv):
    model=[]
    with open(file_path_json, 'r') as f:
        model = json.load(f)
    f.close()
    occur_count = np.loadtxt(file_path_csv, delimiter=',')
    return model,occur_count

def load_data(file_path):
    col_list = ["tweet","date"]
    read_data = pd.read_csv(file_path, usecols=col_list)
    tweet=read_data.tweet.to_list()
    date=read_data.date.to_list()
    return tweet,date

def seperate_tweet(tweet,dates):
    result_list=[]
    date_list=[]
    number_list=[]
    result_date=[]
    for date in dates:
        temp=date_format(date)
        date_list.append(temp)
    current_date=date_list[0]
    result_date.append(current_date)
    for i in range (0,len(date_list)):
        #print current_date
        if current_date-time_internal<date_list[i]:
            if i==len(date_list)-1:
                number_list.append(i)
            else:
                continue
        elif current_date-time_internal>=date_list[i]:
            number_list.append(i)
            current_date=date_list[i]
            result_date.append(current_date)

    #print len(number_list)
    number_index=0
    current_number=number_list[number_index]
    temp_list=[]
    for i in range(0,len(tweet)):
        if i<current_number:
            temp_list.append(tweet[i])
        elif i==current_number:
            temp_list.append(tweet[i])
            #print len(temp_list)
            result_list.append(temp_list)
            temp_list=[]
            if number_index<len(number_list)-1:
                number_index += 1
                current_number = number_list[number_index]

    return result_list,result_date


def date_format(date):
    result=''
    count=0
    flag=0
    number_count=0
    number_count1=0
    for char in date:
        if char=='/':
            count+=1
            if count==1:
                flag=1
            elif (count==2):
                flag=0
        else:
            result+=char
            if flag==1:
                number_count+=1
            elif count==2:
                number_count1+=1
            else:
                continue
    if number_count==1:
        result=result[:4]+"0"+result[4:]
    if number_count1==1:
        result=result[:6]+'0'+result[6:]
    result=int(result)
    return result


def analyze_data(data):
    resultlist = []
    for sentence in data:
        sentence=sentence.split()
        #temp_list = [sentence]
        resultlist.append(sentence)
    return resultlist

def seperate_data(data_set):
    tweet = data_set.tweet.to_list()
    date = data_set.date.tolist()

'''
model = {
            'n_words': self.n_words,
            'prior': self.prior,
            'count': self.count,
            'word_index': self.word_index,
            'NW': self.NW
        }
 '''


def create_model(model,occur_count):
    nbnh = MultinomialNaiveBayesNH()
    nbnh.n_words = model['n_words']
    nbnh.prior=model['prior']
    nbnh.count=model['count']
    nbnh.word_index=model['word_index']
    nbnh.NW=model['NW']
    nbnh.occur_count=occur_count
    return nbnh






def main():
    print "working"

    data,occur_count=load_model('model/model.json','model/model.csv')
    nbnh=create_model(data,occur_count)
    print ("model loaded")
    tweet,date=load_data('test_data/predict.csv')
    print ('data loaded')
    tweet_list,date_list=seperate_tweet(tweet,date)
    print ('Tweets Seperated')
    predict_list=[]
    for tweet in tweet_list:
        test_data=analyze_data(tweet)
    #print test_data
        predict_test=nbnh.test(test_data)
        predict_list.append(predict_test)
        #print predict_test
    date_index=0
    for predict in predict_list:
        predict.insert(0,date_list[date_index])
        date_index+=1
    print predict_list

if __name__ == '__main__':
    main()