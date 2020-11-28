import numpy as np
import pandas as pd
import json
from multinomial_naive_bayesNH import MultinomialNaiveBayesNH
import matplotlib.pyplot as plt


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



def calculate_percent(predict_list):
    resultlist=[]
    for predict in predict_list:
        count_neg=0
        count_pos=1
        for i in predict:
            if i==1:
                count_neg+=1
            elif i==0:
                count_pos+=1
        count_total=count_pos+count_neg
        count_total=float(count_total)
        count_neg=float(count_neg)
        count_percent=count_neg/count_total

        temp=[]
        temp.append(predict[0])
        temp.append(count_percent)
        resultlist.append(temp)
    return resultlist


def calculate_cumulative(predict_list):
    resultlist=[]
    count_neg = 0
    count_pos = 0
    for predict in predict_list:
        for i in predict:
            if i==1:
                count_neg+=1
            elif i==0:
                count_pos+=1
        count_total=count_pos+count_neg
        temp=[]
        temp.append(predict[0])
        temp.append(count_pos)
        temp.append(count_neg)
        temp.append(count_total)
        resultlist.append(temp)
    return resultlist

def plot_graph(data_list):
    x=[]
    y_pos=[]
    y_neg=[]
    y_total=[]
    count=len(data_list)-1
    while count>=0:
        x.append(data_list[count][0])
        y_pos.append(data_list[count][1])
        y_neg.append(data_list[count][2])
        y_total.append(data_list[count][3])
        count-=1
        '''
    for data in data_list:
        x.append(data[0])
        y_pos.append(data[1])
        y_neg.append(data[2])
        y_total.append(data[3])
    '''
    x_axis=[]
    for i in range (0,len(x)):
        x_axis.append(i)
    pos_reverse=Reverse(y_pos)
    neg_reserse=Reverse(y_neg)
    total_reserse=Reverse(y_total)
    plt.plot(x_axis,pos_reverse,label="Positive Sentiments")
    plt.plot(x_axis,neg_reserse,label="Negative Sentiments")
    plt.plot(x_axis,total_reserse, label="Total Sentiments")
    plt.xlabel('Days')
    plt.ylabel('Cumulative Numbers of Sentiments')
    plt.title("(From 01-04-2020 to 30-06-2020)")
    plt.legend(loc='upper left')
    plt.show()


def Reverse(lst):
    return [ele for ele in reversed(lst)]



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
    #print predict_list
    cal=calculate_cumulative(predict_list)
    plot_graph(cal)

if __name__ == '__main__':
    main()