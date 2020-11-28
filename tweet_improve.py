import numpy as np
import pandas as pd
import csv

def first_choice():
    resultList_positive=[]
    resultList_negetive=[]
    list,date,label=read_csv()
    count=0
    positive_count=0
    negative_count=0
    for lines in list:
        line=lines
        line=line.split()
        counts=dict()
        resultsentence = ""
        for word in line:
            if word in counts:
                counts[word]+=1
            else:
                counts[word]=1
        for word in line:
            resultsentence+=word+":"+str(counts[word])+" "
        resultsentence+="Label:"+label[count]
        if label[count]=="positive":
            resultsentence=[resultsentence]
            resultList_positive.append(resultsentence)
            resultList_positive[positive_count].append(date[count])
            positive_count+=1
            #print resultList_positive
        else:
            resultsentence = [resultsentence]
            resultList_negetive.append(resultsentence)
            resultList_negetive[negative_count].append(date[count])
            negative_count += 1
        count+=1
        #print resultsentence
        #print counts
    #print resultList[0]
    return resultList_positive,resultList_negetive
       # print line


def out_put(data_list_positive,data_list_negative,column_list, file_path_pos,file_path_neg):
    #data_list=change_format(data_list)
    #data_list[0].append(12)
    #print data_list[0]
   # for i in range(0, len(date_list)):
    #    data_list[i].append(date_list[i])
    #print data_list
    with open(file_path_pos,'w') as csvfile:
        csvwriter=csv.writer(csvfile)
        csvwriter.writerow(column_list)
        csvwriter.writerows(data_list_positive)
    csvfile.close()

    with open(file_path_neg,'w') as csvfile1:
        csvwriter = csv.writer(csvfile1)
        csvwriter.writerow(column_list)
        csvwriter.writerows(data_list_negative)
    csvfile1.close()

def change_format(data_list):
    resultlist=[]
    for sentence in data_list:
        temp_list=[sentence]
        resultlist.append(temp_list)
    return  resultlist

def read_csv():
    # resultList.append(1)
    # resultList.append(123)
    # print resultList
    col_list = ["tweet","date","label"]
    read_data = pd.read_csv("training_set_processed.csv", usecols=col_list)
    list = read_data.tweet.to_list()
    label=read_data.label.to_list()
    date=read_data.date.tolist()
    return list,date,label

def second_choice():
    resultlist_pos=[]
    resultlist_neg=[]
    data,date,label=read_csv()
    count_pos=0
    count_neg=0
    count=0
    for lines in data:
        #lines+=':'+'positive'
        if label[count]=="positive":
            lines=[lines]
            resultlist_pos.append(lines)
            #print resultlist_pos[0]
            resultlist_pos[count_pos].append(date[count])
            count_pos+=1
        else:
            lines = [lines]
            resultlist_neg.append(lines)
            resultlist_neg[count_neg].append(date[count])
            count_neg += 1
        count+=1
    return resultlist_pos,resultlist_neg

def out_put2(data_list,column_list, file_path):
    #data_list=change_format(data_list)
    #data_list[0].append(12)
    #print data_list[0]
   # for i in range(0, len(date_list)):
    #    data_list[i].append(date_list[i])
    #print data_list
    with open(file_path,'w') as csvfile:
        csvwriter=csv.writer(csvfile)
        csvwriter.writerow(column_list)
        csvwriter.writerows(data_list)
    csvfile.close()

def unique_list(line):
    ulist=[]
    [ulist.append(x) for x in line if x not in ulist]
    return ulist

def main():
    print "working"
    #data_list_positive,data_list_negative=first_choice()
    #out_put(data_list_positive,data_list_negative,["tweet","date"],"twitter_improve_positive.csv","twitter_improve_negative.csv")
    pos,neg=second_choice()
    #print data_list
    #list,date=read_csv()
    #change_char(date)
    #oaut_put2(data_list,["tweet_no_sws","date"],"twitter_improve.csv")
    out_put(pos,neg,["tweet","date"],"twitter_improve_positive.csv","twitter_improve_negative.csv")

if __name__ == '__main__':
    main()