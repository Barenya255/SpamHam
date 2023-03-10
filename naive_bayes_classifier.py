# -*- coding: utf-8 -*-
"""
The following is an implementation of the Naive Bayes' Algorithm to detect spam
mails from non spam mails'
"""

'''The following are the relevant import statements'''
import pandas as pd
import numpy as np
import math
from string import punctuation as punc 
import os


'''The fucntion below imports the data sets from the folders below. The CSV files
were downloaded from Kaggle'''
def imp_data():
    data_set = pd.read_csv("completeSpamAssassin.csv")
    data_set_Spam = pd.read_csv("enronSpamSubset.csv")
    data_set_ling = pd.read_csv("lingSpam.csv")
    data_set.to_numpy()
    return data_set,data_set_Spam,data_set_ling


'''The following is used to make the dictionary of the words, we remove punctuations
and stopwords as best we can.
We return a dictionary of the most frequently used words(top 2000)'''
def make_dictionary(data_set):
    
    words = {}
    for i in range(len(data_set)):
        try:
            x = data_set.loc[i]["Body"].lower()
            x = x.translate(str.maketrans(' ',' ',punc))
            temp = x.split()
            #print("done" + str(i), end = ",")
        except AttributeError:
            temp = []
        for word in temp:
            if words.get(word) == None:
                words[word] = 1
            else:
                words[word] += 1
    #sort the dictionary and take upto frequency 5000
    temp = sorted(words.items(), key = lambda x:x[1], reverse = True)
    words = []
    count = 0
    
    for word in temp:
        words.append(word[0])
        count += 1
        if count == 2000:
            break
    
    words = pre_process(words)
    #print("done")
    diction = {}
    index = 0
    for word in words:
        if diction.get(word) == None:
            diction[word] = index
            #print(index, end = ",")
        index += 1
        
    #print("dictionary made")
    #words = pre_process(words)
    return diction

'''The following function removes the stop words from the data set'''
def pre_process(words):
    from nltk.corpus import stopwords
    
    #words = words.translate(str.maketrans(' ',' ',punc))
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

'''The following is a helper function to the fit function which updates the 
parameters'''       
def train(one_hot, words, train_mail, train_label, p_0, p_1, p):
    #print(type(train_mail))
    #print("training on mail")
    trainer = []
    try:
        trainer = pre_process(train_mail.split())
    except AttributeError:
        print(train_mail)
    if train_label == 0:
        for word in trainer:
            try:
                p_0[words[word]] += 1  
            except KeyError:
                s = 0
                
    if train_label == 1:
        for word in trainer:
            try:
                p_1[words[word]] += 1
            except KeyError:
                s = 0
    return p_0,p_1

'''The fit fucntion simply trains and updates the parameters to the best possible'''
def fit(one_hot, words, data_set,p_0, p_1, p):
    i = 0
    count_0 = 1.0
    count_1 = 1.0
    print("Training_phase_begins")
    for train_mail in data_set["Body"]:
        #train_mail = train_mail.translate(str.maketrans(' ',' ',punc))
        #train_mail.lower()
        print(i, end = ",")
        p_0,p_1 = train(one_hot, words, train_mail, data_set["Label"][i], p_0, p_1, p)
        if data_set["Label"][i] == 0:
            count_0+=1
        else:
            count_1+=1
        i+=1
    p_0 /= count_0
    p_1 /= count_1
    
    p = count_1/(count_0+count_1)
    
    np.save('p_0_values.npy', p_0)
    np.save('p_1_values.npy', p_1)
    np.save('p_values.npy', p)
    
    print("training_done")
    
    return p_0,p_1,p

'''The following calculates the probability density function for the Bernoulli'''
def bernoulli(params, data):
    bern = 1
    for d in range(len(params)):
        bern *= (params[d]**data[d])*((1-params[d])**(1-data[d]))
    return bern


'''The following is used to make prediction'''
def predict(p_0, p_1, p, mail, n):
    one_hot = np.zeros(n)
    for word in mail:
        try:
            one_hot[words[word]] = 1
        except KeyError:
            s = 0
            #print(word + "not evaluated as not in dictionary.")
    #print(one_hot)
    #normalization
    p_0 /= np.linalg.norm(p_0)
    p_1 /= np.linalg.norm(p_1)
    coeff_1 = 0.0
    coeff_2 = 0.0
    for i in range(len(one_hot)):
        if (p_1[i]*(1-p_0[i]))/(p_0[i]*(1-p_1[i])) <= 0:
            print(i)
            print(p_1[i])
            print(p_0[i])
        coeff_1 += one_hot[i]*math.log((p_1[i]*(1-p_0[i]))/(p_0[i]*(1-p_1[i])))
        coeff_2 += math.log((1-p_1[i])/(1-p_0[i]))
    coeff_3 = math.log(p/(1-p))
    decision_boundary = coeff_1 + coeff_2 + coeff_3
    #print(decision_boundary)
    if decision_boundary >= 0:
        return 1
    else:
        return 0
    
'''The following is a helper for reading files'''
def readFile(file_path, p_0, p_1, p):
    
    answer = -1
    try:
        with open(file_path, 'r') as f:
            answer = -1
            temp = f.readlines()
            mail = ""
            for i in range(len(temp)):
                mail += temp[i]
            mail = mail.translate(str.maketrans(' ',' ',punc))
            mail = pre_process(mail.split())
            answer = predict(p_0,p_1,p, mail, d)
    except FileNotFoundError:
        print("fatal error")
    print(answer)
    return answer
    
'''The following calculates accuracy for given csv data sets'''
def calc_accuracy(p_0,p_1,p, data_set, one_hot):
    accuracy = 0.0
    p_0 = np.load('p_0_values.npy')
    p_1 = np.load('p_1_values.npy')
    p = np.load('p_values.npy')
    for i in range(len(data_set)):
        mail = data_set.iloc[i]["Body"]
        if type(mail) != str:
            continue
        mail = mail.translate(str.maketrans(' ',' ',punc))
        mail = pre_process(mail.split())
        if predict(p_0,p_1,p, mail, len(one_hot)) == data_set.iloc[i]["Label"]:
            accuracy += 1
            print("accurate")
        else:
            print("inaccurate")
    accuracy /= len(data_set)
    print("accuracy of the following test data comes out to be: " + str(accuracy))
        
    
'''The following is used to test on the test folder'''
def test_on_folder(one_hot):
    p_0 = np.load('p_0_values.npy')
    p_1 = np.load('p_1_values.npy')
    p = np.load('p_values.npy')
    path = os.getcwd()
    path += "\\test"
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
      
            # call read text file function
            readFile(file_path,p_0,p_1,p)
            
'''The main function'''
if __name__ == "__main__":
    data_set_1, data_set_2, data_set_3 = imp_data()
    words = make_dictionary(data_set_1)
    d = 2000
    p_0 = np.ones(d)
    p_1 = np.ones(d) 
    p = 0.5
    
    one_hot = np.zeros(d)
    '''The test_on_folder function is to be used to test on the test folder'''
    #the following line can be commented out if p_0, p_1 and p are used from the.npy files in the folder
    p_0, p_1, p = fit(one_hot, words, data_set_2, p_0, p_1, p)
    #calc_accuracy(p_0,p_1,p, data_set_1, one_hot)
    test_on_folder(one_hot)
    