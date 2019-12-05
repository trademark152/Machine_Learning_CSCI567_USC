# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:51:38 2016

@author: Azarang
"""

from __future__ import division
from functions import list_duplicates_of
from functions import index_containing_substring
from functions import all_indices
from functions import N_maximum_elements
from functions import feature_maker
from numpy import *
from operator import truediv
import numpy as np
import csv




from keras.layers import Input, Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from hw_utils import genmodel
from hw_utils import loaddata
from hw_utils import normalize
from hw_utils import testmodels
from hw_utils import testmodels_test


    
with open('invited_info_train.txt') as f:
    train = f.read().splitlines()
with open('question_info.txt') as f1:
    question_info = f1.read().splitlines()
with open('user_info.txt') as f2:
    user_info = f2.read().splitlines()
    
# extracting users, questions and their replies. 
user= [] 
ques = []
output = []   
for item in range(0, len(train)):
    user1 = train[item].split("\t")
    user.append(user1[1])
    ques.append(user1[0])
    output.append(user1[2])
    
    
''' =====================user labels==========================================='''    
Y_TRAINING = map(int, output)

''' extracting the users id, tags, and words from the users_info file'''
usersin_info = []
tags_users = []
words_users = []
for r in range(0,len(user_info)):
    xx = user_info[r].split("\t")
    usersin_info.append(xx[0])
    tags_users.append(xx[1])
    words_users.append(xx[2])
    
''' deleting the users without words'''
index_of_null_users = []
for r in range(0, len(words_users)):
    if (words_users[r]=='/'):
        index_of_null_users.append(r)

    
''' changing the words and tags_users list to integer list in the users_info file'''
list_word_users = []
for ff in range(0, len(words_users)):
    if ff in index_of_null_users:
        list_word_users.append(-1)
    else:
        my_list2 = words_users[ff].split("/")
        ttt2 = map(int,my_list2 )
        list_word_users.append(ttt2)
list_tags_users = []
for ee in range(0, len(tags_users)):
    my_list3 = tags_users[ee].split("/")
    ttt3 = map(int, my_list3)
    list_tags_users.append(ttt3)
''' extracting the question id, question tag, words list, number of upvotes, and number of answers'''    
quesin_info = []
tags_ques = []
words_ques=[]
num_upvotes=[]
num_answers=[]
num_top=[]
for r in range(0,len(question_info)):
    yy = question_info[r].split("\t")
    quesin_info.append(yy[0])
    tags_ques.append(yy[1])
    words_ques.append(yy[2])
    num_upvotes.append(yy[4])
    num_answers.append(yy[5])
    num_top.append(yy[6])
''' deleting the questions without words'''
index_of_null_ques = []
for r in range(0, len(words_ques)):
    if (words_ques[r]=='/'):
        index_of_null_ques.append(r)
 
    
''' changing the words list and answers and upvotes to integer lists in the question_info file'''
upvotes = map(int, num_upvotes)
answers = map(int, num_answers)
top_quality = map(int, num_top)
question_tags = map(int, tags_ques)
list_word_ques = []
for ff in range(0, len(words_ques)):
    if ff in index_of_null_ques:
        list_word_ques.append(-1)
    else:
        my_list1 = words_ques[ff].split("/")
        ttt = map(int,my_list1 )
        list_word_ques.append(ttt)

    
''' finding maximum of words in users and questions'''
max_user = 0
for i in range(0, len(list_word_users)):
    if list_word_users[i]!=-1:
        for j in range(0, len(list_word_users[i])):
            if max_user<=list_word_users[i][j]:
                max_user = list_word_users[i][j]
max_ques = 0
for i in range(0, len(list_word_ques)):
    if list_word_ques[i]!=-1:
        for j in range(0, len(list_word_ques[i])):
            if max_ques<=list_word_ques[i][j]:
                max_ques = list_word_ques[i][j]
''' creating user_mat and ques_mat for users and questions'''
user_word_mat = np.zeros((len(usersin_info), max_ques+1))
user_tag_mat = np.zeros((len(usersin_info), 143))
for i in range(0, len(usersin_info)):
    for j in range(0, len(list_tags_users[i])):
        user_tag_mat[i][list_tags_users[i][j]]=1
    if list_word_users[i]!=-1:
        for j in range(0, len(list_word_users[i])):
            if list_word_users[i][j]<=max_ques:
                user_word_mat[i][list_word_users[i][j]]=1

ques_word_mat = np.zeros((len(quesin_info), max_ques+1))
ques_tag_mat = np.zeros((len(quesin_info), 20))
ques_upvote_mat = np.zeros((len(quesin_info),1))
ques_ans_mat = np.zeros((len(quesin_info),1))
ques_top_mat = np.zeros((len(quesin_info),1))
for i in range(0, len(quesin_info)):
    ques_tag_mat[i][question_tags[i]]=1
    if list_word_ques[i]!=-1:
        for j in range(0, len(list_word_ques[i])):
            ques_word_mat[i][list_word_ques[i][j]]=1
ques_upvote_mat = upvotes
ques_ans_mat = answers
ques_top_mat = top_quality
''' mean feature for unavailable users'''
ans_top = [0]*len(upvotes)
for i in range(0, len(upvotes)):
    if ques_ans_mat[i]!=0:
        ans_top[i] = ques_top_mat[i]/ques_ans_mat[i]
    
sum_feature = [0]*20
mean_counter = [0]*20
mean_feature = [0]*20
for i in range(0, len(ans_top)):
    sum_feature[question_tags[i]]=sum_feature[question_tags[i]]+ans_top[i]
    mean_counter[question_tags[i]] = mean_counter[question_tags[i]]+1 
mean_feature = map(truediv, sum_feature, mean_counter)
mean_feature = np.array(mean_feature)

''' ================creating features for training samples====================='''
X_TRAIN = feature_maker(user,usersin_info,user_word_mat,mean_feature,ques_word_mat,ans_top,ques_tag_mat,quesin_info)


''' =======================creating features validation================================'''
with open('validate_nolabel.txt') as fr:
    test = fr.read().splitlines()

test.remove(test[0])
# extracting users, questions. 
user_test= [] 
ques_test = []  
for item in range(0, len(test)):
    user1_test = test[item].split(",")
    user_test.append(user1_test[1])
    ques_test.append(user1_test[0])
    
X_VALID = feature_maker(user_test,usersin_info,user_word_mat,mean_feature,ques_word_mat,ans_top,ques_tag_mat,quesin_info)


'''======================================================================================================'''
'''================================================  Training============================================'''    
    
S = Y_TRAINING
Y_TRAIN = np.zeros((len(S),2))
for i in range(0, len(S)):
    if S[i]==0:
        Y_TRAIN[i][0]=1
    else:
        Y_TRAIN[i][1]=1


a = np.random.permutation(len(Y_TRAIN))

NUM_TEST = 10000;
NUM_LEN = len(Y_TRAIN)
X_tr = zeros([NUM_LEN-NUM_TEST, len(X_TRAIN[0])])
y_tr = zeros([NUM_LEN-NUM_TEST, 2])
for i in range(0,NUM_LEN-NUM_TEST):
    X_tr[i] = X_TRAIN[a[i]]
    y_tr[i] = Y_TRAIN[a[i]]

X_te = zeros([NUM_TEST, len(X_TRAIN[0])])
y_te = zeros([NUM_TEST, 2])
for i in range(0,NUM_TEST):
    X_te[i] = X_TRAIN[a[i+NUM_LEN-NUM_TEST]]
    y_te[i] = Y_TRAIN[a[i+NUM_LEN-NUM_TEST]]


L = len(X_tr[0])
archs = [[L,50,2],[L,50,30,2],[L,500,2],[L,500,300,2],[L,500,500,300,2]]
reg_coeff = [1e-07,1e-06,1e-05]
sgd_decay=[1e-05,1e-04,1e-03]
moms = [0.99]


best_param = testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=reg_coeff, 
				num_epoch=100, batch_size=1000, sgd_lr=5*1e-05, sgd_decays=sgd_decay, sgd_moms=moms, 
					sgd_Nesterov=True, EStop=False, verbose=0)     
        
print best_param
Prob = testmodels_test(X_tr, y_tr, X_VALID, [best_param[0]], actfn='relu', last_act='softmax', reg_coeffs=[best_param[1]], 
				num_epoch=100, batch_size=1000, sgd_lr=5*1e-05, sgd_decays=[best_param[2]], sgd_moms=moms, 
					sgd_Nesterov=True, EStop=False, verbose=0)
    
    
'''==================================== save =========================================='''

with open('CSCI_KAP.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(0,len(X_VALID)):
            writer.writerow([ques_test[i] ,user_test[i], Prob[i][1]])   
    
