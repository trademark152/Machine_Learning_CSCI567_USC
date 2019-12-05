# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:51:01 2016

@author: Lenovo
"""

from __future__ import division
from numpy import *
import numpy as np



from operator import truediv
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
 
    
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
    
    
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices
    

            
def N_maximum_elements(n, list1):
    maximum_indices = [0]*n
    for i in range(0 , n):
        maximum_indices[i] = all_indices(max(list1),list1)[0]
        list1[maximum_indices[i]] = -1
        
    return maximum_indices    
    
    
    
    
def feature_maker(user,usersin_info,user_word_mat,mean_feature,ques_word_mat,ans_top,ques_tag_mat,quesin_info):

    COUNT = len(user)
    user_training = np.zeros((COUNT,20))
    for i in range(0, COUNT):
        print i
        if user[i] in usersin_info:
            for j in range(0,len(usersin_info)):
                if user[i]==usersin_info[j]:
                    current_num = j
            one_indices = []
            current_user=[]
            for j in range(0, len(user_word_mat[current_num])):
                if user_word_mat[current_num][j]==1:
                    one_indices.append(j)
            if len(one_indices)==0:
                #zero_user = np.zeros((1,20))
                user_training[i,:]= mean_feature
            else:
                for j in range(0, len(one_indices)):
                    for k in range(0, len(quesin_info)):
                        if ques_word_mat[k][one_indices[j]]==1:
                            current_tag = ques_tag_mat[k]
                            current_weight = 0
                            if ans_top[k]>0:
                                current_weight = ans_top[k]
                                current_tag = np.multiply(current_tag,current_weight)
                                current_user.append(current_tag)
                current_user = np.array(current_user)
                my_sum = np.zeros((1,20))
                my_counter = np.zeros((1,20))
                if len(current_user)>0:
                    for j in range(0,len(current_user[0])):
                        my_counter[0,j] = sum(current_user[:,j]!=0)
                        my_sum[0,j] = sum(current_user[:,j])
                    normalized_user = my_sum/my_counter
                    normalized_user[isnan(normalized_user)]=0
                    normalized_user = np.array(normalized_user)
                    user_training[i,:] = normalized_user
                else:
                    user_training[i,:] = mean_feature
    else:
        user_training[i,:]= mean_feature

    return user_training