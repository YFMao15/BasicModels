import os
import sys
import numpy as np
from collections import defaultdict
from sklearn import preprocessing

# select data function
def select_data(data_trait,folder_content):
    keys=folder_content.keys()
    for key in keys:
        if (data_trait[0] in key) & (data_trait[1] in key):
            selected_data=folder_content[key]
    return selected_data
    
# calculate conditional probability for Bayesian model 
def classify_data(data):
    keys=data.keys()
    row_num=len(data.keys())
    col_num=len(data['name'])
    temp_list=[]
    for key in keys:
        temp_list.append(data[key])
    temp_list=np.array(temp_list).T.tolist()
    data_E3=[]
    data_E5=[]
    np_data={}
    for temp_a in range(col_num):
        if 'E3' in temp_list[temp_a]:
            temp_column=[]
            for temp_b in range(row_num-2):
                temp_column.append(data[temp_b+1][temp_a])
            data_E3.append(temp_column)
        elif 'E5' in temp_list[temp_a]:
            temp_column=[]
            for temp_b in range(row_num-2):
                temp_column.append(data[temp_b+1][temp_a])
            data_E5.append(temp_column)
    np_data_E3=np.array(data_E3).T
    np_data_E5=np.array(data_E5).T
    np_data['E3']=np_data_E3
    np_data['E5']=np_data_E5
    return np_data

# normalize data before processing the simple classifier
def normalize_data(original_data,data_scale=None):
    normalized_data=defaultdict(list)
    row_num=len(original_data.keys())
    col_num=len(original_data['name'])
    keys=original_data.keys()
    np_data=np.empty((row_num-2,col_num),dtype=float)
    np_logarithmic_data=np.empty((row_num-2,col_num),dtype=float)
    for temp_a in range(row_num-2):
        np_data[temp_a]=original_data[temp_a+1]
    # use logarithm to limit the range
    np_logarithmic_data=np.log10(np_data+0.1*np.ones((row_num-2,col_num),dtype=float))
    # preprocessing function normalize the data in column
    # transpose np_logarithmic_data before the normalization
    if data_scale==None:
        # record the normalization of training data in normalized_scale
        # process the same normalization for testing data
        normalized_scale=preprocessing.StandardScaler().fit(np_logarithmic_data.T)
        temp_normalized_data=normalized_scale.transform(np_logarithmic_data.T)       
    else:
        temp_normalized_data=data_scale.transform(np_logarithmic_data.T)
        normalized_scale=data_scale
    np_normalized_data=temp_normalized_data.T
    for key in keys:
        if isinstance(key,int):
            normalized_data[key]=np_normalized_data[key-1]
        else:
            normalized_data[key]=original_data[key]
    return normalized_data,normalized_scale

