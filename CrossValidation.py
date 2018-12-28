import os
import sys
import numpy as np
from sklearn.model_selection import KFold

def K_fold_splitter(np_data,fold_num):
    keys=np_data.keys()
    np_splitted_data={}
    for key in keys:
        np_splitted_data[key]={'train':[],'test':[]}
        temp_data=np_data[key].T
        kf=KFold(n_splits=10,shuffle=False)
        fold_num=0
        for result_index in kf.split(temp_data):
            train_indices=result_index[0]
            test_indices=result_index[1]
            temp_train_splitted_data=[]
            temp_test_splitted_data=[]
            for index in train_indices:
                temp_train_splitted_data.append(temp_data[index])
            np_splitted_data[key]['train'].append(np.array(temp_train_splitted_data).T)
            for index in test_indices:
                temp_test_splitted_data.append(temp_data[index])
            np_splitted_data[key]['test'].append(np.array(temp_test_splitted_data).T)
    return np_splitted_data
