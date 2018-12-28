import os
import sys
import numpy as np

# covariance calculation for Bayesian model
def covariance_matrix(np_data):
    keys=np_data.keys()
    cov_matrix={}
    for key in keys:
        temp_matrix=np.cov(np_data[key])
        cov_matrix[key]=temp_matrix
        print('\nThe covariance matrix of %s is ' % key)
        print(temp_matrix)
    return cov_matrix

# mean calculation for Bayesian model
def mean_vector(np_data):
    keys=np_data.keys()
    mean_vector={}
    for key in keys:
        temp_vector=np.mean(np_data[key],axis=1)
        mean_vector[key]=temp_vector
        print('\nThe mean vector of %s is ' % key)
        print(temp_vector)
    return mean_vector

# Bayesian model in the classification
def Bayesian_model(np_data,cov_matrix,avg_vector,prior):
    keys=np_data.keys()
    inv_cov_matrix={}
    det_inv_cov_matrix={}
    false_rate=0.0
    for key in keys:
        temp_matrix=np.linalg.inv(cov_matrix[key])
        inv_cov_matrix[key]=temp_matrix
        det_inv_cov_matrix[key]=np.linalg.det(inv_cov_matrix[key])
    
    for key in keys:
        temp_data=np_data[key].T
        row_num=temp_data.shape[0]
        for temp_a in range(row_num):
            # row vector
            temp_vector={}
            temp_vector['E3']=temp_data[temp_a]-avg_vector['E3']
            temp_vector['E5']=temp_data[temp_a]-avg_vector['E5']
            conditional_E3=-np.dot(temp_vector['E3'],np.dot(inv_cov_matrix['E3'],temp_vector['E3']))/2   \
                -np.log(det_inv_cov_matrix['E3'])/2-np.log(2*np.pi)
            judge_E3=conditional_E3+np.log(prior['E3'])
            conditional_E5=-np.dot(temp_vector['E5'],np.dot(inv_cov_matrix['E5'],temp_vector['E5']))/2   \
                -np.log(det_inv_cov_matrix['E5'])/2-np.log(2*np.pi)
            judge_E5=conditional_E5+np.log(prior['E5'])
            if (judge_E3>judge_E5) & (key=='E5'):
                prob_E3=np.e**conditional_E3
                false_rate=false_rate+prob_E3*prior['E3']
            elif (judge_E5>judge_E3) & (key=='E3'):
                prob_E5=np.e**conditional_E5
                false_rate=false_rate+prob_E5*prior['E5']
    return false_rate
