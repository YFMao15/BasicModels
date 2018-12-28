import os
import sys
from FileReader import read_folder
from BayesianModel import covariance_matrix
from BayesianModel import mean_vector
from BayesianModel import Bayesian_model
from DataPlotter import quantile_plotter
from DataPlotter import boundary_plotter
from DataProcessing import classify_data
from DataProcessing import select_data
from DataProcessing import normalize_data


if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='original_data'
    folder_content=read_folder(folder_directory,folder_name)

    # P1, select 2 gene data
    data_trait1=['2','train']
    data_trait2=['2','test']
    data2_train=select_data(data_trait1,folder_content)
    data2_test=select_data(data_trait2,folder_content)
    # normalization process
    np_data2_train=classify_data(data2_train)
    print('Before normalization,')
    #quantile_plotter(np_data2_train)
    data2_normed_train,data2_scale=normalize_data(data2_train)
    data2_normed_test,data2_scale=normalize_data(data2_test,data2_scale)
    np_data2_normed_train=classify_data(data2_normed_train)
    np_data2_normed_test=classify_data(data2_normed_test)
    print('\nAfter normalization,')
    #quantile_plotter(np_data2_normed_train)
    cov_matrix=covariance_matrix(np_data2_normed_train)
    avg_vector=mean_vector(np_data2_normed_train)
    # model on training sets
    false_rate_train={}
    prior1={'E3':0.5,'E5':0.5}
    false_rate_train[1]=Bayesian_model(np_data2_normed_train,cov_matrix,avg_vector,prior1)
    prior2={'E3':1./6,'E5':5./6}
    false_rate_train[2]=Bayesian_model(np_data2_normed_train,cov_matrix,avg_vector,prior2)
    # model on testing sets
    false_rate_test={}
    prior1={'E3':0.5,'E5':0.5}
    false_rate_test[1]=Bayesian_model(np_data2_normed_test,cov_matrix,avg_vector,prior1)
    prior2={'E3':1./6,'E5':5./6}
    false_rate_test[2]=Bayesian_model(np_data2_normed_test,cov_matrix,avg_vector,prior2)
    # plotting the boundary
    boundary_plotter('Bayesian',np_data2_normed_train,cov_matrix,avg_vector,prior1,'Training','1:1 Prior')
    boundary_plotter('Bayesian',np_data2_normed_train,cov_matrix,avg_vector,prior2,'Training','1:5 Prior')
    boundary_plotter('Bayesian',np_data2_normed_test,cov_matrix,avg_vector,prior1,'Testing','1:1 Prior')
    boundary_plotter('Bayesian',np_data2_normed_test,cov_matrix,avg_vector,prior2,'Testing','1:5 Prior')


   


    
