import os
import sys
from FileReader import read_folder
from DataProcessing import classify_data
from DataProcessing import select_data
from DataProcessing import normalize_data
from FisherModel import Fisher_model
from DataPlotter import boundary_plotter
from CrossValidation import K_fold_splitter

if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='original_data'
    folder_content=read_folder(folder_directory,folder_name)

    # P3 Fisher model of classification
    
    data_trait1=['2','train']
    data_trait2=['2','test']
    data2_train=select_data(data_trait1,folder_content)
    data2_test=select_data(data_trait2,folder_content)
    np_data2_train=classify_data(data2_train)
    np_data2_test=classify_data(data2_test)
    avg2_vector,dir2_vector,sct2_matrix,best2_threshold=Fisher_model('train_test','training',np_data2_train)
    Fisher_model('train_test','testing',np_data2_test,avg2_vector,sct2_matrix,dir2_vector,best2_threshold)
    boundary_plotter('Fisher',np_data2_train,dir2_vector,best2_threshold,'Training')
    #boundary_plotter('Fisher',np_data2_test,dir2_vector,best2_threshold,'Testing')
    

    data_trait3=['10','train']
    data_trait4=['10','test']
    data10_train=select_data(data_trait3,folder_content)
    data10_test=select_data(data_trait4,folder_content)
    np_data10_train=classify_data(data10_train)
    np_data10_test=classify_data(data10_test)
    avg10_vector,dir10_vector,sct10_matrix,best10_threshold=Fisher_model('train_test','training',np_data10_train)
    Fisher_model('train_test','testing',np_data10_test,avg10_vector,sct10_matrix,dir10_vector,best10_threshold)
    

    data_trait5=['2','whole']
    data_trait6=['10','whole']
    data2_whole=select_data(data_trait5,folder_content)
    data10_whole=select_data(data_trait6,folder_content)
    np_data2_whole=classify_data(data2_whole)
    np_data10_whole=classify_data(data10_whole)
    np_data2_splitted=K_fold_splitter(np_data2_whole,10)
    np_data10_splitted=K_fold_splitter(np_data10_whole,10)
    avg2_vector,dir2_vector,sct2_matrix,best2_threshold=Fisher_model('cross_val',np_data2_splitted,10,best2_threshold)
    avg10_vector,dir10_vector,sct10_matrix,best10_threshold=Fisher_model('cross_val',np_data10_splitted,10,best10_threshold)
    boundary_plotter('Fisher',np_data2_whole,dir2_vector,best2_threshold,'Cross-Validation')


    
    


