import os
import sys
from FileReader import read_folder
from DataPlotter import boundary_plotter
from DataProcessing import classify_data
from DataProcessing import select_data
from CrossValidation import K_fold_splitter
from SVMModel import singular_vector_machine

if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='original_data'
    folder_content=read_folder(folder_directory,folder_name)

    # P1, SVM implementation
    data_trait1=['2','train']
    data_trait2=['2','test']
    data2_train=select_data(data_trait1,folder_content)
    data2_test=select_data(data_trait2,folder_content)

    np_train_data2=classify_data(data2_train)
    np_test_data2=classify_data(data2_test)
    clf2_model=singular_vector_machine('train_test',np_train_data2,np_test_data2)
    boundary_plotter('SVM',np_test_data2,clf2_model,'Testing')

    data_trait3=['10','train']
    data_trait4=['10','test']
    data10_train=select_data(data_trait3,folder_content)
    data10_test=select_data(data_trait4,folder_content)
    np_train_data10=classify_data(data10_train)
    np_test_data10=classify_data(data10_test)
    clf10_model=singular_vector_machine('train_test',np_train_data10,np_test_data10)

    data_trait5=['2','whole']
    data_trait6=['10','whole']
    whole_data2=select_data(data_trait5,folder_content)
    whole_data10=select_data(data_trait6,folder_content)
    np_whole_data2=classify_data(whole_data2)
    np_whole_data10=classify_data(whole_data10)
    np_splitted_data2=K_fold_splitter(np_whole_data2,10)
    np_splitted_data10=K_fold_splitter(np_whole_data10,10)
    clf2_model=singular_vector_machine('cross_val',np_splitted_data2,10)
    boundary_plotter('SVM',np_test_data2,clf2_model,'Validation')
    clf10_model=singular_vector_machine('cross_val',np_splitted_data10,10)
