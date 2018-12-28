import os
import sys
from FileReader import read_folder
from DataProcessing import classify_data
from DataProcessing import select_data
from DataProcessing import normalize_data
from PerceptronModel import Perceptron
from CrossValidation import K_fold_splitter
from DataPlotter import boundary_plotter

if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='original_data'
    folder_content=read_folder(folder_directory,folder_name)

    # P4 Perceptron model of classification
    
    data_trait1=['2','train']
    data_trait2=['2','test']
    data2_train=select_data(data_trait1,folder_content)
    data2_test=select_data(data_trait2,folder_content)
    np_data2_train=classify_data(data2_train)
    np_data2_test=classify_data(data2_test)
    GD2_weight,inv2_weight=Perceptron('train_test',np_data2_train,np_data2_test)
    #boundary_plotter('perceptron',np_data2_train,GD2_weight,inv2_weight,'Training')
    boundary_plotter('perceptron',np_data2_test,GD2_weight,inv2_weight,'Testing')

    data_trait3=['10','train']
    data_trait4=['10','test']
    data10_train=select_data(data_trait3,folder_content)
    data10_test=select_data(data_trait4,folder_content)
    np_data10_train=classify_data(data10_train)
    np_data10_test=classify_data(data10_test)
    GD10_weight,inv10_weight=Perceptron('train_test',np_data10_train,np_data10_test)

    data_trait5=['2','whole']
    data_trait6=['10','whole']
    data2_whole=select_data(data_trait5,folder_content)
    data10_whole=select_data(data_trait6,folder_content)
    np_data2_whole=classify_data(data2_whole)
    np_data10_whole=classify_data(data10_whole)
    np_data2_splitted=K_fold_splitter(np_data2_whole,10)
    np_data10_splitted=K_fold_splitter(np_data10_whole,10)
    GD2_weight,inv2_weight=Perceptron('cross_val',np_data2_splitted,10)
    GD10_weight,inv10_weight=Perceptron('cross_val',np_data10_splitted,10)
    boundary_plotter('perceptron',np_data2_whole,GD2_weight,inv2_weight,'Cross-validation')
