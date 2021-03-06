import os
import sys
import numpy as np
from DataProcessing import select_data
from DataProcessing import classify_data
from DataProcessing import normalize_data
from FileReader import read_folder
from PCAModel import PCA_function

if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='E3E5_data'
    folder_content=read_folder(folder_directory,folder_name)

    data_trait1=['10','train']
    data_trait2=['10','test']
    data10_train=select_data(data_trait1,folder_content)
    data10_test=select_data(data_trait2,folder_content)
    
    normed_train_data10,normed_scale=normalize_data(data10_train)
    normed_test_data10,normed_scale=normalize_data(data10_test,normed_scale)
    np_normed_train_data10=classify_data(normed_train_data10)
    np_normed_test_data10=classify_data(normed_test_data10)

    n_component=3
    PCA_result=PCA_function(np_normed_train_data10,n_component)
    PCA_function(np_normed_test_data10,n_component)
