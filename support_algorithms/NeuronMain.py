import os
import sys
from FileReader import read_folder
from DataProcessing import classify_data
from DataProcessing import select_data
from DataProcessing import normalize_data
from CrossValidation import K_fold_splitter
from NeuronModel import neuron_model

if __name__=='__main__':
    folder_directory=os.path.dirname(__file__)
    folder_name='original_data'
    folder_content=read_folder(folder_directory,folder_name)

    # P2, Multiple layer perceptron implementation
    data_trait1=['2','train']
    data_trait2=['2','test']
    data2_train=select_data(data_trait1,folder_content)
    data2_test=select_data(data_trait2,folder_content)

    normed_train_data2,scale_data2=normalize_data(data2_train)
    normed_test_data2,scale_data2=normalize_data(data2_test,scale_data2)
    np_normed_train_data2=classify_data(normed_train_data2)
    np_normed_test_data2=classify_data(normed_test_data2)
    # the last dimension of the layer_dims should be the number of classes in the classification 
    batch_size=np_normed_test_data2['E3'].shape[1]+np_normed_test_data2['E5'].shape[1]
    neuron_model('train_test',np_normed_train_data2,np_normed_test_data2,
        max_step=2000,layer_dims=(8,4,2),learning_rate=0.001,batch_size=batch_size)


    data_trait3=['10','train']
    data_trait4=['10','test']
    data10_train=select_data(data_trait3,folder_content)
    data10_test=select_data(data_trait4,folder_content)
    normed_train_data10,scale_data10=normalize_data(data10_train)
    normed_test_data10,scale_data10=normalize_data(data10_test,scale_data10)
    np_normed_train_data10=classify_data(normed_train_data10)
    np_normed_test_data10=classify_data(normed_test_data10)
    batch_size=np_normed_test_data10['E3'].shape[1]+np_normed_test_data10['E5'].shape[1]
    neuron_model('train_test',np_normed_train_data10,np_normed_test_data10,
        max_step=2000,layer_dims=(8,4,2),learning_rate=0.001,batch_size=batch_size)
