import os
import sys
import numpy as np

# add one dimension to predict the threshold
def add_dimension(data):
    dimension=data.shape[1]
    widen_data=np.vstack((np.ones((1,dimension)),data))
    return widen_data

# aggregate all widen samples into one matrix
def sample_matrix(data):
    keys=list(data.keys())
    for key_num in range(len(keys)):
        if key_num==0:
            matrix=data[keys[key_num]].T
        else:
            matrix=np.vstack((matrix,data[keys[key_num]].T*(-1.0)))
    return matrix


# perceptron model
def Perceptron(mode,*args):
    if mode=='train_test':
        np_train_data=args[0]
        np_test_data=args[1]
        keys=np_train_data.keys()
        np_widen_train_data={}
        np_widen_test_data={}
        train_num=0
        for key in keys:
            temp_data=np_train_data[key]
            train_num=train_num+temp_data.shape[1]
            widen_data=add_dimension(temp_data)
            np_widen_train_data[key]=widen_data
            temp_data=np_test_data[key]
            widen_data=add_dimension(temp_data)
            np_widen_test_data[key]=widen_data

        # the first dimension is the class value from add_dimension function
        dimension=np_train_data[key].shape[0]+1
        print('\nThe training sets contain %d dimension of characteristics' % (dimension-1))
        #bias=np.ones((train_num,1))
        E3_num=np_train_data['E3'].shape[1]
        E5_num=np_train_data['E5'].shape[1]
        bias=np.hstack((train_num/E3_num*np.ones((E3_num,1)).T,train_num/E5_num*np.ones((E5_num,1)).T)).T
        GD_weight=np.random.random((dimension,1))
        matrix=sample_matrix(np_widen_train_data)
        learning_rate=0.1*np.ones((dimension,1))
        min_error=[]
        min_error.append(np.linalg.norm(np.dot(matrix,GD_weight)-bias))
        iter_num=0
        iteration_flag=True

        while iteration_flag: 
            partial_diff_value=np.zeros((dimension,1))
            iter_num=iter_num+1
            curr_error=min_error[-1] 
            for dim in range(dimension):
                temp_weight=np.zeros((dimension,1))
                partial_diff_value[dim]=np.dot((np.dot(matrix,GD_weight)-bias).T,matrix[:,dim].reshape(len(matrix[:,dim]),1))*2
                for curr_dim in range(dimension):
                    if curr_dim == dim:
                        temp_weight[curr_dim]=GD_weight[dim]-partial_diff_value[dim]*learning_rate[dim]
                    else:
                        temp_weight[curr_dim]=GD_weight[curr_dim]-0*learning_rate[curr_dim]
                temp_error=np.linalg.norm(np.dot(matrix,temp_weight)-bias)
                while temp_error>curr_error:
                    learning_rate[dim]=learning_rate[dim]*0.5
                    for curr_dim in range(dimension):
                        if curr_dim== dim:
                            temp_weight[curr_dim]=GD_weight[dim]-partial_diff_value[dim]*learning_rate[dim]
                        else:
                            temp_weight[curr_dim]=GD_weight[curr_dim]-0*learning_rate[curr_dim]
                    temp_error=np.linalg.norm(np.dot(matrix,temp_weight)-bias)
                curr_error=temp_error
                for curr_dim in range(dimension):
                    GD_weight[curr_dim]=temp_weight[curr_dim]-0
            learning_rate=0.1*np.ones((dimension,1))
            min_error.append(temp_error)
            print('Iteration %d, Minimal squared error %.3f' % (iter_num,min_error[-1]))
            if abs(min_error[-2]-min_error[-1])<0.001:
                iteration_flag=False

        inv_weight=np.dot(np.dot(np.linalg.inv(np.dot(matrix.T,matrix)),matrix.T),bias)
        print('The weight calculated by gradient descending is:')
        print(GD_weight)
        print('The weight calculated by inversion is:')
        print(inv_weight)

        GD_false_num=0.0
        inv_false_num=0.0
        test_num=0.0
        for key in keys:
            reduced_inv_weight=inv_weight[1:dimension]
            reduced_GD_weight=GD_weight[1:dimension]
            temp_data=np_train_data[key].T
            for count in range(temp_data.shape[0]):
                inv_judge=np.dot(reduced_inv_weight.T,temp_data[count].reshape(dimension-1,1))
                GD_judge=np.dot(reduced_GD_weight.T,temp_data[count].reshape(dimension-1,1))
                if (float(GD_judge+GD_weight[0])<0) & (key=='E3'):
                    GD_false_num=GD_false_num+1
                if (float(inv_judge+inv_weight[0])<0) & (key=='E3'):
                    inv_false_num=inv_false_num+1
                if (float(GD_judge+GD_weight[0])>0) & (key=='E5'):
                    GD_false_num=GD_false_num+1
                if (float(inv_judge+inv_weight[0])>0) & (key=='E5'):
                    inv_false_num=inv_false_num+1
        GD_false_rate=GD_false_num/train_num
        inv_false_rate=inv_false_num/train_num
        print('The false rate of gradient descending calcultion on training sets is %.3f' % (GD_false_rate))
        print('The false rate of inversion matrix calcultion on training sets is %.3f\n' % (inv_false_rate))

        GD_false_num=0.0
        inv_false_num=0.0
        for key in keys:
            reduced_inv_weight=inv_weight[1:dimension]
            reduced_GD_weight=GD_weight[1:dimension]
            temp_data=np_test_data[key].T
            test_num=test_num+temp_data.shape[0]
            for count in range(temp_data.shape[0]):
                inv_judge=np.dot(reduced_inv_weight.T,temp_data[count].reshape(dimension-1,1))
                GD_judge=np.dot(reduced_GD_weight.T,temp_data[count].reshape(dimension-1,1))
                if (float(GD_judge+GD_weight[0])<0) & (key=='E3'):
                    GD_false_num=GD_false_num+1
                if (float(inv_judge+inv_weight[0])<0) & (key=='E3'):
                    inv_false_num=inv_false_num+1
                if (float(GD_judge+GD_weight[0])>0) & (key=='E5'):
                    GD_false_num=GD_false_num+1
                if (float(inv_judge+inv_weight[0])>0) & (key=='E5'):
                    inv_false_num=inv_false_num+1
        GD_false_rate=GD_false_num/test_num
        inv_false_rate=inv_false_num/test_num
        print('The false rate of gradient descending calcultion on testing sets is %.3f' % (GD_false_rate))
        print('The false rate of inversion matrix calcultion on testing sets is %.3f' % (inv_false_rate))
        return GD_weight,inv_weight

    elif mode=='cross_val':
        np_data=args[0]
        fold_num=args[1]
        keys=np_data.keys()
        for curr_fold in range(fold_num):
            print('Fold %d' % curr_fold)
            np_train_data={}
            np_test_data={}
            for key in keys:
                np_train_data[key]=np_data[key]['train'][curr_fold]
                np_test_data[key]=np_data[key]['test'][curr_fold].T
            np_widen_train_data={}
            np_widen_test_data={}
            train_num=0
            for key in keys:
                temp_data=np_train_data[key]
                train_num=train_num+temp_data.shape[1]
                widen_data=add_dimension(temp_data)
                np_widen_train_data[key]=widen_data
                temp_data=np_test_data[key]
                widen_data=add_dimension(temp_data)
                np_widen_test_data[key]=widen_data

            # the first dimension is the class value from add_dimension function
            dimension=np_train_data[key].shape[0]+1
            #bias=np.ones((train_num,1))
            E3_num=np_train_data['E3'].shape[1]
            E5_num=np_train_data['E5'].shape[1]
            bias=np.hstack((train_num/E3_num*np.ones((E3_num,1)).T,train_num/E5_num*np.ones((E5_num,1)).T)).T
            GD_weight=np.random.random((dimension,1))
            matrix=sample_matrix(np_widen_train_data)
            learning_rate=0.1*np.ones((dimension,1))
            min_error=[]
            min_error.append(np.linalg.norm(np.dot(matrix,GD_weight)-bias))
            iter_num=0
            iteration_flag=True

            while iteration_flag: 
                partial_diff_value=np.zeros((dimension,1))
                iter_num=iter_num+1
                curr_error=min_error[-1] 
                for dim in range(dimension):
                    temp_weight=np.zeros((dimension,1))
                    partial_diff_value[dim]=np.dot((np.dot(matrix,GD_weight)-bias).T,matrix[:,dim].reshape(len(matrix[:,dim]),1))*2
                    for curr_dim in range(dimension):
                        if curr_dim == dim:
                            temp_weight[curr_dim]=GD_weight[dim]-partial_diff_value[dim]*learning_rate[dim]
                        else:
                            temp_weight[curr_dim]=GD_weight[curr_dim]-0*learning_rate[curr_dim]
                    temp_error=np.linalg.norm(np.dot(matrix,temp_weight)-bias)
                    while temp_error>curr_error:
                        learning_rate[dim]=learning_rate[dim]*0.5
                        for curr_dim in range(dimension):
                            if curr_dim== dim:
                                temp_weight[curr_dim]=GD_weight[dim]-partial_diff_value[dim]*learning_rate[dim]
                            else:
                                temp_weight[curr_dim]=GD_weight[curr_dim]-0*learning_rate[curr_dim]
                        temp_error=np.linalg.norm(np.dot(matrix,temp_weight)-bias)
                    curr_error=temp_error
                    for curr_dim in range(dimension):
                        GD_weight[curr_dim]=temp_weight[curr_dim]-0
                learning_rate=0.1*np.ones((dimension,1))
                min_error.append(temp_error)
                if abs(min_error[-2]-min_error[-1])<0.001:
                    iteration_flag=False
            print('Iteration %d, Minimal squared error %.3f' % (iter_num,min_error[-1]))

            inv_weight=np.dot(np.dot(np.linalg.inv(np.dot(matrix.T,matrix)),matrix.T),bias)
            #print('The weight calculated by gradient descending is:')
            #print(GD_weight)
            #print('The weight calculated by inversion is:')
            #print(inv_weight)

            GD_false_num=0.0
            inv_false_num=0.0
            test_num=0.0
            for key in keys:
                reduced_inv_weight=inv_weight[1:dimension]
                reduced_GD_weight=GD_weight[1:dimension]
                temp_data=np_test_data[key]
                test_num=test_num+temp_data.shape[0]
                for count in range(temp_data.shape[0]):
                    inv_judge=np.dot(reduced_inv_weight.T,temp_data[count].reshape(dimension-1,1))
                    GD_judge=np.dot(reduced_GD_weight.T,temp_data[count].reshape(dimension-1,1))
                    if (float(GD_judge+GD_weight[0])<0) & (key=='E3'):
                        GD_false_num=GD_false_num+1
                    if (float(inv_judge+inv_weight[0])<0) & (key=='E3'):
                        inv_false_num=inv_false_num+1
                    if (float(GD_judge+GD_weight[0])>0) & (key=='E5'):
                        GD_false_num=GD_false_num+1
                    if (float(inv_judge+inv_weight[0])>0) & (key=='E5'):
                        inv_false_num=inv_false_num+1
            GD_false_rate=GD_false_num/test_num
            inv_false_rate=inv_false_num/test_num
            print('The false rate of gradient descending calcultion on testing sets is %.3f' % (GD_false_rate))
            print('The false rate of inversion matrix calcultion on testing sets is %.3f' % (inv_false_rate))

        return GD_weight,inv_weight

