import os
import sys
import numpy as np

# mean vector calculation in one specific class for Bayesian model
def mean_vector(np_data):
    keys=np_data.keys()
    mean_vector={}
    for key in keys:
        temp_vector=np.mean(np_data[key],axis=1)
        mean_vector[key]=temp_vector
    return mean_vector

def scatter_matrix(np_data):
    keys=np_data.keys()
    scatter_matrix={}
    avg_vector=mean_vector(np_data)
    for key in keys:
        row_num=len(np_data[key][0])
        temp_data=np_data[key].T
        scatter_matrix[key]=np.zeros(shape=np.dot(avg_vector[key],avg_vector[key].T).shape)
        for count in range(row_num):
            temp_sample=temp_data[count]
            # reshape the np.array before transpostion
            temp_sample.shape=(len(temp_sample),1)
            temp_sample_T=temp_sample.reshape((len(temp_sample),1))
            temp_sub_matrix=np.dot(temp_sample_T-avg_vector[key].T,temp_sample-avg_vector[key])
            scatter_matrix[key]=scatter_matrix[key]+temp_sub_matrix
    return scatter_matrix


def Fisher_model(mode,*args):
    if mode=='train_test':
        if args[0]=='training':
        # for training sets
            np_data=args[1]
            avg_vector=mean_vector(np_data)
            sct_matrix=scatter_matrix(np_data)
            value=list(avg_vector.values())
            diff_vector=value[0]-value[1]
            sum_matrix=sum(sct_matrix.values())
            direction_vector=np.dot(sum_matrix,diff_vector)
            thresholds=[]
            avg_E3=np.dot(direction_vector,avg_vector['E3'])
            avg_E5=np.dot(direction_vector,avg_vector['E5'])
            keys=np_data.keys()
            print('The number of chracteristics is %d\n' % avg_vector[list(keys)[0]].shape[0])
            min_false_rate=1.0
            for temp_a in range(1,20):
                thresholds.append(((20-temp_a)*avg_E3+temp_a*avg_E5)/20)
            for threshold in thresholds:
                false_num=0.0
                sample_num=0.0
                for key in keys:
                    temp_data=np_data[key].T
                    row_num=temp_data.shape[0]
                    sample_num=sample_num+row_num
                    for count in range(row_num):
                        judge=np.dot(direction_vector,temp_data[count])-threshold
                        if (judge<0) & (key=='E3'):
                            false_num=false_num+1
                        elif (judge>0) & (key=='E5'):
                            false_num=false_num+1
                false_rate=false_num/sample_num
                if false_rate<min_false_rate:
                    min_false_rate=false_rate
                    best_threshold=threshold
            print('The minimal false rate of %s set is: %.3f \n' % (args[0],min_false_rate))
            print('The corresponding threshold is %.3f\n\n' % best_threshold)
            return avg_vector,direction_vector,sct_matrix,best_threshold

        elif args[0]=='testing':
            # for testing sets
            np_data=args[1]
            avg_vector=args[2]
            sct_matrix=args[3]
            direction_vector=args[4]
            best_threshold=args[5]
            keys=np_data.keys()
            min_false_rate=1.0
            false_num=0.0
            sample_num=0.0
            for key in keys:
                temp_data=np_data[key].T
                row_num=temp_data.shape[0]
                sample_num=sample_num+row_num
                for count in range(row_num):
                    judge=np.dot(direction_vector,temp_data[count])-best_threshold
                    if (judge<0) & (key=='E3'):
                        false_num=false_num+1
                    elif (judge>0) & (key=='E5'):
                        false_num=false_num+1
            false_rate=false_num/sample_num
            print('Under the same threshold as training sets\n')
            print('The false rate of %s set is: %.3f \n' % (args[0],false_rate))
        
    elif mode=='cross_val':
        np_data=args[0]
        fold_num=args[1]
        best_threshold=args[2]
        print('Cross validation for dataset.\n')
        print('The dataset contains %d characteristics.' % (np_data['E3']['train'][0].shape[0]))
        keys=np_data.keys()
        np_train_data={}
        np_test_data={}
        for curr_fold in range(fold_num):
            false_num=0.0
            sample_num=0.0
            np_train_data={}
            np_test_data={}
            for key in keys:
                np_train_data[key]=np_data[key]['train'][curr_fold]
                np_test_data[key]=np_data[key]['test'][curr_fold].T
                sample_num=np_data[key]['test'][curr_fold].shape[1]
            avg_vector=mean_vector(np_train_data)
            sct_matrix=scatter_matrix(np_train_data)
            value=list(avg_vector.values())
            diff_vector=value[0]-value[1]
            sum_matrix=sum(sct_matrix.values())
            direction_vector=np.dot(sum_matrix,diff_vector)
            for key in keys:
                for count in range(np_test_data[key].shape[0]):
                    judge=np.dot(direction_vector,np_test_data[key][count])-best_threshold
                    if (judge<0) & (key=='E3'):
                        false_num=false_num+1
                    elif (judge>0) & (key=='E5'):
                        false_num=false_num+1
            false_rate=false_num/sample_num
            print('Iteration %d, current false rate %.3f' % (curr_fold+1,false_rate))
        return avg_vector,direction_vector,sct_matrix,best_threshold
        
