import os
import sys
import numpy as np
from sklearn import svm

def singular_vector_machine(mode,*args):
    if mode=='train_test':
        np_train_data=args[0]
        np_test_data=args[1]
        keys=list(np_train_data.keys())
        keys_num=len(keys)
        train_class=np.ones((np_train_data[keys[0]].shape[1]))
        test_class=np.ones((np_test_data[keys[0]].shape[1]))
        train_data=[]
        test_data=[]
        for key_num in range(1,keys_num):
            train_class=np.hstack((train_class,(key_num+1)*np.ones((np_train_data[keys[key_num]].shape[1]))))
            test_class=np.hstack((test_class,(key_num+1)*np.ones((np_test_data[keys[key_num]].shape[1]))))
        for key in keys:
            train_num=np_train_data[key].shape[1]
            test_num=np_test_data[key].shape[1]
            for num in range(train_num):
                train_data.append(list((np_train_data[key].T)[num]))
            for num in range(test_num):
                test_data.append(list((np_test_data[key].T)[num]))
        train_data=np.array(train_data)
        test_data=np.array(test_data)

        clf=svm.LinearSVC(C=1.0,tol=1e-5,max_iter=10000)
        clf.fit(train_data,train_class)
        false_num=0.0
        sample_num=0.0
        for key_num in range(keys_num):
            temp_data=np_train_data[keys[key_num]].T
            temp_length=len(temp_data)
            sample_num=sample_num+temp_length
            for num in range(temp_length):
                predict_num=clf.predict(temp_data[num].reshape(1,np_train_data[keys[0]].shape[0]))
                if predict_num[0]!=key_num+1:
                    false_num=false_num+1
        false_rate=false_num/sample_num
        print('The SVM model of %d characteristics:' % np_train_data[keys[0]].shape[0])
        print('The false rate of training sets is: %.3f' % false_rate)

        false_num=0.0
        sample_num=0.0
        for key_num in range(keys_num):
            temp_data=np_test_data[keys[key_num]].T
            temp_length=len(temp_data)
            sample_num=sample_num+temp_length
            for num in range(temp_length):
                predict_num=clf.predict(temp_data[num].reshape(1,np_train_data[keys[0]].shape[0]))
                if predict_num[0]!=key_num+1:
                    false_num=false_num+1
        false_rate=false_num/sample_num
        print('The false rate of testing sets is: %.3f\n' % false_rate)
        return clf

    elif mode=='cross_val':
        np_data=args[0]
        fold_num=10
        keys=np_data.keys()
        print('The SVM model of %d characteristics:' % np_data['E3']['train'][0].shape[0])
        for curr_fold in range(fold_num):
            false_num=0.0
            sample_num=0.0
            np_train_data={}
            np_test_data={}
            for key in keys:
                np_train_data[key]=np_data[key]['train'][curr_fold]
                np_test_data[key]=np_data[key]['test'][curr_fold].T
            keys=list(np_train_data.keys())
            keys_num=len(keys)
            train_class=np.ones((np_train_data[keys[0]].shape[1]))
            test_class=np.ones((np_test_data[keys[0]].shape[1]))
            train_data=[]
            test_data=[]
            for key_num in range(1,keys_num):
                train_class=np.hstack((train_class,(key_num+1)*np.ones((np_train_data[keys[key_num]].shape[1]))))
                test_class=np.hstack((test_class,(key_num+1)*np.ones((np_test_data[keys[key_num]].shape[1]))))
            for key in keys:
                train_num=np_train_data[key].shape[1]
                test_num=np_test_data[key].shape[1]
                for num in range(train_num):
                    train_data.append(list((np_train_data[key].T)[num]))
                for num in range(test_num):
                    test_data.append(list((np_test_data[key].T)[num]))
            train_data=np.array(train_data)
            test_data=np.array(test_data)

            clf=svm.LinearSVC(C=1.0,tol=1e-5,max_iter=10000)
            clf.fit(train_data,train_class)
            for key_num in range(keys_num):
                temp_data=np_test_data[keys[key_num]]
                temp_length=len(temp_data)
                sample_num=sample_num+temp_length
                for num in range(temp_length):
                    predict_num=clf.predict(temp_data[num].reshape(1,np_train_data[keys[0]].shape[0]))
                    if predict_num[0]!=key_num+1:
                        false_num=false_num+1
            false_rate=false_num/sample_num
            print('Current fold number is %d. The false rate of validation sets is: %.3f\n' % (curr_fold+1,false_rate))
        
        return clf

