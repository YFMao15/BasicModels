import os
import sys
import numpy as np
from sympy import Symbol
from sympy import expand
from sympy import solve
from scipy.stats import probplot 
from scipy import interpolate
from matplotlib import pyplot as plt

# plotting the results to show relevance with normal distribution
def quantile_plotter(np_data):
    sub1=plt.subplot(221)
    probplot(np_data['E3'][0],dist='norm',fit=True,plot=sub1)
    sub2=plt.subplot(222)
    probplot(np_data['E3'][1],dist='norm',fit=True,plot=sub2)
    sub3=plt.subplot(223)
    probplot(np_data['E5'][0],dist='norm',fit=True,plot=sub3)
    sub4=plt.subplot(224)
    probplot(np_data['E5'][1],dist='norm',fit=True,plot=sub4)
    plt.show()

def boundary_plotter(mode,np_data,*args):
    if mode=='Bayesian':
        cov_matrix=args[0]
        avg_vector=args[1]
        prior=args[2]
        keys=np_data.keys()
        inv_cov_matrix={}
        det_inv_cov_matrix={}
        for key in keys:
            temp_matrix=np.linalg.inv(cov_matrix[key])
            inv_cov_matrix[key]=temp_matrix
            det_inv_cov_matrix[key]=np.linalg.det(inv_cov_matrix[key])
    
        x=Symbol('x')
        y=Symbol('y')
        random_vector=np.array([x,y])
        # row vector
        temp_vector={}
        temp_vector['E3']=random_vector-avg_vector['E3']
        temp_vector['E5']=random_vector-avg_vector['E5']
        # column vector
        temp_vector_T={}
        temp_vector_T['E3']=temp_vector['E3'].T
        temp_vector_T['E5']=temp_vector['E5'].T
        conditional_E3=-np.dot(temp_vector['E3'],np.dot(inv_cov_matrix['E3'],temp_vector_T['E3']))/2   \
            -np.log(det_inv_cov_matrix['E3'])/2-np.log(2*np.pi)
        judge_E3=conditional_E3+np.log(prior['E3'])
        conditional_E5=-np.dot(temp_vector['E5'],np.dot(inv_cov_matrix['E5'],temp_vector_T['E5']))/2   \
            -np.log(det_inv_cov_matrix['E5'])/2-np.log(2*np.pi)
        judge_E5=conditional_E5+np.log(prior['E5'])
        function=expand(judge_E3)-expand(judge_E5)
        #print('The judging function of E3 is:\n')
        #print(expand(judge_E3))
        #print('\nThe judging function of E5 is:\n')
        #print(expand(judge_E5))
        #print('\nThe function of classifying boundary is:\n')
        #print(function)

        x_range=[]
        y_range=[]
        for x_num in np.arange(-2,5,0.05):
            for y_num in np.arange(-2,5,0.05):
                result=float(function.evalf(subs={x:x_num,y:y_num}))
                if abs(result)<0.03:
                    x_range.append(x_num)
                    y_range.append(y_num)
        # use interpolation to smooth the boundary
        plt.ion()
        # plot in different colors
        sub1=plt.subplot(111)
        sub1.set_title('%s sets classification under %s' % (args[3],args[4]))
        sub1.plot(x_range,y_range,label='boundary',color='red')
        sub1.scatter(np_data['E3'][0],np_data['E3'][1],label='E3 point',color='blue',marker='o',s=60)
        sub1.scatter(np_data['E5'][0],np_data['E5'][1],label='E5 point',color='green',marker='o',s=60)
        plt.pause(10)
        plt.close()


    elif mode=='Fisher':
        direction_vector=args[0]
        best_threshold=args[1]
        x=Symbol('x')
        y=Symbol('y')
        random_vector=np.array([x,y])
        function=np.dot(direction_vector,random_vector)-best_threshold
        result=solve(function,y)
        x_range=range(0,8000,100)
        y_range=[]
        for x_num in x_range:
            y_num=float(result[0].evalf(subs={x:x_num}))
            y_range.append(y_num)

        plt.ion()
        # plot in different colors
        sub1=plt.subplot(121)
        sub1.set_title('%s sets classification under %s model in logarithm ' % (args[2],mode))
        sub1.plot(np.log10(x_range),np.log10(y_range),label='boundary',color='red')
        sub1.scatter(np.log10(np_data['E3'][0]),np.log10(np_data['E3'][1]),label='E3 point',color='blue',marker='o',s=40)
        sub1.scatter(np.log10(np_data['E5'][0]),np.log10(np_data['E5'][1]),label='E5 point',color='green',marker='o',s=40)
        sub2=plt.subplot(122)
        sub2.set_title('%s sets classification under %s model in linearity ' % (args[2],mode))
        sub2.plot(x_range,y_range,label='boundary',color='red')
        sub2.scatter(np_data['E3'][0],np_data['E3'][1],label='E3 point',color='blue',marker='o',s=40)
        sub2.scatter(np_data['E5'][0],np_data['E5'][1],label='E5 point',color='green',marker='o',s=40)
        plt.pause(10)
        plt.close()

    elif mode=='perceptron':
        GD_weight=args[0]
        inv_weight=args[1]
        x=Symbol('x')
        y=Symbol('y')
        random_vector=np.array([1,x,y])
        GD_function=np.dot(GD_weight.T,random_vector)
        inv_function=np.dot(inv_weight.T,random_vector)
        GD_result=solve(GD_function,y)
        inv_result=solve(inv_function,y)
        x_range=range(0,8000,100)
        GD_y_range=[]
        inv_y_range=[]
        for x_num in x_range:
            GD_y_num=float(GD_result[y].evalf(subs={x:x_num}))
            inv_y_num=float(inv_result[y].evalf(subs={x:x_num}))
            GD_y_range.append(GD_y_num)
            inv_y_range.append(inv_y_num)

        plt.ion()
        # plot in different colors
        sub1=plt.subplot(121)
        sub1.set_title('%s sets classification under %s model in logarithm ' % (args[2],mode))
        sub1.plot(np.log10(x_range),np.log10(GD_y_range),label='GD_boundary',color='red')
        sub1.plot(np.log10(x_range),np.log10(inv_y_range),label='inv_boundary',color='black')
        sub1.scatter(np.log10(np_data['E3'][0]),np.log10(np_data['E3'][1]),label='E3 point',color='blue',marker='o',s=40)
        sub1.scatter(np.log10(np_data['E5'][0]),np.log10(np_data['E5'][1]),label='E5 point',color='green',marker='o',s=40)
        sub2=plt.subplot(122)
        sub2.set_title('%s sets classification under %s model in linearity ' % (args[2],mode))
        sub2.plot(x_range,GD_y_range,label='GD_boundary',color='red')
        sub2.plot(x_range,inv_y_range,label='inv_boundary',color='black')
        sub2.scatter(np_data['E3'][0],np_data['E3'][1],label='E3 point',color='blue',marker='o',s=40)
        sub2.scatter(np_data['E5'][0],np_data['E5'][1],label='E5 point',color='green',marker='o',s=40)
        plt.pause(10)
        plt.close()

    elif mode=='SVM':
        clf=args[0]
        x=Symbol('x')
        y=Symbol('y')
        random_vector=np.array([x,y])
        x_range=range(0,8000,100)
        y_range=[]
        function=np.dot(clf.coef_,random_vector)+clf.intercept_
        print('\nThe function of SVM hyper-plane is:')
        print(function)
        result=solve(function,y)
        for x_num in x_range:
            y_num=float(result[y].evalf(subs={x:x_num}))
            y_range.append(y_num)
            
        plt.ion()
        # plot in different colors
        sub1=plt.subplot(121)
        sub1.set_title('%s sets classification under %s model in logarithm ' % (args[1],mode))
        sub1.plot(np.log10(x_range),np.log10(y_range),label='boundary',color='red')
        sub1.scatter(np.log10(np_data['E3'][0]),np.log10(np_data['E3'][1]),label='E3 point',color='blue',marker='o',s=40)
        sub1.scatter(np.log10(np_data['E5'][0]),np.log10(np_data['E5'][1]),label='E5 point',color='green',marker='o',s=40)
        sub2=plt.subplot(122)
        sub2.set_title('%s sets classification under %s model in linearity ' % (args[1],mode))
        sub2.plot(x_range,y_range,label='boundary',color='red')
        sub2.scatter(np_data['E3'][0],np_data['E3'][1],label='E3 point',color='blue',marker='o',s=40)
        sub2.scatter(np_data['E5'][0],np_data['E5'][1],label='E5 point',color='green',marker='o',s=40)
        plt.pause(10)
        plt.close()

    elif mode=='neuron':

        plt.ion()
        # plot in different colors
        sub1=plt.subplot(111)
        sub1.set_title('%s sets classification under %s model in logarithm ' % (args[2],mode))
        sub1.plot(np.log10(x_range),np.log10(GD_y_range),label='GD_boundary',color='red')
        sub1.plot(np.log10(x_range),np.log10(inv_y_range),label='inv_boundary',color='black')
        sub1.scatter(np.log10(np_data['E3'][0]),np.log10(np_data['E3'][1]),label='E3 point',color='blue',marker='o',s=40)
        sub1.scatter(np.log10(np_data['E5'][0]),np.log10(np_data['E5'][1]),label='E5 point',color='green',marker='o',s=40)
        sub2=plt.subplot(122)
        sub2.set_title('%s sets classification under %s model in linearity ' % (args[2],mode))
        sub2.plot(x_range,GD_y_range,label='GD_boundary',color='red')
        sub2.plot(x_range,inv_y_range,label='inv_boundary',color='black')
        sub2.scatter(np_data['E3'][0],np_data['E3'][1],label='E3 point',color='blue',marker='o',s=40)
        sub2.scatter(np_data['E5'][0],np_data['E5'][1],label='E5 point',color='green',marker='o',s=40)
        plt.pause(10)
        plt.close()