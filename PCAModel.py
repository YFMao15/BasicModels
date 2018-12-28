import os
import sys
import numpy as np
from sklearn.decomposition import pca

def PCA_function(np_data,*args):
    n_components=args[0]
    np_all_data=np.hstack((np_data['E3'],np_data['E5']))
    PCA_result=pca.PCA(n_components=n_components)
    PCA_result.fit(np_all_data.T)
    print("The covariance of all features in training data is:")
    print(PCA_result.get_covariance())
    print('\nThe covarience of %d selected values:' % (n_components))
    print(PCA_result.explained_variance_)
    print("\nThe covariance ratio of %d selected values:" % (n_components))
    print(PCA_result.explained_variance_ratio_)
    print('\nThe covariance of noise:')
    print(PCA_result.noise_variance_)
    return PCA_result
