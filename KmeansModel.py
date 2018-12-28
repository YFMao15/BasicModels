import os
import sys
import numpy as np
from sklearn.cluster import KMeans


def feature_projection(np_data,PCA_result):
    keys=np_data.keys()
    np_trans_data={}
    for key in keys:
        np_trans_data[key]=np.dot(PCA_result.components_,np_data[key])
    return np_trans_data


def Kmeans_clustering(mode,np_trans_data,*args):
    if mode=='train':
        np_all_trans_data=np.hstack((np_trans_data['E3'],np_trans_data['E5']))
        k_means=KMeans(n_clusters=2,max_iter=1000)
        k_means.fit(np_all_trans_data.T)
        return k_means
    elif mode=='test':
        k_means=args[0]
        np_all_trans_data=np.hstack((np_trans_data['E3'],np_trans_data['E5']))
        labels=k_means.predict(np_all_trans_data)
        return labels