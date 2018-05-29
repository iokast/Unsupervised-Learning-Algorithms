# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:00:54 2018

@author: Brandon
"""
import pandas as pd
import numpy as np
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import myGMM
from sklearn.neural_network import MLPClassifier



algo_name =         ['original','PCA',  'ICA',  'RP',   'RF']
algo_km_clusters =  [20,        2,      13,     4,      2]
algo_gmm_clusters = [6,         2,      2,      2,      2]

clusters =  [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,25,30]
dims_seg = range(2,20)

time = defaultdict(dict)
for i in range(len(algo_name)):
    # load datasets      
    digits = pd.read_hdf('datasets.hdf','digits_'+algo_name[i]) 
    digitsX = digits.drop('Class',1).copy().values
    digitsY = digits['Class'].copy().values
    
    seg = pd.read_hdf('datasets.hdf','segmentation_'+algo_name[i])    
    segX = seg.drop('Class',1).copy().values
    segY = seg['Class'].copy().values
    
    digitsX = StandardScaler().fit_transform(digitsX)
    segX= StandardScaler().fit_transform(segX)
    
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    
    for k in clusters:
        print(algo_name[i],k)
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        
        st = clock()
        km.fit(digitsX)
        time[k]['Digits km '+ algo_name[i]] = st - clock()
        
        st = clock()
        gmm.fit(digitsX)
        time[k]['Digits gmm '+ algo_name[i]] = st - clock()

        st = clock()
        km.fit(segX)
        time[k]['Seg km '+ algo_name[i]] = st - clock()
        
        st = clock()
        gmm.fit(segX)
        time[k]['Seg gmm '+ algo_name[i]] = st - clock()
        
time = (-pd.DataFrame(time)).T
time.to_csv('./P3_Clustering_Algorithms_Reduced/time_to_cluster.csv')

time = defaultdict(dict)
for i in range(len(algo_name)):
    print(algo_name[i])
    
    # load datasets        
    seg = pd.read_hdf('datasets.hdf','segmentation_'+algo_name[i])    
    segX = seg.drop('Class',1).copy().values
    segY = seg['Class'].copy().values
     
    km = kmeans(random_state=5)
    gmm = myGMM(random_state=5)
    km.set_params(n_clusters=algo_km_clusters[i])
    gmm.set_params(n_components=algo_gmm_clusters[i])
    km.fit(segX)
    gmm.fit(segX)
    segX_km = pd.DataFrame(np.hstack((segX,np.atleast_2d(km.predict(segX)).T)))
    segX_gmm = pd.DataFrame(np.hstack((segX,np.atleast_2d(gmm.predict(segX)).T)))
    
    segX = StandardScaler().fit_transform(segX)
    segX_km = StandardScaler().fit_transform(segX_km)
    segX_gmm = StandardScaler().fit_transform(segX_gmm)
    
    mlp = MLPClassifier(max_iter=2000,early_stopping=False,random_state=5,learning_rate_init = .01, hidden_layer_sizes = (50,50))
    st = clock()
    mlp.fit(segX,segY)
    time['No clusters'][algo_name[i]] = clock() - st
    
    mlp = MLPClassifier(max_iter=2000,early_stopping=False,random_state=5,learning_rate_init = .01, hidden_layer_sizes = (50,50))
    st = clock()
    mlp.fit(segX_km,segY)
    time['K-means'][algo_name[i]] = clock() - st
    
    mlp = MLPClassifier(max_iter=2000,early_stopping=False,random_state=5,learning_rate_init = .01, hidden_layer_sizes = (50,50))
    st = clock()
    mlp.fit(segX_gmm,segY)
    time['GMM'][algo_name[i]] = clock() - st

time = pd.DataFrame(time).T
time.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/time_to_run_NN.csv')