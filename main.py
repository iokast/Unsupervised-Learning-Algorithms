# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 18:56:03 2018

@author: Brandon
"""

import pandas as pd
import numpy as np
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import myGMM,nn_arch,nn_lr, run_NN, appendClusterDimKM, appendClusterDimGMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from helpers import pairwiseDistCorr,reconstructionError, ImportanceSelect
from sklearn.random_projection import SparseRandomProjection
from itertools import product
from sklearn.ensemble import RandomForestClassifier

#%% Setup
np.random.seed(0)
clusters =  [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,25,30]
dims_digits = [2,5,10,15,20,25,30,35,40,45,50,55,60,64]
dims_seg = range(2,20)

# load digits training set      
digits = pd.read_hdf('datasets.hdf','digits_original')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

# load segmentation training set  
seg = pd.read_hdf('datasets.hdf','segmentation_original')     
segX = seg.drop('Class',1).copy().values
segY = seg['Class'].copy().values
le = preprocessing.LabelEncoder()
segY = le.fit_transform(segY)

digitsX = StandardScaler().fit_transform(digitsX)
segX= StandardScaler().fit_transform(segX)

#%% Part 1 - Run k-means and EM clustering algorithms on original datasets

print('Part 1 - Running clustering algoirthms on original datasets...')
SSE = defaultdict(dict)
BIC = defaultdict(dict)
homo = defaultdict(lambda: defaultdict(dict))
compl = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(digitsX)
    gmm.fit(digitsX)
    SSE[k]['Digits SSE'] = km.score(digitsX)
    BIC[k]['Digits BIC'] = gmm.bic(digitsX)
    homo[k]['Digits']['Kmeans'] = homogeneity_score(digitsY,km.predict(digitsX))
    homo[k]['Digits']['GMM'] = homogeneity_score(digitsY,gmm.predict(digitsX))
    compl[k]['Digits']['Kmeans'] = completeness_score(digitsY,km.predict(digitsX))
    compl[k]['Digits']['GMM'] = completeness_score(digitsY,gmm.predict(digitsX))
    adjMI[k]['Digits']['Kmeans'] = ami(digitsY,km.predict(digitsX))
    adjMI[k]['Digits']['GMM'] = ami(digitsY,gmm.predict(digitsX))
    
    km.fit(segX)
    gmm.fit(segX)
    SSE[k]['Segmentation'] = km.score(segX)
    BIC[k]['Segmentation BIC'] = gmm.bic(segX)
    homo[k]['Segmentation']['Kmeans'] = homogeneity_score(segY,km.predict(segX))
    homo[k]['Segmentation']['GMM'] = homogeneity_score(segY,gmm.predict(segX))
    compl[k]['Segmentation']['Kmeans'] = completeness_score(segY,km.predict(segX))
    compl[k]['Segmentation']['GMM'] = completeness_score(segY,gmm.predict(segX))
    adjMI[k]['Segmentation']['Kmeans'] = ami(segY,km.predict(segX))
    adjMI[k]['Segmentation']['GMM'] = ami(segY,gmm.predict(segX))
    print(k, clock()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
#SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
BIC = pd.DataFrame(BIC).T
#ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
homo = pd.Panel(homo)
compl = pd.Panel(compl)
adjMI = pd.Panel(adjMI)


SSE.to_csv('./P1_Clustering_Algorithms_Original/Cluster_Select_Kmeans.csv')
BIC.to_csv('./P1_Clustering_Algorithms_Original/Cluster_Select_GMM.csv')
homo.ix[:,:,'Segmentation'].to_csv('./P1_Clustering_Algorithms_Original/seg_homo.csv')
homo.ix[:,:,'Digits'].to_csv('./P1_Clustering_Algorithms_Original/digits_homo.csv')
compl.ix[:,:,'Segmentation'].to_csv('./P1_Clustering_Algorithms_Original/seg_compl.csv')
compl.ix[:,:,'Digits'].to_csv('./P1_Clustering_Algorithms_Original/digits_compl.csv')
adjMI.ix[:,:,'Segmentation'].to_csv('./P1_Clustering_Algorithms_Original/seg_adjMI.csv')
adjMI.ix[:,:,'Digits'].to_csv('./P1_Clustering_Algorithms_Original/digits_adjMI.csv')

#%% Part 2A & 4A - Run Dimensionality Reduction Algorithm PCA, Run NN with reduced dims

print('Part 2A - Starting PCA for digits dataset...')
pca = PCA(random_state=5)
pca.fit(digitsX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,65))
tmp.to_csv('./P2_Dimensionality_Reduction/digits_PCA_explained_variance_ratio.csv')

print('Part2A - Starting PCA for segmentation dataset...')
pca = PCA(random_state=5)
pca.fit(segX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,20))
tmp.to_csv('./P2_Dimensionality_Reduction/seg_PCA_explained_variance_ratio.csv')

# Run Neural Networks
pca = PCA(random_state=5)  
nn_results = run_NN(dims_digits, pca, digitsX, digitsY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/digits_PCA_nn_results.csv')

pca = PCA(random_state=5)    
nn_results = run_NN(dims_seg, pca, segX, segY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/seg_PCA_nn_results.csv')

#%% Part 2B & 4B - Run Dimensionality Reduction Algorithm ICA, Run NN with reduced dims

print('Part 2B & 4B - Starting ICA for digits dataset...')
ica = FastICA(random_state=5)
kurt = {}
svm = {}
for dim in dims_digits:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(digitsX)
    tmp = pd.DataFrame(tmp)
    tmp2 = tmp.kurt(axis=0)
    kurt[dim] = tmp2.abs().mean()
  
kurt = pd.Series(kurt) 
kurt.to_csv('./P2_Dimensionality_Reduction/digits_ICA_kurtosis.csv')

print('Part 2B - Starting ICA for segmentation dataset...')
ica = FastICA(random_state=5)
kurt = {}
for dim in dims_seg:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(segX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv('./P2_Dimensionality_Reduction/seg_ICA_kurtosis.csv')

# Run Neural Networks
ica = FastICA(random_state=5)  
nn_results = run_NN(dims_digits, ica, digitsX, digitsY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/digits_ICA_nn_results.csv')

ica = FastICA(random_state=5)  
nn_results = run_NN(dims_seg, ica, segX, segY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/seg_ICA_nn_results.csv')

#%% Part 2C & 4C - Run Dimensionality Reduction Algorithm RP, Run NN with reduced dims

print('Part 2C - Starting RP, pairwise distance correlation, for digits dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_digits):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(digitsX), digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/digits_RP_pairwise_distance_corr.csv')

print('Part 2C - Starting RP, pairwise distance correlation, for segmentation dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_seg):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(segX), segX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/seg_RP_pairwise_distance_corr.csv')

print('Part 2C - Starting RP, reconstruction error, for digits dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_digits):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(digitsX)    
    tmp[dim][i] = reconstructionError(rp, digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/digits_RP_reconstruction_error.csv')

print('Part 2C - Starting RP, reconstruction error, for segmentation dataset...')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims_seg):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(segX)  
    tmp[dim][i] = reconstructionError(rp, segX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('./P2_Dimensionality_Reduction/seg_RP_reconstruction_error.csv')

# Run Neural Networks
rp = SparseRandomProjection(random_state=5) 
nn_results = run_NN(dims_digits, rp, digitsX, digitsY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/digits_RP_nn_results.csv')

rp = SparseRandomProjection(random_state=5) 
nn_results = run_NN(dims_seg, rp, segX, segY)     
nn_results.to_csv('./P4_Neural_Networks_Reduced/seg_RP_nn_results.csv')

#%% Part 2D & 4D - Run Dimensionality Reduction Algorithm RF, Run NN with reduced dims

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

print('Part 2D - Starting RF for digits dataset...')
fs_digits = rfc.fit(digitsX,digitsY).feature_importances_ 
print('Part 2D - Starting RF for segmentation dataset...')
fs_seg = rfc.fit(segX,segY).feature_importances_ 

tmp = pd.Series(np.sort(fs_digits)[::-1])
tmp.to_csv('./P2_Dimensionality_Reduction/digits_RF_feature_importance.csv')

tmp = pd.Series(np.sort(fs_seg)[::-1])
tmp.to_csv('./P2_Dimensionality_Reduction/seg_RF_feature_importance.csv')

# Run Neural Networks
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims_digits,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}  
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
gs.fit(digitsX,digitsY)
nn_results = pd.DataFrame(gs.cv_results_)
nn_results.to_csv('./P4_Neural_Networks_Reduced/digits_RF_nn_results.csv')

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims_seg,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}  
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
gs.fit(segX,segY)
nn_results = pd.DataFrame(gs.cv_results_)
nn_results.to_csv('./P4_Neural_Networks_Reduced/seg_RF_nn_results.csv')

#%% Part 2E - Run Dimensionality Reduction Algorithms to create dimension reduced datasets

# Best number of dimensions chosen for each algorithm in Part 1 of analysis doc
dim_digits_PCA = 20 
dim_digits_ICA = 20 
dim_digits_RP = 25 
dim_digits_RF = 35 
dim_seg_PCA = 13 
dim_seg_ICA = 13 
dim_seg_RP = 8 
dim_seg_RF = 4

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

algo_name = ['PCA', 'ICA', 'RP', 'RF']
print('Part 2E - Storing dimensionally reduced datasets for each algorithm...')
filtr = ImportanceSelect(rfc,dim_digits_RF)
algos_digits = [PCA(n_components=dim_digits_PCA,random_state=10), 
           FastICA(n_components=dim_digits_ICA,random_state=10), 
           SparseRandomProjection(n_components=dim_digits_RP,random_state=5),
           ImportanceSelect(rfc,dim_digits_RF)]

filtr = ImportanceSelect(rfc,dim_seg_RF)
algos_seg = [PCA(n_components=dim_seg_PCA,random_state=10), 
           FastICA(n_components=dim_seg_ICA,random_state=10), 
           SparseRandomProjection(n_components=dim_seg_RP,random_state=5),
           ImportanceSelect(rfc,dim_seg_RF)]

for i in range(len(algos_digits)):
    if i == 3:
        digitsX2 = algos_digits[i].fit_transform(digitsX, digitsY)
    else:   
        digitsX2 = algos_digits[i].fit_transform(digitsX)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf('datasets.hdf','digits_'+algo_name[i],complib='blosc',complevel=9)

for i in range(len(algos_seg)):
    if i ==3:
        segX2 = algos_seg[i].fit_transform(segX, segY)
    else:
        segX2 = algos_seg[i].fit_transform(segX)
    seg2 = pd.DataFrame(np.hstack((segX2,np.atleast_2d(segY).T)))
    cols = list(range(seg2.shape[1]))
    cols[-1] = 'Class'
    seg2.columns = cols
    seg2.to_hdf('datasets.hdf','segmentation_'+algo_name[i],complib='blosc',complevel=9)
#    
#%% Part 3 - Run k-means and EM clustering algorithms on each dimensionally reduced dataset

print('Part 3 - Running clustering algoirthms on dimensionally reduced datasets...')
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
    
    SSE = defaultdict(dict)
    BIC = defaultdict(dict)
    homo = defaultdict(lambda: defaultdict(dict))
    compl = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    
    st = clock()
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(digitsX)
        gmm.fit(digitsX)
        SSE[k]['Digits SSE'] = km.score(digitsX)
        BIC[k]['Digits BIC'] = gmm.bic(digitsX)
        homo[k]['Digits']['Kmeans'] = homogeneity_score(digitsY,km.predict(digitsX))
        homo[k]['Digits']['GMM'] = homogeneity_score(digitsY,gmm.predict(digitsX))
        compl[k]['Digits']['Kmeans'] = completeness_score(digitsY,km.predict(digitsX))
        compl[k]['Digits']['GMM'] = completeness_score(digitsY,gmm.predict(digitsX))
        adjMI[k]['Digits']['Kmeans'] = ami(digitsY,km.predict(digitsX))
        adjMI[k]['Digits']['GMM'] = ami(digitsY,gmm.predict(digitsX))
        
        km.fit(segX)
        gmm.fit(segX)
        SSE[k]['Segmentation'] = km.score(segX)
        BIC[k]['Segmentation BIC'] = gmm.bic(segX)
        homo[k]['Segmentation']['Kmeans'] = homogeneity_score(segY,km.predict(segX))
        homo[k]['Segmentation']['GMM'] = homogeneity_score(segY,gmm.predict(segX))
        compl[k]['Segmentation']['Kmeans'] = completeness_score(segY,km.predict(segX))
        compl[k]['Segmentation']['GMM'] = completeness_score(segY,gmm.predict(segX))
        adjMI[k]['Segmentation']['Kmeans'] = ami(segY,km.predict(segX))
        adjMI[k]['Segmentation']['GMM'] = ami(segY,gmm.predict(segX))
        print(k, clock()-st)
        
        
    SSE = (-pd.DataFrame(SSE)).T
    BIC = pd.DataFrame(BIC).T
    homo = pd.Panel(homo)
    compl = pd.Panel(compl)
    adjMI = pd.Panel(adjMI)
    
    SSE.to_csv('./P3_Clustering_Algorithms_Reduced/SSE_'+algo_name[i]+'.csv')
    BIC.to_csv('./P3_Clustering_Algorithms_Reduced/BIC_'+algo_name[i]+'.csv')
    homo.ix[:,:,'Segmentation'].to_csv('./P3_Clustering_Algorithms_Reduced/seg_'+algo_name[i]+'_homo.csv')
    homo.ix[:,:,'Digits'].to_csv('./P3_Clustering_Algorithms_Reduced/digits_'+algo_name[i]+'_homo.csv')
    compl.ix[:,:,'Segmentation'].to_csv('./P3_Clustering_Algorithms_Reduced/seg_'+algo_name[i]+'_compl.csv')
    compl.ix[:,:,'Digits'].to_csv('./P3_Clustering_Algorithms_Reduced/digits_'+algo_name[i]+'_compl.csv')
    adjMI.ix[:,:,'Segmentation'].to_csv('./P3_Clustering_Algorithms_Reduced/seg_'+algo_name[i]+'_adjMI.csv')
    adjMI.ix[:,:,'Digits'].to_csv('./P3_Clustering_Algorithms_Reduced/digits_'+algo_name[i]+'_adjMI.csv')

#%% Part 5 - Rerun neural network learner with dimensionally reduced segmentation dataset with additional cluster feature
    
print('Part 5 - Running neural network with dimensionally reduced Segmentation dataset...')

# Run NN on original dataset without cluster dimension for comparison
grid ={'learning_rate_init':nn_lr,'hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
gs = GridSearchCV(mlp,grid,verbose=10,cv=5)
gs.fit(segX,segY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/seg_original.csv')

algo_name.append('original')

# Run NN on dimensionally reduced and original datasets with addition cluster dimension
for i in range(len(algo_name)):
    #for i in range(4,5):
    # load datasets      
    seg = pd.read_hdf('datasets.hdf','segmentation_'+algo_name[i])    
    segX = seg.drop('Class',1).copy().values
    segY = seg['Class'].copy().values
     
    km = kmeans(random_state=5)
    gmm = myGMM(random_state=5)

    grid ={'addClustKM__n_clusters':clusters,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('addClustKM',appendClusterDimKM(cluster_algo = km)),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(segX,segY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/seg_km_'+algo_name[i]+'.csv')
    
    grid ={'addClustGMM__n_clusters':clusters,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('addClustGMM',appendClusterDimGMM(cluster_algo = gmm)),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(segX,segY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('./P5_Neural_Networks_Reduced_With_Clusters/seg_gmm_'+algo_name[i]+'.csv')
