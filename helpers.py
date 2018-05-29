# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:47:56 2017

@author: jtay
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

nn_arch= [(25,),(25,25),(50),(50,50),(100,25,100)]
nn_lr = [.001, .006, .01, .06, .1, .6, 1]

def run_NN(dims, clf, X, Y):
    grid ={'clf__n_components':dims,'NN__learning_rate_init':nn_lr,'NN__hidden_layer_sizes':nn_arch}      
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('clf',clf),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    gs.fit(X, Y)
    return pd.DataFrame(gs.cv_results_)

	
class appendClusterDimKM(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_algo, n_clusters = 8):
        self.cluster_algo = cluster_algo
        self.n_clusters = n_clusters
    def transform(self, X, *_):
        self.cluster_algo.set_params(n_clusters=self.n_clusters)
        self.cluster_algo.fit(X)
        returned_instances = pd.DataFrame(np.hstack((X,np.atleast_2d(self.cluster_algo.predict(X)).T)))
        return returned_instances
    def fit(self, *_):
        return self
  
  
class appendClusterDimGMM(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_algo, n_clusters = 2):
        self.cluster_algo = cluster_algo
        self.n_clusters = n_clusters
        
    def transform(self, X, *_):
        self.cluster_algo.set_params(n_components=self.n_clusters)
        self.cluster_algo.fit(X)
        returned_instances = pd.DataFrame(np.hstack((X,np.atleast_2d(self.cluster_algo.predict(X)).T)))
        return returned_instances
    
    def fit(self, *_):
        return self

		
class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)
        
        
def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

  
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)
    
	
# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn          
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
                                   