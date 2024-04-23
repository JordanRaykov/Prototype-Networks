# -*- coding: utf-8 -*-

"""
This is a function with the necessary routines to train, evaluate and do predictions 
out of sample using two-layer prototypical network. 

Arguments: train_mode = {train, test, predict}
           proto_select = {fixed; data_driven}
                          'fixed'-value argument allows to directly upload 
                          the values of the basis parameters, consistent 
                          with the selected basis type.
                          'data_driven'-value arguement tries to infer the values 
                          of the basis parameters using Bayesian nonparametric
                          clustering. This is done using the MAP-DP algorithm 
                          from Raykov et al., 2016. What to do when K-means clustering fails: 
                              a simple yet principled alternative algorithm. PloS one, 11(9), 
                              p.e0162259.
                          The kernel function of the MAP-DP algorithm is selected
                          to match the properties of the selected basis type.
          h - integer-value specifying the number of assumed prototypes
          basis_params - an array containing the basis parameters of proto_select = 'fixed'
                          was chosen
          X - input features using for training or out-of-sample predictions
          Y - labels used for training during the training or evaluation depending 
              on the selected value for 
                          
  
@author: Yordan P. Raykov, Luc JW. Evers
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from utils import tpsrbf
from utils import compute_phi
import random
import numpy.matlib

from utis import mapdp_nw


import pickle
import math
from scipy.spatial.distance import cdist
from scipy.special import xlogy

from sklearn import preprocessing


from scipy.special import gammaln
from numpy.linalg import slogdet

from numpy.matlib import repmat
from numpy.linalg import inv


import scipy as sp
from sklearn.decomposition import PCA
from sklearn import datasets
import pickle

from scipy.spatial.distance import cdist
from scipy.special import xlogy

from sklearn import preprocessing

from utils import tpsrbf
from utils import compute_phi
#import GPy
    
def compute_grad(function, X, W, C, N, class_labels, nl, l, d, Beta = None):
    
    T = X[nl].shape[0]
    grad = X[nl] - class_labels
    for k in np.arange(nl-1,l-1,-1):
        if function == 'zlogz':
            z = cdist(X[k], C[k], 'euclidean')
            G = -np.multiply(np.dot(grad, W[k].T), np.log(z) + 1)
            grad = np.zeros([T, d[k-1]])
            for ind1 in range(T):
                grad[ind1,:] = np.dot(G[ind1,:].reshape(1,-1), (np.dot(np.ones([N[k],1]), X[k][ind1,:].reshape(1,-1)) - C[k]))
        elif function == 'gaussian':
            G = -np.multiply(np.dot(grad, W[k].T), compute_phi('gaussian',X[k], C[k], Beta[k]))#tpsrbf(X[k], C[k], Beta[k]))
            grad = np.zeros([T, d[k-1]])
            for ind1 in range(T):
                grad[ind1,:] = np.dot(G[ind1,:].reshape(1,-1), (np.dot(np.ones([N[k],1]), X[k][ind1,:].reshape(1,-1)) - C[k]))
    grad[np.isinf(grad)] = 0
    return grad

from sklearn import preprocessing

def CrossEntropyLoss(yHat, y):
    T = y.shape[0]
    K = y.shape[1]
    E = 0

    for t in range(T):
        for k in range(K):
            E = E - y[t,k]*math.log2(yHat[t,k])
    return E

def NS(X, W, phi, phi_inv, niter):
#Inputs:
#    X: T x D matrix of the layers input data
#    W: dictionary of the wieghts at each layer
#    phi: output of layer before weight multiplication
#    niter: maximum number of training iterations
    
    T = X.shape[0]
    d = W.shape[1]
    
    Y = np.dot(phi,W)
    
    Dx = sp.spatial.distance.cdist(X, X)
    Dy = sp.spatial.distance.cdist(Y, Y)
    
    error = 0.5*np.linalg.norm((Dx - Dy)/(Dx + (Dx == 0)))
    
    eta = 1e-3
    kup = 2.5
    kdown = 0.1
    success = 1
    c = 0
        
    for epoch in range(niter):
        if success == 1:
            grad = np.zeros([T, d])
            for i in range(T):
                grad[i,:] = - np.dot(((Dx[i,:] - Dy[i, :])/((Dy[i, :] + (Dy[i, :] == 0))*(Dx[i, :] + (Dx[i, :] == 0)))), \
                    (Y[i,:] - Y))
           
        Y_new = Y - eta*grad
        W_new = np.dot(phi_inv, Y_new)
        Y_new = np.dot(phi, W_new)
        
        Dy = sp.spatial.distance.cdist(Y_new, Y_new)           
        error_new = 0.5*np.linalg.norm((Dx - Dy)/(Dx + (Dx == 0)))           
    
        if error > error_new:
            success = 1
            eta = eta*kup
            error = error_new
            W = W_new.copy()
            Y = Y_new.copy()
            c = 0
        else:
            success = 0
            eta = eta*kdown
            c += 1
            
        if c > 20:
            break
        
        print('Iter: ', epoch, 'Error: ', error)
        
    return Y, W
def MultiLayer_prototypical_NN(X_train, y_train, basis_params, X_test, y_test, train_mode, proto_select, hypers, basis_type):