# -*- coding: utf-8 -*-
"""
This is function with the necessary routines to train, evaluate and do predictions 
out of sample using single-layer prototypical network. 

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
import tensorflow
from sklearn.gaussian_process.kernels import PairwiseKernel
import keras
from keras import utils as np_utils
#from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from utils import mapdp_nw


def Layer_prototypical_NN(X_train, y_train, basis_params, X_test, y_test, train_mode, proto_select, basis_type, num_proto_cat, num_basis, training_method = 'pseudo-inverse', hypers = None):

    if proto_select == 'fixed':
        if basis_type == 'gaussian':
            C = basis_params[0]
            Beta = basis_params[1]
        else:
            C = basis_params[0]
    elif proto_select == 'inferred':
    # Set hyper-parameters for the MAP-DP clustering
        N0 = hypers.N0
        m0 = hypers.m0 #X[i].mean(1)[:, None]    # Normal-Wishart prior mean
        a0 = hypers.a0 #100             # Normal-Wishart prior scale
        c0 = hypers.c0 # 10/float(10000)    # Normal-Wishart prior degrees of freedom
        B0 = hypers.B0 # np.diag(1/(0.05*X[i].var(1)))  # Normal-Wishart prior precision
        # Run MAPDP to convergence
        mu_basis, z, K, E = mapdp_nw(X_train.T, N0, m0, a0, c0, B0)
        sigma_basis = np.shape(mu_basis)[0]
        for k in np.unique(z):
            sigma_basis = np.hstack((sigma_basis, np.std(X_train[k,:].T)))
        C = mu_basis
        Beta = sigma_basis    
        
    if training_method == 'pseudo-inverse':
        ## Compute the non-linearity Phi for each basis 
        phi = compute_phi(X_train.T, C, Beta)
        phi[np.isnan(phi)] = 0 # null any nan Phi elements
        
        ## Compute the weights of the neural network via pseudo-inverse
        W = np.zeros([C.shape[0], 2])
        T = len(y_train)
        score_cat = np.zeros([T, 2])
        for c in range(2):
            y_c = (y_train == c)
            W[:,c] = np.matmul(np.linalg.pinv(np.matmul(phi.T,phi)), np.matmul(phi.T, y_c))
        
        if train_mode == 'train':
            return W, phi, C, Beta
        
        if train_mode == 'test' or train_mode == 'predict':
            phi_test = compute_phi(X_test.T, C, Beta)
            phi_test[np.isnan(phi_test)] = 0
        
            T = len(y_test)
            score_cat = np.zeros([T, 2])
            for c in range(2):
                score_cat[:,c] = np.matmul(W[:,c].reshape(-1,+1).T, phi_test.T)
        
            Y_hat_test = np.zeros([T,1])
            for i in range(T):
                regularizer = np.sum(score_cat[i,:])
                if regularizer<=0:
                    score_cat[i,0] = random.uniform(0, 1)
                    score_cat[i,1] = 1 - score_cat[i,0]
                    regularizer = 1
                score_cat[i,0] = score_cat[i,0]/regularizer
                score_cat[i,1] = score_cat[i,1]/regularizer
                if score_cat[i,0] > score_cat[i,1]:
                    Y_hat_test[i] = 0
                else:
                    Y_hat_test[i] = 1
        
            prob_predictions = score_cat
            predictions = Y_hat_test  
            if train_mode == 'test':
                ACC = np.sum(predictions == y_test)/len(y_test)
                TPR = np.sum(predictions == 1 and y_test == 1)/len(y_test)
                TNR = np.sum(predictions == 0 and y_test == 0)/len(y_test)
                print('Evaluation metrics:')
                print('Accuracy: ', ACC)
                print('Sensitivity: ', TPR)
                print('Specificity: ', TNR)
                return predictions, prob_predictions, y_test
    elif training_method == 'GP-mode':
        
        d = np.shape(X_train)[1]
        y_trainn = np.tiles(np.arange(1, num_proto_cat), num_basis)
        y_trainn = keras.utils.np_utils.to_categorical(y_trainn, num_proto_cat*num_basis)
        
        kernel = PairwiseKernel(metric='rbf') 
        rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)
        
        #y_train is intput to the softmax layer
        y_train = keras.utils.np_utils.to_categorical(y_train, 2) # Tremor and non-tremor class
        train_rbf = rbf_model.predict(X_train)
        #                      PERCEPTRONS LAYERS
        batch_size = 128
        epochs = 10
        img_size = d
        size = len(C)
        model = Sequential()
        model.add(Dense(img_size, activation='relu', input_shape=(size,)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        nadam=tensorflow.keras.optimizers.Nadam(lr=0.005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=nadam,
                      metrics=['accuracy'])
        #                      TRAINING THE MODEL
        history = model.fit(train_rbf, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1)
        train_prediction_probabilities = model.predict(train_rbf)
        train_prediction = np.argmax(train_prediction_probabilities, axis=1)
        
        if train_mode == 'train':
            return train_prediction, train_prediction_probabilities, y_train
        elif train_mode == 'predict':
            test_rbf = rbf_model.predict(X_test)
            test_prediction_probabilities = model.predict(test_rbf)
            test_prediction = np.argmax(test_prediction_probabilities, axis=1)
            return test_prediction, test_prediction_probabilities, y_test
        elif train_mode == 'test':
            test_rbf = rbf_model.predict(X_test)
            test_prediction_probabilities = model.predict(test_rbf)
            test_prediction = np.argmax(test_prediction_probabilities, axis=1)
            ACC = np.sum(predictions == y_test)/len(y_test)
            TPR = np.sum(predictions == 1 and y_test == 1)/len(y_test)
            TNR = np.sum(predictions == 0 and y_test == 0)/len(y_test)
            print('Evaluation metrics:')
            print('Accuracy: ', ACC)
            print('Sensitivity: ', TPR)
            print('Specificity: ', TNR)
            return predictions, prob_predictions, y_test

 

    