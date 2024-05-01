# -*- coding: utf-8 -*-
"""
This is module with the necessary functions to train, evaluate and do predictions 
out of sample using prototypical neural networks (NN). Key functions include Layer_prototypical_NN and 
MultiLayer_prototypical_NN: Layer_prototypical_NN implements single layer prototypical NN using different 
modes; MultiLayer_prototypical_NN implements a non-standard module for extendeding multiple layers 
of prototypical NNs, based on intermediate dimensionality reduction. 

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
        import tqdm
        import math
        import torch
        import gpytorch
        from matplotlib import pyplot as plt
        from torch.utils.data import TensorDataset, DataLoader
        import urllib.request
        import os
        from scipy.io import loadmat
        from math import floor
        from gpytorch.models import ApproximateGP
        from gpytorch.variational import CholeskyVariationalDistribution
        from gpytorch.variational import VariationalStrategy


        d = np.shape(X_train)[1]
        y_trainn = np.tile(np.arange(1, num_proto_cat + 1), num_basis)
        y_trainn = np_utils.to_categorical(y_trainn, num_proto_cat*num_basis)
        
        kernel = PairwiseKernel(metric='rbf') 
        rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)

        C_tensor = torch.tensor(C)
        y_trainn = torch.tensor(y_trainn)
        
        train_n = int(floor(0.8 * len(C_tensor)))
        
        train_x = C_tensor[:train_n, :].contiguous()
        train_y = y_trainn[:train_n].contiguous()
        
        if torch.cuda.is_available():
            train_x, train_y = train_x.cuda(), train_y.cuda()
        
        
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        class GPModel(ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
                variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
                super(GPModel, self).__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)    
            
        inducing_points = train_x[:500, :]
        rbf_model = GPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if torch.cuda.is_available():
            rbf_model = rbf_model.cuda()
            likelihood = likelihood.cuda()
            
        num_epochs = 4
        rbf_model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': rbf_model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, rbf_model, num_data=train_y.size(0))
        
        epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = rbf_model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
         
        #y_train is intput to the softmax layer
        
        test_dataset = TensorDataset(train_x, train_y)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        
        rbf_model.eval()
        likelihood.eval()
        mean_pred = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = rbf_model(x_batch)
                mean_pred = torch.cat([mean_pred, preds.mean.cpu()])
                
        y_train = np_utils.to_categorical(y_train, 2) # Tremor and non-tremor class
        train_rbf = mean_pred#rbf_model.predict(X_train)
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
            
            test_dataset = TensorDataset(X_test)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            rbf_model.eval()
            likelihood.eval()
            mean_pred = torch.tensor([0.])
            with torch.no_grad():
                for x_batch in test_loader:
                    preds = rbf_model(x_batch)
                    mean_pred = torch.cat([mean_pred, preds.mean.cpu()])
                    
                    
            test_rbf = rbf_model.predict(X_test)
            test_prediction_probabilities = model.predict(test_rbf)
            test_prediction = np.argmax(test_prediction_probabilities, axis=1)
           # test_prediction = model.predict(test_rbf)
            
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

#CRBFtrain(data, class_labels, nl, niter, eta_s, N, d, basis_shape = 'gaussian', basis_parameters = None, use_pca = True, use_gplvm = True, basis_learning = None, inducing_num = None):  
def MultiLayer_prototypical_NN(X_train, y_train, basis_params, X_test, y_test, train_mode, proto_select, basis_type, num_proto_cat, num_basis, training_method = 'pseudo-inverse', hypers = None):
    #Inputs:
    #    data: T x D matrix of the input data
    #    target: T x D_target matrix of the output targets
    #    nl: number of layers for the netwrok
    #    niter: number of iterations for the post training step
    #    eta: starting learning rate of the post training step
    #    N: a list of the number of centers at each layer of the RBF
    #    d: a list of number of the dimension of the data projection after each layer in the RBF
    #    use_pca: if True initializes each layer with PCA for pre-training
    #    basis_learning: learn basis parameters by either fixing them to domain informed values, random subselection or clustering 
      
    X = {}
    W = {}
    C_sum = {}
    C = {}
    Beta = {}
    
    X[0] = data
    T = X[0].shape[0]
    
    for i in range(nl):                
        if i < nl - 1:
            if proto_select == 'random':    
                if basis_type == 'zlogz':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                if basis_type == 'gaussian':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                    Beta[i] = np.random.uniform(0,1,(N[i],C[i].shape[1]))*epsilon
                if basis_type == 'invquad':
                    theta = 1e12; #global scaling parameter for inv quadratic basis function
                    fl_theta = 1e12
                    fl_theta2 = -2*fl_theta
                    theta2 = -2*theta; #for use with centres
                    C_sum[i] = (theta * np.sum(X[i][centres,:]**2, 1) + 1).reshape(1,-1)
                    C[i] = (theta2 * X[i][centres,:]).T
            elif proto_select == 'clustering':
                N0 = 3
                m0 = X[i].mean(1)[:, None]    # Normal-Wishart prior mean
                a0 = 100             # Normal-Wishart prior scale
                c0 = 10/float(10000)    # Normal-Wishart prior degrees of freedom
                B0 = np.diag(1/(0.05*X[i].var(1)))  # Normal-Wishart prior precision
                # # Run MAPDP to convergence
                mu_basis = np.zeros((d[i],1))
                sigma_basis = np.zeros((d[i],1))
                mu, z, K, E = mapdp_nw(X[i].T, N0, m0, a0, c0, B0)
                mu_basis = np.hstack((mu_basis, mu))
                for k in np.unique(z):
                    sigma_basis = np.hstack((sigma_basis, np.std(X[i][k,:].T)))
                C[i] = mu_basis
                Beta[i] = sigma_basis                
            elif basis_learning == 'fixed':
                basis_params
                C[i] = basis_params[i][0]
                if basis_type == 'gaussian':    
                    Beta[i] = basis_params[i][0]
            elif basis_learning == 'inducing':
                if use_gplvm:
                    if d[i] > X[i].shape[1]:
                        raise SyntaxError('use_gplvm should be set to false when projecting from lower dimensions to higher dimensions')
                m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(X[i], d[i], num_inducing=inducing_num, missing_data=True) # Consider inducing points selected using labels
                m.optimize(messages=1, max_iters=5e3)
                C[i] = np.array(m.inducing_inputs)
                print(np.shape(X[i]))
                print(np.shape(C[i]))
            ################################# Basis parameters are set ########

            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
            else:
                phi = compute_phi(basis_shape, X[i], C[i])
                
            if use_pca:
                if d[i] > X[i].shape[1]:
                    raise SyntaxError('use_pca should be set to false when projecting from lower dimensions to higher dimensions')
                X[i+1] = PCA(n_components = d[i]).fit_transform(X[i])
                phi_inv = np.linalg.pinv(phi)
                W[i] = np.dot(phi_inv, X[i+1])
                X[i+1] = np.dot(phi, W[i])
            else:
                phi_inv = np.linalg.pinv(phi)
                W[i] = np.random.randn(N[i],d[i])
                X[i+1] = np.dot(phi, W[i])

            print('Pretrain layer: ', i)
            X[i+1], W[i] = NS(X[i], W[i], phi, phi_inv, 400)
                   
        elif i == nl - 1:
            l = 1
            if basis_type =='zlogz':
                phi = compute_phi(basis_shape, X[0], C[0])
            if basis_type == 'gaussian':
                phi = compute_phi(basis_shape, X[0], C[0], Beta[0])
                        
            W[0] = np.dot(np.linalg.pinv(phi), X[l])
            X[l] = np.dot(phi, W[0])
            for ind2 in range(l,nl-1):
                if basis_type =='zlogz':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2])
                if basis_type == 'gaussian':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
                X[ind2+1] = np.dot(phi, W[ind2])
            
            ## Update centers for the following layer if they are not known
            
            if basis_learning == None:
              #  centres = np.random.choice(np.arange(T), N[i],replace = False)
              #  C[i] = (X[i][centres,:])
                if basis_type == 'zlogz':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                if basis_type == 'gaussian':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                    Beta[i] = Beta[i] = np.random.uniform(0,1,(N[i],C[i].shape[1]))*epsilon # Make basis proportionate to the variance 
                if basis_type == 'invquad':
                    theta = 1e12; #global scaling parameter for inv quadratic basis function
                    fl_theta = 1e12
                    fl_theta2 = -2*fl_theta
                    theta2 = -2*theta; #for use with centres
                    C_sum[i] = (theta * np.sum(X[i][centres,:]**2, 1) + 1).reshape(1,-1)
                    C[i] = (theta2 * X[i][centres,:]).T
                    
            if basis_learning == 'inducing':
                if basis_type == 'zlogz':
                    m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(X[i], X[i].shape[1], num_inducing=inducing_num, missing_data=True) # Consider inducing points selected using labels
                    m.optimize(messages=1, max_iters=5e3)
                    C[i] = np.array(m.inducing_inputs)
                #    centres = np.random.choice(np.arange(T), N[i], replace = False)
                 #   C[i] = (X[i][centres,:])
                
            
            if basis_type =='zlogz':
                print('final layer')
                print(np.shape(C[i]))
                print('shape of X final layer')
                print(np.shape(X[i]))
                phi = compute_phi(basis_shape, X[i], C[i])
            if basis_type == 'gaussian':
                phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
            #phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
            
            # Last layer needs a softmax (logistic link function) to map binary targets
            W[i] = np.dot(np.linalg.pinv(phi), class_labels)
            X[i+1] = np.dot(phi, W[i])
            
    print('pretraining done')
    
    k_up = 2.5; #learning increase rate (1.2 standard)
    k_down = 0.1; #learning decrease rate
    
    for l in range(1,nl):
        if basis_shape =='zlogz':
            phi = compute_phi(basis_shape, X[l-1], C[l-1])
        if basis_shape == 'gaussian':
            phi = compute_phi(basis_shape, X[l-1], C[l-1], Beta[l-1])   
        W[l-1] = np.dot(np.linalg.pinv(phi), X[l])
        X[l] = np.dot(phi, W[l-1])
        for ind2 in range(l,nl):
            if basis_shape =='zlogz':
                phi = compute_phi(basis_shape, X[ind2], C[ind2])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
            X[ind2+1] = np.dot(phi, W[ind2]) 
        print('shape of X[nl]')
        print(np.shape(X[nl]))
        X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
        print('shape of X[nl] after normalizing')
        print(np.shape(X[nl]))
        error = CrossEntropyLoss(X[nl], class_labels)
       # error = np.sqrt(np.sum((target - X[nl])**2))
        
        success = 1
        eta = eta_s
        for it in range(niter):
            if success:
                #grad = X[nl] - class_labels
                grad = compute_grad(basis_shape, X, W, C, N, class_labels, nl, l, d, Beta) 
                
            #    for k in np.arange(nl-1,l-1,-1):
            #        G = -np.multiply(np.dot(grad, W[k].T), (compute_phi(X[k].T,C[k],Beta[k]))**2)
            #        grad = np.zeros([T, d[k-1]])
            #        for t in range(T):
            #            grad[t,:] = np.dot(G[t,:].reshape(1,-1),(np.dot(np.ones([N[k],1]),X[k][t,:].reshape(1,-1)) - C[k].T/theta2))##### WHAT TO put here

            X_old = X.copy()
            W_old = W.copy()
            X[l] = X[l] - eta*grad
            if basis_shape == 'zlogz':            
                phi = compute_phi(basis_shape, X[l-1],C[l-1])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[l-1], C[l-1], Beta[l-1])
            
            W[l-1] = np.dot(np.linalg.pinv(phi), X[l])
            X[l] = np.dot(phi, W[l-1])
            for ind2 in range(l,nl):
                if basis_shape =='zlogz':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2])
                if basis_shape == 'gaussian':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
                X[ind2+1] = np.dot(phi, W[ind2])                        
            
            X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
            error_new = CrossEntropyLoss(X[nl], class_labels)
                                    
            if error > error_new:
                success = 1
                eta = eta*k_up
                error = error_new
            else:
                success = 0
                X = X_old.copy()
                W = W_old.copy()
                eta = eta*k_down
               
            if not it %5:
                print(error_new)
                print(eta)
                print('Layer: ', l, 'Iteration: ', it, 'Error: ', error)
    
    l = nl - 1
    if basis_shape == 'zlogz':
        phi = compute_phi(basis_shape, X[l], C[l])        
    if basis_shape == 'gaussian':
        phi = compute_phi(basis_shape, X[l], C[l], Beta[l])   
    W[l] = np.dot(np.linalg.pinv(phi), class_labels)
    X[nl] = np.dot(phi, W[l])
    X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
    error_new = CrossEntropyLoss(X[nl], class_labels)
    print('Final Error: ', error_new)
    
    return X, W, C, Beta, N
 
def compute_phi(function, X, C, Beta = None, C_sum = None, N = None, theta = None):
    
    #C_sum, N, and theta only needed if function == 'invquad'
    if function == 'gaussian':
        X = X.T
        T = X.shape[1]
        h = C.shape[0]
        d = C.shape[1]
        phi = np.zeros([T, h]) 
        for i in range(h):
            phi[:,i] = tpsrbf(X, C[i,:].reshape(d,1), Beta[i,:].reshape(d,1), ax = 0)
    
    if function == 'invquad':
        T = X.shape[0]
        phi = 1/(theta * np.dot(np.sum(X**2, 1).reshape(-1,1), np.ones([1,N])) + \
         np.dot(X,C) + np.dot(np.ones([T,1]), C_sum))
        
    elif function == 'z2logz':
        z = cdist(X, C, 'euclidean')
        phi = xlogy(z**2,z)
        
    elif function == 'zlogz':
        z = cdist(X, C, 'euclidean')
        phi = xlogy(z,z)
    return phi

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
    