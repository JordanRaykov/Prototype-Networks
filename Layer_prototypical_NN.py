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

    X = data.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
    y = data[outcomevar] #This is the outcome variable
    X_train = np.array(X)
    y_train = np.array(y) #Outcome variable here
    activity_classes = np.unique(data[non_tremor_classes_var])
    
def Layer_prototypical_NN(X, Y .... X_test, Y_test
        data, outcomevar, protovar, non_tremor_classes_var, h, d):

    if proto_select == 'fixed':
        if basis_type == 'gaussian':
            C = basis_params[0]
            Beta = basis_params[1]
        else:
            C = basis_params[0]
    ## Compute the non-linearity Phi for each basis 
    phi = compute_phi(X_train.T, C, Beta)
    phi[np.isnan(phi)] = 0 # null any nan Phi elements

    ## Compute the weights of the neural network via pseudo-inverse
    W = np.zeros([C.shape[0], 2])
    T = len(y_train)
    score_cat = np.zeros([T, 2])
    for c in range(2):
        y_c = (y_train == c)
        W[:,c] = np.matmul(np.linalg.pinv(np.matmul(phi.T,phi)), np.matmul(phi.T,y_c))
        #score_cat[:,c] = np.matmul(W[:,c].reshape(-1,+1).T, phi.T)
        
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
        labels = y_test
        predictions = Y_hat_test    
        return predictions, prob_predictions, labels
        
    ## Approximate prototypical neural network with a Gaussian process plus softmax layer
    y_trainn = keras.utils.np_utils.to_categorical(y_trainn, h*(num_activity + proto_num))
    if basis_type == 'gaussian':
        kernel = PairwiseKernel(metric='rbf') 
        rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)
    else:
        kernel = PairwiseKernel(metric='rbf') 
        rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)
    # Y_hat = np.zeros([T,1])
    # for i in range(T):
    #     regularizer = np.sum(np.exp(score_cat[i,:]))
    #     score_cat[i,0] = np.exp(score_cat[i,0])/regularizer
    #     score_cat[i,1] = np.exp(score_cat[i,1])/regularizer
    #     if score_cat[i,0] > score_cat[i,1]:
    #         Y_hat[i] = 0
    #     else:
    #         Y_hat[i] = 1
    
    if train_mode == 'train':
        if basis_type == 'gaussian'
            return W, phi, C, Beta # trained network parameters
        else:
            return W, phi, C # trained network parameters

    if train_mode == 'test':
        ACC = np.sum(predictions == labels)/len(labels)
        TPR = np.sum(predictions == 1 and labels == 1)/len(labels)
        TNR = np.sum(predictions == 0 and labels == 0)/len(labels)
        return ACC, TPR, TNR
    if train_mode == 'predict':
        return predictions, prob_predictions, labels



d = 45
h = 100
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


predictions = []
prob_predictions = []
labels = []
sensitivity = []
specificity = []

mat_data = 'Y_features_PD_patients_2sec_200Hz_normalized_final.mat'
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['Y_tremor_features_200Hz_norm'];
data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])
data_features = data_features.dropna()

ind_proto_is = np.where(Y_features[:,47]>0)
ind_proto_is = np.array(ind_proto_is)
Y_features[ind_proto_is, 48] = 1 # Make all annotated prototypes to be tremor annotation

# data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])

X = data_features.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"])
Y = data_features["Non-tremor/Tremor"]

outcomevar = 'Non-tremor/Tremor'
outcomevar_id = 49
idcolumn = 'ID'
idcolumn_id = 1
protovar = "Prototype_ID"
activityvar = "Activity"

non_tremor_classes_var = "Non_tremor_activity_labels"

IDlist = {'1', '2', '3', '4', '5', '6', '7', '8'}

non_tremor_activity_label_1 = []
non_tremor_activity_label_2 = []
non_tremor_activity_label_3 = []
non_tremor_activity_label_4 = []
non_tremor_activity_label_5 = []
non_tremor_activity_label_6 = []
non_tremor_activity_label_7 = []

non_tremor_activity_label_1 = np.hstack((np.where((data_features["Activity_label"] == 3) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 3) & (data_features["Non-tremor/Tremor"] == 0)), np.where((data_features["Activity_label"] == 3) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_1 = np.hstack((non_tremor_activity_label_1,np.where((data_features["Activity_label"] == 4) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 4) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 4) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_1 = np.hstack((non_tremor_activity_label_1,np.where((data_features["Activity_label"] == 5.1) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 5.1) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 5.1) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_1 = np.hstack((non_tremor_activity_label_1,np.where((data_features["Activity_label"] == 5.2) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 5.2) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 5.2) & (data_features["Non-tremor/Tremor"] == 98))))

non_tremor_activity_label_2 = np.hstack((np.where((data_features["Activity_label"] == 7.1) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.1) & (data_features["Non-tremor/Tremor"] == 0)), np.where((data_features["Activity_label"] == 7.1) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.2) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.2) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.2) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.3) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.3) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.3) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.4) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.4) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.4) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.5) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.5) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.5) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.6) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.6) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.6) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.7) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.7) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.7) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_2 = np.hstack((non_tremor_activity_label_2,np.where((data_features["Activity_label"] == 7.8) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 7.8) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 7.8) & (data_features["Non-tremor/Tremor"] == 98))))

non_tremor_activity_label_3 = np.hstack((np.where((data_features["Activity_label"] == 8) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 8) & (data_features["Non-tremor/Tremor"] == 0)), np.where((data_features["Activity_label"] == 8) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_3 = np.hstack((non_tremor_activity_label_3,np.where((data_features["Activity_label"] == 10) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 10) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 10) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_3 = np.hstack((non_tremor_activity_label_3,np.where((data_features["Activity_label"] == 13) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 13) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 13) & (data_features["Non-tremor/Tremor"] == 98))))

non_tremor_activity_label_4 = np.hstack((np.where((data_features["Activity_label"] == 9) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 9) & (data_features["Non-tremor/Tremor"] == 0)), np.where((data_features["Activity_label"] == 9) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_4 = np.hstack((non_tremor_activity_label_4,np.where((data_features["Activity_label"] == 11) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 11) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 11) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_4 = np.hstack((non_tremor_activity_label_4,np.where((data_features["Activity_label"] == 12) & (data_features["Non-tremor/Tremor"] == 0)),  np.where((data_features["Activity_label"] == 12) & (data_features["Non-tremor/Tremor"] == 96)), np.where((data_features["Activity_label"] == 12) & (data_features["Non-tremor/Tremor"] == 98))))

non_tremor_activity_label_5 = np.where((data_features["Activity_label"] == 1) & (data_features["Non-tremor/Tremor"] == 98))
non_tremor_activity_label_5 = np.array(non_tremor_activity_label_5)
non_tremor_activity_label_5 = np.hstack((non_tremor_activity_label_5, np.where((data_features["Activity_label"] == 2) & (data_features["Non-tremor/Tremor"] == 98))))
non_tremor_activity_label_5 = np.hstack((non_tremor_activity_label_5, np.where((data_features["Activity_label"] == 6) & (data_features["Non-tremor/Tremor"] == 98))))

non_tremor_activity_label_6 = np.where((data_features["Activity_label"] == 1) & (data_features["Non-tremor/Tremor"] == 0))
non_tremor_activity_label_6 = np.array(non_tremor_activity_label_6)
non_tremor_activity_label_6 = np.hstack((non_tremor_activity_label_6,np.where((data_features["Activity_label"] == 2) & (data_features["Non-tremor/Tremor"] == 0))))
non_tremor_activity_label_6 = np.hstack((non_tremor_activity_label_6,np.where((data_features["Activity_label"] == 6) & (data_features["Non-tremor/Tremor"] == 0))))

non_tremor_activity_label_7 = np.where((data_features["Activity_label"] == 1) & (data_features["Non-tremor/Tremor"] == 96))
non_tremor_activity_label_7 = np.array(non_tremor_activity_label_7)
non_tremor_activity_label_7 = np.hstack((non_tremor_activity_label_7,np.where((data_features["Activity_label"] == 2) & (data_features["Non-tremor/Tremor"] == 96))))
non_tremor_activity_label_7 = np.hstack((non_tremor_activity_label_7,np.where((data_features["Activity_label"] == 6) & (data_features["Non-tremor/Tremor"] == 96))))

Non_tremor_activity_label = np.zeros((len(Y_features),1))
for n in non_tremor_activity_label_1[0]:
    Non_tremor_activity_label[n] = 1
for n in non_tremor_activity_label_2[0]:
    Non_tremor_activity_label[n] = 2
for n in non_tremor_activity_label_3[0]:
    Non_tremor_activity_label[n] = 3
for n in non_tremor_activity_label_4[0]:
    Non_tremor_activity_label[n] = 4
for n in non_tremor_activity_label_5[0]:
    Non_tremor_activity_label[n] = 5
for n in non_tremor_activity_label_6[0]:
    Non_tremor_activity_label[n] = 6
for n in non_tremor_activity_label_7[0]:
    Non_tremor_activity_label[n] = 7
    
Y_features = np.hstack((Y_features, Non_tremor_activity_label))

data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label","Non_tremor_activity_labels"])
data_features = data_features.dropna(axis=0)
Y_features = data_features.to_numpy()
data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label","Non_tremor_activity_labels"])

prob_predictions_out_of_sample_stacked = np.zeros([1,2])
labels_stacked = np.zeros([1,1])
prob_predictions_training_stacked = np.zeros([1,2])
TPR_out_of_sample = []
TPR_training = []
TNR_out_of_sample = []
TNR_training = []
global_threshold = []
AUC_out_of_sample = []
predictions_out_of_sample_stacked = np.zeros([1,1])
predictions_training_stacked = np.zeros([1,1])

auc_per_patient_training_stacked = np.zeros([1,1])
auc_per_patient_out_of_sample_stacked = np.zeros([1,1])

# Make the tremor classes binary
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 2] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 3] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 96] = 0
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 97] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 98] = 0
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 99] = 0

data_features = data_features.drop(columns=["Medication_Intake", "Activity_label"])


num_participants = -1
for i in range(48):
    if i != 24:        
        data_filtered = data_features[data_features[idcolumn] != i+1]
        data_cv = data_features[data_features[idcolumn] == i+1]     
       
        # Train data - all other people in dataframe
        data_train = data_filtered.drop(columns=idcolumn)      
        
        W, phi, C, Beta = RBF_network_proto_and_nontremor_train(data_train, outcomevar, protovar, non_tremor_classes_var, h, d)
        predictions_training, prob_predictions_training, labels_training = RBF_network_proto_and_nontremor_predict(data_train, outcomevar, protovar, non_tremor_classes_var, h, W, phi, C, Beta)

        
        thresholds = np.linspace(0.0,1.0,100)#[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.]
        for threshold in thresholds:          
            for t in range(len(prob_predictions_training)):
                if prob_predictions_training[t,0]>threshold:#threshold:
                    predictions_training[t] = 0
                else:
                    predictions_training[t] = 1
                    
            TP_training = 0
            P_training = 0
            TN_training = 0
            N_training = 0
            for t in range(len(predictions_training)):
                if y_train[t] == 1 and predictions_training[t] == 1:
                    TP_training = TP_training + 1
                if y_train[t] == 1:
                    P_training = P_training + 1
                if y_train[t] == 0 and predictions_training[t] == 0:
                    TN_training = TN_training + 1
                if y_train[t] == 0:
                    N_training = N_training + 1
            print(TN_training/N_training)
            
            if TN_training/N_training > 0.945 and TN_training/N_training < 0.955:
                global_threshold.append(threshold)
                TPR_training.append(TP_training/P_training)
                TNR_training.append(TN_training/N_training)
                predictions_training_stacked = np.vstack((predictions_training_stacked, predictions_training.reshape(-1,+1)))
                print(global_threshold)
                break
        
        pred, prob_predictions_out_of_sample, label = RBF_networkLOOCV_proto_and_nontremor(data_features, i+1, outcomevar, protovar, non_tremor_classes_var, idcolumn, h, d)
        prob_predictions.append(prob_predictions_out_of_sample)
        prob_predictions_out_of_sample_stacked = np.vstack((prob_predictions_out_of_sample_stacked, prob_predictions_out_of_sample))
        labels_array = label.array
        labels_array = labels_array.to_numpy()
        labels_stacked = np.vstack((labels_stacked, labels_array.reshape(-1,+1)))
        labels.append(label)
        predictions_out_of_sample = pred
        for t in range(len(prob_predictions_out_of_sample)):
            if prob_predictions_out_of_sample[t,0]>global_threshold[num_participants]:
                predictions_out_of_sample[t] = 0
            else:
                predictions_out_of_sample[t] = 1            
        predictions_out_of_sample_stacked = np.vstack((predictions_out_of_sample_stacked, predictions_out_of_sample.reshape(-1,+1)))
        TP_out_of_sample = 0
        P_out_of_sample = 0
        TN_out_of_sample = 0
        N_out_of_sample = 0
        for t in range(len(predictions_out_of_sample)):
            if labels_array[t] == 1 and predictions_out_of_sample[t] == 1:
                TP_out_of_sample = TP_out_of_sample + 1
            if labels_array[t] == 1:
                P_out_of_sample = P_out_of_sample + 1
            if labels_array[t] == 0 and predictions_out_of_sample[t] == 0:
                TN_out_of_sample = TN_out_of_sample + 1
            if labels_array[t] == 0:
                N_out_of_sample = N_out_of_sample + 1
        if P_out_of_sample>0:        
            TPR_out_of_sample.append(TP_out_of_sample/P_out_of_sample)
        else:
            TPR_out_of_sample.append(None)
        TNR_out_of_sample.append(TN_out_of_sample/N_out_of_sample)
        if len(np.unique(labels_array)) > 1:
            auc_per_patient_out_of_sample = roc_auc_score(label, prob_predictions_out_of_sample[:, 1])
        else:
            auc_per_patient_out_of_sample = 0
        AUC_out_of_sample.append(auc_per_patient_out_of_sample)
        
        ##Train in-sample and output parameters
        data_in_sample = data_features.drop(columns=["ID"])
        W_in_sample, phi_in_sample, C_in_sample, Beta_in_sample = RBF_network_proto_and_nontremor_train(data_in_sample, outcomevar, protovar, non_tremor_classes_var, h, d)
    
        idt = str(i)
        print('...' + idt + ' processing complete.')   
    
import pickle
with open('save_global_thresholds_single_prototype.pkl', 'wb') as f:
    pickle.dump(global_threshold, f)   
    
import pickle
with open('save_single_proto_trained_visits.pkl', 'wb') as f:
    pickle.dump(W_in_sample, f)
    pickle.dump(phi_in_sample, f)
    pickle.dump(C_in_sample, f)
    pickle.dump(Beta_in_sample, f)
         
with open('save_single_proto_trained_visits.pkl', 'rb') as f:
    W_in_sample_rd = pickle.load(f)
    phi_in_sample_rd = pickle.load(f)
    C_in_sample_rd = pickle.load(f)
    Beta_in_sample_rd = pickle.load(f)
    
    
auc_out_of_sample = roc_auc_score(labels_stacked, prob_predictions_out_of_sample_stacked[:, 1])

#Compute total sensitivity and specificity    

TP_out_of_sample = 0
P_out_of_sample = 0
TN_out_of_sample = 0
N_out_of_sample = 0
for t in range(len(predictions_out_of_sample_stacked)):
    if labels_stacked[t] == 1 and predictions_out_of_sample_stacked[t] == 1:
        TP_out_of_sample = TP_out_of_sample + 1
    if labels_stacked[t] == 1:
        P_out_of_sample  = P_out_of_sample  + 1
    if labels_stacked[t] == 0 and predictions_out_of_sample_stacked[t] == 0:
        TN_out_of_sample  = TN_out_of_sample  + 1
    if labels_stacked[t] == 0:
        N_out_of_sample  = N_out_of_sample  + 1
TPR_total_out_of_sample = TP_out_of_sample/P_out_of_sample
TNR_total_out_of_sample = TN_out_of_sample/N_out_of_sample    
    


#######################################################

from __future__ import print_function
from scipy.special import gammaln
from numpy.linalg import slogdet
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import inv

def mapdp_nw(X, N0, m0, a0, c0, B0, epsilon=1e-6, maxiter=100, fDebug=False):
    '''
    MAP-DP for Normal-Wishart data.

    Inputs:  X  - DxN matrix of data
             N0 - prior count (DP concentration parameter)
             m0 - cluster prior mean
             a0 - cluster prior scale
             c0 - cluster prior degrees of freedom
             B0 - cluster prior precision (inverse covariance)
             epsilon - convergence tolerance
             maxiter - maximum number of iterations, overrides threshold calculation
             fDebug - extra verbose input of algorithm operation

    Outputs: mu - cluster centroids
             z  - data point cluster assignments
             K  - number of clusters
             E  - objective function value for each iteration

    CC BY-SA 3.0 Attribution-Sharealike 3.0, Max A. Little. If you use this
    code in your research, please cite:
    Yordan P. Raykov, Alexis Boukouvalas, Fahd Baig, Max A. Little (2016)
    "What to do when K-means clustering fails: a simple yet principled alternative algorithm",
    PLoS One, (11)9:e0162259
    This implementation follows the description in that paper.
    '''
   
    # Initialization (Alg. 3 line 1)
    (D, N) = X.shape
    assert(D > 0)
    assert(N > 0)
    K = 1
    z = np.zeros((N), dtype=int)  # everybody assigned to first cluster
    Enew = np.inf
    dE = np.inf
    ic = 0  # iteration coung
    E = list()
    # Convergence test (Alg. 3 line 14 and Appendix B)
    while (abs(dE) > epsilon and ic < maxiter):
        Eold = Enew
        dik = np.ones((N, 1)) * np.inf
        for i in range(N):
            dk = np.ones((K+1, 1)) * np.inf
            f = np.empty(K+1)
            Nki = np.ones((K), dtype=int)
            xi = np.atleast_2d(X[:, i]).T  # current data point
            for k in range(K):
                zki = (z == k)
                zki[i] = False
                Nki[k] = zki.sum()
                # Updates meaningless for Nki=0
                if (Nki[k] == 0):
                    continue
                # Update NW cluster hyper parameters (Alg. 3 line 7)
                mki, aki, cki, Bki = nwupd(Nki[k], X[:, zki], m0, a0, c0, B0)

                # Compute Student-t NLL, existing clusters (Alg. 3 line 8)
                dk[k] = stnll(xi, mki, aki, cki, Bki, D)
                # Avoid reinforcement effect at initialization (Appendix B)
                if (ic == 0):
                    Nki[0] = 1
                f[k] = dk[k]-np.log(Nki[k])
            # Compute Student-t NLL, new cluster (Alg. 3 line 9)
            dk[K] = stnll(xi, m0, a0, c0, B0, D)
            f[K] = dk[K]-np.log(N0)
            # Compute MAP assignment (Alg. 3 line 10)
            if(fDebug):
                print(i, 'Compute MAP assignment K=', K, 'f=', f, 'dk=', dk)

            z[i] = np.argmin(f)
            dik[i] = f[z[i]]
            # Create new cluster if required (Alg. 3 line 11-12)
            if (z[i] == K):
                K = K + 1
        # Remove any empty clusters and re-assign (Appendix B)
        Knz = 0
        for k in range(K):
            i = (z == k)
            Nk = i.sum()
            if (Nk > 0):
                z[i] = Knz
                Knz = Knz + 1
        K = Knz
        Nk, _ = np.histogram(z, range(K+1))
        # Compute updated NLL (Alg. 3 line 13)
        Enew = dik.sum()-K*np.log(N0)-np.sum(gammaln(Nk))
        dE = Eold - Enew
        ic += 1
        E.append(Enew)
        print('Iteration %d: K=%d, E=%f, dE=%f\n' % (ic, K, Enew, dE))

    # Compute cluster centroids (Appendix D)
    mu = np.ones((D, K))
    sigma = np.ones((D, K))
    for k in range(K):
        xk = X[:, z == k]
        mu[:, k] = xk.mean(1)
        sigma[:,k] = xk.std(1)
    return mu, sigma, z, K, E


def stnll(x, m, a, c, B, D):
    '''
    Compute Student-t negative log likelihood (Appendix A, eqn. (20))
    '''
    mu = m
    nu = a-D+1
    Lambda = c*float(nu)/(c+1)*B
    S = np.dot(np.dot((x-mu).T, Lambda), (x-mu))
    _, logdetL = slogdet(Lambda)
    return float(nu+D)/2.*np.log(1.+S/float(nu))\
        - 0.5*logdetL+gammaln(nu/2.)\
        - gammaln((float(nu)+D)/2.)+D/2.*np.log(float(nu)*np.pi)


def nwupd(Nki, xki, m0, a0, c0, B0):
    '''
    Update Normal-Wishart hyper parameters (Appendix A, eqns. (18-19))
    '''
    xmki = xki.mean(1)[:, None]
    xmcki = xki-repmat(xmki, 1, Nki)
    Ski = np.dot(xmcki, xmcki.T)
    cki = c0+Nki
    mki = (c0*m0+Nki*xmki)/cki
    xm0cki = xmki-m0
    Bki = inv(inv(B0)+Ski+c0*Nki/cki*np.dot(xm0cki, xm0cki.T))
    aki = a0+Nki
    return mki, aki, cki, Bki

    
    