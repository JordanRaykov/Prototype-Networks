
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 17:26:29 2021

@author: raykovy
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def Logistic_LOOCV(data, ids, outcomevar, idcolumn):
    """
        Intermediate function. 
            
    """
    LOOCV_O = ids
    data_filtered = data[data[idcolumn] != LOOCV_O]
    data_cv = data[data[idcolumn] == LOOCV_O]
   
    # Test data - the person left out of training
    data_test = data_cv.drop(columns=idcolumn)
    X_test = data_test.drop(columns=[outcomevar])
    y_test = data_test[outcomevar] #This is the outcome variable
    
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=idcolumn)
    X_train = data_train.drop(columns=[outcomevar])
    
    X_train= np.array(X_train)
    y_train = np.array(data_train[outcomevar]) #Outcome variable here

    
    from sklearn.linear_model import LogisticRegression
    # Instantiate model with numestimators decision trees
    clf = LogisticRegression(random_state=0, class_weight= "balanced", max_iter=1000).fit(X_train, y_train)
    # Train the model on training data
    clf.fit(X_train, y_train);
    
    # Use the forest's predict method on the test data
    predictions = clf.predict(X_test)
    prob_predictions = clf.predict_proba(X_test)
    labels = y_test
    
    return predictions, prob_predictions, labels


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

outcomevar = 'Non-tremor/Tremor'
outcomevar_id = 49
idcolumn = 'ID'
idcolumn_id = 1

prob_predictions_out_of_sample_stacked = np.zeros([1,2])
labels_stacked = np.zeros([1,1])
prob_predictions_training_stacked = np.zeros([1,2])
TPR_out_of_sample = []
TPR_training = []
TNR_out_of_sample = []
TNR_training = []
TNR_in_sample = []
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

num_participants = -1
for i in range(4):
    if i != 24:
        data_filtered = data_features[data_features[idcolumn] != i+1]
        data_cv = data_features[data_features[idcolumn] == i+1]
    
        X_train = data_filtered.drop(columns=["ID", "Non-tremor/Tremor", "Medication_Intake", "Prototype_ID", "Activity_label"])
        X_train = np.array(X_train)
    
        y_train = np.array(data_filtered['Non-tremor/Tremor']) #Outcome variable here
        label = data_cv['Non-tremor/Tremor']
        # Test data - the person left out of training
        data_test = data_cv.drop(columns=["Medication_Intake","Prototype_ID","Activity_label", "ID", "Non-tremor/Tremor"])
    
    
        logistic = LogisticRegression(random_state=0, class_weight= "balanced", max_iter=1000).fit(X_train, y_train)
        prob_predictions_training = logistic.predict_proba(X_train)
        predictions_training = logistic.predict(X_train)
    
        thresholds = np.linspace(0,1,100)
        for threshold in thresholds:          
            for t in range(len(prob_predictions_training)):
                if prob_predictions_training[t,0]>threshold:
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
            if TN_training/N_training > 0.945 and TN_training/N_training < 0.955:
                global_threshold.append(threshold)
                num_participants = num_participants + 1
                TPR_training.append(TP_training/P_training)
                TNR_training.append(TN_training/N_training)
                predictions_training_stacked = np.vstack((predictions_training_stacked, predictions_training.reshape(-1,+1)))
                break
        # Train data - all other people in dataframe
        data_test = np.array(data_test)
        prob_predictions_out_of_sample = logistic.predict_proba(data_test)
        predictions_out_of_sample = logistic.predict(data_test) 
        labels_array = label.array
        labels_array = labels_array.to_numpy()
    
        # pred, prob_predictions_out_of_sample, label = Logistic_LOOCV(data_features, i+1, outcomevar, idcolumn)
        prob_predictions_out_of_sample_stacked = np.vstack((prob_predictions_out_of_sample_stacked, prob_predictions_out_of_sample))
        labels_stacked = np.vstack((labels_stacked, labels_array.reshape(-1,+1)))
        labels.append(label)
    
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
    
    # Training in-sample
        logistic_in_sample = LogisticRegression(random_state=0, class_weight= "balanced", max_iter=1000).fit(np.vstack((X_train, data_test)), np.vstack((y_train.reshape(-1,+1),labels_array.reshape(-1,+1))))
        prob_predictions_in_sample = logistic_in_sample.predict_proba(data_test)
        predictions_in_sample = logistic_in_sample.predict(data_test)
        for t in range(len(prob_predictions_in_sample)):
            if prob_predictions_in_sample[t,0]>global_threshold[num_participants]:
                predictions_in_sample[t] = 0
            else:
                predictions_in_sample[t] = 1            
    
        TP_in_sample = 0
        P_in_sample = 0
        TN_in_sample = 0
        N_in_sample = 0
        for t in range(len(predictions_in_sample)):
            if labels_array[t] == 1 and predictions_in_sample[t] == 1:
                TP_in_of_sample = TP_in_sample + 1
            if labels_array[t] == 1:
                P_in_sample = P_in_sample + 1
            if labels_array[t] == 0 and predictions_in_sample[t] == 0:
                TN_in_sample = TN_in_sample + 1
            if labels_array[t] == 0:
                N_in_sample = N_in_sample + 1
        TNR_in_sample.append(TN_in_sample/N_in_sample)
    
        idt = str(i)
        print('...' + idt + ' processing complete.')               
    
import pickle
with open('save_global_thresholds_logistic.pkl', 'wb') as f:
    pickle.dump(global_threshold, f)  
    
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
 
    