
# -*- coding: utf-8 -*-
"""
@author: Jordan
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
X = data_features.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"])
Y = data_features["Non-tremor/Tremor"]

outcomevar = 'Non-tremor/Tremor'
outcomevar_id = 49
idcolumn = 'ID'
idcolumn_id = 1

IDlist = {'1', '2', '3', '4', '5', '6', '7', '8'}


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

prob_predictions_out_of_sample_stacked = np.zeros([1,2])
labels_stacked = np.zeros([1,1])
prob_predictions_training_stacked = np.zeros([1,2])
TPR_out_of_sample = []
TPR_proto_out_of_sample = []
TPR_training = []
TNR_out_of_sample = []
TNR_proto_out_of_sample = []
TNR_training = []
global_threshold = []
AUC_out_of_sample = []
AUC_proto_out_of_sample = []
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
for i in range(48):
    if i != 24:

        data_filtered = data_features[data_features[idcolumn] != i+1]
        data_cv = data_features[data_features[idcolumn] == i+1]
       
        
        # Train data - all other people in dataframe
        data_train = data_filtered.drop(columns=idcolumn)
        X_train = data_filtered.drop(columns=['ID','Non-tremor/Tremor','Medication_Intake', 'Prototype_ID', 'Activity_label', 'Non_tremor_activity_labels']) 
    
        X_train= np.array(X_train)
        y_train = np.array(data_train[outcomevar]) #Outcome variable here
        data_test = data_cv.drop(columns=["Medication_Intake","Prototype_ID","Activity_label","Non_tremor_activity_labels"])
        
        
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
        
        label = data_test[outcomevar]
            # Train data - all other people in dataframe
        data_test = data_test.drop(columns=["ID"])
        data_test = data_test.drop(columns=[outcomevar])
        prob_predictions_out_of_sample = logistic.predict_proba(data_test)
        pred = logistic.predict(data_test) 
        
        
        #data_test = data_features.drop(columns=['Medication_Intake', 'Prototype_ID','Activity_label','Non_tremor_activity_labels'])    
        #pred, prob_predictions_out_of_sample, label = Logistic_LOOCV(data_test, i+1, outcomevar, idcolumn)
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
                
        # Compute sensitivity and specificity per strata 
        TPR_per_strata = 0
        TNR_per_strata = 0
        auc_per_strata = 0
        
        TPR_per_proto_strata = 0
        TNR_per_proto_strata = 0
        auc_per_proto_strata = 0
        
        data_curr = data_cv[data_cv["ID"] == i+1]
    
        for kk in range(7):
            TP_out_of_sample = 0
            P_out_of_sample = 0
            TN_out_of_sample = 0
            N_out_of_sample = 0
            strata_indicators = np.where(data_curr["Prototype_ID"] == kk+1)
            strata_indicators = np.array(strata_indicators)   
            for t in strata_indicators[0]:
                if labels_array[t] == 1 and predictions_out_of_sample[t] == 1:
                    TP_out_of_sample = TP_out_of_sample + 1
                if labels_array[t] == 1:
                    P_out_of_sample = P_out_of_sample + 1
                if labels_array[t] == 0 and predictions_out_of_sample[t] == 0:
                    TN_out_of_sample = TN_out_of_sample + 1
                if labels_array[t] == 0:
                    N_out_of_sample = N_out_of_sample + 1
                
            if P_out_of_sample>0:        
                TPR_per_proto_strata = np.hstack((TPR_per_proto_strata, TP_out_of_sample/P_out_of_sample))
            else:
                TPR_per_proto_strata = np.hstack((TPR_per_proto_strata, 0))
            if N_out_of_sample>0:    
                TNR_per_proto_strata = np.hstack((TNR_per_proto_strata, TN_out_of_sample/N_out_of_sample))
            else:
                TNR_per_proto_strata = np.hstack((TNR_per_proto_strata, 0))
            
            if len(np.unique(labels_array[strata_indicators])) > 1:
                auc_per_proto_strata = np.hstack((auc_per_proto_strata, roc_auc_score(labels_array[strata_indicators[0]], prob_predictions_out_of_sample[strata_indicators[0], 1])))
            else:
                auc_per_proto_strata = np.hstack((auc_per_proto_strata,0))
                
        AUC_proto_out_of_sample.append(auc_per_proto_strata)        
        TPR_proto_out_of_sample.append(TPR_per_proto_strata)
        TNR_proto_out_of_sample.append(TNR_per_proto_strata)
                
        for kk in range(7):
            TP_out_of_sample = 0
            P_out_of_sample = 0
            TN_out_of_sample = 0
            N_out_of_sample = 0
            strata_indicators = np.where(data_curr["Non_tremor_activity_labels"] == kk+1)# & (data_features["ID"] == i+1))
            strata_indicators = np.array(strata_indicators)   
            for t in strata_indicators[0]:
                if labels_array[t] == 1 and predictions_out_of_sample[t] == 1:
                    TP_out_of_sample = TP_out_of_sample + 1
                if labels_array[t] == 1:
                    P_out_of_sample = P_out_of_sample + 1
                if labels_array[t] == 0 and predictions_out_of_sample[t] == 0:
                    TN_out_of_sample = TN_out_of_sample + 1
                if labels_array[t] == 0:
                    N_out_of_sample = N_out_of_sample + 1
                
            if P_out_of_sample>0:        
                TPR_per_strata = np.hstack((TPR_per_strata, TP_out_of_sample/P_out_of_sample))
            else:
                TPR_per_strata = np.hstack((TPR_per_strata, 0))
            if N_out_of_sample>0:    
                TNR_per_strata = np.hstack((TNR_per_strata, TN_out_of_sample/N_out_of_sample))
            else:
                TNR_per_strata = np.hstack((TNR_per_strata, 0))
            
            if len(np.unique(labels_array[strata_indicators])) > 1:
                auc_per_strata = np.hstack((auc_per_strata, roc_auc_score(labels_array[strata_indicators[0]], prob_predictions_out_of_sample[strata_indicators[0], 1])))
            else:
                auc_per_strata = np.hstack((auc_per_strata,0))
                            
        AUC_out_of_sample.append(auc_per_strata)        
        TPR_out_of_sample.append(TPR_per_strata)
        TNR_out_of_sample.append(TNR_per_strata)
        idt = str(i)
        print('...' + idt + 'processing complete.')        

tpr_array = np.zeros((8,8))
auc_array = np.zeros((8,8))
for i in range(8):
    tpr_array[i,:] = TPR_out_of_sample[i] 
    tpr_array[i,tpr_array[i,:]==0] = np.nan
    auc_array[i,:] = AUC_out_of_sample[i]  
    auc_array[i,auc_array[i,:]==0] = np.nan
    
mean_tpr_strata = np.nanmean(tpr_array[:8,:], axis=0)
std_tpr_strata = np.nanstd(tpr_array[:8,:], axis=0)
mean_auc_strata = np.nanmean(auc_array[:8,:], axis=0)
std_auc_strata = np.nanstd(auc_array[:8,:], axis=0)

tnr_array = np.zeros((24,8))
for i in range(24):
    tnr_array[i,:] = TNR_out_of_sample[i]   
    tnr_array[i,tnr_array[i,:]==0] = np.nan    
    
mean_tnr_array = np.nanmean(tnr_array[:,:], axis=0)
std_tnr_array = np.nanstd(tnr_array[:,:], axis=0)
mean_tnr_first8 = np.nanmean(tnr_array[:8,:], axis=0)
std_tnr_first8 = np.nanstd(tnr_array[:8,:], axis=0)

tpr_proto_array = np.zeros((8,8))
auc_proto_array = np.zeros((8,8))
for i in range(8):
    tpr_proto_array[i,:] = TPR_out_of_sample[i] 
    tpr_proto_array[i,tpr_proto_array[i,:]==0] = np.nan
    auc_proto_array[i,:] = AUC_proto_out_of_sample[i]  
    auc_proto_array[i,auc_proto_array[i,:]==0] = np.nan
    
mean_tpr_proto_strata = np.nanmean(tpr_proto_array[:8,:], axis=0)
std_tpr_proto_strata = np.nanstd(tpr_proto_array[:8,:], axis=0)
mean_auc_proto_strata = np.nanmean(auc_proto_array[:8,:], axis=0)
std_auc_proto_strata = np.nanstd(auc_proto_array[:8,:], axis=0)

tnr_proto_array = np.zeros((24,8))
for i in range(24):
    tnr_proto_array[i,:] = TNR_out_of_sample[i]      
    tnr_proto_array[i,tnr_proto_array[i,:]==0] = np.nan
    
mean_tnr_proto_array = np.nanmean(tnr_proto_array[:,:], axis=0)
std_tnr_proto_array = np.nanstd(tnr_proto_array[:,:], axis=0)
mean_tnr_proto_first8 = np.nanmean(tnr_proto_array[:8,:], axis=0)
std_tnr_proto_first8 = np.nanstd(tnr_proto_array[:8,:], axis=0)

import pickle
with open('save_TNR_per_nontremor_strata.pkl', 'wb') as f:
    pickle.dump(TNR_out_of_sample, f)
with open('save_TPR_per_prototype_strata.pkl', 'wb') as f:
    pickle.dump(TPR_proto_out_of_sample, f)
with open('save_AUC_per_prototype_strata.pkl', 'wb') as f:
    pickle.dump(AUC_proto_out_of_sample, f)
with open('save_TNR_per_prototype_strata.pkl', 'wb') as f:
    pickle.dump(TNR_proto_out_of_sample, f)
 
    