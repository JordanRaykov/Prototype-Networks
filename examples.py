# -*- coding: utf-8 -*-
"""
Basic examples on data loading on usage of the Prototypical networks from "Insert REFERENCE"

@author: Yordan P. Raykov, Luc JW. Evers
"""
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from utils import load_PD_at_home_data
import h5py
import pandas as pd 

# Example 1: (Computing signal processing features in Python)
phys_data = sio.loadmat('physilog_PD_hbv012.mat')#['phys']
labels_data = sio.loadmat('labels_PD_phys_tremorfog_sharing_prototypes_selection.mat')['labels']
#processed_data = load_PD_at_home_data('physilog_PD_hbv012.mat', 'labels_PD_phys_tremorfog_sharing_prototypes_selection.mat', Ts = 2, Fs = 200)

    ################### UNDONE #####################


## Example 2: (Use pre-computed feature matrices in MATLAB and configure tremor/activity prototype matrices)
# This example loads a feature matrix for including PD@Home participants data computed over 2-second windows
# The example then loads additional annotations and formats the prototype labels described in Evers et al. 2024

mat_data = 'Y_features_PD_patients_2sec_200Hz_normalized.mat'
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['Y_tremor_features_200Hz_norm'];
data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])
data_features = data_features.dropna()

ind_proto_is = np.where(Y_features[:,47]>0)
ind_proto_is = np.array(ind_proto_is)
Y_features[ind_proto_is, 48] = 1 # Make all annotated prototypes to be tremor annotation

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

# Activities of daily living are group into 7 categories as decribed in Evers et al. 2024, Section 4
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

# The tremor classes in Evers et al. 2020 include different phenotypes of annotated tremor activities. 
# For the purpose of most demos in this package, we evaluate the binary tremor detection task 
# so below we apply this step as described in Evers et al. 2024/    

data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 2] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 3] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 96] = 0
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 97] = 1
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 98] = 0
data_features["Non-tremor/Tremor"][data_features["Non-tremor/Tremor"] == 99] = 0

# Once we have encoded the relevant prototype information from prototype tremors and non-tremor activities,
# we drop the medication intake and activity labels from some of the further evaluations. Ignore this step,
# if you are interested in using this info.

data_features = data_features.drop(columns=["Medication_Intake", "Activity_label"]) 
import pickle
# Store to pickle for later use, note that other examples load the data_features dataframe
# as produced here, and stored on the repository page
  
with open('data_features.pkl', 'wb') as f:
    pickle.dump(data_features, f)
    
## Example 3: Load data_features and test Single-Layer Prototypical neural network 
#  with a few different configurations, using leave-one-subject-out cross validation.    
from Prototypical_NN import Layer_prototypical_NN
from utils import compute_basis_params
from sklearn.metrics import roc_auc_score
import pickle
    
with open('data_features.pkl', 'rb') as f:
    data_features = pickle.load(f)
    
protovar = "Prototype_ID"   
nontremorprotovar = "Non_tremor_activity_labels"

global_threshold = []
TPR_training = []
TNR_training = []
TPR_test = []
TNR_test = []
AUC_test = []

predictions_training_stacked = np.zeros([1,1])
prob_predictions_test_stacked = np.zeros([1,2])
predictions_test_stacked = np.zeros([1,1])
labels_stacked = np.zeros([1,1])
auc_per_patient_test_stacked = np.zeros([1,1])

for i in range(8):
    data_filtered = data_features[data_features[idcolumn] != i+1] # Remove current subject ID and use for evaluation 
    data_cv = data_features[data_features[idcolumn] == i+1] # Isolate the current subject ID
   
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=idcolumn)
    # Take the arrays from the data frame
    y_train = np.array(data_train['Non-tremor/Tremor']) # Outcome variable here
    X_train = data_train.drop(columns=['Non-tremor/Tremor']) # Remove the outcome from the feature matrix 
    #Layer_prototypical_NN(X_train, y_train, basis_params, X_test, y_test, train_mode, proto_select, hypers, basis_type):
        
    basis_params = compute_basis_params(data_filtered, protovar, nontremorprotovar)
    predictions_training, prob_predictions_training, labels_training =  Layer_prototypical_NN(X_train, y_train, basis_params, X_train, y_train, train_mode = 'predict', proto_select = 'fixed', basis_type = 'gaussian')
    predictions_training = predictions_training.reshape(+1,-1)[0]
  #  C, Beta, rbf_model, train_prediction, model = RBF_network_proto_and_nontremor_train(data_train, outcomevar, protovar, non_tremor_classes_var, h, d)
  #  predictions_training, prob_predictions_training, labels_training = RBF_network_proto_and_nontremor_predict(data_train, outcomevar, protovar, non_tremor_classes_var, h, rbf_model, model, C, Beta)
    
    thresholds = np.linspace(0.0,1.0,250) # Discretize the AUC and compute thresholds for the classifier
                                          # In Evers et al. 2024, we have used thresholds 
                                          # ensuring specificity of 95%.  
    for threshold in thresholds:          
        for t in range(len(prob_predictions_training)):
            if prob_predictions_training[t,0]>threshold:
                predictions_training[t] = 0
            else:
                predictions_training[t] = 1
        
        # Initialize variables
        print('np.shape(y_train)')
        print(np.shape(y_train))
        print('np.shape(predictions_training)')
        print(np.shape(predictions_training))
        #y_train = y_train.reshape(-1,+1)
        print('np.shape(y_train)')
        print(np.shape(y_train))
        TP_training = np.sum((y_train == 1) & (predictions_training == 1))
        P_training = np.sum(y_train == 1)
        TN_training = np.sum((y_train == 0) & (predictions_training == 0))
        N_training = np.sum(y_train == 0)

        # Calculate TNR and print
        TNR = TN_training / N_training
        print(TNR)

        # Check the condition on the specificity of 95% and append
        if 0.945 < TNR < 0.955:
            global_threshold.append(threshold)
            TPR_training.append(TP_training / P_training)
            TNR_training.append(TNR)
            predictions_training_stacked = np.vstack((predictions_training_stacked, predictions_training.reshape(-1,+1)))
            print(global_threshold)
            break
    # Use estimated thresholds to make out-of-sample predictions with the fully trained classifier
  #  predictions_test, prob_predictions_test, labels_test = RBF_networkLOOCV_proto_and_nontremor
    #Layer_prototypical_NN(X_train, y_train, X_train, y_train, basis_params, train_mode = 'predict', proto_select = 'fixed', basis_type = 'gaussian')
    # Train data - all other people in dataframe
    data_test = data_cv.drop(columns=idcolumn)
    # Take the arrays from the data frame
    y_test = np.array(data_test['Non-tremor/Tremor']) # Outcome variable here
    X_test = data_test.drop(columns=['Non-tremor/Tremor']) # Remove the outcome from the feature matrix 
    basis_params = compute_basis_params(data_filtered, protovar, nontremorprotovar)    
    predictions_test, prob_predictions_test, labels_test = Layer_prototypical_NN(X_train, y_train, basis_params, X_test, y_test, train_mode = 'predict', proto_select = 'fixed', basis_type = 'gaussian')
  #  prob_predictions.append(prob_predictions_out_of_sample)
    prob_predictions_test_stacked = np.vstack((prob_predictions_test_stacked, prob_predictions_test))
    #labels_array = labels_test.array
   # labels_test = labels_test.to_numpy()
    labels_stacked = np.vstack((labels_stacked, labels_test.reshape(-1,+1)))# labels_array.reshape(-1,+1)))

    for t in range(len(prob_predictions_test)):
        if prob_predictions_test[t,0]>global_threshold[i]: # compute the prediction TPR and TNR for the selected threshold during training
            predictions_test[t] = 0
        else:
            predictions_test[t] = 1            
    predictions_test_stacked = np.vstack((predictions_test_stacked, predictions_test.reshape(-1,+1)))
    
    # Initialize variables
    TP_test = np.sum((labels_test == 1) & (predictions_test == 1))
    P_test = np.sum(labels_test == 1)
    TN_test = np.sum((labels_test == 0) & (predictions_test == 0))
    N_test = np.sum(labels_test == 0)
    
    if P_test>0:        
        TPR_test.append(TP_test/P_test)
    else:
        TPR_test.append(None)
    TNR_test.append(TN_test/N_test)
    if len(np.unique(labels_test)) > 1:
        auc_per_patient_test = roc_auc_score(labels_test, prob_predictions_test[:, 1])
    else:
        auc_per_patient_out_of_sample = 0
    AUC_test.append(auc_per_patient_out_of_sample)
    idt = str(i)
    print('...' + idt + ' processing complete.')     
    
# Illustrative plots of performance 
# TPR and TNR in-sample and out-of-sample    
    
    
    
    
    
    
    
# Example of Single-layer Prototypical NN 
X = data.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
y = data[outcomevar] #This is the outcome variable
X_train = np.array(X)
y_train = np.array(y) #Outcome variable here
activity_classes = np.unique(data[non_tremor_classes_var])   


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