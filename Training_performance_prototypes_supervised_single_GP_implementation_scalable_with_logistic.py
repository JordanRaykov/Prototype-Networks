import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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

def tpsrbf(x, mu, beta, ax = None):
    #Add epsilon to avoid zero value for log
    dist = np.divide((x - mu)**2, beta)
    dist = np.sum(dist, axis=0)
    r = dist    
    out = np.exp(-r)
    if r.any()==0:#r == 0):
        out[np.where(r==0)] = 1
    return out

def compute_phi(X, C, Beta):    
    T = X.shape[1]
    h = C.shape[0]      
    r = C.shape[1]
    d = C.shape[1]
    phi = np.zeros([T, h])    
    for i in range(h):
        phi[:,i] = tpsrbf(X, C[i,:].reshape(d,1), Beta[i,:].reshape(d,1), ax = 0)
    return phi
import random
import numpy.matlib

def RBF_network_proto_and_nontremor_train(data, outcomevar, protovar, non_tremor_classes_var, h, d):

    X = data.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
    y = data[outcomevar] #This is the outcome variable
    
    X_train = np.array(X)
    y_train = np.array(y) #Outcome variable here
    
    activity_classes = np.unique(data[non_tremor_classes_var])
    
    proto_num = 7
    num_activity = 5
    C = np.zeros([1, d])
    Beta = np.zeros([1, d])
    mu_basis = np.zeros((d,1))
    sigma_basis = np.zeros((d,1))
    beta = np.zeros([h,d])
    y_trainn = np.zeros([1,1])
    ###Instantiate prototype based basis centers
    for k in range(proto_num):
        id_proto = np.where(data[protovar] == k+1)
        id_proto = np.array(id_proto).reshape(-1)       
        Y_curr_proto = X_train[id_proto,:]  
        print('np.size of Y_curr_proto')
        print(np.shape(Y_curr_proto))
        mu =  np.array(random.choices(Y_curr_proto, k=h)).reshape(h,d)#np.mean(Y_curr_proto, axis=0).reshape(1,d)
        for dd in range(d):
           # beta[:,dd] = np.linspace(0.0001,np.std(Y_curr_proto[:,dd]),h)
            beta[:,dd] = np.std(Y_curr_proto[:,dd])*np.ones((h,))
            beta[:,dd] = beta[:,dd]*0.1
           
        C = np.vstack((C, mu))
        Beta = np.vstack((Beta, beta))
        y_trainn = np.vstack((y_trainn, (np.ones([h,1])*k).reshape(-1,+1)))
    beta = np.zeros([h,d])
    for kk in range(num_activity):
        id_activity = np.where(data[non_tremor_classes_var] == kk+1)
        id_activity = np.array(id_activity).reshape(-1)
        Y_curr_activity = X_train[id_activity,:]
        print('np.size of Y_curr_activity')
        print(np.shape(Y_curr_activity))
        mu = np.array(random.choices(Y_curr_activity, k=h)).reshape(h,d)#np.mean(Y_curr_activity, axis=0).reshape(1,d)
        for dd in range(d):
        #    beta[:,dd] = np.linspace(0.001,np.std(Y_curr_activity[:,dd]),h)
            beta[:,dd] = np.std(Y_curr_activity[:,dd])*np.ones((h,))
            beta[:,dd] = beta[:,dd]*0.1
          #  np.linspace(0.001,np.std(Y_curr_activity[:,dd]),h)
        C = np.vstack((C, mu))
        Beta = np.vstack((Beta, beta))
        y_trainn = np.vstack((y_trainn, (np.ones([h,1])*(kk+proto_num)).reshape(-1,+1)))
        
    
    C = C[1:,:]
    Beta = Beta[1:,:]
    y_trainn = y_trainn[1:,:]
    
    print('shape of C')
    print(np.shape(C))
    print('shape of y_trainn')
    print(np.shape(y_trainn))
    y_trainn = keras.utils.np_utils.to_categorical(y_trainn, h*(num_activity + proto_num))
    kernel = PairwiseKernel(metric='rbf') 
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    C = lda.transform(C)
    rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)
    
    #y_train is intput to the softmax layer
    #y_train = keras.utils.np_utils.to_categorical(y_train, 2) # Tremor and non-tremor class
    X_train_reduced = lda.transform(X_train)
    train_rbf = rbf_model.predict(X_train_reduced)
    
    # Instantiate model with numestimators decision trees
    clf = LogisticRegression(random_state=0, class_weight= "balanced", max_iter=1000)
    # Train the model on training data
    clf.fit(train_rbf, y_train);
    
    # Use the forest's predict method on the test data
    train_prediction = clf.predict(train_rbf)
    
    
    # #                      PERCEPTRONS LAYERS
    # batch_size = 128
    # epochs = 1
    # img_size = d
    # size = len(C)
    # model = Sequential()
    # model.add(Dense(img_size, activation='relu', input_shape=(size,)))
    # model.add(Dropout(0.2))
    # #model.add(Dense(size, activation='softmax'))
    # model.add(Dense(2, activation='softmax'))
    # model.summary()
    # nadam=tensorflow.keras.optimizers.Nadam(lr=0.005)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=nadam,
    #               metrics=['accuracy'])
    # #                      TRAINING THE MODEL
    # history = model.fit(train_rbf, y_train,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1)
    # train_prediction_probabilities = model.predict(train_rbf)
    # train_prediction = np.argmax(train_prediction_probabilities, axis=1)
    

    return C, Beta, rbf_model, train_prediction, clf, lda

def RBF_network_proto_and_nontremor_predict(data, outcomevar, protovar, non_tremor_classes_var, h, rbf_model, model, C, Beta, lda):

    X = data.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
    y = data[outcomevar] #This is the outcome variable
    
    X_test = np.array(X)
    y_test = np.array(y) #Outcome variable here
    X_test_down = lda.transform(X_test)
    test_rbf = rbf_model.predict(X_test_down)
    predictions = model.predict(test_rbf)#np.argmax(prob_predictions, axis=1)
    prob_predictions = model.predict_proba(test_rbf)
    labels = y_test
    
    return predictions, prob_predictions, labels

def RBF_networkLOOCV_proto_and_nontremor(data, ids, outcomevar, protovar, non_tremor_classes_var, idcolumn, h, d):

    LOOCV_O = ids
    data_filtered = data[data[idcolumn] != LOOCV_O]
    data_cv = data[data[idcolumn] == LOOCV_O]
   
    # Test data - the person left out of training
    data_test = data_cv.drop(columns=idcolumn)
    X_test = data_test.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
    y_test = data_test[outcomevar] #This is the outcome variable
    
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=idcolumn)
    X_train = data_train.drop(columns=[outcomevar, protovar, non_tremor_classes_var])
    
    feature_list = list(X_train.columns)
    X_train = np.array(X_train)
    
    import numpy.matlib
    import random
    
    y_train = np.array(data_train[outcomevar]) #Outcome variable here
    
    proto_num = 7
    num_activity = 5
    C = np.zeros([1, d])
    Beta = np.zeros([1, d])
    beta = np.zeros([h,d])
    y_trainn = np.zeros([1,1])
    ###Instantiate prototype based basis centers
    for k in range(proto_num):
        id_proto = np.where(data_train[protovar] == k+1)
        id_proto = np.array(id_proto).reshape(-1)       
        Y_curr_proto = X_train[id_proto,:]      
        mu = np.array(random.choices(Y_curr_proto, k=h)).reshape(h,d)#np.mean(Y_curr_proto, axis=0).reshape(1,d)
        for dd in range(d):
           # beta[:,dd] = np.linspace(0.0001,np.std(Y_curr_proto[:,dd]),h)
            beta[:,dd] = np.std(Y_curr_proto[:,dd])*np.ones((h,))
            beta[:,dd] = beta[:,dd]*0.01
        C = np.vstack((C, mu))
        Beta = np.vstack((Beta, beta))    
        y_trainn = np.vstack((y_trainn, (np.ones([h,1])*k).reshape(-1,+1)))
    beta = np.zeros([h,d])
    for kk in range(num_activity):
        id_activity = np.where(data_train[non_tremor_classes_var] == kk+1)
        id_activity = np.array(id_activity).reshape(-1)
        Y_curr_activity = X_train[id_activity,:]
        mu = np.array(random.choices(Y_curr_activity, k=h)).reshape(h,d)#np.mean(Y_curr_activity, axis=0).reshape(1,d)
        for dd in range(d):
          #  beta[:,dd] = np.linspace(0.001,np.std(Y_curr_activity[:,dd]),h)
            beta[:,dd] = np.std(Y_curr_activity[:,dd])*np.ones((h,))
            beta[:,dd] = beta[:,dd]*0.1
        C = np.vstack((C, mu))
        Beta = np.vstack((Beta, beta))
        y_trainn = np.vstack((y_trainn, (np.ones([h,1])*(kk+proto_num)).reshape(-1,+1)))
        
    C = C[1:,:]
    Beta = Beta[1:,:]
    y_trainn = y_trainn[1:,:]
    y_trainn = keras.utils.np_utils.to_categorical(y_trainn, h*(num_activity + proto_num))
    kernel = PairwiseKernel(metric='rbf') 
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    C = lda.transform(C)
    rbf_model = GaussianProcessRegressor(kernel=kernel).fit(C, y_trainn)
    train_rbf = rbf_model.predict(X_train)
    
    
    
    # Instantiate model with numestimators decision trees
    clf = LogisticRegression(random_state=0, class_weight= "balanced", max_iter=1000)
    # Train the model on training data
    clf.fit(train_rbf, y_train);
    test_rbf = rbf_model.predict(X_test)
    prob_predictions = clf.predict_proba(test_rbf)
    predictions = clf.predict(test_rbf)
    labels = y_test 
    
    return predictions, prob_predictions, labels


d = 45
h = 50
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

ind_proto_is = np.where(Y_features[:,47]>0)
ind_proto_is = np.array(ind_proto_is)
Y_features[ind_proto_is, 48] = 1 # Make all annotated prototypes to be tremor annotation

data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])

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

#non_tremor_activity_label_5 = np.where((data_features["Activity_label"] == 1) & (data_features["Non-tremor/Tremor"] == 98))
#non_tremor_activity_label_5 = np.array(non_tremor_activity_label_5)
#non_tremor_activity_label_5 = np.hstack((non_tremor_activity_label_5, np.where((data_features["Activity_label"] == 2) & (data_features["Non-tremor/Tremor"] == 98))))
#non_tremor_activity_label_5 = np.hstack((non_tremor_activity_label_5, np.where((data_features["Activity_label"] == 6) & (data_features["Non-tremor/Tremor"] == 98))))

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
#for n in non_tremor_activity_label_5[0]:
#    Non_tremor_activity_label[n] = 5
for n in non_tremor_activity_label_6[0]:
    Non_tremor_activity_label[n] = 5
#for n in non_tremor_activity_label_7[0]:
#    Non_tremor_activity_label[n] = 6
    
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


for i in range(8):
    data_filtered = data_features[data_features[idcolumn] != i+1]
    data_cv = data_features[data_features[idcolumn] == i+1]
   
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=idcolumn)
    y_train = np.array(data_features['Non-tremor/Tremor']) #Outcome variable here
    X_train = data_train.drop(columns=[outcomevar])
    C, Beta, rbf_model, train_prediction, model = RBF_network_proto_and_nontremor_train(data_train, outcomevar, protovar, non_tremor_classes_var, h, d)
    predictions_training, prob_predictions_training, labels_training = RBF_network_proto_and_nontremor_predict(data_train, outcomevar, protovar, non_tremor_classes_var, h, rbf_model, model, C, Beta)
    
    thresholds = np.linspace(0.0,1.0,250)#[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.]
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
        if prob_predictions_out_of_sample[t,0]>global_threshold[i]:
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
    
    idt = str(i)
    print('...' + idt + ' processing complete.')   
    
##Train in-sample and output parameters
data_in_sample = data_features.drop(columns=["ID"])
C_in_sample, Beta_in_sample, rbf_model_in_sample, train_prediction_in_sample, model_in_sample = RBF_network_proto_and_nontremor_train(data_in_sample, outcomevar, protovar, non_tremor_classes_var, h, d)
    
import pickle
with open('save_global_thresholds_single_proto_GP.pkl', 'wb') as f:
    pickle.dump(global_threshold, f)  
    
with open('save_single_proto_GP_trained_visits.pkl', 'wb') as f:
    pickle.dump(train_prediction_in_sample, f)
    pickle.dump(rbf_model_in_sample, f)
    pickle.dump(model_in_sample, f)
    pickle.dump(C_in_sample, f)
    pickle.dump(Beta_in_sample, f)