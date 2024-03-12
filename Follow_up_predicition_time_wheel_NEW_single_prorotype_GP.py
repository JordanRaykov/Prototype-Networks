# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:32:38 2022

@author: yorda
"""


# First have to load Prototype thresholds

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.io as pio
import pickle 

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

### Load available follow ups 

mat_data = 'Features_per_second_hbv002.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv002']; 
data_features_with_segments_hbv002 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "id_segment"])
data_features_with_segments_hbv002 = data_features_with_segments_hbv002.dropna()
data_features_hbv002 = data_features_with_segments_hbv002.drop(columns = "id_segment")

mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv012']; 
data_features_with_segments_hbv012 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time", "id_segment"])
data_features_with_segments_hbv012 = data_features_with_segments_hbv012.dropna()
data_features_hbv012 = data_features_with_segments_hbv012.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv013.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv013']; 
data_features_with_segments_hbv013 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time", "id_segment"])
data_features_with_segments_hbv013 = data_features_with_segments_hbv013.dropna()
data_features_hbv013 = data_features_with_segments_hbv013.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv017.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv017']; 
data_features_with_segments_hbv017 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time", "id_segment"])
data_features_with_segments_hbv017 = data_features_with_segments_hbv017.dropna()
data_features_hbv017 = data_features_with_segments_hbv017.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv018.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv018']; 
data_features_with_segments_hbv018 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time", "id_segment"])
data_features_with_segments_hbv018 = data_features_with_segments_hbv018.dropna()
data_features_hbv018 = data_features_with_segments_hbv018.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv022.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv022']; 
data_features_with_segments_hbv022 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv022 = data_features_with_segments_hbv022.dropna()
data_features_hbv022 = data_features_with_segments_hbv022.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv023.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv023']; 
data_features_with_segments_hbv023 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv023 = data_features_with_segments_hbv023.dropna()
data_features_hbv023 = data_features_with_segments_hbv023.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv024.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv024']; 
data_features_with_segments_hbv024 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv024 = data_features_with_segments_hbv024.dropna()
data_features_hbv024 = data_features_with_segments_hbv024.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv038.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv038']; 
data_features_with_segments_hbv038 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv038 = data_features_with_segments_hbv038.dropna()
data_features_hbv038 = data_features_with_segments_hbv038.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv043.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv043']; 
data_features_with_segments_hbv043 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv043 = data_features_with_segments_hbv043.dropna()
data_features_hbv043 = data_features_with_segments_hbv043.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv047.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv047']; 
data_features_with_segments_hbv047 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv047 = data_features_with_segments_hbv047.dropna()
data_features_hbv047 = data_features_with_segments_hbv047.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv051.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv051']; 
data_features_with_segments_hbv051 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time","id_segment"])
data_features_with_segments_hbv051 = data_features_with_segments_hbv051.dropna()
data_features_hbv051 = data_features_with_segments_hbv051.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv054.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv054']; 
data_features_with_segments_hbv054 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time","id_segment"])
data_features_with_segments_hbv054 = data_features_with_segments_hbv054.dropna()
data_features_hbv054 = data_features_with_segments_hbv054.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv063.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv063']; 
data_features_with_segments_hbv063 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time","id_segment"])
data_features_with_segments_hbv063 = data_features_with_segments_hbv063.dropna()
data_features_hbv063 = data_features_with_segments_hbv063.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv074.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv074']; 
data_features_with_segments_hbv074 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv074 = data_features_with_segments_hbv074.dropna()
data_features_hbv074 = data_features_with_segments_hbv074.drop(columns = ["Time","id_segment"])

mat_data = 'Features_per_second_hbv090.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv090']; 
data_features_with_segments_hbv090 = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","Time", "id_segment"])
data_features_with_segments_hbv090 = data_features_with_segments_hbv090.dropna()
data_features_hbv090 = data_features_with_segments_hbv090.drop(columns = ["Time","id_segment"])

### Merge data from all data structures to normalize the features

data_features_array_hbv002 = np.array(data_features_hbv002)
data_features_array_hbv012 = np.array(data_features_hbv012)
data_features_array_hbv013 = np.array(data_features_hbv013)
data_features_array_hbv017 = np.array(data_features_hbv017)
data_features_array_hbv018 = np.array(data_features_hbv018)
data_features_array_hbv022 = np.array(data_features_hbv022)
data_features_array_hbv023 = np.array(data_features_hbv023)
data_features_array_hbv024 = np.array(data_features_hbv024)
data_features_array_hbv038 = np.array(data_features_hbv038)
data_features_array_hbv043 = np.array(data_features_hbv043)
data_features_array_hbv047 = np.array(data_features_hbv047)
data_features_array_hbv051 = np.array(data_features_hbv051)
data_features_array_hbv054 = np.array(data_features_hbv054)
data_features_array_hbv063 = np.array(data_features_hbv063)
data_features_array_hbv074 = np.array(data_features_hbv074)
data_features_array_hbv090 = np.array(data_features_hbv090)

data_merged_follow_ups = np.vstack((data_features_array_hbv002, data_features_array_hbv012, data_features_array_hbv013, data_features_array_hbv017, data_features_array_hbv018, data_features_array_hbv022, data_features_array_hbv023, data_features_array_hbv024, data_features_array_hbv038, data_features_array_hbv043, data_features_array_hbv047, data_features_array_hbv051, data_features_array_hbv054, data_features_array_hbv063, data_features_array_hbv074, data_features_array_hbv090))

data_merged_follow_ups_norm = (data_merged_follow_ups - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv002_norm = (data_features_array_hbv002 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv012_norm = (data_features_array_hbv012 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv013_norm = (data_features_array_hbv013 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv013_norm = (data_features_array_hbv017 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv018_norm = (data_features_array_hbv018 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv022_norm = (data_features_array_hbv022 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv023_norm = (data_features_array_hbv023 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv024_norm = (data_features_array_hbv024 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv038_norm = (data_features_array_hbv038 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv043_norm = (data_features_array_hbv043 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv047_norm = (data_features_array_hbv047 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv051_norm = (data_features_array_hbv051 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv054_norm = (data_features_array_hbv054 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv063_norm = (data_features_array_hbv063 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv074_norm = (data_features_array_hbv074 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)
data_features_array_hbv090_norm = (data_features_array_hbv090 - np.mean(data_merged_follow_ups, axis=0))/np.std(data_merged_follow_ups, axis=0)


import pickle 
with open('save_global_thresholds_single_prototype.pkl', 'rb') as f:
    global_threshold = pickle.load(f)
         
with open('save_single_proto_trained_visits.pkl', 'rb') as f:
    W_in_sample = pickle.load(f)
    phi_in_sample = pickle.load(f)
    C_in_sample = pickle.load(f)
    Beta_in_sample = pickle.load(f)    

id_participant_hbv012 = 0 # Index the participant follow up HBV012
test_rbf_hbv012 = rbf_model_in_sample.predict(data_features_array_hbv012_norm)
prob_predictions_hbv012 = model_in_sample.predict(test_rbf_hbv012)
score_cat = prob_predictions_hbv012
predictions_follow_up_hbv012 = np.argmax(prob_predictions_hbv012, axis=1)
for t in range(len(prob_predictions_hbv012)):
    if prob_predictions_hbv012[t,0]>global_threshold[id_participant_hbv012]:
        predictions_follow_up_hbv012[t] = 0
    else:
        predictions_follow_up_hbv012[t] = 1 
            
  
        
############### Save and use the mean phi indicating distance to the different prototypes ########
h=1
mean_phi = np.zeros((len(prob_predictions_hbv012),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(test_rbf_hbv012[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

#import pickle
with open('save_mean_phi_hbv012_GP.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv012 = mat_contents['accel_time_stamps_hbv012_str']; 
time_stamps_hbv012_hours = np.zeros((len(time_stamps_hbv012), 1))
for t in range(len(time_stamps_hbv012)):
    datem = datetime.strptime(time_stamps_hbv012[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv012_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv012 == 1)
num_bin = len(np.unique(time_stamps_hbv012_hours[ind_tremor_on]))

Y_features_on_hbv012 = data_features_array_hbv012_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv012_hours[ind_tremor_on])
s = 0
tremor_profile_hbv012_per_hour = np.zeros((24, 7))
for t in range(24):
    if t in tremor_hours:
        ind_hour = np.where(time_stamps_hbv012_hours[ind_tremor_on[0]] == t)
        tremor_profile_hbv012_per_hour[t,:] = np.nansum(mean_phi[ind_hour[0], :], axis=0)
    else:
        tremor_profile_hbv012_per_hour[t,:]  = np.zeros((1,7))

    
#with open('save_tremor_profile_proto_GP_hbv012_per_hour.pkl', 'wb') as f:
#    pickle.dump(tremor_profile_hbv012_per_hour, f)
    
#with open('save_tremor_profile_proto_GP_hbv012_per_hour.pkl', 'rb') as f:
#    tremor_profile_hbv012_per_hour = pickle.load(f)

    
tremor_profile_hbv012_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv012_per_hour_reduced[:,0] = tremor_profile_hbv012_per_hour[:,0] + tremor_profile_hbv012_per_hour[:,1]
tremor_profile_hbv012_per_hour_reduced[:,1] = tremor_profile_hbv012_per_hour[:,2] + tremor_profile_hbv012_per_hour[:,3]
tremor_profile_hbv012_per_hour_reduced[:,2] = tremor_profile_hbv012_per_hour[:,4]
tremor_profile_hbv012_per_hour_reduced[:,3] = tremor_profile_hbv012_per_hour[:,5] + tremor_profile_hbv012_per_hour[:,6]
data_duration_sec = np.histogram(time_stamps_hbv012_hours, 24)

#tremor_profile_hbv012_per_hour_plt = np.zeros((24*7, 2))    
tremor_profile_hbv012_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv012_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv012_per_hour_reduced[:,d]/data_duration_sec[0]
    tremor_profile_hbv012_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
    
hbv012_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv012_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv012_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv012_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show()    
    
##### Next participant ##################

phi_test = compute_phi(data_features_array_hbv013_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv013_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv013 = 1 # Index the participant follow up HBV012
predictions_follow_up_hbv013 = Y_hat_test
prob_predictions_follow_up_hbv013 = score_cat

for t in range(len(prob_predictions_follow_up_hbv013)):
    if prob_predictions_follow_up_hbv013[t,0]>global_threshold[id_participant_hbv013]:
        predictions_follow_up_hbv013[t] = 0
    else:
        predictions_follow_up_hbv013[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv013),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv013.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv013.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv013 = mat_contents['accel_time_stamps_hbv013_str'] 
time_stamps_hbv013_hours = np.zeros((len(time_stamps_hbv013), 1))
for t in range(len(time_stamps_hbv013)):
    datem = datetime.strptime(time_stamps_hbv013[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv013_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv013 == 1)
num_bin = len(np.unique(time_stamps_hbv013_hours[ind_tremor_on]))

Y_features_on_hbv013 = data_features_array_hbv013_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv013_hours[ind_tremor_on])
s = 0
tremor_profile_hbv013_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv013_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv013_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv013_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv013_per_hour, f)
    
with open('save_tremor_profile_hbv013_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv013_per_hour = pickle.load(f)
    
tremor_profile_hbv013_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv013_per_hour_reduced[:,0] = tremor_profile_hbv013_per_hour[:,0] + tremor_profile_hbv013_per_hour[:,1]
tremor_profile_hbv013_per_hour_reduced[:,1] = tremor_profile_hbv013_per_hour[:,2] + tremor_profile_hbv013_per_hour[:,3]
tremor_profile_hbv013_per_hour_reduced[:,2] = tremor_profile_hbv013_per_hour[:,4]
tremor_profile_hbv013_per_hour_reduced[:,3] = tremor_profile_hbv013_per_hour[:,5] + tremor_profile_hbv013_per_hour[:,6]
  
tremor_profile_hbv013_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv013_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv013_per_hour_reduced[:,d]
    tremor_profile_hbv013_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv013_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv013_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv013_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv013_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show()    
    
##### Next participant ##################

phi_test = compute_phi(data_features_array_hbv018_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv018_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv018 = 1 # Index the participant follow up HBV018
predictions_follow_up_hbv018 = Y_hat_test
prob_predictions_follow_up_hbv018 = score_cat

for t in range(len(prob_predictions_follow_up_hbv018)):
    if prob_predictions_follow_up_hbv018[t,0]>global_threshold[id_participant_hbv018]:
        predictions_follow_up_hbv018[t] = 0
    else:
        predictions_follow_up_hbv018[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv018),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv018.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv018.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv018 = mat_contents['accel_time_stamps_hbv018_str'] 
time_stamps_hbv018_hours = np.zeros((len(time_stamps_hbv018), 1))
for t in range(len(time_stamps_hbv018)):
    datem = datetime.strptime(time_stamps_hbv018[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv018_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv018 == 1)
num_bin = len(np.unique(time_stamps_hbv018_hours[ind_tremor_on]))

Y_features_on_hbv018 = data_features_array_hbv018_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv018_hours[ind_tremor_on])
s = 0
tremor_profile_hbv018_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv018_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv018_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv018_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv018_per_hour, f)
    
with open('save_tremor_profile_hbv018_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv018_per_hour = pickle.load(f)
    
tremor_profile_hbv018_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv018_per_hour_reduced[:,0] = tremor_profile_hbv018_per_hour[:,0] + tremor_profile_hbv018_per_hour[:,1]
tremor_profile_hbv018_per_hour_reduced[:,1] = tremor_profile_hbv018_per_hour[:,2] + tremor_profile_hbv018_per_hour[:,3]
tremor_profile_hbv018_per_hour_reduced[:,2] = tremor_profile_hbv018_per_hour[:,4]
tremor_profile_hbv018_per_hour_reduced[:,3] = tremor_profile_hbv018_per_hour[:,5] + tremor_profile_hbv018_per_hour[:,6]
  
tremor_profile_hbv018_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv018_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv018_per_hour_reduced[:,d]
    tremor_profile_hbv018_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv018_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv018_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv018_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv018_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show() 

#####Next participant


phi_test = compute_phi(data_features_array_hbv022_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv022_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv022 = 1 # Index the participant follow up HBV018
predictions_follow_up_hbv022 = Y_hat_test
prob_predictions_follow_up_hbv022 = score_cat

for t in range(len(prob_predictions_follow_up_hbv022)):
    if prob_predictions_follow_up_hbv022[t,0]>global_threshold[id_participant_hbv022]:
        predictions_follow_up_hbv022[t] = 0
    else:
        predictions_follow_up_hbv022[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv022),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv022.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv022.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv022 = mat_contents['accel_time_stamps_hbv022_str'] 
time_stamps_hbv022_hours = np.zeros((len(time_stamps_hbv022), 1))
for t in range(len(time_stamps_hbv022)):
    datem = datetime.strptime(time_stamps_hbv022[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv022_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv022 == 1)
num_bin = len(np.unique(time_stamps_hbv022_hours[ind_tremor_on]))

Y_features_on_hbv022 = data_features_array_hbv022_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv022_hours[ind_tremor_on])
s = 0
tremor_profile_hbv022_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv022_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv022_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv022_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv022_per_hour, f)
    
with open('save_tremor_profile_hbv022_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv022_per_hour = pickle.load(f)
    
tremor_profile_hbv022_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv022_per_hour_reduced[:,0] = tremor_profile_hbv022_per_hour[:,0] + tremor_profile_hbv022_per_hour[:,1]
tremor_profile_hbv022_per_hour_reduced[:,1] = tremor_profile_hbv022_per_hour[:,2] + tremor_profile_hbv022_per_hour[:,3]
tremor_profile_hbv022_per_hour_reduced[:,2] = tremor_profile_hbv022_per_hour[:,4]
tremor_profile_hbv022_per_hour_reduced[:,3] = tremor_profile_hbv022_per_hour[:,5] + tremor_profile_hbv022_per_hour[:,6]
  
tremor_profile_hbv022_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv022_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv022_per_hour_reduced[:,d]
    tremor_profile_hbv022_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv022_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv022_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv022_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv022_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show() 



#### Next Participant #######
phi_test = compute_phi(data_features_array_hbv023_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv023_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv023 = 1 # Index the participant follow up HBV023
predictions_follow_up_hbv023 = Y_hat_test
prob_predictions_follow_up_hbv023 = score_cat

for t in range(len(prob_predictions_follow_up_hbv023)):
    if prob_predictions_follow_up_hbv023[t,0]>global_threshold[id_participant_hbv023]:
        predictions_follow_up_hbv023[t] = 0
    else:
        predictions_follow_up_hbv023[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv023),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv023.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv023.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv023 = mat_contents['accel_time_stamps_hbv023_str'] 
time_stamps_hbv023_hours = np.zeros((len(time_stamps_hbv023), 1))
for t in range(len(time_stamps_hbv023)):
    datem = datetime.strptime(time_stamps_hbv023[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv023_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv023 == 1)
num_bin = len(np.unique(time_stamps_hbv023_hours[ind_tremor_on]))

Y_features_on_hbv023 = data_features_array_hbv023_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv023_hours[ind_tremor_on])
s = 0
tremor_profile_hbv023_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv023_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv023_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv023_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv023_per_hour, f)
    
with open('save_tremor_profile_hbv023_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv023_per_hour = pickle.load(f)
    
tremor_profile_hbv023_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv023_per_hour_reduced[:,0] = tremor_profile_hbv023_per_hour[:,0] + tremor_profile_hbv023_per_hour[:,1]
tremor_profile_hbv023_per_hour_reduced[:,1] = tremor_profile_hbv023_per_hour[:,2] + tremor_profile_hbv023_per_hour[:,3]
tremor_profile_hbv023_per_hour_reduced[:,2] = tremor_profile_hbv023_per_hour[:,4]
tremor_profile_hbv023_per_hour_reduced[:,3] = tremor_profile_hbv023_per_hour[:,5] + tremor_profile_hbv023_per_hour[:,6]
  
tremor_profile_hbv023_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv023_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv023_per_hour_reduced[:,d]
    tremor_profile_hbv023_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv023_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv023_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv023_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv023_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show() 


#### Next Participant #######
phi_test = compute_phi(data_features_array_hbv038_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv038_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv038 = 1 # Index the participant follow up HBV038
predictions_follow_up_hbv038 = Y_hat_test
prob_predictions_follow_up_hbv038 = score_cat

for t in range(len(prob_predictions_follow_up_hbv038)):
    if prob_predictions_follow_up_hbv038[t,0]>global_threshold[id_participant_hbv038]:
        predictions_follow_up_hbv038[t] = 0
    else:
        predictions_follow_up_hbv038[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv038),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv038.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv038.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv038 = mat_contents['accel_time_stamps_hbv038_str'] 
time_stamps_hbv038_hours = np.zeros((len(time_stamps_hbv038), 1))
for t in range(len(time_stamps_hbv038)):
    datem = datetime.strptime(time_stamps_hbv038[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv038_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv038 == 1)
num_bin = len(np.unique(time_stamps_hbv038_hours[ind_tremor_on]))

Y_features_on_hbv038 = data_features_array_hbv038_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv038_hours[ind_tremor_on])
s = 0
tremor_profile_hbv038_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv038_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv038_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv038_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv038_per_hour, f)
    
with open('save_tremor_profile_hbv038_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv038_per_hour = pickle.load(f)
    
tremor_profile_hbv038_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv038_per_hour_reduced[:,0] = tremor_profile_hbv038_per_hour[:,0] + tremor_profile_hbv038_per_hour[:,1]
tremor_profile_hbv038_per_hour_reduced[:,1] = tremor_profile_hbv038_per_hour[:,2] + tremor_profile_hbv038_per_hour[:,3]
tremor_profile_hbv038_per_hour_reduced[:,2] = tremor_profile_hbv038_per_hour[:,4]
tremor_profile_hbv038_per_hour_reduced[:,3] = tremor_profile_hbv038_per_hour[:,5] + tremor_profile_hbv038_per_hour[:,6]
  
tremor_profile_hbv038_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv038_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv038_per_hour_reduced[:,d]
    tremor_profile_hbv038_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv038_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv038_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv038_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv038_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show() 


#### Next Participant #######
phi_test = compute_phi(data_features_array_hbv090_norm.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_features_array_hbv090_norm)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
id_participant_hbv090 = 1 # Index the participant follow up HBV023
predictions_follow_up_hbv090 = Y_hat_test
prob_predictions_follow_up_hbv090 = score_cat

for t in range(len(prob_predictions_follow_up_hbv090)):
    if prob_predictions_follow_up_hbv090[t,0]>global_threshold[id_participant_hbv090]:
        predictions_follow_up_hbv090[t] = 0
    else:
        predictions_follow_up_hbv090[t] = 1   
        

############### Save and use the mean phi indicating distance to the different prototypes ########
h=100
mean_phi = np.zeros((len(prob_predictions_follow_up_hbv090),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.sum(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)
    
mean_phi = mean_phi/np.sum(mean_phi, axis = 1).reshape(-1,+1)

with open('save_mean_phi_hbv090.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)
    
mat_data = 'Features_per_second_hbv090.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv090 = mat_contents['accel_time_stamps_hbv090_str'] 
time_stamps_hbv090_hours = np.zeros((len(time_stamps_hbv090), 1))
for t in range(len(time_stamps_hbv090)):
    datem = datetime.strptime(time_stamps_hbv090[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv090_hours[t] = datem.hour
ind_tremor_on = np.where(predictions_follow_up_hbv090 == 1)
num_bin = len(np.unique(time_stamps_hbv090_hours[ind_tremor_on]))

Y_features_on_hbv090 = data_features_array_hbv090_norm[ind_tremor_on[0],:]
tremor_hours = np.unique(time_stamps_hbv090_hours[ind_tremor_on])
s = 0
tremor_profile_hbv090_per_hour = np.zeros((num_bin, 7))
for t in tremor_hours:
    ind_hour = np.where(time_stamps_hbv090_hours[ind_tremor_on[0]] == t)
    tremor_profile_hbv090_per_hour[s,:] = np.nanmean(mean_phi[ind_hour[0], :], axis=0)
    s = s+1
    
with open('save_tremor_profile_hbv090_per_hour.pkl', 'wb') as f:
    pickle.dump(tremor_profile_hbv090_per_hour, f)
    
with open('save_tremor_profile_hbv090_per_hour.pkl', 'rb') as f:
    tremor_profile_hbv090_per_hour = pickle.load(f)
    
tremor_profile_hbv090_per_hour_reduced = np.zeros((24,4))
tremor_profile_hbv090_per_hour_reduced[:,0] = tremor_profile_hbv090_per_hour[:,0] + tremor_profile_hbv090_per_hour[:,1]
tremor_profile_hbv090_per_hour_reduced[:,1] = tremor_profile_hbv090_per_hour[:,2] + tremor_profile_hbv090_per_hour[:,3]
tremor_profile_hbv090_per_hour_reduced[:,2] = tremor_profile_hbv090_per_hour[:,4]
tremor_profile_hbv090_per_hour_reduced[:,3] = tremor_profile_hbv090_per_hour[:,5] + tremor_profile_hbv090_per_hour[:,6]
  
tremor_profile_hbv090_per_hour_plt = np.zeros((24*4, 2))  
for d in range(4):
    tremor_profile_hbv090_per_hour_plt[24*d:24*(d+1),0] = tremor_profile_hbv090_per_hour_reduced[:,d]
    tremor_profile_hbv090_per_hour_plt[24*d:24*(d+1),1] = np.ones((24,))*(d+1)
     
hbv090_time_wheel_prototypes = pd.DataFrame(data = tremor_profile_hbv090_per_hour_plt, columns=["prototype_proportion","tremor_profile" ])
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']*4
hbv090_time_wheel_prototypes['hour'] = hours_array
pio.renderers.default = "browser"
fig = px.line_polar(hbv090_time_wheel_prototypes, r='prototype_proportion', theta='hour', color="tremor_profile", line_close=True, title='Time of Day', width=600, height=500)
fig.show() 


        
id_indicator = np.ones((len(prob_predictions_follow_up), 1))*(id_participant+1)
        
#hbv002_follow_up_tremor = np.hstack((Y_features, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

# Convert matlab time to python
from datetime import datetime
from datetime import timedelta

mat_data = 'Time_stamps_hbv002.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv002 = mat_contents['accel_time_stamps_hbv002_str']; 
time_stamps_hbv002_hours = np.zeros((len(time_stamps_hbv002), 1))
for t in range(len(time_stamps_hbv002)):
    datem = datetime.strptime(time_stamps_hbv002[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv002_hours[t] = datem.hour

    
hbv002_follow_up_tremor = np.hstack((Y_features, time_stamps_hbv002_hours, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

hbv002_follow_up_tremor_df = pd.DataFrame(data=hbv002_follow_up_tremor, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","id_segment", "Time", "Tremor", "Probability_tremor_0","Probability_tremor_1", "ID"])


# Create a summary plot displaying the tremor duration, the tremor severity and the tremor profile throught the different times of day
# Time of day and time of the week wheel - https://stackoverflow.com/questions/40352607/time-wheel-in-python3-pandas
ind_tremor_on = np.where(predictions_follow_up == 1)
tremor_hours = np.unique(time_stamps_hbv002_hours[ind_tremor_on])
num_bin = len(np.unique(time_stamps_hbv002_hours[ind_tremor_on]))
tremor_duration_sec = np.histogram(time_stamps_hbv002_hours[ind_tremor_on], num_bin)
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
tremor_duration_sec_per_hour = np.vstack((np.zeros((6,1)),tremor_duration_sec[0].reshape(-1,+1), np.zeros((1,1))))
hbv002_time_wheel = pd.DataFrame(data = tremor_duration_sec_per_hour, columns=["duration_sec"])
hbv002_time_wheel['hour'] = hours_array

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.line_polar(hbv002_time_wheel, r='duration_sec', theta='hour', line_close=True, title='Time of Day', width=600, height=500)
fig.show()

###############################################################################
mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv012']; 
data_features_with_segments = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "id_segment"])
data_features_with_segments = data_features_with_segments.dropna()
data_features = data_features_with_segments.drop(columns = "id_segment")

id_participant = 0 # Index the participant follow up
data_test_follow_up = np.array(data_features)
prob_predictions_follow_up = logistic_in_sample.predict_proba(data_test_follow_up)
predictions_follow_up = logistic_in_sample.predict(data_test_follow_up)
for t in range(len(prob_predictions_follow_up)):
    if prob_predictions_follow_up[t,0]>global_threshold[id_participant]:
        predictions_follow_up[t] = 0
    else:
        predictions_follow_up[t] = 1  
        
id_indicator = np.ones((len(prob_predictions_follow_up), 1))*(id_participant+1)
        
#hbv002_follow_up_tremor = np.hstack((Y_features, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

# Convert matlab time to python
from datetime import datetime
from datetime import timedelta

mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv012 = mat_contents['accel_time_stamps_hbv012_str']; 
time_stamps_hbv012_hours = np.zeros((len(time_stamps_hbv012), 1))
for t in range(len(time_stamps_hbv012)):
    datem = datetime.strptime(time_stamps_hbv012[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv012_hours[t] = datem.hour

    
hbv012_follow_up_tremor = np.hstack((Y_features, time_stamps_hbv012_hours, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

hbv012_follow_up_tremor_df = pd.DataFrame(data=hbv012_follow_up_tremor, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","id_segment", "Time", "Tremor", "Probability_tremor_0","Probability_tremor_1", "ID"])


# Create a summary plot displaying the tremor duration, the tremor severity and the tremor profile throught the different times of day
# Time of day and time of the week wheel - https://stackoverflow.com/questions/40352607/time-wheel-in-python3-pandas
ind_tremor_on = np.where(predictions_follow_up == 1)
tremor_hours = np.unique(time_stamps_hbv012_hours[ind_tremor_on])
num_bin = len(np.unique(time_stamps_hbv012_hours[ind_tremor_on]))
tremor_duration_sec = np.histogram(time_stamps_hbv012_hours[ind_tremor_on], num_bin)
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
tremor_duration_sec_per_hour = tremor_duration_sec[0].reshape(-1,+1)
hbv012_time_wheel = pd.DataFrame(data = tremor_duration_sec_per_hour, columns=["duration_sec"])
hbv012_time_wheel['hour'] = hours_array

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.line_polar(hbv012_time_wheel, r='duration_sec', theta='hour', line_close=True, title='Time of Day', width=600, height=500)
fig.show()

###Compute average power of the tremor in each time bin
Y_features_on = Y_features[ind_tremor_on[0],:]
tremor_mean_intensity = np.zeros((len(tremor_hours), 1))
tremor_min_intensity = np.zeros((len(tremor_hours), 1))
tremor_max_intensity = np.zeros((len(tremor_hours), 1))
for t in range(len(tremor_hours)):
    ind_hour = np.where(time_stamps_hbv012_hours[ind_tremor_on] == t)
    tremor_mean_intensity[t] = np.mean(Y_features_on[ind_hour, 5])
    tremor_min_intensity[t] = np.min(Y_features_on[ind_hour, 5])
    tremor_max_intensity[t] = np.max(Y_features_on[ind_hour, 5])
tremor_intensity = np.vstack((tremor_mean_intensity, tremor_min_intensity, tremor_max_intensity))
tremor_intensity_measure = np.vstack((np.ones((len(tremor_mean_intensity),1))*1, np.ones((len(tremor_mean_intensity),1))*2, np.ones((len(tremor_mean_intensity),1))*3))
    
tremor_intensity_wheel = np.hstack((tremor_intensity, tremor_intensity_measure))
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
hbv012_time_wheel = pd.DataFrame(data = tremor_intensity_wheel, columns=["tremor_intensity_dB", "measure"])
hbv012_time_wheel['hour'] = hours_array

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.line_polar(hbv012_time_wheel, r='tremor_intensity_dB', theta='hour', color = "measure", line_close=True, title='Time of Day', width=600, height=500)
fig.show()


########################## Add the prediction from the prototypes###########

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
import pickle 
with open('save_single_proto_trained_visits.pkl', 'rb') as f:
    W_in_sample = pickle.load(f)
    phi_in_sample = pickle.load(f)
    C_in_sample = pickle.load(f)
    Beta_in_sample = pickle.load(f)
    
    
mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv012']; 
data_features_with_segments = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Time","id_segment"])
data_features_with_segments = data_features_with_segments.dropna()
data_features = data_features_with_segments.drop(columns = ["Time", "id_segment"])

id_participant = 0 # Index the participant follow up
data_test_follow_up = np.array(data_features)

phi_test = compute_phi(data_test_follow_up.T, C_in_sample, Beta_in_sample)
phi_test[np.isnan(phi_test)] = 0
T = len(data_test_follow_up)
score_cat = np.zeros([T, 2])
for c in range(2):
    score_cat[:,c] = np.matmul(W_in_sample[:,c].reshape(-1,+1).T, phi_test.T)
    
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
            
predictions_follow_up = Y_hat_test
prob_predictions_follow_up = score_cat

for t in range(len(prob_predictions_follow_up)):
    if prob_predictions_follow_up[t,0]>global_threshold[id_participant]:
        predictions_follow_up[t] = 0
    else:
        predictions_follow_up[t] = 1   

h=100
mean_phi = np.zeros((len(prob_predictions_follow_up),7))
for proto_id in range(7):
    mean_phi[:,proto_id] = np.mean(phi_test[:,proto_id*h:(proto_id+1)*h], axis=1)

import pickle
with open('save_mean_phi.pkl', 'wb') as f:
    pickle.dump(mean_phi, f)


id_indicator = np.ones((len(prob_predictions_follow_up), 1))*(id_participant+1)
        
#hbv002_follow_up_tremor = np.hstack((Y_features, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

# Convert matlab time to python
from datetime import datetime
from datetime import timedelta

mat_data = 'Features_per_second_hbv012.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv012 = mat_contents['accel_time_stamps_hbv012_str']; 
time_stamps_hbv012_hours = np.zeros((len(time_stamps_hbv012), 1))
for t in range(len(time_stamps_hbv012)):
    datem = datetime.strptime(time_stamps_hbv012[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv012_hours[t] = datem.hour

    
hbv012_follow_up_tremor = np.hstack((Y_features, time_stamps_hbv012_hours, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

hbv012_follow_up_tremor_df = pd.DataFrame(data=hbv012_follow_up_tremor, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","id_segment", "Time", "Tremor", "Probability_tremor_0","Probability_tremor_1", "ID"])


# Create a summary plot displaying the tremor duration, the tremor severity and the tremor profile throught the different times of day
# Time of day and time of the week wheel - https://stackoverflow.com/questions/40352607/time-wheel-in-python3-pandas
ind_tremor_on = np.where(predictions_follow_up == 1)
tremor_hours = np.unique(time_stamps_hbv012_hours[ind_tremor_on])
num_bin = len(np.unique(time_stamps_hbv012_hours[ind_tremor_on]))
tremor_duration_sec = np.histogram(time_stamps_hbv012_hours[ind_tremor_on], num_bin)
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
tremor_duration_sec_per_hour = tremor_duration_sec[0].reshape(-1,+1)
hbv012_time_wheel = pd.DataFrame(data = tremor_duration_sec_per_hour, columns=["duration_sec"])
hbv012_time_wheel['hour'] = hours_array

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.line_polar(hbv012_time_wheel, r='duration_sec', theta='hour', line_close=True, title='Time of Day', width=600, height=500)
fig.show()










###############################################################################
mat_data = 'Features_per_second_hbv013.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['accel_features_hbv013']; 
data_features_with_segments = pd.DataFrame(data=Y_features, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "id_segment"])
data_features_with_segments = data_features_with_segments.dropna()
data_features = data_features_with_segments.drop(columns = "id_segment")

id_participant = 1 # Index the participant follow up
data_test_follow_up = np.array(data_features)
prob_predictions_follow_up = logistic_in_sample.predict_proba(data_test_follow_up)
predictions_follow_up = logistic_in_sample.predict(data_test_follow_up)
for t in range(len(prob_predictions_follow_up)):
    if prob_predictions_follow_up[t,0]>global_threshold[id_participant]:
        predictions_follow_up[t] = 0
    else:
        predictions_follow_up[t] = 1  
        
id_indicator = np.ones((len(prob_predictions_follow_up), 1))*(id_participant+1)

        
#hbv002_follow_up_tremor = np.hstack((Y_features, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

# Convert matlab time to python
from datetime import datetime
from datetime import timedelta

mat_data = 'Features_per_second_hbv013.mat' #Load a structure with the features during follow up
mat_contents = sio.loadmat(mat_data)
time_stamps_hbv013 = mat_contents['accel_time_stamps_hbv013_str']; 
time_stamps_hbv013_hours = np.zeros((len(time_stamps_hbv013), 1))
for t in range(len(time_stamps_hbv013)):
    datem = datetime.strptime(time_stamps_hbv013[t], "%d-%m-%Y %H:%M:%S")
    time_stamps_hbv013_hours[t] = datem.hour

    
hbv013_follow_up_tremor = np.hstack((Y_features, time_stamps_hbv013_hours, predictions_follow_up.reshape(-1,+1), prob_predictions_follow_up, id_indicator))

hbv013_follow_up_tremor_df = pd.DataFrame(data=hbv013_follow_up_tremor, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10", "Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45","id_segment", "Time", "Tremor", "Probability_tremor_0","Probability_tremor_1", "ID"])


# Create a summary plot displaying the tremor duration, the tremor severity and the tremor profile throught the different times of day
# Time of day and time of the week wheel - https://stackoverflow.com/questions/40352607/time-wheel-in-python3-pandas
ind_tremor_on = np.where(predictions_follow_up == 1)
tremor_hours = np.unique(time_stamps_hbv013_hours[ind_tremor_on])
num_bin = len(np.unique(time_stamps_hbv013_hours[ind_tremor_on]))
tremor_duration_sec = np.histogram(time_stamps_hbv013_hours[ind_tremor_on], num_bin)
hours_array = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
tremor_duration_sec_per_hour = tremor_duration_sec[0].reshape(-1,+1)
hbv013_time_wheel = pd.DataFrame(data = tremor_duration_sec_per_hour, columns=["duration_sec"])
hbv013_time_wheel['hour'] = hours_array

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.line_polar(hbv013_time_wheel, r='duration_sec', theta='hour', line_close=True, title='Time of Day', width=600, height=500)
fig.show()

