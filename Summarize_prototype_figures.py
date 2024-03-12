# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:00:53 2023

@author: pmzyr
"""

tpr_per_strata = np.zeros((8,1))
counter = np.zeros((8,1))
sum_xx = np.zeros((8,1))
var_x = np.zeros((8,1))
for i in range(8):
    curr_tpr = TPR_proto_out_of_sample[i]
    for j in range(8):
        if curr_tpr[j]>0:
            tpr_per_strata[j] = tpr_per_strata[j] + curr_tpr[j]
            sum_xx[j] = sum_xx[j] + curr_tpr[j]**2
            counter[j] = counter[j] + 1
            
for i in range(8):
    tpr_per_strata[i] = tpr_per_strata[i]/counter[i]
    sum_xx[i] = sum_xx[i]/counter[i]
    var_x[i] = sum_xx[i] - tpr_per_strata[i]**2
    
tnr_per_strata = np.zeros((8,1))
counter = np.zeros((8,1))
sum_xx = np.zeros((8,1))
var_x = np.zeros((8,1))
for i in range(32):
    curr_tnr = TNR_out_of_sample[i]
    for j in range(8):
        if curr_tnr[j]>0:
            tnr_per_strata[j] = tnr_per_strata[j] + curr_tnr[j]
            sum_xx[j] = sum_xx[j] + curr_tnr[j]**2
            counter[j] = counter[j] + 1
            
for i in range(8):
    tnr_per_strata[i] = tnr_per_strata[i]/counter[i]
    sum_xx[i] = sum_xx[i]/counter[i]
    var_x[i] = sum_xx[i] - tnr_per_strata[i]**2
