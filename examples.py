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

# Usage example:
#phys_data = sio.loadmat('phys_cur_PD_merged.mat')['phys']
labels_data = sio.loadmat('labels_PD_phys_tremorfog_sharing_prototypes_selection.mat')['labels']
processed_data = load_PD_at_home_data('phys_cur_PD_merged.mat', 'labels_PD_phys_tremorfog_sharing_prototypes_selection.mat', Ts = 2, Fs = 200)


data = {}
with h5py.File(phys_cur_PD_merged.mat, 'r') as f:
    for k, v in f.items():
        data[k] = {name: np.array(v2) for name, v2 in v.items()}
        
with h5py.File('phys_cur_PD_merged.mat', 'r') as f:
    print(f.items())
    
    
# Example of Single-layer Prototypical NN 
   