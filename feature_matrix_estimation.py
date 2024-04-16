# -*- coding: utf-8 -*-
"""

@author: Jordan Raykov

Signal processing feature estimation for tremor detection/estimation pipelines
"""

import numpy as np
from scipy.signal import spectrogram, welch
from scipy.stats import entropy

def feature_estimate(Y_tremor, Fs, Ts = 1, featureMatrixType=1):
    windowSize = int(Ts * Fs)
    Num_windows = len(Y_tremor) // windowSize
    
    Yx = Y_tremor[:, 0]
    tremor_feature_STDx = np.zeros((Num_windows,1))
    tremor_feature_entropyx = np.zeros((Num_windows, 1))
    for t in range(Num_windows):
        window_data = Yx[t * windowSize : (t+1) * windowSize]
        tremor_feature_STDx[t] = np.std(window_data)
        tremor_feature_entropyx[t] = entropy(np.histogram(window_data, bins='auto')[0], base=2)  # Compute Shannon entropy
    
    Yy = Y_tremor[:, 1]
    tremor_feature_STDy = np.zeros(Num_windows)
    tremor_feature_entropyy = np.zeros(Num_windows)
    for t in range(Num_windows):
        window_data = Yy[t * windowSize : (t+1) * windowSize]
        tremor_feature_STDy[t] = np.std(window_data)
        tremor_feature_entropyy[t] = entropy(np.histogram(window_data, bins='auto')[0], base=2)  # Compute Shannon entropy
    
    Yz = Y_tremor[:, 2]
    tremor_feature_STDz = np.zeros(Num_windows)
    tremor_feature_entropyz = np.zeros(Num_windows)
    for t in range(Num_windows):
        window_data = Yz[t * windowSize : (t+1) * windowSize]
        tremor_feature_STDz[t] = np.std(window_data)
        tremor_feature_entropyz[t] = entropy(np.histogram(window_data, bins='auto')[0], base=2)  # Compute Shannon entropy
    
    ## Compute some spectral features for axis x
    if len(Y_tremor[:, 0]) > windowSize:
        interval_count = len(Y_tremor) // (Fs * Ts)
        interval_length = Fs * Ts
        
        ## x-axis
        YYx = np.zeros((interval_count, interval_length))
        for i in range(interval_count):
            YYx[i, :] = Y_tremor[(i * interval_length):((i + 1) * interval_length), 0]
        
        pxx_x = np.zeros((interval_count, windowSize // 2 + 1))
        for kk in range(interval_count):
            f_x, pxx_x[kk, :] = welch(YYx[kk, :], fs=Fs, nperseg=windowSize)
        
        Fmax = 15  # Hz
        ind_x = f_x < Fmax
        pxx_select_x = pxx_x[:, ind_x]
        pxx_reshape_x = pxx_select_x.T
        
        t = np.arange(0, (YYx.shape[1] / Fs) * YYx.shape[0], YYx.shape[1] / Fs)
        
        gait_freq = (f_x > 0.3) & (f_x < 2)
        tremor_freq = (f_x > 4) & (f_x < 8)
        high_tremor_freq = (f_x > 8) & (f_x < 12)
        total_energy_freq = (f_x > 0.3) & (f_x < 12)
        
      #  gait_max_power_x = np.max(pxx_reshape_x[gait_freq, :], axis=0)
      #  tremor_max_power_x = np.max(pxx_reshape_x[tremor_freq, :], axis=0)
      #  high_tremor_max_power_x = np.max(pxx_reshape_x[high_tremor_freq, :], axis=0)
      #  total_max_power_x = np.max(pxx_reshape_x, axis=0)
        
        gait_max_power_x = np.max(10 * np.log10(pxx_reshape_x[gait_freq, :]), axis=0)
        ind_max_power_gait_x = np.argmax(pxx_reshape_x[gait_freq, :], axis=0)
        tremor_max_power_x = np.max(10 * np.log10(pxx_reshape_x[tremor_freq, :]), axis=0)
        ind_max_power_tremor_x = np.argmax(pxx_reshape_x[tremor_freq, :], axis=0)
        high_tremor_max_power_x = np.max(10 * np.log10(pxx_reshape_x[high_tremor_freq, :]), axis=0)
        ind_max_power_high_tremor_x = np.argmax(pxx_reshape_x[high_tremor_freq, :], axis=0)
        total_max_power_x = np.max(10 * np.log10(pxx_reshape_x[total_energy_freq, :]), axis=0)
        ind_max_power_total_x = np.argmax(pxx_reshape_x[total_energy_freq, :], axis=0)
        
        gait_power_x = np.sum(10 * np.log10(pxx_reshape_x[gait_freq, :]), axis=0) / np.sum(gait_freq)
        tremor_power_x = np.sum(10 * np.log10(pxx_reshape_x[tremor_freq, :]), axis=0) / np.sum(tremor_freq)
        high_tremor_power_x = np.sum(10 * np.log10(pxx_reshape_x[high_tremor_freq, :]), axis=0) / np.sum(high_tremor_freq)
        total_power_x = np.sum(10 * np.log10(pxx_reshape_x[total_energy_freq, :]), axis=0) / np.sum(total_energy_freq)
        
        sx, f, tx, px = spectrogram(Y_tremor[:, 0], fs=Fs, window='hamming', nperseg=int(Fs), noverlap=0)
        px = np.abs(sx)**2 / (int(Fs) * 0.5)
        se_x = entropy(px, axis=0)
        
        ## y-axis
        YYy = np.zeros((interval_count, interval_length))
        for i in range(interval_count):
            YYy[i, :] = Y_tremor[(i * interval_length):((i + 1) * interval_length), 1]
        
        pxx_y = np.zeros((interval_count, windowSize // 2 + 1))
        for kk in range(interval_count):
            f_y, pxx_y[kk, :] = welch(YYy[kk, :], fs=Fs, nperseg=windowSize)
        
        Fmax = 15  # Hz
        ind_y = f_y < Fmax
        pxx_select_y = pxx_y[:, ind_y]
        pxx_reshape_y = pxx_select_y.T
        
        t = np.arange(0, (YYy.shape[1] / Fs) * YYy.shape[0], YYy.shape[1] / Fs)
        
        gait_freq = (f_y > 0.3) & (f_y < 2)
        tremor_freq = (f_y > 4) & (f_y < 8)
        high_tremor_freq = (f_y > 8) & (f_y < 12)
        total_energy_freq = (f_y > 0.3) & (f_y < 12)
        
        gait_max_power_y = np.max(10 * np.log10(pxx_reshape_y[gait_freq, :]), axis=0)
        ind_max_power_gait_y = np.argmax(pxx_reshape_y[gait_freq, :], axis=0)
        tremor_max_power_y = np.max(10 * np.log10(pxx_reshape_y[tremor_freq, :]), axis=0)
        ind_max_power_tremor_y = np.argmax(pxx_reshape_y[tremor_freq, :], axis=0)
        high_tremor_max_power_y = np.max(10 * np.log10(pxx_reshape_y[high_tremor_freq, :]), axis=0)
        ind_max_power_high_tremor_y = np.argmax(pxx_reshape_y[high_tremor_freq, :], axis=0)
        total_max_power_y = np.max(10 * np.log10(pxx_reshape_y[total_energy_freq, :]), axis=0)
        ind_max_power_total_y = np.argmax(pxx_reshape_y[total_energy_freq, :], axis=0)
        
        gait_power_y = np.sum(10 * np.log10(pxx_reshape_y[gait_freq, :]), axis=0) / np.sum(gait_freq)
        tremor_power_y = np.sum(10 * np.log10(pxx_reshape_y[tremor_freq, :]), axis=0) / np.sum(tremor_freq)
        high_tremor_power_y = np.sum(10 * np.log10(pxx_reshape_y[high_tremor_freq, :]), axis=0) / np.sum(high_tremor_freq)
        total_power_y = np.sum(10 * np.log10(pxx_reshape_y[total_energy_freq, :]), axis=0) / np.sum(total_energy_freq)
        
        sy, f, ty, py = spectrogram(Y_tremor[:, 1], fs=Fs, window='hamming', nperseg=int(Fs), noverlap=0)
        py = np.abs(sy)**2 / (int(Fs) * 0.5)
        se_y = entropy(py, axis=0)
        
        ## z-axis
        YYz = np.zeros((interval_count, interval_length))
        for i in range(interval_count):
            YYz[i, :] = Y_tremor[(i * interval_length):((i + 1) * interval_length), 2]
        
        pxx_z = np.zeros((interval_count, windowSize // 2 + 1))
        for kk in range(interval_count):
            f_z, pxx_z[kk, :] = welch(YYz[kk, :], fs=Fs, nperseg=windowSize)
        
        Fmax = 15  # Hz
        ind_z = f_z < Fmax
        pxx_select_z = pxx_z[:, ind_z]
        pxx_reshape_z = pxx_select_z.T
        
        t = np.arange(0, (YYz.shape[1] / Fs) * YYz.shape[0], YYz.shape[1] / Fs)
        
        gait_freq = (f_z > 0.3) & (f_z < 2)
        tremor_freq = (f_z > 4) & (f_z < 8)
        high_tremor_freq = (f_z > 8) & (f_z < 12)
        total_energy_freq = (f_z > 0.3) & (f_z < 12)
        
        gait_max_power_z = np.max(10 * np.log10(pxx_reshape_z[gait_freq, :]), axis=0)
        ind_max_power_gait_z = np.argmax(pxx_reshape_z[gait_freq, :], axis=0)
        tremor_max_power_z = np.max(10 * np.log10(pxx_reshape_z[tremor_freq, :]), axis=0)
        ind_max_power_tremor_z = np.argmax(pxx_reshape_z[tremor_freq, :], axis=0)
        high_tremor_max_power_z = np.max(10 * np.log10(pxx_reshape_z[high_tremor_freq, :]), axis=0)
        ind_max_power_high_tremor_z = np.argmax(pxx_reshape_z[high_tremor_freq, :], axis=0)
        total_max_power_z = np.max(10 * np.log10(pxx_reshape_z[total_energy_freq, :]), axis=0)
        ind_max_power_total_z = np.argmax(pxx_reshape_z[total_energy_freq, :], axis=0)
        
        gait_power_z = np.sum(10 * np.log10(pxx_reshape_z[gait_freq, :]), axis=0) / np.sum(gait_freq)
        tremor_power_z = np.sum(10 * np.log10(pxx_reshape_z[tremor_freq, :]), axis=0) / np.sum(tremor_freq)
        high_tremor_power_z = np.sum(10 * np.log10(pxx_reshape_z[high_tremor_freq, :]), axis=0) / np.sum(high_tremor_freq)
        total_power_z = np.sum(10 * np.log10(pxx_reshape_z[total_energy_freq, :]), axis=0) / np.sum(total_energy_freq)
        
        sz, f, tz, pz = spectrogram(Y_tremor[:, 2], fs=Fs, window='hamming', nperseg=int(Fs), noverlap=0)
        pz = np.abs(sz)**2 / (int(Fs) * 0.5)
        se_z = entropy(pz, axis=0)
        
        if featureMatrixType == 2:   
            cepstral_coeffs_x = cepstral_coefficients()
            cepstral_coeffs_y = cepstral_coefficients()
            cepstral_coeffs_z = cepstral_coefficients()
            
        ## Create the appropriate format feature matrix
        if featureMatrixType == 1:
            tremor_features = np.concatenate((tremor_feature_STDx, tremor_feature_entropyx, gait_max_power_x.T, gait_power_x.T, tremor_max_power_x.T, tremor_power_x.T, high_tremor_max_power_x.T, high_tremor_power_x.T, 
                                              total_max_power_x.T, total_power_x.T, se_x, f_x[ind_max_power_gait_x], f_x[ind_max_power_tremor_x], f_x[ind_max_power_high_tremor_x], f_x[ind_max_power_total_x],
                                              tremor_feature_STDy, tremor_feature_entropyy, gait_max_power_y.T, gait_power_y.T, tremor_max_power_y.T, tremor_power_y.T, high_tremor_max_power_y.T, high_tremor_power_y.T, 
                                              total_max_power_y.T, total_power_y.T, se_y, f_y[ind_max_power_gait_y], f_y[ind_max_power_tremor_y], f_y[ind_max_power_high_tremor_y], f_y[ind_max_power_total_y],
                                              tremor_feature_STDz, tremor_feature_entropyz, gait_max_power_z.T, gait_power_z.T, tremor_max_power_z.T, tremor_power_z.T, high_tremor_max_power_z.T, high_tremor_power_z.T, 
                                              total_max_power_z.T, total_power_z.T, se_z, f_z[ind_max_power_gait_z], f_z[ind_max_power_tremor_z], f_z[ind_max_power_high_tremor_z], f_z[ind_max_power_total_z]),axis=0)
        elif featureMatrixType == 2:   
            tremor_features = np.concatenate((tremor_feature_STDx, tremor_feature_entropyx, gait_max_power_x.T, gait_power_x.T, tremor_max_power_x.T, tremor_power_x.T, high_tremor_max_power_x.T, high_tremor_power_x.T, 
                                              total_max_power_x.T, total_power_x.T, se_x, f_x[ind_max_power_gait_x], f_x[ind_max_power_tremor_x], f_x[ind_max_power_high_tremor_x], f_x[ind_max_power_total_x], cepstral_coeffs_x,
                                              tremor_feature_STDy, tremor_feature_entropyy, gait_max_power_y.T, gait_power_y.T, tremor_max_power_y.T, tremor_power_y.T, high_tremor_max_power_y.T, high_tremor_power_y.T, 
                                              total_max_power_y.T, total_power_y.T, se_y, f_y[ind_max_power_gait_y], f_y[ind_max_power_tremor_y], f_y[ind_max_power_high_tremor_y], f_y[ind_max_power_total_y], cepstral_coeffs_y,
                                              tremor_feature_STDz, tremor_feature_entropyz, gait_max_power_z.T, gait_power_z.T, tremor_max_power_z.T, tremor_power_z.T, high_tremor_max_power_z.T, high_tremor_power_z.T, 
                                              total_max_power_z.T, total_power_z.T, se_z, f_z[ind_max_power_gait_z], f_z[ind_max_power_tremor_z], f_z[ind_max_power_high_tremor_z], f_z[ind_max_power_total_z], cepstral_coeffs_z),axis=0)
        return 
    
    
def cepstral_coefficients(
    self,
    total_power_col: str, low_frequency, high_frequency, filter_length, n_dct_filters):

    self.print_process(self.cepstral_coefficients)
        
    # compute filter points
    freqs = np.linspace(low_frequency, high_frequency, num=filter_length+2)
    filter_points = np.floor((self.window_length + 1) / self.resampling_frequency * freqs).astype(int)  

    # construct filterbank
    filters = np.zeros((len(filter_points)-2, int(self.window_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
        filters[j, filter_points[j+1] : filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])

    # filter signal 
    power_filtered = self.df_windows[total_power_col].apply(lambda x: np.dot(filters, x))
    log_power_filtered = power_filtered.apply(lambda x: 10.0 * np.log10(x))

    # generate cepstral coefficients
    dct_filters = np.empty((n_dct_filters, filter_length))
    dct_filters[0, :] = 1.0 / np.sqrt(filter_length)

    samples = np.arange(1, 2 * filter_length, 2) * np.pi / (2.0 * filter_length)

    for i in range(1, n_dct_filters):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_length)

    cepstral_coefs = log_power_filtered.apply(lambda x: np.dot(dct_filters, x))

    return pd.DataFrame(np.vstack(cepstral_coefs), columns=['cc_{}'.format(j+1) for j in range(n_dct_filters)])