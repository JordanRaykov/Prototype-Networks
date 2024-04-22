# -*- coding: utf-8 -*-
"""
Utility functions for estimating tremor features and composing the Prototypical neural networks for tremor classification

@authors: Yordan P. Raykov, Luc JW. Evers
"""

import numpy as np
from scipy.signal import spectrogram, welch
from scipy.stats import entropy
import h5py

from scipy.special import gammaln
from numpy.linalg import slogdet

from numpy.matlib import repmat
from numpy.linalg import inv

import random

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

def load_matlab_data(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for k, v in f.items():
            data[k] = {name: np.array(v2) for name, v2 in v.items()}
    return data

def load_PD_at_home_data(phys_data_matlab, labels_data_matlab, Ts = 2, Fs = 200):
    # This function loads example PD@Home data used for training of tremor classifiers
    # Arguments: phys_data - MATLAB structure containing raw sensor data from Physilog sensors, see Evers et al. 2020 for details
    #            labels_data - MATLAB structure containing video annotations of activities of daily living and annotated prototypes of tremor, see Evers et al. 2020 and "Insert Reference" for details
    
    phys_data = load_matlab_data(phys_data_matlab)
    labels_data = load_matlab_data(labels_data_matlab)
    
    IDs = ["hbv012", "hbv013", "hbv017", "hbv018", "hbv022", "hbv023", "hbv038", "hbv090"]
    Arms = ["Right", "Left", "Left", "Right", "Right", "Left", "Right", "Left"]
    Fs = 200
    tremor_features_with_tremor = []

    for n in range(len(IDs)):
        for id in range(25):
            if phys_data[id]['id'][0][0] == IDs[n]:
                current_id_phys = id
            if labels_data[id]['id'][0][0] == IDs[n]:
                current_id_labels = id

        if Arms[n] == "Left":
            ind_arm = 'LW'
        else:
            ind_arm = 'RW'

        Yl1 = labels_data[current_id_labels]['premed_tremorfinal'][0][0]
        Yl2 = labels_data[current_id_labels]['postmed_tremorfinal'][0][0]
        premed_tremor_start = labels_data[current_id_labels]['premed_tremorstart'][0][0] * Fs
        premed_tremor_end = labels_data[current_id_labels]['premed_tremorend'][0][0] * Fs
        activity_tremor_premed_gap_start = int(np.floor((labels_data[current_id_labels]['premed_tremorstart'][0][0] - labels_data[current_id_labels]['premedstart'][0][0]) * Fs))
        activity_tremor_premed_gap_end = int(np.floor((labels_data[current_id_labels]['premed_tremorend'][0][0] - labels_data[current_id_labels]['premedend'][0][0]) * Fs))

        Y_activity_label_premed = labels_data[current_id_labels]['premedfinal'][0][0]
        if activity_tremor_premed_gap_start > 0:
            Y_activity_label_premed = Y_activity_label_premed[activity_tremor_premed_gap_start:]
        else:
            Yl1 = Yl1[-activity_tremor_premed_gap_start:]
            premed_tremor_start -= activity_tremor_premed_gap_start

        if activity_tremor_premed_gap_end < 0:
            Y_activity_label_premed = Y_activity_label_premed[:activity_tremor_premed_gap_end]
        else:
            Yl1 = Yl1[:-activity_tremor_premed_gap_end]
            premed_tremor_end -= activity_tremor_premed_gap_end

        postmed_tremor_start = labels_data[current_id_labels]['postmed_tremorstart'][0][0] * Fs
        activity_tremor_postmed_gap_start = int(np.floor((labels_data[current_id_labels]['postmed_tremorstart'][0][0] - labels_data[current_id_labels]['postmedstart'][0][0]) * Fs))
        postmed_tremor_end = labels_data[current_id_labels]['postmed_tremorend'][0][0] * Fs
        activity_tremor_postmed_gap_end = int(np.floor((labels_data[current_id_labels]['postmed_tremorend'][0][0] - labels_data[current_id_labels]['postmedend'][0][0]) * Fs))

        Y_activity_label_postmed = labels_data[current_id_labels]['postmedfinal'][0][0]
        if activity_tremor_postmed_gap_start > 0:
            Y_activity_label_postmed = Y_activity_label_postmed[activity_tremor_postmed_gap_start:]
        else:
            Yl2 = Yl2[-activity_tremor_postmed_gap_start:]
            postmed_tremor_start -= activity_tremor_postmed_gap_start

        if activity_tremor_postmed_gap_end < 0:
            Y_activity_label_postmed = Y_activity_label_postmed[:activity_tremor_postmed_gap_end]
        else:
            Yl2 = Yl2[:-activity_tremor_postmed_gap_end]
            postmed_tremor_end -= activity_tremor_postmed_gap_end

        Ylabel = np.concatenate((Yl1, Yl2))
        Y_activity_label = np.concatenate((Y_activity_label_premed, Y_activity_label_postmed))

        tremor_type_concat = []
        tremor_data_concat = []
        num_proto = len(labels_data[current_id_labels]['table_tremorproto'][0][0]['Start'][0])

        Y_proto = np.zeros(len(phys_data[current_id_phys][ind_arm][0][0]['accel'][:, 0]))
        for t in range(num_proto):
            tremor_type_start = labels_data[current_id_labels]['table_tremorproto'][0][0]['Start'][0][t]
            tremor_type_end = labels_data[current_id_labels]['table_tremorproto'][0][0]['End'][0][t]
            tremor_type = labels_data[current_id_labels]['table_tremorproto'][0][0]['Code'][0][t]
            tremor_ind = (phys_data[current_id_phys][ind_arm][0][0]['accel'][:, 0] > tremor_type_start) & (phys_data[current_id_phys][ind_arm][0][0]['accel'][:, 0] < tremor_type_end)
            Y_proto[tremor_ind] = int(tremor_type)
            tremor_data_per_type = phys_data[current_id_phys][ind_arm][0][0]['accel'][tremor_ind, 1:4]
            tremor_type_concat = np.ones(len(tremor_data_per_type)) * int(tremor_type)
            tremor_type_concat = np.concatenate((tremor_type_concat, tremor_type_concat))
            tremor_data_concat = np.concatenate((tremor_data_concat, phys_data[current_id_phys][ind_arm][0][0]['accel'][tremor_ind, 1:4]))

        Y_proto = np.concatenate((Y_proto[int(premed_tremor_start):int(premed_tremor_end)], Y_proto[int(postmed_tremor_start):int(postmed_tremor_end)]))
        Y_tremor_premed = phys_data[current_id_phys][ind_arm][0][0]['accel'][int(np.floor(premed_tremor_start)):int(round(premed_tremor_end)), 1:4]
        Y_tremor_postmed = phys_data[current_id_phys][ind_arm][0][0]['accel'][int(np.floor(postmed_tremor_start)):int(round(postmed_tremor_end)), 1:4]
        Y_tremor = np.concatenate((Y_tremor_premed * 9.8, Y_tremor_postmed * 9.8))
        Y_med_ind_1 = np.concatenate((np.ones(len(Y_tremor_premed)), np.ones(len(Y_tremor_postmed)) * 2))

        Y_proto_dd = Y_proto
        Y_proto_data = Y_tremor[Y_proto_dd > 0] 
        Y_tremor_features = feature_estimate(Y_tremor, Fs = 200, Ts = 2)
        Y_tremor_features = Y_tremor_features[~np.isnan(Y_tremor_features).any(axis=1)]

        Y_med_ind_1d = np.zeros(len(Y_tremor_features))
        for t in range(int(np.floor(len(Y_med_ind_1) / (Fs * Ts)))):
            Y_med_ind_1d[t] = np.mode(Y_med_ind_1[t * (Fs * Ts):(t + 1) * (Fs * Ts)])

        Y_proto_d = np.zeros(len(Y_tremor_features))
        for t in range(int(np.floor(len(Y_proto) / (Fs * Ts)))):
            Y_proto_d[t] = np.mode(Y_proto[t * (Fs * Ts):(t + 1) * (Fs * Ts)])

        Ylabeld = np.zeros(len(Y_tremor_features))
        for t in range(int(np.floor(len(Ylabel) / (Fs * Ts)))):
            score_1 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '1')
            score_2 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '2')
            score_3 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '3')
            score_97 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '97')

            score_0 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '0')
            score_96 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '96')
            score_98 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '98')
            score_99 = np.sum(Ylabel[t * (Fs * Ts):(t + 1) * (Fs * Ts)] == '99')

            score_tremor = score_1 + score_2 + score_3 + score_97
            score_nontremor = score_0 + score_96 + score_98 + score_99
            if 2 * score_tremor > score_nontremor:
                Ylabeld[t] = 1
            else:
                Ylabeld[t] = 0

        Y_activity_label_down = np.zeros(len(Y_tremor_features))
        for t in range(int(np.floor(len(Y_activity_label) / (Fs * Ts)))):
            Y_activity_label_down[t] = np.mode(Y_activity_label[t * (Fs * Ts):(t + 1) * (Fs * Ts)])


        Y_tremor_features_full = np.column_stack((n * np.ones(len(Y_tremor_features)), Y_tremor_features, Y_med_ind_1d, Y_proto_d, Ylabeld, Y_activity_label_down))

        tremor_features_with_tremor.append({
            'patient_id': IDs[n],
            'features': Y_tremor_features,
            'features_full': Y_tremor_features_full,
            'Proto': Y_proto_data,
            'Proto_labels': Y_proto_dd[Y_proto_dd > 0],
            'Labels_raw': Ylabeld
        })

    return tremor_features_with_tremor


def feature_estimate(Y_tremor, Fs, Ts = 1, featureMatrixType=1):
    # This function estimates one of two different meature matrices descriptive of tremor episodes
    
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

def compute_basis_params(data, protovar1, protovar2, estimation_method = 'data_driven'):    
    
    basis_params = []
    if estimation_method == 'data_driven':
        d = np.shape(data)[1] - 2
        num_proto_1 = len(np.unique(data[protovar1])) - 1
        num_proto_2 = len(np.unique(data[protovar2])) - 1
        C = np.zeros([1, d])
        Beta = np.zeros([1, d])
        mu_basis = np.zeros((d,1))
        sigma_basis = np.zeros((d,1))
        beta = np.zeros([num_proto_1, d])
    
        ###Instantiate prototype based basis centers
        for k in range(num_proto_1):
            id_proto1 = np.where(data[protovar1] == k+1)
            id_proto1 = np.array(id_proto1).reshape(-1) 
            data_features = np.array(data.drop(columns=[protovar1, protovar2]))
            data_features_proto1 = data_features[id_proto1,:]  
            
            mu =  np.array(random.choices(data_features_proto1, k=num_proto_1)).reshape(num_proto_1, d)
            for dd in range(d):
                beta[:,dd] = np.std(np.array(data_features_proto1[:,dd]))*np.ones((num_proto_1,))
                beta[:,dd] = beta[:,dd]*0.1 # choose poriton of the variance scale parameter
           
            C = np.vstack((C, mu))
            Beta = np.vstack((Beta, beta))
    
        beta = np.zeros([num_proto_2,d])
        for kk in range(num_proto_2):
            id_proto2 = np.where(data[protovar2] == kk+1)
            id_proto2 = np.array(id_proto2).reshape(-1)
            data_features_proto2 = data_features[id_proto2,:]  

            mu = np.array(random.choices(data_features_proto2, k=num_proto_2)).reshape(num_proto_2,d)
            for dd in range(d):
                beta[:,dd] = np.std(np.array(data_features_proto2[:,dd]))*np.ones((num_proto_2,))
                beta[:,dd] = beta[:,dd]*0.1
            C = np.vstack((C, mu))
            Beta = np.vstack((Beta, beta))
        
        C = C[1:,:]
        Beta = Beta[1:,:]
    
        basis_params.append(C)
        basis_params.append(Beta)
        return basis_params
    
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