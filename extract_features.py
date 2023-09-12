import os
import pandas as pd
import librosa
import librosa.display
import glob 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import logging
import pickle
from sklearn.mixture import GaussianMixture
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import lfilter

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.append(os.getcwd() + '/antispoof_methods/')
sys.path.append(os.getcwd() + '/antispoof_methods/CQCC/')
# sys.path.append(os.getcwd() + '/antispoof_methods/LFCC/')

from antispoof_methods.CQCC.CQT_toolbox_2013.cqt import cqt
from antispoof_methods.gmm import extract_cqcc
from antispoof_methods.LFCC_pipeline import lfcc
# from antispoof_methods.LFCC.python.gmm import Deltas

# feature extraction functions
def Deltas(x, width=3):
    hlen = int(np.floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]


def extract_mfcc(audio_data, sr):

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr)

    return mfccs.T

def extract_dwt(audio_data, wavelet_name='db1'):

    wvlt = pywt.Wavelet(wavelet_name)

    coeffs = pywt.wavedec(audio_data, wvlt, mode='constant', level=5)


############## min of coeff and clip longer coefficients at the end  ##############

    # coeff_sizes = [cf.shape[0] for cf in coeffs]
    # min_coeff = min(coeff_sizes)

    # coeffs_ls = []
    # for cf in coeffs:
    #     N = cf.shape[0] - min_coeff
    #     # cf_padded = np.pad(cf, (0, N), 'constant')
    #     cf_clipped = cf[:min_coeff]
    #     coeffs_ls.append(cf_clipped)

    
    # coeffs_array = np.vstack(coeffs_ls)

    # return coeffs_array.T

############## max of coeff and pad zeros at the end  ##############

    # coeff_sizes = [cf.shape[0] for cf in coeffs]
    # max_coeff = max(coeff_sizes)

    # coeffs_ls = []
    # for cf in coeffs:
    #     N = max_coeff - cf.shape[0]
    #     cf_padded = np.pad(cf, (0, N), 'constant')
    #     coeffs_ls.append(cf_padded)

    
    # coeffs_array = np.vstack(coeffs_ls)

    # return coeffs_array.T

############### Original ################

    sigma_feat = []
    skew_feat = []
    kurt_feat = []

    for cf in coeffs:
        # print(cf.shape)
        sigma_feat.append(np.log(np.var(cf)))
        skew_feat.append(skew(cf))
        kurt_feat.append(kurtosis(cf))
    
    return np.array(sigma_feat + skew_feat + kurt_feat).reshape(1, -1)

################################################


def extract_lfcc(audio_data, sr, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000):

    lfccs = lfcc(sig=audio_data,
                 fs=sr,
                 num_ceps=num_ceps,
                 low_freq=low_freq,
                 high_freq=high_freq).T
    
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = np.vstack(feats)
    
    return lfccs.T



def extract_features(audio_file, features='cqcc'):

    audio_data, sr = librosa.load(audio_file, res_type='kaiser_fast')
    audio_data = librosa.util.normalize(audio_data)
    
    if features == 'cqcc':
        return extract_cqcc(audio_data, sr)
    
    if features == 'dwt':
        return extract_dwt(audio_data)
    
    if features == 'lfcc':
        return extract_lfcc(audio_data, sr)
    
    if features == 'mfcc':
        return extract_mfcc(audio_data, sr)
    

def extract_features_all(audio_data_dir, data_type, class_type, features='cqcc', audio_features_folder='./audio_features/'):

    # input data path
    data_dir = audio_data_dir + data_type + class_type

    # output data path
    out_folder = audio_features_folder + features + '/' + data_type
    out_file = out_folder + class_type
    print(out_file)

    wav_files = list(glob.glob(os.path.join(data_dir + '/', '*.wav')))
    wav_files.sort()
    print(wav_files)
    print(len(wav_files))

    audio_features = []
    if not os.path.exists(out_file + '.npy'):
    
        audio_features_ls = []
        for i, wf in enumerate(wav_files):
            print('Done Extracting and Saving audio file {}'.format(i))
            audio_features = extract_features(wf, features=features)

            print(audio_features.shape)
            # print(audio_features)

            audio_features_ls.append(audio_features)

        audio_features = np.vstack(audio_features_ls)

        # print(cqcc_features)
        print(audio_features.shape)

        if not os.path.exists(out_folder):
                os.makedirs(out_folder)

        np.save(out_file, audio_features)

    else:
        print("{} features already exist".format(features))

    return audio_features



def train_gmm(data_label, n_comp, data_path, features='cqcc', model_path='models/'):

    # load data
    train_data = np.load(data_path)

    print(train_data.shape)

    logging.info('Start GMM training.')

    gmm_dict_file = '_'.join(('gmm', data_label, features, str(n_comp), 'clippeddwt', '.pkl'))

    # gmm_dict_file = '_'.join(('gmm', data_label, features, str(n_comp), '.pkl'))

    gmm = GaussianMixture(n_components=n_comp,
                              random_state=None,
                              covariance_type='diag',
                              max_iter=100,
                              verbose=2,
                              verbose_interval=1)
    
    # gmm.means_init = train_data.mean(axis=0)
         
    gmm.fit(train_data)

    model_save_path = model_path + gmm_dict_file
    with open(model_save_path, "wb") as f:
        pickle.dump(gmm, f)

    return gmm


def train_svm(laser_data_path, original_data_path, data_dist, features='cqcc', model_path='models/'):

    # load data
    laser_data = np.load(laser_data_path)
    original_data = np.load(original_data_path)

    train_laser_label = np.array(['laser']*laser_data.shape[0])
    train_orig_label = np.array(['original']*original_data.shape[0])

    print(laser_data.shape)
    print(original_data.shape)

    print('Start SVM training.')

    # svm_dict_file = '_'.join(('svm', data_label, features, str(n_comp), 'clippeddwt', '.pkl'))

    svm_dict_file = '_'.join(('svm', features, 'SVC', data_dist, '.pkl'))

    # svm = make_pipeline(StandardScaler(), LinearSVC(verbose=2, random_state=0, dual=True))
    svm = make_pipeline(StandardScaler(), SVC(verbose=True, random_state=0))

    X_train = np.concatenate((laser_data, original_data))
    print(X_train.shape)

    y_train = np.concatenate((train_laser_label, train_orig_label))
    print(y_train.shape)

    svm.fit(X_train, y_train)

    print(svm[1].get_params())

    model_save_path = model_path + svm_dict_file
    with open(model_save_path, "wb") as f:
        pickle.dump(svm, f)

    return svm



if __name__ == "__main__":

    ########### Declarations #############
    audio_dir = '/home/hashim/PHD/audio_data/AllAudioSamples/'

    data_type = 'train/'

    data_dist = 'random'

    class_type = 'original'

    data_dir = audio_dir + data_type + class_type

    features = 'dwt' # cqcc, dwt, lfcc, mfcc

    audio_features_folder = './audio_features/'

    train_GMM = False

    train_SVM = True



    ############# feature Extraction ##########

    audio_features = extract_features_all(audio_dir, data_type, class_type, features=features, audio_features_folder=audio_features_folder)


    ############# training GMM #############

    if train_GMM:
        features_path = audio_features_folder + features + '/' + data_type + class_type + '.npy'
        gmm = train_gmm(class_type, 16, features_path, features=features, model_path='./models/')
        
    if train_SVM:
        laser_path = audio_features_folder + features + '/' + data_type + 'laser' + '.npy'
        orig_path = audio_features_folder + features + '/' + data_type + 'original' + '.npy'
        
        svm = train_svm(laser_path, orig_path, data_dist, features=features)


