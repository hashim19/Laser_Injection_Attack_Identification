import numpy as np
import pandas as pd
import os
import glob
import pickle
from collections import defaultdict
from sklearn.metrics import accuracy_score

from extract_features import extract_features



def score_svm(test_dir, feature_folder, features='cqcc', data_dist='random', model_path='./models/'):

    svm_dict_file = '_'.join(('svm', features, 'SVC', data_dist, '.pkl'))
    # svm_dict_file = '_'.join(('svm', features, 'LinearSVC', data_dist, '.pkl'))

    svm = pickle.load(open(model_path + svm_dict_file, 'rb'))

    data_path = feature_folder + '/' + features + '/' + data_type + '/'

    out_filename = 'laser_original_scores_' + features + '_svm_SVC_' + data_dist + '.csv'
    out_fullfile = feature_folder + '/' + features + '/' + out_filename

    ############# read test wav files ###########
    laser_wav_files = list(glob.glob(os.path.join(test_dir + data_type + '/' + 'laser' + '/', '*.wav')))
    laser_wav_files.sort()
    print(laser_wav_files)
    print(len(laser_wav_files))

    orig_wav_files = list(glob.glob(os.path.join(test_dir + data_type + '/' + 'original' + '/', '*.wav')))
    orig_wav_files.sort()
    print(orig_wav_files)
    print(len(orig_wav_files))

    # score_dict = defaultdict(list)

    score_dict = {'Predicted_Label': [], 'GT_Label': []}

    # iterate over laser audios and add results to the scores dictionary
    for i, lf in enumerate(laser_wav_files):

        print('Computing results for laser audio {}'.format(i))

        score_dict['GT_Label'].append('laser')

        audio_features = extract_features(lf, features=features)

        predicted_label = svm.predict(audio_features)[0]

        score_dict['Predicted_Label'].append(predicted_label)

    # iterate over laser audios and add results to the scores dictionary
    for i, of in enumerate(orig_wav_files):

        print('Computing results for original audio {}'.format(i))

        score_dict['GT_Label'].append('original')

        audio_features = extract_features(of, features=features)

        predicted_label = svm.predict(audio_features)[0]

        score_dict['Predicted_Label'].append(predicted_label)

    score_pd = pd.DataFrame(score_dict)

    score_pd.to_csv(out_fullfile)

    return score_pd



def score_gmm(test_dir, feature_folder, n_comp=2, features='cqcc', data_type='test', model_path='./models/'):

    # load models
    # gmm_dict_file_orig = '_'.join(('gmm', 'original', features, str(n_comp), 'clippeddwt', '.pkl')) 
    # gmm_dict_file_laser = '_'.join(('gmm', 'laser', features, str(n_comp), 'clippeddwt', '.pkl'))

    gmm_dict_file_orig = '_'.join(('gmm', 'original', features, str(n_comp), '.pkl'))
    gmm_dict_file_laser = '_'.join(('gmm', 'laser', features, str(n_comp), '.pkl'))


    gmm_original = pickle.load(open(model_path + gmm_dict_file_orig, 'rb'))
    gmm_laser = pickle.load(open(model_path + gmm_dict_file_laser, 'rb'))

    data_path = feature_folder + '/' + features + '/' + data_type + '/'
    
    out_filename = 'laser_original_scores_' + features + str(n_comp) + '_svm_' + '.csv'
    out_fullfile = feature_folder + '/' + features + '/' + out_filename
    
    # read test data
    # laser_data = np.load(data_path + 'laser.npy').T
    # original_data = np.load(data_path + 'original.npy').T

    ############# read test wav files ###########
    laser_wav_files = list(glob.glob(os.path.join(test_dir + data_type + '/' + 'laser' + '/', '*.wav')))
    laser_wav_files.sort()
    print(laser_wav_files)
    print(len(laser_wav_files))

    orig_wav_files = list(glob.glob(os.path.join(test_dir + data_type + '/' + 'original' + '/', '*.wav')))
    orig_wav_files.sort()
    print(orig_wav_files)
    print(len(orig_wav_files))

    # score_dict = defaultdict(list)

    score_dict = {'Score': [], 'Predicted_Label': [], 'GT_Label': [], 'Original_Score': [], 'Laser_Score': []}

    
    # iterate over laser audios and add results to the scores dictionary
    for i, lf in enumerate(laser_wav_files):

        print('Computing results for laser audio {}'.format(i))

        score_dict['GT_Label'].append('laser')

        audio_features = extract_features(lf, features=features)

        original_score = gmm_original.score(audio_features)
        laser_score = gmm_laser.score(audio_features) 

        scr = original_score - laser_score
        score_dict['Score'].append(scr)

        if scr < 0:
            score_dict['Predicted_Label'].append('laser')
        else:
            score_dict['Predicted_Label'].append('original')

        score_dict['Original_Score'].append(original_score)
        score_dict['Laser_Score'].append(laser_score)

    # iterate over original audios and add results to the scores dictionary
    for of in orig_wav_files:

        print('Computing results for original audio {}'.format(i))

        score_dict['GT_Label'].append('original')

        audio_features = extract_features(of, features=features)

        original_score = gmm_original.score(audio_features)
        laser_score = gmm_laser.score(audio_features) 

        scr = original_score - laser_score
        score_dict['Score'].append(scr)

        if scr < 0:
            score_dict['Predicted_Label'].append('laser')
        else:
            score_dict['Predicted_Label'].append('original')

        score_dict['Original_Score'].append(original_score)
        score_dict['Laser_Score'].append(laser_score)

    score_pd = pd.DataFrame(score_dict)

    score_pd.to_csv(out_fullfile)

    return score_pd
        




if __name__ == "__main__":

    audio_data_dir = '/home/hashim/PHD/audio_data/AllAudioSamples/'
    audio_features_folder = './audio_features/'
    features = 'mfcc' # cqcc, dwt, lfcc, mfcc
    data_type = 'test_TI/'
    data_dist = 'TI'

    
    # gmm_scores = score_gmm(audio_data_dir, audio_features_folder, n_comp=16, features=features, model_path='./models/')
    
    # print("accuracy = {}".format(accuracy_score(gmm_scores['GT_Label'], gmm_scores['Predicted_Label'])))

    svm_scores = score_svm(audio_data_dir, audio_features_folder, data_dist=data_dist, features=features, model_path='./models/')

    print("accuracy = {}".format(accuracy_score(svm_scores['GT_Label'], svm_scores['Predicted_Label'])))



