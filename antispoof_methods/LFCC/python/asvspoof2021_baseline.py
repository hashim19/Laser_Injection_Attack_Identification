from gmm import train_gmm
from os.path import exists
import pickle


# configs - feature extraction e.g., LFCC or CQCC
features = 'lfcc'

# configs - GMM parameters
ncomp = 512

# GMM pickle file
dict_file = 'gmm_LA_lfcc.pkl'
dict_file_final = 'gmm_lfcc_asvspoof21_la.pkl'

# configs - train & dev data - if you change these datasets
db_folder = '/path/to/ASVspoof_root/'
train_folders = [db_folder + 'LA/ASVspoof2019_LA_train/flac/']  # [db_folder + 'LA/ASVspoof2019_LA_train/flac/', db_folder + 'LA/ASVspoof2019_LA_dev/flac/']
train_keys = [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt']  # [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt']

audio_ext = '.flac'

# train bona fide & spoof GMMs
if not exists(dict_file):
    gmm_bona = train_gmm(data_label='bonafide', features=features,
                         train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
                         dict_file=dict_file, ncomp=ncomp,
                         init_only=True)
    gmm_spoof = train_gmm(data_label='spoof', features=features,
                          train_keys=train_keys, train_folders=train_folders, audio_ext=audio_ext,
                          dict_file=dict_file, ncomp=ncomp,
                          init_only=True)

    gmm_dict = dict()
    gmm_dict['bona'] = gmm_bona._get_parameters()
    gmm_dict['spoof'] = gmm_spoof._get_parameters()
    with open(dict_file, "wb") as tf:
        pickle.dump(gmm_dict, tf)


gmm_dict = dict()
with open(dict_file + '_bonafide_init_partial.pkl', "rb") as tf:
    gmm_dict['bona'] = pickle.load(tf)

with open(dict_file + '_spoof_init_partial.pkl', "rb") as tf:
    gmm_dict['spoof'] = pickle.load(tf)

with open(dict_file_final, "wb") as f:
    pickle.dump(gmm_dict, f)

