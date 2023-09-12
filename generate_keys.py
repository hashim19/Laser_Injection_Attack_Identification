import os
import pandas as pd
import librosa
import librosa.display
import glob 
import numpy as np
import random

pth_to_data = '/home/hashim/PHD/audio_data/AllAudioSamples/'
Percent = 0.75
train_out_filename = 'train_laser_injection_data'
test_out_filename = 'test_laser_injection_data'


laser_dir = pth_to_data + 'Laser'
# laser_files = list(glob.glob('*.wav'))
laser_files = os.listdir(laser_dir)
laser_files = [lf.split('.')[0] for lf in laser_files if lf.endswith('.wav')]
laser_files.sort()

print(laser_files)


orig_dir = pth_to_data + 'Original'
# orig_files = list(glob.glob(os.path.join(orig_dir, '*.wav')))
orig_files = os.listdir(orig_dir)
orig_files = [of.split('.')[0] for of in orig_files if of.endswith('.wav')]
orig_files.sort()

# all_files = [laser_files, orig_files]

# print(all_files)

train_out_data = {'filename': [], 'Label': []}
test_out_data = {'filename': [], 'Label': []}

train_laser_files = random.sample(laser_files, int(len(laser_files)*Percent))
test_laser_files = [lf for lf in laser_files if lf not in train_laser_files]

train_orig_files = random.sample(orig_files, int(len(orig_files)*Percent))
test_orig_files = [lf for lf in orig_files if lf not in train_orig_files]

# print(train_laser_files)
# print(len(train_laser_files))
# print(len(laser_files))
# print(len(test_laser_files))

# print('test')
# print(len(train_orig_files))
# print(len(test_orig_files))

# generating tarining keys

for f_name in train_laser_files:
    train_out_data['filename'].append('Laser/' + f_name)
    train_out_data['Label'].append('spoof')

for f_name in train_orig_files:
    train_out_data['filename'].append('Original/' + f_name)
    train_out_data['Label'].append('bonafide')

print(train_out_data)

train_out_pd = pd.DataFrame(train_out_data)
train_out_pd.to_csv(train_out_filename + '.csv')

# generating testing keys 
for f_name in test_laser_files:
    test_out_data['filename'].append('Laser/' + f_name)
    test_out_data['Label'].append('spoof')

for f_name in test_orig_files:
    test_out_data['filename'].append('Original/' + f_name)
    test_out_data['Label'].append('bonafide')

print(test_out_data)

test_out_pd = pd.DataFrame(test_out_data)
test_out_pd.to_csv(test_out_filename + '.csv')