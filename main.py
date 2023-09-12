from sys import exec_prefix
from utils import read_wave, write_wave, frame_generator, vad_collector

import numpy as np
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
import webrtcvad
import os
import glob
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import read, write 
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


Apply_VAD = False


# audio files directory
# audio_dir = '/home/hashim/PHD/Final Recordings/Abdul_mix_intructions'

# split_audio_dir = audio_dir.split('/')
# chunk_dir = audio_dir.replace(split_audio_dir[-1], '') + split_audio_dir[-1] +'_chunk'

# orig_chunk_dir = chunk_dir +  '/orig_chunk/'
# laser_left_chunk_dir = chunk_dir + '/laser_left_chunk/'
# laser_right_chunk_dir = chunk_dir + '/laser_right_chunk/'

# print(chunk_dir)

# # create the required directories
# if not os.path.exists(chunk_dir):
#     os.makedirs(chunk_dir)

# if not os.path.exists(orig_chunk_dir):
#     os.makedirs(orig_chunk_dir)

# if not os.path.exists(laser_left_chunk_dir):
#     os.makedirs(laser_left_chunk_dir)

# if not os.path.exists(laser_right_chunk_dir):
#     os.makedirs(laser_right_chunk_dir)

# get the paths of all the audio files
# wav_fpaths = list(Path(audio_dir).glob("**/**/*.wav"))

audio_dir = '/home/hashim/PHD/Final Recordings/Abdul_mix_intructions'

wav_files = list(glob.glob(os.path.join(audio_dir, '**/*.wav')))
wav_files.sort()
print(wav_files)


# if Apply_VAD:

#     # Apply VAD to the audio files
#     vad = webrtcvad.Vad(int(0.5))

#     for wav_path in wav_files:

#         wav_path_split = wav_path.split('/')

#         # read the audio file
#         audio, sample_rate = read_wave(wav_path)

#         # create the frames of the audio file
#         frames = frame_generator(30, audio, sample_rate)

#         # create only voices segments
#         segments = vad_collector(sample_rate, 30, 300, vad, frames)

#         for i, segment in enumerate(segments):
#             path = 'chunk{}.wav'.format(int(i))
            
#             if wav_path_split[-2] == 'Original':

#                 fullpath = orig_chunk_dir + path
            
#             if wav_path_split[-2] == 'Laser_left':

#                 fullpath = laser_left_chunk_dir + path

#             if wav_path_split[-2] == 'Laser_right':

#                 fullpath = laser_right_chunk_dir + path 

#             print(' Writing %s' % (fullpath,))
            
#             try:
#                 write_wave(fullpath, segment, sample_rate)
#             except:
#                 print('could not write the wave file')


# audio = AudioSegment.from_wav(wav_files[0])

# print(audio)

fs, audio_data = read(wav_files[-2])

for wf in wav_files:

    print(wf)
    wf_split = wf.split('/')

    out_filename = audio_dir + '/' + wf_split[-2] + '_' + wf_split[-1].split('.')[0] 
    
    try:
        fs, audio_data = read(wf)
    except:
        print('could not read the wave file')
        continue

# plt.figure(figsize=(12,9))
# plt.plot(audio_data)
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# # f, t, Sxx = signal.spectrogram(audio_data, 1)

    plt.figure(figsize=(12,9))
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(audio_data, Fs=fs)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.show()
    plt.savefig(out_filename + '.png')