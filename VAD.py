# Run the VAD on 10 ms of silence. The result should be False.
from numpy.core.numeric import outer
import webrtcvad
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
import glob
import os 
import librosa

from resemblyzer import preprocess_wav
from utils import read_wave, write_wave, frame_generator, vad_collector

vad = webrtcvad.Vad(2)

# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))


def apply_vad(in_path, out_path, file_type='mp3'):

    wav_files = list(glob.glob(os.path.join(in_path, '*.' + file_type)))
    wav_files.sort()

    # print(wav_files)
    # create vad object
    vad = webrtcvad.Vad(int(0))

    for wf in wav_files:

        wf_split = wf.split('/')
        # tag = wf_split[-2].split("_")[0]

        file_name = wf_split[-1].split(".")[0]

        print(file_name)

        # audio_data, sr = librosa.load(wf, res_type='kaiser_fast')
        audio_data, sr = read_wave(wf)

        # create the frames of the audio file
        frames = frame_generator(30, audio_data, sr)
        
        # create only voices segments
        segments = vad_collector(sr, 30, 300, vad, frames)

        for i, segment in enumerate(segments):
            
            fullpath = out_path + file_name + '_{:03d}'.format(i) + '.mp3'

            print(' Writing %s' % (fullpath,))

            # print(len(segment))

            try:
                # if vad.is_speech(segment, sr):
                write_wave(fullpath, segment, sr)
            except:
                print('could not write the file')


if __name__ == "__main__":

    # in_dir = '/home/hashim/PHD/audio_data/Laser_Original/Laser/'
    # out_dir = '/home/hashim/PHD/audio_data/Laser_Original/Laser_cleaned_1/'

    in_dir = '/home/hashim/PHD/audio_data/Laser_Original_New/Laser/'
    out_dir = '/home/hashim/PHD/audio_data/Laser_Original_New/Laser_cleaned/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    apply_vad(in_dir, out_dir)