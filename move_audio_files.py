import shutil, os
import glob 

audio_dir = '/home/hashim/PHD/lab_recordings/Kareem_All_Instructions/'

# wav_files = list(glob.glob(os.path.join(audio_dir, '**/**/*channel2.mp3')))
wav_files = list(glob.glob(os.path.join(audio_dir, '**/**/*.mp3')))
wav_files.sort()

# print(wav_files)

dest_folder = '/home/hashim/PHD/audio_data/Kareem/'
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

Orig_folder = dest_folder + 'Original/'
Laser_folder = dest_folder + 'Laser/'

if not os.path.exists(Orig_folder):
    os.makedirs(Orig_folder)

if not os.path.exists(Laser_folder):
    os.makedirs(Laser_folder)

laser_count = 0
orig_count = 0

for wf in wav_files:

    print(wf)

    wf_split = wf.split('/')

    tag = wf_split[-2].split("_")[0]

    channel = wf_split[-1]

    # print(channel)

    # print(tag)

    if tag == 'Direct' or tag == 'Speaker':

        new_filename = 'original_{:03d}'.format(orig_count) + '.mp3'
        print(new_filename)
        shutil.copy(wf, Orig_folder + new_filename)
        orig_count+=1

    if channel == 'output_channel2.mp3' and (tag == 'Right Mic' or tag == 'Right'):

        new_filename = 'laser_{:03d}'.format(laser_count) + '.mp3'
        print(new_filename)
        shutil.copy(wf, Laser_folder + new_filename)
        laser_count+=1

    if channel == 'output_channel1.mp3' and (tag == 'Left Mic' or tag == 'Left'):

        new_filename = 'laser_{:03d}'.format(laser_count) + '.mp3'
        print(new_filename)
        shutil.copy(wf, Laser_folder + new_filename)
        laser_count+=1
