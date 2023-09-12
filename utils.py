from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import split_on_silence
import os

import collections
import contextlib
import sys
import wave

import webrtcvad

from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 


def plot_projections(X, classes, ax=None, colors=None, markers=None, legend=True, title="", **kwargs):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Compute the 2D projections. You could also project to another number of dimensions (e.g. 
    # for a 3D plot) or use a different different dimensionality reduction like PCA or TSNE.
    reducer = UMAP(n_neighbors=2, **kwargs)
    projs = reducer.fit_transform(X)

    # Draw the projections
    classes = np.array(classes)
    colors = colors or _my_colors
    un_markers = pd.unique(markers)
    # print(un_markers)
    # for i, class_label in enumerate(np.unique(classes)):
    for i, class_label in enumerate(classes): 
        # class_projs = projs[classes == class_label]
        class_projs = projs[i]
        marker = "o" if markers is None else markers[i]
        label = class_label if legend else None
        # print(label)
        # print(marker)
        ax.scatter(*class_projs.T, c=[colors[i]], marker=marker, label=label)

    if legend:
        ax.legend(title="classes", ncol=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    
    return projs


def silence_based_conversion(path = "sample.wav", dump_dir = '/sample_dir'):

    # open the audio file
    audio_data = AudioSegment.from_wav(path)

    play(audio_data)


    # slice the audio based on silence
    chunks = split_on_silence(audio_data,
        # must be silent for at least 0.5 seconds
        # or 500 ms. adjust this value based on user
        # requirement. if the speaker stays silent for 
        # longer, increase this value. else, decrease it.
        min_silence_len = 600,
  
        # consider it silent if quieter than -16 dBFS
        # adjust this per requirement
        silence_thresh = -20
    )

    print(chunks)

    # create a directory to store the audio chunks.
    try:
        os.mkdir(dump_dir)
    except(FileExistsError):
        pass
  
    # move into the directory to
    # store the audio files.
    os.chdir(dump_dir)

    i = 0
    # process each chunk
    for chunk in chunks:
              
        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration = 10)
  
        # add 0.5 sec silence to beginning and 
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent
  
        # export audio chunk and save it in 
        # the current directory.
        print("saving chunk{0}.wav".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate ='192k', format ="wav")
  
        # the name of the newly created chunk
        filename = 'chunk'+str(i)+'.wav'
  
        print("Processing chunk "+str(i))

    # os.chdir('..')


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


if __name__ == "__main__":

    audio_path = '/home/hashim/PHD/Final Recordings/Abdul_mix_intructions/Direct/output_channel1.wav'

    chunk_storage_dir = '/home/hashim/PHD/Final Recordings/Abdul_mix_intructions/chunk/'

    # silence_based_conversion(path=audio_path, dump_dir=chunk_storage_dir)

    audio, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad(int(0.5))

    frames = frame_generator(30, audio, sample_rate)
    # frames = list(frames)

    # print(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # print(segments)

    # segments = list(segments)

    # print(segments)

    for i, segment in enumerate(segments):
        path = 'chunk{}.wav'.format(int(i))
        fullpath = chunk_storage_dir + path
        print(' Writing %s' % (fullpath,))
        write_wave(fullpath, segment, sample_rate)





