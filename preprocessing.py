import argparse
import os
import pickle

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


parser = argparse.ArgumentParser(description='Process audio files into data for FeedDicts')
parser.add_argument('--audiodir', type=str,
    help='Directory of audio files')
parser.add_argument('--savedir', type=str,
    help='Directory to save numpy arrays and other data for FeedDict')
parser.add_argument('--bpm', type=int, default=128,
    help='Should be the same for all audio files')
parser.add_argument('--n_beats', type=int, default=256,
    help='Number of beats to take from each song. Songs must be at least n_beats / BPM minutes long')
parser.add_argument('--scaler_type', type=str, default='minmax',
    help='Type of scaler to normalize data -- either "standard" or "minmax"')
parser.add_argument('--max_size', type=float, default=0.5,
    help='Maximum size for numpy arrays in GB')
parser.add_argument('--frames', type=int, default=128,
    help='Number of frames in each STFT. NOT RECCOMENDED TO CHANGE')
parser.add_argument('--channels', type=int, default=256,
    help='Maximum size for numpy arrays in GB. NOT RECCOMENDED TO CHANGE')
args = parser.parse_args()


# function for pickling objects
def save_pickle(obj, filename):
    path = os.path.join(savedir, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# function for saving memmap arrays
def save_memmap(file_idx, batch, label):
    print('Saving {}: file {}...\n'.format(label, file_idx))
    path = os.path.join(savedir, label,
                        '{}_{}.npy'.format(label, file_idx))
    np.save(path, batch)
    return path


if __name__ == '__main__':

    audiodir = args.audiodir                                   # directory where audio files are located
    savedir = args.savedir                                     # directory to save project files
    bpm = args.bpm                                             # all audio should have the same BPM
    n_beats = args.n_beats                                     # number of beats to take from each audio sample
    frames = args.frames                                       # number of time frames per beat in STFT
    channels = args.channels                                   # number of channels in STFT
    max_size_gb = args.max_size                                # max size of memmap files
    scaler_type = args.scaler_type


    n_fft = (channels - 1) * 2                                 # number of FFTs for STFT
    hop_length = n_fft // 4                                    # hop length for STFT
    n_samples = hop_length * (frames - 1)                      # number of samples per beat
    sr = bpm * n_samples / 60                                  # sample rate
    min_audio_len = n_samples * n_beats                        # minimum samples per audio track
    max_size = max_size_gb * 1e9                               # GB to bytes
    bytes_per_beat = 32 * frames * channels                    # size of STFT for each beat
    max_songs = int(max_size / (n_beats * bytes_per_beat))     # max songs per memmap
    max_beats = int(max_size / bytes_per_beat)                 # max beats per memmap


    # create directories
    for dir in [savedir, os.path.join(savedir, 'time_series_data'),
                os.path.join(savedir, 'randomized_data')]:
        if not os.path.exists(dir): os.makedirs(dir)

    # save information
    data_config = dict(n_beats=n_beats, frames=frames, channels=channels, n_fft=n_fft,
        hop_length=hop_length, n_samples=n_samples, sample_rate=sr)
    print('Saving config...\n')
    save_pickle(data_config, 'data_config.pkl')


    # initialize variables for loop for saving time series data
    audio_files = [os.path.join(audiodir, f) for f in os.listdir(audiodir)]
    n_audio_files = len(audio_files)
    batch = np.zeros([max_songs, n_beats, frames, channels], np.float32)
    file_idx = 0
    song_idx = 0

    # file paths to be used later
    time_series_files = []

    # create scaler
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler((-1.0, 1.0))

    # loop through audio files
    for n, f in enumerate(audio_files):

        # load audio file and check for minimum length
        print('Opening audio file: {}...\n'.format(f))
        audio, _ = librosa.load(f, sr)
        assert len(audio) >= min_audio_len

        # get individual beats in audio
        for beat_idx in range(n_beats):
            beat = audio[n_samples * beat_idx:n_samples * (beat_idx + 1)]

            # transform beat with STFT
            stft = librosa.stft(beat, n_fft=n_fft)
            mag, phase = librosa.magphase(stft)
            mag = mag.T

            # fit scaler to beat along the frequency dimension
            scaler.partial_fit(mag)

            # append STFT data to batch
            batch[song_idx][beat_idx] = mag

        # save batch if it's full
        if song_idx == max_songs - 1:
            print('Saving time series data: file {}...\n'.format(file_idx))
            path = os.path.join(savedir, 'time_series_data',
                'time_series_data_{}.npy'.format(file_idx))
            np.save(path, batch)

            # append file path
            time_series_files.append(path)

            # set loop indices
            song_idx = 0
            file_idx += 1

        # save remaining batch
        elif n == n_audio_files - 1:
            path = save_memmap(file_idx, batch, 'time_series_data')
            time_series_files.append(path)

        else:
            song_idx += 1

    if song_idx != 0:
        path = save_memmap(file_idx, batch[:song_idx], 'time_series_data')
        time_series_files.append(path)

    # save scaler after fitting
    print('Saving scaler...\n')
    path = os.path.join(savedir, 'scaler.pkl')
    with open(path, 'wb') as f:
        pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)


    # loop for scaling data and get indices for shuffling
    print('Scaling data...\n')
    indices = []
    for file_idx, f in enumerate(time_series_files):

        # get array shape
        arr = np.load(f)
        shape = arr.shape

        # loop through songs
        for song_idx in range(shape[0]):

            # loop through beats -- transform data and append indices
            for beat_idx in range(shape[1]):
                arr[song_idx][beat_idx] = scaler.transform(arr[song_idx][beat_idx])
                indices.append((file_idx, song_idx, beat_idx))

        np.save(f, arr)


    # shuffle indices
    np.random.shuffle(indices)

    # initialize loop variable for saving randomly shuffled data
    batch = np.zeros([max_beats, frames, channels], np.float32)
    file_idx = 0
    beat_idx = 0

    # loop through indices
    for idx in indices:
        i, j, k = idx

        # append beat to batch
        time_series_batch =  np.load(time_series_files[i])
        batch[beat_idx] = time_series_batch[j][k]

        # save batch if it's full
        if beat_idx == max_beats - 1:
            save_memmap(file_idx, batch, 'randomized_data')
            beat_idx = 0
            file_idx += 1

        else:
            beat_idx += 1

    # save remaining batch
    if beat_idx != 0:
        save_memmap(file_idx, batch[:beat_idx], 'randomized_data')