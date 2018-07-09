import os
import pickle

import librosa
import numpy as np
import soundfile as sf


class PostProcessor:
    def __init__(self, dir):
        config_path = os.path.join(dir, 'data_config.pkl')
        scaler_path = os.path.join(dir, 'scaler.pkl')
        assert os.path.exists(config_path)
        assert os.path.exists(scaler_path)

        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            for key, value in config.items():
                setattr(self, key, value)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def magspec_to_audio(self, data, n_iter=100, concat=False):
        shape = data.shape
        n_magspecs = np.prod(shape[:-2])
        count = 0

        if len(shape) == 2:
            audio = self.__inverse_transform(data, n_iter)

        elif len(shape) == 3:
            beats = shape[0]
            audio = np.zeros([beats, self.n_samples])

            for i in range(beats):
                audio[i] = self.__inverse_transform(data[i], n_iter)
                count += 1
                print('{} % done\r'.format(int(100 * count / n_magspecs)))

            if concat:
                audio = np.resize(audio, [beats * self.n_samples])

        elif len(shape) == 4:
            songs = shape[0]
            beats = shape[1]
            audio = np.zeros([songs, beats, self.n_samples])

            for i in range(songs):
                for j in range(beats):
                    audio[i][j] = self.__inverse_transform(data[i][j], n_iter)
                    count += 1
                    print('{} % done\r'.format(int(100 * count / n_magspecs)))

            if concat:
                audio = np.resize(audio, [songs, beats * self.n_samples])

        else:
            raise ValueError('Incorrect data shape')

        return audio

    def write(self, file, audio):
        sf.write(file, audio, int(self.sample_rate))

    @staticmethod
    def inv_magphase(mag, phase_angle):
        phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
        return mag * phase

    def __inverse_transform(self, data, n_iter):
        data = self.scaler.inverse_transform(data)
        data = data.T
        complex_specgram = self.inv_magphase(data, 0.0)

        for i in range(n_iter):
            audio = librosa.core.istft(complex_specgram, win_length=self.n_fft)

            if i != n_iter - 1:
                complex_specgram = librosa.core.stft(audio, n_fft=self.n_fft)
                _, phase = librosa.magphase(complex_specgram)
                phase_angle = np.angle(phase)
                complex_specgram = self.inv_magphase(data, phase_angle)

        return audio