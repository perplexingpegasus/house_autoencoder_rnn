import os
import pickle
from itertools import cycle

import numpy as np


class FeedDict:
    def __init__(self, datadir, logdir, split=0.75, random_state=0):
        assert split >= 0 and split < 1

        files = [os.path.join(datadir, f)
                for f in os.listdir(datadir)
                if f.endswith('.npy')]
        np.random.shuffle(files)

        self.training_data = dict(files=None, cur_path=None,
            cur_array=None, cur_array_len=0, idx=0)

        if split == 0:
            self.test_data = None
            self.training_data['files'] = cycle(files)
            self.__change_array()

        else:
            n_files = len(files)
            split_idx = int(n_files * split)
            assert split_idx < n_files

            self.test_data = self.training_data.copy()
            self.training_data['files'] = cycle(files[:split_idx])
            self.test_data['files'] = cycle(files[split_idx:])

            self.__change_array()
            self.__change_array(test=True)

        self.shape = self.training_data['cur_array'].shape[1:]
        self.logdir = logdir

    def __change_array(self, test=False):
        data = self.test_data if test else self.training_data
        new_path = next(data['files'])

        if new_path != data['cur_path']:
            data['cur_path'] = new_path
            data['cur_array'] = np.load(new_path)
            data['cur_array_len'] = data['cur_array'].shape[0]
        data['idx'] = 0

    def next_batch(self, batch_size, test=False):
        data = self.test_data if test else self.training_data

        start = data['idx']
        remaining = data['cur_array_len'] - start

        if remaining >= batch_size:
            stop = start + batch_size
            return data['cur_array'][start:stop]

        else:
            stop = batch_size - remaining
            batch = data['cur_array'][start:]
            self.__change_array()
            batch = np.concatenate((batch, data['cur_array'][:stop]))

        data['idx'] = stop
        return batch

    @classmethod
    def load(cls, datadir, logdir):
        if os.path.exists(logdir):
            with open(logdir, 'rb') as f:
                fd = pickle.load(f)
            if type(fd) == cls:
                return fd
        return cls(datadir, logdir)

    def save(self):
        with open(self.logdir, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)