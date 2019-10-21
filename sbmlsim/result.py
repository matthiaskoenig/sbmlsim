"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""
import logging
import numpy as np
import pandas as pd
from typing import List


class Result(object):
    """Result of simulation(s)."""

    def __init__(self, frames: List[pd.DataFrame]):
        """

        :param frames: iterable of pd.DataFrame
        """
        if isinstance(frames, pd.DataFrame):
            frames = [frames]

        # empty array for storage
        df = frames[0]
        self.index = df.index
        self.columns = df.columns
        self.frames = frames

        # store data in numpy
        self.data = np.empty((self.nrow, self.ncol, self.nframes)) * np.nan
        for k, df in enumerate(self.frames):
            self.data[:, :, k] = df.values

    def __len__(self):
        return len(self.frames)

    def statistics_df(self):
        df = pd.DataFrame({
            'mean'
        })
        pass


    @property
    def nrow(self):
        return len(self.index)

    @property
    def ncol(self):
        return len(self.columns)

    @property
    def nframes(self):
        return len(self.frames)

    @property
    def mean(self):
        logging.warning("no caching of mean !")
        if len(self) == 1:
            logging.warning("mean() on Result with len==1 is not defined")
            return self.frames[0]
        else:
            return pd.DataFrame(np.mean(self.data, axis=2), columns=self.columns)

    @property
    def std(self):
        return pd.DataFrame(np.std(self.data, axis=2), columns=self.columns)

    @property
    def min(self):
        return pd.DataFrame(np.min(self.data, axis=2), columns=self.columns)

    @property
    def max(self):
        return pd.DataFrame(np.max(self.data, axis=2), columns=self.columns)

    def to_hdf5(self, path):
        """Store to HDF5"""
        with pd.HDFStore(path) as store:
            for k, frame in enumerate(self.frames):
                key = "df{}".format(k)
                store[key] = frame

    @staticmethod
    def from_hdf5(path):
        """Read from HDF5"""
        with pd.HDFStore(path) as store:
            frames = [store[key] for key in store.keys()]

        return Result(frames=frames)
