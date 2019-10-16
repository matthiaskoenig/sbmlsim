"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""

import logging
import numpy as np
import pandas as pd
from typing import List


# FIXME: hashing
# TODO: serialization of results:
# HDF5 (reading and writing)

class TaskResult(object):
    """
    stores

    """
    def __init__(self, model: str, sims, result):
        pass


class Result(object):
    """Result of a single timecourse simulation. """

    def __init__(self, frames: List[pd.DataFrame]):
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
