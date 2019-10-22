"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""
import logging
import numpy as np
import pandas as pd
from typing import List

from cached_property import cached_property
# FIXME: invalidate the cache on changes !!!

logger = logging.getLogger(__name__)


class Result(object):
    """Result of simulation(s)."""

    def __init__(self, frames: List[pd.DataFrame]):
        """

        :param frames: iterable of pd.DataFrame
        """
        if isinstance(frames, pd.DataFrame):
            frames = [frames]

        # empty array for storage
        self.frames = frames

        if len(frames) > 0:
            df = frames[0]
            self.index = df.index
            self.columns = df.columns

            # store data in numpy
            self.data = np.empty((self.nrow, self.ncol, self.nframes)) * np.nan
            for k, df in enumerate(self.frames):
                self.data[:, :, k] = df.values
        else:
            logging.warning("Empty Result, no DataFrames.")

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        lines = [
            str(type(self)),
            f"DataFrames: {len(self)}",
            f"Shape: {self.data.shape}",
            f"Size (bytes): {self.data.nbytes}"
        ]
        return "\n".join(lines)


    def statistics_df(self):
        df = pd.DataFrame({
            'mean'
        })
        pass


    @cached_property
    def nrow(self):
        return len(self.index)

    @cached_property
    def ncol(self):
        return len(self.columns)

    @cached_property
    def nframes(self):
        return len(self.frames)

    @cached_property
    def mean(self):
        if len(self) == 1:
            logging.warning("For a single simulation the mean is the actual simulation")
            return self.frames[0]
        else:
            return pd.DataFrame(np.mean(self.data, axis=2), columns=self.columns)

    @cached_property
    def std(self):
        return pd.DataFrame(np.std(self.data, axis=2), columns=self.columns)

    @cached_property
    def min(self):
        return pd.DataFrame(np.min(self.data, axis=2), columns=self.columns)

    @cached_property
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
