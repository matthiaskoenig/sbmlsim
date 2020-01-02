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

    def __init__(self, frames: List[pd.DataFrame], udict=None, ureg=None):
        """

        :param frames: iterable of pd.DataFrame
        """
        if isinstance(frames, pd.DataFrame):
            frames = [frames]

        # empty array for storage
        self.frames = frames
        # units dictionary for lookup
        self.udict = udict
        self.ureg = ureg

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
            logging.warning("For a single simulation the mean returns the single simulation")
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
        """Store complete results as HDF5"""
        with pd.HDFStore(path, complib="zlib", complevel=9) as store:
            for k, frame in enumerate(self.frames):
                key = "df{}".format(k)
                store.put(key, frame)

    @staticmethod
    def from_hdf5(path):
        """Read from HDF5"""
        with pd.HDFStore(path) as store:
            frames = [store[key] for key in store.keys()]

        return Result(frames=frames)
