"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""
from typing import List
import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import List

from sbmlsim.simulation import ScanSim, Dimension

from cached_property import cached_property
# FIXME: invalidate the cache on changes !!!

logger = logging.getLogger(__name__)


class XResult(object):

    @classmethod
    def from_dfs(cls, scan: ScanSim, dfs: List[pd.DataFrame]) -> xr.Dataset:
        """Structure is based on the underlying scan."""
        df = dfs[0]
        # add time dimension (FIXME: internal dimension depend on simulation type)
        shape = [len(df)]
        dims = ["time"]
        coords = {"time": df.time.values}
        for scan_dim in scan.dimensions:  # type: ScanDimension
            shape.append(len(scan_dim))
            dim = scan_dim.dimension
            dims.append(dim)
            coords[dim] = scan_dim.index

        print("shape:", shape)
        print("dims:", dims)
        # print(coords)

        indices = scan.indices()

        ds = xr.Dataset()
        columns = dfs[0].columns
        data = np.empty(shape=shape)
        for k_col, column in enumerate(columns):
            if column == "time":
                # not storing the "time" column (encoded as dimension)
                continue

            for k_df, df in enumerate(dfs):
                index = tuple([...] + list(indices[k_df]))  # trick to get the ':' in first time dimension
                # print(index)
                # print(index)
                data[index] = df[column].values

            # create DataArray for given column
            da = xr.DataArray(data=data, dims=dims, coords=coords)
            ds[column] = da

        return ds


class Result(object):
    """Result of simulation(s).

    Results only store the raw data without any units.
    The SimulationExperiment context, i.e., especially the model definition is
    required to resolve the units.
    """

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
            self.data = np.empty((self.nrow, self.ncol, self.nframes))
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
            logging.debug("For a single simulation the mean returns the single simulation")
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
