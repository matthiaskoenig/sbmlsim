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


@xr.register_dataset_accessor("sim")
class Result:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @classmethod
    def from_dfs(cls, dfs: List[pd.DataFrame], scan: ScanSim=None, udict=None) -> xr.Dataset:
        """Structure is based on the underlying scan."""
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if udict is None:
            udict = {}

        # add time dimension
        # FIXME: internal dimensions different for other simulation types
        df = dfs[0]
        shape = [len(df)]
        dims = ["time"]
        coords = {"time": df.time.values}
        del df

        if scan is None:
            logger.error("dummy scan created!")
            # Create a dummy scan
            scan = ScanSim(
                simulation=None,
                dimensions=[Dimension("dim1", index=np.arange(1))]
            )

        # add additional scan dimensions
        for scan_dim in scan.dimensions:  # type: ScanDimension
            shape.append(len(scan_dim))
            dim = scan_dim.dimension
            coords[dim] = scan_dim.index
            dims.append(dim)

        # print("shape:", shape)
        # print("dims:", dims)
        # print(coords)

        indices = scan.indices()

        ds = xr.Dataset()
        columns = dfs[0].columns
        data = np.empty(shape=shape)
        for k_col, column in enumerate(columns):
            if column == "time":
                # not storing "time" column (encoded as dimension)
                continue

            for k_df, df in enumerate(dfs):
                index = tuple([...] + list(indices[k_df]))  # trick to get the ':' in first time dimension
                data[index] = df[column].values

            # create DataArray for given column
            da = xr.DataArray(data=data, dims=dims, coords=coords)

            # set unit
            if column in udict:
                da.attrs["units"] = udict[column]

            ds[column] = da

        return ds


    '''
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
    '''