"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""
from typing import List, Dict
import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import List
from pint import UnitRegistry
from cached_property import cached_property


from sbmlsim.simulation import ScanSim, Dimension
from sbmlsim.utils import deprecated, timeit

logger = logging.getLogger(__name__)


class XResult:
    def __init__(self, xdataset: xr.Dataset, scan: ScanSim,
                 udict: Dict = None, ureg: UnitRegistry=None, _df: pd.DataFrame = None):
        self.xds = xdataset
        self.scan = scan
        self.udict = udict
        self.ureg = ureg

        # set the DataFrame if a one-dimensional scan
        self._df = _df

    def __getitem__(self, key) -> xr.DataArray:
        return self.xds[key]

    def __getattr__(self, name):
        """Provide dot access to keys."""
        if name in {'xds', 'scan', 'udict', 'ureg'}:
            # local field lookup
            return getattr(self, name)
        else:
            # forward lookup to xds
            return getattr(self.xds, name)

    @classmethod
    def from_dfs(cls, dfs: List[pd.DataFrame], scan: ScanSim=None,
                 udict: Dict = None, ureg: UnitRegistry = None) -> 'XResult':
        """Structure is based on the underlying scan."""
        logger.error("Start creating XResult")
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
        columns = dfs[0].columns

        data_dict = {col: np.empty(shape=shape) for col in columns if col != "time"}
        for k_df, df in enumerate(dfs):
            for k_col, column in enumerate(columns):

                if column == "time":
                    # not storing "time" column (encoded as dimension)
                    continue

                # FIXME: speedup by restructuring data
                index = tuple([...] + list(indices[k_df]))  # trick to get the ':' in first time dimension
                data_dict[column][index] = df[column].values

        # Create the DataSet
        _df = None
        if len(dfs) == 1:
            # store a copy of single timecourse for serialization
            # FIXME: this is a hack for now
            _df = dfs[0].copy()

        ds = xr.Dataset({key: xr.DataArray(data=data, dims=dims, coords=coords) for key, data in data_dict.items()})
        return XResult(xdataset=ds, scan=scan, udict=udict, ureg=ureg, _df=_df)



    @deprecated
    def to_hdf5(self, path):
        """Store complete results as HDF5"""
        # FIXME: handle the special case, where we can convert things to
        # a dataframe

        logger.warning("to_hdf5 only partially implemented")
        # FIXME: new serialization format (netCDF?)

        if self._df is not None:
            with pd.HDFStore(path, complib="zlib", complevel=9) as store:
                for k, frame in enumerate([self._df]):
                    key = "df{}".format(k)
                    store.put(key, frame)

        if self._df is not None:
            path_tsv = f"{str(path)}.tsv"
            print(path_tsv)
            self._df.to_csv(path_tsv, sep="\t", index=False)

        else:
            logger.error("No dataframe found.")


    @deprecated
    @staticmethod
    def from_hdf5(path):
        """Read from HDF5"""
        logger.warning("'from_hdf5' not implemented")
        # FIXME: implement new serialization format

        # with pd.HDFStore(path) as store:
        #    frames = [store[key] for key in store.keys()]
        #
        # return Result(frames=frames)

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
'''