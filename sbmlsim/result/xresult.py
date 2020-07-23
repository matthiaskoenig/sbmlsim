"""
Helpers for working with simulation results.
Handles the storage of simulations.
"""
from typing import List, Dict
import logging
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum

from sbmlsim.simulation import ScanSim, Dimension
from sbmlsim.utils import deprecated, timeit
from sbmlsim.units import UnitRegistry


logger = logging.getLogger(__name__)


class XResult:
    """
    Always check if returned quantities have units or not!

    FIXME: write the units in the attrs
    FIXME: helper method for to DataFrame
    """
    class XResultType(Enum):
        TIMECOURSE = 1
        STEADYSTATE = 2

    def __init__(self, xdataset: xr.Dataset,
                 udict: Dict = None, ureg: UnitRegistry=None,
                 type: XResultType = XResultType.TIMECOURSE):
        self.xds = xdataset
        self.udict = udict
        self.ureg = ureg
        self.type = type

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

    # TODO: general dimension reduction (for complete dataset)
    '''
    def dim_op(self, f_op, key=None):
        data = self.xds[key] if key else self.xds
        f = getattr(data, "f_op")
        res = f(dim=self._redop_dims(), skipna=True)
        if key: 
            return res.values * self.ureg(self.udict[key])
        else:
            return res
    '''

    def is_timecourse(self) -> bool:
        """Check if timecourse"""
        result = True
        xds = self.xds
        if len(xds.dims) == 2:
            for dim in xds.dims:
                if dim == "_time":
                    continue
                else:
                    if xds.dims[dim] != 1:
                        result = False
        else:
            return False
        return result

    def to_mean_dataframe(self):
        res = {}
        for col in self.xds:
            res[col] = self.dim_mean(key=col)
        return pd.DataFrame(res)

    def dim_mean(self, key):
        """Mean over all added dimensions"""
        return self.xds[key].mean(dim=self._redop_dims(), skipna=True).values * self.ureg(self.udict[key])

    def dim_std(self, key):
        """Standard deviation over all added dimensions"""
        return self.xds[key].std(dim=self._redop_dims(), skipna=True).values * self.ureg(self.udict[key])

    def dim_min(self, key):
        """Minimum over all added dimensions"""
        return self.xds[key].min(dim=self._redop_dims(), skipna=True).values * self.ureg(self.udict[key])

    def dim_max(self, key):
        """Maximum over all added dimensions"""
        return self.xds[key].max(dim=self._redop_dims(), skipna=True).values * self.ureg(self.udict[key])

    def _redop_dims(self) -> List[str]:
        """Dimensions for reducing operations."""
        if self.type == self.XResultType.TIMECOURSE:
            return [dim_id for dim_id in self.dims if dim_id != "_time"]
        else:
            return [dim_id for dim_id in self.dims if dim_id != "_time"]

    @classmethod
    def from_dfs(cls, dfs: List[pd.DataFrame], scan: ScanSim=None,
                 udict: Dict = None, ureg: UnitRegistry = None) -> 'XResult':
        """Structure is based on the underlying scan."""
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if udict is None:
            udict = {}

        df = dfs[0]
        num_dfs = len(dfs)

        # add time dimension
        shape = [len(df)]
        dims = ["_time"]  # FIXME: internal dimensions different for other simulation types
        coords = {"_time": df.time.values}
        columns = df.columns
        del df

        # Additional dimensions
        if scan is None:
            dimensions = [Dimension("_dfs", index=np.arange(num_dfs))]
        else:
            dimensions = scan.dimensions

        # add additional dimensions
        for dimension in dimensions:  # type: Dimension
            shape.append(len(dimension))
            dim_id = dimension.dimension
            coords[dim_id] = dimension.index
            dims.append(dim_id)

        indices = Dimension.indices_from_dimensions(dimensions)
        data_dict = {col: np.empty(shape=shape) for col in columns}
        for k_df, df in enumerate(dfs):
            for k_col, column in enumerate(columns):

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
        for key in data_dict:
            if key in udict:
                ds[key].attrs["units"] = udict[key]
        return XResult(xdataset=ds, udict=udict, ureg=ureg)

    def to_netcdf(self, path_nc):
        """Store results as netcdf."""
        self.xds.to_netcdf(path_nc)

    def to_tsv(self, path_tsv):
        '''
        if self._df is not None:
            with pd.HDFStore(path, complib="zlib", complevel=9) as store:
                for k, frame in enumerate([self._df]):
                    key = "df{}".format(k)
                    store.put(key, frame)
        '''
        xds = self.xds  # type: xr.Dataset

        # Check if data can be converted to DataFrame (only timecourse data)
        if (len(xds.dims) == 2):
            for dim in xds.dims:
                if dim == "_time":
                    continue
                else:
                    if xds.dims[dim] > 1:
                        logger.warning("No TSV created for higher dimensional data.")
                        return

        data = {v: xds[v].values.flatten() for v in xds.keys()}
        df = pd.DataFrame(data)
        df.to_csv(path_tsv, sep="\t", index=False)

    @staticmethod
    def from_netcdf(path):
        """Read from netCDF"""
        ds = xr.open_dataset(path)
        return XResult(xdataset=ds)


if __name__ == "__main__":
    from sbmlsim.model import RoadrunnerSBMLModel
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR
    r = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)._model
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres = XResult.from_dfs(dfs)
    print(xres)