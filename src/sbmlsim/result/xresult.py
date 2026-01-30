"""Module for encoding simulation results and processed data."""

from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from pymetadata import log

from sbmlsim.simulation import Dimension, ScanSim
from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


class XResult:
    """Result of simulations.

    A wrapper around xr.Dataset which adds unit support via
    dictionary lookups.
    """

    def __init__(self, xdataset: xr.Dataset, uinfo: Optional[UnitsInformation] = None):
        self.xds = xdataset
        self.uinfo = uinfo

    def __getitem__(self, key) -> xr.DataArray:
        """Get item."""
        try:
            return self.xds[key]
        except KeyError as err:
            logger.error(f"Key '{key}' not in {self.xds}" f"\n{err}")
            raise err

    def __getattr__(self, name):
        """Provide dot access to keys."""
        if name in {"xds", "scan", "uinfo"}:
            # local field lookup
            return getattr(self, name)
        else:
            # forward lookup to xds
            return getattr(self.xds, name)

    def __str__(self) -> str:
        """Get string."""
        return f"<XResult: {self.xds.__repr__()},\n{self.uinfo}>"

    def dim_mean(self, key: str) -> xr.Dataset:
        """Get mean over all added dimensions."""
        try:
            data = self.xds[key].mean(
                dim=self._redop_dims(), skipna=True
            ).values * self.uinfo.ureg(self.uinfo[key])
        except KeyError as err:
            logger.error(
                f"Key '{key}' does not exist in XResult. Add the "
                f"key to the experiment via add_selections in "
                f"'Experiment.datagenerators'."
            )
            raise err
        return data

    def dim_std(self, key):
        """Get standard deviation over all added dimensions."""
        return self.xds[key].std(
            dim=self._redop_dims(), skipna=True
        ).values * self.uinfo.ureg(self.uinfo[key])

    def dim_min(self, key):
        """Get minimum over all added dimensions."""
        return self.xds[key].min(
            dim=self._redop_dims(), skipna=True
        ).values * self.uinfo.ureg(self.uinfo[key])

    def dim_max(self, key):
        """Get maximum over all added dimensions."""
        return self.xds[key].max(
            dim=self._redop_dims(), skipna=True
        ).values * self.uinfo.ureg(self.uinfo[key])

    def _redop_dims(self) -> List[str]:
        """Dimensions for reducing operations."""
        return [dim_id for dim_id in self.dims if dim_id != "_time"]

    @classmethod
    def from_dfs(
        cls,
        dfs: List[pd.DataFrame],
        scan: ScanSim = None,
        uinfo: UnitsInformation = None,
    ) -> "XResult":
        """Create XResult from DataFrames.

        Structure is based on the underlying scans
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if uinfo is None:
            uinfo = UnitsInformation(udict={}, ureg=None)

        df = dfs[0]
        num_dfs = len(dfs)

        # add time dimension
        shape = [len(df)]
        dims = ["_time"]
        coords = {"_time": df.time.values}
        columns = df.columns
        del df

        # Additional dimensions
        if scan is None:
            dimensions = [Dimension("_dfs", index=np.arange(num_dfs))]
        else:
            dimensions = scan.dimensions  # type: ignore

        # add additional dimensions
        dimension: Dimension
        for dimension in dimensions:
            shape.append(len(dimension))
            dim_id = dimension.dimension
            coords[dim_id] = dimension.index
            dims.append(dim_id)

        indices = Dimension.indices_from_dimensions(dimensions)
        data_dict = {col: np.empty(shape=shape) for col in columns}
        for k_df, df in enumerate(dfs):
            for column in columns:
                index = tuple(
                    [...] + list(indices[k_df])
                )  # trick to get the ':' in first time dimension
                data = data_dict[column]
                data[index] = df[column].values

        # Create the DataSet
        ds = xr.Dataset(
            {
                key: xr.DataArray(data=data, dims=dims, coords=coords)
                for key, data in data_dict.items()
            }
        )
        for key in data_dict:
            if key in uinfo:
                # set units attribute
                ds[key].attrs["units"] = uinfo[key]
        return XResult(xdataset=ds, uinfo=uinfo)

    def to_netcdf(self, path_nc):
        """Store results as netcdf."""
        self.xds.to_netcdf(path_nc)

    def is_timecourse(self) -> bool:
        """Check if timecourse."""
        # FIXME: better implementation necessary
        is_tc = True
        xds = self.xds
        if len(xds.dims) == 2:
            for dim in xds.dims:
                if dim == "_time":
                    continue
                else:
                    if xds.sizes[dim] != 1:
                        is_tc = False
        else:
            return False
        return is_tc

    def to_mean_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with mean data."""
        res = {}
        for col in self.xds:
            res[col] = self.dim_mean(key=col)
        return pd.DataFrame(res)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self.is_timecourse():
            # only timecourse data can be uniquely converted to DataFrame
            # higher dimensional data will be flattened.
            logger.warning("Higher dimensional data, data will be mean.")

        data = {v: self.xds[v].values.flatten() for v in self.xds.keys()}
        df = pd.DataFrame(data)
        return df

    def to_tsv(self, path_tsv):
        """Write data to tsv."""
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(path_tsv, sep="\t", index=False)
        else:
            logger.warning("Could not write TSV")

    @staticmethod
    def from_netcdf(path):
        """Read from netCDF."""
        ds = xr.open_dataset(path)
        return XResult(xdataset=ds, uinfo=None)
