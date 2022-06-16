"""Module for encoding simulation results and processed data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from sbmlutils import log
from sbmlutils.console import console

from sbmlsim.simulation import Dimension, ScanSim


logger = log.get_logger(__name__)


class XResult:
    """Data structure for storing results.

    Results is always structured data.
    A wrapper around xr.Dataset from xarray
    """

    def __init__(self, xdataset: xr.Dataset):
        """Initialize XResult."""
        self.xds = xdataset

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get item."""
        try:
            return self.xds[key]
        except KeyError as err:
            logger.error(f"Key '{key}' not in {self.xds}" f"\n{err}")
            raise err

    def __getattr__(self, attr: str):
        """Provide dot access to keys."""
        if attr in {"xds", "scan"}:
            # local field lookup
            return getattr(self, attr)
        else:
            # forward lookup to xds
            return getattr(self.xds, attr)

    def __str__(self) -> str:
        """Get string."""
        return self.xds.__str__()

    def __repr__(self) -> str:
        """Get string representation."""
        return self.xds.__repr__()

    @staticmethod
    def from_dfs(
        dfs: List[pd.DataFrame],
        scan: ScanSim = None,
        udict: Optional[Dict[str, str]] = None,
    ) -> XResult:
        """Create XResult from DataFrames.

        Structure is based on the underlying scans which were performed.
        An optional unit dictionary can be provided.
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

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

        # create DataSet
        ds = xr.Dataset(
            {
                key: xr.DataArray(data=data, dims=dims, coords=coords)
                for key, data in data_dict.items()
            }
        )

        # set units attribute
        if udict:
            for key in data_dict:
                if key in udict:
                    ds[key].attrs["units"] = udict[key]
        return XResult(xdataset=ds)

    def to_netcdf(self, path: Path):
        """Store results as netcdf4/HDF5."""
        self.xds.to_netcdf(path, format="NETCDF4", engine="h5netcdf")

    @staticmethod
    def from_netcdf(path: Path) -> XResult:
        """Read from netCDF."""
        ds: xr.Dataset = xr.open_dataset(path)
        return XResult(xdataset=ds)

    def to_mean_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with mean data."""
        res = {}
        for col in self.xds:
            res[col] = self.mean_all_dims(key=col)
        return pd.DataFrame(res)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame.

        Only timecourse simulations without scan
        """
        if not self.is_no_scan():
            # only timecourse data can be uniquely converted to DataFrame
            # higher dimensional data will be flattened.
            logger.warning("Higher dimensional data, data will be mean.")

        data = {v: self.xds[v].values.flatten() for v in self.xds.keys()}
        df = pd.DataFrame(data)
        return df

    def is_scan(self) -> bool:
        """Check if scan.

        Checks if additional dimensions besides `_time` exist.
        """
        is_scan = False
        if len(self.xds.dims) == 2:
            for dim in self.xds.dims:
                if dim == "_time":
                    continue
                else:
                    # check length
                    if self.xds.dims[dim] != 1:
                        is_scan = True
        else:
            return True
        return is_scan

    def to_tsv(self, path_tsv):
        """Write data to tsv."""
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(path_tsv, sep="\t", index=False)
        else:
            logger.warning("Could not write TSV")

    def mean_all_dims(self, key: str) -> xr.Dataset:
        """Get mean over all dimensions.

        Skips NA.
        """
        return self.xds[key].mean(dim=self._redop_dims(), skipna=True)

    def std_all_dims(self, key: str) -> xr.Dataset:
        """Get standard deviation over all dimensions besides time.

        Skips NA.
        """
        return self.xds[key].std(dim=self._redop_dims(), skipna=True)

    def min_all_dims(self, key: str) -> xr.Dataset:
        """Get minimum over all dimensions besides time.

        Skips NA.
        """
        return self.xds[key].min(dim=self._redop_dims(), skipna=True)

    def max_all_dims(self, key: str) -> xr.Dataset:
        """Get maximum over all dimensions besides time.

        Skips NA.
        """
        return self.xds[key].max(dim=self._redop_dims(), skipna=True)

    def _redop_dims(self) -> List[str]:
        """Dimensions for reducing operations.

        All dimensions besides time.
        """
        return [dim_id for dim_id in self.dims if dim_id != "_time"]


if __name__ == "__main__":
    from sbmlsim.model import RoadrunnerSBMLModel
    from sbmlsim.resources import REPRESSILATOR_SBML

    r = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML).model
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres = XResult.from_dfs(dfs)
    console.print(xres)
