"""Module for encoding simulation results and processed data."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.simulation import Dimension, ScanSim
from sbmlsim.units import Quantity, UnitsInformation

logger = logging.getLogger(__name__)


class XResult:
    """Result of simulations.

    A wrapper around xr.Dataset which adds unit support via
    dictionary lookups.
    """

    def __init__(self, xdataset: xr.Dataset, uinfo: UnitsInformation | None = None):
        """Initialize XResult.

        Args:
            xdataset: Dataset with the simulation results.
            uinfo: Units information for the variables in the dataset.
        """
        self.xds = xdataset
        self.uinfo = uinfo

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get item.

        Args:
            key: Variable key.

        Returns:
            DataArray for the key.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            return self.xds[key]
        except KeyError as err:
            logger.error("Key '%s' not in %s\n%s", key, self.xds, err)
            raise err

    def __getattr__(self, name: str) -> Any:
        """Provide dot access to keys.

        Args:
            name: Attribute name.

        Returns:
            Attribute of the wrapped dataset.
        """
        if name in {"xds", "scan", "uinfo"}:
            # local field lookup
            return getattr(self, name)
        # forward lookup to xds
        return getattr(self.xds, name)

    def __str__(self) -> str:
        """Get string."""
        return f"<XResult: {self.xds.__repr__()},\n{self.uinfo}>"

    def _quantity(self, key: str, values: np.ndarray) -> Quantity:
        """Create quantity with the units of the key.

        Args:
            key: Variable key.
            values: Values for the quantity.

        Returns:
            Quantity with units of the key.

        Raises:
            ValueError: If no units information is available.
        """
        if self.uinfo is None:
            raise ValueError(f"No units information available for key '{key}'.")
        return values * self.uinfo.ureg(self.uinfo[key])

    def dim_mean(self, key: str) -> Quantity:
        """Get mean over all added dimensions.

        Args:
            key: Variable key.

        Returns:
            Mean values with units.

        Raises:
            KeyError: If the key does not exist in the result.
        """
        try:
            values = self.xds[key].mean(dim=self._redop_dims(), skipna=True).values
        except KeyError as err:
            logger.error(
                "Key '%s' does not exist in XResult. Add the key to the experiment "
                "via add_selections in 'Experiment.datagenerators'.",
                key,
            )
            raise err
        return self._quantity(key, values)

    def dim_std(self, key: str) -> Quantity:
        """Get standard deviation over all added dimensions.

        Args:
            key: Variable key.

        Returns:
            Standard deviation values with units.
        """
        values = self.xds[key].std(dim=self._redop_dims(), skipna=True).values
        return self._quantity(key, values)

    def dim_min(self, key: str) -> Quantity:
        """Get minimum over all added dimensions.

        Args:
            key: Variable key.

        Returns:
            Minimum values with units.
        """
        values = self.xds[key].min(dim=self._redop_dims(), skipna=True).values
        return self._quantity(key, values)

    def dim_max(self, key: str) -> Quantity:
        """Get maximum over all added dimensions.

        Args:
            key: Variable key.

        Returns:
            Maximum values with units.
        """
        values = self.xds[key].max(dim=self._redop_dims(), skipna=True).values
        return self._quantity(key, values)

    def _redop_dims(self) -> list[str]:
        """Dimensions for reducing operations.

        Returns:
            All dimensions besides the time dimension.
        """
        return [str(dim_id) for dim_id in self.xds.dims if dim_id != "_time"]

    @classmethod
    def from_dfs(
        cls,
        dfs: list[pd.DataFrame],
        scan: ScanSim | None = None,
        uinfo: UnitsInformation | None = None,
    ) -> "XResult":
        """Create XResult from DataFrames.

        Structure is based on the underlying scans.

        Args:
            dfs: DataFrames of the individual simulations.
            scan: Scan defining the additional dimensions.
            uinfo: Units information.

        Returns:
            XResult combining the DataFrames.
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if uinfo is None:
            uinfo = UnitsInformation(udict={}, ureg=None)  # ty: ignore[invalid-argument-type]  # FIXME(cross-area): UnitsInformation.ureg should allow None

        df = dfs[0]
        num_dfs = len(dfs)

        # add time dimension
        shape = [len(df)]
        dims = ["_time"]
        coords = {"_time": df.time.values}
        columns = df.columns
        del df

        # Additional dimensions
        dimensions: list[Dimension]
        if scan is None:
            dimensions = [Dimension("_dfs", index=np.arange(num_dfs))]
        else:
            dimensions = scan.dimensions  # ty: ignore[invalid-assignment]  # FIXME(cross-area): ScanSim.dimensions attribute shadows the method of the same name

        # add additional dimensions
        for dimension in dimensions:
            shape.append(len(dimension))
            dim_id = dimension.dimension
            coords[dim_id] = dimension.index
            dims.append(dim_id)

        indices = Dimension.indices_from_dimensions(dimensions)
        data_dict = {col: np.empty(shape=shape) for col in columns}
        for k_df, df in enumerate(dfs):
            for column in columns:
                # trick to get the ':' in first time dimension
                index = (..., *indices[k_df])
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

    def to_netcdf(self, path_nc: str | Path) -> None:
        """Store results as netcdf.

        Args:
            path_nc: Path to the netCDF file.
        """
        self.xds.to_netcdf(path_nc)

    def is_timecourse(self) -> bool:
        """Check if timecourse.

        Returns:
            True if the result is a single timecourse.
        """
        # FIXME: better implementation necessary
        is_tc = True
        xds = self.xds
        if len(xds.dims) == 2:
            for dim in xds.dims:
                if dim == "_time":
                    continue
                if xds.sizes[dim] != 1:
                    is_tc = False
        else:
            return False
        return is_tc

    def to_mean_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with mean data.

        Returns:
            DataFrame with the mean over all dimensions.
        """
        res = {}
        for col in self.xds:
            res[col] = self.dim_mean(key=str(col))
        return pd.DataFrame(res)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame.

        Returns:
            DataFrame with flattened data.
        """
        if not self.is_timecourse():
            # only timecourse data can be uniquely converted to DataFrame
            # higher dimensional data will be flattened.
            logger.warning("Higher dimensional data, data will be mean.")

        data = {v: self.xds[v].values.flatten() for v in self.xds}
        return pd.DataFrame(data)

    def to_tsv(self, path_tsv: str | Path) -> None:
        """Write data to tsv.

        Args:
            path_tsv: Path to the TSV file.
        """
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(path_tsv, sep="\t", index=False)
        else:
            logger.warning("Could not write TSV")

    @staticmethod
    def from_netcdf(path: str | Path) -> "XResult":
        """Read from netCDF.

        Args:
            path: Path to the netCDF file.

        Returns:
            XResult without units information.
        """
        ds = xr.open_dataset(path)
        return XResult(xdataset=ds, uinfo=None)
