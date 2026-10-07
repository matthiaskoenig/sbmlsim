"""Module for encoding simulation results and processed data.

The results of the simulations of a scan are ragged: every simulation keeps
the time points of its own output, e.g. the steps of the integrator. An
`XResult` holds them in one `xarray.Dataset` with the dimension `_point`
first, the dimensions of the scan after it, and the time as a variable like
every selection; a simulation with fewer points is padded with `NaN`.
`interpolate` puts the simulations on a common grid, the dimension `_time`,
which is what the reductions over the scan (`dim_mean`, ...) need.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from sbmlsim.result.timecourse import TimecourseResult
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
        return self.uinfo.ureg.Quantity(values, self.uinfo[key])

    def dim_mean(self, key: str, times: ArrayLike | None = None) -> Quantity:
        """Get mean over all added dimensions.

        The simulations are interpolated onto `times` first, see
        `interpolate`, onto the union of their time points without them.

        Args:
            key: Variable key.
            times: the times of the result.

        Returns:
            The values over the time, with units.
        """
        return self._reduce(key, "mean", times)

    def dim_std(self, key: str, times: ArrayLike | None = None) -> Quantity:
        """Get standard deviation over all added dimensions.

        The simulations are interpolated onto `times` first, see
        `interpolate`, onto the union of their time points without them.

        Args:
            key: Variable key.
            times: the times of the result.

        Returns:
            The values over the time, with units.
        """
        return self._reduce(key, "std", times)

    def dim_min(self, key: str, times: ArrayLike | None = None) -> Quantity:
        """Get minimum over all added dimensions.

        The simulations are interpolated onto `times` first, see
        `interpolate`, onto the union of their time points without them.

        Args:
            key: Variable key.
            times: the times of the result.

        Returns:
            The values over the time, with units.
        """
        return self._reduce(key, "min", times)

    def dim_max(self, key: str, times: ArrayLike | None = None) -> Quantity:
        """Get maximum over all added dimensions.

        The simulations are interpolated onto `times` first, see
        `interpolate`, onto the union of their time points without them.

        Args:
            key: Variable key.
            times: the times of the result.

        Returns:
            The values over the time, with units.
        """
        return self._reduce(key, "max", times)

    def _reduce(self, key: str, operation: str, times: ArrayLike | None) -> Quantity:
        """Reduce a variable over the dimensions of the scan.

        Args:
            key: Variable key.
            operation: `mean`, `std`, `min` or `max`.
            times: the times to interpolate onto, see `interpolate`.

        Returns:
            The reduced values with units.

        Raises:
            KeyError: If the key does not exist in the result.
        """
        if key not in self.xds:
            logger.error(
                "Key '%s' does not exist in XResult. Add the key to the "
                "selections of the experiment.",
                key,
            )
            raise KeyError(key)
        xres = self
        if "_point" in self.xds.dims:
            xres = self.interpolate(
                self.time_points() if times is None else times, keys=[key]
            )
        elif times is not None:
            xres = self.interpolate(times, keys=[key])
        array = xres.xds[key]
        values = getattr(array, operation)(dim=xres._redop_dims(), skipna=True).values
        return self._quantity(key, values)

    def time_points(self) -> np.ndarray:
        """Get the union of the time points of the simulations, sorted."""
        if "_point" not in self.xds.dims:
            return np.asarray(self.xds["_time"].values, dtype=float)
        times = np.asarray(self.xds["time"].values, dtype=float).ravel()
        return np.unique(times[np.isfinite(times)])

    def is_ragged(self) -> bool:
        """Check whether the simulations have different time points."""
        if "_point" not in self.xds.dims:
            return False
        time = np.asarray(self.xds["time"].values, dtype=float)
        if time.ndim == 1:
            return False
        flat = time.reshape(time.shape[0], -1)
        return bool(
            np.isnan(flat).any() or not np.allclose(flat, flat[:, :1], equal_nan=True)
        )

    def interpolate(
        self, times: ArrayLike, keys: Sequence[str] | None = None
    ) -> "XResult":
        """Get the result on a common grid of times.

        Every simulation is interpolated linearly onto the times; a time
        outside of a simulation is `NaN`. The variable `time` is the grid.

        Args:
            times: the times of the grid.
            keys: the variables to interpolate, all by default.

        Returns:
            The result with the dimension `_time`, whose coordinate are the
            times, and the dimensions of the scan.
        """
        grid = np.asarray(times, dtype=float).ravel()
        if "_point" not in self.xds.dims:
            xds = self.xds if keys is None else self.xds[[*keys]]
            interpolated = xds.interp(_time=grid)
            return XResult(xdataset=interpolated, uinfo=self.uinfo)

        time = self.xds["time"]
        scan_dims = [str(d) for d in time.dims if d != "_point"]
        t = np.asarray(time.values, dtype=float)
        shape = t.shape[1:]
        coords: dict[str, Any] = {"_time": grid}
        for dim in scan_dims:
            if dim in self.xds.coords:
                coords[dim] = self.xds.coords[dim].values
        variables: dict[str, xr.DataArray] = {
            "time": xr.DataArray(
                data=np.broadcast_to(
                    grid.reshape((-1,) + (1,) * len(shape)), (grid.size, *shape)
                ).copy(),
                dims=["_time", *scan_dims],
                coords=coords,
                attrs=self.xds["time"].attrs,
            )
        }
        selected = list(self.xds.data_vars) if keys is None else list(keys)
        for key in selected:
            if key == "time":
                continue
            values = np.asarray(self.xds[key].values, dtype=float)
            out = np.full((grid.size, *shape), np.nan)
            for index in np.ndindex(*shape):
                tk = t[(slice(None), *index)]
                vk = values[(slice(None), *index)]
                mask = np.isfinite(tk)
                if mask.any():
                    out[(slice(None), *index)] = np.interp(
                        grid, tk[mask], vk[mask], left=np.nan, right=np.nan
                    )
            variables[str(key)] = xr.DataArray(
                data=out,
                dims=["_time", *scan_dims],
                coords=coords,
                attrs=self.xds[key].attrs,
            )
        return XResult(xdataset=xr.Dataset(variables), uinfo=self.uinfo)

    def _redop_dims(self) -> list[str]:
        """Dimensions for reducing operations.

        Returns:
            All dimensions besides the time dimension.
        """
        return [
            str(dim_id) for dim_id in self.xds.dims if dim_id not in {"_time", "_point"}
        ]

    @classmethod
    def from_timecourses(
        cls,
        results: Sequence[TimecourseResult],
        scan: ScanSim | None = None,
        uinfo: UnitsInformation | None = None,
    ) -> "XResult":
        """Create XResult from the results of timecourse simulations.

        Structure is based on the underlying scans: the result of the k-th
        simulation is placed at the indices of the k-th combination of the
        dimensions of the scan. Without a scan several results are entries of
        the dimension `_dfs`, a single result has no further dimension. The
        results keep their time points, see the module.

        Args:
            results: results of the individual simulations, which share their
                columns.
            scan: Scan defining the additional dimensions.
            uinfo: Units information.

        Returns:
            XResult combining the results.

        Raises:
            ValueError: if there are no results, a result has no time or the
                results differ in their columns.
        """
        if not results:
            raise ValueError("No results of timecourse simulations.")

        if uinfo is None:
            uinfo = UnitsInformation(udict={}, ureg=None)  # ty: ignore[invalid-argument-type]  # FIXME(cross-area): UnitsInformation.ureg should allow None

        first = results[0]
        columns = first.columns
        if "time" not in columns:
            raise ValueError(
                f"The results have no column 'time', the columns are {columns}."
            )
        for result in results:
            if result.columns != columns:
                raise ValueError(
                    f"The results differ in their columns: {columns} and "
                    f"{result.columns}."
                )
        n_point = max(len(result) for result in results)

        # Additional dimensions
        dimensions: list[Dimension]
        if scan is not None:
            dimensions = scan.dimensions
        elif len(results) > 1:
            dimensions = [Dimension("_dfs", index=np.arange(len(results)))]
        else:
            dimensions = []

        shape = [n_point]
        dims = ["_point"]
        coords: dict[str, np.ndarray] = {"_point": np.arange(n_point)}
        for dimension in dimensions:
            shape.append(len(dimension))
            dim_id = dimension.dimension
            coords[dim_id] = dimension.index
            dims.append(dim_id)

        # one array for all columns with the column as the first axis, so a
        # result is copied with a single assignment and every variable is a
        # contiguous block of it
        indices = Dimension.indices_from_dimensions(dimensions) if dimensions else [()]
        data = np.full(shape=(len(columns), *shape), fill_value=np.nan)
        for k, result in enumerate(results):
            data[(slice(None), slice(0, len(result)), *indices[k])] = result.values.T

        # Create the DataSet, a name which appears twice is its first column
        keys = {key: columns.index(key) for key in dict.fromkeys(columns)}
        ds = xr.Dataset(
            {
                key: xr.DataArray(data=data[k], dims=dims, coords=coords)
                for key, k in keys.items()
            }
        )
        for key in keys:
            if key in uinfo:
                # set units attribute
                ds[key].attrs["units"] = uinfo[key]
        return XResult(xdataset=ds, uinfo=uinfo)

    @classmethod
    def from_dfs(
        cls,
        dfs: pd.DataFrame | Sequence[pd.DataFrame],
        scan: ScanSim | None = None,
        uinfo: UnitsInformation | None = None,
    ) -> "XResult":
        """Create XResult from DataFrames.

        The DataFrames are converted to `TimecourseResult`, see
        `from_timecourses`, which the simulator uses directly.

        Args:
            dfs: DataFrames of the individual simulations.
            scan: Scan defining the additional dimensions.
            uinfo: Units information.

        Returns:
            XResult combining the DataFrames.
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        results = [
            TimecourseResult(
                columns=tuple(str(c) for c in df.columns),
                values=df.to_numpy(dtype=float),
            )
            for df in dfs
        ]
        return cls.from_timecourses(results=results, scan=scan, uinfo=uinfo)

    def to_netcdf(self, path_nc: str | Path) -> None:
        """Store results as netcdf.

        Args:
            path_nc: Path to the netCDF file.
        """
        self.xds.to_netcdf(path_nc)

    def is_timecourse(self) -> bool:
        """Check if timecourse.

        Returns:
            True if the result is a single timecourse, i.e. every dimension
            besides the time has the size one.
        """
        return all(
            self.xds.sizes[dim] == 1
            for dim in self.xds.dims
            if dim not in {"_time", "_point"}
        )

    def to_mean_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with mean data.

        Returns:
            DataFrame with the mean over all dimensions, in the units of the
            variables.
        """
        res = {}
        for col in self.xds:
            res[col] = self.dim_mean(key=str(col)).magnitude
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
        df = pd.DataFrame(data)
        if "time" in df.columns:
            # the padding of a ragged result
            df = df[df["time"].notna()].reset_index(drop=True)
        return df

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
