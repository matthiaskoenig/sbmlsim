"""The result of a scan, see `sbmlsim.simulator.Simulator.run`.

A `ScanResult` wraps one `xarray.Dataset`:

- dimensions: the dimensions of the scan in their order, then `time` when
  every simulation has the same output times and `_point` otherwise, i.e.
  `(*dims, time)` or `(*dims, _point)`, the layout of the timecourses of
  pkpdutils;
- variables: one per selection over `(*dims, time)` or `(*dims, _point)`; in
  the ragged layout of `_point` every simulation keeps its own time points
  and the variable `time` over `(*dims, _point)` holds them, padded with
  `NaN`; `status` over the dimensions of the scan for a run with
  `on_error="flag"`;
- coordinates: the labels of every dimension, every changed target of a
  dimension of values along its dimension, and `time` on a grid;
- `attrs`: `dims`, the dimensions of the scan in their order, `units`, the
  unit of every variable and coordinate, `scan` and `integrator_settings`,
  the provenance, and `errors` for a run with `on_error="flag"`.

`xarray` does the rest: `res["[X]"].sel(dose=10)`, `res.ds.to_dataframe()`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from sbmlsim.result.timecourse import interpolate
from sbmlsim.units import Quantity, ureg

#: the dimension of the time of a grid
TIME = "time"
#: the dimension of the output points of the ragged layout
POINT = "_point"
#: the dimension of `ScanResult.summary`
STATISTIC = "statistic"
#: the statistics of `ScanResult.summary`
STATISTICS: tuple[str, ...] = ("mean", "sd", "cv", "min", "max")
#: the attribute of a netCDF file which holds the attributes of the result
NETCDF_ATTRS = "sbmlsim"


class ScanResult:
    """The result of a scan, see the module.

    Attributes:
        ds: the dataset.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset which follows the layout of the module."""
        self.ds = ds

    def __repr__(self) -> str:
        """Get the representation."""
        sizes = ", ".join(f"{dim}: {size}" for dim, size in self.ds.sizes.items())
        return f"ScanResult({sizes}; {', '.join(self.variables)})"

    @property
    def units(self) -> dict[str, str]:
        """Get the unit of every variable and coordinate."""
        return self.ds.attrs.get("units", {})

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the dimensions of the scan which the result has, in their order."""
        return tuple(d for d in self.ds.attrs.get("dims", []) if d in self.ds.dims)

    @property
    def ragged(self) -> bool:
        """Check whether every simulation keeps its own time points."""
        return POINT in self.ds.dims

    @property
    def variables(self) -> list[str]:
        """Get the variables of the selections, without `time` and `status`."""
        return [str(name) for name in self.ds.data_vars if name not in (TIME, "status")]

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get a variable or a coordinate.

        Raises:
            KeyError: if the result has no variable or coordinate of the name.
        """
        try:
            return self.ds[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is no variable or coordinate of the result, its "
                f"variables are {self.variables}."
            ) from None

    def __contains__(self, key: object) -> bool:
        """Check whether the result has a variable or coordinate."""
        return key in self.ds.variables

    def quantity(self, key: str) -> Quantity:
        """Get the values of a variable or coordinate with its unit."""
        values = np.asarray(self[key].values, dtype=float)
        return ureg.Quantity(values, self.units.get(key, ""))

    def sel(self, **indexers: Any) -> ScanResult:
        """Select by labels, see `xarray.Dataset.sel`."""
        return ScanResult(self.ds.sel(indexers))

    def isel(self, **indexers: Any) -> ScanResult:
        """Select by positions, see `xarray.Dataset.isel`."""
        return ScanResult(self.ds.isel(indexers))

    def time_points(self) -> np.ndarray:
        """Get the times of the grid, or the union of the time points of the simulations."""
        times = np.asarray(self.ds[TIME].values, dtype=float).ravel()
        if not self.ragged:
            return times
        return np.unique(times[np.isfinite(times)])

    def _grid(self, times: ArrayLike | Quantity) -> np.ndarray:
        """Get times as numbers in the time unit of the result."""
        if isinstance(times, Quantity):
            unit = self.units.get(TIME) or "dimensionless"
            return np.asarray(times.to(unit).magnitude, dtype=float).ravel()
        return np.asarray(times, dtype=float).ravel()

    def interpolate(self, times: ArrayLike | Quantity) -> ScanResult:
        """Get the result on a grid of times.

        Every simulation is interpolated linearly onto the times, see
        `sbmlsim.result.timecourse.interpolate`; a time outside of a
        simulation is `NaN`. The variables without a time stay as they are.

        Args:
            times: the times of the grid, numbers in the time unit of the
                result or a quantity.

        Returns:
            The result with the dimension `time`.
        """
        grid = self._grid(times)
        tdim = POINT if self.ragged else TIME
        time_dims = self.ds[TIME].dims
        variables: dict[str, Any] = {}
        for name, array in self.ds.data_vars.items():
            if name == TIME:
                continue
            if tdim not in array.dims:
                variables[str(name)] = array
                continue
            order = (
                time_dims
                if self.ragged
                else (*[d for d in array.dims if d != TIME], TIME)
            )
            values = np.asarray(array.transpose(*order).values, dtype=float)
            times_of = (
                np.asarray(self.ds[TIME].values, dtype=float)
                if self.ragged
                else np.broadcast_to(
                    np.asarray(self.ds[TIME].values, dtype=float), values.shape
                )
            )
            out = np.full((*values.shape[:-1], grid.size), np.nan)
            for index in np.ndindex(*values.shape[:-1]):
                out[index] = interpolate(times_of[index], values[index], grid)
            variables[str(name)] = ([*order[:-1], TIME], out)
        coords = {
            name: coord
            for name, coord in self.ds.coords.items()
            if name != TIME and tdim not in coord.dims
        }
        coords[TIME] = grid
        return ScanResult(
            xr.Dataset(variables, coords=coords, attrs=dict(self.ds.attrs))
        )

    def summary(
        self,
        dims: str | Sequence[str] | None = None,
        statistics: Sequence[str] = STATISTICS,
        quantiles: Sequence[float] = (),
    ) -> ScanResult:
        """Get statistics of the variables over dimensions of the scan.

        A result in the ragged layout is interpolated onto the union of its
        time points first. `sd` is the sample standard deviation and `cv` the
        ratio of `sd` and `mean`; a quantile `q` is the statistic `q<q>`,
        e.g. `q0.05`. `NaN`, e.g. of a failed point, is skipped. The unit of a
        variable is the unit of its statistics, except of `cv`, a ratio.

        Args:
            dims: the dimensions to reduce, every dimension of the scan by
                default.
            statistics: statistics of `STATISTICS`.
            quantiles: quantiles between 0 and 1.

        Returns:
            The result with the dimension `statistic` instead of the reduced
            dimensions.

        Raises:
            ValueError: if a dimension is no dimension of the scan, a
                statistic is unknown or a quantile is outside of [0, 1].
        """
        reduced = (
            list(self.dims)
            if dims is None
            else [dims]
            if isinstance(dims, str)
            else list(dims)
        )
        unknown = sorted(set(reduced) - set(self.dims))
        if unknown:
            raise ValueError(
                f"{unknown} are no dimensions of the scan, its dimensions are "
                f"{list(self.dims)}."
            )
        wrong = sorted(set(statistics) - set(STATISTICS))
        if wrong:
            raise ValueError(
                f"Unknown statistics {wrong}, the statistics are {list(STATISTICS)}."
            )
        outside = [q for q in quantiles if not 0.0 <= q <= 1.0]
        if outside:
            raise ValueError(f"The quantiles {outside} are outside of [0, 1].")
        source = self.interpolate(self.time_points()) if self.ragged else self
        ds = source.ds.drop_vars("status", errors="ignore")
        parts: list[xr.Dataset] = []
        labels: list[str] = []
        for statistic in statistics:
            if statistic == "mean":
                part = ds.mean(dim=reduced, skipna=True)
            elif statistic == "sd":
                part = ds.std(dim=reduced, skipna=True, ddof=1)
            elif statistic == "cv":
                part = ds.std(dim=reduced, skipna=True, ddof=1) / ds.mean(
                    dim=reduced, skipna=True
                )
            elif statistic == "min":
                part = ds.min(dim=reduced, skipna=True)
            else:
                part = ds.max(dim=reduced, skipna=True)
            parts.append(part)
            labels.append(statistic)
        for q in quantiles:
            parts.append(ds.quantile(q, dim=reduced, skipna=True).drop_vars("quantile"))
            labels.append(f"q{q:g}")
        summary = xr.concat(parts, dim=pd.Index(labels, name=STATISTIC))
        summary.attrs = dict(self.ds.attrs)
        return ScanResult(summary)

    def to_netcdf(self, path: str | Path) -> None:
        """Write the result as netCDF, the attributes as JSON."""
        ds = self.ds.copy()
        ds.attrs = {NETCDF_ATTRS: json.dumps(self.ds.attrs)}
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ScanResult:
        """Read a result written by `to_netcdf`."""
        ds = xr.load_dataset(path)
        ds.attrs = json.loads(ds.attrs.get(NETCDF_ATTRS, "{}"))
        return cls(ds)
