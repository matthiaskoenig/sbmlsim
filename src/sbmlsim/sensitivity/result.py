"""The result of a sensitivity analysis.

`SensitivityResult` wraps an `xarray.Dataset` with one variable per observable
and index, `<observable>.<index>` (e.g. `auc.ST`, `auc.ST_conf`,
`[S2].normalized`), over `(parameter, *other dimensions, [time])`: the
parameters of the design, every label of the other dimensions of the scan
(e.g. doses, conditions) and, for a timecourse on a grid, every time point.
`attrs` carry the units of every variable, the method, its options and the
provenance of the scan.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.result.scan import TIME, _json_default
from sbmlsim.sensitivity.classification import sensitivity_classification

#: the dimension of the parameters of the design
PARAMETER = "parameter"

#: the second dimension of the parameters of a second order index
PARAMETER_2 = "parameter_2"

#: the attribute which carries the attributes as JSON in a netCDF file
NETCDF_ATTRS = "sbmlsim"


class SensitivityResult:
    """The indices of a sensitivity analysis, see the module.

    Attributes:
        ds: the dataset.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset of indices.

        Args:
            ds: the dataset.

        Raises:
            ValueError: if the dataset has no dimension `parameter`.
        """
        if PARAMETER not in ds.dims:
            raise ValueError(f"A sensitivity result has the dimension '{PARAMETER}'.")
        self.ds = ds

    @property
    def units(self) -> dict[str, str]:
        """Get the unit of every variable."""
        return dict(self.ds.attrs.get("units", {}))

    @property
    def method(self) -> str:
        """Get the method of the analysis, e.g. `sobol`."""
        return str(self.ds.attrs.get("method", ""))

    @property
    def parameters(self) -> list[str]:
        """Get the parameters of the indices."""
        return [str(p) for p in self.ds[PARAMETER].values.tolist()]

    @property
    def observables(self) -> list[str]:
        """Get the observables, in the order of their first variable."""
        return list(
            dict.fromkeys(str(name).rsplit(".", 1)[0] for name in self.ds.data_vars)
        )

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get the variable of an index of an observable, e.g. `auc.ST`."""
        return self.ds[key]

    def __contains__(self, key: object) -> bool:
        """Check whether the result has a variable."""
        return key in self.ds.data_vars

    def index(
        self, name: str, observables: Sequence[str] | None = None
    ) -> xr.DataArray:
        """Stack an index of the scalar observables into `(parameter, observable, ...)`.

        Args:
            name: the index, e.g. `ST`.
            observables: the observables, every scalar observable which has
                the index by default.

        Returns:
            The index over `(parameter, observable, *other dimensions)`.

        Raises:
            KeyError: if no scalar observable has the index.
        """
        chosen = [
            o
            for o in (observables or self.observables)
            if f"{o}.{name}" in self.ds.data_vars
            and TIME not in self.ds[f"{o}.{name}"].dims
        ]
        if not chosen:
            raise KeyError(
                f"No scalar observable of the result has the index '{name}'."
            )
        stacked = xr.concat(
            [self.ds[f"{o}.{name}"] for o in chosen],
            dim=pd.Index(chosen, name="observable"),
        )
        dims = [
            PARAMETER,
            "observable",
            *(d for d in stacked.dims if d not in (PARAMETER, "observable")),
        ]
        return stacked.transpose(*dims)

    def sel(self, **indexers: Any) -> SensitivityResult:
        """Select labels, see `xarray.Dataset.sel`."""
        return SensitivityResult(self.ds.sel(**indexers))

    def isel(self, **indexers: Any) -> SensitivityResult:
        """Select positions, see `xarray.Dataset.isel`."""
        return SensitivityResult(self.ds.isel(**indexers))

    def to_dataframe(self, name: str) -> pd.DataFrame:
        """Get a variable as a table, a row per parameter.

        Args:
            name: the variable, e.g. `auc.ST`.

        Returns:
            The table; the other dimensions are its columns.
        """
        data = self.ds[name]
        others = [d for d in data.dims if d != PARAMETER]
        series: pd.Series = data.to_series()
        if others:
            return series.unstack(others)
        return series.to_frame(name)

    def classify(self, name: str) -> xr.DataArray:
        """Classify every value of an index, see `sensitivity_classification`.

        Args:
            name: the variable, e.g. `auc.normalized`.

        Returns:
            The classes as strings (`high`, `medium`, `low`, `negligible`), `""`
            for `NaN`.
        """
        data = self.ds[name]
        classes = np.vectorize(
            lambda v: "" if np.isnan(v) else str(sensitivity_classification(float(v))),
            otypes=[object],
        )(data.values)
        return data.copy(data=classes.astype(str))

    def to_netcdf(self, path: str | Path) -> None:
        """Write the result as netCDF, the attributes as JSON.

        Args:
            path: the file.
        """
        ds = self.ds.copy()
        ds.attrs = {NETCDF_ATTRS: json.dumps(self.ds.attrs, default=_json_default)}
        ds.to_netcdf(path, engine="h5netcdf")

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SensitivityResult:
        """Read a result written by `to_netcdf`.

        Args:
            path: the file.

        Returns:
            The result.
        """
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            loaded = ds.load()
        loaded.attrs = json.loads(loaded.attrs.pop(NETCDF_ATTRS, "{}"))
        return cls(loaded)
