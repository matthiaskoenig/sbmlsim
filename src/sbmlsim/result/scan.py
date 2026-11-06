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
  dimension of values along its dimension (`<dimension>.<target>` when the
  target is also a variable of the result, which keeps the plain name), and
  `time` on a grid;
- `attrs`: `dims`, the dimensions of the scan in their order, `units`, the
  unit of every variable and coordinate, `scan` and `integrator_settings`,
  the provenance, `changes`, the times of the changes after the start of the
  simulations in the time of the result, and `errors` for a run with
  `on_error="flag"`.

The result of a block of points of a scan, see the argument `points` of
`Simulator.run`, has the dimension `_index` of the flat indices of its points
in C order instead of the dimensions of the scan, every coordinate of a
dimension along `_index` and the attribute `block` (`start`, `stop`, the
`shape` and `dims` of the scan and the dimension of every coordinate,
`coords`); `ScanResult.combine` combines the blocks of a scan to the result of
the scan.

The output of a simulation has two time points at the time of a change after
the start, the state before the change and the state after it, as at an event
of the model; the time of a grid then appears twice. The idioms of xarray
which need a unique index raise at such a time, e.g. `ds.interp`, `ds.sel`
with `method="nearest"` and `ds.reindex`: `ScanResult.interpolate` puts a
result on other times, `sel(time=t)` at a change selects both rows and
`isel` one of them.

`nca` and `to_timecourses` hand a PK analysis and a timecourse over to
pkpdutils, whose layout `(*dims, time)` the result has.

`xarray` does the rest: `res["[X]"].sel(dose=10)`, `res.ds.to_dataframe()`.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from sbmlsim.result.netcdf import (
    Compression,
    json_attributes,
    read_netcdf,
    write_netcdf,
)
from sbmlsim.result.timecourse import (
    GridWeights,
    apply_weights,
    grid_weights,
    interpolate,
    without_states_before,
)
from sbmlsim.units import (
    DimensionalityError,
    Quantity,
    UnitRegistry,
    setup_matplotlib,
)
from sbmlsim.units import ureg as package_ureg

if TYPE_CHECKING:
    from pkpdutils import NCAResult, Timecourses

#: the dimension of the time of a grid
TIME = "time"
#: the dimension of the output points of the ragged layout
POINT = "_point"
#: the dimension of the points of a block of a scan, their flat indices in C order
INDEX = "_index"
#: the attribute of a block of a scan, see `ScanResult.combine`
BLOCK = "block"
#: the attribute of the times of the changes after the start
CHANGES = "changes"
#: the dimension of `ScanResult.summary`
STATISTIC = "statistic"
#: the statistics of `ScanResult.summary`
STATISTICS: tuple[str, ...] = ("mean", "sd", "cv", "min", "max")
#: the variable of the outcome of every point, reserved like `TIME`
STATUS = "status"
#: the variable of the flags of the analysis of a PK observable, `<id>.flags`
FLAGS = "flags"
#: the most elements (points of the scan times time points) of the union of the
#: time points of a ragged result which `ScanResult.summary` interpolates onto
MAX_UNION_ELEMENTS = 10_000_000
#: the most messages of failed points a result keeps and a chunk reports
MAX_ERRORS: int = 10
#: the most variables the representation of a result names, a model has hundreds
REPR_VARIABLES = 8


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
        names = self.variables
        shown = ", ".join(names[:REPR_VARIABLES])
        if len(names) > REPR_VARIABLES:
            shown += f", ... ({len(names)} variables)"
        return f"ScanResult({sizes}; {shown})"

    @property
    def units(self) -> dict[str, str]:
        """Get the unit of every variable and coordinate."""
        return self.ds.attrs.get("units", {})

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the dimensions of the scan which the result has, in their order."""
        return tuple(d for d in self.ds.attrs.get("dims", []) if d in self.ds.dims)

    @property
    def is_block(self) -> bool:
        """Check whether the result is a block of a scan, see `combine`."""
        return BLOCK in self.ds.attrs

    def _check_whole(self, method: str) -> None:
        """Check that the result is no block of a scan, for a method.

        Raises:
            ValueError: for a block of a scan.
        """
        if self.is_block:
            raise ValueError(
                f"`{method}` needs the result of the whole scan, this one is a "
                f"block of its points: combine the blocks first, see "
                f"`ScanResult.combine`."
            )

    @property
    def ragged(self) -> bool:
        """Check whether every simulation keeps its own time points."""
        return POINT in self.ds.dims

    @property
    def variables(self) -> list[str]:
        """Get the variables of the selections, without `time` and `status`."""
        return [str(name) for name in self.ds.data_vars if name not in (TIME, STATUS)]

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

    @property
    def q(self) -> _Quantities:
        """Get the variables and coordinates of the result as quantities.

        `res.q["cp"]` is `res.quantity("cp")`, the values with their unit,
        e.g. `res.q["time"].to("hr")`. Quantities plot with matplotlib in
        their units, see `sbmlsim.units.setup_matplotlib`.
        """
        setup_matplotlib()
        return _Quantities(self)

    def quantity(self, key: str, ureg: UnitRegistry | None = None) -> Quantity:
        """Get the values of a variable or coordinate with its unit.

        The `cv` of a summary is dimensionless; a variable with a dimension
        `statistic` which holds statistics of different units needs a
        selection of the statistic first. A timecourse of one point of a
        ragged result, e.g. `res.sel(dose="high")`, is its time points without
        the padding, so `res.sel(dose="high").q["cp"].max()` is its peak.

        Args:
            key: the variable or coordinate.
            ureg: the unit registry of the quantity, the one of the package
                by default; e.g. the registry of a simulation experiment.

        Raises:
            ValueError: if the key holds labels, has no unit, or mixes units.
        """
        array = self[key]
        if array.dtype.kind not in "fiub":
            raise ValueError(f"'{key}' holds labels, not values, so it is no quantity.")
        if key not in self.units:
            raise ValueError(f"'{key}' has no unit in the units of the result.")
        unit = self.units[key]
        overrides: dict[str, str] = self.ds.attrs.get("statistic_units", {})
        if overrides and STATISTIC in array.coords and key in self.ds.data_vars:
            labels = np.atleast_1d(array.coords[STATISTIC].values)
            found = {overrides.get(str(label), unit) for label in labels}
            if len(found) > 1:
                raise ValueError(
                    f"'{key}' has statistics of different units, select a "
                    f"statistic first, e.g. sel(statistic='mean')."
                )
            unit = found.pop()
        values = np.asarray(array.values, dtype=float)
        if array.dims == (POINT,) and self.ds[TIME].dims == (POINT,):
            # one point of a ragged result: its time points, not the padding
            values = values[~np.isnan(np.asarray(self.ds[TIME].values, dtype=float))]
        registry = ureg if ureg is not None else package_ureg
        return registry.Quantity(values, unit)

    def nca(self, observable: str) -> NCAResult:
        """Get the analysis of a PK observable as the result of pkpdutils.

        The kept parameters of the observable, `<id>.<parameter>`, with their
        units and its flags over the dimensions of the scan. A point which
        failed has `NaN` parameters and the flags `NCAFlag.NO_DATA`, which
        pkpdutils gives a sample without data.

        Args:
            observable: the id of the PK observable.

        Returns:
            The `pkpdutils.NCAResult`.

        Raises:
            KeyError: if the result does not keep the flags of the observable.
        """
        from pkpdutils import NCAFlag, NCAResult

        prefix = f"{observable}."
        flags = f"{prefix}{FLAGS}"
        if flags not in self.ds.data_vars:
            raise KeyError(
                f"The result does not keep the PK observable '{observable}': keep "
                f"its flags '{flags}', e.g. keep=['{observable}']."
            )
        data_vars: dict[str, xr.DataArray] = {}
        for name in self.ds.data_vars:
            name = str(name)
            if not name.startswith(prefix):
                continue
            values = self.ds[name]
            parameter = name.removeprefix(prefix)
            if parameter == FLAGS:
                values = values.fillna(int(NCAFlag.NO_DATA)).astype(int)
            data_vars[parameter] = values.assign_attrs(units=self.units.get(name, ""))
        return NCAResult(xr.Dataset(data_vars).reset_coords(drop=True))

    def to_timecourses(self, key: str, **kwargs: Any) -> Timecourses:
        """Get a timecourse of the result as the timecourses of pkpdutils.

        pkpdutils takes a time once: of the states before and after a change
        at its time (`changes` of the attributes) the timecourse has the
        state after it, as a `PK` observable analyses it.

        Args:
            key: the variable, a timecourse.
            **kwargs: the other arguments of `Timecourses.from_arrays`, e.g.
                `dose` and `route`.

        Returns:
            The `pkpdutils.Timecourses` over the dimensions of the scan.

        Raises:
            ValueError: if the variable is no timecourse of the result.
        """
        from pkpdutils import Timecourses

        tdim = POINT if self.ragged else TIME
        if key not in self.ds.data_vars or tdim not in self.ds[key].dims:
            raise ValueError(f"'{key}' is no timecourse of the result.")
        dims = tuple(self.dims)
        values = self.ds[key].transpose(*dims, tdim).values
        time = (
            self.ds[TIME].transpose(*dims, POINT).values
            if self.ragged
            else self.ds[TIME].values
        )
        changes = np.asarray(self.ds.attrs.get(CHANGES, []), dtype=float)
        if changes.size:
            shape = values.shape
            flat_time = np.broadcast_to(time, shape).reshape(-1, shape[-1])
            flat_time, flat_values = without_states_before(
                flat_time,
                values.reshape(-1, shape[-1]),
                [changes] * flat_time.shape[0],
            )
            if self.ragged or not np.all(flat_time == flat_time[:1]):
                time = flat_time.reshape(*shape[:-1], -1)
            else:
                time = flat_time[0]
            values = flat_values.reshape(*shape[:-1], -1)
        coords = {dim: self.ds[dim].values for dim in dims}
        return Timecourses.from_arrays(
            time,
            values,
            time_unit=self.units.get(TIME) or "dimensionless",
            unit=self.units.get(key) or "dimensionless",
            dims=dims,
            coords=coords or None,
            **kwargs,
        )

    def to(
        self, units: Mapping[str, Any], ureg: UnitRegistry | None = None
    ) -> ScanResult:
        """Get the result with variables and coordinates in other units.

        The labels, the dimensions and every other variable stay as they are,
        so a converted result selects, tabulates and plots as before, e.g.
        `res.to({"time": "hr", "pk.cmax": "µM"})["pk.cmax"].to_pandas()`.
        Converting `time` converts the times of the changes as well. The
        `cv` of a summary is dimensionless and stays as it is.

        Args:
            units: the variable or coordinate and its unit, a string or a unit
                of pint.
            ureg: the unit registry of the units, the one of the package by
                default; e.g. the registry of a simulation experiment.

        Returns:
            A new result, the result itself does not change.

        Raises:
            KeyError: if a key is no variable or coordinate of the result.
            ValueError: if a key holds labels, has no unit, or its unit cannot
                be converted into the unit asked for.
        """
        registry = ureg if ureg is not None else package_ureg
        ds = self.ds.copy()
        attrs = copy.deepcopy(ds.attrs)
        units_of: dict[str, str] = attrs.setdefault("units", {})
        overrides: dict[str, str] = attrs.get("statistic_units", {})
        for key, unit in units.items():
            array = self[key]
            if array.dtype.kind not in "fiu":
                raise ValueError(
                    f"'{key}' holds labels, not values, so it has no unit."
                )
            source = units_of.get(key)
            if not source:
                raise ValueError(f"'{key}' has no unit in the units of the result.")
            target = registry.Unit(unit) if isinstance(unit, str) else unit
            values = np.array(array.values, dtype=float)
            converted = np.array(values)
            # the statistics of a summary in another unit, e.g. `cv`, stay
            mask: Any = slice(None)
            if overrides and STATISTIC in array.dims and key in ds.data_vars:
                axis = array.dims.index(STATISTIC)
                keep = np.array(
                    [str(label) not in overrides for label in array[STATISTIC].values]
                )
                mask = (slice(None),) * axis + (keep,)
            try:
                converted[mask] = (
                    registry.Quantity(values[mask], source).to(target).magnitude
                )
            except DimensionalityError as err:
                raise ValueError(
                    f"'{key}' in '{source}' cannot be converted into '{unit}': {err}"
                ) from err
            if key in ds.coords:
                ds = ds.assign_coords({key: (array.dims, converted, array.attrs)})
            else:
                ds[key] = (array.dims, converted, array.attrs)
            units_of[key] = str(target)
            if key == TIME and attrs.get(CHANGES):
                attrs[CHANGES] = (
                    registry.Quantity(np.asarray(attrs[CHANGES], dtype=float), source)
                    .to(target)
                    .magnitude.tolist()
                )
        ds.attrs = attrs
        return ScanResult(ds)

    def sel(self, **indexers: Any) -> ScanResult:
        """Select by labels, see `xarray.Dataset.sel`."""
        return ScanResult(self.ds.sel(indexers))

    def isel(self, **indexers: Any) -> ScanResult:
        """Select by positions, see `xarray.Dataset.isel`."""
        return ScanResult(self.ds.isel(indexers))

    def time_points(self) -> np.ndarray:
        """Get the times of the grid, or the union of the times of the simulations.

        A time which a simulation has twice, the states before and after a
        change or an event of the model at it, is twice in the union, so an
        interpolation onto the union keeps the jump, see `interpolate`.
        """
        times = np.asarray(self.ds[TIME].values, dtype=float)
        if not self.ragged:
            return times.ravel()
        rows = times.reshape(-1, times.shape[-1])
        # the occurrence of every time point among the equal ones before it
        position = np.arange(rows.shape[-1])
        repeated = np.zeros(rows.shape, dtype=bool)
        repeated[:, 1:] = rows[:, 1:] == rows[:, :-1]
        occurrence = position - np.maximum.accumulate(
            np.where(repeated, 0, position), axis=-1
        )
        finite = np.isfinite(rows)
        pairs = np.unique(
            np.stack([rows[finite], occurrence[finite].astype(float)]), axis=1
        )
        return pairs[0]

    def _grid(self, times: ArrayLike | Quantity) -> np.ndarray:
        """Get times as numbers in the time unit of the result.

        Raises:
            ValueError: see `time_magnitudes`.
        """
        return time_magnitudes(times, self.units.get(TIME), "The result")

    def interpolate(self, times: ArrayLike | Quantity) -> ScanResult:
        """Get the result on a grid of times.

        Every simulation is interpolated linearly onto the times, see
        `sbmlsim.result.timecourse.grid_weights`; a time outside of a
        simulation is `NaN`. A simulation has the states before and after a
        change and an event of the model at its time, and time points closer
        than the resolution of its time (`sbmlsim.simulator.plan.time_tolerance`
        of its first and last time) are one jump: a time of the grid before a
        change is interpolated toward the state before it, the time of a
        change (`changes` of the attributes) sees the state after it, and the
        first of two equal times of the grid the state before it. The
        variables without a time stay as they are.

        Args:
            times: the times of the grid, numbers in the time unit of the
                result or a quantity.

        Returns:
            The result with the dimension `time`.

        Raises:
            ValueError: if the result is a block of a scan, or the times are a
                quantity and the result has no unit of time or another one,
                see `time_magnitudes`.
        """
        self._check_whole("interpolate")
        grid = self._grid(times)
        tdim = POINT if self.ragged else TIME
        time_dims = self.ds[TIME].dims
        changes = np.asarray(self.ds.attrs.get(CHANGES, []), dtype=float)
        variables: dict[str, Any] = {}
        weights: GridWeights | None = None
        for name, array in self.ds.data_vars.items():
            if name == TIME:
                continue
            if tdim not in array.dims:
                variables[str(name)] = array
                continue
            if self.ragged:
                order = time_dims
                values = np.asarray(array.transpose(*order).values, dtype=float)
                times_of = np.asarray(self.ds[TIME].values, dtype=float)
                out = np.full((*values.shape[:-1], grid.size), np.nan)
                for index in np.ndindex(*values.shape[:-1]):
                    out[index] = interpolate(
                        times_of[index],
                        values[index],
                        grid,
                        _resolution(times_of[index]),
                        changes,
                    )
            else:
                order = (*[d for d in array.dims if d != TIME], TIME)
                values = np.asarray(array.transpose(*order).values, dtype=float)
                if weights is None:
                    times_of = np.asarray(self.ds[TIME].values, dtype=float)
                    weights = grid_weights(
                        times_of, grid, _resolution(times_of), changes
                    )
                out = apply_weights(weights, values)
            variables[str(name)] = ([*order[:-1], TIME], out)
        coords = {
            name: coord
            for name, coord in self.ds.coords.items()
            if name != TIME and tdim not in coord.dims
        }
        coords[TIME] = grid
        return ScanResult(
            xr.Dataset(variables, coords=coords, attrs=copy.deepcopy(self.ds.attrs))
        )

    def summary(
        self,
        dims: str | Sequence[str] | None = None,
        statistics: Sequence[str] = STATISTICS,
        quantiles: Sequence[float] = (),
        times: ArrayLike | Quantity | None = None,
    ) -> ScanResult:
        """Get statistics of the variables over dimensions of the scan.

        A result in the ragged layout is interpolated onto the union of its
        time points first, or onto `times`; the union has a time point per step
        of every simulation, so it is refused above `MAX_UNION_ELEMENTS`
        elements (points of the scan times time points).

        `sd` is the sample standard deviation and `cv` the ratio of `sd` and
        `mean`; a quantile `q` is the statistic `q<q>`, e.g. `q0.05`. `NaN`,
        e.g. of a failed point, is skipped. The unit of a variable is the unit
        of its statistics, except of `cv`, a ratio, which `statistic_units` in
        the attributes records.

        Args:
            dims: the dimensions to reduce, every dimension of the scan by
                default.
            statistics: statistics of `STATISTICS`.
            quantiles: quantiles between 0 and 1.
            times: the times to interpolate onto before, numbers in the time
                unit of the result or a quantity; the union of the time points
                of a ragged result by default.

        Returns:
            The result with the dimension `statistic` instead of the reduced
            dimensions.

        Raises:
            ValueError: if the result is a block of a scan, a dimension is no
                dimension of the scan, a statistic is unknown, a quantile is
                outside of [0, 1] or the union of the time points of a ragged
                result is too large.
        """
        self._check_whole("summary")
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
        if times is not None:
            source = self.interpolate(times)
        elif self.ragged:
            union = self.time_points()
            n_points = int(np.prod([self.ds.sizes[d] for d in self.dims]))
            if n_points * union.size > MAX_UNION_ELEMENTS:
                raise ValueError(
                    f"The union of the time points of the {n_points} points has "
                    f"{union.size} times, {n_points * union.size} elements per "
                    f"variable, above {MAX_UNION_ELEMENTS}. Pass times= to "
                    f"summary, call interpolate(times) first, or run with time= "
                    f"for a common grid."
                )
            source = self.interpolate(union)
        else:
            source = self
        ds = source.ds.drop_vars(STATUS, errors="ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            parts, labels = _statistics(ds, reduced, statistics, quantiles)
        summary = xr.concat(parts, dim=pd.Index(labels, name=STATISTIC))
        summary.attrs = copy.deepcopy(self.ds.attrs)
        # the labels of the statistics carry no unit
        summary.attrs.setdefault("units", {})[STATISTIC] = ""
        if "cv" in labels:
            summary.attrs["statistic_units"] = {"cv": "dimensionless"}
        return ScanResult(summary)

    def to_netcdf(self, path: str | Path, compression: Compression = "zstd") -> None:
        """Write the result as compressed netCDF4, the attributes as JSON.

        The attributes must be JSON serializable, numpy scalars and arrays
        are written as numbers and lists.

        Args:
            path: the file.
            compression: `"zstd"`, `"gzip"`, which every reader of HDF5 reads,
                or `None`, see `sbmlsim.result.netcdf`.
        """
        write_netcdf(self.ds, path, compression)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ScanResult:
        """Read a result written by `to_netcdf`.

        Raises:
            ValueError: if the file has no attributes of a result.
        """
        return cls(read_netcdf(path))

    @classmethod
    def combine(cls, parts: Sequence[ScanResult]) -> ScanResult:
        """Combine the results of the blocks of a scan to the result of the scan.

        A block is the result of a run of some points of a scan, see the
        argument `points` of `sbmlsim.simulator.Simulator.run`. The blocks,
        in any order, in memory or read with `from_netcdf`, cover every point
        of the scan once; the result is the result of a run of the whole scan,
        bitwise: the blocks are sorted by their first point, the dimension
        `_point` of a ragged result is padded with `NaN` to the longest block,
        the arrays are concatenated along `_index` and reshaped into the
        dimensions of the scan, and a coordinate of a dimension takes its
        values along its dimension. The `errors` of a run with
        `on_error="flag"` are the ones of the blocks in their order, at most
        `MAX_ERRORS`.

        A block has the layout of `_index` (see the module) and not the
        dimensions of the scan: `dims` of a block is empty, `summary` and
        `interpolate` raise for it, and `to` and `q` treat `_index` as any
        dimension; they are meant for the combined result.

        Args:
            parts: the results of the blocks.

        Returns:
            The result of the scan.

        Raises:
            ValueError: if there is no block, a result is no block of a scan,
                the blocks are results of other scans (an attribute other than
                `block` and `errors`, a variable or a coordinate differs), or
                they miss points of the scan or have a point more than once.
        """
        if not parts:
            raise ValueError("There are no blocks to combine.")
        for part in parts:
            if BLOCK not in part.ds.attrs or INDEX not in part.ds.dims:
                raise ValueError(
                    "A result to combine is no block of a scan, see the argument "
                    "`points` of `Simulator.run`."
                )
        ordered = sorted(parts, key=lambda part: int(part.ds.attrs[BLOCK]["start"]))
        _check_same_scan(ordered)
        first = ordered[0].ds
        block = first.attrs[BLOCK]
        shape = tuple(int(size) for size in block["shape"])
        dims = [str(dim) for dim in block["dims"]]
        _check_coverage(ordered, math.prod(shape))
        rows = max(part.ds.sizes.get(POINT, 0) for part in ordered)
        padded = [pad_points(part.ds, rows) for part in ordered]
        data_vars: dict[str, xr.Variable] = {}
        for name, variable in first.data_vars.items():
            if INDEX not in variable.dims:
                data_vars[str(name)] = variable.variable
                continue
            ordered_dims = variable.transpose(INDEX, ...).dims
            values = np.concatenate(
                [ds[name].transpose(INDEX, ...).values for ds in padded]
            )
            data_vars[str(name)] = xr.Variable(
                [*dims, *ordered_dims[1:]],
                values.reshape(*shape, *values.shape[1:]),
                attrs=variable.attrs,
            )
        coords: dict[str, xr.Variable] = {}
        for name, coordinate in first.coords.items():
            if name == INDEX:
                continue
            if INDEX not in coordinate.dims:
                coords[str(name)] = coordinate.variable
                continue
            dim = str(block["coords"][name])
            axis = dims.index(dim)
            values = np.concatenate([part.ds[name].values for part in ordered])
            along = tuple(slice(None) if k == axis else 0 for k in range(len(shape)))
            coords[str(name)] = xr.Variable(
                dim, values.reshape(shape)[along], attrs=coordinate.attrs
            )
        attrs = copy.deepcopy(dict(first.attrs))
        del attrs[BLOCK]
        if "errors" in attrs:
            errors = [e for part in ordered for e in part.ds.attrs.get("errors", [])]
            attrs["errors"] = errors[:MAX_ERRORS]
        return cls(xr.Dataset(data_vars, coords=coords, attrs=attrs))


def _definition(ds: xr.Dataset) -> dict[str, Any]:
    """Get what the blocks of one scan share, see `ScanResult.combine`.

    The attributes but `block` (without its `start` and `stop`) and `errors`,
    as JSON types, the dimensions of the variables and of the coordinates and
    the values of the coordinates which are not along `_index`, e.g. the grid
    of times.
    """
    attrs = {k: v for k, v in ds.attrs.items() if k not in (BLOCK, "errors")}
    block = {k: v for k, v in ds.attrs[BLOCK].items() if k not in ("start", "stop")}
    definition: dict[str, Any] = json_attributes({**attrs, BLOCK: block})
    definition["variables"] = {
        str(name): [str(dim) for dim in variable.dims]
        for name, variable in ds.data_vars.items()
    }
    definition["coordinates"] = {
        str(name): [str(dim) for dim in coordinate.dims]
        if INDEX in coordinate.dims
        else coordinate.values.tolist()
        for name, coordinate in ds.coords.items()
    }
    return definition


def _check_same_scan(ordered: Sequence[ScanResult]) -> None:
    """Check that the blocks are results of one scan, see `ScanResult.combine`.

    Raises:
        ValueError: if an attribute, a variable or a coordinate differs.
    """
    expected = _definition(ordered[0].ds)
    for part in ordered[1:]:
        other = _definition(part.ds)
        differ = sorted(
            key
            for key in expected.keys() | other.keys()
            if expected.get(key) != other.get(key)
        )
        if differ:
            raise ValueError(
                f"The blocks are results of other scans, they differ in "
                f"{', '.join(differ)}."
            )


def _check_coverage(ordered: Sequence[ScanResult], size: int) -> None:
    """Check that the sorted blocks cover the points of a scan once.

    Raises:
        ValueError: naming the missing points or the points in more than one
            block, or if a block has not the points of its attribute.
    """
    missing: list[range] = []
    doubled: list[range] = []
    reached = 0
    for part in ordered:
        start = int(part.ds.attrs[BLOCK]["start"])
        stop = int(part.ds.attrs[BLOCK]["stop"])
        if part.ds.sizes[INDEX] != stop - start:
            raise ValueError(
                f"The block {range(start, stop)} has {part.ds.sizes[INDEX]} points."
            )
        if start > reached:
            missing.append(range(reached, start))
        elif start < reached:
            doubled.append(range(start, min(stop, reached)))
        reached = max(reached, stop)
    if reached < size:
        missing.append(range(reached, size))
    if reached > size:
        raise ValueError(
            f"The blocks have points beyond the {size} points of the scan, up to "
            f"{reached}."
        )
    if missing:
        raise ValueError(
            f"The blocks miss the points {', '.join(map(str, missing))} of the "
            f"scan of {size} points."
        )
    if doubled:
        raise ValueError(
            f"The blocks hold points more than once: {', '.join(map(str, doubled))}."
        )


def pad_points(ds: xr.Dataset, rows: int) -> xr.Dataset:
    """Pad the dimension `_point` of a ragged result with `NaN` to a number of rows.

    The results of parts of the points of a scan, e.g. its blocks or the
    batches of a propagation, have the native time points of their longest
    point; padded to the longest part they concatenate to the result of all
    the points, which a run pads alike.

    Args:
        ds: the dataset of a result.
        rows: the rows of `_point`, at least the ones of the dataset.

    Returns:
        The dataset with `_point` of `rows`, the dataset itself if it has no
        `_point` or already `rows` of it.

    Raises:
        ValueError: for fewer rows than the dataset has.
    """
    if POINT not in ds.dims or ds.sizes[POINT] == rows:
        return ds
    if ds.sizes[POINT] > rows:
        raise ValueError(
            f"The result has {ds.sizes[POINT]} rows of '{POINT}', more than {rows}."
        )

    def padded(variable: xr.Variable) -> xr.Variable:
        if POINT not in variable.dims:
            return variable
        values = variable.values
        shape = list(values.shape)
        shape[variable.dims.index(POINT)] = rows
        dtype = np.result_type(values.dtype, np.float64)
        filled = np.full(shape, np.nan, dtype=dtype)
        filled[tuple(slice(0, size) for size in values.shape)] = values
        return xr.Variable(variable.dims, filled, attrs=variable.attrs)

    return xr.Dataset(
        {str(name): padded(v.variable) for name, v in ds.data_vars.items()},
        coords={str(name): padded(c.variable) for name, c in ds.coords.items()},
        attrs=ds.attrs,
    )


def time_magnitudes(
    times: ArrayLike | Quantity, unit: str | None, owner: str = "The model"
) -> np.ndarray:
    """Get times as numbers in a unit of time.

    Args:
        times: numbers in the unit or a quantity.
        unit: the unit of time of a model or a result; `None`, empty or
            `dimensionless` if it has none.
        owner: what has the unit, for a message.

    Returns:
        The times, one dimensional.

    Raises:
        ValueError: if the times are a quantity and the owner has no unit of
            time, or one the quantity cannot be converted to.
    """
    if not isinstance(times, Quantity):
        return np.asarray(times, dtype=float).ravel()
    if not unit or unit == "dimensionless":
        if times.dimensionless:
            return np.asarray(times.to("dimensionless").magnitude, dtype=float).ravel()
        raise ValueError(
            f"{owner} has no unit of time, give the times as numbers, not in "
            f"'{times.units}'."
        )
    try:
        return np.asarray(times.to(unit).magnitude, dtype=float).ravel()
    except DimensionalityError as err:
        raise ValueError(
            f"The times in '{times.units}' cannot be converted to the unit of "
            f"time '{unit}' of {owner[0].lower()}{owner[1:]}."
        ) from err


def _resolution(time: np.ndarray) -> float:
    """Get the resolution of the time of a simulation from its time points.

    It is `time_tolerance` of the first and the last finite time. Only the
    steps of the integrator have two time points closer than it, an event of
    the model, and their first and last time are the start and the end of
    the simulation, so it is the resolution of its plan; with a `time_shift`
    it is larger, which takes a step shorter than it as a jump, a change of
    the value of the order of the resolution.
    """
    # the simulator imports the results
    from sbmlsim.simulator.plan import time_tolerance

    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return 0.0
    return time_tolerance(float(finite[0]), float(finite[-1]))


def _statistics(
    ds: xr.Dataset,
    reduced: list[str],
    statistics: Sequence[str],
    quantiles: Sequence[float],
) -> tuple[list[xr.Dataset], list[str]]:
    """Reduce a dataset over dimensions, a part per statistic."""
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
    return parts, labels


class _Quantities:
    """The variables and coordinates of a result as quantities, see `ScanResult.q`."""

    def __init__(self, result: ScanResult) -> None:
        """Wrap a result."""
        self._result = result

    def __getitem__(self, key: str) -> Quantity:
        """Get the values of a variable or coordinate with its unit.

        Raises:
            KeyError: if the result has no variable or coordinate of the name.
            ValueError: if it holds labels or has no unit.
        """
        return self._result.quantity(key)

    def __contains__(self, key: object) -> bool:
        """Check whether the result has a variable or coordinate with a unit."""
        return key in self._result and key in self._result.units

    def __repr__(self) -> str:
        """Get the representation."""
        return f"Quantities of {self._result!r}"
