"""Module handling data (experiment and simulation)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import pint
import xarray as xr
from pint.facets.plain import PlainUnit

from sbmlsim.result import ScanResult
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.simulation import Scan
from sbmlsim.simulator.formula import compile_formula, reduce_formula
from sbmlsim.units import (
    DimensionalityError,
    Quantity,
    UnitRegistry,
    UnitsInformation,
)

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)


#: the dimension of the rows of a dataset
ROW = "row"

#: the dimensions a reduction of data runs along: the first one an array has
REDUCED_DIMS = (TIME, POINT, ROW)


def to_quantity(array: xr.DataArray, ureg: UnitRegistry) -> Quantity:
    """Get the values of data as a quantity.

    Args:
        array: the values of a `Data`, see `Data.get_data`.
        ureg: the registry of the quantity, e.g. the one of the experiment.

    Returns:
        The values with the unit of `attrs["units"]`.

    Raises:
        ValueError: if the array holds labels or has no unit.
    """
    if array.dtype.kind not in "fiub":
        raise ValueError(
            f"'{array.name}' holds labels, not values, so it is no quantity."
        )
    unit = array.attrs.get("units")
    if unit is None:
        raise ValueError(f"'{array.name}' has no unit, so it is no quantity.")
    return ureg.Quantity(np.array(array.values, dtype=float), unit)


def evaluate_function(
    formula: str,
    variables: Mapping[str, xr.DataArray | float],
    ureg: UnitRegistry,
) -> xr.DataArray:
    """Evaluate the formula of a `Data` of type FUNCTION on its data.

    The formula is the math of PEtab, see `sbmlsim.simulator.formula`. The
    arrays are broadcast by the names of their dimensions, so `y / dose` over
    `(dose, time)` and `(dose,)` needs no reshaping; two arrays must have the
    same coordinates of a dimension they share. `max` and `min` of a single
    argument reduce it along its time (`time`, or `_point` of a ragged
    result) or, without one, along the rows of a dataset, ignoring `NaN`, the
    padding of a scan; the other dimensions stay, so `Y/max(Y)` normalizes
    every simulation of a scan to its own maximum. With two or more arguments
    they are the elementwise maximum and minimum of PEtab. `mean` and `at`
    need the time points of a simulation, they are reductions of the
    observables of a scan. An array without a unit is evaluated as plain
    numbers, a formula without units is dimensionless.

    Args:
        formula: the formula.
        variables: the arrays of the data and the numbers of the parameters,
            by the identifiers of the formula.
        ureg: the registry of the units.

    Returns:
        The value of the formula over the broadcast dimensions, with its unit.

    Raises:
        ValueError: if the formula is not valid math, reads an identifier
            which is not a variable, uses `mean` or `at`, or combines arrays
            with different coordinates of one dimension.
    """
    reduced = reduce_formula(formula)
    scope: dict[str, xr.DataArray | float] = dict(variables)
    for reduction in reduced.reductions:
        if reduction.function in ("mean", "at"):
            raise ValueError(
                f"'{reduction.function}' in the formula '{formula}' needs the time "
                f"points of a simulation: it is a reduction of the observables of a "
                f"scan, see sbmlsim.simulation.observables."
            )
        x = _evaluate_part(reduction.arguments[0], scope, formula, ureg)
        scope[reduction.symbol] = _extreme(reduction.function, x)
    return _evaluate_part(reduced.outer, scope, formula, ureg)


def _evaluate_part(
    part: str,
    scope: Mapping[str, xr.DataArray | float],
    formula: str,
    ureg: UnitRegistry,
) -> xr.DataArray:
    """Evaluate a part of a formula without reductions, broadcasting by name.

    Raises:
        ValueError: if the part reads an identifier without a value, or two
            arrays have different coordinates of one dimension.
    """
    compiled = compile_formula(part)
    missing = [symbol for symbol in compiled.symbols if symbol not in scope]
    if missing:
        raise ValueError(
            f"The formula '{formula}' reads {missing}, which have no values."
        )
    arrays = {
        symbol: value
        for symbol in compiled.symbols
        if isinstance(value := scope[symbol], xr.DataArray)
    }
    try:
        aligned = xr.align(*arrays.values(), join="exact") if arrays else ()
    except ValueError as err:
        raise ValueError(
            f"The data of the formula '{formula}' has different coordinates of a "
            f"dimension: {err}"
        ) from err
    broadcast = dict(zip(arrays, xr.broadcast(*aligned), strict=True)) if arrays else {}
    if broadcast:
        # one order for all: the reduced dimensions (time, points, rows) last,
        # the others as the broadcast has them, so the time stays last
        first = next(iter(broadcast.values()))
        order = [d for d in first.dims if d not in REDUCED_DIMS] + [
            d for d in REDUCED_DIMS if d in first.dims
        ]
        broadcast = {k: v.transpose(*order) for k, v in broadcast.items()}
    arguments = [
        _argument(broadcast[symbol], ureg) if symbol in broadcast else scope[symbol]
        for symbol in compiled.symbols
    ]
    value = compiled.apply(arguments)
    if isinstance(value, Quantity):
        magnitude, unit = np.asarray(value.magnitude, dtype=float), str(value.units)
    else:
        magnitude, unit = np.asarray(value, dtype=float), "dimensionless"
    template = next(iter(broadcast.values()), None)
    if template is None:
        return xr.DataArray(magnitude, attrs={"units": unit})
    return xr.DataArray(
        np.broadcast_to(magnitude, template.shape).copy(),
        dims=template.dims,
        coords=template.coords,
        attrs={"units": unit},
    )


def _argument(array: xr.DataArray, ureg: UnitRegistry) -> Any:
    """Get the value of an array for a formula: a quantity, plain numbers without a unit."""
    if array.attrs.get("units") is None:
        return np.asarray(array.values, dtype=float)
    return to_quantity(array, ureg)


def _extreme(function: str, x: xr.DataArray) -> xr.DataArray:
    """Reduce data to its largest or smallest value along its reduced dimension.

    The dimension is the first of `REDUCED_DIMS` the array has; an array
    without one is a value per simulation and stays. `NaN`, the padding of a
    ragged result, is ignored, and only `NaN` gives `NaN`.
    """
    dim = next((d for d in REDUCED_DIMS if d in x.dims), None)
    if dim is None:
        return x
    ufunc = np.fmax if function == "max" else np.fmin
    reduced = x.reduce(lambda values, axis: ufunc.reduce(values, axis=axis), dim=dim)
    reduced.attrs = dict(x.attrs)
    return reduced


def check_sel(
    data: Data,
    sel: Mapping[str, Any],
    labels: Mapping[str, list[Any] | None],
) -> None:
    """Check that a selection names dimensions and labels of its source.

    Args:
        data: the data of the selection, named in the error.
        sel: the selection, see `Data`.
        labels: the labels of every dimension of the source, `None` for a
            dimension whose labels are not checked, e.g. the time of a task
            before it ran.

    Raises:
        ValueError: if a dimension is not one of the source, or a label is not
            one of its dimension; the message names the ones which exist.
    """
    for dim, label in sel.items():
        if dim not in labels:
            raise ValueError(
                f"{data} selects the dimension '{dim}', which its source has not: "
                f"{list(labels)}."
            )
        known = labels[dim]
        if known is None:
            continue
        wanted = (
            list(label) if isinstance(label, list | tuple | np.ndarray) else [label]
        )
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise ValueError(
                f"{data} selects {unknown} of the dimension '{dim}', whose labels "
                f"are {known}."
            )


def _select(
    array: xr.DataArray,
    sel: Mapping[str, Any],
    data: Data,
    source: xr.Dataset | xr.DataArray,
) -> xr.DataArray:
    """Select the labels of `sel` from data, see `Data`.

    A dimension of the source (the result of a task, or the array itself for
    a function) which the array has not is skipped: the array is constant
    along it. A dimension without labels is selected by position.

    Raises:
        ValueError: if a dimension is not one of the source, or a label is not
            one of its dimension, see `check_sel`.
    """
    check_sel(
        data,
        sel,
        {
            str(dim): (
                source[dim].values.tolist()
                if dim in source.coords
                else list(range(size))
            )
            for dim, size in source.sizes.items()
        },
    )
    by_label: dict[str, Any] = {}
    by_position: dict[str, Any] = {}
    for dim, label in sel.items():
        labelled = dim in source.coords
        if dim in array.dims:
            value = list(label) if isinstance(label, tuple | np.ndarray) else label
            (by_label if labelled else by_position)[dim] = value
    if by_label:
        array = array.sel(by_label)
    if by_position:
        array = array.isel(by_position)
    return array


def _rows(dset: pd.DataFrame, sel: Mapping[str, Any], data: Data) -> pd.DataFrame:
    """Select the rows of a dataset whose columns have the values of `sel`.

    Raises:
        ValueError: if a column is not one of the dataset, or no row is left.
    """
    rows = dset
    for column, value in sel.items():
        if column not in dset.columns:
            raise ValueError(
                f"{data} selects rows by the column '{column}', which the dataset "
                f"has not: {list(dset.columns)}."
            )
        values = (
            list(value) if isinstance(value, list | tuple | np.ndarray) else [value]
        )
        rows = rows[rows[column].isin(values)]
    if sel and rows.empty:
        raise ValueError(
            f"{data} selects {dict(sel)}, which no row of the dataset has."
        )
    return rows


def _dimension_values(
    experiment: SimulationExperiment, task_id: str, result: ScanResult
) -> dict[str, str]:
    """Get the values of the dimensions which a result stores under a plain name.

    The scan core stores the values a dimension sets to a target as a
    coordinate `<target>` when the target is no variable of the result and as
    `<dimension>.<target>` when it is; every array of a task names them
    `<dimension>.<target>`, so that the name does not depend on what else the
    task keeps.

    Args:
        experiment: the experiment of the task.
        task_id: the task of the result.
        result: the result of the task.

    Returns:
        The plain name of every such coordinate and its qualified name.
    """
    simulation = experiment._simulations[experiment._tasks[task_id].simulation_id]
    if not isinstance(simulation, Scan):
        return {}
    values: dict[str, str] = {}
    for dimension in simulation.dimensions:
        for target in dimension.values:
            if (
                target not in values
                and target in result.ds.coords
                and result.ds[target].dims == (dimension.id,)
            ):
                values[target] = f"{dimension.id}.{target}"
    return values


class Data:
    """Data of a simulation experiment.

    A column of a dataset, a selection of the results of a task or a function
    of other data. It is a promise which is fulfilled when the experiment runs.
    """

    class Types(Enum):
        """Data types."""

        TASK = 1
        DATASET = 2
        FUNCTION = 3

    def __init__(
        self,
        index: str,
        task: str | None = None,
        dataset: str | None = None,
        function: str | None = None,
        variables: dict[str, Data] | None = None,
        parameters: dict[str, float] | None = None,
        sid: str | None = None,
        sel: Mapping[str, Any] | None = None,
    ):
        """Construct data.

        Args:
            index: what the data is called, i.e., a selection of the results of
                a task (`"S"` is the amount, `"[S]"` the concentration and
                `"time"` the time), a column of a dataset or the name of a
                function.
            task: id of the task whose results are selected.
            dataset: id of the dataset whose column is selected.
            function: formula of the data, a function of `variables` and
                `parameters`.
            variables: the data the function reads, by the identifier in the
                formula.
            parameters: the numbers the function reads, by the identifier in
                the formula.
            sid: id of the data, if not given `<task or dataset>__<index>` for
                an amount and `<task or dataset>__conc__<index>` for a
                concentration (`"[S]"`), with `__` for a dot of the index
                (`pk.cmax` is `<task>__pk__cmax`) and `_x<hex>_` for every
                other character which is no letter, digit or underscore (the
                rate of change `X'` is `<task>__X_x27_`), so it is a valid SId.
            sel: labels of dimensions to select, `{dim: label}` keeps one point
                and drops the dimension, `{dim: [labels]}` keeps the dimension;
                for a dataset the values of columns whose rows are kept, see
                `get_data`. The sid does not depend on it, give `sid` to tell
                apart two data of one index with different selections.

        Raises:
            ValueError: if none of `task`, `dataset` and `function` is given.
        """
        #: the selection as given, `"[S]"` selects the concentration of `S`
        self.selection: str = index
        #: the name of the data, the selection without the brackets
        self.index: str = (
            index[1:-1] if index.startswith("[") and index.endswith("]") else index
        )
        self.task_id: str | None = task
        self.dset_id: str | None = dataset
        self.function: str | None = function
        self.variables: dict[str, Data] = variables if variables is not None else {}
        self.parameters: dict[str, float] = parameters if parameters is not None else {}
        self.unit: str | None = None
        self._sid = sid
        self.sel: dict[str, Any] = dict(sel) if sel else {}

        if (not self.task_id) and (not self.dset_id) and (not self.function):
            raise ValueError(
                "Either 'task_id', 'dset_id' or 'function' required for Data."
            )

    def __repr__(self) -> str:
        """Get string."""
        s: str
        if self.is_task():
            s = f"Data(Task|selection={self.selection}, task_id={self.task_id})"
        elif self.is_dataset():
            s = f"Data(DataSet|selection={self.selection}, dset_id={self.dset_id})"
        elif self.is_function():
            s = f"Data(Function|selection={self.selection}, function={self.function})"
        return s

    @property
    def sid(self) -> str:
        """Get id."""
        sid: str
        if self._sid:
            return self._sid
        name = _sid_part(self.index)
        if self.selection != self.index:
            name = f"conc__{name}"
        if self.task_id:
            sid = f"{self.task_id}__{name}"
        elif self.dset_id:
            sid = f"{self.dset_id}__{name}"
        else:
            sid = name

        return sid

    def is_task(self) -> bool:
        """Check if task."""
        return self.task_id is not None

    def is_dataset(self) -> bool:
        """Check if dataset."""
        return self.dset_id is not None

    def is_function(self):
        """Check if function."""
        return self.function is not None

    @property
    def name(self) -> str:
        """Get name."""
        name: str
        dtype = self.dtype
        if dtype in [Data.Types.TASK, Data.Types.DATASET]:
            name = self.index
        elif dtype == Data.Types.FUNCTION:
            if len(self.variables) == 1:
                name = next(iter(self.variables.values())).index
            else:
                name = self.index
        return name

    @property
    def dtype(self) -> Data.Types:
        """Get data type."""
        if self.task_id:
            dtype = Data.Types.TASK
        elif self.dset_id:
            dtype = Data.Types.DATASET
        elif self.function:
            dtype = Data.Types.FUNCTION
        else:
            raise ValueError("DataType could not be determined!")
        return dtype

    # TODO: calculations
    # TODO: conversion factors for units, necessary to store
    # TODO: storage of definitions on simulation.

    def to_dict(self):
        """Convert to dictionary."""
        # FIXME: ensure that the data is evaluated (via get_data) before
        #        it is serialized. Currently only the plotted variables are
        #        evaluated (-> units can not be resolved for the remainder).

        return {
            "type": self.dtype,
            "index": self.index,
            # the selection tells an amount `S` and a concentration `[S]` apart
            "selection": self.selection,
            "unit": self.unit,
            "task": self.task_id,
            "dataset": self.dset_id,
            "function": self.function,
            "variables": self.variables if self.variables else None,
            "sel": self.sel or None,
        }

    def get_data(
        self,
        experiment: SimulationExperiment,
        to_units: str | None = None,
    ) -> xr.DataArray:
        """Get the values of the data from an experiment which ran.

        The values are a labelled array named by the sid of the data, with the
        unit in `attrs["units"]` (`None` for labels), see `to_quantity`:

        - a task: the variable or coordinate of its `ScanResult` with its
          coordinates, a timecourse over `(*dims, time)` or `(*dims, _point)`
          for a ragged result padded with `NaN`, a value per simulation over
          `(*dims)`; `time` is the time, the plain name of a symbol its
          timecourse, also when the scan changes it, `<dimension>.<target>`
          the values a dimension sets and a coordinate of a dimension are
          over the dimension, and a dimension id gives its labels; every
          array of a task names the values of a dimension
          `<dimension>.<target>` among its coordinates;
        - a dataset: the column over the dimension `row`, whose coordinate is
          the index of the dataset;
        - a function: its formula on its variables and parameters, see
          `evaluate_function`.

        `sel` selects labels of the dimensions of a task or a function (a
        dimension of the scan which the data has not is skipped) and the rows
        of a dataset by the values of its columns. `unit` is set to the unit of
        the values.

        Args:
            experiment: the experiment whose datasets and results are read.
            to_units: the unit to convert the values to, their own unit
                without.

        Returns:
            The values.

        Raises:
            KeyError: if the dataset has no column of the index or no unit of
                it, or the result of the task has no variable or coordinate of
                the selection, e.g. the timecourse of a changed target which
                no data of the experiment read.
            ValueError: if the dataset is no `DataSet`, the result of the task
                is no `ScanResult` or its selection has no unit, a function has
                no formula, or `sel` names a dimension, label or column which
                does not exist.
            DimensionalityError: if the values cannot be converted to
                `to_units`.
        """
        if self.dtype == Data.Types.DATASET:
            array = self._dataset_array(experiment)
        elif self.dtype == Data.Types.TASK:
            array = self._task_array(experiment)
        else:
            array = self._function_array(experiment)
        array.name = self.sid
        self.unit = array.attrs.get("units")
        if to_units is not None:
            try:
                quantity = to_quantity(array, experiment.ureg).to(to_units)
            except DimensionalityError:
                logger.error("Could not convert '%s' to units '%s'.", self, to_units)
                raise
            array = array.copy(data=np.asarray(quantity.magnitude, dtype=float))
            array.attrs = {"units": to_units}
        return array

    def _dataset_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Get the column of the dataset over its rows, see `get_data`."""
        if not experiment._datasets:
            experiment._datasets = experiment.datasets()
        dset = experiment._datasets[str(self.dset_id)]
        if not isinstance(dset, DataSet):
            raise ValueError(
                f"DataSet '{self.dset_id}' is not a DataSet, but type '{type(dset)}'"
            )
        if dset.empty:
            logger.error("Adding empty dataset '%s' for '%s'.", dset, self.dset_id)
        uindex = self.index[:-3] if self.index.endswith(("_se", "_sd")) else self.index
        if self.index not in dset.columns:
            error_msg = (
                f"Data column with key '{self.index}' does not exist in dataset: "
                f"'{self.dset_id}'."
            )
            logger.error(error_msg)
            raise KeyError(error_msg)
        try:
            unit = dset.uinfo[uindex]
        except KeyError:
            logger.error(
                "Units missing for key '%s' in dataset: '%s'. Add missing units to "
                "dataset.",
                uindex,
                self.dset_id,
            )
            raise
        rows = _rows(dset, self.sel, self)
        return xr.DataArray(
            np.asarray(rows[self.index].values),
            dims=(ROW,),
            coords={ROW: np.asarray(rows.index.values)},
            attrs={"units": unit},
        )

    def _task_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Get the variable or coordinate of the result of the task, see `get_data`."""
        result = experiment.results[str(self.task_id)]
        if not isinstance(result, ScanResult):
            raise ValueError(
                f"The result of the task '{self.task_id}' is no ScanResult: "
                f"{type(result)}."
            )
        values = _dimension_values(experiment, str(self.task_id), result)
        name = self._qualified_name(result, values)
        if name not in result:
            if self.selection == TIME:
                raise KeyError(
                    f"The task '{self.task_id}' keeps no time: its data read only "
                    f"values per simulation; a timecourse of it read in data(), a "
                    f"figure or a fit mapping keeps the time."
                )
            raise KeyError(
                f"'{self.selection}' is not in the result of the task "
                f"'{self.task_id}', its variables are {result.variables}: the "
                f"data a task keeps is the data read in data(), the figures and "
                f"the fit mappings of the experiment."
            )
        array = result[name]
        numeric = array.dtype.kind in "fiub"
        unit = result.units.get(name) if numeric else None
        if unit == "":
            unit = "dimensionless"
        if unit is None and numeric:
            raise ValueError(
                f"'{self.selection}' of the task '{self.task_id}' has no unit in "
                f"the result."
            )
        array = _select(array, self.sel, self, result.ds)
        return xr.DataArray(
            array.values, dims=array.dims, coords=array.coords, attrs={"units": unit}
        ).rename({p: q for p, q in values.items() if p in array.coords})

    def _qualified_name(self, result: ScanResult, values: Mapping[str, str]) -> str:
        """Get the name in the result of the data, resolving `<dimension>.<target>`.

        The values a dimension sets to a target are `<dimension>.<target>`,
        which the result stores under that name when the target is also a
        variable, else under the plain `<target>`; `<dimension>.<coordinate>`
        is a coordinate of the dimension.

        Args:
            result: the result of the task.
            values: the plain names of the values of the dimensions in the
                result and their qualified names, see `_dimension_values`.

        Raises:
            KeyError: if the index is the plain name of a target whose values
                the result has but not its timecourse, or names a dimension
                and a target it does not change.
        """
        index = self.selection
        plain = {qualified: name for name, qualified in values.items()}
        if index in plain:
            return plain[index]
        if index in values:
            raise KeyError(
                f"'{index}' is not in the result of the task '{self.task_id}': "
                f"the values its scan sets are Data('{values[index]}'), and the "
                f"timecourse of '{index}' needs the data to be registered in the "
                f"experiment, read in data(), a figure or a fit mapping."
            )
        if index in result:
            return index
        dimension, _, target = index.partition(".")
        if target and dimension in result.ds.dims:
            if target in result.ds.coords and result[target].dims == (dimension,):
                return target
            changed = [
                values.get(str(t), str(t))
                for t in result.ds.coords
                if t != dimension and result.ds[t].dims == (dimension,)
            ]
            raise KeyError(
                f"'{index}' is not in the result of the task '{self.task_id}': "
                f"the dimension '{dimension}' has the values {sorted(changed)}."
            )
        return index

    def _function_array(self, experiment: SimulationExperiment) -> xr.DataArray:
        """Evaluate the function on its variables and parameters, see `get_data`."""
        if self.function is None:
            raise ValueError(f"Data '{self}' has no function.")
        variables: dict[str, xr.DataArray | float] = {}
        for key, variable in self.variables.items():
            d = experiment._data[variable] if isinstance(variable, str) else variable
            variables[key] = d.get_data(experiment=experiment)
        variables.update(self.parameters)
        array = evaluate_function(self.function, variables, experiment.ureg)
        return _select(array, self.sel, self, array)


#: a character which a generated sid encodes
_NOT_IN_SID = re.compile(r"[^a-zA-Z0-9_]")


def _sid_part(index: str) -> str:
    """Encode an index for a sid: a dot as `__`, any other character as `_x<hex>_`.

    Every character which is no letter, digit or underscore is encoded, so a
    selection of roadrunner (`X'`, `eigenReal(X)`, `X[1]`) gives a valid SId.
    """
    return _NOT_IN_SID.sub(
        lambda m: "__" if m.group() == "." else f"_x{ord(m.group()):x}_", index
    )


def _own_units[T: DataSet | DataSeries](result: T) -> T:
    """Give a new DataSet or DataSeries a copy of its units information."""
    uinfo = getattr(result, "uinfo", None)
    if isinstance(uinfo, UnitsInformation):
        result.uinfo = UnitsInformation(dict(uinfo.udict), ureg=uinfo.ureg)
    return result


def _unify_units(
    df: pd.DataFrame,
    value_key: str,
    unit_key: str,
    ureg: UnitRegistry,
    error_keys: list[str],
) -> str | None:
    """Convert the rows of a column to the unit of its first row with a value.

    The rows without a value or without a unit are ignored (a column without any
    value uses the rows which have a unit). Rows with another unit are converted in
    place, the value, the error columns `error_keys` and the unit column, and the
    unit of the first row is returned (None if no row has a unit).

    :raises ValueError: naming the column and its units, if a unit cannot be read
        (no string, undefined or with a factor) or converted into the first unit by
        a factor, or if a column which is converted has values which are no numbers
    """
    has_unit = df[unit_key].notna()
    present = has_unit & df[value_key].notna()
    # a row which carries a number (a value, an sd or an se) needs its unit, the unit
    # of an empty row is ignored
    numbers = df[value_key].notna()
    for key in error_keys:
        if key in df.columns:
            numbers = numbers | df[key].notna()
    carries = has_unit & numbers
    rows = present if present.any() else (carries if carries.any() else has_unit)
    units = df.loc[rows, unit_key].unique()
    if len(units) == 0:
        return None
    target = units[0]
    # the units of all rows with a number are converted, also of those without a value
    other = df.loc[carries, unit_key].unique()
    units = np.concatenate([[target], other[other != target]])
    if len(units) == 1:
        return str(target)

    # every unit is read once, so an error names the unit which is wrong
    parsed = {unit: _parse_unit(value_key, unit, units, ureg) for unit in units}
    factors: dict[str, float] = {}
    for unit in units[1:]:
        try:
            zero = float(ureg.Quantity(0.0, parsed[unit]).to(parsed[target]).magnitude)
            factors[unit] = float(
                ureg.Quantity(1.0, parsed[unit]).to(parsed[target]).magnitude
            )
        except pint.errors.PintError as err:
            dimensions = {u: str(parsed[u].dimensionality) for u in units}
            raise ValueError(
                f"Column '{value_key}' has the units {list(units)}, which cannot "
                f"be converted into '{target}' (dimensions {dimensions}): {err}"
            ) from err
        if zero != 0.0:
            raise ValueError(
                f"Column '{value_key}' has the units {list(units)}, the unit "
                f"'{unit}' has an offset to '{target}' (e.g. degC) and cannot be "
                "converted by a factor, use one unit for the column"
            )
    logger.info(
        "Column '%s' has the units %s, the rows are converted to '%s'",
        value_key,
        list(units),
        target,
    )
    keys = [key for key in [value_key, *error_keys] if key in df.columns]
    for key in keys:
        try:
            df[key] = df[key].astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Column '{key}' has the units {list(units)}, which are converted "
                f"to '{target}', but values which are no numbers: {err}"
            ) from err
    for unit, factor in factors.items():
        # all rows of the unit which carry a number, so an sd without a value follows
        mask = carries & (df[unit_key] == unit)
        for key in keys:
            df.loc[mask, key] = df.loc[mask, key] * factor
    df.loc[has_unit, unit_key] = target
    return str(target)


def _parse_unit(
    value_key: str, unit: object, units: Iterable[object], ureg: UnitRegistry
) -> PlainUnit:
    """Read a unit of a column.

    Args:
        value_key: the column.
        unit: the unit of some of its rows.
        units: all units of the column, for the message.
        ureg: the unit registry.

    Raises:
        ValueError: naming the column, its units and the unit, if the unit is
            no string or not a unit of the registry.
    """
    if not isinstance(unit, str):
        raise ValueError(
            f"Column '{value_key}' has the units {list(units)}, the unit "
            f"{unit!r} is no string"
        )
    try:
        return ureg.Quantity(1.0, unit).units
    # pint raises errors of many kinds for a unit it cannot read: its own, a
    # ValueError for a factor, a TypeError or the TokenError of its parser
    except Exception as err:
        raise ValueError(
            f"Column '{value_key}' has the units {list(units)}, the unit '{unit}' "
            f"cannot be read: {err}"
        ) from err


class DataSeries(pd.Series):
    """DataSet - a pd.Series with additional unit information."""

    # additional properties
    _metadata = ["uinfo"]  # noqa: RUF012 -- pandas declares it as an instance variable

    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataSet

    def __finalize__(  # ty: ignore[override-of-final-method]
        self, other: object, method: str | None = None, **kwargs: Any
    ) -> Self:
        """Finalize and give the new object its own unit information."""
        # pandas hands the metadata on by reference and only __finalize__ (final in
        # the typing of pandas, not enforced) sees the new object with its metadata;
        # the results of concat and merge have no uinfo at all
        return _own_units(super().__finalize__(other, method=method, **kwargs))


class DataSet(pd.DataFrame):
    """DataSet, a pd.DataFrame with additional unit information."""

    # additional properties
    _metadata = ["uinfo", "Q_"]  # noqa: RUF012 -- pandas declares it as an instance variable

    # a column of a DataSet is a DataSeries; pandas declares the constructor of
    # the columns as an attribute, a property would be read-only
    _constructor_sliced = DataSeries

    @property
    def _constructor(self):
        return DataSet

    def __finalize__(  # ty: ignore[override-of-final-method]
        self, other: object, method: str | None = None, **kwargs: Any
    ) -> Self:
        """Finalize and give the new object its own unit information."""
        # pandas hands the metadata on by reference and only __finalize__ (final in
        # the typing of pandas, not enforced) sees the new object with its metadata;
        # the results of concat and merge have no uinfo at all
        return _own_units(super().__finalize__(other, method=method, **kwargs))

    def get_quantity(self, key: str):
        """Return quantity for given key.

        Requires using the numpy data instead of the series.
        """
        return self.uinfo.ureg.Quantity(
            self[key].values,
            self.uinfo[key],
        )

    def __repr__(self) -> str:
        """Return DataFrame with all columns."""
        pd.set_option("display.max_columns", None)
        s = super().__repr__()
        pd.reset_option("display.max_columns")
        return str(s)

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, ureg: UnitRegistry, udict: dict[str, str] | None = None
    ) -> DataSet:
        """Create DataSet from given pandas.DataFrame.

        The DataFrame can have various formats which should be handled.
        Standard formats are
        1. units annotations based on '*_unit' columns, with additional '*_sd'
           or '*_se' units
        2. units annotations based on 'unit' column which is applied on
           'mean', 'value', 'sd' and 'se' columns

        The unit of a column is the unit of its first row with a value; the unit
        of a row without a number (no value, sd or se) is ignored, and a column
        without any unit has no unit. Rows with another unit of the same
        dimension are converted to it, their values, sd and se, and their unit
        column is rewritten. The data frame of the caller is not changed.

        :param df: pandas.DataFrame
        :param ureg: the unit registry
        :param udict: optional units of columns

        :return: dataset

        :raises ValueError: naming the column and its units, if a unit of a column
            with several units cannot be read (no string, undefined or with a
            factor), cannot be converted by a factor (another dimension or an
            offset such as degC), or if such a column has values which are no
            numbers
        """
        if not isinstance(ureg, UnitRegistry):
            raise ValueError(
                f"ureg must be a UnitRegistry, but '{ureg}' is '{type(ureg)}'"
            )
        if df.empty:
            raise ValueError(f"DataFrame cannot be empty, check DataFrame: {df}")

        # the caller's data frame is not changed
        df = df.copy()

        if udict is None:
            udict = {}

        # all units from udict and DataFrame
        all_udict: dict[str, str] = {}

        for key in df.columns:
            # handle '*_unit columns'
            if key.endswith("_unit"):
                # parse the item and unit in dict; a row without a value has no
                # unit, which is not a unit of the column
                item_key = key[0:-5]
                if item_key not in df.columns:
                    logger.error(
                        "Missing * column '%s' for unit column: '%s'", item_key, key
                    )
                    continue
                unit = _unify_units(
                    df,
                    item_key,
                    key,
                    ureg,
                    [f"{item_key}_sd", f"{item_key}_se"],
                )
                if unit is None:
                    logger.error("Column '%s' units are missing", key)
                else:
                    all_udict[item_key] = unit

            elif key == "unit":
                # add unit to "mean" and "value"
                for key in ["mean", "value", "median"]:
                    if (key in df.columns) and f"{key}_unit" not in df.columns:
                        # FIXME: probably not a good idea to add columns while iterating over them
                        df[f"{key}_unit"] = df.unit
                        error_keys = [f"{key}_sd", f"{key}_se"]
                        if key == "mean":
                            error_keys += ["sd", "se"]
                        unit = _unify_units(df, key, f"{key}_unit", ureg, error_keys)
                        if unit is None:
                            logger.error("Column 'unit' has no unit for '%s'", key)
                        else:
                            udict[key] = unit

                        # rename the sd and se columns to mean_sd and mean_se
                        if key == "mean":
                            for err_key in ["sd", "se"]:
                                if (
                                    err_key not in df.columns
                                    and f"mean_{err_key}" in df.columns
                                ):
                                    df[err_key] = df[f"mean_{err_key}"]

                                if f"mean_{err_key}" in df.columns:
                                    # remove existing mean_sd column
                                    del df[f"mean_{err_key}"]
                                    logger.warning(
                                        "Removing existing column: 'mean_%s' from DataSet. Column should be named: '%s'",
                                        err_key,
                                        err_key,
                                    )

                                df.rename(
                                    columns={f"{err_key}": f"mean_{err_key}"},
                                    inplace=True,
                                )

                # remove unit column
                del df["unit"]

            elif key in ["count", "n"]:
                # add special units for count
                if f"{key}_unit" not in df.columns:
                    udict[key] = "dimensionless"

        # add external definitions
        if udict:
            for key, unit in udict.items():
                if key in all_udict:
                    logger.error("Duplicate unit definition for: '%s'", key)
                else:
                    all_udict[key] = unit
                    # add the unit column of a column of the data frame
                    if key in df.columns:
                        df[f"{key}_unit"] = unit

        dset = DataSet(df)
        dset.uinfo = UnitsInformation(all_udict, ureg=ureg)
        dset.Q_ = dset.uinfo.ureg.Quantity
        return dset

    def unit_conversion(self, key, factor: Quantity) -> None:
        """Convert the units of the given key in the dataset via `key * factor`.

        Changes values in place in the DataSet.

        The quantity in the dataset is multiplied with the conversion factor.
        In addition to the key, also the respective error measures are
        converted with the same factor, i.e.
        - {key}
        - {key}_sd
        - {key}_se
        - {key}_min
        - {key}_max

        FIXME: in addition base keys should be updated in the table,
        i.e. if key in [mean, median, min, max, sd, se, cv] then the other
        keys should be updated;
        use default set of keys for automatic conversion

        :param key: column key in dataset (this column is unit converted)
        :param factor: multiplicative Quantity factor for conversion
        :return: None
        """
        if key in self.columns:
            if key not in self.uinfo:
                raise ValueError(
                    f"Unit conversion only possible on keys which have units! "
                    f"No unit defined for key '{key}'"
                )

            # unit conversion and simplification
            new_quantity = self.uinfo.Q_(self[key], self.uinfo[key]) * factor
            new_quantity = new_quantity.to_base_units().to_reduced_units()

            # updated values
            self[key] = new_quantity.magnitude

            # update error measures
            for err_key in [f"{key}_sd", f"{key}_se", f"{key}_min", f"{key}_max"]:
                if err_key in self.columns:
                    # error keys not stored in udict, only the base quantity
                    new_err_quantity = (
                        self.uinfo.Q_(self[err_key], self.uinfo[key]) * factor
                    )
                    new_err_quantity = (
                        new_err_quantity.to_base_units().to_reduced_units()
                    )
                    self[err_key] = new_err_quantity.magnitude

            # updated units
            new_units = new_quantity.units
            new_units_str = (
                str(new_units).replace("**", "^").replace(" ", "")
            )  # '{:~}'.format(new_units)
            self.uinfo[key] = new_units_str

            if f"{key}_unit" in self.columns:
                self[f"{key}_unit"] = new_units_str
        else:
            logger.error(
                "Key '%s' not in DataSet, unit conversion not applied: '%s'",
                key,
                factor,
            )


def load_pkdb_dataframe(
    sid, data_path: Path | list[Path], sep="\t", comment="#", **kwargs
) -> pd.DataFrame:
    """Load TSV data from PKDB figure or table id.

    This is a simple helper functions to directly loading the TSV data.
    It is recommended to use `pkdb_analysis` methods instead.

    This function will be removed.

    E.g. for 'Amchin1999_Tab1' the file
        data_path / 'Amchin1999' / '.Amchin1999.tsv'
    is loaded.

    :param sid: figure or table id
    :param data_path: base path of data or iterable of data_paths
    :param sep: separator
    :param comment: comment characters
    :param kwargs: additional kwargs for csv parsing
    :return: pandas DataFrame
    :raises FileNotFoundError: if the dataset is in none of the data paths
    """
    study = sid.split("_")[0]
    if isinstance(data_path, Path):
        data_path = [data_path]

    # use the first path which exists
    paths = [p / study / f".{sid}.tsv" for p in data_path]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Dataset '{sid}' not found, none of the files exists: "
            f"{', '.join(str(p) for p in paths)}"
        )

    try:
        df = pd.read_csv(path, sep=sep, comment=comment, **kwargs)
    except pd.errors.ParserError as err:
        logger.error("Could not read DataFrame for '%s' at '%s'.", sid, path)
        raise err

    # FIXME: handle unnecessary UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
    # At this point we only work with numpy arrays, units not important here
    return df.dropna(how="all")  # drop all NA rows


def load_pkdb_dataframes_by_substance(
    sid, data_path, **kwargs
) -> dict[str, pd.DataFrame]:
    """Load dataframes from given PKDB figure/table id split on substance.

    The DataFrame is split on the 'substance' key.

    This is a simple helper functions to directly loading the TSV data.
    It is recommended to use `pkdb_analysis` methods instead.

    This function will be removed.

    :param sid:
    :param data_path:
    :param kwargs:
    :return: dict[substance, pd.DataFrame]
    """
    df = load_pkdb_dataframe(sid=sid, data_path=data_path, na_values=["na"], **kwargs)
    frames = {}
    for substance in df.substance.unique():
        frames[substance] = df.copy()[df.substance == substance]
    return frames
