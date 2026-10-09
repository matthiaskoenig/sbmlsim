"""Module handling data (experiment and simulation)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sbmlsim.result import ScanResult
from sbmlsim.simulator.formula import compile_formula, evaluate_reduced, reduce_formula
from sbmlsim.units import (
    DimensionalityError,
    Quantity,
    UnitRegistry,
    UnitsInformation,
)

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)


def evaluate_function(formula: str, variables: Mapping[str, Any]) -> Any:
    """Evaluate the formula of a `Data` of type FUNCTION on its data.

    The formula is the math of PEtab, see `sbmlsim.simulator.formula`, with
    the reductions of `evaluate_reduced`: `max` and `min` of a single argument
    reduce it along its last axis, the time of a simulation, and ignore `NaN`,
    the padding of a scan, so `Y/max(Y)` normalizes every simulation of a scan
    to its own maximum. With two or more arguments they are the elementwise
    maximum and minimum of PEtab. `mean` and `at` need the time points of a
    simulation, which data has not; they are reductions of the observables of
    a scan. A formula whose identifiers are all reductions or numbers is a
    value per simulation, a number for a single simulation.

    Args:
        formula: the formula.
        variables: the values of the identifiers of the formula, the arrays
            or quantities of the data and the numbers of the parameters.

    Returns:
        The value of the formula, a quantity if the variables are quantities.

    Raises:
        ValueError: if the formula is not valid math, reads an identifier
            which is not a variable, or uses `mean` or `at` on an array.
    """
    value = evaluate_reduced(formula, variables)
    reduced = reduce_formula(formula)
    per_simulation = bool(reduced.reductions) and all(
        symbol in reduced.placeholders
        or np.ndim(getattr(variables[symbol], "magnitude", variables[symbol])) == 0
        for symbol in compile_formula(reduced.outer).symbols
    )
    if per_simulation and np.ndim(getattr(value, "magnitude", value)) > 0:
        return value[..., 0]
    return value


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
            sid: id of the data, `<task or dataset>__<index>` if not given.

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
            sid = self._sid
        elif self.task_id:
            sid = f"{self.task_id}__{self.index}"
        elif self.dset_id:
            sid = f"{self.dset_id}__{self.index}"
        elif self.function:
            sid = self.index

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

    # todo: dimensions, data type
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
            "unit": self.unit,
            "task": self.task_id,
            "dataset": self.dset_id,
            "function": self.function,
            "variables": self.variables if self.variables else None,
        }

    def get_data(
        self,
        experiment: SimulationExperiment,
        to_units: str | None = None,
    ) -> Quantity:
        """Get the values of the data from an experiment which ran.

        A dataset gives its column, a task the variable or coordinate of its
        `ScanResult` and a function its formula evaluated on its variables
        and parameters; `unit` is set to the unit of the values. The data of a
        task is in the layout of its result, `(*dims, time)` on a common grid
        of times or `(*dims, _point)` for a ragged result, whose simulations
        keep their own time points padded with `NaN`; a coordinate, e.g. a
        changed target, is over its dimension. The values are quantities of
        the unit registry of the experiment.

        Args:
            experiment: the experiment whose datasets and results are read.
            to_units: the unit to convert the values to, their own unit
                without.

        Returns:
            The values with their unit.

        Raises:
            KeyError: if the dataset has no column of the index or no unit of
                it, or the result of the task has no variable or coordinate of
                the selection.
            ValueError: if the dataset is no `DataSet`, the result of the task
                is no `ScanResult` or its selection holds labels or has no
                unit, or a function has no formula.
            DimensionalityError: if the values cannot be converted to
                `to_units`.
        """
        # the type of the data is the first of task, dataset and function
        if self.dtype == Data.Types.DATASET and self.dset_id is not None:
            # read dataset data
            if not experiment._datasets:
                experiment._datasets = experiment.datasets()
            dset = experiment._datasets[self.dset_id]
            if not isinstance(dset, DataSet):
                raise ValueError(
                    f"DataSet '{self.dset_id}' is not a DataSet, but "
                    f"type '{type(dset)}'\n"
                    f"{dset}"
                )
            if dset.empty:
                logger.error("Adding empty dataset '%s' for '%s'.", dset, self.dset_id)

            # data with units
            if self.index.endswith("_se") or self.index.endswith("_sd"):
                uindex = self.index[:-3]
            else:
                uindex = self.index

            if self.index not in dset.columns:
                error_msg = (
                    f"Data column with key '{self.index}' does not "
                    f"exist in dataset: '{self.dset_id}'."
                )
                logger.error(error_msg)
                raise KeyError(error_msg)
            try:
                self.unit = dset.uinfo[uindex]
            except KeyError as err:
                logger.error(
                    "Units missing for key '%s' in dataset: '%s'. Add missing "
                    "units to dataset.",
                    uindex,
                    self.dset_id,
                )
                raise err
            x = dset[self.index].values * dset.uinfo.ureg(dset.uinfo[uindex])

        elif self.dtype == Data.Types.TASK and self.task_id is not None:
            result = experiment.results[self.task_id]
            if not isinstance(result, ScanResult):
                raise ValueError(
                    f"The result of the task '{self.task_id}' is no ScanResult: "
                    f"{type(result)}."
                )
            if self.selection not in result:
                raise KeyError(
                    f"'{self.selection}' is not in the result of the task "
                    f"'{self.task_id}', its variables are {result.variables}: add "
                    f"it to the selections of the experiment."
                )
            # the values in the layout of the result, the time last
            x = result.quantity(self.selection, ureg=experiment.ureg)
            self.unit = result.units[self.selection]

        elif self.dtype == Data.Types.FUNCTION:
            # evaluate with actual data
            if self.function is None:
                raise ValueError(f"Data '{self}' has no function.")
            variables = {}
            for var_key, variable in self.variables.items():
                # lookup via key
                if isinstance(variable, str):
                    variables[var_key] = experiment._data[variable].get_data(
                        experiment=experiment
                    )
                elif isinstance(variable, Data):
                    variables[var_key] = variable.get_data(experiment=experiment)
            for par_key, par_value in self.parameters.items():
                variables[par_key] = par_value

            x = evaluate_function(self.function, variables)
            if not isinstance(x, Quantity):
                # a formula of plain numbers evaluates to a number, e.g. a
                # function of parameters alone; it is dimensionless
                x = experiment.ureg.Quantity(x, "dimensionless")
            self.unit = str(x.units)

        # convert units to requested units
        if to_units is not None:
            try:
                x = x.to(to_units)
            except DimensionalityError as err:
                logger.error(
                    "Could not convert '%s' to units '%s' with data \n'%s'",
                    self,
                    to_units,
                    x,
                )
                raise err
            except AttributeError as err:
                logger.error(
                    "Could not convert '%s' with data '%s (%s)' to units '%s'",
                    self,
                    x,
                    type(x),
                    to_units,
                )
                raise err

        return x


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


class DataSet(pd.DataFrame):
    """DataSet, a pd.DataFrame with additional unit information."""

    # additional properties
    _metadata = ["uinfo", "Q_"]  # noqa: RUF012 -- pandas declares it as an instance variable

    @property
    def _constructor(self):
        return DataSet

    @property
    def _constructor_sliced(self):
        return DataSeries

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

        :param df: pandas.DataFrame
        :param uinfo: optional units information

        :return: dataset
        """
        if not isinstance(ureg, UnitRegistry):
            raise ValueError(
                f"ureg must be a UnitRegistry, but '{ureg}' is '{type(ureg)}'"
            )
        if df.empty:
            raise ValueError(f"DataFrame cannot be empty, check DataFrame: {df}")

        if udict is None:
            udict = {}

        # all units from udict and DataFrame
        all_udict: dict[str, str] = {}

        for key in df.columns:
            # handle '*_unit columns'
            if key.endswith("_unit"):
                # parse the item and unit in dict
                units = df[key].unique()
                if len(units) > 1:
                    logger.error(
                        "Column '%s' units are not unique: '%s' in \n%s", key, units, df
                    )
                elif len(units) == 0:
                    logger.error("Column '%s' units are missing: '%s'", key, units)
                item_key = key[0:-5]
                if item_key not in df.columns:
                    logger.error(
                        "Missing * column '%s' for unit column: '%s'", item_key, key
                    )
                else:
                    all_udict[item_key] = units[0]

            elif key == "unit":
                # add unit to "mean" and "value"
                for key in ["mean", "value", "median"]:
                    if (key in df.columns) and f"{key}_unit" not in df.columns:
                        # FIXME: probably not a good idea to add columns while iterating over them
                        df[f"{key}_unit"] = df.unit
                        unit_keys = df.unit.unique()
                        if len(df.unit.unique()) > 1:
                            logger.error(
                                "More than one unit in 'unit' column will create issues in unit conversion, filter data to reduce units: '%s'",
                                df.unit.unique(),
                            )
                        udict[key] = unit_keys[0]

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
