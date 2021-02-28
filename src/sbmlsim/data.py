"""Module handling data (experiment and simulation)."""
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sbmlsim.combine import mathml
from sbmlsim.result import XResult
from sbmlsim.units import DimensionalityError, Quantity, UnitRegistry
from sbmlsim.utils import deprecated


logger = logging.getLogger(__name__)


class Data(object):
    """Data.

    Main data generator class which uses data either from
    experimental data, simulations or via function calculations.

    All transformation of data and a tree of data operations.
    """

    class Types(Enum):
        """Data types."""

        TASK = 1
        DATASET = 2
        FUNCTION = 3

    def __init__(
        self,
        experiment,
        index: str,
        task: str = None,
        dataset: str = None,
        function=None,
        variables=None,
    ):
        self.experiment = experiment
        self.index = index
        self.task_id = task
        self.dset_id = dataset
        self.function = function
        self.variables = variables
        self.unit = None

        if (not self.task_id) and (not self.dset_id) and (not self.function):
            raise ValueError(
                "Either 'task_id', 'dset_id' or 'function' required for Data."
            )

        # register data in simulation
        if experiment._data is None:
            experiment._data = {}

        experiment._data[self.sid] = self

    def __str__(self) -> str:
        """Get string."""
        if self.is_task():
            s = f"Data(index={self.index}, task_id={self.task_id})|Task"
        elif self.is_dataset():
            s = f"Data(index={self.index}, dset_id={self.dset_id})|DataSet"
        elif self.is_function():
            s = f"Data(index={self.index}, function={self.function})|Function"
        return s

    @property
    def sid(self) -> str:
        """Get id."""
        if self.task_id:
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
    def dtype(self) -> str:
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

        d = {
            "type": self.dtype,
            "index": self.index,
            "unit": self.unit,
            "task": self.task_id,
            "dataset": self.dset_id,
            "function": self.function,
            "variables": self.variables if self.variables else None,
        }
        return d

    def get_data(self, to_units: str = None):
        """Return actual data from the data object.

        :param to_units: units to convert to
        :return:
        """
        # Necessary to resolve the data
        if self.dtype == Data.Types.DATASET:
            # read dataset data
            dset = self.experiment._datasets[self.dset_id]
            if not isinstance(dset, DataSet):
                raise ValueError(
                    f"DataSet '{self.dset_id}' is not a DataSet, but "
                    f"type '{type(dset)}'\n"
                    f"{dset}"
                )
            if dset.empty:
                logger.error(f"Adding empty dataset '{dset}' for '{self.dset_id}'.")

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
                self.unit = dset.udict[uindex]
            except KeyError as err:
                logger.error(
                    f"Units missing for key '{uindex}' in dataset: "
                    f"'{self.dset_id}'. Add missing units to dataset."
                )
                raise err
            x = dset[self.index].values * dset.ureg(dset.udict[uindex])

        elif self.dtype == Data.Types.TASK:
            # read results of task
            xres = self.experiment.results[self.task_id]  # type: XResult
            if not isinstance(xres, XResult):
                raise ValueError("Only Result objects supported in task data.")

            self.unit = xres.udict[self.index]
            # FIXME: complete data must be kept
            # print(xres)
            x = xres.dim_mean(self.index)
            # x = xres[self.index]

        elif self.dtype == Data.Types.FUNCTION:
            # evaluate with actual data
            astnode = mathml.formula_to_astnode(self.function)
            variables = {}
            for k, v in self.variables.items():
                # lookup via key
                if isinstance(v, str):
                    variables[k] = self.experiment._data[v].data
                elif isinstance(v, Data):
                    variables[k] = v.data

            x = mathml.evaluate(astnode=astnode, variables=variables)
            self.unit = str(x.units)  # check if this is correct

        # convert units to requested units
        if to_units is not None:
            try:
                x = x.to(to_units)
            except DimensionalityError as err:
                logger.error(
                    f"Could not convert '{str(self)}' with "
                    f"data '{x} ({type(x)})' to "
                    f"units '{to_units}'"
                )
                raise err
            except AttributeError as err:
                logger.error(
                    f"Could not convert '{str(self)}' with "
                    f"data '{x} ({type(x)})' to "
                    f"units '{to_units}'"
                )
                raise err

        return x

    data = property(get_data)


class DataFunction(object):
    """Functional data calculation.

    The idea ist to provide an object which can calculate a generic math function
    based on given input symbols.

    Important challenge is to handle the correct functional evaluation.
    """

    def __init__(self, index, formula, variables):
        self.index = index
        self.formula = formula
        self.variables = variables


class DataSeries(pd.Series):
    """DataSet - a pd.Series with additional unit information."""

    # additional properties
    _metadata = ["udict", "ureg"]

    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataSet


class DataSet(pd.DataFrame):
    """DataSet.

     pd.DataFrame with additional unit information in the form
    of a unit dictionary 'udict' (Dict[str, str]) mapping column
    keys to units. The UnitRegistry is the UnitRegistry conversions
    are calculated on.
    """

    # additional properties
    _metadata = ["udict", "ureg"]

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
        return self.ureg.Quantity(
            # downcasting !
            self[key].values,
            self.udict[key],
        )

    def __repr__(self) -> str:
        """Return DataFrame with all columns."""
        pd.set_option("max_columns", None)
        s = super().__repr__()
        pd.reset_option("max_columns")
        return str(s)

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, ureg: UnitRegistry, udict: Dict[str, str] = None
    ) -> "DataSet":
        """Create DataSet from given pandas.DataFrame.

        The DataFrame can have various formats which should be handled.
        Standard formats are
        1. units annotations based on '*_unit' columns, with additional '*_sd'
           or '*_se' units
        2. units annotations based on 'unit' column which is applied on
           'mean', 'value', 'sd' and 'se' columns

        :param df: pandas.DataFrame
        :param udict: optional unit dictionary
        :param ureg:
        :return:
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
        all_udict = {}

        for key in df.columns:
            # handle '*_unit columns'
            if key.endswith("_unit"):
                # parse the item and unit in dict
                units = df[key].unique()
                if len(units) > 1:
                    logger.error(
                        f"Column '{key}' units are not unique: '{units}' in \n" f"{df}"
                    )
                elif len(units) == 0:
                    logger.error(f"Column '{key}' units are missing: '{units}'")
                    print(df.head())
                item_key = key[0:-5]
                if item_key not in df.columns:
                    logger.error(
                        f"Missing * column '{item_key}' for unit " f"column: '{key}'"
                    )
                else:
                    all_udict[item_key] = units[0]

            elif key == "unit":
                # add unit to "mean" and "value"
                for key in ["mean", "value", "median"]:
                    if (key in df.columns) and not (f"{key}_unit" in df.columns):
                        # FIXME: probably not a good idea to add columns while iterating over them
                        df[f"{key}_unit"] = df.unit
                        unit_keys = df.unit.unique()
                        if len(df.unit.unique()) > 1:
                            logger.error(
                                f"More than one unit in 'unit' column will create issues in "
                                f"unit conversion, filter data to reduce units: '{df.unit.unique()}'"
                            )
                        udict[key] = unit_keys[0]

                        # rename the sd and se columns to mean_sd and mean_se
                        if key == "mean":
                            for err_key in ["sd", "se"]:
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
                    logger.error(f"Duplicate unit definition for: '{key}'")
                else:
                    all_udict[key] = unit
                    # add the unit columns to the data frame
                    setattr(df, f"{key}_unit", unit)

        dset = DataSet(df)
        dset.udict = all_udict
        dset.ureg = ureg
        return dset

    def unit_conversion(self, key, factor: Quantity, filter=None):
        """Convert the units of the given key in the dataset.

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
            if key not in self.udict:
                raise ValueError(
                    f"Unit conversion only possible on keys which have units! "
                    f"No unit defined for key '{key}'"
                )

            # unit conversion and simplification
            new_quantity = self.ureg.Quantity(self[key], self.udict[key]) * factor
            new_quantity = new_quantity.to_base_units().to_reduced_units()

            # updated values
            self[key] = new_quantity.magnitude

            # update error measures
            for err_key in [f"{key}_sd", f"{key}_se", f"{key}_min", f"{key}_max"]:
                if err_key in self.columns:
                    # error keys not stored in udict, only the base quantity
                    new_err_quantity = (
                        self.ureg.Quantity(self[err_key], self.udict[key]) * factor
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
            self.udict[key] = new_units_str

            if f"{key}_unit" in self.columns:
                self[f"{key}_unit"] = new_units_str
        else:
            logger.error(
                f"Key '{key}' not in DataSet, unit conversion not applied: '{factor}'"
            )


# @deprecated
def load_pkdb_dataframe(
    sid, data_path: [Path, List[Path]], sep="\t", comment="#", **kwargs
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
    """
    study = sid.split("_")[0]
    if isinstance(data_path, Path):
        data_path = [data_path]

    for p in data_path:
        path = p / study / f".{sid}.tsv"
        if path.exists():
            # use the first path which exists
            break
    if not path.exists():
        ValueError(f"file path not found in data_path: {data_path}")

    df = pd.read_csv(path, sep=sep, comment=comment, **kwargs)
    # FIXME: handle unnecessary UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
    # At this point we only work with numpy arrays, units not important here
    df = df.dropna(how="all")  # drop all NA rows
    return df


# @deprecated
def load_pkdb_dataframes_by_substance(
    sid, data_path, **kwargs
) -> Dict[str, pd.DataFrame]:
    """Load dataframes from given PKDB figure/table id split on substance.

    The DataFrame is split on the 'substance' key.

    This is a simple helper functions to directly loading the TSV data.
    It is recommended to use `pkdb_analysis` methods instead.

    This function will be removed.

    :param sid:
    :param data_path:
    :param kwargs:
    :return: Dict[substance, pd.DataFrame]
    """
    df = load_pkdb_dataframe(sid=sid, data_path=data_path, na_values=["na"], **kwargs)
    frames = {}
    for substance in df.substance.unique():
        frames[substance] = df.copy()[df.substance == substance]
    return frames
