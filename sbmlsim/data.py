
from enum import Enum
import logging
from typing import Dict
import pandas as pd
import numpy as np
from sbmlsim.units import Quantity, UnitRegistry
from sbmlsim.combine import mathml
from sbmlsim.result import XResult

logger = logging.getLogger(__name__)


class Data(object):
    """Main data generator class which uses data either from
    experimental data, simulations or via function calculations.

    Alls transformation of data and a tree of data operations.

    # Possible Data:
    # simulation result: access via id
    # Dataset access via id

    # FIXME: must also handle all the unit conversions
    """
    class Types(Enum):
        TASK = 1
        DATASET = 2
        FUNCTION = 3

    def __init__(self, experiment,
                 index: str, unit: str=None,
                 task: str = None,
                 dataset: str = None,
                 function=None, variables=None):
        self.experiment = experiment
        self.index = index
        self.unit = unit
        self.task_id = task
        self.dset_id = dataset
        self.function = function
        self.variables = variables

        if (not self.task_id) and (not self.dset_id) and (not self.function):
            raise ValueError(f"Either 'task_id', 'dset_id' or 'function' "
                             f"required for Data.")

        # register data in simulation
        if experiment._data is None:
            experiment._data = {}

        experiment._data[self.sid] = self

    @property
    def sid(self):
        if self.task_id:
            sid = f"{self.task_id}__{self.index}"
        elif self.dset_id:
            sid = f"{self.dset_id}__{self.index}"
        elif self.function:
            sid = self.index

        return sid

    def is_task(self):
        return self.task_id is not None

    def is_dataset(self):
        return self.dset_id is not None

    def is_function(self):
        return self.function is not None

    @property
    def dtype(self):
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
        """ Convert to dictionary. """
        d = {
            "type": self.dtype,
            "index": self.index,
            "unit": self.unit,
            "task": self.task_id,
            "dataset": self.dset_id,
            "function": self.function,
            "variables": self.variables if self.variables else None
        }
        return d

    @property
    def data(self):
        """Returns actual data from the data object"""
        # FIXME: data caching & store conversion factors

        # Necessary to resolve the data

        if self.dtype == Data.Types.DATASET:
            # read dataset data
            dset = self.experiment._datasets[self.dset_id]
            if not isinstance(dset, DataSet):
                raise ValueError(dset)
            if dset.empty:
                logger.error(f"Empty dataset in adding data: {dset}")

            # data with units
            if self.index.endswith("_se") or self.index.endswith("_sd"):
                uindex = self.index[:-3]
            else:
                uindex = self.index
            x = dset[self.index].values * dset.ureg(dset.udict[uindex])

        elif self.dtype == Data.Types.TASK:
            # read results of task
            xres = self.experiment.results[self.task_id]  # type: XResult
            if not isinstance(xres, XResult):
                raise ValueError("Only Result objects supported in task data.")

            # FIXME: complete data must be kept
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

        # convert units
        if self.unit:
            x = x.to(self.unit)

        return x


class DataFunction(object):
    """ Functional data calculation.

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
    _metadata = ['udict', 'ureg']

    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataSet


class DataSet(pd.DataFrame):
    """
    DataSet - a pd.DataFrame with additional unit information in the form
              of a unit dictionary 'udict' (Dict[str, str]) mapping column
              keys to units. The UnitRegistry is the UnitRegistry conversions
              are calculated on.
    """
    # additional properties
    _metadata = ['udict', 'ureg']

    @property
    def _constructor(self):
        return DataSet

    @property
    def _constructor_sliced(self):
        return DataSeries

    def get_quantity(self, key):
        """Returns quantity for given key.

        Requires using the numpy data instead of the series.
        """
        return self.ureg.Quantity(
            # downcasting !
            self[key].values, self.udict[key]
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame, ureg: UnitRegistry, udict: Dict[str, str]=None) -> 'DataSet':
        """Creates DataSet from given pandas.DataFrame.

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
            raise ValueError(f"ureg must be a UnitRegistry, but '{ureg}' is '{type(ureg)}'")
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
                    logger.error(f"Column '{key}' units are not unique: "
                                 f"'{units}'")
                item_key = key[0:-5]
                if item_key not in df.columns:
                    logger.error(f"Missing * column '{item_key}' for unit "
                                 f"column: '{key}'")
                else:
                    all_udict[item_key] = units[0]

            elif key == "unit":
                # add unit to "mean" and "value"
                for key in ["mean", "value"]:
                    if (key in df.columns) and not (f"{key}_unit" in df.columns):
                        # FIXME: probably not a good idea to add columns while iterating over them
                        df[f"{key}_unit"] = df.unit
                        unit_keys = df.unit.unique()
                        if len(df.unit.unique()) > 1:
                            logger.error("More than 1 unit in 'unit' column !")
                        udict[key] = unit_keys[0]

                        # rename the sd and se columns to mean_sd and mean_se
                        if key == 'mean':
                            for err_key in ['sd', 'se']:
                                df.rename(columns={f'{err_key}': f'mean_{err_key}'}, inplace=True)

                # remove unit column
                del df['unit']

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

    def unit_conversion(self, key, factor: Quantity):
        """Also converts the corresponding errors"""
        if key in self.columns:
            if key not in self.udict:
                raise ValueError(
                    f"Unit conversion only possible on keys which have units! "
                    f"No unit defined for key '{key}'")

            # unit conversion and simplification
            new_quantity = self.ureg.Quantity(self[key], self.udict[key]) * factor
            new_quantity = new_quantity.to_base_units().to_reduced_units()

            # updated values
            self[key] = new_quantity.magnitude

            # update error measures
            for err_key in [f"{key}_sd", f"{key}_se"]:
                if err_key in self.columns:
                    # error keys not stored in udict, only the base quantity
                    new_err_quantity = self.ureg.Quantity(self[err_key], self.udict[key]) * factor
                    new_err_quantity = new_err_quantity.to_base_units().to_reduced_units()
                    self[err_key] = new_err_quantity.magnitude

            # updated units
            new_units = new_quantity.units
            new_units_str = str(new_units).replace("**", "^").replace(" ", "")  # '{:~}'.format(new_units)
            self.udict[key] = new_units_str

            if f"{key}_unit" in self.columns:
                self[f"{key}_unit"] = new_units_str
        else:
            logger.error(f"Key '{key}' not in DataSet, unit conversion not applied: '{factor}'")


def load_pkdb_dataframe(sid, data_path, sep="\t", comment="#", **kwargs) -> pd.DataFrame:
    """ Loads data from given pkdb figure/table id.

    :param sid:
    :param data_path:
    :param sep: separator
    :param comment: comment characters
    :param kwargs: additional kwargs for csv parsing
    :return: pandas DataFrame
    """
    study = sid.split('_')[0]
    path = data_path / study / f'{sid}.tsv'

    if not path.exists():
        path = data_path / study / f'.{sid}.tsv'

    return pd.read_csv(path, sep=sep, comment=comment, **kwargs)


def load_pkdb_dataframes_by_substance(sid, data_path, **kwargs) -> Dict[str, pd.DataFrame]:
    """ Load dataframes from given pkdb figure/table id split on substance.

    :param sid:
    :param data_path:
    :param kwargs:
    :return: Dict[substance, pd.DataFrame]
    """
    df = load_pkdb_dataframe(sid=sid, data_path=data_path, na_values=['na'], **kwargs)
    frames = {}
    for substance in df.substance.unique():
        frames[substance] = df[df.substance == substance]
    return frames


if __name__ == "__main__":
    f1 = DataFunction(
        index="test", formula="(x + y + z)/x",
        variables={
         'x': 0.1 * np.ones(shape=[1, 10]),
         'y': 3.0 * np.ones(shape=[1, 10]),
         'z': 2.0 * np.ones(shape=[1, 10]),
        })
    res = f1.data()
    print(res)