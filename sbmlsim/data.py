import pandas as pd
from enum import Enum
import logging

from sbmlsim.result import Result
from sbmlsim.utils import deprecated

from pint import Quantity

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
                 function=None, data=None):
        self.experiment = experiment
        self.index = index
        self.unit = unit
        self.task_id = task
        self.dset_id = dataset
        self.function = function
        # self._data = data

    @property
    def dtype(self):
        if self.task_id:
            dtype = Data.Types.TASK
        elif self.dset_id:
            dtype = Data.Types.DATASET
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
            result = self.experiment.results[self.task_id]  # type: Result
            if not isinstance(result, Result):
                raise ValueError("Only Result objects supported in task data.")
            x = result.mean[self.index].values * result.ureg(result.udict[self.index])

        # convert units
        if self.unit:
            x = x.to(self.unit)

        return x


class DataFunction(object):
    """TODO: Data based on functions, i.e. data based on data.

    These are the more complicated data generators.
    1. calculatable from existing data,
    2. data can be directly serialized
    """
    pass


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
    DataSet - a pd.DataFrame with additional unit information.
    """
    # additional properties
    _metadata = ['udict', 'ureg']

    @property
    def _constructor(self):
        return DataSet

    @property
    def _constructor_sliced(self):
        return DataSeries

    @classmethod
    def from_df(cls, data: pd.DataFrame, udict: dict, ureg) -> 'DataSet':
        """DataSet from pandas.DataFrame"""
        if udict is None:
            udict = {}

        for key, unit in udict.items():
            # add the unit columns to the data frame
            setattr(data, f"{key}_unit", unit)

        # handle special unit column
        if "unit" in data.columns:
            
            # add unit to "mean", "value", "sd", "se" columns
            for key in ["mean", "value", "sd", "se"]:
                if not f"{key}_unit" in data.columns:
                    setattr(data, f"{key}_unit", data.unit)
                    unit_keys = data.unit.unique()
                    if len(data.unit.unique()) > 1:
                        logger.error("More than 1 unit in 'unit' column !")
                    udict[key] = unit_keys[0]
                    
        dset = DataSet(data)
        dset.udict = udict
        dset.ureg = ureg
        return dset



    def unit_conversion(self, key, factor: Quantity):
        """Also converts the corresponding errors"""
        if key in self.columns:
            self[key] = (self[key] * factor)
            new_units = (self.get_quantity(key) * factor).to_base_units().to_reduced_units().units
            new_units_str = str(new_units).replace("**", "^").replace(" ", "")  # '{:~}'.format(new_units)
            self.udict[key] = new_units_str
            for err_key in [f"{key}_sd", f"{key}_se"]:
                if err_key in self.columns:
                    self[err_key] = self[err_key] * factor
                    # error keys not stored in udict
            # if unit is stored in tsv these must be updated
            if f"{key}_unit" in self.columns:
                self[f"{key}_unit"] = new_units_str

            # special handling of unspecific mean or value
            # FIXME: make this work in general
            # FIXME: add validation rules on PKDB that no unit_* columns
            if key in {"mean"}:
                if "unit" in self.columns:
                    self["unit"] = new_units_str
                for key_additional in {'sd', 'se', 'value'}:
                    if key_additional in self.columns:
                        self[key_additional] = (self[key_additional] * factor)
                    self.udict[key_additional] = new_units_str



        else:
            logger.warning(f"Key '{key}' not in DataSet: '{id(self)}'")

    def get_quantity(self, key):
        """Returns quantity for given key.

        Requires using the numpy data instead of the series.
        """
        return self.ureg.Quantity(
            # downcasting !
            self[key].values, self.udict[key]
        )


def load_dataframe(sid, data_path, sep="\t", comment="#", **kwargs) -> pd.DataFrame:
    """ Loads data from given figure/table id."""
    study = sid.split('_')[0]
    path = data_path / study / f'{sid}.tsv'

    if not path.exists():
        path = data_path / study / f'.{sid}.tsv'

    return pd.read_csv(path, sep=sep, comment=comment, **kwargs)





if __name__ == "__main__":
    df = pd.DataFrame({'col1': [1, 2, 3],
                       'col2': [2, 3, 4],
                       "col3": [4, 5, 6]})
    print(df)
    dset = DataSet.from_df(df, udict={"col1": "mM"}, ureg="test")
    print(dset)
    print(dset.udict)
    dset2 = dset[dset.col1 > 1]
    print(dset2)
    print(dset2.udict)
