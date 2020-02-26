import pandas as pd
from enum import Enum
import logging

# from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)


class Data(object):
    """Main data generator class which uses data either from
    experimental data or simulations.

    # Possible Data:
    # simulation result: access via id
    # Dataset access via id

    # FIXME: must also handle all the unit conversions
    """
    class Types(Enum):
        SIMULATION = 1
        DATASET = 2
        FUNCTION = 3

    def __init__(self, experiment,  # FIXME: annotate SimulationExperiment
                 index: str, unit: str=None,
                 simulation: str=None,
                 dataset: str=None,
                 function=None, data=None):
        self.experiment = experiment
        self.index = index
        self.unit = unit
        self.simulation = simulation
        self.dataset = dataset
        self.function = function
        self._data = data

    def get_type(self):
        if self.simulation:
            dtype = Data.Types.SIMULATION
        elif self.dataset:
            dtype = Data.Types.DATASET
        return dtype

    # todo: dimensions, data type
    # TODO: calculations

    @property
    def data(self):
        """Returns actual data from the data object"""
        # FIXME: data caching

        # Necessary to resolve the data
        dtype = self.get_type()
        if dtype == Data.Types.DATASET:
            # read dataset data
            data = self.experiment.datasets[self.dataset]
        elif dtype == Data.Types.SIMULATION:
            # read simulation data
            data = self.experiment.simulations[self.simulation]

        # Make a dataset with units out of the data
        if isinstance(data, DataSet):
            dset = data
        elif isinstance(data, pd.DataFrame):
            dset = DataSet.from_df(data=data, udict=None, ureg=None)

        if dset.empty:
            logger.error(f"Empty dataset in adding data: {dset}")

        # data with units
        x = dset[self.index].values * dset.ureg(dset.udict[self.index])

        # convert units
        if self.unit:
            x = x.to(self.unit)

        return x


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
    """DataSet - a pd.DataFrame with additional unit information.
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
    def from_df(cls, data: pd.DataFrame, udict: dict, ureg):
        dset = DataSet(data)
        dset.udict = udict
        dset.ureg = ureg
        return dset


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
