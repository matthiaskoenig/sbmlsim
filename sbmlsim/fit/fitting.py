from typing import List, Dict, Iterable, Set

import numpy as np
from sbmlsim.data import Data


class FitExperiment(object):
    """
    A Simulation Experiment used in a fitting.
    """

    def __init__(self, experiment,
                 weight: float = 1.0, mappings: Set[str] = None,
                 fit_parameters: List['FitParameter'] = None):
        """A Simulation experiment used in a fitting.

        :param experiment:
        :param weight: weight in the global fitting
        :param mappings: mappings to use from experiments (None uses all mappings)
        :param fit_parameters: LOCAL parameters only changed in this simulation
                                experiment
        """
        self.experiment_class = experiment
        self.mappings = mappings
        self.weight = weight
        if fit_parameters is None:
            self.fit_parameters = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.experiment_class} {self.mappings})"


class FitMapping(object):
    """Mapping of reference data to obeservables in the model."""

    def __init__(self, experiment: 'sbmlsim.experiment.SimulationExperiment',
                 reference: 'FitData', observable: 'FitData'):
        """FitMapping.

        :param reference: reference data (mostly experimental data)
        :param observable: observable in model
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable


class FitParameter(object):
    """Parameter adjusted in a parameter optimization."""

    def __init__(self, parameter_id: str,
                 start_value: float = None,
                 lower_bound: float = -np.Inf, upper_bound: float = np.Inf,
                 unit: str = None):
        """FitParameter.

        :param parameter_id: id of parameter in the model
        :param start_value: initial value for fitting
        :param lower_bound: bounds for fitting
        :param upper_bound: bounds for fitting

        """
        self.pid = parameter_id
        self.start_value = start_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.unit = unit

    def __str__(self):
        return f"{self.__class__.__name__}<{self.pid} = {self.start_value} " \
               f"[{self.lower_bound} - {self.upper_bound}]>"


class FitData(object):
    """Data used in a fit.

    This is either data from a dataset, a simulation results from
    a task or functional data, i.e. calculated from other data.
    """
    def __init__(self, experiment: 'SimulationExperiment',
                 xid: str, yid: str,
                 xid_sd: str=None, xid_se: str=None,
                 yid_sd: str=None, yid_se: str=None,
                 dataset: str=None, task: str=None, function: str=None):

        self.dset_id = dataset
        self.task_id = task
        self.function = function

        # FIXME: simplify
        self.x = Data(experiment=experiment, index=xid,
                      task=self.task_id, dataset=self.dset_id, function=self.function)
        self.y = Data(experiment=experiment, index=yid,
                      task=self.task_id, dataset=self.dset_id, function=self.function)
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None
        if xid_sd:
            self.x_sd = Data(experiment=experiment, index=xid_sd, task=self.task_id,
                             dataset=self.dset_id, function=self.function)
        if xid_se:
            self.x_se = Data(experiment=experiment, index=xid_se, task=self.task_id,
                             dataset=self.dset_id, function=self.function)
        if yid_sd:
            self.y_sd = Data(experiment=experiment, index=yid_sd, task=self.task_id,
                             dataset=self.dset_id, function=self.function)
        if yid_se:
            self.y_se = Data(experiment=experiment, index=yid_se, task=self.task_id,
                             dataset=self.dset_id, function=self.function)

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

    def get_data(self) -> Dict:
        """Returns actual data."""
        result = FitDataResult()
        for key in ["x", "y", "x_sd", "x_se", "y_sd", "y_se"]:
            d = getattr(self, key)
            if d is not None:
                setattr(result, key, d.data)

        return result


class FitDataResult(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None

    def __str__(self):
        return str(self.__dict__)
