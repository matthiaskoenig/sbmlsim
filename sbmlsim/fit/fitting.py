from typing import List, Dict
from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
import numpy as np

class FitData(object):

    def __init__(self, xid: str, yid: str,
                 xid_sd: str=None, xid_se: str=None,
                 yid_sd: str=None, yid_se: str=None,
                 dataset: str=None, task: str=None, function: str=None):
        self.xid = xid
        self.xid_sd = xid_sd
        self.xid_se = xid_se
        self.yid = yid
        self.yid_sd = yid_sd
        self.yid_se = yid_se


class FitMapping(object):

    def __init__(self, experiment: SimulationExperiment,
                 reference: FitData, observable: FitData):
        """

        :param reference: experimental reference data
        :param simulation: simulation outcome
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable

        # TODO: create data on experiment


class FitParameter(object):
    def __init__(self, parameter_id: str, model_id: str,
                 start_value: float = None,
                 lower_bound: float = -np.Inf, upper_bound: float = np.Inf):
        """

        :param pid: id of parameter in the model
        :param start_value: initial value for fitting
        :param lower_bound: bounds for fitting
        :param upper_bound: bounds for fitting
        """
        self.pid = parameter_id
        self.mid = model_id
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class FitExperiment(object):
    """
    A Simulation Experiment used in a fitting.
    """

    def __init__(self, experiment, weight: float=1.0, mappings: List[str]=None,
                 fit_parameters: List[FitParameter]=None):
        """A Simulation experiment used in a fitting.

        :param experiment:
        :param weight: weight in the global fitting
        :param mappings: mappings to use from experiments (None uses all mappings)
        :param fit_parameters: LOCAL parameters only changed in this simulation
                                experiment
        """
        self.experiment = experiment
        if mappings is None:
            mappings = []
        self.mappings = mappings
        self.weight = weight
        if fit_parameters is None:
            self.fit_parameters = []


class OptimizationProblem(object):
    """Defines the complete optimization problem."""
    def __init__(self, fit_experiments: List[FitExperiment],
                 fit_parameters: List[FitParameter],
                 validation_experiments: List[FitExperiment]=[]):

        self.experiments = fit_experiments
        self.parameters = fit_parameters
        self.validation_experiments = validation_experiments

    def _residuals(self):
        """ Calculates residuals

        :return:
        """
        # TODO: interpolation (make this fast (c++ and numba))

        # TODO implement (non-linear least square optimization

        # TODO: make this fast (c++ and numba)
        raise NotImplementedError

    def optimize(self):
        """Runs optimization problem.

        Starts many optimizations with sampling, parallelization, ...

        Returns list of optimal parameters with respective value of objective value
        and residuals (per fit experiment).
        -> local optima of optimization

        """
        # TODO implement (non-linear least square optimization)
        # TODO: make fast and native parallelization (cluster deployment)
        raise NotImplementedError

    def validate(self):
        """ Runs the validation with optimal parameters or set of local
        optima.

        :return:
        """
        # TODO implement (non-linear least square optimization
        raise NotImplementedError