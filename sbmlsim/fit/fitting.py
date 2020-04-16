import numpy as np
from scipy import optimize
from typing import List, Dict, Iterable
from sbmlsim.data import Data


from sbmlsim.data import Data


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

        self.dataset = dataset
        self.task = task
        self.function = function

        # FIXME: simplify
        self.x = Data(experiment=experiment, index=xid,
                 task=self.task, dataset=self.dataset, function=self.function)
        self.y = Data(experiment=experiment, index=yid,
                 task=self.task, dataset=self.dataset, function=self.function)
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None
        if xid_sd:
            self.x_sd = Data(experiment=experiment, index=xid_sd, task=self.task,
                        dataset=self.dataset, function=self.function)
        if xid_se:
            self.x_se = Data(experiment=experiment, index=xid_se, task=self.task,
                        dataset=self.dataset, function=self.function)
        if yid_sd:
            self.y_sd = Data(experiment=experiment, index=yid_sd, task=self.task,
                        dataset=self.dataset, function=self.function)
        if yid_se:
            self.y_se = Data(experiment=experiment, index=yid_se, task=self.task,
                             dataset=self.dataset, function=self.function)

    def get_data(self):
        for key in ["x", "y", "x_sd", "x_se", "y_sd", "y_se"]:
            d = getattr(self, key)
            if d is not None:
                data = d.data
                print(f"FitData: {key} = {data}")


class FitMapping(object):
    """Mapping of reference data to obeservables in the model."""

    def __init__(self, experiment: 'sbmlsim.experiment.SimulationExperiment',
                 reference: FitData, observable: FitData):
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
                 lower_bound: float = -np.Inf, upper_bound: float = np.Inf):
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

    def __str__(self):
        return f"{self.__class__.__name__}<{self.pid} = {self.start_value} " \
               f"[{self.lower_bound} - {self.upper_bound}]>"


class FitExperiment(object):
    """
    A Simulation Experiment used in a fitting.
    """

    def __init__(self, experiment: 'sbmlsim.simulation.SimulationExperiment',
                 weight: float=1.0, mappings: List[str]=None,
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
            mappings = experiment._fit_mappings
        self.mappings = mappings
        self.weight = weight
        if fit_parameters is None:
            self.fit_parameters = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.experiment} {self.mappings})"


class FitResult(object):
    def __init__(self, parameters, status, trajectory):
        """ Storage of result.

        :param parameters:
        :param status:
        :param trajectory:
        """
        self.parameters = parameters
        self.status = status
        self.trajectory = trajectory


class OptimizationProblem(object):
    """Parameter optimization problem."""

    def __init__(self, fit_experiments: Iterable[FitExperiment],
                 fit_parameters: Iterable[FitParameter]):
        """Optimization problem.

        :param fit_experiments:
        :param fit_parameters:
        """
        self.experiments = fit_experiments
        self.parameters = fit_parameters

    @property
    def x0(self):
        """Initial values of parameters."""
        return [p.start_value for p in self.parameters]

    @property
    def bounds(self):
        """Bounds of parameters."""
        lb = [p.lower_bound for p in self.parameters]
        ub = [p.upper_bound for p in self.parameters]
        return [lb, ub]

    def __str__(self):
        """String representation."""
        info = []
        info.append("-"*80)
        info.append(self.__class__.__name__)
        info.append("-" * 80)
        info.append("Experiments")
        info.extend([f"\t{e}" for e in self.experiments])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        info.append("-" * 80)
        return "\n".join(info)

    def report(self):
        print(str(self))

    def timecourse_sims(self):
        """Collect all the timecourse_simulations which must be run

        :return:
        """
        # TODO:

        self.simulations = None

    def residuals(self, p, **kwargs):
        """ Calculates residuals

        :return:
        """
        m = 1
        residuals = np.zeros(shape=(m,))
        # **kwargs is dictionary of current parameters

        # [1] simulate all simulation experiments with current parameters
        # TODO: simulation
        # get tasks to run

        # inject parameters in tasks

        # [2] interpolate observables with reference time points
        # TODO: interpolation (make this fast (c++ and numba))

        # [3] calculate residuals between simulation and reference output
        raise NotImplementedError

    def optimize(self):
        """Runs optimization problem.

        Starts many optimizations with sampling, parallelization, ...

        Returns list of optimal parameters with respective value of objective value
        and residuals (per fit experiment).
        -> local optima of optimization

        """
        x0 = self.x0
        bounds = self.bounds
        results = optimize.least_squares(fun=self.residual, x0=p0, bounds=(-np.inf, np.inf),
                                            kwargs={"x": x, "y": y})
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