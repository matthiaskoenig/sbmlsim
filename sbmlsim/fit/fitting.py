from typing import List, Dict, Iterable
from collections import defaultdict, namedtuple
import numpy as np
from scipy import optimize
from scipy import interpolate

from sbmlsim.data import Data
from sbmlsim.simulation import TimecourseSim, ScanSim

from sbmlsim.plot.plotting_matplotlib import plt

class FitDataResult(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None

    def __str__(self):
        return(str(self.__dict__))

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

        # parameter information
        self.pids = [p.pid for p in self.parameters]
        self.x0 = [p.start_value for p in self.parameters]
        self.units = [p.unit for p in self.parameters]
        lb = [p.lower_bound for p in self.parameters]
        ub = [p.upper_bound for p in self.parameters]
        self.bounds = [lb, ub]

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

    def run_tasks(self, p):
        """Run tasks"""

        results = {}
        # TODO: necessary to execute all tasks, which are used in fit models
        # with the respective parameters.
        # FIXME: much faster by getting models and simulations once with fast updates
        # of unit converted values
        for fit_experiment in self.experiments:
            sim_experiment = fit_experiment.experiment  # type: sbmlsim.simulation.SimulationExperiment
            for mapping_id in fit_experiment.mappings:
                mapping = sim_experiment._fit_mappings[mapping_id]  # type: FitMapping
                for fit_data in [mapping.reference, mapping.observable]:
                    if fit_data.is_task():
                        task_id = fit_data.task_id
                        task = sim_experiment._tasks[task_id]
                        model = sim_experiment._models[task.model_id]
                        simulation = sim_experiment._simulations[task.simulation_id]

                        # Overwrite initial changes in the simulation
                        changes = {}
                        Q_ = sim_experiment.Q_
                        for k, value in enumerate(p):
                            changes[self.pids[k]] = Q_(value, self.units[k])
                        # print("Update: ", changes)
                        simulation.timecourses[0].changes.update(changes)


                        # set model in simulator
                        sim_experiment.simulator.set_model(model=model)

                        # set selections based on data
                        # FIXME: selections must be based on fit mappings
                        selections = set()
                        for d in sim_experiment._data.values():  # type: Data
                            if d.is_task():
                                selections.add(d.index)
                        selections = sorted(list(selections))
                        # print(f"Setting selections: {selections}")
                        sim_experiment.simulator.set_timecourse_selections(selections=selections)


                if isinstance(simulation, TimecourseSim):
                    sim_experiment._results[task_id] = sim_experiment.simulator.run_timecourse(simulation)
                elif isinstance(simulation, ScanSim):
                    sim_experiment._results[task_id] = sim_experiment.simulator.run_scan(simulation)
                else:
                    raise ValueError(f"Unsupported simulation type: "
                                     f"{type(simulation)}")

        # Get the new data for the simulation experiment
        # sim_experiment.evaluate_mappings()

    def residuals(self, p, make_plots=False, title=None):
        """ Calculates residuals

        :return:
        """
        print(f"parameters: {p}")
        # run simulations
        self.run_tasks(p)

        # get data for residual calculation
        parts = []
        for fit_experiment in self.experiments:
            sim_experiment = fit_experiment.experiment
            for key, mapping in sim_experiment._fit_mappings.items():
                # Get actual data from the results

                # print("observable")
                data_obs = mapping.observable.get_data()
                # print(data_obs)

                # convert reference data to observable
                # print("reference")
                data_ref = mapping.reference.get_data()
                # print(data_ref)
                data_ref.x = data_ref.x.to(data_obs.x.units)
                data_ref.y = data_ref.y.to(data_obs.y.units)
                # print(data_ref)

                # interpolation
                # TODO: interpolation (make this fast (c++ and numba))
                f = interpolate.interp1d(x=data_obs.x.magnitude, y=data_obs.y.magnitude, copy=False, assume_sorted=True)
                yobs = f(data_ref.x.magnitude)

                # calculate residuals
                res = yobs - data_ref.y.magnitude
                parts.append(res)

                if make_plots:
                    plt.plot(data_ref.x.magnitude, data_ref.y.magnitude, "s", color="black", label="reference_data")
                    plt.plot(data_obs.x.magnitude, data_obs.y.magnitude, "-", color="black", label="observable")
                    plt.plot(data_ref.x.magnitude, yobs, "o", color="blue", label="interpolation")
                    plt.plot(data_ref.x.magnitude, res, "o", color="darkorange", label="residuals")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    if title:
                        plt.title(title)
                    plt.legend()
                    plt.show()

        residuals = np.concatenate(parts)
        return residuals


    def optimize(self, **kwargs):
        """Runs optimization problem.

        Starts many optimizations with sampling, parallelization, ...

        Returns list of optimal parameters with respective value of objective value
        and residuals (per fit experiment).
        -> local optima of optimization

        """
        return optimize.least_squares(
            fun=self.residuals, x0=self.x0, bounds=self.bounds, **kwargs
        )
