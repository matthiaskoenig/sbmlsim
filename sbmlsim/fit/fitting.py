from typing import List, Dict, Iterable, Set

from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from scipy import interpolate
import seaborn as sns
import pandas as pd

from sbmlsim.data import Data
from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.utils import timeit
from sbmlsim.plot.plotting_matplotlib import plt  # , GridSpec


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
                 weight: float=1.0, mappings: Set[str]=None,
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

    def residuals(self, p, complete_data=False):
        """ Calculates residuals

        :return:
        """
        print(f"\t{p}")
        # run simulations
        self.run_tasks(p)

        # get data for residual calculation
        parts = []
        if complete_data:
            residual_data = {}
        for fit_experiment in self.experiments:
            sim_experiment = fit_experiment.experiment
            for key, mapping in sim_experiment._fit_mappings.items():
                if key not in fit_experiment.mappings:
                    continue

                # Get actual data from the results
                data_obs = mapping.observable.get_data()

                # convert & prepare reference data to observable
                # -------------------------------------
                # FIXME: Do once outside of residuals!
                data_ref = mapping.reference.get_data()
                data_ref.x = data_ref.x.to(data_obs.x.units)
                data_ref.y = data_ref.y.to(data_obs.y.units)
                x_ref = data_ref.x.magnitude
                y_ref = data_ref.y.magnitude

                y_ref_err = None
                if data_ref.y_sd is not None:
                    y_ref_err = data_ref.y_sd.to(data_obs.y.units).magnitude
                elif data_ref.y_se is not None:
                    y_ref_err = data_ref.y_se.to(data_obs.y.units).magnitude
                # handle special case of all NaN errors
                if y_ref_err is not None and np.all(np.isnan(y_ref_err)):
                    y_ref_err = None

                # remove NaN
                x_ref = x_ref[~np.isnan(y_ref)]
                if y_ref_err is not None:
                    y_ref_err = y_ref_err[~np.isnan(y_ref)]
                y_ref = y_ref[~np.isnan(y_ref)]
                # -------------------------------------

                # interpolation
                # TODO: interpolation (make this fast (c++ and numba))
                # FIXME: make a fast interpolation via the datapoints left and right of experimental
                # points (or directly request the necessary data points)
                f = interpolate.interp1d(x=data_obs.x.magnitude, y=data_obs.y.magnitude, copy=False, assume_sorted=True)
                y_obs = f(x_ref)

                # calculate weights based on errors
                if y_ref_err is None:
                    weights = np.ones_like(y_ref)
                else:
                    # handle special case that all errors are NA (no normalization possible)
                    weights = 1.0 / y_ref_err  # the larger the error, the smaller the weight
                    weights[np.isnan(weights)] = np.nanmax(
                        weights)  # NaNs are filled with minimal errors, i.e. max weights
                    weights = weights / np.min(
                        weights)  # normalize minimal weight to 1.0

                # experiment based weight
                weights = weights * fit_experiment.weight

                # calculate residuals
                res = y_obs - y_ref
                res_weighted = res * weights
                parts.append(res_weighted)

                if complete_data:
                    residual_data[f"{sim_experiment.sid}_{key}"] = {
                        "experiment": sim_experiment.sid,
                        "mapping": key,
                        "data_obs": data_obs,
                        "data_ref": data_ref,
                        "x_ref": x_ref,
                        "y_ref": y_ref,
                        "y_ref_err": y_ref_err,
                        "y_obs": y_obs,
                        "res": res,
                        "res_weighted": res_weighted,
                        "weights": weights,
                        "cost": 0.5 * np.sum(np.power(res_weighted, 2))
                    }

        if complete_data:
            return residual_data
        else:
            return np.concatenate(parts)

    @timeit
    def _optimize_single(self, x0=None, optimizer="least square", **kwargs) -> scipy.optimize.OptimizeResult:
        """ Runs single optimization with x0 start values.

        :param x0: parameter start vector (important for deterministic optimizers)
        :param optimizer: optimization algorithm and method
        :param kwargs:
        :return:
        """
        if x0 is None:
            x0 = self.x0

        if optimizer == "least square":
            opt_result = optimize.least_squares(
                fun=self.residuals, x0=x0, bounds=self.bounds, **kwargs
            )
            opt_result.x0 = x0  # store start value
            return opt_result
        else:
            raise ValueError(f"optimizer is not supported: {optimizer}")

    def optimize(self, size=10, seed=None,
                 optimizer="least square", sampling="loguniform",
                 max_bound=1E10, min_bound=1E-10,
                 **kwargs) -> List[scipy.optimize.OptimizeResult]:
        """Run multiple optimizations."""

        # create the sample parameters
        x0_values = np.zeros(shape=(size, len(self.parameters)))
        # parameter set are the x0 values
        x0_values[0, :] = self.x0
        # remaining samples are random samples
        if seed:
            np.random.seed(seed)
        for k, p in enumerate(self.parameters):
            lb = p.lower_bound if not np.isinf(p.lower_bound) else -max_bound
            ub = p.upper_bound if not np.isinf(p.upper_bound) else max_bound

            # uniform sampling
            if sampling == "uniform":
                x0_values[1:,k] = np.random.uniform(lb, ub, size=size-1)
            elif sampling == "loguniform":
                # only working with positive values
                if lb <= 0.0:
                    lb = min_bound
                lb_log = np.log10(lb)
                ub_log = np.log10(ub)

                values_log = np.random.uniform(lb_log, ub_log, size=size)
                x0_values[:, k] = np.power(10, values_log)

        x0_samples = pd.DataFrame(x0_values, columns=[p.pid for p in self.parameters])
        print("samples:")
        print(x0_samples)

        fits = []
        for k in range(size):
            x0 = x0_samples.values[k, :]
            print(f"[{k+1}/{size}] optimize from x0={x0}")
            fits.append(
                self._optimize_single(x0=x0, optimizer=optimizer, **kwargs)
            )
        return fits

    def process_fits(self, fits: List[scipy.optimize.OptimizeResult]):
        """Process the optimization results."""
        results = []
        pids = [p.pid for p in self.parameters]
        for fit in fits:
            res = {
                'status': fit.status,
                'success': fit.success,
                'cost': fit.cost,
                'optimality': fit.optimality,
                # 'message': fit.message
            }
            # add parameter columns
            for k, pid in enumerate(pids):
                res[pid] = fit.x[k]
            res['x'] = fit.x
            res['x0'] = fit.x0

            results.append(res)
        df = pd.DataFrame(results)
        df.sort_values(by=["cost"], inplace=True)
        return df


    def plot_residuals(self, res_data_start, res_data_fit=None,
                       titles=["initial", "fit"], filepath=None):
        """ Plot residual data.

        :param res_data_start: initial residual data
        :return:
        """

        for sid in res_data_start.keys():
            fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

            axes = [(a1, a3, a5), (a2, a4, a6)]
            if titles is None:
                titles = ["Initial", "Fit"]
            for k, res_data in enumerate([res_data_start, res_data_fit]):
                if res_data is None:
                    continue

                ax1, ax2, ax3 = axes[k]
                title = titles[k]

                # get data
                rdata = res_data[sid]
                data_obs = rdata['data_obs']
                data_ref = rdata['data_ref']
                x_ref = rdata['x_ref']
                y_ref = rdata['y_ref']
                y_ref_err = rdata['y_ref_err']
                y_obs = rdata['y_obs']
                res = rdata['res']
                res_weighted = rdata['res_weighted']
                weights = rdata['weights']
                cost = rdata['cost']

                for ax in (ax1, ax2, ax3):
                    ax.axhline(y=0, color="black")

                if y_ref_err is None:
                    ax1.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax1.errorbar(x_ref, y_ref, yerr=y_ref_err,
                                 marker="s", color="black", label="reference_data")
                ax1.plot(data_obs.x.magnitude, data_obs.y.magnitude, "-",
                         color="blue", label="observable")
                ax1.plot(x_ref, y_obs, "o", color="blue",
                         label="interpolation")
                for ax in (ax1, ax2):
                    ax.plot(x_ref, res, "o", color="darkorange",
                            label="residuals")
                ax1.fill_between(x_ref, res, np.zeros_like(res),
                                 alpha=0.4, color="darkorange", label="__nolabel__")

                ax2.plot(x_ref, res_weighted, "o", color="darkgreen",
                         label="weighted residuals")
                ax2.fill_between(x_ref, res_weighted,
                                 np.zeros_like(res), alpha=0.4, color="darkgreen",
                                 label="__nolabel__")

                res_weighted2 = np.power(res_weighted, 2)
                ax3.plot(x_ref, res_weighted2, "o", color="darkred",
                         label="(weighted residuals)^2")
                ax3.fill_between(x_ref, res_weighted2,
                                 np.zeros_like(res), alpha=0.4, color="darkred",
                                 label="__nolabel__")

                for ax in (ax1, ax2):
                    plt.setp(ax.get_xticklabels(), visible=False)

                # ax3.set_xlabel("x")
                for ax in (ax2, ax3):
                    ax.set_xlim(ax1.get_xlim())

                if title:
                    full_title = "{}: {} (cost={:.4e})".format(
                        sid, title, cost
                    )
                    ax1.set_title(full_title)
                for ax in (ax1, ax2, ax3):
                    plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.set_ylabel("y")
                    ax.legend()

            # adapt axes
            if res_data_fit is not None:
                for axes in [(a1, a2), (a3, a4), (a5, a6)]:
                    ax1, ax2 = axes
                    ylim1 = ax1.get_ylim()
                    ylim2 = ax2.get_ylim()
                    for ax in axes:
                        ax.set_ylim([min(ylim1[0],ylim2[0]), max(ylim1[1],ylim2[1])])

            if filepath is not None:
                fig.savefig(filepath / f"{sid}.png")
            plt.show()

    def fit_report(self, fits: List[scipy.optimize.OptimizeResult]):
        """ Readable report of optimization.
        """
        # plot top fit
        fit_results = self.process_fits(fits)

        pd.set_option('display.max_columns', None)
        print("-" * 80)
        print(fit_results)
        print("-" * 80)
        print("Optimal parameters:")
        fitted_pars = dict(zip(
            [p.pid for p in self.parameters],
            fit_results.x[0]
        ))
        for key, value in fitted_pars.items():
            print("\t{:<15} {}".format(key, value))
        print("-" * 80)

    def plot_waterfall(self, fits: List[scipy.optimize.OptimizeResult]):
        """Process the optimization results."""
        df = self.process_fits(fits)


        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.plot(range(len(df)), 1 + (df.cost-df.cost[0]), '-o', color="black")
        ax.set_xlabel("index (Ordered optimizer run)")
        ax.set_ylabel("Offsetted cost value (relative to best start)")
        ax.set_yscale("log")
        ax.set_title("Waterfall plot")

        plt.show()

    def plot_correlation(self, fits: List[scipy.optimize.OptimizeResult]):
        """Process the optimization results."""
        df = self.process_fits(fits)

        sns.set(style="ticks", color_codes=True)
        pids = [p.pid for p in self.parameters]
        sns.pairplot(data=df[pids])
        plt.show()

    def plot_costs(self, res_data_start, res_data_fit, filepath=None):
        """Plots bar diagram of costs for set of residuals

        :param res_data_start:
        :param res_data_fit:
        :param filepath:
        :return:
        """
        data = []
        types = ["initial", "fit"]

        for sid in res_data_start.keys():
            for k, res_data in enumerate([res_data_start, res_data_fit]):
                rdata = res_data[sid]

                data.append({
                    'id': sid,
                    'experiment': rdata['experiment'],
                    'mapping': rdata['mapping'],
                    'cost': rdata['cost'],
                    'type': types[k]
                })
        cost_df = pd.DataFrame(data, columns=["id", "experiment", "mapping", "cost", "type"])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        sns.set_color_codes("pastel")
        sns.barplot(ax=ax, x="cost", y="id", hue="type", data=cost_df)
        ax.set_xscale("log")
        if filepath:
            fig.savefig(filepath)
        plt.show()
