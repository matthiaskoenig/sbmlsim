from typing import List, Dict, Iterable, Set
import numpy as np
import scipy
from scipy import optimize
from scipy import interpolate
from collections import defaultdict
from pathlib import Path
from enum import Enum


import time
import pandas as pd
import logging
from dataclasses import dataclass

from sbmlsim.data import Data
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulation import TimecourseSim, ScanSim
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.utils import timeit
from sbmlsim.plot.plotting_matplotlib import plt  # , GridSpec
import seaborn as sns

from .fitting import FitExperiment, FitParameter

logger = logging.getLogger(__name__)

@dataclass
class RuntimeErrorOptimizeResult:
    status: str = -1
    success: bool = False
    duration: float = -1.0
    cost: float = np.Inf
    optimality: float = np.Inf

class SamplingType(Enum):
    LOGUNIFORM = 1
    UNIFORM = 2

class OptimizerType(Enum):
    LEAST_SQUARE = 1
    DIFFERENTIAL_EVOLUTION = 2

class OptimizationProblem(object):
    """Parameter optimization problem."""

    def __init__(self, fit_experiments: Iterable[FitExperiment],
                 fit_parameters: Iterable[FitParameter], simulator=None,
                 base_path=None, data_path=None):
        """Optimization problem.

        :param fit_experiments:
        :param fit_parameters:
        """
        self.fit_experiments = FitExperiment.reduce(fit_experiments)
        self.parameters = fit_parameters

        # parameter information
        self.pids = [p.pid for p in self.parameters]
        self.punits = [p.unit for p in self.parameters]
        lb = [p.lower_bound for p in self.parameters]
        ub = [p.upper_bound for p in self.parameters]
        self.bounds = [lb, ub]
        self.x0 = [p.start_value for p in self.parameters]

        # Create experiment runner (loads the experiments & all models)
        exp_classes = {fit_exp.experiment_class for fit_exp in self.fit_experiments}
        self.runner = ExperimentRunner(
            experiment_classes=exp_classes,
            simulator=simulator,
            base_path=base_path,
            data_path=data_path
        )

        # prepare reference data for all mappings (full length lists)
        self.experiment_keys = []
        self.mapping_keys = []
        self.xid_observable = []
        self.yid_observable = []
        self.x_references = []
        self.y_references = []
        self.y_errors = []
        self.weights = []
        self.weights_mapping = []

        self.models = []
        self.simulations = []
        self.selections = []

        # Collect information for simulations
        for fit_experiment in self.fit_experiments:

            # get simulation experiment
            sid = fit_experiment.experiment_class.__name__
            sim_experiment = self.runner.experiments[sid]

            # FIXME: selections should be based on fit mappings
            selections = set()
            for d in sim_experiment._data.values():  # type: Data
                if d.is_task():
                    selections.add(d.index)
            selections = list(selections)

            # use all fit_mappings if None are provided
            if fit_experiment.mappings is None:
                fit_experiment.mappings = list(sim_experiment._fit_mappings.keys())
                fit_experiment.weights = [1.0] * len(fit_experiment.mappings)

            # collect information for single mapping
            for k, mapping_id in enumerate(fit_experiment.mappings):
                # weight of mapping

                weight = fit_experiment.weights[k]

                # sanity checks
                if mapping_id not in sim_experiment._fit_mappings:
                    raise ValueError(f"Mapping key '{mapping_id}' not defined in "
                                     f"SimulationExperiment '{sim_experiment}'.")

                mapping = sim_experiment._fit_mappings[mapping_id]  # type: FitMapping

                if mapping.observable.task_id is None:
                    raise ValueError(f"Only observables from tasks supported: "
                                     f"'{mapping.observable}'")
                if mapping.reference.dset_id is None:
                    raise ValueError(f"Only references from datasets supported: "
                                     f"'{mapping.reference}'")

                task_id = mapping.observable.task_id
                task = sim_experiment._tasks[task_id]
                model = sim_experiment._models[task.model_id]  # type: RoadrunnerSBMLModel
                # only set the tolerances at the beginning
                # FIXME: better setting of parameters

                simulation = sim_experiment._simulations[task.simulation_id]

                if not isinstance(simulation, TimecourseSim):
                    raise ValueError(f"Only TimecourseSims supported in fitting: "
                                     f"'{simulation}")

                # observable units
                obs_xid = mapping.observable.x.index
                obs_yid = mapping.observable.y.index
                obs_x_unit = model.udict[obs_xid]
                obs_y_unit = model.udict[obs_yid]

                # prepare data
                data_ref = mapping.reference.get_data()
                data_ref.x = data_ref.x.to(obs_x_unit)
                data_ref.y = data_ref.y.to(obs_y_unit)
                x_ref = data_ref.x.magnitude
                y_ref = data_ref.y.magnitude

                y_ref_err = None
                if data_ref.y_sd is not None:
                    y_ref_err = data_ref.y_sd.to(obs_y_unit).magnitude
                elif data_ref.y_se is not None:
                    y_ref_err = data_ref.y_se.to(obs_y_unit).magnitude
                # handle special case of all NaN errors
                if y_ref_err is not None and np.all(np.isnan(y_ref_err)):
                    y_ref_err = None

                # remove NaN
                x_ref = x_ref[~np.isnan(y_ref)]
                if y_ref_err is not None:
                    y_ref_err = y_ref_err[~np.isnan(y_ref)]
                y_ref = y_ref[~np.isnan(y_ref)]

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

                # lookup maps
                self.models.append(model)
                self.simulations.append(simulation)
                self.selections.append(selections)

                # store information
                self.experiment_keys.append(sid)
                self.mapping_keys.append(mapping_id)
                self.xid_observable.append(obs_xid)
                self.yid_observable.append(obs_yid)
                self.x_references.append(x_ref)
                self.y_references.append(y_ref)
                self.y_errors.append(y_ref_err)
                self.weights.append(weights)
                self.weights_mapping.append(weight)

    def __str__(self):
        """String representation."""
        info = []
        info.append("-"*80)
        info.append(self.__class__.__name__)
        info.append("-" * 80)
        info.append("Experiments")
        info.extend([f"\t{e}" for e in self.runner.experiments])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        info.append("-" * 80)
        return "\n".join(info)

    def report(self, output_path=None):
        """Print and write report."""
        info = str(self)
        print(info)
        if output_path is not None:
            filepath = output_path / "optimization_problem.txt"
            with open(filepath, "w") as fout:
                fout.write(info)

    def run_optimization(self, size: int = 5, seed: int = 1234,
                         plot_results: bool = True, output_path=None, **kwargs) -> List[scipy.optimize.OptimizeResult]:
        """Runs the given optimization problem.

        This changes the internal state of the optimization problem
        and provides simple access to parameters and costs after
        optimization.

        :param size: number of optimization with different start values
        :param seed: random seed (for sampling of parameters)
        :param plot_results: should standard plots be generated
        :param output_path: path (directory) to store results
        :param kwargs: additional arguments for optimizer, e.g. xtol
        :return: list of optimization results
        """
        self.report(output_path=output_path)

        # optimize
        fits = self.optimize(size=size, seed=seed, **kwargs)
        self.fit_report(output_path=output_path)

        if plot_results:
            if len(fits) > 1:
                self.plot_waterfall(output_path=output_path)
                self.plot_correlation(output_path=output_path)

            # plot top fit
            self.plot_costs(output_path=output_path)
            self.plot_residuals(output_path=output_path)

        return fits

    def optimize(self, size=10, seed=None,
                 optimizer="least square", sampling=SamplingType.UNIFORM,
                 max_bound=1E10, min_bound=1E-10,
                 **kwargs) -> List[scipy.optimize.OptimizeResult]:
        """Run parameter optimization"""
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
            if sampling == SamplingType.UNIFORM:
                x0_values[1:, k] = np.random.uniform(lb, ub, size=size-1)
            elif sampling == SamplingType.LOGUNIFORM:
                # only working with positive values
                if lb <= 0.0:
                    lb = min_bound
                lb_log = np.log10(lb)
                ub_log = np.log10(ub)

                values_log = np.random.uniform(lb_log, ub_log, size=size-1)
                x0_values[1:, k] = np.power(10, values_log)

        x0_samples = pd.DataFrame(x0_values, columns=[p.pid for p in self.parameters])
        print("samples:")
        print(x0_samples)

        fits = []
        # TODO: parallelization
        for k in range(size):
            x0 = x0_samples.values[k, :]
            print(f"[{k+1}/{size}] optimize from x0={x0}")
            fits.append(
                self._optimize_single(x0=x0, optimizer=optimizer, **kwargs)
            )

        self.fit_results = self.process_fits(fits)
        self.xopt = self.fit_results.x.iloc[0]
        return fits

    @timeit
    def _optimize_single(self, x0=None, optimizer=OptimizerType.LEAST_SQUARE,
                         **kwargs) -> scipy.optimize.OptimizeResult:
        """ Runs single optimization with x0 start values.

        :param x0: parameter start vector (important for deterministic optimizers)
        :param optimizer: optimization algorithm and method
        :param kwargs:
        :return:
        """
        if x0 is None:
            x0 = self.x0

        if optimizer == OptimizerType.LEAST_SQUARE:
            # scipy least square optimizer
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            ts = time.time()
            try:
                opt_result = optimize.least_squares(
                    fun=self.residuals, x0=x0, bounds=self.bounds, **kwargs
                )
            except RuntimeError:
                logger.error("RuntimeError in ODE integration")
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = (te - ts)
            return opt_result

        elif optimizer == OptimizerType.DIFFERENTIAL_EVOLUTION:
            # scipy differential evolution
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
            ts = time.time()
            try:
                de_bounds = [(p.lower_bound, p.upper_bound) for k, p in enumerate(self.parameters)]
                opt_result = optimize.differential_evolution(
                    func=self.cost_least_square, bounds=de_bounds, **kwargs
                )
            except RuntimeError:
                logger.error("RuntimeError in ODE integration")
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = (te - ts)
            opt_result.cost = self.cost_least_square(opt_result.x)
            return opt_result

        else:
            raise ValueError(f"optimizer is not supported: {optimizer}")

    def cost_least_square(self, x):
        res_weighted = self.residuals(x)
        return 0.5 * np.sum(np.power(res_weighted, 2))

    def residuals(self, x, complete_data=False):
        """Calculates residuals for given parameter vector.

        :param x: parameter vector
        :param complete_data: boolean flag to return additional information
        :return: vector of weighted residuals
        """
        print(f"\t{x}")
        parts = []
        if complete_data:
            residual_data = defaultdict(list)

        # simulate all mappings for all experiments
        simulator = self.runner.simulator  # type: SimulatorSerial
        Q_ = self.runner.Q_

        for k, mapping_id in enumerate(self.mapping_keys):
            # Overwrite initial changes in the simulation
            changes = {
                self.pids[i]: Q_(value, self.punits[i]) for i, value in enumerate(x)
            }
            self.simulations[k].timecourses[0].changes.update(changes)

            # set model in simulator (FIXME: update only when necessary)
            simulator.set_model(model=self.models[k])
            simulator.set_integrator_settings(variable_step_size=True,
                                              relative_tolerance=1E-6, absolute_tolerance=1E-8)
            simulator.set_timecourse_selections(selections=self.selections[k])

            # FIXME: normalize simulations and parameters once outside of loop
            simulation = self.simulations[k]  # type: TimecourseSim
            simulation.normalize(udict=simulator.udict, ureg=simulator.ureg)

            # run simulation
            # logger.warning(f"Running simulation: {k} - {self.experiment_keys[k]} - {mapping_id}")
            df = simulator._timecourses([simulation])[0]

            # interpolation of simulation results
            f = interpolate.interp1d(
                x=df[self.xid_observable[k]],
                y=df[self.yid_observable[k]],
                copy=False, assume_sorted=True
            )
            y_obsip = f(self.x_references[k])

            # scaling of residuals
            # scale = 1E8

            # calculate weighted residuals
            parts.append(
                (y_obsip - self.y_references[k]) * self.weights[k] * self.weights_mapping[k] # * scale
            )

            if complete_data:
                res = (y_obsip - self.y_references[k]) # * scale
                res_weighted = res * self.weights[k] * self.weights_mapping[k]
                residual_data["x_obs"].append(df[self.xid_observable[k]])
                residual_data["y_obs"].append(df[self.yid_observable[k]])
                residual_data["y_obsip"].append(y_obsip)
                residual_data["residuals"].append(res)
                residual_data["residuals_weighted"].append(res_weighted)
                residual_data["cost"].append(0.5 * np.sum(np.power(res_weighted, 2)))

        if complete_data:
            return residual_data
        else:
            return np.concatenate(parts)

    def process_fits(self, fits: List[scipy.optimize.OptimizeResult]):
        """Process the optimization results."""
        results = []
        pids = [p.pid for p in self.parameters]
        # print(fits)
        for fit in fits:
            res = {
                # 'status': fit.status,
                'success': fit.success,
                'duration': fit.duration,
                'cost': fit.cost,
                 # 'optimality': fit.optimality,
                'message': fit.message if hasattr(fit, "message") else None
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

    def fit_report(self, output_path: Path=None):
        """ Readable report of optimization. """
        pd.set_option('display.max_columns', None)
        info = [
            "-" * 80,
            str(self.fit_results),
            "-" * 80,
            "Optimal parameters:",
        ]

        xopt = self.fit_results.x.iloc[0]
        fitted_pars = {}
        for k, p in enumerate(self.parameters):
            fitted_pars[p.pid] = (xopt[k], p.unit)

        for key, value in fitted_pars.items():
            info.append(
                "\t'{}': Q_({}, '{}'),".format(key, value[0], value[1])
            )
        info.append("-" * 80)
        info = "\n".join(info)
        print(info)
        if output_path is not None:
            filepath = output_path / "fit_report.txt"
            with open(filepath, "w") as fout:
                fout.write(info)

    def plot_waterfall(self, output_path=None):
        """Process the optimization results."""
        df = self.fit_results
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.plot(range(len(df)), 1 + (df.cost-df.cost.iloc[0]), '-o', color="black")
        ax.set_xlabel("index (Ordered optimizer run)")
        ax.set_ylabel("Offsetted cost value (relative to best start)")
        # ax.set_yscale("log")
        ax.set_title("Waterfall plot")
        plt.show()

        if output_path is not None:
            filepath = output_path / "01_waterfall.svg"
            fig.savefig(filepath, bbox_inches="tight")

    def plot_costs(self, xstart=None, xopt=None, output_path: Path=None):
        """Plots bar diagram of costs for set of residuals

        :param res_data_start:
        :param res_data_fit:
        :param filepath:
        :return:
        """
        if xstart is None:
            xstart = self.x0
        if xopt is None:
            xopt = self.xopt

        print("xstart", xstart)
        print("xopt", xopt)

        res_data_start = self.residuals(x=xstart, complete_data=True)
        res_data_fit = self.residuals(x=xopt, complete_data=True)

        data = []
        types = ["initial", "fit"]

        for k in range(len(self.mapping_keys)):
            for kdata, res_data in enumerate([res_data_start, res_data_fit]):

                data.append({
                    'id': f"{self.experiment_keys[k]}_{self.mapping_keys[k]}",
                    'experiment': self.experiment_keys[k],
                    'mapping': self.mapping_keys[k],
                    'cost': res_data['cost'][k],
                    'type': types[kdata]
                })
        cost_df = pd.DataFrame(data, columns=["id", "experiment", "mapping", "cost", "type"])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        sns.set_color_codes("pastel")
        sns.barplot(ax=ax, x="cost", y="id", hue="type", data=cost_df)
        ax.set_xscale("log")
        plt.show()
        if output_path:
            filepath = output_path / "02_costs.svg"
            fig.savefig(filepath, bbox_inches="tight")


    def plot_correlation(self, output_path=None):
        """Process the optimization results."""
        df = self.fit_results

        pids = [p.pid for p in self.parameters]
        sns.set(style="ticks", color_codes=True)
        # sns_plot = sns.pairplot(data=df[pids + ["cost"]], hue="cost", corner=True)

        npars = len(pids)
        # x0 =
        fig, axes = plt.subplots(nrows=npars, ncols=npars, figsize=(5*npars, 5*npars))
        size = 20 * (df.cost.min() / df.cost.values)
        alpha = 0.6*df.cost.min() / df.cost.values
        for kx, pidx in enumerate(pids):
            for ky, pidy in enumerate(pids):
                if npars == 1:
                    ax = axes
                else:
                    ax = axes[kx][ky]

                # optimal values
                if kx >= ky:
                    # start values
                    for ks in range(len(size)):
                        # x = df[pidx].values[ks]
                        # y = df[pidy].values[ks]
                        if 'x0' in df.columns:
                            xstart = df.x0[ks][kx]
                            ystart = df.x0[ks][ky]
                            ax.plot(xstart, ystart, "x", color="black")
                        # axes[kx][ky].plot(x, y, "o", color="black", markersize=size[ks], alpha=alpha[ks])

                    # optimal values
                    ax.scatter(df[pidx], df[pidy], c=df.cost, s=np.power(size,2), alpha=1.0,
                                         cmap="RdBu")
                    ax.set_xlabel(pidx)
                    ax.set_ylabel(pidy)

                    ax.plot(self.xopt[kx], self.xopt[ky], "s",
                                      color="darkgreen", markersize=30,
                                      alpha=0.9)

                # trajectory
                if kx < ky:
                    # start values
                    for ks in range(len(size)):
                        x = df[pidx].values[ks]
                        y = df[pidy].values[ks]
                        if 'x0' in df.columns:
                            xstart = df.x0[ks][kx]
                            ystart = df.x0[ks][ky]
                            # start point
                            ax.plot(xstart, ystart, "x", color="black")
                            # end point
                            ax.plot(x, y, "o", color="black")
                            # search
                            ax.plot([xstart, x], [ystart, y], "-", color="black")

                            #print("xstart", xstart)
                            #print("ystart", ystart)
                            #print("x", x)
                            #print("y", y)
                # plt.colorbar()


        plt.show()
        if output_path is not None:
            filepath = output_path / "03_parameter_correlation.svg"
            # sns_plot.savefig(filepath, bbox_inches="tight")
            fig.savefig(filepath, bbox_inches="tight")

    def plot_residuals(self, xstart=None, xopt=None, output_path: Path = None):
        """ Plot residual data.

        :param res_data_start: initial residual data
        :return:
        """
        if xstart is None:
            xstart = self.x0
        if xopt is None:
            xopt = self.xopt

        titles = ["initial", "fit"]
        res_data_start = self.residuals(x=xstart, complete_data=True)
        res_data_fit = self.residuals(x=xopt, complete_data=True)

        for k, mapping_id in enumerate(self.mapping_keys):
            fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

            axes = [(a1, a3, a5), (a2, a4, a6)]
            if titles is None:
                titles = ["Initial", "Fit"]

            # global reference data
            sid = self.experiment_keys[k]
            mapping_id = self.mapping_keys[k]
            weights = self.weights[k]
            x_ref = self.x_references[k]
            y_ref = self.y_references[k]
            y_ref_err = self.y_errors[k]
            x_id = self.xid_observable[k]
            y_id = self.yid_observable[k]

            for kdata, res_data in enumerate([res_data_start, res_data_fit]):
                ax1, ax2, ax3 = axes[kdata]
                title = titles[kdata]

                # calculated data in residuals

                x_obs = res_data['x_obs'][k]
                y_obs = res_data['y_obs'][k]
                y_obsip = res_data['y_obsip'][k]
                res = res_data['residuals'][k]
                res_weighted = res_data['residuals_weighted'][k]
                cost = res_data['cost'][k]

                for ax in (ax1, ax2, ax3):
                    ax.axhline(y=0, color="black")
                    ax.set_ylabel(y_id)
                ax3.set_xlabel(x_id)

                if y_ref_err is None:
                    ax1.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax1.errorbar(x_ref, y_ref, yerr=y_ref_err,
                                 marker="s", color="black", label="reference_data")
                ax1.plot(x_obs, y_obs, "-",
                         color="blue", label="observable")
                ax1.plot(x_ref, y_obsip, "o", color="blue",
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
                    full_title = "{}_{}: {} (cost={:.3e})".format(
                        sid, mapping_id, title, cost
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
                        ax.set_ylim([min(ylim1[0], ylim2[0]), max(ylim1[1],ylim2[1])])

            plt.show()
            if output_path is not None:
                fig.savefig(output_path / f"residuals_{sid}_{mapping_id}.svg",
                            bbox_inches="tight")
