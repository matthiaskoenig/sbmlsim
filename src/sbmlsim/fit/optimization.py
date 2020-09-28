import logging
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.fit.objects import FitExperiment, FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import DimensionalityError
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


@dataclass
class RuntimeErrorOptimizeResult:
    status: str = -1
    success: bool = False
    duration: float = -1.0
    cost: float = np.Inf
    optimality: float = np.Inf


class OptimizerType(Enum):
    LEAST_SQUARE = 1
    DIFFERENTIAL_EVOLUTION = 2


class WeightingLocalType(Enum):
    """Weighting of the data points within a single fit mapping.

    This decides how the data points within a single fit mapping are
    weighted. One can account for the errors or not.
    """

    NO_WEIGHTING = 1  # data points are weighted equally
    ABSOLUTE_ONE_OVER_WEIGHTING = 2  # data points are weighted as 1/(error-min(error))
    RELATIVE_ONE_OVER_WEIGHTING = 2  # data points are weighted as 1/(error-min(error))


class ResidualType(Enum):
    """How are the residuals calculated? Are the absolute residuals used,
    or are the residuals normalized based on the data points, i.e., relative
    residuals.

    Relative residuals make different fit mappings comparable.
    """

    ABSOLUTE_RESIDUALS = 1  # (no local effects)
    RELATIVE_RESIDUALS = 2  # (local effects) induces local weighting by normalizing every residual by absolute value
    ABSOLUTE_NORMED_RESIDUALS = (
        3  # (no local effects) absolute residuals normed per mean reference data
    )


class OptimizationProblem(object):
    """Parameter optimization problem."""

    def __init__(
        self,
        opid,
        fit_experiments: Iterable[FitExperiment],
        fit_parameters: Iterable[FitParameter],
        base_path=None,
        data_path=None,
    ):
        """Optimization problem.

        The problem must be pickable for parallelization !
        So initialize must be run to create the non-pickable instances.

        :param opid: id for optimization problem
        :param fit_experiments:
        :param fit_parameters:
        """
        self.opid = opid
        self.fit_experiments = FitExperiment.reduce(fit_experiments)
        self.parameters = fit_parameters
        if self.parameters is None or len(self.parameters) == 0:
            logger.error(
                f"{opid}: parameters in optimization problem cannot be empty, "
                f"but '{self.parameters}'"
            )

        # parameter information
        self.pids = [p.pid for p in self.parameters]
        self.punits = [p.unit for p in self.parameters]
        lb = [p.lower_bound for p in self.parameters]
        ub = [p.upper_bound for p in self.parameters]
        self.bounds = [lb, ub]
        self.x0 = [p.start_value for p in self.parameters]

        # paths
        self.base_path = base_path
        self.data_path = data_path

    def __repr__(self):
        return f"<OptimizationProblem: {self.opid}>"

    def __str__(self):
        """String representation."""
        info = []
        info.append("-" * 80)
        info.append(f"{self.__class__.__name__}: {self.opid}")
        info.append("-" * 80)
        info.append("Experiments")

        # FIXME: full serialization of experiments!
        # FIXME: runner only available after initialization
        info.extend([f"\t{e}" for e in self.fit_experiments])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        info.append("-" * 80)
        return "\n".join(info)

    def initialize(
        self, weighting_local: WeightingLocalType, residual_type: ResidualType
    ):
        # weighting in fitting and handling of residuals
        if weighting_local is None:
            raise ValueError("'weighting_local' is required.")
        if residual_type is None:
            raise ValueError("'residual_type' is required.")
        logger.info(f"weighting_local: {weighting_local}")
        logger.info(f"residual_type: {residual_type}")

        self.weighting_local = weighting_local
        self.residual_type = residual_type

        # Create experiment runner (loads the experiments & all models)
        exp_classes = {fit_exp.experiment_class for fit_exp in self.fit_experiments}

        self.runner = ExperimentRunner(
            experiment_classes=exp_classes,
            base_path=self.base_path,
            data_path=self.data_path,
        )

        # prepare reference data for all mappings (full length lists)
        self.experiment_keys = []
        self.mapping_keys = []
        self.xid_observable = []
        self.yid_observable = []
        self.x_references = []
        self.y_references = []
        self.y_errors = []
        self.y_errors_type = []
        self.weights_local = []  # weights for data points
        self.weights_global_user = []  # user defined weights per mapping

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
                # user defined weight of fit mapping
                weight_global_user = fit_experiment.weights[k]

                # sanity checks
                if mapping_id not in sim_experiment._fit_mappings:
                    raise ValueError(
                        f"Mapping key '{mapping_id}' not defined in "
                        f"SimulationExperiment '{sim_experiment}'."
                    )

                mapping = sim_experiment._fit_mappings[mapping_id]  # type: FitMapping

                if mapping.observable.task_id is None:
                    raise ValueError(
                        f"Only observables from tasks supported: "
                        f"'{mapping.observable}'"
                    )
                if mapping.reference.dset_id is None:
                    raise ValueError(
                        f"Only references from datasets supported: "
                        f"'{mapping.reference}'"
                    )

                task_id = mapping.observable.task_id
                task = sim_experiment._tasks[task_id]
                model = sim_experiment._models[
                    task.model_id
                ]  # type: RoadrunnerSBMLModel
                simulation = sim_experiment._simulations[task.simulation_id]

                if not isinstance(simulation, TimecourseSim):
                    raise ValueError(
                        f"Only TimecourseSims supported in fitting: " f"'{simulation}"
                    )

                # observable units
                obs_xid = mapping.observable.x.index
                obs_yid = mapping.observable.y.index
                obs_x_unit = model.udict[obs_xid]
                obs_y_unit = model.udict[obs_yid]

                # prepare data
                data_ref = mapping.reference.get_data()
                try:
                    data_ref.x = data_ref.x.to(obs_x_unit)
                except DimensionalityError as e:
                    logger.error(
                        f"{sid}.{mapping_id}: Unit convertion fails for '{data_ref.x}' to '{obs_x_unit}"
                    )
                    raise e
                try:
                    data_ref.y = data_ref.y.to(obs_y_unit)
                except DimensionalityError as e:
                    logger.error(
                        f"{sid}.{mapping_id}: Unit convertion fails for '{data_ref.y}' to '{obs_y_unit}'."
                    )
                    raise e
                x_ref = data_ref.x.magnitude
                y_ref = data_ref.y.magnitude

                # Use errors for weighting (tries SD and falls back on SE)
                # FIXME: SE & SD not handled uniquely, i.e., slightly different weighting
                y_ref_err = None
                if data_ref.y_sd is not None:
                    y_ref_err = data_ref.y_sd.to(obs_y_unit).magnitude
                    y_ref_err_type = "SD"
                elif data_ref.y_se is not None:
                    y_ref_err = data_ref.y_se.to(obs_y_unit).magnitude
                    y_ref_err_type = "SE"
                else:
                    y_ref_err_type = None

                # handle missing data (0.0 and NaN
                if y_ref_err is not None:
                    # remove 0.0 from y-error
                    y_ref_err[(y_ref_err == 0.0)] = np.NAN
                    if np.all(np.isnan(y_ref_err)):
                        # handle special case of all NaN errors
                        y_ref_err = None
                    else:
                        # some NaNs could exist (err is maximal error of all data points)
                        y_ref_err[np.isnan(y_ref_err)] = np.nanmax(y_ref_err)

                # remove zero values for relative errors (no inf residuals)
                if self.residual_type == ResidualType.RELATIVE_RESIDUALS:
                    nonzero_mask = y_ref != 0.0
                    if not np.all(nonzero_mask):
                        logger.debug(
                            f"Zero (0.0) values in y data in experiment '{sid}' "
                            f"mapping '{mapping_id}' removed: {y_ref}"
                        )
                        x_ref = x_ref[nonzero_mask]
                        if y_ref_err is not None:
                            y_ref_err = y_ref_err[nonzero_mask]
                        y_ref = y_ref[nonzero_mask]

                # remove NaN from y-data
                nonnan_mask = ~np.isnan(y_ref)
                if not np.all(nonnan_mask):
                    logger.debug(
                        f"Removing NaN values in '{sid}:{mapping_id}' y data: {y_ref}"
                    )
                x_ref = x_ref[nonnan_mask]
                y_ref = y_ref[nonnan_mask]
                if y_ref_err is not None:
                    y_ref_err = y_ref_err[nonnan_mask]

                # at this point all x_ref, y_ref and y_ref_err (if not None) should be numerical
                if np.any(~np.isfinite(x_ref)):
                    raise ValueError(
                        f"{fit_experiment}.{mapping_id}: NaN or INF in x_ref: {x_ref}"
                    )
                if np.any(~np.isfinite(y_ref)):
                    raise ValueError(
                        f"{fit_experiment}.{mapping_id}: NaN or INF in y_ref: {y_ref}"
                    )
                if y_ref_err is not None:
                    if np.any(~np.isfinite(y_ref_err)):
                        raise ValueError(
                            f"{fit_experiment}.{mapping_id}: NaN or INF in y_ref_err: {y_ref_err}"
                        )

                # calculate local weights based on errors
                weights = np.ones_like(y_ref)  # local weights are by default 1.0

                if self.weighting_local != WeightingLocalType.NO_WEIGHTING:
                    if y_ref_err is not None:
                        if (
                            self.weighting_local
                            == WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING
                        ):
                            weights = (
                                1.0 / y_ref_err
                            )  # the larger the error, the smaller the weight
                            weights = weights / np.max(
                                weights
                            )  # normalize maximal weight to 1.0, weights in (0, 1]
                        elif (
                            self.weighting_local
                            == WeightingLocalType.RELATIVE_ONE_OVER_WEIGHTING
                        ):
                            weights = (
                                y_ref / y_ref_err
                            )  # the larger the error, the smaller the weight
                        else:
                            raise ValueError(
                                f"Local weighting not supported: {self.weighting_local}"
                            )
                    else:
                        logger.warning(
                            f"'{sid}.{mapping_id}': Using '{self.weighting_local}' with "
                            f"no errors in reference data."
                        )

                if False:
                    print("-" * 80)
                    print(f"{fit_experiment}.{mapping_id}")
                    print(f"weights: {weights}")
                    print(f"y_ref: {y_ref}")
                    print(f"y_ref_err: {y_ref_err}")

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
                self.y_errors_type.append(y_ref_err_type)
                self.weights_local.append(weights)
                self.weights_global_user.append(weight_global_user)

    def set_simulator(self, simulator):
        """Sets the simulator on the runner and the experiments.

        :param simulator:
        :return:
        """
        self.runner.set_simulator(simulator)

    def report(self, output_path=None):
        """Print and write report."""
        info = str(self)
        print(info)
        if output_path is not None:
            filepath = output_path / "00_fit_problem.txt"
            with open(filepath, "w") as fout:
                fout.write(info)

    def optimize(
        self,
        size=10,
        seed=None,
        verbose=False,
        optimizer: OptimizerType = OptimizerType.LEAST_SQUARE,
        sampling: SamplingType = SamplingType.UNIFORM,
        weighting_local: WeightingLocalType = WeightingLocalType.NO_WEIGHTING,
        residual_type: ResidualType = ResidualType.ABSOLUTE_RESIDUALS,
        variable_step_size=True,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-8,
        **kwargs,
    ) -> Tuple[List[scipy.optimize.OptimizeResult], List]:
        """Run parameter optimization"""
        # additional settings for optimization
        self.weighting_local = weighting_local
        self.residual_type = residual_type
        self.variable_step_size = variable_step_size
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

        if optimizer == OptimizerType.LEAST_SQUARE:
            # initial value samples for local optimizer
            x_samples = create_samples(
                parameters=self.parameters,
                size=size,
                sampling=sampling,
                seed=seed,
            )
        else:
            if seed is not None:
                np.random.seed(seed)

        fits = []
        trajectories = []
        for k in range(size):
            if optimizer == OptimizerType.LEAST_SQUARE:
                x0 = x_samples.values[k, :]
            else:
                x0 = None
            if verbose:
                print(f"[{k+1}/{size}] x0={x0}")
            fit, trajectory = self._optimize_single(
                x0=x0, optimizer=optimizer, **kwargs
            )
            if verbose:
                print("\t{:8.4f} [s]".format(fit.duration))

            fits.append(fit)
            trajectories.append(trajectory)
        return fits, trajectories

    @timeit
    def _optimize_single(
        self, x0=None, optimizer=OptimizerType.LEAST_SQUARE, **kwargs
    ) -> Tuple[scipy.optimize.OptimizeResult, List]:
        """Runs single optimization with x0 start values.

        :param x0: parameter start vector (important for deterministic optimizers)
        :param optimizer: optimization algorithm and method
        :param kwargs:
        :return:
        """
        if x0 is None:
            x0 = self.x0

        # logarithmic parameters for optimizer
        x0log = np.log10(x0)

        self._trajectory = []
        if optimizer == OptimizerType.LEAST_SQUARE:
            # scipy least square optimizer
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            ts = time.time()
            try:
                boundslog = [
                    np.log10([p.lower_bound for p in self.parameters]),
                    np.log10([p.upper_bound for p in self.parameters]),
                ]
                if "method" in kwargs and kwargs["method"] == "lm":
                    # no bounds supported on lm
                    logger.warning("No bounds on Levenberg-Marquardt optimizations")
                    opt_result = optimize.least_squares(
                        fun=self.residuals, x0=x0log, **kwargs
                    )
                else:
                    opt_result = optimize.least_squares(
                        fun=self.residuals, x0=x0log, bounds=boundslog, **kwargs
                    )
            except RuntimeError as err:
                logger.error(f"RuntimeError in ODE integration (optimize): {err}")
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0log
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            opt_result.x = np.power(10, opt_result.x)
            return opt_result, deepcopy(self._trajectory)

        elif optimizer == OptimizerType.DIFFERENTIAL_EVOLUTION:
            # scipy differential evolution
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
            ts = time.time()
            try:
                de_bounds_log = [
                    (np.log10(p.lower_bound), np.log10(p.upper_bound))
                    for k, p in enumerate(self.parameters)
                ]
                opt_result = optimize.differential_evolution(
                    func=self.cost_least_square, bounds=de_bounds_log, **kwargs
                )
            except RuntimeError as err:
                logger.error(f"RuntimeError in ODE integration (optimize): {err}")
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0log
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            opt_result.cost = self.cost_least_square(opt_result.x)
            opt_result.x = np.power(10, opt_result.x)
            return opt_result, deepcopy(self._trajectory)

        else:
            raise ValueError(f"optimizer is not supported: {optimizer}")

    def cost_least_square(self, xlog):
        res_weighted = self.residuals(xlog)
        return 0.5 * np.sum(np.power(res_weighted, 2))

    def residuals(self, xlog, complete_data=False):
        """Calculates residuals for given parameter vector.

        :param x: logarithmic parameter vector
        :param complete_data: boolean flag to return additional information
        :return: vector of weighted residuals
        """
        # Necessary to work in logarithmic parameter space to account for xtol
        # in largely varying parameters
        # see https://github.com/scipy/scipy/issues/7632
        # print(f"\t{xlog}")
        x = np.power(10, xlog)
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

            # set model in simulator
            simulator.set_model(model=self.models[k])

            # print(simulator.r.integrator)
            simulator.set_timecourse_selections(selections=self.selections[k])

            # FIXME: normalize simulations and parameters once outside of loop
            simulation = self.simulations[k]  # type: TimecourseSim
            simulation.normalize(udict=simulator.udict, ureg=simulator.ureg)

            # run simulation
            # logger.warning(f"Running simulation: {k} - {self.experiment_keys[k]} - {mapping_id}")

            try:
                df = simulator._timecourses([simulation])[0]

                # interpolation of simulation results
                f = interpolate.interp1d(
                    x=df[self.xid_observable[k]],
                    y=df[self.yid_observable[k]],
                    copy=False,
                    assume_sorted=True,
                )
                y_obsip = f(self.x_references[k])

                # calculate absolute & relative residuals
                res_abs = y_obsip - self.y_references[k]

            except RuntimeError as err:
                # something went wrong in the integration (setting high residuals & cost)
                logger.error(
                    f"RuntimeError in ODE integration (residuals for {x}): {err}"
                )
                res_abs = 5.0 * self.y_references[k]  # total error

            res_abs_normed = res_abs / self.y_references[k].mean()
            with np.errstate(invalid="ignore"):
                res_rel = res_abs / self.y_references[k]
            # no cost contribution of zero values
            res_rel[np.isnan(res_rel)] = 0
            res_rel[np.isinf(res_rel)] = 0

            # select correct residuals
            if self.residual_type == ResidualType.ABSOLUTE_RESIDUALS:
                res = res_abs
            elif self.residual_type == ResidualType.ABSOLUTE_NORMED_RESIDUALS:
                res = res_abs_normed
            elif self.residual_type == ResidualType.RELATIVE_RESIDUALS:
                res = res_rel

            # apply local weighting & user defined weighting (in the cost function the weighted residuals are squared)
            # sum(w_i * r_i^2) = sum((w_i^0.5*r_i)^2)
            resw = (
                res
                * np.sqrt(self.weights_local[k])
                * np.sqrt(self.weights_global_user[k])
            )  # FIXME: handle square of global weights correctly
            parts.append(resw)

            # if False:
            #    print("residuals:", self.experiment_keys[k], 'mapping_id:', mapping_id, res)
            #    print("residuals weighted:", self.experiment_keys[k], 'mapping_id:', mapping_id, resw)

            # for post_processing
            if complete_data:

                residual_data["x_obs"].append(df[self.xid_observable[k]])
                residual_data["y_obs"].append(df[self.yid_observable[k]])
                residual_data["y_obsip"].append(y_obsip)
                residual_data["residuals"].append(res)
                residual_data["residuals_weighted"].append(resw)
                residual_data["res_abs"].append(res_abs)
                residual_data["res_abs_normed"].append(res_abs_normed)
                residual_data["res_rel"].append(res_rel)
                # FIXME: this depends on loss function
                residual_data["cost"].append(0.5 * np.sum(np.power(resw, 2)))

        if complete_data:
            return residual_data
        else:
            res_all = np.concatenate(parts)
            # store the local step
            self._trajectory.append((deepcopy(x), 0.5 * np.sum(np.power(res_all, 2))))
            return res_all

    # --------------------------
    # Plotting
    # --------------------------
    @timeit
    def plot_costs(
        self, x, xstart=None, output_path: Path = None, show_plots: bool = True
    ):
        """Plots bar diagram of costs for set of residuals

        :param res_data_start:
        :param res_data_fit:
        :param filepath:
        :return:
        """
        if xstart is None:
            xstart = self.x0

        res_data_start = self.residuals(xlog=np.log10(xstart), complete_data=True)
        res_data_fit = self.residuals(xlog=np.log10(x), complete_data=True)

        data = []
        types = ["initial", "fit"]

        for k in range(len(self.mapping_keys)):
            for kdata, res_data in enumerate([res_data_start, res_data_fit]):

                data.append(
                    {
                        "id": f"{self.experiment_keys[k]}_{self.mapping_keys[k]}",
                        "experiment": self.experiment_keys[k],
                        "mapping": self.mapping_keys[k],
                        "cost": res_data["cost"][k],
                        "type": types[kdata],
                    }
                )
        cost_df = pd.DataFrame(
            data, columns=["id", "experiment", "mapping", "cost", "type"]
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        sns.set_color_codes("pastel")
        sns.barplot(ax=ax, x="cost", y="id", hue="type", data=cost_df)
        ax.set_xscale("log")
        if show_plots:
            plt.show()
        if output_path:
            filepath = output_path / "03_costs_mappings.svg"
            fig.savefig(filepath, bbox_inches="tight")

            tsv_path = output_path / "03_costs_mappings.tsv"
            cost_df.to_csv(tsv_path, sep="\t", index=False)

        return cost_df

    @timeit
    def plot_fits(self, x, output_path: Path = None, show_plots: bool = True):
        """Plot fitted curves with experimental data.

        Overview of all fit mappings.

        :param x: parameters to evaluate
        :return:
        """
        n_plots = len(self.mapping_keys)
        fig, axes = plt.subplots(
            nrows=n_plots, ncols=2, figsize=(10, 5 * n_plots), squeeze=False
        )

        # residual data and simulations of optimal paraemters
        res_data = self.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.mapping_keys):

            # global reference data
            sid = self.experiment_keys[k]
            mapping_id = self.mapping_keys[k]
            x_ref = self.x_references[k]
            y_ref = self.y_references[k]
            y_ref_err = self.y_errors[k]
            y_ref_err_type = self.y_errors_type[k]
            x_id = self.xid_observable[k]
            y_id = self.yid_observable[k]

            for ax in axes[k]:
                ax.set_title(f"{sid} {mapping_id}")
                ax.set_xlabel(x_id)
                ax.set_ylabel(y_id)

                # calculated data in residuals
                x_obs = res_data["x_obs"][k]
                y_obs = res_data["y_obs"][k]

                # FIXME: add residuals

                # plot data
                if y_ref_err is None:
                    ax.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label=f"reference_data ± {y_ref_err_type}",
                    )
                # plot simulation
                ax.plot(x_obs, y_obs, "-", color="blue", label="observable")
                ax.legend()

            axes[k][1].set_yscale("log")
            axes[k][1].set_ylim(bottom=0.3 * np.nanmin(y_ref))

        if show_plots:
            plt.show()
        if output_path is not None:
            fig.savefig(output_path / f"00_fits_{self.opid}.svg", bbox_inches="tight")

    @timeit
    def plot_residuals(
        self, x, xstart=None, output_path: Path = None, show_plots: bool = True
    ):
        """Plot residual data.

        :param res_data_start: initial residual data
        :return:
        """
        if xstart is None:
            xstart = self.x0

        titles = ["initial", "fit"]
        res_data_start = self.residuals(xlog=np.log10(xstart), complete_data=True)
        res_data_fit = self.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.mapping_keys):
            fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(
                nrows=3, ncols=2, figsize=(10, 10)
            )

            axes = [(a1, a3, a5), (a2, a4, a6)]
            if titles is None:
                titles = ["Initial", "Fit"]

            # global reference data
            sid = self.experiment_keys[k]
            mapping_id = self.mapping_keys[k]
            weights = self.weights_local[k]
            x_ref = self.x_references[k]
            y_ref = self.y_references[k]
            y_ref_err = self.y_errors[k]
            x_id = self.xid_observable[k]
            y_id = self.yid_observable[k]

            for kdata, res_data in enumerate([res_data_start, res_data_fit]):
                ax1, ax2, ax3 = axes[kdata]
                title = titles[kdata]

                # calculated data in residuals

                x_obs = res_data["x_obs"][k]
                y_obs = res_data["y_obs"][k]
                y_obsip = res_data["y_obsip"][k]

                res = res_data["residuals"][k]
                res_weighted = res_data["residuals_weighted"][k]
                res_abs = res_data["res_abs"][k]
                res_rel = res_data["res_rel"][k]

                cost = res_data["cost"][k]

                for ax in (ax1, ax2, ax3):
                    ax.axhline(y=0, color="black")
                    ax.set_ylabel(y_id)
                ax3.set_xlabel(x_id)

                if y_ref_err is None:
                    ax1.plot(x_ref, y_ref, "s", color="black", label="reference_data")
                else:
                    ax1.errorbar(
                        x_ref,
                        y_ref,
                        yerr=y_ref_err,
                        marker="s",
                        color="black",
                        label="reference_data",
                    )

                ax1.plot(x_obs, y_obs, "-", color="blue", label="observable")
                ax1.plot(x_ref, y_obsip, "o", color="blue", label="interpolation")
                for ax in (ax1, ax2):
                    ax.plot(x_ref, res_abs, "o", color="darkorange", label="obs-ref")
                ax1.fill_between(
                    x_ref,
                    res_abs,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkorange",
                    label="__nolabel__",
                )

                ax2.plot(
                    x_ref,
                    res_weighted,
                    "o",
                    color="darkgreen",
                    label="weighted residuals",
                )
                ax2.fill_between(
                    x_ref,
                    res_weighted,
                    np.zeros_like(res_weighted),
                    alpha=0.4,
                    color="darkgreen",
                    label="__nolabel__",
                )

                res_weighted2 = np.power(res_weighted, 2)
                ax3.plot(
                    x_ref,
                    res_weighted2,
                    "o",
                    color="darkred",
                    label="(weighted residuals)^2",
                )
                ax3.fill_between(
                    x_ref,
                    res_weighted2,
                    np.zeros_like(res),
                    alpha=0.4,
                    color="darkred",
                    label="__nolabel__",
                )

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
                    # plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.set_ylabel("y")
                    # ax.set_yscale("log")
                    ax.legend()

            # adapt axes
            if res_data_fit is not None:
                for axes in [(a1, a2), (a3, a4), (a5, a6)]:
                    ax1, ax2 = axes
                    ylim1 = ax1.get_ylim()
                    ylim2 = ax2.get_ylim()
                    # for ax in axes:
                    #    ax.set_ylim([min(ylim1[0], ylim2[0]), max(ylim1[1],ylim2[1])])
            if show_plots:
                plt.show()
            if output_path is not None:
                fig.savefig(
                    output_path / f"06_residuals_{sid}_{mapping_id}.svg",
                    bbox_inches="tight",
                )
