"""Optimization of parameter fitting problem."""

import logging
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Set, Sized, Tuple

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.fit.objects import FitExperiment, FitMapping, FitParameter
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import DimensionalityError
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


@dataclass
class RuntimeErrorOptimizeResult:
    """Error in opimization."""

    status: str = "-1"
    success: bool = False
    duration: float = -1.0
    cost: float = np.Inf
    optimality: float = np.Inf


class OptimizerType(Enum):
    """Type of optimization.

    Least square
    """

    LEAST_SQUARE = 1
    DIFFERENTIAL_EVOLUTION = 2


class FittingType(Enum):
    """Type of fitting (absolute changes or relative changes to baseline).

    Decides how to fit the data. If the various datasets have large offsets
    a fitting of the relative changes to baseline can work.
    As baseline the simulations should contain a pre-simulation.
    """

    ABSOLUTE_VALUES = 1  # absolute values are use
    ABSOLUTE_CHANGES_BASELINE = 2  # relative values in respect to baseline are used
    RELATIVE_CHANGES_BASELINE = 3  # relative values in respect to baseline are used


class WeightingLocalType(Enum):
    """Weighting of the data points within a single fit mapping.

    This decides how the data points within a single fit mapping are
    weighted. One can account for the errors or not.
    """

    NO_WEIGHTING = 1  # data points are weighted equally
    ABSOLUTE_ONE_OVER_WEIGHTING = 2  # data points are weighted as 1/(error-min(error))
    RELATIVE_ONE_OVER_WEIGHTING = 3  # FIXME: check that this is working and documented


class ResidualType(Enum):
    """Handling of the residuals.

    How are the residuals calculated? Are the absolute residuals used,
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
        fit_experiments: Collection[FitExperiment],
        fit_parameters: Collection[FitParameter],
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
        self.opid: str = opid
        # self.fit_experiments = FitExperiment.reduce(fit_experiments)
        self.fit_experiments = fit_experiments
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

    def __repr__(self) -> str:
        """Get representation."""
        return f"<OptimizationProblem: {self.opid}>"

    def __str__(self) -> str:
        """Get string representation.

        This can be run before initialization.
        """
        info = []
        info.append("-" * 80)
        info.append(f"{self.__class__.__name__}: {self.opid}")
        info.append("-" * 80)
        info.append("Experiments")
        info.extend([f"\t{e}" for e in self.fit_experiments])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        return "\n".join(info)

    def report(self, path: Path = None, print_output: bool = True) -> str:
        """Print and write report.

        Can only be called after initialization.
        """
        core_info = self.__str__()
        all_info = [
            core_info,
            "Settings",
            f"\tfitting_type: {self.fitting_type}",
            f"\tresidual_type: {self.residual_type}",
            f"\tweighting local: {self.weighting_local}",
            "Data",
        ]
        for key in [
            "experiment_keys",
            "mapping_keys",
            "xid_observable",
            "yid_observable",
            "x_references",
            "y_references",
            "y_errors",
            "y_errors_type",
            "weights",
            "weights_points",
            "weights_curve",
        ]:
            all_info.append(f"\t{key}: {str(getattr(self, key))}")

        info = "\n".join(all_info)

        if print_output:
            print(info)
        if path:
            with open(path, "w") as f:
                f.write(info)
        return info

    def initialize(
        self,
        fitting_type: FittingType,
        weighting_local: WeightingLocalType,
        residual_type: ResidualType,
    ):
        """Initialize Optimization problem."""
        # weighting in fitting and handling of residuals
        if weighting_local is None:
            raise ValueError("'weighting_local' is required.")
        if residual_type is None:
            raise ValueError("'residual_type' is required.")
        if fitting_type is None:
            logger.warning(
                "No FittingType provided for fitting, defaulting to absolute"
            )
            fitting_type = FittingType.ABSOLUTE_VALUES

        logger.debug(f"fitting_type: {fitting_type}")
        logger.debug(f"weighting_local: {weighting_local}")
        logger.debug(f"residual_type: {residual_type}")

        self.fitting_type = fitting_type
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
        self.weights = []  # total weights for points (errors + user)
        self.weights_points = []  # weights for data points based on errors
        self.weights_curve = []  # user defined weights per mapping/curve

        self.models = []
        self.xmodel = np.empty(shape=(len(self.pids)))
        self.simulations = []
        self.selections = []

        # Collect information for simulations
        for fit_experiment in self.fit_experiments:  # type: FitExperiment

            # get simulation experiment
            sid = fit_experiment.experiment_class.__name__
            sim_experiment = self.runner.experiments[sid]

            # FIXME: selections should be based on fit mappings
            selections_set: Set[str] = set()
            for d in sim_experiment._data.values():  # type: Data
                if d.is_task():
                    selections_set.add(d.index)
            selections: List[str] = list(selections_set)

            # use all fit_mappings if None are provided
            if fit_experiment.mappings is None:
                fit_experiment.mappings = list(sim_experiment._fit_mappings.keys())
                fit_experiment.weights = [1.0] * len(fit_experiment.mappings)

            # collect information for single mapping
            for k, mapping_id in enumerate(fit_experiment.mappings):
                # sanity checks
                if mapping_id not in sim_experiment._fit_mappings:
                    raise ValueError(
                        f"Mapping key '{mapping_id}' not defined in "
                        f"SimulationExperiment\n"
                        f"{sim_experiment}\n"
                        f"{fit_experiment}"
                    )

                mapping: FitMapping = sim_experiment._fit_mappings[mapping_id]

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

                # get weight for curve
                if fit_experiment.use_mapping_weights:
                    # use provided mapping weights
                    weight_curve = mapping.weight
                    fit_experiment.weights[k] = weight_curve
                    if weight_curve is None:
                        raise ValueError(
                            f"If `use_mapping_weights` is set on a FitExperiment "
                            f"then all mappings must have a weight. But "
                            f"weight '{weight_curve}' in {mapping}."
                        )
                else:
                    weight_curve = fit_experiment.weights[k]

                if weight_curve < 0:
                    raise ValueError(
                        f"Mapping weights must be positive but "
                        f"weight '{mapping.weight}' in {mapping}"
                    )

                task_id = mapping.observable.task_id
                task = sim_experiment._tasks[task_id]
                model = sim_experiment._models[
                    task.model_id
                ]  # type: RoadrunnerSBMLModel
                simulation = sim_experiment._simulations[task.simulation_id]

                if not isinstance(simulation, TimecourseSim):
                    raise ValueError(
                        f"Only TimecourseSims supported in fitting: '{simulation}"
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
                        f"{sid}.{mapping_id}: Unit conversion fails for '{data_ref.x}' "
                        f"to '{obs_x_unit}"
                    )
                    raise e
                try:
                    data_ref.y = data_ref.y.to(obs_y_unit)
                except DimensionalityError as e:
                    logger.error(
                        f"{sid}.{mapping_id}: Unit conversion fails for '{data_ref.y}' "
                        f"to '{obs_y_unit}'."
                    )
                    raise e
                x_ref = data_ref.x.magnitude
                y_ref = data_ref.y.magnitude

                if self.fitting_type == FittingType.ABSOLUTE_CHANGES_BASELINE:
                    # Use absolute changes to baseline, which is the first point
                    y_ref = (
                        y_ref - y_ref[0]
                    )  # FIXME: will break if first data point is bad;

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

                # handle missing data (0.0 and NaN)
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
                # FIXME: check how this works with the relative changes (due to 0.0 at first data point)
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

                # FIXME: relative_changes to baseline errors;

                # --- WEIGHTS ---
                # calculate local weights based on errors (i.e. weights for data
                # points
                # FIXME: the weights must be normalized for a mapping based on data points;
                # FIXME: also make sure that data points with error contribute
                # correctly compared to no errors
                weight_points = np.ones_like(y_ref)  # local weights are by default 1.0
                if self.weighting_local == WeightingLocalType.NO_WEIGHTING:
                    pass
                else:
                    if y_ref_err is not None:
                        if (
                            self.weighting_local
                            == WeightingLocalType.ABSOLUTE_ONE_OVER_WEIGHTING
                        ):
                            # the larger the error, the smaller the weight
                            weight_points = 1.0 / y_ref_err
                        elif (
                            self.weighting_local
                            == WeightingLocalType.RELATIVE_ONE_OVER_WEIGHTING
                        ):
                            # the larger the error, the smaller the weight
                            weight_points = y_ref / y_ref_err
                        else:
                            raise ValueError(
                                f"Local weighting not supported: {self.weighting_local}"
                            )
                    else:
                        logger.warning(
                            f"'{sid}.{mapping_id}': Using '{self.weighting_local}' with "
                            f"no errors in reference data."
                        )

                # normalize weights to mean=1.0 for given curve
                # this makes the weights comparable
                weight_points = weight_points / np.mean(weight_points)

                # apply local weighting & user defined weighting
                # (in the cost function the weighted residuals are squared)
                # sum(w_i * r_i^2) = sum((w_i^0.5*r_i)^2)
                weight = np.sqrt(weight_points * weight_curve)
                if np.any(weight < 0):
                    raise ValueError("Negative weights encountered.")

                # --- STORE INITIAL PARAMETERS ---
                # store initial model parameters
                for k, pid in enumerate(self.pids):
                    pid_value = model.r[pid]
                    if pid in model.changes:
                        try:
                            # model changes have units
                            pid_value = model.changes[pid].magnitude
                        except AttributeError:
                            pid_value = model.changes[pid]
                    self.xmodel[k] = pid_value

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
                self.weights_points.append(weight_points)
                self.weights_curve.append(weight_curve)
                self.weights.append(weight)

                # debug info
                if False:
                    print("-" * 80)
                    print(f"{fit_experiment}.{mapping_id}")
                    print(f"weight: {weight}")
                    print(f"weight_curve: {weight_curve}")
                    print(f"weight_points: {weight_points}")
                    print(f"y_ref: {y_ref}")
                    print(f"y_ref_err: {y_ref_err}")

            # Print mappings with calculated weights
            print(fit_experiment)

    def set_simulator(self, simulator):
        """Set the simulator on the runner and the experiments.

        :param simulator:
        :return:
        """
        self.runner.set_simulator(simulator)

    def optimize(
        self,
        size=10,
        seed=None,
        verbose=False,
        optimizer: OptimizerType = OptimizerType.LEAST_SQUARE,
        sampling: SamplingType = SamplingType.UNIFORM,
        fitting_type: FittingType = FittingType.ABSOLUTE_VALUES,
        weighting_local: WeightingLocalType = WeightingLocalType.NO_WEIGHTING,
        residual_type: ResidualType = ResidualType.ABSOLUTE_RESIDUALS,
        variable_step_size=True,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-8,
        **kwargs,
    ) -> Tuple[List[scipy.optimize.OptimizeResult], List]:
        """Run parameter optimization."""

        # additional settings for optimization
        self.fitting_type = fitting_type
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
        self, x0: np.ndarray = None, optimizer=OptimizerType.LEAST_SQUARE, **kwargs
    ) -> Tuple[scipy.optimize.OptimizeResult, List]:
        """Run single optimization with x0 start values.

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

    def cost_least_square(self, xlog: np.ndarray) -> float:
        """Get least square costs for parameters."""
        res_weighted = self.residuals(xlog)
        return 0.5 * np.sum(np.power(res_weighted, 2))

    def residuals(self, xlog: np.ndarray, complete_data=False):
        """Calculate residuals for given parameter vector.

        :param x: logarithmic parameter vector
        :param complete_data: boolean flag to return additional information
        :return: vector of weighted residuals
        """
        # Necessary to work in logarithmic parameter space to account for xtol
        # in largely varying parameters
        # see https://github.com/scipy/scipy/issues/7632
        # print(f"\t{xlog}")
        x = np.power(10, xlog)

        # FIXME: handle parts better
        parts = []
        if complete_data:
            residual_data = defaultdict(list)

        # simulate all mappings for all experiments
        simulator = self.runner.simulator  # type: SimulatorSerial
        Q_ = self.runner.Q_

        for k, _ in enumerate(self.mapping_keys):
            # update initial changes
            changes = {
                self.pids[ix]: Q_(value, self.punits[ix]) for ix, value in enumerate(x)
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
            try:
                # FIXME: just simulate at the requested timepoints with step
                df = simulator._timecourses([simulation])[0]

                # interpolation of simulation results
                f = interpolate.interp1d(
                    x=df[self.xid_observable[k]],
                    y=df[self.yid_observable[k]],
                    copy=False,
                    assume_sorted=True,
                )
                y_obsip = f(self.x_references[k])
                if self.fitting_type == FittingType.ABSOLUTE_CHANGES_BASELINE:
                    y_obsip = y_obsip - y_obsip[0]

                # calculate absolute & relative residuals
                res_abs = y_obsip - self.y_references[k]

            except RuntimeError as err:
                # something went wrong in the integration (setting high residuals & cost)
                logger.error(
                    f"RuntimeError in ODE integration (residuals for {x}): {err}"
                )
                res_abs = 5.0 * self.y_references[k]  # total error

            res_abs_normed = res_abs / self.y_references[k].mean()

            with np.errstate(divide="ignore", invalid="ignore"):
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

            resw = res * self.weights[k]
            parts.append(resw)

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
            # FIXME
            self._trajectory.append((deepcopy(x), 0.5 * np.sum(np.power(res_all, 2))))
            return res_all


class OptimizationAnalysis:
    """Class for creating plots and results."""

    def __init__(self, optimization_problem: OptimizationProblem):
        self.op = optimization_problem

    @staticmethod
    def _save_fig(fig, path: Path, show_plots: bool = True):
        if show_plots:
            plt.show()
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @timeit
    def plot_fits(self, x, path: Path = None, show_plots: bool = True):
        """Plot fitted curves with experimental data.

        Overview of all fit mappings.

        :param x: parameters to evaluate
        :return:
        """
        n_plots = len(self.op.mapping_keys)
        fig, axes = plt.subplots(
            nrows=n_plots, ncols=2, figsize=(10, 5 * n_plots), squeeze=False
        )

        # residual data and simulations of optimal paraemters
        res_data = self.op.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.op.mapping_keys):

            # global reference data
            sid = self.op.experiment_keys[k]
            mapping_id = self.op.mapping_keys[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            y_ref_err_type = self.op.y_errors_type[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

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

        self._save_fig(fig, path=path, show_plots=show_plots)

    @timeit
    def plot_residuals(self, x, output_path: Path = None, show_plots: bool = True):
        """Plot residual data.

        :param res_data_start: initial residual data
        :return:
        """
        titles = ["model", "fit"]
        res_data_start = self.op.residuals(
            xlog=np.log10(self.op.xmodel), complete_data=True
        )
        res_data_fit = self.op.residuals(xlog=np.log10(x), complete_data=True)

        for k, mapping_id in enumerate(self.op.mapping_keys):
            fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(
                nrows=3, ncols=2, figsize=(10, 10)
            )

            axes = [(a1, a3, a5), (a2, a4, a6)]
            if titles is None:
                titles = ["Model", "Fit"]

            # global reference data
            sid = self.op.experiment_keys[k]
            mapping_id = self.op.mapping_keys[k]
            # weights = self.op.weights_points[k]
            x_ref = self.op.x_references[k]
            y_ref = self.op.y_references[k]
            y_ref_err = self.op.y_errors[k]
            x_id = self.op.xid_observable[k]
            y_id = self.op.yid_observable[k]

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
                # res_rel = res_data["res_rel"][k]

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
                    # ylim1 = ax1.get_ylim()
                    # ylim2 = ax2.get_ylim()
                    # # for ax in axes:
                    # #    ax.set_ylim([min(ylim1[0], ylim2[0]), max(ylim1[1],ylim2[1])])

            if show_plots:
                plt.show()
            if output_path is not None:
                fig.savefig(
                    output_path / f"06_residuals_{sid}_{mapping_id}.svg",
                    bbox_inches="tight",
                )

    @timeit
    def plot_costs(self, x, path: Path = None, show_plots: bool = True) -> pd.DataFrame:
        """Plot cost function comparison.

        # FIXME: separate calculation of cost DataFrame
        """
        xparameters = {
            # model parameters
            "model": self.op.xmodel,
            # initial values of fit parameter
            "initial": self.op.x0,
            # provided parameters
            "fit": x,
        }
        data = []
        costs = {}
        for key, xpar in xparameters.items():
            res_data = self.op.residuals(xlog=np.log10(xpar), complete_data=True)
            costs[key] = res_data["cost"]
            for k, _ in enumerate(self.op.mapping_keys):
                data.append(
                    {
                        "id": f"{self.op.experiment_keys[k]}_{self.op.mapping_keys[k]}",
                        "experiment": self.op.experiment_keys[k],
                        "mapping": self.op.mapping_keys[k],
                        "cost": res_data["cost"][k],
                        "type": key,
                    }
                )

        cost_df = pd.DataFrame(
            data, columns=["id", "experiment", "mapping", "cost", "type"]
        )

        min_cost = np.min(
            [
                np.min(costs["fit"]),
                np.min(costs["model"]),
                np.min(costs["initial"]),
            ]
        )
        max_cost = np.max(
            [
                np.max(costs["fit"]),
                np.max(costs["model"]),
                np.max(costs["initial"]),
            ]
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        ax.plot(
            [min_cost * 0.5, max_cost * 2],
            [min_cost * 0.5, max_cost * 2],
            "--",
            color="black",
        )
        # ax.plot(costs["initial"], costs["fit"], linestyle="", marker="s", label="initial")
        ax.plot(
            costs["model"],
            costs["fit"],
            linestyle="",
            marker="o",
            label="model",
            color="black",
            markersize="10",
            alpha=0.8,
        )

        for k, exp_key in enumerate(self.op.experiment_keys):
            ax.annotate(
                exp_key,
                xy=(
                    costs["model"][k],
                    costs["fit"][k],
                ),
                fontsize="x-small",
                alpha=0.7,
            )

        ax.set_xlabel("initial cost")
        ax.set_ylabel("fit cost")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.set_xlim(min_cost * 0.5, max_cost * 2)
        ax.set_ylim(min_cost * 0.5, max_cost * 2)
        ax.legend()

        # sns.set_color_codes("pastel")
        # sns.barplot(ax=ax, x="cost", y="id", hue="type", data=cost_df)
        # ax.set_xscale("log")
        if show_plots:
            plt.show()
        if path:
            fig.savefig(path, bbox_inches="tight")

        return cost_df
