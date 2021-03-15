"""Optimization of parameter fitting problem."""

import logging
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate, optimize

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.fit.objects import FitExperiment, FitMapping, FitParameter
from sbmlsim.fit.options import (
    OptimizationAlgorithmType,
    ResidualType,
    WeightingPointsType, WeightingCurvesType,
)
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.serialization import ObjectJSONEncoder, to_json
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


class OptimizationProblem(ObjectJSONEncoder):
    """Parameter optimization problem."""

    def __init__(
        self,
        opid: str,
        fit_experiments: Collection[FitExperiment],
        fit_parameters: Collection[FitParameter],
        base_path: Path = None,
        data_path: Path = None,
    ):
        """Optimization problem.

        The problem must be pickable for parallelization !
        So initialize must be run to create the non-pickable instances.

        :param opid: id for optimization problem
        :param fit_experiments:
        :param fit_parameters:
        """
        super(OptimizationProblem, self).__init__()
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

        # set in initialization
        self.runner: Optional[ExperimentRunner] = None
        self.residual_type: Optional[ResidualType] = None
        self.weighting_curves: Optional[WeightingCurvesType] = None
        self.weighting_points: Optional[WeightingPointsType] = None

        self.experiment_keys: List[str] = []
        self.mapping_keys: List[str] = []
        self.xid_observable: List[str] = []
        self.yid_observable: List[str] = []
        self.x_references: List[Any] = []
        self.y_references: List[Any] = []
        self.y_errors: List[Any] = []
        self.y_errors_type: List[str] = []
        self.weights: List[
            Any
        ] = []  # total weights for points (data points and curve weights)
        self.weights_points: List[Any] = []  # weights for data points based on errors
        self.weights_curves: List[Any] = []  # user defined weights per mapping/curve

        self.models: List[Any] = []
        self.xmodel: np.ndarray = np.empty(shape=(len(self.pids)))
        self.simulations: List[Any] = []
        self.selections: List[Any] = []

    def __repr__(self) -> str:
        """Get representation."""
        return f"<OptimizationProblem: {self.opid}>"

    def __str__(self) -> str:
        """Get string representation.

        This can be run before initialization.
        """
        info = [
            "-" * 80,
            f"{self.__class__.__name__}: {self.opid}",
            "-" * 80,
            "Experiments",
        ]
        info.extend([f"\t{e}" for e in self.fit_experiments])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        return "\n".join(info)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = dict()
        for key in ["opid", "fit_experiments", "parameters", "base_path", "data_path"]:
            d[key] = self.__dict__[key]
        return d

    def to_json(self, path: Optional[Path] = None) -> Union[str, Path]:
        """Store OptimizationResult as json.

        Uses the to_dict method.
        """
        return to_json(object=self, path=path)

    def report(self, path: Path = None, print_output: bool = True) -> str:
        """Print and write report.

        Can only be called after initialization.
        """
        core_info = self.__str__()
        all_info = [
            core_info,
            "Settings",
            f"\tfitting_type: {self.fitting_strategy}",
            f"\tresidual_type: {self.residual_type}",
            f"\tweighting local: {self.weighting_points}",
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
        residual_type: Optional[ResidualType],
        weighting_curves: Optional[WeightingCurvesType],
        weighting_points: Optional[WeightingPointsType],
        variable_step_size: bool = True,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-6,
    ) -> None:
        """Initialize Optimization problem.

        Performs precalculations, resolving data, calculating weights.
        Creates and attaches simulator for the given problem.

        :param residual_type: handling of residuals
        :param weighting_curves: weighting of curves (fit mappings)
        :param weighting_points: weighting of points
        :param absolute_tolerance: absolute tolerance of simulator
        :param relative_tolerance: relative tolerance of simulator
        :param variable_step_size: use variable step size in solver
        """
        if residual_type is None:
            raise ValueError("'residual_type' is required.")
        if weighting_curves is None:
            raise ValueError("'weighting_curves' is required.")
        if weighting_points is None:
            raise ValueError("'weighting_points' is required.")

        self.residual_type = residual_type
        self.weighting_curves = weighting_curves
        self.weighting_points = weighting_points

        # Create experiment runner (loads the experiments & all models)
        exp_classes = {fit_exp.experiment_class for fit_exp in self.fit_experiments}

        self.runner = ExperimentRunner(
            experiment_classes=exp_classes,
            base_path=self.base_path,
            data_path=self.data_path,
        )

        # Collect information for simulations
        fit_exp: Callable
        for fit_experiment in self.fit_experiments:

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
                    weight_curve_user = mapping.weight
                    fit_experiment.weights[k] = weight_curve_user
                    if weight_curve_user is None:
                        raise ValueError(
                            f"If `use_mapping_weights` is set on a FitExperiment "
                            f"then all mappings must have a weight. But "
                            f"weight '{weight_curve_user}' in {mapping}."
                        )
                else:
                    weight_curve_user = fit_experiment.weights[k]

                if weight_curve_user < 0:
                    raise ValueError(
                        f"Mapping weights must be positive but "
                        f"weight '{mapping.weight}' in {mapping}"
                    )

                task_id = mapping.observable.task_id
                task = sim_experiment._tasks[task_id]
                model: RoadrunnerSBMLModel = sim_experiment._models[task.model_id]
                simulation = sim_experiment._simulations[task.simulation_id]

                if not isinstance(simulation, TimecourseSim):
                    raise ValueError(
                        f"Only TimecourseSims supported in fitting: '{simulation}"
                    )

                # observable units
                obs_xid = mapping.observable.x.index
                obs_yid = mapping.observable.y.index
                obs_x_unit = model.uinfo[obs_xid]
                obs_y_unit = model.uinfo[obs_yid]

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

                # FIXME: this should all happen at the end in residual calculation
                if self.residual_type == ResidualType.ABSOLUTE_CHANGES_BASELINE:
                    # Use absolute changes to baseline, which is the first point
                    y_ref = (y_ref - y_ref[0])

                # Use errors for weighting (tries SD and falls back on SE)
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

                # --- WEIGHTS ---

                # weight points
                weight_points: np.ndarray
                if self.weighting_points == WeightingPointsType.NO_WEIGHTING:
                    # local weights are by default 1.0
                    weight_points = np.ones_like(y_ref)
                elif self.weighting_points == WeightingPointsType.ERROR_WEIGHTING:
                    if y_ref_err is None:
                        logger.warning(
                            f"'{sid}.{mapping_id}': Using '{self.weighting_points}' "
                            f"with no errors in reference data."
                        )
                    else:
                        # the larger the error, the smaller the weight
                        weight_points = 1.0 / y_ref_err

                # curve weight
                weight_curve: float
                if self.weighting_curves == WeightingCurvesType.NO_WEIGHTING:
                    weight_curve = weight_curve_user
                elif self.weighting_curves == WeightingCurvesType.POINTS:
                    weight_curve = weight_curve_user / len(weight_points)
                elif self.weighting_curves == WeightingCurvesType.MEAN:
                    weight_curve = weight_curve_user / np.mean(y_ref)
                elif self.weighting_curves == WeightingCurvesType.MEAN_AND_POINTS:
                    weight_curve = weight_curve_user / np.mean(y_ref) / len(
                        weight_points)

                # total weight
                # apply local weighting & user defined weighting
                # (in the cost function the weighted residuals are squared)
                # sum(w_i * r_i^2) = sum((w_i^0.5*r_i)^2)
                weight = np.sqrt(weight_curve * weight_points)
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
                # weights
                self.weights.append(weight)
                self.weights_points.append(weight_points)
                self.weights_curves.append(weight_curve)

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
            # print(fit_experiment)

        # set simulator instance with arguments
        simulator = SimulatorSerial(
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            variable_step_size=variable_step_size,
        )
        self.set_simulator(simulator)

    def set_simulator(self, simulator):
        """Set the simulator on the runner and the experiments.

        :param simulator:
        :return:
        """
        self.runner.set_simulator(simulator)

    def optimize(
        self,
        size: Optional[int] = 5,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        sampling: SamplingType = SamplingType.UNIFORM,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[optimize.OptimizeResult], List]:
        """Run parameter optimization.

        To change the weighting or handling of residuals reinitialize the optimization
        algorithm.
        """
        # create samples
        x_samples: pd.DataFrame
        if algorithm == OptimizationAlgorithmType.LEAST_SQUARE:
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
            if algorithm == OptimizationAlgorithmType.LEAST_SQUARE:
                x0 = x_samples.values[k, :]
            else:
                x0 = None

            logger.debug(f"[{k+1}/{size}] x0={x0}")
            fit, trajectory = self._optimize_single(
                x0=x0, algorithm=algorithm, **kwargs
            )
            logger.debug("\t{:8.4f} [s]".format(fit.duration))

            fits.append(fit)
            trajectories.append(trajectory)
        return fits, trajectories

    @timeit
    def _optimize_single(
        self,
        x0: np.ndarray = None,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        **kwargs,
    ) -> Tuple[scipy.optimize.OptimizeResult, List]:
        """Run single optimization with x0 start values.

        :param x0: parameter start vector (important for deterministic optimizers)
        :param algorithm: optimization algorithm and method
        :param kwargs:
        :return:
        """
        # FIXME: this should not be necessary, handle outside
        if x0 is None:
            x0 = self.x0

        # logarithmic parameters for optimizer
        x0log = np.log10(x0)

        self._trajectory = []
        if algorithm == OptimizationAlgorithmType.LEAST_SQUARE:
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
                    opt_result = scipy.optimize.least_squares(
                        fun=self.residuals, x0=x0log, **kwargs
                    )
                else:
                    opt_result = scipy.optimize.least_squares(
                        fun=self.residuals, x0=x0log, bounds=boundslog, **kwargs
                    )
            except RuntimeError as err:
                logger.error(
                    f"RuntimeError in ODE integration (optimize) for '{self.pids} = {x0}': \n{err}"
                )
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0log
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            opt_result.x = np.power(10, opt_result.x)
            return opt_result, deepcopy(self._trajectory)

        elif algorithm == OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION:
            # scipy differential evolution
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
            ts = time.time()
            try:
                de_bounds_log = [
                    (np.log10(p.lower_bound), np.log10(p.upper_bound))
                    for k, p in enumerate(self.parameters)
                ]
                opt_result = scipy.optimize.differential_evolution(
                    func=self.cost_least_square, bounds=de_bounds_log, **kwargs
                )
            except RuntimeError as err:
                logger.error(
                    f"RuntimeError in ODE integration (optimize) for '{self.pids} = {x0}': \n{err}"
                )
                opt_result = RuntimeErrorOptimizeResult()
                opt_result.x = x0log
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            opt_result.cost = self.cost_least_square(opt_result.x)
            opt_result.x = np.power(10, opt_result.x)
            return opt_result, deepcopy(self._trajectory)

        else:
            raise ValueError(f"optimizer is not supported: {algorithm}")

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
        simulator: SimulatorSerial = self.runner.simulator
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
            simulation: TimecourseSim = self.simulations[k]
            simulation.normalize(uinfo=simulator.uinfo)

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
                if (
                    self.fitting_strategy
                    == FittingStrategyType.ABSOLUTE_CHANGES_BASELINE
                ):
                    y_obsip = y_obsip - y_obsip[0]

                # calculate absolute & relative residuals
                res_abs = y_obsip - self.y_references[k]

            except RuntimeError as err:
                # error in integration (setting high residuals & cost)
                logger.error(
                    f"RuntimeError in ODE integration ('{self.pids} = {x}'): \n{err}"
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
