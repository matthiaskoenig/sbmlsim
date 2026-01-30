"""Optimization of parameter fitting problem."""

import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from pymetadata import log
from pymetadata.console import console
from scipy import interpolate, optimize

from sbmlsim.experiment import ExperimentRunner
from sbmlsim.fit.objects import FitExperiment, FitMapping, FitParameter
from sbmlsim.fit.options import (
    LossFunctionType,
    OptimizationAlgorithmType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.serialization import ObjectJSONEncoder, to_json
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import DimensionalityError
from sbmlsim.utils import timeit


logger = log.get_logger(__name__)


@dataclass
class RuntimeErrorOptimizeResult:
    """Error in opimization."""

    status: str = "-1"
    success: bool = False
    duration: float = -1.0
    cost: float = np.inf
    optimality: float = np.inf


class OptimizationProblem(ObjectJSONEncoder):
    """Parameter optimization problem."""

    def __init__(
        self,
        opid: str,
        fit_experiments: list[FitExperiment],
        fit_parameters: list[FitParameter],
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
        self.fit_experiments = []
        for fit_exp in fit_experiments:
            if fit_exp.exclude:
                logger.warning(f"FitExperiment excluded: {fit_exp}")
            else:
                self.fit_experiments.append(fit_exp)
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
        self.residual: Optional[ResidualType] = None
        self.weighting_curves: Optional[WeightingCurvesType] = None
        self.weighting_points: Optional[WeightingPointsType] = None

        self.experiment_keys: list[str] = []
        self.mapping_keys: list[str] = []
        self.xid_observable: list[str] = []
        self.yid_observable: list[str] = []
        self.x_references: list[Any] = []
        self.y_references: list[Any] = []
        self.y_errors: list[Any] = []
        self.y_errors_type: list[str] = []
        self.weights: list[
            Any
        ] = []  # total weights for points (data points and curve weights)
        self.weights_points: list[Any] = []  # weights for data points based on errors
        self.weights_curves: list[Any] = []  # user defined weights per mapping/curve

        self.models: list[Any] = []
        self.xmodel: np.ndarray = np.empty(shape=(len(self.pids)))
        self.simulations: list[Any] = []
        self.selections: list[Any] = []

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

    def to_dict(self) -> dict[str, Any]:
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
            f"\tresidual_type: {self.residual}",
            f"\tweighting_curves: {self.weighting_curves}",
            f"\tweighting_points: {self.weighting_points}",
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
            "weights_curves",
        ]:
            all_info.append(f"\t{key}: {str(getattr(self, key))}")

        info = "\n".join(all_info)

        if print_output:
            console.log(info)
        if path:
            with open(path, "w") as f:
                f.write(info)
        return info

    def initialize(
        self,
        residual: Optional[ResidualType],
        loss_function: LossFunctionType,
        weighting_curves: list[WeightingCurvesType],
        weighting_points: Optional[WeightingPointsType],
        variable_step_size: bool = True,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-6,
    ) -> None:
        """Initialize Optimization problem.

        Performs precalculations, resolving data, calculating weights.
        Creates and attaches simulator for the given problem.

        :param residual: handling of residuals
        :param loss_function: loss function for residual transformation
        :param weighting_curves: list of options for weighting curves (fit mappings)
        :param weighting_points: weighting of points
        :param absolute_tolerance: absolute tolerance of simulator
        :param relative_tolerance: relative tolerance of simulator
        :param variable_step_size: use variable step size in solver
        """
        if weighting_curves is None:
            # no weighting by default
            weighting_curves = []
        if isinstance(weighting_curves, WeightingCurvesType):
            raise TypeError(
                f"weighting_curves must be a 'list[WeightingCurvesType]', "
                f"but '{type(weighting_curves)}' given."
            )

        if residual is None:
            raise ValueError("'residual_type' is required.")
        if weighting_points is None:
            raise ValueError("'weighting_points' is required.")

        self.residual = residual
        self.loss_function = loss_function
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
        # fit_exp: Callable
        for fit_experiment in self.fit_experiments:
            # get simulation experiment
            sid = fit_experiment.experiment_class.__name__
            sim_experiment = self.runner.experiments[sid]

            # FIXME: selections should be based on fit mappings; this will reduce
            # selections and speed up calculations
            selections_set: set[str] = set()
            # for d in sim_experiment._data.values():  # type: Data
            #     if d.is_task():
            #         selections_set.add(d.selection)

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
                obs_xid = mapping.observable.x.selection
                obs_yid = mapping.observable.y.selection
                selections_set.add(obs_xid)
                selections_set.add(obs_yid)
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

                if self.residual in [
                    ResidualType.ABSOLUTE_TO_BASELINE,
                    ResidualType.NORMALIZED_TO_BASELINE,
                ]:
                    # Changes to baseline, which is the first point
                    y_ref = y_ref - y_ref[0]

                # --- errors on data ---
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

                # handle special case of all NaN
                if y_ref_err is not None and np.all(np.isnan(y_ref_err)):
                    y_ref_err = None
                    y_ref_err_type = None

                # handle missing data (0.0 and NaN)
                if y_ref_err is not None:
                    # remove 0.0 from y-error
                    y_ref_err[(y_ref_err == 0.0)] = np.nan
                    if np.all(np.isnan(y_ref_err)):
                        # handle special case of all NaN errors
                        logger.warning(
                            f"Errors are all NaN '{sid}.{mapping_id}' y data: "
                            f"'{y_ref_err}'"
                        )
                        y_ref_err = None
                        y_ref_err_type = None
                    else:
                        # FIXME: this must be based on coefficient of variation
                        # some NaNs could exist (err is maximal error of all points)
                        y_ref_err[np.isnan(y_ref_err)] = np.nanmax(y_ref_err)

                # remove NaN from y-data
                nonnan_mask = ~np.isnan(y_ref)
                if not np.all(nonnan_mask):
                    logger.debug(
                        f"Removing NaN values in '{sid}.{mapping_id}' y data: '{y_ref}'"
                    )
                x_ref = x_ref[nonnan_mask]
                y_ref = y_ref[nonnan_mask]
                if y_ref_err is not None:
                    y_ref_err = y_ref_err[nonnan_mask]

                # at this point all x_ref, y_ref and y_ref_err
                for data_key, data in [
                    ("x_ref", x_ref),
                    ("y_ref", y_ref),
                    ("y_ref_err", x_ref),
                ]:
                    if data_key == "y_ref_err" and data is None:
                        # skip tests if no error data
                        continue
                    if np.any(~np.isfinite(data)):
                        raise ValueError(
                            f"{fit_experiment}.{mapping_id}: NaN or INF in "
                            f"'{data_key}': '{data}'"
                        )

                # --- WEIGHTS ---

                # weight points (default to 1.0)
                weight_points: np.ndarray
                if self.weighting_points == WeightingPointsType.NO_WEIGHTING:
                    # local weights are by default 1.0
                    weight_points = np.ones_like(y_ref)

                elif self.weighting_points == WeightingPointsType.ERROR_WEIGHTING:
                    # Challenging to combine datasets with errors and without
                    # due to the weighting based on the error

                    if y_ref_err is not None:
                        # Scale with coefficient of variation (1/CV)
                        # the larger the error, the smaller the weight
                        # weight_points = 1.0 / y_ref_err
                        # CV = SD/mean; scaling with 1/CV (CV=1 -> w=1; CV=0.1 -> w=10);
                        # The weighting must be normalized to the curve!, i.e. be a
                        # unitless quantity approximately the same for the different
                        # datasets.
                        weight_points = np.abs(y_ref / y_ref_err)
                        # weight_points = 1.0 / y_ref_err  # scale with error;
                    else:
                        logger.warning(
                            f"'{sid}.{mapping_id}': Using '{self.weighting_points}' "
                            f"with no errors in reference data."
                        )
                        # Weights must be comparable to datasets with data (1/CV)
                        # Assuming an error with CV of 0.5 -> w=2
                        weight_points = 2 * np.ones_like(y_ref)

                else:
                    raise ValueError(
                        f"Unsupported WeightingPointsType: '{weighting_points}'"
                    )

                # curve weight
                weight_curve: float = 1.0
                if WeightingCurvesType.MAPPING in self.weighting_curves:
                    weight_curve = weight_curve * weight_curve_user
                if WeightingCurvesType.POINTS in self.weighting_curves:
                    weight_curve = weight_curve / len(y_ref)
                # if WeightingCurvesType.MEAN in self.weighting_curves:
                #     weight_curve = weight_curve_user / np.mean(y_ref)

                # total weight (apply local weighting & user defined weighting)
                # w{k} * w{k,i}
                weight = weight_curve * weight_points
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

                selections: list[str] = list(selections_set)

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
                    console.log(f"{fit_experiment}.{mapping_id}")
                    console.log(f"weight: {weight}")
                    console.log(f"weight_curve: {weight_curve}")
                    console.log(f"weight_points: {weight_points}")
                    console.log(f"y_ref: {y_ref}")
                    console.log(f"y_ref_err: {y_ref_err}\n")

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
    ) -> Tuple[list[optimize.OptimizeResult], list]:
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
    ) -> Tuple[scipy.optimize.OptimizeResult, list]:
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

        Optimization is performed in logarithmic parameter space to
        account for xtol in largely varying parameters.
        see https://github.com/scipy/scipy/issues/7632

        :param xlog: logarithmic parameter vector
        :param complete_data: boolean flag to return additional information
        :return: vector of weighted residuals
        """
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
            simulator.set_timecourse_selections(selections=self.selections[k])

            # FIXME: normalize simulations and parameters once outside of loop
            simulation: TimecourseSim = self.simulations[k]
            simulation.normalize(uinfo=simulator.uinfo)

            # run simulation
            try:
                # FIXME: just simulate at the requested timepoints with step
                df = simulator._timecourses([simulation])[0]

                # interpolation of simulation results and requested time points
                f = interpolate.interp1d(
                    x=df[self.xid_observable[k]],
                    y=df[self.yid_observable[k]],
                    copy=False,
                    assume_sorted=True,
                )
                y_obsip = f(self.x_references[k])

                if self.residual in {
                    ResidualType.ABSOLUTE_TO_BASELINE,
                    ResidualType.NORMALIZED_TO_BASELINE,
                }:
                    # subtract simulation baseline
                    y_obsip = y_obsip - y_obsip[0]

                # calculate absolute residuals (f(x_{i}) - y_{i})
                res_abs = y_obsip - self.y_references[k]

            except RuntimeError as err:
                # error in integration (setting high residuals & cost)
                logger.error(
                    f"RuntimeError in ODE integration ('{self.pids} = {x}'): \n{err}"
                )
                res_abs = 5.0 * self.y_references[k]  # total error

            # with np.errstate(divide="ignore", invalid="ignore"):
            res_norm = res_abs / np.mean(self.y_references[k])

            # select correct residuals
            residuals: np.ndarray
            if self.residual == ResidualType.ABSOLUTE:
                residuals = res_abs
            elif self.residual == ResidualType.NORMALIZED:
                residuals = res_norm
            elif self.residual == ResidualType.ABSOLUTE_TO_BASELINE:
                residuals = res_abs
            elif self.residual == ResidualType.NORMALIZED_TO_BASELINE:
                residuals = res_norm
            else:
                raise ValueError(f"ResidualType not supported: '{self.residual}'")

            # weighted residuals
            # total cost:
            # 0.5 * sum(residuals_weighted^2)
            # the square root is required to ensure weighting with w in the squared
            # residuals;
            # this is not exactly the definition of typical weights.
            residuals_weighted = residuals * np.sqrt(self.weights[k])

            # apply loss function
            if self.loss_function == LossFunctionType.LINEAR:
                pass
            elif self.loss_function == LossFunctionType.SOFT_L1:
                residuals_weighted = 2 * (np.power(1 + residuals_weighted, 0.5) - 1)
            elif self.loss_function == LossFunctionType.CAUCHY:
                residuals_weighted = np.log(1 + residuals_weighted)
            elif self.loss_function == LossFunctionType.ARCTAN:
                residuals_weighted = np.arctan(residuals_weighted)

            parts.append(residuals_weighted)

            # for post_processing
            if complete_data:
                residual_data["x_obs"].append(df[self.xid_observable[k]])
                residual_data["y_obs"].append(df[self.yid_observable[k]])
                residual_data["y_obsip"].append(y_obsip)
                residual_data["residuals"].append(residuals)
                residual_data["weights_curve"].append(self.weights_curves[k])
                residual_data["residuals_weighted"].append(residuals_weighted)
                residual_data["res_abs"].append(res_abs)
                residual_data["res_norm"].append(res_norm)
                residual_data["cost"].append(
                    0.5 * np.sum(np.power(residuals_weighted, 2))
                )

        if complete_data:
            return residual_data
        else:
            res_all = np.concatenate(parts)
            # store the local step
            self._trajectory.append((deepcopy(x), 0.5 * np.sum(np.power(res_all, 2))))
            return res_all
