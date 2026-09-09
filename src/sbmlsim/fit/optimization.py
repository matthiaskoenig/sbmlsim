"""Optimization of parameter fitting problem."""

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate

from sbmlsim.console import console
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit.objects import (
    UNUSED_KINDS,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    MappingKind,
)
from sbmlsim.fit.options import (
    FitSettings,
    LossFunctionType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.sampling import SamplingType, create_samples
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.serialization import ObjectJSONEncoder, to_json
from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.units import DimensionalityError, Quantity
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)


def apply_loss_function(
    residuals: np.ndarray, loss_function: LossFunctionType
) -> np.ndarray:
    """Apply the loss function to the residuals.

    The cost of the optimization is `0.5 * sum(rho(residuals**2))` with `rho` the
    loss function. The optimizers minimize `0.5 * sum(f**2)`, so the residuals are
    transformed to `sign(r) * sqrt(rho(r**2))`, which gives exactly this cost and
    keeps the sign of the residual. The loss functions are the loss functions of
    `scipy.optimize.least_squares`.

    Args:
        residuals: weighted residuals of a fit mapping.
        loss_function: loss function to apply.

    Returns:
        Transformed residuals.

    Raises:
        ValueError: if the loss function is not supported.
    """
    if loss_function == LossFunctionType.LINEAR:
        return residuals

    # z = r^2 >= 0, so all loss functions are well defined
    z = np.square(residuals)
    rho: np.ndarray
    if loss_function == LossFunctionType.SOFT_L1:
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss_function == LossFunctionType.CAUCHY:
        rho = np.log1p(z)
    elif loss_function == LossFunctionType.ARCTAN:
        rho = np.arctan(z)
    else:
        raise ValueError(f"LossFunctionType not supported: '{loss_function}'")

    return np.sign(residuals) * np.sqrt(rho)


#: keys of an optimization result which are kept: the parameters and the cost,
#: which the reports and the metrics are made from, plus the scalars which say
#: how the run went. The optimizers also return the residuals `fun`, the
#: jacobian `jac`, the gradient and the active mask of the last step, which are
#: as large as the data and which nothing reads; they are dropped
RESULT_KEYS: tuple[str, ...] = (
    "x",
    "x0",
    "cost",
    "success",
    "status",
    "message",
    "duration",
    "optimality",
    "nfev",
)


def minimal_result(
    opt_result: scipy.optimize.OptimizeResult,
) -> scipy.optimize.OptimizeResult:
    """Reduce the result of an optimization to what is reported.

    Args:
        opt_result: result of one of the optimizers of scipy.

    Returns:
        A result with the keys of `RESULT_KEYS` which are present.
    """
    return scipy.optimize.OptimizeResult(
        {key: opt_result[key] for key in RESULT_KEYS if key in opt_result}
    )


class FitTimeout(Exception):
    """A single optimization ran longer than its budget.

    The optimizers of scipy cannot be interrupted, so the objective raises this
    once the budget is spent, which ends the optimization. The runs which
    finished are kept, see `OptimizationProblem.optimize`.
    """


class RuntimeErrorOptimizeResult(scipy.optimize.OptimizeResult):
    """Result of an optimization which did not finish.

    An optimization fails with an error of the integrator, with a timeout or
    with any other error of the objective. This *is* a
    `scipy.optimize.OptimizeResult`, i.e., a dictionary with attribute access,
    so that a failed run is stored, serialized and reported like a successful
    one and a fit keeps the runs which worked.
    """

    def __init__(
        self,
        x: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        cost: float = np.inf,
        message: str = "RuntimeError in ODE integration.",
        duration: float = -1.0,
    ):
        """Initialize the result of an optimization which did not finish.

        Args:
            x: parameters the optimization reached.
            x0: parameters it started from.
            cost: cost of `x`.
            message: what went wrong.
            duration: seconds the optimization ran.
        """
        super().__init__(
            status=-1,
            success=False,
            duration=duration,
            cost=cost,
            optimality=np.inf,
            x=x,
            x0=x0,
            message=message,
        )


class OptimizationProblem(ObjectJSONEncoder):
    """Parameter optimization problem."""

    def __init__(
        self,
        opid: str,
        mapping_collections: list[FitMappingCollection],
        fit_parameters: list[FitParameter],
        base_path: Path | None = None,
        data_path: Path | None = None,
    ):
        """Optimization problem.

        The problem must be pickable for parallelization !
        So initialize must be run to create the non-pickable instances.

        :param opid: id for optimization problem
        :param mapping_collections:
        :param fit_parameters:
        """
        super().__init__()
        self.opid: str = opid
        self.mapping_collections = []
        for collection in mapping_collections:
            if collection.exclude:
                logger.warning("FitMappingCollection excluded: %s", collection)
            else:
                self.mapping_collections.append(collection)
        if not fit_parameters:
            raise ValueError(
                f"'{opid}': an OptimizationProblem requires fit parameters, but "
                f"'{fit_parameters}' were given."
            )
        pids = [p.pid for p in fit_parameters]
        if len(pids) > len(set(pids)):
            raise ValueError(
                f"'{opid}': duplicate fit parameters are not allowed: '{sorted(pids)}'."
            )
        self.parameters = fit_parameters

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
        self.runner: ExperimentRunner | None = None
        self.settings: FitSettings | None = None

        # cost of every step of the running optimization, for the trace plot
        self._trajectory: list[float] = []
        # best step of the running optimization, which a run that is
        # interrupted keeps; the trajectory does not store the parameters
        self._best: tuple[np.ndarray, float] | None = None
        # deadline of the running optimization, set by `_optimize_single`
        self._deadline: float | None = None
        self.xmodel: np.ndarray = np.empty(shape=(len(self.pids)))
        self._reset_mappings()

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the definition of the problem, not its resolved data.

        An initialized problem holds the experiment runner with the models and
        the unit registry, which cannot be pickled. The workers of a pool
        initialize the problem themselves, so what they need is the
        definition: a pickled problem is uninitialized, whether the original
        was initialized or not.
        """
        fresh = OptimizationProblem(
            opid=self.opid,
            mapping_collections=self.mapping_collections,
            fit_parameters=self.parameters,
            base_path=self.base_path,
            data_path=self.data_path,
        )
        return fresh.__dict__

    def _reset_mappings(self) -> None:
        """Reset the data collected for the fit mappings.

        The data of the mappings is collected in `initialize`, which can be called
        more than once, e.g., to run an optimization and to analyze it afterwards.
        """
        self.experiment_keys: list[str] = []
        self.mapping_keys: list[str] = []
        self.mapping_kinds: list[MappingKind] = []
        # the collection of `mapping_collections` every mapping comes from
        self.collection_indices: list[int] = []
        self.xid_observable: list[str] = []
        self.yid_observable: list[str] = []
        self.x_references: list[Any] = []
        self.y_references: list[Any] = []
        self.y_errors: list[Any] = []
        self.y_errors_type: list[str | None] = []
        # total weights for points (data points and curve weights)
        self.weights: list[Any] = []
        self.weights_points: list[Any] = []  # weights for data points based on errors
        self.weights_curves: list[Any] = []  # user defined weights per mapping/curve

        self.models: list[Any] = []
        self.simulations: list[Any] = []
        self.selections: list[Any] = []
        # indices of the mappings which share a simulation, see `_group_mappings`
        self.mapping_groups: list[list[int]] = []

    def indices(self, kind: MappingKind | None = None) -> list[int]:
        """Get the indices of the fit mappings of a kind.

        Args:
            kind: kind of the mappings, all mappings if `None`.

        Returns:
            Indices into the resolved data of the mappings.
        """
        if kind is None:
            return list(range(len(self.mapping_keys)))
        return [k for k, mk in enumerate(self.mapping_kinds) if mk is kind]

    @property
    def training_indices(self) -> list[int]:
        """Indices of the fit mappings which are fitted."""
        return self.indices(MappingKind.TRAINING)

    @property
    def validation_indices(self) -> list[int]:
        """Indices of the fit mappings which are only evaluated."""
        return self.indices(MappingKind.VALIDATION)

    def mapping_counts(self) -> dict[MappingKind, int]:
        """Get the number of resolved fit mappings per kind."""
        return {
            kind: len(self.indices(kind)) for kind in MappingKind if self.indices(kind)
        }

    @property
    def is_initialized(self) -> bool:
        """Check if the problem was initialized, i.e., the data is resolved."""
        return self.runner is not None and bool(self.mapping_keys)

    @property
    def settings_initialized(self) -> FitSettings:
        """Settings of the problem, set in `initialize`.

        Raises:
            ValueError: if the problem was not initialized.
        """
        if self.settings is None:
            raise ValueError(
                f"OptimizationProblem '{self.opid}' must be initialized first."
            )
        return self.settings

    @property
    def residual(self) -> ResidualType:
        """Handling of the residuals, see `FitSettings`."""
        return self.settings_initialized.residual

    @property
    def loss_function(self) -> LossFunctionType:
        """Loss function of the fit, see `FitSettings`."""
        return self.settings_initialized.loss_function

    @property
    def weighting_curves(self) -> Sequence[WeightingCurvesType]:
        """Weighting of the curves, see `FitSettings`."""
        return self.settings_initialized.weighting_curves

    @property
    def weighting_points(self) -> WeightingPointsType:
        """Weighting of the data points, see `FitSettings`."""
        return self.settings_initialized.weighting_points

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
        info.extend([f"\t{e}" for e in self.mapping_collections])
        info.append("Parameters")
        info.extend([f"\t{p}" for p in self.parameters])
        return "\n".join(info)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {}
        for key in [
            "opid",
            "mapping_collections",
            "parameters",
            "base_path",
            "data_path",
        ]:
            d[key] = self.__dict__[key]
        return d

    def to_json(self, path: Path | None = None) -> str | Path:
        """Store OptimizationResult as json.

        Uses the to_dict method.
        """
        return to_json(object=self, path=path)

    def report(self, path: Path | None = None, print_output: bool = True) -> str:
        """Print and write report.

        Can only be called after initialization.
        """
        core_info = self.__str__()
        counts = ", ".join(
            f"{count} {kind.value}" for kind, count in self.mapping_counts().items()
        )
        all_info = [
            core_info,
            str(self.settings_initialized),
            f"Mappings: {len(self.mapping_keys)} ({counts})",
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
            all_info.append(f"\t{key}: {getattr(self, key)!s}")

        info = "\n".join(all_info)

        if print_output:
            console.log(info)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(info)
        return info

    def initialize(self, settings: FitSettings, force: bool = False) -> None:
        """Initialize the optimization problem for the given settings.

        Resolves the data of the fit mappings, converts it to the units of the
        model, calculates the weights and attaches a simulator. The problem is
        only initialized once for a given set of settings: a fit and the report
        of the fit use the same problem, and resolving the data twice repeats
        the work and every message about the data.

        Args:
            settings: settings of the fit, they decide how the residuals and
                the weights are calculated.
            force: initialize again even if the settings did not change.

        Raises:
            TypeError: if the settings are not a `FitSettings`.
        """
        if not isinstance(settings, FitSettings):
            raise TypeError(
                f"'settings' must be a 'FitSettings', but '{type(settings)}' given."
            )
        if not force and self.is_initialized and self.settings == settings:
            logger.debug("%s: already initialized, skipping", self.opid)
            return

        self.settings = settings
        self._validate_parameters()
        # initialize can be called more than once, e.g. for the report of a fit
        self._reset_mappings()

        # Create experiment runner (loads the experiments & all models)
        exp_classes: set[type[SimulationExperiment]] = {
            collection.experiment_class for collection in self.mapping_collections
        }

        self.runner = ExperimentRunner(
            experiment_classes=list(exp_classes),
            base_path=self.base_path,
            data_path=self.data_path,
        )

        # Collect information for simulations
        for collection_index, mapping_collection in enumerate(self.mapping_collections):
            # get simulation experiment
            sid = mapping_collection.experiment_class.__name__
            sim_experiment = self.runner.experiments[sid]

            # FIXME: selections should be based on fit mappings; this will reduce
            # selections and speed up calculations
            selections_set: set[str] = set()
            # for d in sim_experiment._data.values():  # type: Data
            #     if d.is_task():
            #         selections_set.add(d.selection)

            # a FitMappingCollection without mappings uses all mappings of the experiment
            mapping_collection.resolve_mappings(sim_experiment._fit_mappings.keys())

            # collect information for single mapping
            for k, mapping_id in enumerate(mapping_collection.mappings):
                # sanity checks
                if mapping_id not in sim_experiment._fit_mappings:
                    raise ValueError(
                        f"Mapping key '{mapping_id}' not defined in "
                        f"SimulationExperiment\n"
                        f"{sim_experiment}\n"
                        f"{mapping_collection}"
                    )

                mapping: FitMapping = sim_experiment._fit_mappings[mapping_id]

                if mapping_collection.kind in UNUSED_KINDS:
                    # the outliers and the data the model does not describe are
                    # used neither in the fit nor in the evaluation
                    continue

                if mapping.observable.task_id is None:
                    raise ValueError(
                        f"Only observables from tasks supported: '{mapping.observable}'"
                    )
                if mapping.reference.dset_id is None:
                    raise ValueError(
                        f"Only references from datasets supported: "
                        f"'{mapping.reference}'"
                    )

                # get weight for curve
                if mapping_collection.use_mapping_weights:
                    # use provided mapping weights
                    weight_curve_user = mapping.weight
                    mapping_collection.weights[k] = weight_curve_user
                    if weight_curve_user is None:
                        raise ValueError(
                            f"If `use_mapping_weights` is set on a FitMappingCollection "
                            f"then all mappings must have a weight. But "
                            f"weight '{weight_curve_user}' in {mapping}."
                        )
                else:
                    weight_curve_user = mapping_collection.weights[k]

                if weight_curve_user is not None and weight_curve_user < 0:
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
                if data_ref.x is None or data_ref.y is None:
                    raise ValueError(
                        f"{sid}.{mapping_id}: reference data requires x and y data."
                    )
                try:
                    data_ref.x = data_ref.x.to(obs_x_unit)
                except DimensionalityError as e:
                    logger.error(
                        "%s.%s: Unit conversion fails for '%s' to '%s",
                        sid,
                        mapping_id,
                        data_ref.x,
                        obs_x_unit,
                    )
                    raise e
                try:
                    data_ref.y = data_ref.y.to(obs_y_unit)
                except DimensionalityError as e:
                    logger.error(
                        "%s.%s: Unit conversion fails for '%s' to '%s'.",
                        sid,
                        mapping_id,
                        data_ref.y,
                        obs_y_unit,
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
                            "Errors are all NaN '%s.%s' y data: '%s'",
                            sid,
                            mapping_id,
                            y_ref_err,
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
                        "Removing NaN values in '%s.%s' y data: '%s'",
                        sid,
                        mapping_id,
                        y_ref,
                    )
                x_ref = x_ref[nonnan_mask]
                y_ref = y_ref[nonnan_mask]
                if y_ref_err is not None:
                    y_ref_err = y_ref_err[nonnan_mask]

                # at this point all x_ref, y_ref and y_ref_err must be finite
                for data_key, data in [
                    ("x_ref", x_ref),
                    ("y_ref", y_ref),
                    ("y_ref_err", y_ref_err),
                ]:
                    if data is None:
                        # no error data on the mapping
                        continue
                    if np.any(~np.isfinite(data)):
                        raise ValueError(
                            f"{mapping_collection}.{mapping_id}: NaN or INF in "
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
                        # Weights must be comparable to datasets with data (1/CV)
                        # Assuming an error with CV of 0.5 -> w=2, the mappings
                        # without errors are in the report of the problem
                        weight_points = 2 * np.ones_like(y_ref)

                else:
                    raise ValueError(
                        f"Unsupported WeightingPointsType: '{self.weighting_points}'"
                    )

                # curve weight
                weight_curve: float = 1.0
                if WeightingCurvesType.MAPPING in self.weighting_curves:
                    if weight_curve_user is None:
                        raise ValueError(
                            f"{sid}.{mapping_id}: weight of mapping is required."
                        )
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

                selections: list[str] = list(selections_set)

                # lookup maps
                self.models.append(model)
                self.simulations.append(simulation)
                self.selections.append(selections)

                # store information
                self.experiment_keys.append(sid)
                self.mapping_keys.append(mapping_id)
                self.mapping_kinds.append(mapping_collection.kind)
                self.collection_indices.append(collection_index)
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

                logger.debug(
                    "%s.%s: weight_curve=%s, weight_points=%s",
                    sid,
                    mapping_id,
                    weight_curve,
                    weight_points,
                )

        # initial parameter values of the models
        self._store_model_parameters()
        self._group_mappings()

        if not self.training_indices:
            raise ValueError(
                f"'{self.opid}': no training data, at least one fit mapping must "
                f"be '{MappingKind.TRAINING.value}'."
            )

        # set simulator instance with arguments
        simulator = SimulatorSerial(
            absolute_tolerance=settings.absolute_tolerance,
            relative_tolerance=settings.relative_tolerance,
            variable_step_size=settings.variable_step_size,
        )
        self.set_simulator(simulator)

    @property
    def parameter_scale(self) -> ParameterScaleType:
        """Get the space the optimizer searches the parameters in."""
        return self.settings_initialized.parameter_scale

    def to_scale(self, x: Any) -> np.ndarray:
        """Transform parameters of the model into the space of the optimizer."""
        return np.asarray(self.parameter_scale.to_scale(x), dtype=float)

    def from_scale(self, x: Any) -> np.ndarray:
        """Transform parameters of the optimizer into the units of the model."""
        return np.asarray(self.parameter_scale.from_scale(x), dtype=float)

    def _validate_parameters(self) -> None:
        """Check that the parameters can be optimized.

        An optimization on a logarithmic scale, which is the default, requires
        finite positive bounds and start values; on the linear scale the bounds
        only have to be finite.

        Raises:
            ValueError: if a bound or a start value does not suit the scale.
        """
        scale = self.parameter_scale
        space = f"'{scale.name}' parameter space"
        for p in self.parameters:
            for key in ["lower_bound", "upper_bound"]:
                value = getattr(p, key)
                if not np.isfinite(value):
                    raise ValueError(
                        f"{self.opid}: the optimization requires a finite "
                        f"'{key}', but FitParameter '{p.pid}' has '{value}'."
                    )
                if scale.is_log and value <= 0.0:
                    raise ValueError(
                        f"{self.opid}: the optimization is performed in {space}, "
                        f"which requires a positive '{key}', but FitParameter "
                        f"'{p.pid}' has '{value}'."
                    )
            if scale.is_log and p.start_value is not None and p.start_value <= 0.0:
                raise ValueError(
                    f"{self.opid}: the optimization is performed in {space}, "
                    f"which requires a positive 'start_value', but FitParameter "
                    f"'{p.pid}' has '{p.start_value}'."
                )

    def _group_mappings(self) -> None:
        """Group the fit mappings which are simulated together.

        Several fit mappings read different observables of the same simulation,
        e.g. the plasma concentration and the amount in urine of one dosing.
        The simulation of such a group runs once per evaluation of the
        residuals with the selections of all its mappings, which is where the
        time of a fit goes.
        """
        groups: dict[tuple[int, int], list[int]] = {}
        for k in range(len(self.mapping_keys)):
            key = (id(self.models[k]), id(self.simulations[k]))
            groups.setdefault(key, []).append(k)
        self.mapping_groups = list(groups.values())

        logger.debug(
            "%s: %s fit mappings in %s simulations",
            self.opid,
            len(self.mapping_keys),
            len(self.mapping_groups),
        )

    def _store_model_parameters(self) -> None:
        """Store the initial values of the fitted parameters in the models.

        The values are read from the first model, a model which starts from
        different values is reported.

        Raises:
            ValueError: if a model is not loaded in roadrunner.
        """
        for k_model, model in enumerate(self.models):
            if model.r is None:
                raise ValueError(f"Model '{model}' is not loaded in roadrunner.")

            for k, pid in enumerate(self.pids):
                pid_value = model.r[pid]
                if pid in model.changes:
                    change = model.changes[pid]
                    # model changes have units
                    pid_value = (
                        change.magnitude if isinstance(change, Quantity) else change
                    )
                if k_model == 0:
                    self.xmodel[k] = pid_value
                elif not np.isclose(self.xmodel[k], pid_value):
                    logger.warning(
                        "%s: models start from different values for '%s': "
                        "'%s' != '%s'; the value of the first model is reported.",
                        self.opid,
                        pid,
                        self.xmodel[k],
                        pid_value,
                    )

    def parameter_set_model(self, sid: str = "model") -> ParameterSet:
        """Get the initial values of the fitted parameters in the model.

        The set is the reference a fitted set is compared against in a report.

        Args:
            sid: identifier of the set.

        Returns:
            Parameter set of the values the models start from.
        """
        return ParameterSet.from_model(
            parameters=self.parameters, x=self.xmodel, sid=sid
        )

    def set_simulator(self, simulator: SimulatorSerial | None) -> None:
        """Set the simulator on the runner and the experiments."""
        self.runner_initialized.set_simulator(simulator)

    @property
    def runner_initialized(self) -> ExperimentRunner:
        """Runner of the problem, created in `initialize`."""
        if self.runner is None:
            raise ValueError(
                f"OptimizationProblem '{self.opid}' must be initialized first."
            )
        return self.runner

    def optimize(
        self,
        size: int = 5,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        sampling: SamplingType = SamplingType.UNIFORM,
        seed: int | None = None,
        timeout: float | None = None,
        on_run_finished: (
            Callable[[int, scipy.optimize.OptimizeResult, list[float]], None] | None
        ) = None,
        **kwargs,
    ) -> tuple[list[scipy.optimize.OptimizeResult], list[list[float]]]:
        """Run parameter optimization.

        The problem must be initialized, i.e., the settings of the fit are the
        settings it was initialized with.

        Args:
            size: number of optimizations, every one starts from its own sample.
            algorithm: optimization algorithm.
            sampling: sampling of the start values of the local optimizer.
            seed: seed of the sampling.
            timeout: seconds a single optimization may run, no limit if `None`.
                A run which is out of time keeps the parameters it reached.
            on_run_finished: called with the index, the fit and the trajectory
                of every finished optimization, i.e., to report the progress of
                a fit and to store the runs while it runs.
            kwargs: additional arguments of the optimizer.

        Returns:
            The fits and the trajectories of the optimizations. A run which
            failed is a `RuntimeErrorOptimizeResult` with its message, the
            other runs are unaffected.
        """
        starts = self.start_values(
            size=size, algorithm=algorithm, sampling=sampling, seed=seed
        )
        seeds = self.run_seeds(size=size, algorithm=algorithm, seed=seed)

        fits: list[scipy.optimize.OptimizeResult] = []
        trajectories: list[list[float]] = []
        for k in range(size):
            fit, trajectory = self.optimize_run(
                x0=starts[k],
                algorithm=algorithm,
                timeout=timeout,
                run=k,
                size=size,
                run_seed=seeds[k],
                **kwargs,
            )
            fits.append(fit)
            trajectories.append(trajectory)
            if on_run_finished is not None:
                on_run_finished(k, fit, trajectory)
        return fits, trajectories

    def start_values(
        self,
        size: int,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        sampling: SamplingType = SamplingType.UNIFORM,
        seed: int | None = None,
    ) -> list[np.ndarray | None]:
        """Create the start values of the optimization runs.

        The start values are created for all runs at once, so that they only
        depend on the seed and the number of runs and not on how the runs are
        distributed over the workers of a parallel fit.

        Args:
            size: number of optimizations.
            algorithm: optimization algorithm. The global optimizer draws its
                own samples, its runs start from `None`.
            sampling: sampling of the start values of the local optimizer.
            seed: seed of the sampling.

        Returns:
            One start vector per run, `None` for the global optimizer.
        """
        if algorithm != OptimizationAlgorithmType.LEAST_SQUARE:
            return [None] * size
        x_samples: pd.DataFrame = create_samples(
            parameters=self.parameters,
            size=size,
            sampling=sampling,
            seed=seed,
        )
        return [x_samples.values[k, :] for k in range(size)]

    @staticmethod
    def run_seeds(
        size: int,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        seed: int | None = None,
    ) -> list[int | None]:
        """Create the seed of every optimization run.

        The global optimizer draws its own population, so every run needs its
        own seed: with one seed for all runs they all return the same result.
        The local optimizer is deterministic, its runs differ in the start
        values and do not need a seed.

        Args:
            size: number of optimizations.
            algorithm: optimization algorithm.
            seed: seed of the fit, `None` for runs which are not reproducible.

        Returns:
            One seed per run, `None` if the runs do not need one.
        """
        if algorithm == OptimizationAlgorithmType.LEAST_SQUARE or seed is None:
            return [None] * size
        return [
            int(s) for s in np.random.SeedSequence(seed).generate_state(max(size, 1))
        ]

    def optimize_run(
        self,
        x0: np.ndarray | None = None,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        timeout: float | None = None,
        run: int = 0,
        size: int = 1,
        run_seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[scipy.optimize.OptimizeResult, list[float]]:
        """Run a single optimization, which never raises.

        This is one repeat of a fit, i.e., what the serial runner loops over and
        what a worker of a parallel fit executes. An optimization which fails is
        a `RuntimeErrorOptimizeResult` with its message, so that the repeat is
        stored and reported like a successful one and the other repeats are
        unaffected.

        Args:
            x0: start values of the run, `None` for the global optimizer.
            algorithm: optimization algorithm.
            timeout: seconds the optimization may run, no limit if `None`.
            run: index of the run, for the log messages.
            size: number of runs, for the log messages.
            run_seed: seed of the run, `None` if it does not need one.
            kwargs: additional arguments of the optimizer.

        Returns:
            The fit and the cost of every step of the optimization.
        """
        if run_seed is not None:
            kwargs["rng"] = run_seed
        logger.debug("[%s/%s] x0=%s", run + 1, size, x0)
        try:
            fit, trajectory = self._optimize_single(
                x0=x0, algorithm=algorithm, timeout=timeout, **kwargs
            )
        except Exception as err:
            # one run must not lose the results of the other runs
            logger.error(
                "%s: optimization %s/%s failed: %s: %s",
                self.opid,
                run + 1,
                size,
                type(err).__name__,
                err,
            )
            return (
                RuntimeErrorOptimizeResult(
                    x=np.asarray(x0, dtype=float) if x0 is not None else None,
                    x0=np.asarray(x0, dtype=float) if x0 is not None else None,
                    message=f"{type(err).__name__}: {err}",
                ),
                [],
            )
        logger.debug("	%s [s]", format(fit.duration, "8.4f"))
        return fit, trajectory

    @timeit
    def _optimize_single(
        self,
        x0: np.ndarray | None = None,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        timeout: float | None = None,
        **kwargs,
    ) -> tuple[scipy.optimize.OptimizeResult, list]:
        """Run single optimization with x0 start values.

        Args:
            x0: parameter start vector (important for deterministic optimizers).
            algorithm: optimization algorithm and method.
            timeout: seconds the optimization may run, no limit if `None`.
            kwargs: additional arguments of the optimizer.

        Returns:
            The fit and the trajectory of the optimization.

        Raises:
            ValueError: if the algorithm is not supported or start values are
                required and missing.
        """
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._trajectory = []
        self._best = None
        try:
            return self._optimize_single_run(x0=x0, algorithm=algorithm, **kwargs)
        finally:
            # the deadline is the budget of this run, the residuals are
            # evaluated again when the fit is reported
            self._deadline = None

    def _optimize_single_run(
        self,
        x0: np.ndarray | None = None,
        algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
        **kwargs,
    ) -> tuple[scipy.optimize.OptimizeResult, list]:
        """Run a single optimization, see `_optimize_single`.

        Raises:
            ValueError: if the algorithm is not supported or start values are
                required and missing.
        """
        # FIXME: this should not be necessary, handle outside
        if x0 is None:
            if any(value is None for value in self.x0):
                raise ValueError(
                    f"{self.opid}: an optimization without start values requires a "
                    f"'start_value' on every FitParameter: '{self.parameters}'."
                )
            x0 = np.array(self.x0, dtype=float)

        # the optimizer searches the scaled space, see `ParameterScaleType`
        x0log: np.ndarray = self.to_scale(x0)

        if algorithm == OptimizationAlgorithmType.LEAST_SQUARE:
            # scipy least square optimizer
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            ts = time.time()
            try:
                boundslog = [
                    self.to_scale([p.lower_bound for p in self.parameters]),
                    self.to_scale([p.upper_bound for p in self.parameters]),
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
            except (RuntimeError, FitTimeout) as err:
                logger.error(
                    "%s in ODE integration (optimize) for '%s = %s': \n%s",
                    type(err).__name__,
                    self.pids,
                    x0,
                    err,
                )
                opt_result = self._interrupted_result(err, x0log)
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            opt_result.x = self.from_scale(opt_result.x)
            return minimal_result(opt_result), list(self._trajectory)

        if algorithm == OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION:
            # scipy differential evolution
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
            ts = time.time()
            try:
                de_bounds_log = [
                    (self.to_scale(p.lower_bound), self.to_scale(p.upper_bound))
                    for k, p in enumerate(self.parameters)
                ]
                opt_result = scipy.optimize.differential_evolution(
                    func=self.cost_least_square, bounds=de_bounds_log, **kwargs
                )
            except (RuntimeError, FitTimeout) as err:
                logger.error(
                    "%s in ODE integration (optimize) for '%s = %s': \n%s",
                    type(err).__name__,
                    self.pids,
                    x0,
                    err,
                )
                opt_result = self._interrupted_result(err, x0log)
            te = time.time()
            opt_result.x0 = x0  # store start value
            opt_result.duration = te - ts
            if not isinstance(opt_result, RuntimeErrorOptimizeResult):
                # differential evolution reports `fun`, the cost is evaluated.
                # An interrupted run already carries the best cost of its
                # trajectory, evaluating it again would hit the deadline once
                # more
                opt_result.cost = self.cost_least_square(np.asarray(opt_result.x))
            opt_result.x = self.from_scale(opt_result.x)
            return minimal_result(opt_result), list(self._trajectory)

        raise ValueError(f"optimizer is not supported: {algorithm}")

    def _simulate_groups(
        self,
        simulator: SimulatorSerial,
        changes: dict[str, Quantity],
        evaluated: set[int],
        x: np.ndarray,
    ) -> dict[int, pd.DataFrame | None]:
        """Simulate the groups of fit mappings for the given parameters.

        The mappings of a group share a simulation, so it runs once with the
        selections of all of them; `_group_mappings` builds the groups.

        Args:
            simulator: simulator of the problem.
            changes: parameters to set on the simulations.
            evaluated: indices of the fit mappings which are evaluated.
            x: parameter values, for the message of a failed integration.

        Returns:
            The result of the simulation of every evaluated mapping, `None` if
            its integration failed.
        """
        results: dict[int, pd.DataFrame | None] = {}
        for group in self.mapping_groups:
            indices = [k for k in group if k in evaluated]
            if not indices:
                continue

            k0 = indices[0]
            simulation: TimecourseSim = self.simulations[k0]
            simulation.timecourses[0].changes.update(changes)

            simulator.set_model(model=self.models[k0])
            simulator.set_timecourse_selections(
                selections=sorted({s for k in indices for s in self.selections[k]})
            )
            simulation.normalize(uinfo=simulator.uinfo)

            df: pd.DataFrame | None
            try:
                # FIXME: just simulate at the requested timepoints with step
                df = simulator._timecourses([simulation])[0]
            except RuntimeError as err:
                logger.error(
                    "RuntimeError in ODE integration ('%s = %s'): \n%s",
                    self.pids,
                    x,
                    err,
                )
                df = None

            for k in indices:
                results[k] = df

        return results

    def _interrupted_result(
        self, err: Exception, x0log: np.ndarray
    ) -> RuntimeErrorOptimizeResult:
        """Get the result of a run which did not finish.

        The best parameters the optimizer reached are kept, so a run which ran
        out of time still contributes what it found.
        """
        best = self._best
        return RuntimeErrorOptimizeResult(
            x=self.to_scale(best[0]) if best is not None else x0log,
            cost=best[1] if best is not None else np.inf,
            message=f"{type(err).__name__}: {err}",
        )

    def cost_least_square(self, xlog: np.ndarray) -> float:
        """Get least square costs for parameters."""
        res_weighted: np.ndarray = self.residuals(xlog)  # ty: ignore[invalid-assignment]
        return float(0.5 * np.sum(np.square(res_weighted)))

    def residuals(
        self, xlog: np.ndarray, complete_data: bool = False
    ) -> np.ndarray | dict[str, list[Any]]:
        """Calculate residuals for given parameter vector.

        Optimization is performed in logarithmic parameter space to
        account for xtol in largely varying parameters.
        see https://github.com/scipy/scipy/issues/7632

        Args:
            xlog: logarithmic parameter vector.
            complete_data: return the simulations, residuals and costs of every
                fit mapping instead of the vector of weighted residuals.

        Returns:
            Vector of weighted residuals, or the complete data of the mappings.

        Raises:
            ValueError: if no simulator is set or the residuals are not supported.
        """
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise FitTimeout(
                f"'{self.opid}': the optimization did not finish in its budget."
            )
        x = self.from_scale(xlog)

        # FIXME: handle parts better
        parts = []
        if complete_data:
            residual_data = defaultdict(list)

        # simulate all mappings for all experiments
        simulator: SimulatorSerial | None = self.runner_initialized.simulator
        if simulator is None:
            raise ValueError(f"No simulator set on OptimizationProblem '{self.opid}'.")
        Q_ = self.runner_initialized.Q_

        # the parameters are the same for every mapping, the quantities are
        # created once and not once per mapping
        changes = {
            self.pids[ix]: Q_(value, self.punits[ix]) for ix, value in enumerate(x)
        }
        evaluated = {
            k
            for k in range(len(self.mapping_keys))
            # the optimization only uses the training data, the validation data
            # is simulated for the evaluation of a fit
            if complete_data or self.mapping_kinds[k] is MappingKind.TRAINING
        }
        results = self._simulate_groups(
            simulator=simulator, changes=changes, evaluated=evaluated, x=x
        )

        df: pd.DataFrame | None = None
        for k, mapping_key in enumerate(self.mapping_keys):
            if k not in evaluated:
                continue

            df = results[k]
            if df is not None:
                # interpolation of simulation results and requested time points
                f = interpolate.interp1d(
                    x=df[self.xid_observable[k]],
                    y=df[self.yid_observable[k]],
                    copy=False,
                    assume_sorted=True,
                )
                try:
                    y_obsip = f(self.x_references[k])
                except ValueError as err:
                    console.print(f"Interpolation error in mapping key: {mapping_key}")
                    raise err

                if self.residual in {
                    ResidualType.ABSOLUTE_TO_BASELINE,
                    ResidualType.NORMALIZED_TO_BASELINE,
                }:
                    # subtract simulation baseline
                    y_obsip = y_obsip - y_obsip[0]

                # calculate absolute residuals (f(x_{i}) - y_{i})
                res_abs = y_obsip - self.y_references[k]
            else:
                # the integration failed, setting high residuals & cost
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
            residuals_weighted = apply_loss_function(
                residuals_weighted, self.loss_function
            )

            if self.mapping_kinds[k] is MappingKind.TRAINING:
                # only the training data enters the cost
                parts.append(residuals_weighted)

            # for post_processing
            if complete_data:
                if df is None:
                    raise ValueError(
                        f"'{mapping_key}': no simulation results, the complete data "
                        f"of a failed simulation cannot be evaluated."
                    )
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
        res_all = np.concatenate(parts)

        # the cost of the step, the trace plot is the only thing which uses it
        cost = float(0.5 * np.sum(np.power(res_all, 2)))
        self._trajectory.append(cost)
        if self._best is None or cost < self._best[1]:
            # the parameters of the best step, kept for a run which is
            # interrupted; storing them for every step is what a trajectory
            # used to do and no report reads them
            self._best = (x.copy(), cost)
        return res_all
