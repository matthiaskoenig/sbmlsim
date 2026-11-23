"""The fit of an optimization problem in sbmlsim-core, the contract `fit/1` (#556).

An optimizer in python evaluates one point at a time: it asks the problem for
the residuals of a point, the problem simulates the conditions of that point
on the threads of the engine, the optimizer waits for all of them, and the
jacobian of a step is one batch of its own (#568). The engine fits a problem
itself instead, from one request: its optimizer, the runs of the starts of a
multi start fit, the profiles of the identifiability of its optimum and the
two sides of every profile all run in rust, each of them on the threads of
the same batch, and the fit reaches python once, as an answer.

What the engine needs is a request of the contract `fit/1`: the parameters of
the search space, the simulation of every condition, the curves of the data
with the weight of every point of them, and what the optimizer may spend.
This module builds that request from an initialized `OptimizationProblem`,
and `gaps_of` says what keeps a problem out of it: an objective the request
cannot carry, an observable which is no column of a simulation, or data of
more than one model.

The residuals of the request are the ones of the problem, point for point:
the residual of a point is `(f(x) - y) * sqrt(w)` on both sides, with the
weights `w` of the `FitSettings` of the problem
(`OptimizationProblem.weights`), which the engine takes as a standard
deviation of `1/sqrt(w)` (`CurveRequest.noise`, the engine divides by it);
the residuals of `ResidualType.NORMALIZED` divide by the mean of the data of
their curve, which is one value per curve and goes into that deviation as
well. So the fit of the engine minimizes the cost of the problem, and
`tests/fit/test_fit_core.py` holds the engine to it.

The conditions are the simulation groups of the problem, every one of them
the request of the plan of its group with the values of the parameters at the
start of the fit (`OptimizationProblem.evaluated_plan`), i.e. the simulation
the problem itself would run; the engine sets the parameters of the fit as
the varying targets of that request, which take the place of their values at
the start. A group whose data is read from observables of its experiment, or
whose simulation is a scan, is no condition of a request: its output is not
the columns of a simulation at the times of its data.

A value of that request is a number, so a value which an optimizer moves
cannot be one: the number would hold it where the fit started. That is a
PEtab v2 condition which assigns an entity an estimated parameter, e.g.
`[B] = initial_B` or `[A] = initial_A1 + initial_A2`, which the executor of
the engine refuses because it cannot fold such a formula into a number. A
formula which is a linear form over the parameters of the fit (`linear_form`)
is taken out of the plan and given to the condition as a derived value
instead (`_derived_of`, the field `derived` of sbmlsim-core
`DERIVED_VERSION`), which the engine evaluates at the values the optimizer is
at and sets on its target before every simulation; every other formula keeps
its gap.

A formula **at a change** is folded by the executor as well, but by
simulating the segment which ends at the change and reading the state there
(`executor_core._Fold`), so the number it folds into is the one of the point
the request was built at. A formula which reads a constant of the simulation
folds into the same number at every point of the search; one which reads
anything the simulation moves, e.g. the `-0.5 * Aurine` of a collection
which is halved, would be held where the fit started while the optimizer
moves the state it reads. A plan whose formulas at a change are all linear
forms over the entities of the model gives them to the engine as forms
(`executor_core.is_formed`, the field `forms` of a change of sbmlsim-core
`CHANGE_FORMS_VERSION`), which it evaluates from its own state; any other
formula which reads more than constants is the gap `change_value`
(`_gaps_at_the_folds`), so such a problem is fitted in python instead of
answering the residuals of another problem.

The times of a curve are the output times of its condition and not the times
of the data: `Plan.time_shift` is applied to the time of a result on the way
back and a request carries no shift, so a simulation whose data is on a
shifted axis measures at the times of its plan (`_curve`,
`OptimizationProblem._x_plan`).
"""

from __future__ import annotations

import inspect
import logging
import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from packaging.version import InvalidVersion, Version

from sbmlsim.engine import Engine, UnsupportedModelError
from sbmlsim.experiment.sids import to_sid
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.observable_sbml import (
    OBSERVABLE_PREFIX,
    Entities,
    add_formula,
    add_observable,
    observable_id,
    observable_model,
)
from sbmlsim.fit.options import (
    ObjectiveType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
)
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.simulator.formula import compile_formula, linear_form
from sbmlsim.simulator.observables import FormulaNode

if TYPE_CHECKING:  # pragma: no cover
    from scipy.optimize import OptimizeResult

    from sbmlsim.fit.objects import ObservableModel
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.options import FitSettings
    from sbmlsim.fit.provenance import FitProvenance
    from sbmlsim.fit.result import OptimizationResult
    from sbmlsim.model.model_core import CoreSBMLModel
    from sbmlsim.simulator.plan import Assignment, Plan

if TYPE_CHECKING:
    # `sbmlsim.fit.identifiability` imports the runner, which imports this
    # module, so the settings of a profile are a type and no import of its own
    from sbmlsim.fit.identifiability import ProfileSettings

logger = logging.getLogger(__name__)

#: the contract of a fit of the engine
CONTRACT = "fit/1"

#: the release of sbmlsim-core whose curve reads `after_change`, the state
#: after a change of a measurement, see `reads_after_change`
AFTER_CHANGE_VERSION = "0.16.0"

#: the release of sbmlsim-core whose condition reads `derived`, the values of
#: a condition which follow the parameters of the fit, see `reads_derived`
DERIVED_VERSION = "0.17.0"

#: the release of sbmlsim-core whose change reads `forms`, the linear formulas
#: of a change which the engine evaluates from its own state, each of them a
#: constant and its `terms`, see `reads_change_forms`
CHANGE_FORMS_VERSION = "0.20.0"

#: the release of sbmlsim-core whose curve takes a positive infinity as the
#: standard deviation of a measurement, a point which carries no information,
#: see `reads_infinite_noise`
INFINITE_NOISE_VERSION = "0.21.0"

#: the model of the fit of a problem which has observables of its own, kept
#: per problem with the entities it was built from, see `fit_model`; a weak
#: dictionary, so it holds no problem alive, and `__getstate__` pickles the
#: definition of a problem, so a worker builds its own
_FIT_MODELS: weakref.WeakKeyDictionary[OptimizationProblem, tuple[Entities, Any]] = (
    weakref.WeakKeyDictionary()
)

#: the scale of a parameter in a request of the engine, by the scale of the fit
SCALES: dict[ParameterScaleType, str] = {
    ParameterScaleType.LINEAR: "lin",
    ParameterScaleType.LOG: "log",
    ParameterScaleType.LOG10: "log10",
}

#: the algorithm of a request of the engine, by the algorithm of the fit; the
#: differential evolution of the engine takes the block of its population
ALGORITHMS: dict[OptimizationAlgorithmType, str] = {
    OptimizationAlgorithmType.LEAST_SQUARE: "trust_region",
    OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION: "differential_evolution",
}

#: where a jacobian of the engine comes from: the forward sensitivities of the
#: conditions of the request, which are the derivatives of the residuals, or a
#: forward difference quotient of a relative step of `1e-3`
SENSITIVITIES = "sensitivities"
DIFFERENCE_QUOTIENT = "difference_quotient"

#: the iterations of the optimizer of a point of a profile, which fits the
#: other parameters while one of them is held at a value: the fit of the
#: request takes none of its own, so that the profiles are walked around the
#: optimum the caller gives, see `profile_points`
MAX_ITERATIONS_OF_A_POINT: int = 100

#: the codes a run of the engine stops with which are a point its optimizer
#: cannot improve on (`basin::TerminationCode`), i.e. the `success` of a run
#: of scipy: a tolerance of the gradient, of the step or of the cost, and the
#: codes of a trust region whose step collapsed (`NumericalNoProgress`, a
#: rejected trial which changes no parameter, and `SolverFailed`, a step the
#: solver cannot take), which is what the `xtol` of
#: `scipy.optimize.least_squares` stops at. A code which is a budget of the
#: run (its iterations, its time), a stop of the caller or a failure of the
#: model is none of them, and the code itself is the message of the run.
CONVERGED: frozenset[str] = frozenset(
    {
        "GradientTolerance",
        "RelativeGradientTolerance",
        "ProjectedGradientTolerance",
        "ParamTolerance",
        "RelativeParamTolerance",
        "CostTolerance",
        "RelativeCostTolerance",
        "TargetCost",
        "SolverConverged",
        "NoImprovement",
        "NumericalNoProgress",
        "SolverFailed",
        "RhoTolerance",
    }
)

#: the residuals the engine computes, the ones which are a weight of a point
RESIDUALS: frozenset[ResidualType] = frozenset(
    {ResidualType.ABSOLUTE, ResidualType.NORMALIZED}
)

#: the noise distributions whose residual of the data is the residual the
#: engine computes, a difference divided by a standard deviation: the normal
#: one as it is and the log-normal one on a logarithmic column of the model,
#: see `_gaps_of_the_objective`. The residual of a Laplace distribution is
#: `sign(r) sqrt(2 |r|)` of that difference, which no curve of the contract
#: expresses.
LIKELIHOOD_DISTRIBUTIONS: frozenset[str] = frozenset({"normal", "log-normal"})

#: the prefix of the observable of the model which is the logarithm of the
#: column of a mapping of log-normal noise, see `_log_observable`
LOG_PREFIX: Final = "log__"


@dataclass(frozen=True)
class Gap:
    """Something of an optimization problem which the engine cannot fit.

    Attributes:
        code: what the gap is about, e.g. `objective` or `observable`.
        message: what keeps the problem out of the engine, a sentence.
    """

    code: str
    message: str


def loaded_model(problem: OptimizationProblem) -> Any:
    """Get the loaded model of a problem, the one the engine fits.

    Args:
        problem: the initialized problem.

    Returns:
        The `LoadedModel` of the bindings of sbmlsim-core, which fits a
        request of `fit/1` against the model.

    Raises:
        ValueError: if the problem has no model, more than one, or a model
            which is not loaded in sbmlsim-core, see `gaps_of`.
    """
    models = _models_of(problem)
    if len(models) != 1:
        raise ValueError(
            f"'{problem.opid}': the engine fits one model, but the fit mappings "
            f"of the problem have {len(models)} of them."
        )
    model = fit_model(problem)
    loaded = getattr(model, "loaded", None)
    if loaded is None:
        raise ValueError(
            f"'{problem.opid}': the model '{model}' is not loaded in the engine "
            f"core, so the engine cannot fit it; the problem simulates with "
            f"'{problem.engine.value}'."
        )
    return loaded


def fit_model(problem: OptimizationProblem) -> Any:
    """Get the model the engine fits, with the observables of the problem.

    A mapping which observes a formula of the selections is a column of the
    model of the fit and not of the model of the problem
    (`sbmlsim.fit.observable_sbml`), so a problem which has one is fitted
    against a model of its own, loaded from the SBML of the model of the
    problem with those observables added. The model of the problem itself is
    not touched, and it is the model of a problem without such a mapping.

    The model is kept per problem, so the SBML of a fit is read and compiled
    once however often a request is built, and a problem which is
    initialized again gets a new one.

    Args:
        problem: the initialized problem.

    Returns:
        The model, a `CoreSBMLModel`.

    Raises:
        ValueError: if the problem has no model or more than one.
    """
    models = _models_of(problem)
    if len(models) != 1:
        raise ValueError(
            f"'{problem.opid}': the engine fits one model, but the fit mappings "
            f"of the problem have {len(models)} of them."
        )
    model = models[0]
    entities = _observable_entities(problem)
    if not entities:
        return model
    kept = _FIT_MODELS.get(problem)
    if kept is not None and kept[0] == entities:
        return kept[1]
    loaded = observable_model(model, entities)
    _FIT_MODELS[problem] = (entities, loaded)
    return loaded


def fits_requests(loaded: Any) -> bool:
    """Whether a loaded model of the bindings fits a request of `fit/1` itself.

    `LoadedModel.fit` is new in sbmlsim-core 0.12.0, which the floor of
    sbmlsim is at, so what answers `False` here is a model which is not
    loaded in sbmlsim-core, and the optimizer in python fits the problem
    of such a model as it did before.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.

    Returns:
        Whether the engine of the bindings fits a request.
    """
    return loaded is not None and hasattr(loaded, "fit")


def reads_after_change(loaded: Any) -> bool:
    """Whether the bindings read the `after_change` of a curve of a request.

    `CurveRequest.after_change` is new in sbmlsim-core
    `AFTER_CHANGE_VERSION`, and a curve of the engine does not deny the
    fields it does not know: bindings before it **ignore** the field rather
    than refusing it, so every point would read the state before its change
    and the residuals would be wrong without an error. The fit of a problem
    whose data is compared with the state after a change therefore stays in
    python with such bindings, see `_gaps_at_the_changes`.

    A pre-release or development build of the version carries the field, so
    the release of the version is compared and not the version itself,
    i.e. `0.16.0.dev1` reads it.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.

    Returns:
        Whether the bindings read the field.
    """
    return _reads(loaded, AFTER_CHANGE_VERSION)


def _reads(loaded: Any, release: str) -> bool:
    """Whether the bindings of a loaded model are of a release or newer.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.
        release: the release the field was added in.

    Returns:
        Whether the installed bindings are that release or newer.
    """
    if loaded is None:
        return False
    import sbmlsim_core

    try:
        installed = Version(sbmlsim_core.__version__).release
    except InvalidVersion:
        return False
    return installed >= Version(release).release


def reads_derived(loaded: Any) -> bool:
    """Whether the bindings read the derived values of a condition.

    A condition of `fit/1` carries the values which follow the parameters of
    the fit from sbmlsim-core `DERIVED_VERSION` on (`_derived_of`). The
    request of a condition has no `deny_unknown_fields`, so an older engine
    would take the request without them and simulate the condition without
    the value, which is another fit and not an error; such a problem keeps
    the gap of the formula instead.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.

    Returns:
        Whether the bindings read them.
    """
    return _reads(loaded, DERIVED_VERSION)


def reads_change_forms(loaded: Any) -> bool:
    """Whether the bindings read a change of a linear formula as a form.

    A change of `simulation/1` carries `constant + sum(factor * entity)` which
    the engine evaluates from its own state at the change from sbmlsim-core
    `CHANGE_FORMS_VERSION` on (`executor_core.form_of`). Unlike the request of
    a condition, a `Change` does deny unknown fields, so an older engine would
    refuse the request with an error and not simulate it without the forms;
    the gap of the formula is kept for a clear message and not for safety.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.

    Returns:
        Whether the bindings read them.
    """
    return _reads(loaded, CHANGE_FORMS_VERSION)


def reads_infinite_noise(loaded: Any) -> bool:
    """Whether the bindings take an infinite standard deviation of a curve.

    The `noise` of a curve of `fit/1` is the standard deviation of every
    measurement of it, and a positive infinity is a measurement which carries
    no information from sbmlsim-core `INFINITE_NOISE_VERSION` on: its
    residual is the difference over infinity and its row of the jacobian is
    zeros, so the point stays in the residual vector of the request and adds
    nothing to the cost and nothing to the gradient. An older engine refuses
    such a request, which `_gaps_of_the_request` reports as `unsupported`, so
    the gate buys a clear message and not safety.

    Args:
        loaded: the loaded model, `None` for a model which is not loaded in
            sbmlsim-core.

    Returns:
        Whether the bindings take it.
    """
    return _reads(loaded, INFINITE_NOISE_VERSION)


def _loaded_of(problem: OptimizationProblem) -> Any:
    """Get the loaded model of a problem, `None` if it has not exactly one."""
    models = _models_of(problem)
    if len(models) != 1:
        return None
    return getattr(models[0], "loaded", None)


def gaps_of(problem: OptimizationProblem) -> list[Gap]:
    """Get what keeps the fit of a problem out of the engine.

    A problem without gaps is a fit of the engine: everything its objective
    computes is in a request of `fit/1`, so the optimizer of the engine
    minimizes the cost of the problem. A problem with gaps is fitted by the
    optimizer in python, which computes its objective itself.

    Args:
        problem: the initialized problem.

    Returns:
        The gaps of the problem, empty for a fit of the engine.
    """
    gaps: list[Gap] = []
    if problem.engine is not Engine.CORE:
        gaps.append(
            Gap(
                "engine",
                f"the problem simulates with '{problem.engine.value}', and the "
                f"fit of a request runs in sbmlsim-core",
            )
        )
        return gaps
    models = _models_of(problem)
    if len(models) != 1:
        return [
            Gap(
                "models",
                f"the mappings of the problem read {len(models)} models and a "
                f"request of '{CONTRACT}' carries one",
            )
        ]
    if not fits_requests(getattr(models[0], "loaded", None)):
        import sbmlsim_core

        return [
            Gap(
                "bindings",
                f"the bindings of sbmlsim-core {sbmlsim_core.__version__} do not "
                f"fit a request of '{CONTRACT}', which 0.12.0 is the first to do",
            )
        ]
    settings = problem.settings_initialized
    objective = _gaps_of_the_objective(problem)
    if objective:
        return objective
    if settings.objective is ObjectiveType.LEAST_SQUARES:
        # a likelihood applies neither the loss function nor the residual
        # type of the settings, its residual is the one of its noise model
        if settings.loss_function.name != "LINEAR":
            return [
                Gap(
                    "loss_function",
                    f"the residuals of the fit go through the loss function "
                    f"'{settings.loss_function.name.lower()}', which the engine "
                    f"does not apply",
                )
            ]
        if settings.residual not in RESIDUALS:
            return [
                Gap(
                    "residual",
                    f"the residuals of the fit are "
                    f"'{settings.residual.name.lower()}', which subtract a "
                    f"baseline of the simulation",
                )
            ]
    models = _models_of(problem)
    if len(models) != 1:
        gaps.append(
            Gap(
                "models",
                f"the fit mappings of the problem have {len(models)} models, and "
                f"a request fits one",
            )
        )
    for parameter in problem.parameters:
        if parameter.is_external:
            gaps.append(
                Gap(
                    "parameter",
                    f"the parameter '{parameter.pid}' writes "
                    f"'{parameter.target_id}', which is no entity of the model",
                )
            )
    gaps.extend(_gaps_of_the_targets(problem))
    gaps.extend(_gaps_of_the_bounds(problem))
    gaps.extend(_gaps_of_the_values(problem))
    gaps.extend(_gaps_at_the_folds(problem))
    for group, mappings in enumerate(problem.mapping_groups):
        training = [
            k for k in mappings if problem.mapping_kinds[k] is MappingKind.TRAINING
        ]
        if not training:
            continue
        gaps.extend(_gaps_of_a_group(problem, group, training))
    if gaps:
        return gaps
    return _gaps_of_the_request(problem)


def fixed_noise(problem: OptimizationProblem, mapping: int) -> np.ndarray | None:
    """Get the scale of the noise of a fit mapping when the fit does not move it.

    The engine divides the residual of a point by a number, so a noise model
    is a curve of a request only when its scale is the same at every point of
    the search: a formula which reads the simulation (`NoiseModel.observable`,
    a selection of it) or a parameter of the fit has a scale which follows the
    parameters and is none.

    A noise parameter which is no parameter of the fit is its nominal value,
    which is fixed, and a placeholder value which is a formula reads the
    selections of the simulation and is not.

    Args:
        problem: the initialized problem.
        mapping: index of the fit mapping.

    Returns:
        The scale of the noise of every measurement, `None` when it follows
        the parameters of the fit or the simulation, or when the noise model
        is not defined at the nominal values.
    """
    from sbmlsim.fit.petab_v2.likelihood import noise_model_of, noise_values

    noise = noise_model_of(problem, mapping)
    if noise.observable is not None or problem.noise_selections[mapping]:
        return None
    if any(
        isinstance(value, str)
        for values in noise.placeholder_values
        for value in values
    ):
        return None
    pids = set(problem.pids)
    if any(parameter.pid in pids for parameter in noise.parameters):
        return None
    if any(symbol in pids for symbol in noise.symbols):
        return None
    size = len(problem.y_references[mapping])
    try:
        return np.asarray(noise_values(noise, size=size), dtype=float)
    except ValueError:
        return None


def _gaps_of_the_objective(problem: OptimizationProblem) -> list[Gap]:
    """Get what keeps the objective of a problem out of a request.

    Weighted least squares, the default, is the objective of a request. A
    likelihood is one too when its residuals are the ones the engine
    computes: the negative log-likelihood of a measurement is
    `0.5 res**2 + t` with `res` the normalized residual and `t` the term of
    the scale of its noise (`likelihood_residuals`), so a noise model whose
    scale the fit does not move (`fixed_noise`) has a constant `t`, which
    shifts the cost and has no derivative, and a normal `res` is the residual
    of a curve of `noise` that scale. A log-normal `res` is that of a curve
    whose column is the logarithm of the observable, which the model of the
    fit carries (`_log_observable`), and whose values are the logarithms of
    the measurements.

    The constant of the scales is `likelihood_constant`, which the cost of a
    run of the engine is shifted by, so the cost of such a fit is the
    negative log-likelihood as a fit in python reports it.

    A posterior adds the residuals of the priors, which are functions of the
    parameters alone and no part of a request, a Laplace distribution has a
    residual no curve expresses (`LIKELIHOOD_DISTRIBUTIONS`), and a noise
    whose scale is a parameter of the fit, e.g. `sd_obs`, or which reads the
    simulation is one the engine would hold at the value it started from.

    Args:
        problem: the initialized problem.

    Returns:
        The gaps, empty for an objective the engine minimizes.
    """
    objective = problem.settings_initialized.objective
    if objective is ObjectiveType.LEAST_SQUARES:
        return []
    if objective is not ObjectiveType.LIKELIHOOD:
        return [
            Gap(
                "objective",
                f"the fit minimizes the objective '{objective.value}', whose "
                f"priors are functions of the parameters and no residual of a "
                f"request",
            )
        ]
    from sbmlsim.fit.petab_v2.likelihood import noise_model_of

    gaps: list[Gap] = []
    for mapping in problem.indices(MappingKind.TRAINING):
        key = f"{problem.experiment_keys[mapping]}.{problem.mapping_keys[mapping]}"
        noise = noise_model_of(problem, mapping)
        distribution = str(noise.distribution.value)
        if distribution not in LIKELIHOOD_DISTRIBUTIONS:
            gaps.append(
                Gap(
                    "objective",
                    f"the noise of '{key}' is '{distribution}', whose residual "
                    f"is no difference divided by a standard deviation",
                )
            )
            continue
        if fixed_noise(problem, mapping) is None:
            gaps.append(
                Gap(
                    "objective",
                    f"the scale of the noise of '{key}', '{noise.formula}', "
                    f"follows the parameters of the fit or its simulation, and "
                    f"a curve of a request divides by a number",
                )
            )
    return gaps


def likelihood_constant(problem: OptimizationProblem) -> float:
    """Get what the cost of a run of the engine misses for a likelihood.

    The engine minimizes the half sum of the squares of the residuals of the
    curves, which are the residuals of the data of the likelihood. The
    negative log-likelihood is
    `0.5 sum(data**2) + 0.5 sum(scale**2) - 0.5 sum(offsets)` with the
    residuals of the scale and the offsets of `likelihood_residuals`, both of
    which are constant for a noise the fit does not move (`fixed_noise`), so
    this is the constant a run of the engine is shifted by.

    Args:
        problem: the initialized problem, without the gaps of its objective.

    Returns:
        The constant, `0.0` for a fit of least squares.
    """
    if problem.settings_initialized.objective is ObjectiveType.LEAST_SQUARES:
        return 0.0
    from sbmlsim.fit.petab_v2.likelihood import (
        likelihood_offsets,
        likelihood_residuals,
        noise_model_of,
    )

    constant = 0.0
    for mapping in problem.indices(MappingKind.TRAINING):
        noise = noise_model_of(problem, mapping)
        measurement = np.asarray(problem.y_references[mapping], dtype=float)
        sigma = fixed_noise(problem, mapping)
        if sigma is None:
            raise ValueError(
                f"'{problem.opid}': the noise of the mapping "
                f"'{problem.mapping_keys[mapping]}' is not fixed, see "
                f"'fixed_noise'."
            )
        # the residuals of the scale do not depend on the simulation, so the
        # measurement itself stands in for it here
        _, scale = likelihood_residuals(
            measurement, measurement, sigma, noise.distribution
        )
        offsets = likelihood_offsets(measurement, noise.distribution)
        constant += 0.5 * float(np.sum(np.square(scale))) - 0.5 * float(np.sum(offsets))
    return constant


def _gaps_of_the_bounds(problem: OptimizationProblem) -> list[Gap]:
    """Get the gaps of the bounds of the parameters of a problem.

    An optimizer of the engine searches between finite, increasing bounds,
    and the lower bound of a parameter of a logarithmic scale is positive
    (`check_parameters` of the engine). A problem whose bounds the engine
    refuses is fitted in python, e.g. one of a parameter of a prior which has
    no bounds of its own.
    """
    gaps: list[Gap] = []
    factors = _factors_of(problem)
    scales = problem.scales_initialized
    for k, parameter in enumerate(problem.parameters):
        lower = float(parameter.lower_bound) * factors[k]
        upper = float(parameter.upper_bound) * factors[k]
        if not math.isfinite(lower) or not math.isfinite(upper):
            gaps.append(
                Gap(
                    "parameter",
                    f"the parameter '{parameter.pid}' has the bounds "
                    f"[{lower}, {upper}], and an optimizer of the engine searches "
                    f"between finite bounds",
                )
            )
        elif lower >= upper:
            gaps.append(
                Gap(
                    "parameter",
                    f"the parameter '{parameter.pid}' has the bounds "
                    f"[{lower}, {upper}], which are not increasing",
                )
            )
        elif SCALES[scales[k]] != "lin" and lower <= 0.0:
            gaps.append(
                Gap(
                    "parameter",
                    f"the parameter '{parameter.pid}' is searched on the scale "
                    f"'{SCALES[scales[k]]}' and has the lower bound {lower}, which "
                    f"is not positive",
                )
            )
    return gaps


def _gaps_of_the_request(problem: OptimizationProblem) -> list[Gap]:
    """Get what the engine refuses in the request of the fit of a problem.

    The checks above are the ones of the fit, of its data and of its
    parameters. What is left is everything the engine says about the
    simulation of a condition, which is the authority on it and which no
    check here repeats: a formula before the initialization the engine cannot
    fold into a number, a selection which is no entity of the model, a
    formula at a change which reads no output of the engine, and whatever a
    later release
    adds (the module docstring of `sbmlsim.simulator.executor_core` lists
    them). The request of the fit is built and what it raises is the gap, so
    a problem the engine would refuse is fitted in python instead of failing.

    Args:
        problem: the problem, whose other gaps are closed.

    Returns:
        The gap the engine names, empty for a request it accepts.
    """
    try:
        fit_request(problem)
    except UnsupportedModelError as err:
        return [Gap("unsupported", f"the engine does not simulate the fit: {err}")]
    except (ValueError, KeyError, TypeError) as err:
        return [
            Gap(
                "request",
                f"the request of the fit cannot be built: {type(err).__name__}: {err}",
            )
        ]
    return []


def fit_request(
    problem: OptimizationProblem,
    *,
    starts: Sequence[Sequence[float] | np.ndarray] = (),
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    max_iterations: int = 100,
    max_seconds: float | None = None,
    gradient_tolerance: float | None = None,
    step_tolerance: float | None = None,
    cost_tolerance: float | None = None,
    evolution: Mapping[str, Any] | None = None,
    identifiability: Mapping[str, Any] | None = None,
    jacobian: str | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Build the request of the contract `fit/1` of the fit of a problem.

    Args:
        problem: the initialized problem, without gaps (`gaps_of`).
        starts: the points the fit starts from, the values of the parameters
            in their own units, one run of the fit per point; the start of
            the parameters themselves without.
        algorithm: the optimizer of the fit.
        max_iterations: the largest number of iterations of a run.
        max_seconds: the time the fit may take, after which the optimizer
            stops at its best point; no budget without one.
        gradient_tolerance: the tolerance of the scaled gradient a step
            converges at; the one of the optimizer without.
        step_tolerance: the tolerance of the step a run converges at,
            relative to the point it steps from, the `xtol` of
            `scipy.optimize.least_squares`; no test of the step without one,
            which spends the last iterations of a run on the noise of the
            integrator (#607).
        cost_tolerance: the tolerance of the change of the cost a run
            converges at, relative to the cost before the step, the `ftol` of
            `scipy.optimize.least_squares`; no test of the cost without one.
        evolution: the population of a differential evolution and how it is
            drawn, the fields of `Evolution` of the engine, e.g. the `seed`.
        identifiability: the identifiability of the optimum, the fields of
            `Identifiability` of the engine (`parameters`, `walk`,
            `profiles`, `fisher`); none of it without.
        jacobian: where the jacobian of a step comes from, `"auto"`,
            `"sensitivities"` or `"difference_quotient"`; `"auto"` without.
        threads: the threads of the batch of an evaluation; the CPUs of the
            machine without.

    Returns:
        The request, a dictionary of the fields of the contract.

    Raises:
        ValueError: if the problem is not initialized, if it has no model or
            more than one, or if a start is not one value per parameter.
    """
    if not problem.is_initialized:
        raise ValueError(
            f"'{problem.opid}': the request of a fit needs the resolved data "
            f"of the problem, which 'initialize' builds."
        )
    conditions, curves = _conditions_and_curves(problem, fit_model(problem))
    optimizer: dict[str, Any] = {
        "algorithm": _algorithm(algorithm, evolution),
        "max_iterations": int(max_iterations),
    }
    if max_seconds is not None:
        optimizer["max_seconds"] = float(max_seconds)
    for name, tolerance in (
        ("gradient_tolerance", gradient_tolerance),
        ("step_tolerance", step_tolerance),
        ("cost_tolerance", cost_tolerance),
    ):
        if tolerance is not None:
            optimizer[name] = float(tolerance)
    request: dict[str, Any] = {
        "contract": CONTRACT,
        "parameters": _parameters(problem),
        "conditions": conditions,
        "curves": curves,
        "optimizer": optimizer,
    }
    if len(starts):
        request["starts"] = [_start(problem, start) for start in starts]
    if identifiability is not None:
        request["identifiability"] = dict(identifiability)
    if jacobian is not None:
        request["jacobian"] = jacobian
    if threads is not None:
        request["threads"] = int(threads)
    return request


def _models_of(problem: OptimizationProblem) -> list[Any]:
    """Get the models of the fit mappings of a problem, every one of them once."""
    return list({id(model): model for model in problem.models}.values())


def _parameters(problem: OptimizationProblem) -> list[dict[str, Any]]:
    """Get the parameters of a request, the search space of the problem.

    The values of a parameter are the ones of the model, which the engine
    sets: the factor of its target carries the units of the parameter into
    the model, see `_factors_of`.

    The start of a parameter is `problem.xmodel`, the value of the model as
    loaded, clamped into its bounds: the engine refuses a start outside them
    and a nominal value of a PEtab problem may be outside them, which PEtab
    reports and answers by starting at the bound (case 0026 of the test
    suite, whose `initial_A1` is 0.5 in a `[1, 10]`). A start inside its
    bounds is left as it is, so this changes nothing for a problem whose
    model is inside them, which is every other one.
    """
    factors = _factors_of(problem)
    scales = problem.scales_initialized
    parameters = []
    for k, parameter in enumerate(problem.parameters):
        factor = factors[k]
        lower = float(parameter.lower_bound) * factor
        upper = float(parameter.upper_bound) * factor
        parameters.append(
            {
                "id": parameter.target_id,
                "start": min(max(float(problem.xmodel[k]) * factor, lower), upper),
                "lower": lower,
                "upper": upper,
                "scale": SCALES[scales[k]],
            }
        )
    return parameters


def _start(
    problem: OptimizationProblem, start: Sequence[float] | np.ndarray
) -> list[float]:
    """Get a start of a request, the values of the parameters in the model."""
    values = np.asarray(start, dtype=float)
    if values.shape != (len(problem.parameters),):
        raise ValueError(
            f"'{problem.opid}': a start of the fit is one value per parameter, "
            f"but '{values.shape}' values are given for "
            f"{len(problem.parameters)} parameters."
        )
    return [
        float(value * factor)
        for value, factor in zip(values, _factors_of(problem), strict=True)
    ]


def _factors_of(problem: OptimizationProblem) -> list[float]:
    """Get the factor of every parameter into the units of its target.

    The parameters of a problem are in their own units and the factor of a
    target carries a value into the model (`OptimizationProblem.evaluated_plan`),
    which is where the values of a request are: the engine sets the entity of
    the model.
    """
    factors = [1.0] * len(problem.parameters)
    for targets in problem._group_targets:
        for _, index, factor in targets:
            factors[index] = float(factor)
    return factors


def _algorithm(
    algorithm: OptimizationAlgorithmType, evolution: Mapping[str, Any] | None
) -> Any:
    """Get the algorithm of the optimizer of a request."""
    name = ALGORITHMS[algorithm]
    if algorithm is OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION:
        return {name: dict(evolution or {})}
    return name


def _conditions_and_curves(
    problem: OptimizationProblem, model: CoreSBMLModel
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Get the conditions and the curves of a request.

    A condition is a simulation group of the problem which has training data,
    the request of its plan with the values of the parameters at the start of
    the fit; a curve is a training fit mapping, the data of the problem with
    the weight of every point of it.

    The curves are in the order of the fit mappings of the problem, which is
    the order of the residuals it computes: the residuals of the engine are
    the ones of the problem in the same places, and a report reads them by
    the index of a mapping.
    """
    from sbmlsim.simulator.executor_core import request_of

    loaded = _loaded_of(problem)

    settings = model.request_settings()
    x = np.asarray(problem.xmodel, dtype=float)
    parts = {k: _parts_of(problem, k) for k in problem.indices(MappingKind.TRAINING)}
    conditions: list[dict[str, Any]] = []
    of_the_group: dict[int, str] = {}
    for group, mappings in enumerate(problem.mapping_groups):
        training = [
            k for k in mappings if problem.mapping_kinds[k] is MappingKind.TRAINING
        ]
        if not training:
            continue
        plan = problem.evaluated_plan(group, x, [])
        # a value which follows a parameter of the fit is no number of the
        # request: the engine evaluates it before every simulation
        plan, derived = (
            _derived_of(problem, plan, model) if reads_derived(loaded) else (plan, [])
        )
        selections = list(problem._selections_of(group, training))
        # the observable of a mapping is a column of the model of the fit, so
        # the condition selects it instead of what its formula reads, and a
        # mapping of log-normal noise reads its logarithmic column
        for k in training:
            for part in parts[k]:
                column = part.column or _observable_column(
                    problem, group, problem.yid_observable[k]
                )
                if _is_log_normal(problem, k):
                    column = _log_observable(column)
                if column not in selections:
                    selections.append(column)
        condition = _condition_id(problem, group, training[0])
        of_the_group[group] = condition
        entry: dict[str, Any] = {
            "id": condition,
            "request": request_of(
                plan, model, selections, settings, forms=reads_change_forms(loaded)
            ),
        }
        if derived:
            entry["derived"] = derived
        conditions.append(entry)
    groups = {k: group for group, ks in enumerate(problem.mapping_groups) for k in ks}
    curves = [
        _curve(problem, k, of_the_group[groups[k]], groups[k], part, len(parts[k]) > 1)
        for k in problem.indices(MappingKind.TRAINING)
        for part in parts[k]
    ]
    return conditions, curves


def _condition_id(problem: OptimizationProblem, group: int, mapping: int) -> str:
    """Get the id of the condition of a group, which its curves read it by."""
    return (
        f"{problem.experiment_keys[mapping]}|{problem.simulation_keys[mapping]}|{group}"
    )


def _curve(
    problem: OptimizationProblem,
    mapping: int,
    condition: str,
    group: int,
    part: _Part,
    numbered: bool,
) -> dict[str, Any]:
    """Get the curve of a training fit mapping, its data and its weights.

    The engine divides the residual of a point by its standard deviation, so
    the weight `w` of the problem is `1/sqrt(w)` here, and the mean of the
    data of a normalized residual goes into it as well.

    A curve measures at the output times of its condition, which are the ones
    of the plan of its group: `Plan.time_shift` is applied to the time of a
    result on the way back (`sbmlsim.simulator.executor_core.request_of`), so
    the time of a point here is the output time of the plan it reads
    (`OptimizationProblem._x_plan`, the merged time of `merge_times`) and not
    the time of the data, which the engine would not find for a simulation
    whose time is shifted (#613).

    A point which is compared with the state after a change carries its flag
    (`after_change_of`), and a curve no point of which is gets no field at
    all, which is the request every release of the engine reads.

    Args:
        problem: the initialized problem.
        mapping: index of the training fit mapping.
        condition: the id of the condition of its group.
        group: the simulation group of the mapping.
        part: the part of the mapping the curve is (`_parts_of`), which is
            the whole of it for a mapping which is one curve.
        numbered: whether the mapping has more than one part, whose curves
            carry the positions of their points in their id, so that the id
            of a curve of a request is unique.

    Returns:
        The curve, a dictionary of the fields of the contract.
    """
    at = list(part.at)
    rows = problem._x_plan[mapping]
    times = (
        np.asarray(problem.x_references[mapping], dtype=float)
        if rows is None
        else np.asarray(rows, dtype=float)
    )
    values = np.asarray(problem.y_references[mapping], dtype=float)
    column = part.column or _observable_column(
        problem, group, problem.yid_observable[mapping]
    )
    noise = _noise_of(problem, mapping)
    if _is_log_normal(problem, mapping):
        # the residual of a log-normal noise is the difference of the
        # logarithms, which the logarithmic column of the model gives
        column = _log_observable(column)
        values = np.log(values)
    key = problem.mapping_keys[mapping]
    curve: dict[str, Any] = {
        "id": f"{key}[{at[0]}:{at[-1] + 1}]" if numbered else key,
        "condition": condition,
        "column": column,
        "times": [float(time) for time in times[at]],
        "values": [float(value) for value in values[at]],
        "noise": [float(noise[point]) for point in at],
    }
    after = after_change_of(problem, group, mapping)
    if after is not None:
        curve["after_change"] = [bool(flag) for flag in after[at]]
    return curve


def _noise_of(problem: OptimizationProblem, mapping: int) -> np.ndarray:
    """Get the standard deviation of every measurement of a fit mapping.

    The engine divides the residual of a point by this number. For weighted
    least squares it is the weight of the point, `1/sqrt(w)` of the problem,
    with the mean of the data of a normalized residual in it; for a
    likelihood it is the scale of the noise model, which the weights, the
    residual type and the loss function of the settings play no part in, as
    they do not in python (`OptimizationProblem.residuals`).

    Args:
        problem: the initialized problem.
        mapping: index of the fit mapping.

    Returns:
        The standard deviation of every measurement of the mapping.

    Raises:
        ValueError: if the noise of a likelihood is not fixed, which
            `_gaps_of_the_objective` reports as a gap.
    """
    if problem.settings_initialized.objective is not ObjectiveType.LEAST_SQUARES:
        sigma = fixed_noise(problem, mapping)
        if sigma is None:
            raise ValueError(
                f"'{problem.opid}': the noise of the mapping "
                f"'{problem.mapping_keys[mapping]}' is not fixed, see "
                f"'fixed_noise'."
            )
        return sigma
    weights = np.asarray(problem.weights[mapping], dtype=float)
    reference = 1.0
    if problem.residual is ResidualType.NORMALIZED:
        # the mean of the whole mapping, which is what the problem divides by
        reference = float(np.mean(problem.y_references[mapping]))
    # a weight of exactly zero is a measurement which carries no information,
    # whose standard deviation is a positive infinity: the engine takes it
    # from `INFINITE_NOISE_VERSION` on, see `reads_infinite_noise`
    with np.errstate(divide="ignore"):
        return reference / np.sqrt(weights)


def _gaps_of_the_targets(problem: OptimizationProblem) -> list[Gap]:
    """Get the gaps of the targets the parameters of a problem write.

    The engine sets one entity of the model per parameter, the same one in
    every condition of the fit: a parameter which writes two entities, which
    writes none in a group, or which writes different ones in two groups is
    no parameter of a request.
    """
    gaps: list[Gap] = []
    expected: dict[int, tuple[str, float]] = {}
    for group, targets in enumerate(problem._group_targets):
        if not problem.mapping_groups[group]:
            continue
        seen: dict[int, tuple[str, float]] = {}
        for target, index, factor in targets:
            if index in seen:
                gaps.append(
                    Gap(
                        "parameter",
                        f"the parameter '{problem.parameters[index].pid}' writes "
                        f"'{seen[index][0]}' and '{target}' in the same "
                        f"simulation, and a parameter of a request writes one "
                        f"entity of the model",
                    )
                )
            seen[index] = (target, float(factor))
        if not expected:
            expected = seen
            continue
        for index, (target, factor) in seen.items():
            if index not in expected:
                continue
            if expected[index] != (target, factor):
                gaps.append(
                    Gap(
                        "parameter",
                        f"the parameter '{problem.parameters[index].pid}' writes "
                        f"'{expected[index][0]}' in one simulation of the fit and "
                        f"'{target}' in another one",
                    )
                )
        if set(seen) != set(expected):
            missing = sorted(
                problem.parameters[index].pid for index in set(expected) ^ set(seen)
            )
            gaps.append(
                Gap(
                    "parameter",
                    f"the parameters {missing} write an entity of the model in "
                    f"one simulation of the fit and none in another one, where "
                    f"the engine sets every parameter in every condition",
                )
            )
    return gaps


def _gaps_of_a_group(
    problem: OptimizationProblem, group: int, training: Sequence[int]
) -> list[Gap]:
    """Get the gaps of a simulation group of a problem and of its training data."""
    gaps: list[Gap] = []
    if problem._group_points[group] is not None:
        gaps.append(
            Gap(
                "scan",
                f"the simulation of the group {group} is a scan, whose points "
                f"are simulations of their own",
            )
        )
    if problem._group_graphs[group] is not None:
        gaps.append(
            Gap(
                "observable",
                f"the data of the group {group} is read from observables of its "
                f"experiment which need the output of its simulation, a value "
                f"per simulation, a reduction, a `PK` or a `Custom` observable, "
                f"and are no column of it",
            )
        )
    if problem.group_derived[group]:
        gaps.append(
            Gap(
                "derived",
                f"the simulation of the group {group} has derived changes, which "
                f"read the values of the parameters",
            )
        )
    for k in training:
        gaps.extend(_gaps_of_a_mapping(problem, group, k))
    return gaps


def _gaps_of_a_mapping(
    problem: OptimizationProblem, group: int, mapping: int
) -> list[Gap]:
    """Get the gaps of the data of a training fit mapping of a problem."""
    key = problem.mapping_keys[mapping]
    gaps: list[Gap] = []
    observable = _gap_of_an_observable(problem, mapping)
    if observable is not None:
        gaps.append(observable)
        return gaps
    if problem._rows[mapping] is None:
        gaps.append(
            Gap(
                "x",
                f"the x of the data of the mapping '{key}' is no time of its "
                f"simulation, and a curve of a request measures over the time",
            )
        )
        return gaps
    weights = np.asarray(problem.weights[mapping], dtype=float)
    # a weight of exactly zero is a measurement which carries no information,
    # whose standard deviation `_noise_of` gives as a positive infinity: the
    # engine takes that from `INFINITE_NOISE_VERSION` on and its residual and
    # its row of the jacobian are zero. Every other weight which is no
    # standard deviation stays a gap: an infinite weight is an error of
    # exactly zero, whose noise is `0.0`, and a negative one has no square
    # root. `FitMapping.weight` of zero is the second source of a zero
    # weight, which zeroes every point of its mapping.
    unusable = np.isnan(weights) | np.isinf(weights) | (weights < 0.0)
    if np.any(unusable):
        gaps.append(
            Gap(
                "weights",
                f"a point of the mapping '{key}' has the weight "
                f"'{float(weights[int(np.argmax(unusable))])}', which is no "
                f"standard deviation of a curve",
            )
        )
    elif np.any(weights == 0.0) and not reads_infinite_noise(_loaded_of(problem)):
        import sbmlsim_core

        gaps.append(
            Gap(
                "weights",
                f"a point of the mapping '{key}' has the weight '0.0', a "
                f"measurement which carries no information, whose standard "
                f"deviation is the infinity which the bindings of sbmlsim-core "
                f"{sbmlsim_core.__version__} do not take, and "
                f"{INFINITE_NOISE_VERSION} is the first which does",
            )
        )
    if problem.residual is ResidualType.NORMALIZED:
        mean = float(np.mean(problem.y_references[mapping]))
        if mean == 0.0 or not math.isfinite(mean):
            gaps.append(
                Gap(
                    "weights",
                    f"the data of the mapping '{key}' has the mean '{mean}', "
                    f"which its normalized residuals divide by",
                )
            )
    gaps.extend(_gaps_at_the_changes(problem, group, mapping))
    return gaps


def after_change_of(
    problem: OptimizationProblem, group: int, mapping: int
) -> np.ndarray | None:
    """Get which points of a fit mapping read the state after a change.

    The output of a simulation has the state before a change and the one
    after it at the time of the change. A point of a request reads the first
    of those rows, the state before the change, and one whose flag is true
    reads the last row of its time, the state after it
    (`CurveRequest.after_change` of the engine): that is
    `DataAtChangeType.AFTER` for every point and the default for the first
    point of a curve, see `OptimizationProblem._first_after`.

    Args:
        problem: the initialized problem.
        group: the simulation group of the mapping.
        mapping: index of the fit mapping.

    Returns:
        One flag per point of the mapping, in the order of its data, or
        `None` when no point of it reads the state after a change, which is
        every mapping of a simulation without changes.
    """
    times = np.asarray(problem.x_references[mapping], dtype=float)
    plan = problem.plans[group]
    if plan is None or not plan.change_times or times.size == 0:
        return None
    at_change = plan.at_change(times - plan.time_shift)
    before = problem._before_change[mapping]
    after = at_change >= 0 if before is None else (at_change >= 0) & (before < 0)
    if not bool(np.any(after)):
        return None
    return np.asarray(after, dtype=bool)


def _gap_of_an_observable(problem: OptimizationProblem, mapping: int) -> Gap | None:
    """Get the gap of the observable of a training fit mapping.

    An observable which is a formula of the selections is written into the
    model of the fit as a parameter of an assignment rule
    (`sbmlsim.fit.observable_sbml`), so it is a column of a simulation like
    any other and no gap. What is still a gap is a placeholder whose value
    differs between the points of one curve, which is one value of the model
    and several of the data, and a formula the model cannot carry, e.g. one
    which reads the concentration of a species it has not.

    Args:
        problem: the initialized problem.
        mapping: index of the training fit mapping.

    Returns:
        The gap, or `None` for an observable the model of the fit carries.
    """
    observable = problem.observable_models[mapping]
    if observable is None:
        return None
    key = problem.mapping_keys[mapping]
    model = _models_of(problem)[0] if len(_models_of(problem)) == 1 else None
    if model is None:
        return None
    rows = _placeholder_rows(observable, problem.x_references[mapping])
    if rows is None:
        return Gap(
            "observable",
            f"the observable of the mapping '{key}' has the placeholders "
            f"{list(observable.placeholders)} without a value per measurement",
        )
    try:
        for values in rows.values():
            add_observable(
                observable_id(key), observable, values, model.symbols, Entities()
            )
    except ValueError as err:
        return Gap(
            "observable",
            f"the observable '{observable.formula}' of the mapping '{key}' is no "
            f"entity of the model of the fit: {err}",
        )
    return None


def _placeholder_rows(
    observable: ObservableModel, times: Any
) -> dict[tuple[float | str, ...], dict[str, float | str]] | None:
    """Get the values of the placeholders of an observable, by the row they are.

    A placeholder has a value per measurement and the model of the fit
    carries one value, so an observable whose placeholders differ between the
    measurements of one curve becomes one observable per row of values, each
    read by the points which have that row (`_parts_of`).

    Args:
        observable: the observable of a fit mapping.
        times: the times of the data of the mapping, whose number the rows
            have to match.

    Returns:
        The value of every placeholder by the row it belongs to, in the order
        the rows first appear, a single empty row for an observable without
        placeholders, or `None` if the observable has placeholders without a
        value per measurement.
    """
    if not observable.placeholders:
        return {(): {}}
    rows = observable.placeholder_values
    if len(rows) != len(times):
        return None
    out: dict[tuple[float | str, ...], dict[str, float | str]] = {}
    for row in rows:
        key = tuple(row)
        if key not in out:
            out[key] = dict(zip(observable.placeholders, row, strict=True))
    return out


def _observable_names(
    key: str, rows: Mapping[tuple[float | str, ...], object]
) -> dict[tuple[float | str, ...], str]:
    """Get the column of every row of the placeholder values of an observable."""
    if len(rows) == 1:
        return {next(iter(rows)): observable_id(key)}
    return {row: f"{observable_id(key)}__r{n}" for n, row in enumerate(rows)}


def _observable_entities(problem: OptimizationProblem) -> Entities:
    """Get the observables of the training mappings of a problem.

    An observable whose placeholders have more than one row of values over
    the points of its curve becomes one observable per row, which the parts
    of the mapping read (`_parts_of`).

    Args:
        problem: the initialized problem, without the gap `observable`.

    Returns:
        The entities of the model of the fit, empty for a problem none of
        whose mappings observes a formula.
    """
    entities = Entities()
    models = _models_of(problem)
    if len(models) != 1:
        return entities
    symbols = models[0].symbols
    for mapping in problem.indices(MappingKind.TRAINING):
        observable = problem.observable_models[mapping]
        if observable is None:
            continue
        rows = _placeholder_rows(observable, problem.x_references[mapping])
        if rows is None:
            continue
        key = problem.mapping_keys[mapping]
        names = _observable_names(key, rows)
        for row, values in rows.items():
            add_observable(names[row], observable, values, symbols, entities)
    for k_group, graph in enumerate(problem._group_pointwise):
        if graph is None:
            continue
        # the nodes of the graph are ordered, so a formula reads only the
        # observables before it, whose parameters `names` carries: nothing of
        # a formula is substituted as text. `FormulaNode.factor` is the factor
        # from the derived unit of a formula into its declared one, which
        # `evaluate` applies to its values (`value *= factor`), so the
        # parameter carries the same single multiplication and its column is
        # the value of python.
        names: dict[str, str] = {}
        for node in graph.nodes:
            if not isinstance(node, FormulaNode):  # pragma: no cover - pointwise
                continue
            formula = node.formula
            if node.factor != 1.0:
                formula = f"({formula}) * {node.factor!r}"
            target = _group_observable(k_group, node.id)
            add_formula(target, formula, symbols, entities, names=dict(names))
            names[node.id] = target
    for mapping in problem.indices(MappingKind.TRAINING):
        if not _is_log_normal(problem, mapping):
            continue
        for part in _parts_of(problem, mapping):
            column = part.column or problem.yid_observable[mapping]
            add_formula(_log_observable(column), f"log({column})", symbols, entities)
    return entities


def _is_log_normal(problem: OptimizationProblem, mapping: int) -> bool:
    """Whether the noise of a fit mapping of a likelihood is log-normal.

    The residual of a log-normal noise is the difference of the logarithms,
    which the curve of the mapping reads from a logarithmic column of the
    model (`_log_observable`); a fit of least squares has no noise model.
    """
    if problem.settings_initialized.objective is ObjectiveType.LEAST_SQUARES:
        return False
    from sbmlsim.fit.petab_v2.likelihood import noise_model_of

    return bool(noise_model_of(problem, mapping).distribution.is_log)


def _group_observable(group: int, observable: str) -> str:
    """Get the id of the parameter of an observable of a simulation group.

    Two experiments of one fit may define an observable of the same id with
    different formulas, e.g. a fraction of a dose which each of them derives
    from its own selections, so the parameter of the model of the fit carries
    the group of the observable and not its id alone.

    Args:
        group: index of the simulation group of the observable.
        observable: the id of the observable of the experiment.

    Returns:
        The id of its parameter in the model of the fit.
    """
    return f"{OBSERVABLE_PREFIX}g{group}__{to_sid(observable)}"


def _observable_column(problem: OptimizationProblem, group: int, column: str) -> str:
    """Get the column of the model of the fit a curve of a group reads.

    A column which is an observable of the experiment of the group is the
    parameter the model of the fit carries for it (`_group_observable`);
    every other column is a selection of the model and is itself.
    """
    graph = problem._group_pointwise[group]
    if graph is None:
        return column
    if column not in {node.id for node in graph.nodes}:
        return column
    return _group_observable(group, column)


def _log_observable(column: str) -> str:
    """Get the id of the observable of the model which is the logarithm of a column.

    The column is a selection of the model (`X`, `[X]`) or the observable of
    a mapping which the model of the fit carries already, and the id is one
    SId whatever it is, so a concentration and an observable of a part both
    get one.
    """
    return f"{OBSERVABLE_PREFIX}{LOG_PREFIX}{to_sid(column)}"


@dataclass(frozen=True)
class _Part:
    """One curve of a training fit mapping.

    A mapping is one curve of the request, except for an observable whose
    placeholders have a value per measurement: the model of the fit carries
    one value per observable, so such a mapping is one curve per run of
    consecutive points which share their values, each reading the observable
    of those values. The runs are in the order of the points, so the
    residuals of the request stay the residuals of the problem in their
    order.

    Attributes:
        column: the column of the simulation the curve reads, `None` for the
            selection of the mapping itself.
        at: the positions of the points of the curve in the data of the
            mapping, consecutive and increasing.
    """

    column: str | None
    at: tuple[int, ...]


def _parts_of(problem: OptimizationProblem, mapping: int) -> list[_Part]:
    """Get the curves of a training fit mapping of a problem.

    Args:
        problem: the initialized problem, without the gap `observable`.
        mapping: index of the training fit mapping.

    Returns:
        The parts, one for a mapping which is one curve.
    """
    points = tuple(range(len(problem.x_references[mapping])))
    observable = problem.observable_models[mapping]
    if observable is None:
        return [_Part(None, points)]
    rows = _placeholder_rows(observable, problem.x_references[mapping])
    if rows is None:
        return [_Part(None, points)]
    names = _observable_names(problem.mapping_keys[mapping], rows)
    if not observable.placeholders:
        return [_Part(names[()], points)]
    parts: list[_Part] = []
    at: list[int] = []
    row: tuple[float | str, ...] | None = None
    for point, values in enumerate(observable.placeholder_values):
        current = tuple(values)
        if row is not None and current != row:
            parts.append(_Part(names[row], tuple(at)))
            at = []
        row = current
        at.append(point)
    if row is not None:
        parts.append(_Part(names[row], tuple(at)))
    return parts


def _derived_of(
    problem: OptimizationProblem, plan: Plan, model: CoreSBMLModel
) -> tuple[Plan, list[dict[str, Any]]]:
    """Get the plan of a condition without its derived values, and those values.

    A pre-initialization assignment whose formula is a linear form over the
    parameters of the fit cannot be folded into a number: the number would
    hold the value where the fit started while the optimizer moves the
    parameter, so the executor refuses it (`sbmlsim.simulator.executor_core`,
    the gap `unsupported`). It is taken out of the plan and given to the
    engine as a derived value of the condition instead, which the engine
    evaluates before every simulation.

    Only the pre-initialization of the plan itself is read this way; an
    assignment of its steady state and a formula which is no linear form over
    the parameters stay in the plan and keep their gap.

    Args:
        problem: the initialized problem.
        plan: the plan of the condition.
        model: the model of the fit, which gives the values of the symbols a
            formula reads and the fit does not estimate.

    Returns:
        The plan without the assignments which became derived values, which
        is the given one when there are none, and those values as the
        dictionaries of the contract.
    """
    targets = _parameter_targets(problem)
    kept: list[Assignment] = []
    derived: list[dict[str, Any]] = []
    for assignment in plan.preinit:
        form = (
            None
            if assignment.formula is None
            else linear_form(
                str(assignment.formula),
                targets,
                _known_values(str(assignment.formula), targets, plan, model),
            )
        )
        if form is None:
            kept.append(assignment)
            continue
        constant, terms = form
        derived.append(
            {
                "target": assignment.target,
                "constant": constant,
                "terms": [{"parameter": at, "factor": factor} for at, factor in terms],
            }
        )
    if not derived:
        return plan, []
    return replace(plan, preinit=tuple(kept)), derived


def _gaps_of_the_values(problem: OptimizationProblem) -> list[Gap]:
    """Get the gaps of the values of the conditions which follow a parameter.

    A value of a request of the engine is a number, and the executor folds a
    formula it can evaluate without the model into one
    (`sbmlsim.simulator.executor_core`), the formula of a constant parameter
    of the model included. A parameter of the fit **is** such a constant
    parameter, so a value like `[A] = initial_A1 + initial_A2` folds into the
    number it has at the start of the fit, which would hold it there while
    the optimizer moves the parameter: the engine would answer the residuals
    of another problem without an error.

    A value of the pre-initialization which is a linear form over the
    parameters is carried by the condition instead (`_derived_of`), so it
    follows them; every other value which reads a parameter of the fit is a
    gap, and so is one of a steady state or of a change, which a condition
    carries no derived value for.

    The value the fit itself writes is a gap in the same places: the engine
    sets the parameters of a request on the model before every simulation,
    which overrides the pre-initialization of a condition, but
    `Plan.with_values` replaces the assignment of a target *wherever* the
    plan has one, so a parameter whose target a change or the
    pre-initialization of a steady state sets reaches the request as the
    number of the point it was built at (`_held_values`). A plain number is
    enough for that, no formula is needed.

    Args:
        problem: the initialized problem.

    Returns:
        The gaps, one per value which reads a parameter of the fit and is not
        carried as a derived value, and one per parameter whose target such a
        value holds.
    """
    carries = reads_derived(_loaded_of(problem))
    targets = _parameter_targets(problem)
    model = fit_model(problem)
    x = np.asarray(problem.xmodel, dtype=float)
    gaps: list[Gap] = []
    # one entry per parameter whose target a value of a plan holds: the group
    # and the place of the first of them, and how many of them there are, so
    # a protocol of many doses of an estimated dose is one gap
    held: dict[str, tuple[int, str, int]] = {}
    for group, mappings in enumerate(problem.mapping_groups):
        if not any(problem.mapping_kinds[k] is MappingKind.TRAINING for k in mappings):
            continue
        plan = problem.evaluated_plan(group, x, [])
        for assignment, where in _held_values(plan):
            if assignment.target not in targets:
                continue
            seen = held.get(assignment.target)
            held[assignment.target] = (
                (seen[0], seen[1], seen[2] + 1) if seen else (group, where, 1)
            )
        for assignment, where, derivable in _plan_formulas(plan):
            if assignment.formula is None:
                continue
            read = sorted(
                symbol
                for symbol in compile_formula(str(assignment.formula)).symbols
                if symbol in targets
            )
            if not read:
                continue
            if (
                derivable
                and carries
                and linear_form(
                    str(assignment.formula),
                    targets,
                    _known_values(str(assignment.formula), targets, plan, model),
                )
                is not None
            ):
                continue
            gaps.append(
                Gap(
                    "parameter_value",
                    f"the simulation of the group {group} sets '{assignment.target}' "
                    f"{where} to '{assignment.formula}', which reads the parameter"
                    f"{'s' if len(read) > 1 else ''} "
                    f"{', '.join(repr(symbol) for symbol in read)} of the fit: a "
                    f"value of a request of the engine is a number, which would "
                    f"hold it where the fit started",
                )
            )
    for target, (group, where, n) in held.items():
        place = where if n == 1 else f"at {n} values, the first of them {where},"
        gaps.append(
            Gap(
                "parameter_value",
                f"the parameter '{problem.parameters[targets[target]].pid}' writes "
                f"'{target}', which the simulation of the group {group} sets "
                f"{place} as well: a value of a request of the engine is a "
                f"number, and the parameter of the condition overrides the "
                f"pre-initialization of the request alone, so the value would "
                f"hold the target where the fit started",
            )
        )
    return gaps


def _held_values(plan: Plan) -> list[tuple[Assignment, str]]:
    """Get every value of a plan which a parameter of the condition cannot override.

    The engine sets the parameters of a request on the model before every
    simulation of a condition, which overrides the pre-initialization of that
    condition: a value there follows the search whatever the plan carries. It
    does not override a value the engine applies later, neither a change nor
    the pre-initialization of a steady state, so the number of such a value
    is the one of the point the request was built at.

    Args:
        plan: the plan of the condition.

    Returns:
        The assignment and where it is, for every value of the plan the
        engine would hold where the request was built.
    """
    found: list[tuple[Assignment, str]] = []
    if plan.steady_state is not None:
        found += [
            (assignment, "before the initialization of its steady state")
            for assignment in plan.steady_state.preinit
        ]
    found += [
        (assignment, f"at the time {event.time}")
        for event in plan.events
        for assignment in event.assignments
    ]
    return found


def _gaps_at_the_folds(problem: OptimizationProblem) -> list[Gap]:
    """Get the gaps of the changes a request carries as a folded number.

    A value of a request is a number, and the executor folds a formula at a
    change into one by simulating the segment which ends at the change and
    reading the state there (`executor_core._Fold`). That number is the one
    of the point the request was built at: a formula which reads a state,
    e.g. `0.5 * Aurine` of a collection which is halved, folds into the
    amount the model had there at the start of the fit and the engine holds
    the change at it while the optimizer moves the simulation, so the engine
    would answer the residuals of another problem without an error.

    There are three gates, in this order. A formula folds into the same number
    at every point of the search when every symbol it reads is a constant of
    the simulation (`_constant_at_the_changes`), and is no gap then. Else it
    is no gap when the bindings read the forms of a change
    (`reads_change_forms`) and every formula at a change of its plan is a
    linear form (`executor_core.is_formed`, all or nothing per plan, so one
    formula which is no form folds the forms of its plan as well): the
    engine evaluates them from its own state. Else it is the gap.

    The number of the pre-initialization is not such a constant: it is the
    value before the initialization and the simulation moves it on the way to
    the change, which is why `_known_values`, the values the executor folds a
    formula *before the initialization* with, is not the test here. A formula
    which reads a parameter of the fit is the gap of `_gaps_of_the_values`,
    which is more precise about it, so this reports only the symbols which
    are neither.

    The test is conservative: a symbol a change of the plan sets to a number
    is held at that number by the engine as well, so a second dose
    `PODOSE + 25` of a plan which doses by a change would fold correctly,
    and reporting it costs a fit the engine instead of its correctness.

    A formula before the initialization, of the plan or of its steady state,
    is folded without simulating and raises in the executor for a symbol it
    cannot resolve, which `_gaps_of_the_request` reports as `unsupported`.

    Args:
        problem: the initialized problem.

    Returns:
        The gaps, one per formula the executor folds by simulating, in the
        order they were met, naming its first change and how many changes
        carry it: the gaps of a problem are reported as one message, which a
        protocol of many doses or collections would otherwise fill with one
        entry per change.
    """
    from sbmlsim.simulator.executor_core import is_formed

    targets = _parameter_targets(problem)
    model = fit_model(problem)
    forms = reads_change_forms(_loaded_of(problem))
    x = np.asarray(problem.xmodel, dtype=float)
    # one entry per formula, in the order they were met: its first change and
    # the number of the changes which carry it
    found: dict[tuple[str, str], tuple[int, float, list[str], int]] = {}
    for group, mappings in enumerate(problem.mapping_groups):
        if not any(problem.mapping_kinds[k] is MappingKind.TRAINING for k in mappings):
            continue
        plan = problem.evaluated_plan(group, x, [])
        constant = _constant_at_the_changes(plan, model)
        # the request writes the forms all or nothing per plan, and this is the
        # plan the request is built from: `_conditions_and_curves` asks
        # `is_formed` of the plan `_derived_of` returns, which is
        # `replace(plan, preinit=...)` and so carries these very events
        formed = forms and is_formed(plan, model)
        for event in plan.events:
            for assignment in event.assignments:
                if assignment.formula is None:
                    continue
                formula = str(assignment.formula)
                read = sorted(
                    symbol
                    for symbol in compile_formula(formula).symbols
                    if symbol not in constant and symbol not in targets
                )
                if not read:
                    continue
                # the engine evaluates the forms of a plan whose formulas are all
                # forms from its own state at the change, so they follow the
                # search where a folded number would not
                if formed:
                    continue
                key = (assignment.target, formula)
                if key in found:
                    first = found[key]
                    found[key] = (first[0], first[1], first[2], first[3] + 1)
                else:
                    found[key] = (group, event.time, read, 1)
    gaps: list[Gap] = []
    for (target, formula), (group, time, read, n) in found.items():
        where = (
            f"at the time {time}"
            if n == 1
            else f"at {n} changes, the first of the group {group} at the time {time},"
        )
        gaps.append(
            Gap(
                "change_value",
                f"the simulation of the group {group} sets '{target}' {where} to "
                f"'{formula}', which reads "
                f"{', '.join(repr(symbol) for symbol in read)} of its simulation: "
                f"the executor folds such a formula into the number it has where "
                f"the request was built, which would hold the change there",
            )
        )
    return gaps


def _constant_at_the_changes(plan: Plan, model: CoreSBMLModel) -> set[str]:
    """Get the symbols whose value at a change of a plan is a constant.

    A symbol is one when the model can only read it, i.e. it is a parameter
    which is the variable of no rule and of no initial assignment, and
    nothing of the plan assigns it at a time, so neither the integration nor
    a change moves it: the number the executor folds a formula of such
    symbols into is the same at every point of the search. A parameter the
    pre-initialization sets by a formula is none, since a derived value of the
    condition moves it with the search (`_derived_of`), while one it sets to a
    number holds that number through the simulation.

    Args:
        plan: the plan of the condition.
        model: the model of the fit.

    Returns:
        The symbols a formula at a change may read.
    """
    assigned = {
        assignment.target for event in plan.events for assignment in event.assignments
    }
    by_formula = {
        assignment.target
        for assignment in plan.preinit
        if assignment.formula is not None
    }
    symbols = model.symbols
    return {
        parameter
        for parameter in symbols.parameters
        if parameter not in assigned
        and parameter not in symbols.initial_assignments
        and parameter not in symbols.assignment_rules
        and parameter not in symbols.rate_rules
        and parameter not in by_formula
    }


def _plan_formulas(plan: Plan) -> list[tuple[Assignment, str, bool]]:
    """Get every assignment of a plan which carries a formula.

    Args:
        plan: the plan of a condition.

    Returns:
        The assignment, where it is, and whether the condition can carry it
        as a derived value, which only the pre-initialization of the plan
        itself can be (`_derived_of`).
    """
    found: list[tuple[Assignment, str, bool]] = [
        (assignment, "before the initialization", True) for assignment in plan.preinit
    ]
    if plan.steady_state is not None:
        found += [
            (assignment, "before the initialization of its steady state", False)
            for assignment in plan.steady_state.preinit
        ]
    found += [
        (assignment, f"at the time {event.time}", False)
        for event in plan.events
        for assignment in event.assignments
    ]
    return found


def _parameter_targets(problem: OptimizationProblem) -> dict[str, int]:
    """Get the position in the parameters of the fit of every entity one writes."""
    return {parameter.target_id: at for at, parameter in enumerate(problem.parameters)}


def _known_values(
    formula: str, targets: Mapping[str, int], plan: Plan, model: CoreSBMLModel
) -> dict[str, float]:
    """Get the values of the symbols of a formula the fit does not estimate.

    These are the two the executor folds itself
    (`executor_core._preinit_value`): a symbol the same pre-initialization
    sets to a number, and a constant parameter of the model, the variable of
    no rule and of no initial assignment, whose value nothing the
    pre-initialization sets can move. A symbol of neither is left out, so the
    form is refused and the value keeps its gap.

    Args:
        formula: the formula of the assignment.
        targets: the entities the parameters of the fit write.
        plan: the plan of the condition, whose pre-initialization gives the
            numbers.
        model: the model of the fit.

    Returns:
        The value of every symbol which is neither a parameter of the fit nor
        unknown.
    """
    numbers = {
        assignment.target: assignment.value
        for assignment in plan.preinit
        if assignment.value is not None
    }
    symbols = model.symbols
    known: dict[str, float] = {}
    for symbol in compile_formula(formula).symbols:
        if symbol in targets:
            continue
        if symbol in numbers:
            known[symbol] = float(numbers[symbol])
            continue
        if (
            symbol in symbols.parameters
            and symbol not in symbols.initial_assignments
            and symbol not in symbols.assignment_rules
            and symbol not in symbols.rate_rules
        ):
            try:
                known[symbol] = float(model.loaded_value(symbol))
            except KeyError:
                continue
    return known


def _gaps_at_the_changes(
    problem: OptimizationProblem, group: int, mapping: int
) -> list[Gap]:
    """Get the gap of the data of a mapping at the time of a change.

    A point which is compared with the state after a change is a point of a
    request since sbmlsim-core 0.16.0, which reads the `after_change` of a
    curve (`after_change_of`): bindings before it ignore the field instead of
    refusing it, so such a point would quietly be compared with the state
    before its change, and the fit of a problem which has one stays in python
    there.
    """
    after = after_change_of(problem, group, mapping)
    if after is None:
        return []
    if reads_after_change(_loaded_of(problem)):
        return []
    import sbmlsim_core

    times = np.asarray(problem.x_references[mapping], dtype=float)
    which = times[after]
    return [
        Gap(
            "data_at_change",
            f"the mapping '{problem.mapping_keys[mapping]}' measures at the time "
            f"of a change, {list(which[:3])}, and compares with the state after "
            f"it, which the bindings of sbmlsim-core "
            f"{sbmlsim_core.__version__} do not read, and "
            f"{AFTER_CHANGE_VERSION} is the first which does",
        )
    ]


def reports_runs(loaded: Any) -> bool:
    """Whether the bindings report the runs of a fit while the fit runs.

    The `on_run` of `LoadedModel.fit` is new in sbmlsim-core 0.13.0 and the
    fit itself in 0.12.0, so this is what bindings of a release between the
    two answer with `False`, whose fits report their runs when they
    answered, see `run`.

    Args:
        loaded: the loaded model of the bindings.

    Returns:
        Whether a fit of the bindings takes an `on_run`.
    """
    fit = getattr(loaded, "fit", None)
    if fit is None:
        return False
    # the bindings are rust and carry the signature of a fit as its text
    text = getattr(fit, "__text_signature__", None)
    if text:
        return "on_run" in text
    try:
        return "on_run" in inspect.signature(fit).parameters
    except TypeError, ValueError:
        return False


def run(
    problem: OptimizationProblem,
    settings: FitSettings | None = None,
    *,
    size: int = 1,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    seed: int | None = None,
    sampling: SamplingType = SamplingType.LOGUNIFORM_LHS,
    indices: Sequence[int] | None = None,
    threads: int | None = None,
    timeout: float | None = None,
    max_iterations: int = 100,
    gradient_tolerance: float | None = None,
    step_tolerance: float | None = None,
    cost_tolerance: float | None = None,
    evolution: Mapping[str, Any] | None = None,
    jacobian: str | None = None,
    identifiability: Mapping[str, Any] | None = None,
    provenance: FitProvenance | None = None,
    on_run: Callable[[int, OptimizeResult], None] | None = None,
) -> OptimizationResult:
    """Run the fit of a problem in the engine and get the result of the fit.

    The engine runs the starts of a request next to each other on the threads
    of one batch, every one of them an optimizer of its own, and answers with
    every run and with the best of them. The starts are the ones of the
    problem (`OptimizationProblem.start_values`), so a fit of a seed starts
    where the fit of that seed in python starts.

    The whole fit is one request, which the engine fills its pool with: the
    runs of a fit are not equally long and a request per chunk of them would
    end every chunk with its longest run, which costs more than the fit of
    all of them (1.07x to 1.34x of the fits of the HCTZ example, 16 starts in
    chunks of four on four threads). The engine reports a run of the request
    as soon as it finished instead, which `on_run` passes on and which a fit
    that is interrupted keeps: its result is the runs which finished and
    carries `interrupted`.

    Args:
        problem: the initialized problem, without gaps (`gaps_of`).
        settings: the settings of the fit, which the result is stored with;
            the ones the problem is initialized with by default. Settings
            which differ from them initialize the problem again.
        size: the number of runs of the fit, one per start.
        algorithm: the optimizer of every run.
        seed: the seed of the sampling of the starts.
        sampling: the sampling of the starts.
        indices: the runs of the fit to run, a piece of it; all of them
            without, see `sbmlsim.fit.optimization.run_indices`.
        threads: the threads of the batch of an evaluation; the CPUs of the
            machine without.
        timeout: the seconds a single run may take, after which its optimizer
            stops at its best point; no limit without.
        max_iterations: the largest number of iterations of a run.
        gradient_tolerance: the tolerance of the scaled gradient a step
            converges at; the one of the optimizer without.
        step_tolerance: the tolerance of the step a run converges at, see
            `fit_request`.
        cost_tolerance: the tolerance of the change of the cost a run
            converges at, see `fit_request`.
        evolution: the population of a differential evolution and how it is
            drawn, see `fit_request`.
        jacobian: where the jacobian of a step comes from, see `fit_request`.
        identifiability: the identifiability of the optimum, see
            `fit_request`; none of it without.
        provenance: what the fit ran with, stored with the result.
        on_run: called with the index of a run of the fit and its result as
            soon as the engine reported that run, e.g. to store the runs
            while the fit runs. Bindings which do not report the runs of a
            fit call it with every run when the fit answered.

    Returns:
        The result of the fit, a run per start of it in the order of the
        starts, which `OptimizationResult.df_fits` sorts by their cost.

    Raises:
        ValueError: if the problem is not initialized, if it has no model or
            more than one, or if a differential evolution of more than one run
            is asked for: the population of a request is drawn from one seed,
            so the runs of such a fit would be the same run.
        FitError: if the engine refuses the request or a run of it fails, see
            the bindings of sbmlsim-core.
    """
    from sbmlsim.fit.result import OptimizationResult

    if settings is not None:
        problem.initialize(settings)
    settings = problem.settings_initialized
    if algorithm is OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION and size > 1:
        raise ValueError(
            f"'{problem.opid}': a differential evolution of the engine draws its "
            f"population from one seed, so the {size} runs of the fit would be "
            f"the same run; a fit of several runs is a least squares one."
        )
    drawn = problem.start_values(size, algorithm, sampling, seed)
    runs = list(range(size)) if indices is None else list(indices)
    starts = [drawn[k] for k in runs]
    loaded = loaded_model(problem)
    request = fit_request(
        problem,
        starts=[start for start in starts if start is not None],
        algorithm=algorithm,
        max_iterations=max_iterations,
        max_seconds=timeout,
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        cost_tolerance=cost_tolerance,
        evolution=evolution,
        identifiability=identifiability,
        jacobian=jacobian,
        threads=threads,
    )
    reported: dict[int, OptimizeResult] = {}

    def report(index: int, run: Mapping[str, Any]) -> None:
        """Keep a run of the fit which the engine reported and pass it on."""
        fit = _fit_of(problem, run, algorithm, request)
        reported[index] = fit
        if on_run is not None:
            on_run(runs[index], fit)

    streams = reports_runs(loaded)
    interrupted = False
    try:
        answer = loaded.fit(
            request, threads=threads, **({"on_run": report} if streams else {})
        )
    except KeyboardInterrupt:
        # the runs which were reported are the fit, as the repeats which
        # finished are the fit of an optimizer in python
        if not reported:
            raise
        logger.warning(
            "'%s': the fit was interrupted, it keeps the %d of its %d runs "
            "which finished.",
            problem.opid,
            len(reported),
            len(starts),
        )
        interrupted = True
        fits = [reported[index] for index in sorted(reported)]
    else:
        fits = _fits_of(problem, answer, algorithm, request)
        if not streams and on_run is not None:
            # bindings which do not report a run while the fit runs report
            # every one of them when it answered
            for index, fit in enumerate(fits):
                on_run(runs[index], fit)
    result = OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        # the engine answers with the cost of every run and not with the cost
        # of its steps, so a result of the engine has no trace of a run
        trajectories=[[] for _ in fits],
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
        units=problem.parameter_units,
        provenance=provenance,
        prior_correlation=problem.prior_correlation,
    )
    result.interrupted = interrupted
    return result


def _fits_of(
    problem: OptimizationProblem,
    answer: Mapping[str, Any],
    algorithm: OptimizationAlgorithmType,
    request: Mapping[str, Any],
) -> list[OptimizeResult]:
    """Get the runs of an answer of the engine as the runs of a fit.

    A run of the engine is a run of scipy with the keys a report reads
    (`RESULT_KEYS`): the values of its optimum and of its start in the units
    of the parameters, its cost, whether its optimizer converged (`CONVERGED`)
    and what it took. The engine reports no gradient of its optimum, so a run
    has no `optimality`.
    """
    return [
        _fit_of(problem, run, algorithm, request, answer["jacobian"])
        for run in answer["runs"]
    ]


def _fit_of(
    problem: OptimizationProblem,
    run: Mapping[str, Any],
    algorithm: OptimizationAlgorithmType,
    request: Mapping[str, Any],
    jacobian: str | None = None,
) -> OptimizeResult:
    """Get one run of the engine as the run of a fit, see `_fits_of`.

    A run says where its jacobian came from; `jacobian` is what a run of
    bindings which do not answers with, the one of the whole fit.

    The cost of a run of a likelihood is shifted by `likelihood_constant`,
    the residuals of the scales of its noise and their offsets, so it is the
    negative log-likelihood a fit in python reports.
    """
    from scipy.optimize import OptimizeResult

    factors = np.asarray(_factors_of(problem), dtype=float)
    status = str(run["status"])
    constant = likelihood_constant(problem)
    return OptimizeResult(
        x=np.asarray(run["values"], dtype=float) / factors,
        x0=np.asarray(run["start"], dtype=float) / factors,
        cost=float(run["cost"]) + constant,
        success=status in CONVERGED,
        message=status,
        duration=float(run["seconds"]),
        nfev=int(run["evaluations"]),
        algorithm=algorithm.name,
        optimizer_kwargs={
            **dict(request.get("optimizer", {})),
            "jacobian": run.get("jacobian", jacobian),
            "engine": "core",
        },
    )


def gaps_of_a_fit(
    problem: OptimizationProblem,
    algorithm: OptimizationAlgorithmType,
    size: int,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[Gap], dict[str, Any]]:
    """Get the gaps of a whole fit of the engine and the settings of its optimizer.

    The objective of the problem has to be a request (`gaps_of`), and so have
    the optimizer of the fit and its arguments: the arguments of a fit are the
    ones of `scipy.optimize.least_squares` or of
    `scipy.optimize.differential_evolution`, and the settings of the engine
    are its iterations, the tolerances of its gradient, of its step and of its
    cost (`gtol`, `xtol` and `ftol`) and the population of a differential
    evolution, so an argument which is none of them is a gap. The tolerances
    the arguments do not give are `default_tolerance` of the relative
    tolerance of the integrator, as a fit in python gets them (#607).

    Args:
        problem: the initialized problem.
        algorithm: the optimizer of the fit.
        size: the number of runs of the fit.
        kwargs: the arguments of the optimizer of the fit.

    Returns:
        The gaps of the fit, empty for a fit the engine runs, and the settings
        of `run` which its arguments give.
    """
    gaps = gaps_of(problem)
    settings: dict[str, Any] = {}
    evolution: dict[str, Any] = {}
    evolving = algorithm is OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
    if evolving and size > 1:
        gaps.append(
            Gap(
                "runs",
                f"a differential evolution of the engine draws its population "
                f"from one seed, so the {size} runs of the fit would be the "
                f"same run",
            )
        )
    for key, value in (kwargs or {}).items():
        if key == "gtol":
            settings["gradient_tolerance"] = float(value)
        elif key == "xtol":
            settings["step_tolerance"] = float(value)
        elif key == "ftol":
            settings["cost_tolerance"] = float(value)
        elif key == "method" and value == "trf":
            continue
        elif key == "maxiter" and evolving:
            settings["max_iterations"] = int(value)
        elif key == "seed" and evolving:
            evolution["seed"] = int(value)
        elif key == "popsize" and evolving:
            # the population of scipy is a multiple of the parameters, the one
            # of the engine is the members themselves
            evolution["population"] = int(value) * len(problem.parameters)
        elif key == "mutation" and evolving and isinstance(value, int | float):
            evolution["f"] = float(value)
        elif key == "recombination" and evolving:
            evolution["cr"] = float(value)
        elif key == "polish" and evolving:
            evolution["polish"] = bool(value)
        else:
            gaps.append(
                Gap(
                    "optimizer",
                    f"the argument '{key}' of the optimizer of the fit is no "
                    f"setting of the optimizer of the engine",
                )
            )
    if evolution:
        settings["evolution"] = evolution
    # the tolerances a fit in python gets from `default_tolerance`, which the
    # optimizer of the engine has no default of: without them it tests neither
    # its step nor the change of its cost and spends its last iterations on the
    # noise of the integrator, 2.3 times the evaluations of a fit of the HCTZ
    # example to the same optima (#607)
    from sbmlsim.fit.optimization import default_tolerance

    tolerance = default_tolerance(problem.settings_initialized.relative_tolerance)
    for name in ("gradient_tolerance", "step_tolerance", "cost_tolerance"):
        settings.setdefault(name, tolerance)
    return gaps, settings


def walk_of(settings: ProfileSettings) -> dict[str, Any]:
    """Get the walk of the engine of the settings of a profile likelihood.

    The walk of the engine is the walk of `sbmlsim.fit.identifiability`: it
    steps away from the optimum until the cost reaches the threshold of the
    parameter, reducing a step which raises the cost by more than
    `max_cost_fraction` of the distance to the threshold and enlarging one
    which raises it by little, so the fields are the ones of the settings. The
    `delta` of the engine is the rise of the cost which bounds a parameter,
    i.e. half the quantile of the chi square distribution, which is what
    `ProfileSettings.threshold` adds to the cost of the optimum.

    Args:
        settings: the settings of the analysis.

    Returns:
        The `walk` of the identifiability of a request.
    """
    return {
        "points": settings.max_points,
        "step": settings.initial_step,
        "min_step": settings.min_step,
        "max_step": settings.max_step,
        "factor": settings.step_factor,
        "rise": settings.max_cost_fraction,
        "delta": settings.delta / 2.0,
    }


def gaps_of_the_profiles(
    problem: OptimizationProblem, settings: ProfileSettings
) -> list[Gap]:
    """Get the gaps of a profile likelihood of the engine.

    The objective of the problem has to be a request (`gaps_of`), and the
    walk of the analysis has to be the walk of the engine (`walk_of`): a scan
    which holds the other parameters at the optimum instead of fitting them
    again is none of it, and neither is an argument of the optimizer of a
    point which is no setting of the engine. A fit of one parameter is none of
    it either: every point of its profile holds that parameter and fits
    nothing, which the optimizer of the engine refuses and python evaluates as
    the cost at the point.

    Args:
        problem: the initialized problem.
        settings: the settings of the analysis.

    Returns:
        The gaps of the analysis, empty for one the engine walks.
    """
    gaps = gaps_of(problem)
    if len(problem.parameters) < 2:
        gaps.append(
            Gap(
                "parameters",
                "the fit has one parameter, so a point of its profile holds "
                "that parameter and has none left to fit again",
            )
        )
    if not settings.reoptimize:
        gaps.append(
            Gap(
                "reoptimize",
                "the analysis scans the cost with the other parameters at the "
                "optimum, where a profile of the engine fits them again at "
                "every point",
            )
        )
    for key in settings.optimizer_kwargs:
        if key not in {"gtol"}:
            gaps.append(
                Gap(
                    "optimizer",
                    f"the argument '{key}' of the optimizer of a point is no "
                    f"setting of the optimizer of the engine",
                )
            )
    return gaps


def profile_points(
    problem: OptimizationProblem,
    settings: ProfileSettings,
    at: Sequence[float] | np.ndarray,
    pids: Sequence[str],
    *,
    threads: int | None = None,
    jacobian: str | None = None,
) -> dict[str, tuple[list[tuple[np.ndarray, float, bool]], ...]]:
    """Walk the profiles of the parameters of a problem in the engine.

    The engine walks the two sides of every profile next to each other on the
    threads of one batch, so the whole analysis is one call and not a process
    per scan. The optimum is the one of the caller: the fit of the request
    takes no iteration of its own, it is the point the profiles are walked
    around.

    Args:
        problem: the initialized problem, of sbmlsim-core.
        settings: the settings of the analysis.
        at: the optimum, in the space the parameters are searched.
        pids: the parameters to profile, in the order of the profiles.
        threads: the threads of the batch of the analysis, all cores of the
            machine without.
        jacobian: where the jacobian of a step of a point comes from, see
            `fit_request`.

    Returns:
        The points of the two sides of every profile by parameter id, from the
        optimum outward, as the scans of `sbmlsim.fit.identifiability` carry
        them: the parameters of a point in the space the search runs in, its
        cost and whether the optimizer of the point converged.
    """
    at = np.asarray(at, dtype=float)
    points: dict[str, Any] = {"max_iterations": MAX_ITERATIONS_OF_A_POINT}
    if "gtol" in settings.optimizer_kwargs:
        points["gradient_tolerance"] = float(settings.optimizer_kwargs["gtol"])
    request = fit_request(
        problem,
        # the starts of a request are the values of the parameters, the walk
        # and the optimum of the caller are in the space they are searched
        starts=[np.asarray(problem.from_scale(at), dtype=float)],
        # the optimum is the one of the caller, the engine does not look for
        # another one: the fit of no iteration answers with that point and the
        # points of the profiles fit the other parameters themselves
        max_iterations=0,
        jacobian=jacobian,
        identifiability={
            "parameters": list(pids),
            "walk": walk_of(settings),
            "profiles": True,
            "fisher": False,
            "optimizer": points,
        },
    )
    answer = loaded_model(problem).fit(request, threads=threads)
    walked: dict[str, tuple[list[tuple[np.ndarray, float, bool]], ...]] = {}
    for profile in answer["profiles"]:
        index = problem.pids.index(profile["id"])
        below: list[tuple[np.ndarray, float, bool]] = []
        above: list[tuple[np.ndarray, float, bool]] = []
        for point in profile["points"]:
            side = above if point["x"] > at[index] else below
            side.append(
                (
                    _theta_of(problem, index, point),
                    float(point["cost"]),
                    point["status"] in CONVERGED,
                )
            )
        # the points of the engine ascend, the scan of a side runs outward
        below.reverse()
        walked[profile["id"]] = (below, above)
    return walked


def _theta_of(
    problem: OptimizationProblem, index: int, point: Mapping[str, Any]
) -> np.ndarray:
    """Get the parameters of a point of a profile in the space of the search.

    A point of the engine carries the value the parameter is held at and the
    values the other parameters reached, in their order in the fit, which is
    the order of the problem without the one which is held.
    """
    values = list(point["values"])
    values.insert(index, float(point["value"]))
    return np.asarray(problem.to_scale(np.asarray(values, dtype=float)), dtype=float)


def gaps_of_a_jacobian(problem: OptimizationProblem) -> list[Gap]:
    """Get what keeps the jacobian of the residuals of a problem out of the engine.

    The gaps of the fit of the problem (`gaps_of`), and the bindings which do
    not resolve the problem of a request (`LoadedModel.fit_problem`, new in
    sbmlsim-core 0.12.0). Whether the conditions of the problem carry the
    forward sensitivities is no gap of the request: the engine answers that
    when it resolves the problem, see `residual_jacobian`.

    Args:
        problem: the initialized problem.

    Returns:
        The gaps, empty for a problem whose jacobian the engine computes.
    """
    gaps = gaps_of(problem)
    if gaps:
        return gaps
    if problem.settings_initialized.objective is not ObjectiveType.LEAST_SQUARES:
        return [
            Gap(
                "objective",
                f"the fit minimizes the objective "
                f"'{problem.settings_initialized.objective.value}', whose "
                f"residuals are the ones of the data and of the scales of its "
                f"noise, while a jacobian of the engine has the rows of the "
                f"curves of its request, the ones of the data",
            )
        ]
    loaded = _loaded_of(problem)
    if not hasattr(loaded, "fit_problem"):
        import sbmlsim_core

        return [
            Gap(
                "bindings",
                f"the bindings of sbmlsim-core {sbmlsim_core.__version__} do not "
                f"resolve the problem of a request of '{CONTRACT}', which 0.12.0 "
                f"is the first to do",
            )
        ]
    return gaps


def residual_jacobian(
    problem: OptimizationProblem,
    x: Sequence[float] | np.ndarray,
    *,
    threads: int | None = None,
) -> np.ndarray:
    """Get the jacobian of the residuals of a problem from the engine, exactly.

    The forward sensitivities of the conditions are the derivatives of the
    residuals themselves, one simulation per condition, where a difference
    quotient takes two simulations per parameter and condition and is exact
    only to the noise of the integrator over its step. The whole jacobian is
    one batch of the engine, which walks the conditions on its threads.

    The jacobian is the one of the data of the fit, a row per point of the
    curves of the request, which are the training mappings of the problem; the
    residuals of the priors of a fit of the posterior are no part of a request
    and no part of this jacobian. Its columns are the derivatives in the space
    the problem searches: the engine searches the entity of the model, so a
    parameter of its own unit has the factor of its target in its column, and
    the factor of a parameter which is searched as a logarithm cancels.

    Args:
        problem: the initialized problem, without gaps (`gaps_of_a_jacobian`).
        x: the point, in the space the problem searches.
        threads: the threads of the batch, every CPU of the process without.

    Returns:
        The jacobian, a row per data point and a column per parameter.

    Raises:
        ValueError: if the problem is not initialized, if it has no model or
            more than one, if `x` is not one value per parameter, or if a
            condition of the problem carries no forward sensitivities, which
            is what the engine refuses the problem with.
    """
    request = fit_request(
        problem,
        starts=[np.asarray(problem.from_scale(np.asarray(x, dtype=float)))],
        # the jacobian is no fit: the optimizer of the request takes no step
        max_iterations=0,
        jacobian=SENSITIVITIES,
        threads=threads,
    )
    core = loaded_model(problem).fit_problem(request, threads=threads)
    jacobian = np.asarray(core.jacobian(_point_of(problem, x)), dtype=float)
    return jacobian * _column_factors(problem)


def _point_of(
    problem: OptimizationProblem, x: Sequence[float] | np.ndarray
) -> list[float]:
    """Get the point of the search space of the engine of a point of a problem.

    The engine searches the entity of the model, which the factor of its
    target carries a parameter of another unit into (`_factors_of`), on the
    scale of the parameter.
    """
    values = np.asarray(problem.from_scale(np.asarray(x, dtype=float)), dtype=float)
    return [
        float(scale.to_scale(value * factor))
        for scale, value, factor in zip(
            problem.scales_initialized, values, _factors_of(problem), strict=True
        )
    ]


def _column_factors(problem: OptimizationProblem) -> np.ndarray:
    """Get what carries the columns of a jacobian of the engine into the space of the problem.

    A parameter of a linear scale is searched as the value of its entity in
    the model, which is the value of the parameter times the factor of its
    target, so the derivative of the engine is the one of the problem divided
    by that factor. The factor of a parameter which is searched as a logarithm
    is an offset of the search space and cancels in its derivative.
    """
    return np.array(
        [
            1.0 if scale.is_log else factor
            for scale, factor in zip(
                problem.scales_initialized, _factors_of(problem), strict=True
            )
        ],
        dtype=float,
    )
