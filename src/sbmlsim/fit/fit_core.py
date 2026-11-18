"""The fit of an optimization problem in the engine core, the contract `fit/1` (#556).

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
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sbmlsim.engine import Engine
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.options import (
    ObjectiveType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
)
from sbmlsim.fit.sampling import SamplingType

if TYPE_CHECKING:  # pragma: no cover
    from scipy.optimize import OptimizeResult

    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.options import FitSettings
    from sbmlsim.fit.provenance import FitProvenance
    from sbmlsim.fit.result import OptimizationResult
    from sbmlsim.model.model_core import CoreSBMLModel

logger = logging.getLogger(__name__)

#: the contract of a fit of the engine
CONTRACT = "fit/1"

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
            which is not loaded in the engine core, see `gaps_of`.
    """
    models = _models_of(problem)
    if len(models) != 1:
        raise ValueError(
            f"'{problem.opid}': the engine fits one model, but the fit mappings "
            f"of the problem have {len(models)} of them."
        )
    model = models[0]
    loaded = getattr(model, "loaded", None)
    if loaded is None:
        raise ValueError(
            f"'{problem.opid}': the model '{model}' is not loaded in the engine "
            f"core, so the engine cannot fit it; the problem simulates with "
            f"'{problem.engine.value}'."
        )
    return loaded


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
                f"fit of a request runs in the engine core",
            )
        )
        return gaps
    settings = problem.settings_initialized
    if settings.objective is not ObjectiveType.LEAST_SQUARES:
        return [
            Gap(
                "objective",
                f"the fit minimizes the objective '{settings.objective.value}', "
                f"whose noise parameters and priors are no part of a request",
            )
        ]
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
                f"the residuals of the fit are '{settings.residual.name.lower()}', "
                f"which subtract a baseline of the simulation",
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
    for group, mappings in enumerate(problem.mapping_groups):
        training = [
            k for k in mappings if problem.mapping_kinds[k] is MappingKind.TRAINING
        ]
        if not training:
            continue
        gaps.extend(_gaps_of_a_group(problem, group, training))
    return gaps


def fit_request(
    problem: OptimizationProblem,
    *,
    starts: Sequence[Sequence[float] | np.ndarray] = (),
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    max_iterations: int = 100,
    max_seconds: float | None = None,
    gradient_tolerance: float | None = None,
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
    model = _models_of(problem)[0] if len(_models_of(problem)) == 1 else None
    if model is None:
        raise ValueError(
            f"'{problem.opid}': the engine fits one model, but the fit mappings "
            f"of the problem have {len(_models_of(problem))} of them."
        )
    conditions, curves = _conditions_and_curves(problem, model)
    optimizer: dict[str, Any] = {
        "algorithm": _algorithm(algorithm, evolution),
        "max_iterations": int(max_iterations),
    }
    if max_seconds is not None:
        optimizer["max_seconds"] = float(max_seconds)
    if gradient_tolerance is not None:
        optimizer["gradient_tolerance"] = float(gradient_tolerance)
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
    """
    factors = _factors_of(problem)
    scales = problem.scales_initialized
    parameters = []
    for k, parameter in enumerate(problem.parameters):
        factor = factors[k]
        parameters.append(
            {
                "id": parameter.target_id,
                "start": float(problem.xmodel[k]) * factor,
                "lower": float(parameter.lower_bound) * factor,
                "upper": float(parameter.upper_bound) * factor,
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

    settings = model.request_settings()
    x = np.asarray(problem.xmodel, dtype=float)
    conditions: list[dict[str, Any]] = []
    of_the_group: dict[int, str] = {}
    for group, mappings in enumerate(problem.mapping_groups):
        training = [
            k for k in mappings if problem.mapping_kinds[k] is MappingKind.TRAINING
        ]
        if not training:
            continue
        plan = problem.evaluated_plan(group, x, [])
        selections = problem._selections_of(group, training)
        condition = _condition_id(problem, group, training[0])
        of_the_group[group] = condition
        conditions.append(
            {
                "id": condition,
                "request": request_of(plan, model, selections, settings),
            }
        )
    groups = {k: group for group, ks in enumerate(problem.mapping_groups) for k in ks}
    curves = [
        _curve(problem, k, of_the_group[groups[k]])
        for k in problem.indices(MappingKind.TRAINING)
    ]
    return conditions, curves


def _condition_id(problem: OptimizationProblem, group: int, mapping: int) -> str:
    """Get the id of the condition of a group, which its curves read it by."""
    return (
        f"{problem.experiment_keys[mapping]}|{problem.simulation_keys[mapping]}|{group}"
    )


def _curve(
    problem: OptimizationProblem, mapping: int, condition: str
) -> dict[str, Any]:
    """Get the curve of a training fit mapping, its data and its weights.

    The engine divides the residual of a point by its standard deviation, so
    the weight `w` of the problem is `1/sqrt(w)` here, and the mean of the
    data of a normalized residual goes into it as well.
    """
    weights = np.asarray(problem.weights[mapping], dtype=float)
    scale = 1.0
    if problem.residual is ResidualType.NORMALIZED:
        scale = float(np.mean(problem.y_references[mapping]))
    noise = [float(scale / math.sqrt(weight)) for weight in weights]
    return {
        "id": problem.mapping_keys[mapping],
        "condition": condition,
        "column": problem.yid_observable[mapping],
        "times": [float(time) for time in problem.x_references[mapping]],
        "values": [float(value) for value in problem.y_references[mapping]],
        "noise": noise,
    }


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
    if problem._group_graphs[group] is not None or (
        problem._group_pointwise[group] is not None
    ):
        gaps.append(
            Gap(
                "observable",
                f"the data of the group {group} is read from observables of its "
                f"experiment, which are evaluated on the output of its "
                f"simulation and are no column of it",
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
    if problem.observable_models[mapping] is not None:
        gaps.append(
            Gap(
                "observable",
                f"the mapping '{key}' observes a model of its own, which is no "
                f"column of a simulation",
            )
        )
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
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        gaps.append(
            Gap(
                "weights",
                f"a point of the mapping '{key}' has the weight "
                f"'{np.min(weights)}', which is no standard deviation of a curve",
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


def _gaps_at_the_changes(
    problem: OptimizationProblem, group: int, mapping: int
) -> list[Gap]:
    """Get the gap of the data of a mapping at the time of a change.

    The output of a simulation has the state before a change and the one
    after it at the time of the change, and the engine reads the row of the
    state before it: a data point which is compared with the state after a
    change, which `DataAtChangeType.AFTER` is for every point and the default
    for the first point of a curve, is no point of a request.
    """
    times = np.asarray(problem.x_references[mapping], dtype=float)
    plan = problem.plans[group]
    if plan is None or not plan.change_times or times.size == 0:
        return []
    at_change = plan.at_change(times - plan.time_shift)
    before = problem._before_change[mapping]
    after = at_change >= 0 if before is None else (at_change >= 0) & (before < 0)
    if not bool(np.any(after)):
        return []
    which = times[after]
    return [
        Gap(
            "data_at_change",
            f"the mapping '{problem.mapping_keys[mapping]}' measures at the time "
            f"of a change, {list(which[:3])}, and compares with the state after "
            f"it, where the engine reads the state before it",
        )
    ]


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
    evolution: Mapping[str, Any] | None = None,
    jacobian: str | None = None,
    identifiability: Mapping[str, Any] | None = None,
    provenance: FitProvenance | None = None,
) -> OptimizationResult:
    """Run the fit of a problem in the engine and get the result of the fit.

    The whole fit is one request: the engine runs the `size` starts of the fit
    next to each other on the threads of one batch, every one of them an
    optimizer of its own, and answers with every run and with the best of
    them. The starts are the ones of the problem
    (`OptimizationProblem.start_values`), so a fit of a seed starts where the
    fit of that seed in python starts.

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
        evolution: the population of a differential evolution and how it is
            drawn, see `fit_request`.
        jacobian: where the jacobian of a step comes from, see `fit_request`.
        identifiability: the identifiability of the optimum, see
            `fit_request`; none of it without.
        provenance: what the fit ran with, stored with the result.

    Returns:
        The result of the fit, the runs of its starts sorted by their cost.

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
    starts = problem.start_values(size, algorithm, sampling, seed)
    if indices is not None:
        starts = [starts[k] for k in indices]
    request = fit_request(
        problem,
        starts=[start for start in starts if start is not None],
        algorithm=algorithm,
        max_iterations=max_iterations,
        max_seconds=timeout,
        gradient_tolerance=gradient_tolerance,
        evolution=evolution,
        identifiability=identifiability,
        jacobian=jacobian,
        threads=threads,
    )
    answer = loaded_model(problem).fit(request, threads=threads)
    return OptimizationResult(
        parameters=problem.parameters,
        fits=_fits_of(problem, answer, algorithm, request),
        # the engine answers with the cost of every run and not with the cost
        # of its steps, so a result of the engine has no trace of a run
        trajectories=[[] for _ in answer["runs"]],
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
        units=problem.parameter_units,
        provenance=provenance,
        prior_correlation=problem.prior_correlation,
    )


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
    from scipy.optimize import OptimizeResult

    factors = np.asarray(_factors_of(problem), dtype=float)
    optimizer = dict(request.get("optimizer", {}))
    fits: list[OptimizeResult] = []
    for run in answer["runs"]:
        status = str(run["status"])
        fits.append(
            OptimizeResult(
                x=np.asarray(run["values"], dtype=float) / factors,
                x0=np.asarray(run["start"], dtype=float) / factors,
                cost=float(run["cost"]),
                success=status in CONVERGED,
                message=status,
                duration=float(run["seconds"]),
                nfev=int(run["evaluations"]),
                algorithm=algorithm.name,
                optimizer_kwargs={
                    **optimizer,
                    "jacobian": answer["jacobian"],
                    "engine": "core",
                },
            )
        )
    return fits


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
    are its iterations, the tolerance of its gradient and the population of a
    differential evolution, so an argument which is none of them is a gap.

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
    return gaps, settings
