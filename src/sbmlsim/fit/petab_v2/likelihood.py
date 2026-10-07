"""The log-likelihood of an optimization problem.

A fit of `sbmlsim` is a weighted least squares fit, PEtab defines the
objective of a problem as the likelihood of its measurements under a noise
model. This module calculates that log-likelihood for the evaluation of a
problem, the optimizer does not use it.

The noise model of a fit mapping is its `sbmlsim.fit.objects.NoiseModel`, i.e.
the noise formula and the distribution of the observable it was read from. The
simulation `y` is the median of the distribution of the measurement `m` and
the noise formula gives its scale `s`, which are the definitions of PEtab v2:

    normal        log p = -0.5 log(2 pi s^2)       - 0.5 ((m - y) / s)^2
    log-normal    log p = -0.5 log(2 pi s^2 m^2)   - 0.5 ((log m - log y) / s)^2
    laplace       log p = -log(2 s)                - |m - y| / s
    log-laplace   log p = -log(2 s m)              - |log m - log y| / s

`chi2` is the sum of the squares of the `normalized_residuals`, and
`log_prior` the log density of the prior of every parameter, which
`unnorm_log_posterior` adds to the log-likelihood.

The functions of arrays (`log_density`, `normalized_residuals`,
`noise_values`) know nothing of a problem, `log_likelihood`, `chi2` and
`gradient` simulate one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sbmlsim.fit.objects import NoiseDistribution, NoiseModel
from sbmlsim.fit.options import ResidualType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.log import some_ids

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)

#: the placeholder of the noise of a measurement, i.e. the standard deviation
#: of its data. PEtab v2 declares the placeholders of an observable in
#: `noisePlaceholders`; the `noiseParameter${n}_${observableId}` names of v1
#: are gone
NOISE_PLACEHOLDER = "sd"

#: scale of the noise of a fit mapping which has neither a noise model nor
#: errors on its data, in the unit of the observable
DEFAULT_SIGMA = 1.0

#: relative step of the finite differences of the gradient, the rule of
#: `sbmlsim.fit.fisher`
DEFAULT_STEP = 1e-6


# --- THE FUNCTIONS OF ARRAYS ---


def log_density(
    measurement: ArrayLike,
    simulation: ArrayLike,
    sigma: ArrayLike,
    distribution: NoiseDistribution = NoiseDistribution.NORMAL,
) -> np.ndarray:
    """Get the log density of every measurement under its noise model.

    Args:
        measurement: the measured values.
        simulation: the simulated values at the measurements, the median of
            the distribution.
        sigma: scale of the noise, one value or one value per measurement.
        distribution: distribution of the noise.

    Returns:
        The log density of every measurement, see the module for the
        definitions.

    Raises:
        ValueError: if the arrays do not have one shape, if a scale is not a
            positive finite number, if a measurement or a simulation is not
            finite, or if a measurement or a simulation of a logarithmic
            distribution is not positive.
    """
    m, y, s, distribution = _checked(measurement, simulation, sigma, distribution)
    residual = _residual(m, y, s, distribution)
    # the density of the measurement and not of its logarithm, i.e. the
    # jacobian `1 / m` of the transformation is part of it
    scale = s * m if distribution.is_log else s
    if distribution in {NoiseDistribution.NORMAL, NoiseDistribution.LOG_NORMAL}:
        return -0.5 * np.log(2.0 * np.pi * np.square(scale)) - 0.5 * np.square(residual)
    return -np.log(2.0 * scale) - np.abs(residual)


def normalized_residuals(
    measurement: ArrayLike,
    simulation: ArrayLike,
    sigma: ArrayLike,
    distribution: NoiseDistribution = NoiseDistribution.NORMAL,
) -> np.ndarray:
    """Get the residual of every measurement in units of the scale of its noise.

    The residual is `(m - y) / s`, and `(log m - log y) / s` for a
    logarithmic distribution, whose square summed over the measurements is
    the `chi2` of PEtab.

    Args:
        measurement: the measured values.
        simulation: the simulated values at the measurements.
        sigma: scale of the noise, one value or one value per measurement.
        distribution: distribution of the noise.

    Returns:
        The normalized residual of every measurement.

    Raises:
        ValueError: see `log_density`.
    """
    return _residual(*_checked(measurement, simulation, sigma, distribution))


def _residual(
    m: np.ndarray, y: np.ndarray, s: np.ndarray, distribution: NoiseDistribution
) -> np.ndarray:
    """Get the normalized residual of checked arrays, see `normalized_residuals`."""
    if distribution.is_log:
        return (np.log(m) - np.log(y)) / s
    return (m - y) / s


def _checked(
    measurement: ArrayLike,
    simulation: ArrayLike,
    sigma: ArrayLike,
    distribution: NoiseDistribution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NoiseDistribution]:
    """Check the arrays of a log density, see `log_density`.

    Returns:
        The measurements, the simulations, the scale of every measurement and
        the distribution.
    """
    m = np.asarray(measurement, dtype=float)
    y = np.asarray(simulation, dtype=float)
    if m.shape != y.shape:
        raise ValueError(
            f"The log density requires a simulation for every measurement, but "
            f"the measurements have the shape '{m.shape}' and the simulations "
            f"'{y.shape}'."
        )
    try:
        s = np.broadcast_to(np.asarray(sigma, dtype=float), m.shape)
    except ValueError as err:
        raise ValueError(
            f"The log density requires one scale of the noise or one per "
            f"measurement, but the measurements have the shape '{m.shape}' and "
            f"the scales '{np.shape(sigma)}'."
        ) from err
    if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError(
            f"The scale of the noise must be a positive finite number, but is "
            f"'{s[~np.isfinite(s) | (s <= 0.0)]}'."
        )
    for name, values in (("measurement", m), ("simulation", y)):
        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"The log density requires finite values, but "
                f"'{int(np.sum(~np.isfinite(values)))}' of the '{name}' values "
                f"are not: '{values[~np.isfinite(values)]}'."
            )

    distribution = NoiseDistribution(distribution)
    if distribution.is_log:
        for name, values in (("measurement", m), ("simulation", y)):
            if np.any(values <= 0.0):
                raise ValueError(
                    f"The distribution '{distribution.value}' requires positive "
                    f"values, but '{int(np.sum(values <= 0.0))}' of the "
                    f"'{name}' values are not: '{values[values <= 0.0]}'."
                )
    return m, y, s, distribution


def noise_values(
    noise: NoiseModel,
    size: int,
    values: Mapping[str, float] | None = None,
    simulation: ArrayLike | None = None,
    selections: Mapping[str, ArrayLike] | None = None,
) -> np.ndarray:
    """Get the scale of the noise of every measurement of a fit mapping.

    The symbols of the noise formula are resolved in this order: a placeholder
    is the value of the measurement, the symbol of the observable is the
    simulation, a selection of the simulation is its value at the
    measurement, which is what a condition or the fit gave it, a parameter is
    the value of `values` and, without one, the nominal value of the noise
    model.

    Args:
        noise: noise model of the fit mapping.
        size: number of measurements.
        values: values of parameters by their id, e.g. the values of a
            parameter set.
        simulation: the simulated values at the measurements, for a noise
            formula which holds the observable.
        selections: values of the selections of the simulation at the
            measurements, see `NoiseModel.symbols`.

    Returns:
        The scale of the noise, one value per measurement.

    Raises:
        ValueError: if the noise model has placeholders and not one value of
            them per measurement, if the formula holds the observable and no
            simulation is given, if a symbol has no value, or if a scale is
            not a positive finite number.
    """
    context = f"noise formula '{noise.formula}'"
    if noise.placeholders and len(noise.placeholder_values) != size:
        raise ValueError(
            f"{context}: '{len(noise.placeholder_values)}' values of the "
            f"placeholders '{list(noise.placeholders)}' for '{size}' "
            f"measurements."
        )
    try:
        formula = noise.formula_model
    except ValueError as err:
        raise ValueError(f"{context}: {err}") from err
    parameters: dict[str, float] = {p.pid: p.value for p in noise.parameters}
    parameters.update(values or {})
    selections = selections or {}

    variables: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for symbol in formula.symbols:
        if symbol == noise.observable:
            if simulation is None:
                raise ValueError(
                    f"{context}: the formula holds the observable "
                    f"'{noise.observable}', so it requires the simulation."
                )
            variables[symbol] = np.asarray(simulation, dtype=float)
        elif symbol in selections:
            variables[symbol] = np.asarray(selections[symbol], dtype=float)
        elif symbol in parameters:
            variables[symbol] = np.full(size, float(parameters[symbol]))
        else:
            missing.append(symbol)
    if missing:
        raise ValueError(
            f"{context}: the formula uses '{missing}', which are neither "
            f"placeholders nor selections of the simulation nor parameters with "
            f"a value. A noise formula is a number, a parameter or a formula of "
            f"them."
        )

    sigma = formula.evaluate(variables, size=size)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            f"{context}: the scale of the noise must be a positive finite "
            f"number, but is '{sigma[~np.isfinite(sigma) | (sigma <= 0.0)]}'."
        )
    return sigma


def default_noise_model(errors: ArrayLike | None) -> NoiseModel:
    """Get the noise model of a fit mapping which does not define one.

    The noise is normal and its standard deviation is the error of the
    reference data as the problem resolves it (the SD, the SE without one, the
    largest error of the curve for a point without one), through the
    placeholder `NOISE_PLACEHOLDER`, and `DEFAULT_SIGMA` for data without
    errors. This is what the export writes for such a mapping, so
    the log-likelihood of a problem and of its PEtab problem agree.

    Args:
        errors: the errors of the reference data of the mapping, i.e.
            `OptimizationProblem.y_errors[k]`, `None` if it has none.

    Returns:
        The noise model.
    """
    if errors is None:
        return NoiseModel(formula=repr(DEFAULT_SIGMA))
    return NoiseModel(
        formula=NOISE_PLACEHOLDER,
        placeholders=(NOISE_PLACEHOLDER,),
        placeholder_values=tuple(
            (float(error),) for error in np.asarray(errors, dtype=float)
        ),
    )


# --- THE FUNCTIONS OF A PROBLEM ---


def noise_model_of(problem: OptimizationProblem, k: int) -> NoiseModel:
    """Get the noise model of a fit mapping of an initialized problem.

    Args:
        problem: initialized optimization problem.
        k: index of the fit mapping.

    Returns:
        The noise model of the mapping, `default_noise_model` if it has none.
    """
    noise = problem.noise_models[k]
    if noise is not None:
        return noise
    return default_noise_model(problem.y_errors[k])


def nominal_parameters(problem: OptimizationProblem) -> ParameterSet:
    """Get the nominal values of the parameters of a problem.

    The nominal value of a parameter is its start value, which is the
    `nominalValue` of the parameter table for a problem which was read, and
    the value of the model for a parameter without a start value.

    Args:
        problem: initialized optimization problem.

    Returns:
        The parameter set `nominal`.
    """
    x = [
        float(problem.xmodel[k]) if p.start_value is None else float(p.start_value)
        for k, p in enumerate(problem.parameters)
    ]
    return ParameterSet.from_fit_parameters(
        parameters=problem.parameters,
        x=x,
        sid="nominal",
        provenance="nominal values of the problem",
    )


def _check_problem(problem: OptimizationProblem) -> None:
    """Check that the log-likelihood of a problem is defined.

    Raises:
        ValueError: if the problem is not initialized, or if its residuals are
            relative to the baseline of a curve.
    """
    if not problem.is_initialized:
        raise ValueError(
            f"'{problem.opid}': the log-likelihood requires the resolved "
            f"mappings, call `initialize(settings)` first."
        )
    if problem.residual in {
        ResidualType.ABSOLUTE_TO_BASELINE,
        ResidualType.NORMALIZED_TO_BASELINE,
    }:
        raise ValueError(
            f"'{problem.opid}': the residual '{problem.residual.name}' shifts "
            f"the data to the baseline of its curve, so the problem does not "
            f"hold the measurements the noise model describes. Initialize the "
            f"problem with the residual 'ABSOLUTE' or 'NORMALIZED' for its "
            f"log-likelihood."
        )


def _noise_terms(
    problem: OptimizationProblem, pset: ParameterSet
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, NoiseDistribution]]:
    """Simulate a problem and get the noise of its training data.

    Args:
        problem: initialized optimization problem.
        pset: the parameters to simulate at.

    Returns:
        Per fit mapping of the training data its name, the measurements, the
        simulations, the scale of the noise and the distribution.

    Raises:
        ValueError: if a simulation failed or if the noise model of a mapping
            cannot be evaluated.
    """
    evaluations = problem.evaluations(pset.x(problem.pids))
    terms = []
    for k in problem.training_indices:
        key = f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}"
        noise = noise_model_of(problem, k)
        measurement = np.asarray(problem.y_references[k], dtype=float)
        simulation = evaluations[k].prediction
        try:
            sigma = noise_values(
                noise,
                size=measurement.size,
                values=pset.values,
                simulation=simulation,
                selections=evaluations[k].selections,
            )
        except ValueError as err:
            raise ValueError(f"'{problem.opid}', fit mapping '{key}': {err}") from err
        terms.append((key, measurement, simulation, sigma, noise.distribution))
    return terms


def log_likelihood(
    problem: OptimizationProblem, parameters: ParameterSet | None = None
) -> float:
    """Get the log-likelihood of the training data of a problem.

    The problem is simulated at the parameters and the log density of every
    measurement of the training data is summed, see `log_density`. The noise
    model of a fit mapping is `noise_model_of`. The settings of the fit, i.e.
    the residual, the weights and the loss function, do not enter.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate the log-likelihood at, with the
            values of the parameters of the fit and, optionally, of parameters
            of the noise formulas. `nominal_parameters` by default.

    Returns:
        The log-likelihood.

    Raises:
        ValueError: if the problem is not initialized, if its residuals are
            relative to the baseline, if a simulation failed or if the noise
            model of a mapping cannot be evaluated.
        KeyError: if the parameters lack a parameter of the fit.
    """
    _check_problem(problem)
    pset = parameters if parameters is not None else nominal_parameters(problem)
    total = 0.0
    for key, measurement, simulation, sigma, distribution in _noise_terms(
        problem, pset
    ):
        try:
            density = log_density(
                measurement, simulation, sigma, distribution=distribution
            )
        except ValueError as err:
            raise ValueError(f"'{problem.opid}', fit mapping '{key}': {err}") from err
        total += float(np.sum(density))
    return total


def chi2(problem: OptimizationProblem, parameters: ParameterSet | None = None) -> float:
    """Get the chi2 of the training data of a problem.

    The sum of the squares of the `normalized_residuals` of the training
    data, which is the `chi2` of PEtab.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate chi2 at, see `log_likelihood`.

    Returns:
        The chi2.

    Raises:
        ValueError: see `log_likelihood`.
        KeyError: if the parameters lack a parameter of the fit.
    """
    _check_problem(problem)
    pset = parameters if parameters is not None else nominal_parameters(problem)
    total = 0.0
    for key, measurement, simulation, sigma, distribution in _noise_terms(
        problem, pset
    ):
        try:
            residuals = normalized_residuals(
                measurement, simulation, sigma, distribution=distribution
            )
        except ValueError as err:
            raise ValueError(f"'{problem.opid}', fit mapping '{key}': {err}") from err
        total += float(np.sum(np.square(residuals)))
    return total


def log_prior(
    problem: OptimizationProblem, parameters: ParameterSet | None = None
) -> dict[str, float]:
    """Get the log density of the prior of every parameter of a problem.

    The prior of a parameter is its `Prior`, truncated at its bounds, and the
    uniform distribution over its bounds without one, which is `0` for a
    parameter with an infinite bound, i.e. the improper flat prior.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate the priors at,
            `nominal_parameters` by default.

    Returns:
        The log prior by the id of the parameter.

    Raises:
        ValueError: if the problem is not initialized, or if the parameters
            of a prior do not fit its distribution.
        KeyError: if the parameters lack a parameter of the fit.
    """
    if parameters is None and not problem.is_initialized:
        raise ValueError(
            f"'{problem.opid}': the nominal parameters require the resolved "
            f"problem, call `initialize(settings)` first."
        )
    pset = parameters if parameters is not None else nominal_parameters(problem)
    priors: dict[str, float] = {}
    for parameter in problem.parameters:
        value = float(pset.values[parameter.pid])
        lower, upper = parameter.lower_bound, parameter.upper_bound
        if parameter.prior is not None:
            try:
                priors[parameter.pid] = parameter.prior.log_density(value, lower, upper)
            except ValueError as err:
                raise ValueError(
                    f"'{problem.opid}', parameter '{parameter.pid}': {err}"
                ) from err
        elif not lower <= value <= upper:
            priors[parameter.pid] = -np.inf
        elif np.isfinite(lower) and np.isfinite(upper):
            priors[parameter.pid] = -float(np.log(upper - lower))
        else:
            priors[parameter.pid] = 0.0
    return priors


def unnorm_log_posterior(
    problem: OptimizationProblem, parameters: ParameterSet | None = None
) -> float:
    """Get the unnormalized log posterior of a problem.

    The sum of the `log_likelihood` and the `log_prior` of the parameters,
    which is the `unnorm_log_posterior` of PEtab.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate it at, `nominal_parameters` by
            default.

    Returns:
        The unnormalized log posterior.

    Raises:
        ValueError: see `log_likelihood` and `log_prior`.
        KeyError: if the parameters lack a parameter of the fit.
    """
    pset = parameters if parameters is not None else nominal_parameters(problem)
    return log_likelihood(problem, pset) + float(sum(log_prior(problem, pset).values()))


#: the orders of the differences of the gradient
GRADIENT_ORDERS: tuple[int, ...] = (2, 4)


#: the fall back of a difference which is logged: fewer points than the
#: order needs, or the secant of the bounds
FALL_BACK_THREE_POINTS = "three points"
FALL_BACK_SECANT = "secant"


def stencil(
    value: float,
    h: float,
    lower_bound: float,
    upper_bound: float,
    order: int = 2,
    name: str = "?",
) -> list[tuple[float, float]]:
    """Get the points and the weights of the difference of a derivative.

    The derivative is the sum of the weights times the values of the function
    at the points. The points stay inside the bounds and keep the step. The
    first difference which has all its points inside the bounds is taken:

    | order | difference | points | error |
    | --- | --- | --- | --- |
    | 4 | central | `x -2h ... x +2h`, four | `h^4` |
    | 4 | forward | `x ... x +4h`, five | `h^4` |
    | 4 | backward | `x -4h ... x`, five | `h^4` |
    | 2 or 4 | central | `x -h`, `x +h` | `h^2` |
    | 2 or 4 | forward | `x ... x +2h`, three | `h^2` |
    | 2 or 4 | backward | `x -2h ... x`, three | `h^2` |
    | any | secant of the bounds | the bounds | the distance of the bounds |

    A step which is shrunk to the distance to a bound, which is what a
    central difference next to a bound needs, divides the error of the
    function by a vanishing step, so the difference is one sided with the
    full step instead. The points are tested, not the distances to the
    bounds: a point which is one ulp outside a bound is not simulated.

    The fall back to a lower order and the secant are logged as a warning,
    the derivative of a network next to a bound is then less exact than the
    order says.

    Args:
        value: the value of the parameter.
        h: the step.
        lower_bound: lower bound of the parameter.
        upper_bound: upper bound of the parameter.
        order: order of the central difference, `2` or `4`.
        name: id of the parameter in the warnings.

    Returns:
        The points with their weights.

    Raises:
        ValueError: if the order is not `2` or `4`, if the step is not
            positive, or if the value is outside the bounds or the bounds are
            equal.
    """
    points, fall_back = _stencil(value, h, lower_bound, upper_bound, order)
    if fall_back is not None:
        _warn_fall_back(fall_back, [name], order)
    return points


def _warn_fall_back(fall_back: str, names: list[str], order: int) -> None:
    """Log the fall back of the differences of parameters, once for all of them.

    Args:
        fall_back: `FALL_BACK_THREE_POINTS` or `FALL_BACK_SECANT`.
        names: ids of the parameters.
        order: order of the difference which was asked for.
    """
    listed = some_ids(names, n=20)
    if fall_back == FALL_BACK_SECANT:
        logger.warning(
            "The bounds of the parameters %s are closer than the steps of the "
            "difference, their derivatives are the secants of the bounds.",
            listed,
        )
    else:
        logger.warning(
            "The bounds of the parameters %s leave less than four steps of "
            "room on both sides for the difference of five points (order %s), "
            "their derivatives are differences of three points.",
            listed,
            order,
        )


def _stencil(
    value: float, h: float, lower_bound: float, upper_bound: float, order: int
) -> tuple[list[tuple[float, float]], str | None]:
    """Get the points and the weights of a difference and its fall back.

    Args:
        value: the value of the parameter.
        h: the step.
        lower_bound: lower bound of the parameter.
        upper_bound: upper bound of the parameter.
        order: order of the central difference, `2` or `4`.

    Returns:
        The points with their weights, see `stencil`, and the fall back,
        `None` for the difference of the order.

    Raises:
        ValueError: see `stencil`.
    """
    if order not in GRADIENT_ORDERS:
        raise ValueError(
            f"The order of the difference is one of {GRADIENT_ORDERS}, not '{order}'."
        )
    if not h > 0.0:
        raise ValueError(f"The step of the difference must be positive, not '{h}'.")
    below = value - lower_bound
    above = upper_bound - value
    if not (below >= 0.0 and above >= 0.0) or (below == 0.0 and above == 0.0):
        raise ValueError(
            f"the difference requires the value inside the bounds "
            f"[{lower_bound} - {upper_bound}] with room for a step, but it is "
            f"'{value}'"
        )

    def inside(*points: float) -> bool:
        """Test that the points are inside the bounds."""
        return all(lower_bound <= point <= upper_bound for point in points)

    if order == 4:
        if inside(value - 2.0 * h, value + 2.0 * h):
            return [
                (value - 2.0 * h, 1.0 / (12.0 * h)),
                (value - h, -8.0 / (12.0 * h)),
                (value + h, 8.0 / (12.0 * h)),
                (value + 2.0 * h, -1.0 / (12.0 * h)),
            ], None
        if inside(value + 4.0 * h):
            return [
                (value, -25.0 / (12.0 * h)),
                (value + h, 48.0 / (12.0 * h)),
                (value + 2.0 * h, -36.0 / (12.0 * h)),
                (value + 3.0 * h, 16.0 / (12.0 * h)),
                (value + 4.0 * h, -3.0 / (12.0 * h)),
            ], None
        if inside(value - 4.0 * h):
            return [
                (value, 25.0 / (12.0 * h)),
                (value - h, -48.0 / (12.0 * h)),
                (value - 2.0 * h, 36.0 / (12.0 * h)),
                (value - 3.0 * h, -16.0 / (12.0 * h)),
                (value - 4.0 * h, 3.0 / (12.0 * h)),
            ], None
    if inside(value - h, value + h):
        points = [(value - h, -0.5 / h), (value + h, 0.5 / h)]
    elif inside(value + 2.0 * h):
        points = [
            (value, -1.5 / h),
            (value + h, 2.0 / h),
            (value + 2.0 * h, -0.5 / h),
        ]
    elif inside(value - 2.0 * h):
        points = [
            (value, 1.5 / h),
            (value - h, -2.0 / h),
            (value - 2.0 * h, 0.5 / h),
        ]
    else:
        # the bounds are closer than the steps of a difference
        distance = upper_bound - lower_bound
        return [
            (lower_bound, -1.0 / distance),
            (upper_bound, 1.0 / distance),
        ], FALL_BACK_SECANT
    return points, FALL_BACK_THREE_POINTS if order == 4 else None


def gradient(
    problem: OptimizationProblem,
    parameters: ParameterSet | None = None,
    step: float = DEFAULT_STEP,
    order: int = 2,
) -> pd.Series:
    """Get the gradient of the log-likelihood by finite differences.

    The differences are taken on the linear scale, i.e. in the units of the
    model, with the step `step * max(|x|, 1)` for a parameter of the value
    `x`, and are central differences, see `stencil`. The model is not
    simulated outside the bounds of a parameter: next to a bound the
    difference is one sided with the full step. A parameter without bounds
    has the central difference. The parameters whose difference falls back
    to fewer points or to the secant of the bounds are logged in one warning
    per fall back and call.

    A difference divides the error of a simulation by the step, so the
    problem is initialized with `FitSettings` of tight tolerances and
    `variable_step_size=False`: with a variable step size the data is
    interpolated on the steps of the integrator, which differ between two
    simulations, and the simulations of one problem differ by `1e-6` however
    tight the tolerances are. The gradient logs a warning in this case.

    The error of the central difference of three points grows with the
    square of the step and the third derivative. The log-likelihood of a
    model which oscillates is strongly curved in the parameters of a network:
    the difference of five points, `order=4`, has an error which grows with
    the fourth power of the step, for four simulations per parameter instead
    of two.

    The gradient is defined inside the bounds only: a parameter outside its
    bounds raises, while `log_likelihood` evaluates the same parameters.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate the gradient at,
            `nominal_parameters` by default.
        step: relative step of the differences.
        order: order of the central difference, `2` or `4`.

    Returns:
        The derivative of the log-likelihood by every parameter of the fit,
        indexed by the ids of the parameters.

    Raises:
        ValueError: if the step is not positive, if the order is not `2` or
            `4`, if a parameter is outside its bounds or its bounds are
            equal, or if the log-likelihood cannot be calculated, see
            `log_likelihood`.
    """
    if not step > 0.0:
        raise ValueError(f"The step of the gradient must be positive, not '{step}'.")
    _check_problem(problem)
    if problem.settings_initialized.variable_step_size:
        logger.warning(
            "'%s': the gradient is calculated with `variable_step_size=True`, "
            "the differences of its simulations are of the size of the step "
            "'%s'. Initialize the problem with `variable_step_size=False` and "
            "tight tolerances for a gradient.",
            problem.opid,
            step,
        )
    pset = parameters if parameters is not None else nominal_parameters(problem)
    x = pset.x(problem.pids)

    def shifted(pid: str, value: float) -> ParameterSet:
        """Get the parameter set with one value replaced."""
        return ParameterSet(
            sid=pset.sid, values={**pset.values, pid: value}, units=dict(pset.units)
        )

    derivatives: dict[str, float] = {}
    # the parameters whose difference fell back, logged once per gradient
    fall_backs: dict[str, list[str]] = {}
    # the log-likelihood at the parameters, which a one sided difference uses
    at_parameters: float | None = None
    for k, parameter in enumerate(problem.parameters):
        pid = parameter.pid
        value = float(x[k])
        try:
            points, fall_back = _stencil(
                value=value,
                h=step * max(abs(value), 1.0),
                lower_bound=float(parameter.lower_bound),
                upper_bound=float(parameter.upper_bound),
                order=order,
            )
        except ValueError as err:
            raise ValueError(
                f"'{problem.opid}': the gradient of the parameter '{pid}': {err}."
            ) from err
        if fall_back is not None:
            fall_backs.setdefault(fall_back, []).append(pid)
        derivative = 0.0
        for point, weight in points:
            if point == value:
                if at_parameters is None:
                    at_parameters = log_likelihood(problem, pset)
                derivative += weight * at_parameters
            else:
                derivative += weight * log_likelihood(problem, shifted(pid, point))
        derivatives[pid] = derivative
    for fall_back in (FALL_BACK_THREE_POINTS, FALL_BACK_SECANT):
        if fall_back in fall_backs:
            _warn_fall_back(fall_back, fall_backs[fall_back], order)
    return pd.Series(derivatives, name="gradient", dtype=float)
