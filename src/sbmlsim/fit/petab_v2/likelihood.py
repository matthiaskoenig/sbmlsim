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

The functions of arrays (`log_density`, `noise_values`) know nothing of a
problem, `log_likelihood` and `gradient` simulate one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sympy as sp
from numpy.typing import ArrayLike
from petab.v2.math import sympify_petab

from sbmlsim.fit.objects import NoiseDistribution, NoiseModel
from sbmlsim.fit.options import ResidualType
from sbmlsim.fit.parameters import ParameterSet

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
            positive finite number, or if a measurement or a simulation of a
            logarithmic distribution is not positive.
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

    distribution = NoiseDistribution(distribution)
    if distribution.is_log:
        for name, values in (("measurement", m), ("simulation", y)):
            if np.any(values <= 0.0):
                raise ValueError(
                    f"The distribution '{distribution.value}' requires positive "
                    f"values, but '{int(np.sum(values <= 0.0))}' of the "
                    f"'{name}' values are not: '{values[values <= 0.0]}'."
                )
        residual = (np.log(m) - np.log(y)) / s
        # the density of the measurement and not of its logarithm, i.e. the
        # jacobian `1 / m` of the transformation is part of it
        scale = s * m
    else:
        residual = (m - y) / s
        scale = s

    if distribution in {NoiseDistribution.NORMAL, NoiseDistribution.LOG_NORMAL}:
        return -0.5 * np.log(2.0 * np.pi * np.square(scale)) - 0.5 * np.square(residual)
    return -np.log(2.0 * scale) - np.abs(residual)


def _evaluate(formula: str | float, variables: Mapping[str, Any], context: str) -> Any:
    """Evaluate a formula of the math of PEtab on the values of its symbols.

    Args:
        formula: the formula, or a number.
        variables: value of every symbol, a number or an array.
        context: what the formula belongs to, for the message of an error.

    Returns:
        The value of the formula, an array if one of its symbols is one.

    Raises:
        ValueError: if a symbol of the formula has no value.
    """
    if isinstance(formula, int | float):
        return float(formula)
    expression = sympify_petab(formula)
    symbols = sorted(expression.free_symbols, key=str)
    missing = [str(symbol) for symbol in symbols if str(symbol) not in variables]
    if missing:
        raise ValueError(
            f"{context}: the formula '{formula}' uses '{missing}', which are "
            f"neither placeholders nor parameters with a value. A noise formula "
            f"is a number, a parameter or a formula of them."
        )
    function = sp.lambdify(symbols, expression, modules="numpy")
    return function(*[variables[str(symbol)] for symbol in symbols])


def noise_values(
    noise: NoiseModel,
    size: int,
    values: Mapping[str, float] | None = None,
    simulation: ArrayLike | None = None,
) -> np.ndarray:
    """Get the scale of the noise of every measurement of a fit mapping.

    The symbols of the noise formula are resolved in this order: a placeholder
    is the value of the measurement, the symbol of the observable is the
    simulation, a parameter is the value of `values` and, without one, the
    nominal value of the noise model. A parameter which a problem estimates is
    therefore evaluated at the value it is given, it is not estimated.

    Args:
        noise: noise model of the fit mapping.
        size: number of measurements.
        values: values of parameters by their id, e.g. the values of a
            parameter set.
        simulation: the simulated values at the measurements, for a noise
            formula which holds the observable.

    Returns:
        The scale of the noise, one value per measurement.

    Raises:
        ValueError: if the noise model has placeholders and not one value of
            them per measurement, if a symbol has no value, or if a scale is
            not a positive finite number.
    """
    context = f"noise formula '{noise.formula}'"
    parameters: dict[str, Any] = {p.pid: p.value for p in noise.parameters}
    parameters.update(values or {})

    variables: dict[str, Any] = dict(parameters)
    if noise.observable is not None and simulation is not None:
        variables[noise.observable] = np.asarray(simulation, dtype=float)
    if noise.placeholders:
        if len(noise.placeholder_values) != size:
            raise ValueError(
                f"{context}: '{len(noise.placeholder_values)}' values of the "
                f"placeholders '{list(noise.placeholders)}' for '{size}' "
                f"measurements."
            )
        for k, placeholder in enumerate(noise.placeholders):
            variables[placeholder] = np.array(
                [
                    float(_evaluate(row[k], parameters, context))
                    for row in noise.placeholder_values
                ],
                dtype=float,
            )

    sigma = np.array(
        np.broadcast_to(
            np.asarray(_evaluate(noise.formula, variables, context), dtype=float),
            (size,),
        )
    )
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            f"{context}: the scale of the noise must be a positive finite "
            f"number, but is '{sigma[~np.isfinite(sigma) | (sigma <= 0.0)]}'."
        )
    return sigma


def default_noise_model(errors: ArrayLike | None) -> NoiseModel:
    """Get the noise model of a fit mapping which does not define one.

    The noise is normal with the standard deviation of the reference data,
    through the placeholder `NOISE_PLACEHOLDER`, and with `DEFAULT_SIGMA` for
    data without errors. This is what the export writes for such a mapping, so
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
    predictions = problem.predictions(pset.x(problem.pids))

    total = 0.0
    for k in problem.training_indices:
        key = f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}"
        noise = noise_model_of(problem, k)
        measurement = np.asarray(problem.y_references[k], dtype=float)
        simulation = predictions[k]
        try:
            sigma = noise_values(
                noise,
                size=measurement.size,
                values=pset.values,
                simulation=simulation,
            )
            density = log_density(
                measurement, simulation, sigma, distribution=noise.distribution
            )
        except ValueError as err:
            raise ValueError(f"'{problem.opid}', fit mapping '{key}': {err}") from err
        total += float(np.sum(density))
    return total


def gradient(
    problem: OptimizationProblem,
    parameters: ParameterSet | None = None,
    step: float = DEFAULT_STEP,
) -> pd.Series:
    """Get the gradient of the log-likelihood by central finite differences.

    The differences are taken on the linear scale, i.e. in the units of the
    model, with the step `step * max(|x|, 1)` for a parameter of the value
    `x`. A difference divides the error of a simulation by the step, so the
    problem is initialized with `FitSettings` of tight tolerances and
    `variable_step_size=False`: with a variable step size the data is
    interpolated on the steps of the integrator, which differ between two
    simulations, and the simulations of one problem differ by `1e-6` however
    tight the tolerances are. The gradient logs a warning in this case.

    Args:
        problem: initialized optimization problem.
        parameters: parameters to evaluate the gradient at,
            `nominal_parameters` by default.
        step: relative step of the differences.

    Returns:
        The derivative of the log-likelihood by every parameter of the fit,
        indexed by the ids of the parameters.

    Raises:
        ValueError: if the step is not positive, or if the log-likelihood
            cannot be calculated, see `log_likelihood`.
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
    for k, pid in enumerate(problem.pids):
        h = step * max(abs(float(x[k])), 1.0)
        plus = log_likelihood(problem, shifted(pid, float(x[k]) + h))
        minus = log_likelihood(problem, shifted(pid, float(x[k]) - h))
        derivatives[pid] = (plus - minus) / (2.0 * h)
    return pd.Series(derivatives, name="gradient", dtype=float)
