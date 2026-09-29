"""Tests of the log-likelihood of a problem."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.fisher import jacobian
from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType, ResidualType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2.likelihood import (
    default_noise_model,
    gradient,
    log_density,
    log_likelihood,
    noise_values,
    nominal_parameters,
)

#: simulations and measurements of the case `sciml_problem_import/001` of the
#: PEtab SciML test suite, normal noise with the scale 0.05
SCIML_001 = Path(__file__).parents[1] / "data" / "petab" / "sciml_001_llh.tsv"
#: the `llh` of the `solutions.yaml` of the case and its `tol_llh`
SCIML_001_LLH = 33.02909543616689
SCIML_001_TOL = 1e-3


def test_log_density_normal() -> None:
    """The density of the normal distribution, calculated by hand."""
    # -0.5 * log(2 pi 0.25) - 0.5 * ((2 - 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi * 0.25) - 2.0
    assert expected == pytest.approx(-2.2257913526447273, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_log_normal() -> None:
    """The simulation is the median and the density is the one of `m`."""
    # -0.5 * log(2 pi 0.25 * 4) - 0.5 * ((log 2 - log 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi) - 2.0 * np.log(1.5) ** 2
    assert expected == pytest.approx(-1.2477424409910038, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_laplace() -> None:
    """The density of the Laplace distribution, calculated by hand."""
    # -log(2 * 0.5) - |2 - 3| / 0.5
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LAPLACE)
    assert density == pytest.approx([-2.0], rel=1e-14)


def test_log_density_log_laplace() -> None:
    """The density of the log-Laplace distribution, calculated by hand."""
    # -log(2 * 0.5 * 2) - |log 2 - log 3| / 0.5
    expected = -np.log(2.0) - 2.0 * np.log(1.5)
    assert expected == pytest.approx(-1.5040773967762742, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    assert density == pytest.approx([expected], rel=1e-14)


@pytest.mark.parametrize("distribution", list(NoiseDistribution))
def test_log_density_is_a_density(distribution: NoiseDistribution) -> None:
    """The density of the measurement integrates to one.

    This is what tells the density of `m` from the density of `log m`, i.e.
    it pins the `1 / m` of the logarithmic distributions.
    """
    start = 1e-6 if distribution.is_log else -27.0
    m = np.linspace(start, start + 60.0, 600001)
    density = np.exp(log_density(m, np.full_like(m, 3.0), 0.25, distribution))
    assert np.trapezoid(density, m) == pytest.approx(1.0, abs=1e-4)


def test_log_density_has_its_maximum_at_the_measurement() -> None:
    """A simulation which hits the measurement is the most likely one."""
    for distribution in NoiseDistribution:
        at, off = log_density([2.0, 2.0], [2.0, 2.5], 0.5, distribution)
        assert at > off


def test_log_density_takes_a_scale_per_measurement() -> None:
    """The scale is one value or one per measurement."""
    density = log_density([1.0, 1.0], [1.0, 1.0], [0.5, 2.0])
    assert density == pytest.approx(
        [-0.5 * np.log(2.0 * np.pi * 0.25), -0.5 * np.log(2.0 * np.pi * 4.0)]
    )


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_log_density_requires_a_positive_scale(sigma: float) -> None:
    """A scale which is not a positive number is an error and not a `nan`."""
    with pytest.raises(ValueError, match="positive finite"):
        log_density([1.0], [1.0], sigma)


def test_log_density_requires_positive_values_for_a_log_distribution() -> None:
    """The logarithm of a measurement which is not positive does not exist."""
    with pytest.raises(ValueError, match="measurement"):
        log_density([0.0], [1.0], 0.5, NoiseDistribution.LOG_NORMAL)
    with pytest.raises(ValueError, match="simulation"):
        log_density([1.0], [-1.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    # the normal distribution takes them
    assert np.isfinite(log_density([-1.0], [0.0], 0.5)).all()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("distribution", list(NoiseDistribution))
def test_log_density_requires_finite_values(
    value: float, distribution: NoiseDistribution
) -> None:
    """A value which is not finite is an error and not a `nan` or `-inf`."""
    with pytest.raises(ValueError, match=r"finite.*measurement"):
        log_density([1.0, value], [1.0, 1.0], 0.5, distribution)
    with pytest.raises(ValueError, match=r"finite.*simulation"):
        log_density([1.0, 1.0], [value, 1.0], 0.5, distribution)


def test_log_density_requires_one_shape() -> None:
    """Every measurement has a simulation."""
    with pytest.raises(ValueError, match="shape"):
        log_density([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="scale"):
        log_density([1.0, 2.0], [1.0, 2.0], [0.5, 0.5, 0.5])


def test_log_likelihood_of_the_sciml_test_suite() -> None:
    """The `llh` of a case of the PEtab SciML test suite is reproduced.

    This pins the sign and the constant of the log-likelihood against a value
    which another tool calculated.
    """
    df = pd.read_csv(SCIML_001, sep="\t", float_precision="round_trip")
    assert len(df) == 20
    llh = float(np.sum(log_density(df.measurement, df.simulation, 0.05)))
    assert llh == pytest.approx(SCIML_001_LLH, abs=SCIML_001_TOL)
    assert llh == pytest.approx(SCIML_001_LLH, rel=1e-12)


def test_noise_values_of_a_number() -> None:
    """A noise formula which is a number is the scale of every measurement."""
    sigma = noise_values(NoiseModel(formula="0.05"), size=3)
    assert sigma.tolist() == [0.05, 0.05, 0.05]


def test_noise_values_of_a_placeholder() -> None:
    """A placeholder has a value per measurement."""
    noise = NoiseModel(
        formula="0.1 + 2 * sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (1.0,)),
    )
    assert noise_values(noise, size=2) == pytest.approx([1.1, 2.1])


def test_noise_values_of_a_parameter() -> None:
    """A parameter is evaluated at the given value, the nominal one without."""
    noise = NoiseModel(
        formula="sigma_a ^ 2",
        parameters=(NoiseParameter(pid="sigma_a", value=3.0, estimate=True),),
    )
    assert noise_values(noise, size=2) == pytest.approx([9.0, 9.0])
    assert noise_values(noise, size=2, values={"sigma_a": 2.0}) == pytest.approx(
        [4.0, 4.0]
    )


def test_noise_values_of_a_placeholder_which_is_a_parameter() -> None:
    """The value of a placeholder is a number or a formula of parameters."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), ("sigma_a",), ("2 * sigma_a",)),
        parameters=(NoiseParameter(pid="sigma_a", value=3.0),),
    )
    assert noise_values(noise, size=3) == pytest.approx([0.5, 3.0, 6.0])


def test_noise_values_of_the_observable() -> None:
    """A noise which is proportional to the simulation."""
    noise = NoiseModel(formula="0.1 * obs_a + 0.01", observable="obs_a")
    sigma = noise_values(noise, size=2, simulation=[1.0, 2.0])
    assert sigma == pytest.approx([0.11, 0.21])


def test_noise_values_of_the_observable_require_the_simulation() -> None:
    """A noise formula of the observable is not evaluated without it."""
    noise = NoiseModel(formula="0.1 * obs_a", observable="obs_a")
    with pytest.raises(ValueError, match="requires the simulation"):
        noise_values(noise, size=2)


def test_noise_values_of_a_placeholder_named_like_a_parameter() -> None:
    """A placeholder is the value of the measurement, whatever else has its name."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (0.25,)),
        parameters=(NoiseParameter(pid="sd", value=3.0),),
    )
    sigma = noise_values(noise, size=2, values={"sd": 7.0})
    assert sigma.tolist() == [0.5, 0.25]


def test_noise_values_require_every_symbol() -> None:
    """A symbol without a value is named."""
    with pytest.raises(ValueError, match="k_unknown"):
        noise_values(NoiseModel(formula="2 * k_unknown"), size=2)


def test_noise_values_require_a_value_per_measurement() -> None:
    """The values of the placeholders are the ones of the measurements."""
    noise = NoiseModel(
        formula="sd", placeholders=("sd",), placeholder_values=((0.5,), (1.0,))
    )
    with pytest.raises(ValueError, match="'2' values"):
        noise_values(noise, size=3)


def test_noise_values_require_a_positive_scale() -> None:
    """A noise formula which is not positive is an error."""
    with pytest.raises(ValueError, match="positive finite"):
        noise_values(NoiseModel(formula="-0.05"), size=2)


def test_a_noise_model_requires_a_value_per_placeholder() -> None:
    """A measurement has a value for every placeholder."""
    with pytest.raises(ValueError, match="placeholders"):
        NoiseModel(
            formula="a + b", placeholders=("a", "b"), placeholder_values=((0.5,),)
        )


def test_default_noise_model() -> None:
    """Without a noise model the noise is the standard deviation of the data."""
    noise = default_noise_model(np.array([0.5, 0.25]))
    assert noise.distribution is NoiseDistribution.NORMAL
    assert noise_values(noise, size=2).tolist() == [0.5, 0.25]
    # and the scale is one for data without errors
    assert noise_values(default_noise_model(None), size=2).tolist() == [1.0, 1.0]


# --- THE LOG-LIKELIHOOD OF A PROBLEM ---


@pytest.fixture
def settings_likelihood() -> FitSettings:
    """Get settings with which the cost is the sum of squares of the data.

    The integrator is tighter than in a fit and its output grid is fixed: a
    finite difference of the log-likelihood divides the error of a simulation
    by the step, and with a variable step size two simulations of a problem
    differ by `1e-6`, see `sbmlsim.fit.petab_v2.likelihood.gradient`.
    """
    return FitSettings(
        residual=ResidualType.ABSOLUTE,
        parameter_scale=ParameterScaleType.LINEAR,
        variable_step_size=False,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )


@pytest.fixture
def op_unit_noise(
    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
) -> OptimizationProblem:
    """Get the initialized problem with a normal noise of scale one."""
    op_hctz_pk.initialize(settings_likelihood)
    op_hctz_pk.noise_models = [
        NoiseModel(formula="1.0") for _ in op_hctz_pk.mapping_keys
    ]
    return op_hctz_pk


def test_the_hctz_problem_has_a_log_likelihood(
    op_hctz_pk: OptimizationProblem, definition_hctz_pk: FitDefinition
) -> None:
    """The reference problem has a log-likelihood, with the settings of its fit."""
    op_hctz_pk.initialize(definition_hctz_pk.settings)
    assert all(noise is None for noise in op_hctz_pk.noise_models)

    llh = log_likelihood(op_hctz_pk)
    assert np.isfinite(llh)
    assert llh < 0.0


def test_log_likelihood_of_the_nominal_parameters_by_default(
    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
) -> None:
    """The nominal values are the start values of the parameters."""
    op_hctz_pk.initialize(settings_likelihood)
    nominal = nominal_parameters(op_hctz_pk)
    assert nominal.sid == "nominal"
    assert nominal.values == {p.pid: p.start_value for p in op_hctz_pk.parameters}

    llh = log_likelihood(op_hctz_pk)
    assert llh == pytest.approx(log_likelihood(op_hctz_pk, nominal), rel=1e-8)
    # and other parameters are another likelihood
    other = ParameterSet(
        sid="other", values={pid: 2.0 * v for pid, v in nominal.values.items()}
    )
    assert log_likelihood(op_hctz_pk, other) != pytest.approx(llh, rel=1e-3)


def test_log_likelihood_is_the_sum_over_the_training_data(
    op_hctz_pk: OptimizationProblem, settings_likelihood: FitSettings
) -> None:
    """The validation data and the outliers do not enter."""
    op_hctz_pk.initialize(settings_likelihood)
    assert op_hctz_pk.validation_indices
    nominal = nominal_parameters(op_hctz_pk)
    predictions = op_hctz_pk.predictions(nominal.x(op_hctz_pk.pids))
    assert sorted(predictions) == op_hctz_pk.training_indices

    expected = 0.0
    for k in op_hctz_pk.training_indices:
        errors = op_hctz_pk.y_errors[k]
        sigma = errors if errors is not None else 1.0
        expected += float(
            np.sum(log_density(op_hctz_pk.y_references[k], predictions[k], sigma))
        )
    assert log_likelihood(op_hctz_pk) == pytest.approx(expected, rel=1e-8)


def test_log_likelihood_of_a_unit_noise_is_the_cost(
    op_unit_noise: OptimizationProblem,
) -> None:
    """With a normal noise of scale one the cost is the log-likelihood.

    `llh = -n/2 log(2 pi) - 0.5 sum((y - m)^2)`, and the second term is the
    cost of a fit of the absolute residuals without weights.
    """
    problem = op_unit_noise
    nominal = nominal_parameters(problem)
    x = nominal.x(problem.pids)
    n = sum(len(problem.y_references[k]) for k in problem.training_indices)

    cost = problem.cost_least_square(problem.to_scale(x))
    assert log_likelihood(problem, nominal) == pytest.approx(
        -0.5 * n * np.log(2.0 * np.pi) - cost, rel=1e-8
    )


def test_gradient_of_a_unit_noise_is_the_gradient_of_the_cost(
    op_unit_noise: OptimizationProblem,
) -> None:
    """The gradient is `-J' r` of the residuals of the fit."""
    problem = op_unit_noise
    nominal = nominal_parameters(problem)
    x = nominal.x(problem.pids)

    residuals = np.asarray(problem.residuals(problem.to_scale(x)), dtype=float)
    expected = -jacobian(problem, problem.to_scale(x)).T @ residuals

    grad = gradient(problem, nominal)
    assert list(grad.index) == problem.pids
    # a difference of two log-likelihoods has the rounding error of their
    # size, which is not relative to a small derivative
    scale = float(np.max(np.abs(expected)))
    assert grad.to_numpy() == pytest.approx(expected, rel=1e-4, abs=1e-6 * scale)
    assert np.all(grad.to_numpy() != 0.0)


def test_log_likelihood_uses_the_noise_model_of_a_mapping(
    op_unit_noise: OptimizationProblem,
) -> None:
    """A parameter of the noise is evaluated at the value of the set."""
    problem = op_unit_noise
    nominal = nominal_parameters(problem)
    n = sum(len(problem.y_references[k]) for k in problem.training_indices)
    unit = log_likelihood(problem, nominal)

    problem.noise_models = [
        NoiseModel(
            formula="sigma_a",
            parameters=(NoiseParameter(pid="sigma_a", value=1.0, estimate=True),),
        )
        for _ in problem.mapping_keys
    ]
    assert log_likelihood(problem, nominal) == pytest.approx(unit, rel=1e-8)

    # llh(s) = -n log(s) - n/2 log(2 pi) - cost / s^2
    cost = -unit - 0.5 * n * np.log(2.0 * np.pi)
    wide = ParameterSet(sid="wide", values={**nominal.values, "sigma_a": 2.0})
    assert log_likelihood(problem, wide) == pytest.approx(
        -n * np.log(2.0) - 0.5 * n * np.log(2.0 * np.pi) - cost / 4.0, rel=1e-8
    )
    # the gradient is the one of the parameters of the fit
    assert list(gradient(problem, wide).index) == problem.pids


def test_log_likelihood_requires_an_initialized_problem(
    op_hctz_iv: OptimizationProblem,
) -> None:
    """The data of the problem has to be resolved."""
    with pytest.raises(ValueError, match="initialize"):
        log_likelihood(op_hctz_iv)


def test_log_likelihood_requires_the_measurements(
    op_hctz_iv: OptimizationProblem,
) -> None:
    """Data which is shifted to its baseline is not what the noise describes."""
    op_hctz_iv.initialize(FitSettings(residual=ResidualType.ABSOLUTE_TO_BASELINE))
    with pytest.raises(ValueError, match="baseline"):
        log_likelihood(op_hctz_iv)


def test_log_likelihood_names_the_mapping_of_an_error(
    op_unit_noise: OptimizationProblem,
) -> None:
    """An error of a noise model says which mapping it belongs to."""
    problem = op_unit_noise
    k = problem.training_indices[0]
    problem.noise_models[k] = NoiseModel(formula="k_unknown")
    with pytest.raises(ValueError, match=problem.mapping_keys[k]):
        log_likelihood(problem)


def test_log_likelihood_of_a_log_distribution_requires_positive_data(
    op_unit_noise: OptimizationProblem,
) -> None:
    """The amount in the urine is zero at the first measurement."""
    problem = op_unit_noise
    k = next(
        k
        for k in problem.training_indices
        if np.any(np.asarray(problem.y_references[k]) <= 0.0)
    )
    problem.noise_models[k] = NoiseModel(
        formula="0.5", distribution=NoiseDistribution.LOG_NORMAL
    )
    with pytest.raises(ValueError, match="positive") as excinfo:
        log_likelihood(problem)
    assert problem.mapping_keys[k] in str(excinfo.value)


def test_gradient_of_a_failed_simulation_raises(
    op_unit_noise: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step which the model cannot simulate is an error and not a `nan`."""
    simulator = op_unit_noise.runner_initialized.simulator
    assert simulator is not None

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CVODE failed")

    monkeypatch.setattr(simulator, "_timecourses", fail)
    with pytest.raises(ValueError, match="failed") as excinfo:
        gradient(op_unit_noise)
    # the first mapping which is simulated, at the values of every parameter
    k = op_unit_noise.training_indices[0]
    assert op_unit_noise.mapping_keys[k] in str(excinfo.value)
    assert all(pid in str(excinfo.value) for pid in op_unit_noise.pids)


def _record_predictions(
    problem: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> list[np.ndarray]:
    """Record the parameters every simulation of the problem is run at."""
    evaluated: list[np.ndarray] = []
    predictions = problem.predictions

    def record(
        x: np.ndarray, indices: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        evaluated.append(np.asarray(x, dtype=float).copy())
        return predictions(x, indices=indices)

    monkeypatch.setattr(problem, "predictions", record)
    return evaluated


def test_gradient_stays_inside_the_bounds(
    op_unit_noise: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parameter smaller than the step is not simulated below its bound.

    `KI__HCTZEX_k` has the lower bound `1e-10`, the step `1e-6` of the
    central difference would take it to a negative value.
    """
    problem = op_unit_noise
    pid = "KI__HCTZEX_k"
    k = problem.pids.index(pid)
    parameter = problem.parameters[k]
    assert parameter.lower_bound == 1e-10
    nominal = nominal_parameters(problem)
    small = ParameterSet(sid="small", values={**nominal.values, pid: 1e-8})

    evaluated = _record_predictions(problem, monkeypatch)
    grad = gradient(problem, small)
    assert np.all(np.isfinite(grad.to_numpy()))
    assert all(
        p.lower_bound <= x[j] <= p.upper_bound
        for x in evaluated
        for j, p in enumerate(problem.parameters)
    )
    # the central difference with the step shrunk to the distance to the bound
    assert min(x[k] for x in evaluated) == pytest.approx(parameter.lower_bound)


def test_gradient_at_a_bound_is_one_sided(
    op_unit_noise: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parameter at its lower bound of zero has the forward difference."""
    problem = op_unit_noise
    pid = "KI__HCTZEX_k"
    k = problem.pids.index(pid)
    monkeypatch.setattr(problem.parameters[k], "lower_bound", 0.0)
    nominal = nominal_parameters(problem)
    at_bound = ParameterSet(sid="bound", values={**nominal.values, pid: 0.0})

    evaluated = _record_predictions(problem, monkeypatch)
    grad = gradient(problem, at_bound)
    assert min(x[k] for x in evaluated) == 0.0
    assert all(x[k] >= 0.0 for x in evaluated)
    # the derivative is the one next to the bound
    near = ParameterSet(sid="near", values={**nominal.values, pid: 1e-8})
    assert grad[pid] == pytest.approx(gradient(problem, near)[pid], rel=1e-2)


def test_log_likelihood_requires_the_parameters_of_the_fit(
    op_unit_noise: OptimizationProblem,
) -> None:
    """A parameter set which lacks a parameter is an error."""
    with pytest.raises(KeyError, match="does not contain"):
        log_likelihood(op_unit_noise, ParameterSet(sid="empty", values={}))


def test_gradient_warns_about_a_variable_step_size(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The differences of simulations on a variable grid are noise."""
    assert fit_settings.variable_step_size
    op_hctz_iv.initialize(fit_settings)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
        grad = gradient(op_hctz_iv)
    assert "variable_step_size" in caplog.text
    assert np.all(np.isfinite(grad.to_numpy()))


def test_gradient_requires_a_positive_step(
    op_unit_noise: OptimizationProblem,
) -> None:
    """A step of zero divides by zero."""
    with pytest.raises(ValueError, match="step"):
        gradient(op_unit_noise, step=0.0)


def test_gradient_requires_the_parameters_inside_their_bounds(
    op_unit_noise: OptimizationProblem,
) -> None:
    """A parameter outside its bounds has no difference inside them."""
    problem = op_unit_noise
    pid = "KI__HCTZEX_k"
    nominal = nominal_parameters(problem)
    outside = ParameterSet(sid="outside", values={**nominal.values, pid: 0.0})
    with pytest.raises(ValueError, match=rf"{pid}.*bounds"):
        gradient(problem, outside)
