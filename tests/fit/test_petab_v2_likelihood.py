"""Tests of the log-likelihood of a problem."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.fisher import jacobian
from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
from sbmlsim.fit.optimization import MappingEvaluation, OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType, ResidualType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2.likelihood import (
    chi2,
    default_noise_model,
    gradient,
    log_density,
    log_likelihood,
    noise_values,
    nominal_parameters,
    normalized_residuals,
    stencil,
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


def test_noise_values_of_a_selection() -> None:
    """A symbol which the simulation selects has its value at every measurement.

    The simulation knows the value a condition or the fit gave the parameter,
    so it takes precedence over the values and the nominal value.
    """
    noise = NoiseModel(
        formula="sigma_a * [S]",
        parameters=(NoiseParameter(pid="sigma_a", value=3.0),),
    )
    sigma = noise_values(
        noise,
        size=2,
        values={"sigma_a": 7.0},
        selections={"sigma_a": np.array([1.0, 2.0]), "[S]": np.array([0.5, 0.5])},
    )
    assert sigma == pytest.approx([0.5, 1.0])


def test_noise_values_of_a_placeholder_which_reads_a_selection() -> None:
    """The formula of a placeholder value is evaluated at its measurement."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=(("2 * k",), ("k",), (0.5,)),
    )
    sigma = noise_values(noise, size=3, selections={"k": np.array([1.0, 3.0, 9.0])})
    assert sigma == pytest.approx([2.0, 3.0, 0.5])


def test_the_symbols_of_a_noise_model() -> None:
    """The symbols are what a noise formula reads besides placeholders and observable."""
    noise = NoiseModel(
        formula="sigma_a * obs_a + sd",
        placeholders=("sd",),
        placeholder_values=(("k1 * 2",), (0.5,)),
        observable="obs_a",
    )
    assert noise.symbols == ("k1", "sigma_a")


def test_a_selection_of_the_measurements_of_a_noise_model() -> None:
    """A selection of the measurements keeps their placeholder values."""
    noise = NoiseModel(
        formula="sd", placeholders=("sd",), placeholder_values=((1.0,), (2.0,))
    )
    assert noise.select(np.array([False, True])).placeholder_values == ((2.0,),)


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


def test_a_noise_model_coerces_its_fields() -> None:
    """The distribution is the enum and the sequences are tuples."""
    noise = NoiseModel(
        formula="sd",
        distribution="log-normal",  # ty: ignore[invalid-argument-type]
        placeholders=["sd"],  # ty: ignore[invalid-argument-type]
        placeholder_values=[[0.5], [0.25]],  # ty: ignore[invalid-argument-type]
        parameters=[NoiseParameter(pid="sigma_a", value=1.0)],  # ty: ignore[invalid-argument-type]
    )
    assert noise.distribution is NoiseDistribution.LOG_NORMAL
    assert noise.placeholders == ("sd",)
    assert noise.placeholder_values == ((0.5,), (0.25,))
    assert noise.parameters == (NoiseParameter(pid="sigma_a", value=1.0),)
    # a frozen dataclass of tuples compares and hashes
    assert noise == NoiseModel(
        formula="sd",
        distribution=NoiseDistribution.LOG_NORMAL,
        placeholders=("sd",),
        placeholder_values=((0.5,), (0.25,)),
        parameters=(NoiseParameter(pid="sigma_a", value=1.0),),
    )
    assert hash(noise)


@pytest.mark.parametrize("formula", ["", "  "])
def test_a_noise_model_requires_a_formula(formula: str) -> None:
    """An empty noise formula is an error."""
    with pytest.raises(ValueError, match="formula"):
        NoiseModel(formula=formula)


def test_a_noise_model_requires_a_distribution_of_petab() -> None:
    """A distribution which PEtab does not define is named."""
    with pytest.raises(ValueError, match="'gauss' is not one of PEtab"):
        NoiseModel(formula="0.1", distribution="gauss")  # ty: ignore[invalid-argument-type]


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


def test_normalized_residuals() -> None:
    """The residual is the difference in units of the scale of the noise."""
    m = np.array([1.0, 2.0])
    y = np.array([1.5, 1.0])
    sigma = np.array([0.5, 0.5])
    assert normalized_residuals(m, y, sigma) == pytest.approx([-1.0, 2.0])
    assert normalized_residuals(
        m, y, sigma, distribution=NoiseDistribution.LAPLACE
    ) == pytest.approx([-1.0, 2.0])
    for distribution in [NoiseDistribution.LOG_NORMAL, NoiseDistribution.LOG_LAPLACE]:
        assert normalized_residuals(
            m, y, sigma, distribution=distribution
        ) == pytest.approx((np.log(m) - np.log(y)) / sigma)


def test_chi2_of_a_unit_noise_is_twice_the_cost(
    op_unit_noise: OptimizationProblem,
) -> None:
    """With a normal noise of scale one chi2 is the sum of squares."""
    problem = op_unit_noise
    nominal = nominal_parameters(problem)
    cost = problem.cost_least_square(problem.to_scale(nominal.x(problem.pids)))
    assert chi2(problem, nominal) == pytest.approx(2.0 * cost, rel=1e-8)
    assert chi2(problem) == pytest.approx(2.0 * cost, rel=1e-8)


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

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CVODE failed")

    monkeypatch.setattr("sbmlsim.fit.optimization.execute", fail)
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
    evaluations = problem.evaluations

    def record(
        x: np.ndarray, indices: list[int] | None = None
    ) -> dict[int, MappingEvaluation]:
        evaluated.append(np.asarray(x, dtype=float).copy())
        return evaluations(x, indices=indices)

    monkeypatch.setattr(problem, "evaluations", record)
    return evaluated


def test_gradient_stays_inside_the_bounds(
    op_unit_noise: OptimizationProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parameter smaller than the step is not simulated below its bound.

    `KI__HCTZEX_k` has the lower bound `1e-10`, the step `1e-6` of the
    central difference would take it to a negative value. The difference is
    the forward one with the full step: a step which is shrunk to the distance
    to the bound divides the error of the simulation by `1e-8`.
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
    points = sorted({float(x[k]) for x in evaluated})
    assert points == pytest.approx([1e-8, 1e-8 + 1e-6, 1e-8 + 2e-6], rel=1e-12)


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


# --- THE STENCIL OF A DIFFERENCE ---


def _derivative(points: list[tuple[float, float]], f: Any) -> float:
    return sum(weight * f(point) for point, weight in points)


@pytest.mark.parametrize("order", [2, 4])
def test_the_central_difference(order: int) -> None:
    """A parameter with room on both sides has the central difference."""
    points = stencil(1.0, 0.1, 0.0, 2.0, order=order)
    assert len(points) == {2: 2, 4: 4}[order]
    assert min(p for p, _ in points) == pytest.approx(1.0 - order / 2 * 0.1)
    assert max(p for p, _ in points) == pytest.approx(1.0 + order / 2 * 0.1)
    # exact for a polynomial of the order
    assert _derivative(points, lambda x: x**order) == pytest.approx(order * 1.0)
    assert _derivative(points, lambda x: 3.0 * x + 1.0) == pytest.approx(3.0)


def test_the_difference_next_to_a_bound_keeps_its_step() -> None:
    """A bound closer than the step gives a one sided difference of full step."""
    forward = stencil(0.01, 0.1, 0.0, 2.0)
    assert [p for p, _ in forward] == pytest.approx([0.01, 0.11, 0.21])
    assert _derivative(forward, lambda x: x**2) == pytest.approx(0.02)
    backward = stencil(1.99, 0.1, 0.0, 2.0)
    assert [p for p, _ in backward] == pytest.approx([1.99, 1.89, 1.79])
    assert _derivative(backward, lambda x: x**2) == pytest.approx(3.98)
    # at the bounds
    assert [p for p, _ in stencil(0.0, 0.1, 0.0, 2.0)] == pytest.approx([0.0, 0.1, 0.2])
    at_upper = stencil(2.0, 0.1, 0.0, 2.0)
    assert [p for p, _ in at_upper] == pytest.approx([2.0, 1.9, 1.8])
    assert _derivative(at_upper, lambda x: x**2) == pytest.approx(4.0)


def test_the_difference_of_bounds_closer_than_the_steps() -> None:
    """Bounds closer than the steps of a difference give the secant of the interval."""
    points = stencil(0.5, 1.0, 0.0, 1.0)
    assert [p for p, _ in points] == [0.0, 1.0]
    assert _derivative(points, lambda x: x**2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": 3}, r"order.*\(2, 4\)"),
        ({"h": 0.0}, "step of the difference must be positive"),
        ({"value": 3.0}, "inside the bounds"),
        ({"value": 1.0, "lower_bound": 1.0, "upper_bound": 1.0}, "room for a step"),
    ],
)
def test_a_stencil_which_does_not_exist(kwargs: dict, message: str) -> None:
    """The order, the step and the bounds of a difference are checked."""
    arguments: dict[str, Any] = {
        "value": 1.0,
        "h": 0.1,
        "lower_bound": 0.0,
        "upper_bound": 2.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        stencil(**arguments)


def test_the_gradient_of_the_fourth_order(op_unit_noise: OptimizationProblem) -> None:
    """The five point difference agrees with the three point one."""
    problem = op_unit_noise
    second = gradient(problem)
    fourth = gradient(problem, order=4)
    assert list(fourth.index) == problem.pids
    np.testing.assert_allclose(fourth.to_numpy(), second.to_numpy(), rtol=1e-3)
    with pytest.raises(ValueError, match=r"order of the difference is one of \(2, 4\)"):
        gradient(problem, order=3)


def _points(*args: Any, **kwargs: Any) -> list[float]:
    return [p for p, _ in stencil(*args, **kwargs)]


def test_the_stencil_never_leaves_the_bounds() -> None:
    """A point one ulp outside a bound is not simulated."""
    # `value - h` is one ulp below the lower bound
    lower = -0.04584310855955209
    points = _points(0.03633893549909927, 0.08218204405865137, lower, 41.0576)
    assert min(points) >= lower
    # a small positive lower bound, the value within two ulp of `lower + h`
    rng = np.random.default_rng(1)
    for _ in range(2000):
        lower = float(rng.choice([1e-12, 1e-10, 1e-8]))
        h = float(rng.uniform(0.1, 1.0)) * lower * 10.0
        value = float(lower + h + rng.integers(-2, 3) * np.spacing(lower + h))
        if value < lower:
            continue
        for order in (2, 4):
            points = _points(value, h, lower, 100.0, order=order)
            assert min(points) >= lower
            assert max(points) <= 100.0


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_the_one_sided_five_point_difference(degree: int) -> None:
    """The one sided five point differences are exact up to the degree 4."""
    x0 = 0.7

    def f(x: float) -> float:
        return x**degree

    exact = degree * x0 ** (degree - 1)
    forward = stencil(x0, 0.1, x0, 5.0, order=4)
    backward = stencil(x0, 0.1, 0.0, x0, order=4)
    assert _points(x0, 0.1, x0, 5.0, order=4) == pytest.approx(
        [x0 + k * 0.1 for k in range(5)]
    )
    assert _points(x0, 0.1, 0.0, x0, order=4) == pytest.approx(
        [x0 - k * 0.1 for k in range(5)]
    )
    assert _derivative(forward, f) == pytest.approx(exact)
    assert _derivative(backward, f) == pytest.approx(exact)


@pytest.mark.parametrize(
    ("value", "lower", "upper", "order", "count"),
    [
        # the central difference of three points
        (1.0, 0.0, 2.0, 2, 2),
        # the forward and backward differences of three points
        (0.0, 0.0, 2.0, 2, 3),
        (2.0, 0.0, 2.0, 2, 3),
        # the central difference of five points
        (1.0, 0.0, 2.0, 4, 4),
        # the one sided differences of five points next to a bound
        (0.0, 0.0, 2.0, 4, 5),
        (0.05, 0.0, 2.0, 4, 5),
        (0.15, 0.0, 2.0, 4, 5),
        (1.95, 0.0, 2.0, 4, 5),
        # the room `2 h` below is enough for the central difference
        (0.3, 0.0, 2.0, 4, 4),
        (0.5, 0.0, 2.0, 4, 4),
    ],
)
def test_the_stencil_of_a_position(
    value: float, lower: float, upper: float, order: int, count: int
) -> None:
    """The order 4 keeps its order next to a bound."""
    points = _points(value, 0.1, lower, upper, order=order)
    assert len(points) == count
    assert all(lower <= p <= upper for p in points)


@pytest.mark.parametrize(
    ("value", "lower", "upper", "order", "power"),
    [
        (1.0, 0.0, 2.0, 2, 2),
        (0.0, 0.0, 2.0, 2, 2),
        (2.0, 0.0, 2.0, 2, 2),
        (1.0, 0.0, 2.0, 4, 4),
        (0.0, 0.0, 2.0, 4, 4),
        (2.0, 0.0, 2.0, 4, 4),
    ],
)
def test_the_truncation_order_of_a_stencil(
    value: float, lower: float, upper: float, order: int, power: int
) -> None:
    """The error of a difference falls with the power of the step."""

    def error(h: float) -> float:
        points = stencil(value, h, lower, upper, order=order)
        return abs(_derivative(points, np.exp) - np.exp(value))

    assert error(0.04) / error(0.02) == pytest.approx(2.0**power, rel=0.15)


def test_the_order_4_falls_back_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Less than `4 h` of room on both sides gives the difference of three points."""
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
        points = stencil(0.12, 0.1, 0.0, 0.26, order=4, name="p1")
    assert len(points) == 2
    assert "'p1'" in caplog.text
    assert "three points" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
        stencil(1.0, 0.1, 0.0, 2.0, order=4, name="p1")
        stencil(1.0, 0.1, 0.0, 2.0, order=2, name="p1")
    assert not caplog.text


def test_the_gradient_warns_once_about_the_parameters_next_to_a_bound(
    op_unit_noise: OptimizationProblem,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A network has hundreds of elements, one warning names all of them."""
    problem = op_unit_noise
    nominal = nominal_parameters(problem)
    # two parameters without room for five points, one without room for three
    for pid, room in [
        (problem.pids[0], 1.5),
        (problem.pids[1], 1.5),
        (problem.pids[2], 0.5),
    ]:
        k = problem.pids.index(pid)
        value = nominal.values[pid]
        h = 1e-6 * max(abs(value), 1.0)
        monkeypatch.setattr(problem.parameters[k], "lower_bound", value - room * h)
        monkeypatch.setattr(problem.parameters[k], "upper_bound", value + room * h)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
        gradient(problem, nominal, order=4)
    records = [r.getMessage() for r in caplog.records]
    assert len(records) == 2
    three, secant = records
    assert f"{problem.pids[:2]}" in three
    assert "three points" in three
    assert f"{problem.pids[2:3]}" in secant
    assert "secant" in secant


def test_the_secant_warns(caplog: pytest.LogCaptureFixture) -> None:
    """The secant has a step different from the step of the difference."""
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.likelihood"):
        points = stencil(0.5, 1.0, 0.0, 1.0, name="p1")
    assert [p for p, _ in points] == [0.0, 1.0]
    assert "'p1'" in caplog.text
    assert "secant" in caplog.text
