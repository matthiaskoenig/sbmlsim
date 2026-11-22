"""The likelihood and the posterior as the objective of a fit (#428)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitSettings, ParameterSet, fit_core
from sbmlsim.fit.objects import NoiseDistribution
from sbmlsim.fit.options import ObjectiveType, ParameterScaleType, ResidualType
from sbmlsim.fit.petab_v2 import gaps_of_problem
from sbmlsim.fit.petab_v2.likelihood import (
    likelihood_offsets,
    likelihood_residuals,
    log_density,
    log_likelihood,
    log_prior,
)
from sbmlsim.fit.petab_v2.reader import PetabReader
from tests.fit.test_petab_v2_reader import SETTINGS, write_problem

LIKELIHOOD = replace(SETTINGS, objective=ObjectiveType.LIKELIHOOD)


def _reader(
    tmp_path: Path,
    noise_formula: str = "0.05",
    distribution: str = "normal",
    sigma: bool = False,
    prior: str = "",
) -> PetabReader:
    """Read the problem of the prey with a noise model of its own.

    Args:
        tmp_path: directory of the problem.
        noise_formula: the noise formula of the observable.
        distribution: the noise distribution of the observable.
        sigma: whether the parameter `sigma` is estimated, and only it, so the
            parameters of the model are their nominal values.
        prior: the prior distribution and its parameters of `alpha`, e.g.
            `normal;1.3;0.1`.
    """
    path = write_problem(tmp_path, {"prey_o": "prey"}, {"e1": None})
    observables = pd.read_csv(tmp_path / "observables.tsv", sep="\t")
    observables["noiseFormula"] = noise_formula
    observables["noiseDistribution"] = distribution
    observables.to_csv(tmp_path / "observables.tsv", sep="\t", index=False)
    parameters = pd.read_csv(tmp_path / "parameters.tsv", sep="\t")
    parameters["lowerBound"] = 0.1
    if prior:
        name, *values = prior.split(";")
        parameters["priorDistribution"] = ""
        parameters["priorParameters"] = ""
        parameters.loc[parameters["parameterId"] == "alpha", "priorDistribution"] = name
        parameters.loc[parameters["parameterId"] == "alpha", "priorParameters"] = (
            ";".join(values)
        )
    if sigma:
        parameters["estimate"] = False
        parameters = pd.concat(
            [
                parameters,
                pd.DataFrame(
                    [
                        {
                            "parameterId": "sigma",
                            "lowerBound": 1e-6,
                            "upperBound": 10.0,
                            "nominalValue": 1.0,
                            "estimate": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    parameters.to_csv(tmp_path / "parameters.tsv", sep="\t", index=False)
    return PetabReader.from_yaml(path)


@pytest.mark.parametrize(
    ("distribution", "noise_formula"),
    [
        ("normal", "0.05"),
        ("log-normal", "0.1"),
        ("laplace", "0.05"),
        ("log-laplace", "0.1"),
        ("normal", "0.01 + 0.1 * prey_o"),
    ],
)
def test_the_cost_is_the_negative_log_likelihood(
    tmp_path: Path, distribution: str, noise_formula: str
) -> None:
    """The half sum of squares of the residuals is the likelihood, up to its offsets."""
    reader = _reader(tmp_path, noise_formula, distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    for factor in (1.0, 1.1):
        pset = reader.nominal_parameters(problem)
        x = np.asarray(pset.x(problem.pids)) * factor
        cost = problem.cost_least_square(problem.to_scale(x))
        pset = ParameterSet(sid="x", values=dict(zip(problem.pids, x, strict=True)))
        assert cost == pytest.approx(-log_likelihood(problem, pset), rel=1e-12)


def test_the_cost_of_the_posterior_holds_the_priors(tmp_path: Path) -> None:
    """The prior of a parameter enters, a parameter without one is uniform."""
    reader = _reader(tmp_path, prior="normal;1.3;0.1")
    problem = reader.to_optimization_problem()
    problem.initialize(replace(SETTINGS, objective=ObjectiveType.POSTERIOR))
    pset = reader.nominal_parameters(problem)
    x = np.asarray(pset.x(problem.pids)) * 1.05
    pset = ParameterSet(sid="x", values=dict(zip(problem.pids, x, strict=True)))
    priors = log_prior(problem, pset)
    expected = -(log_likelihood(problem, pset) + priors["alpha"])
    assert problem.cost_least_square(problem.to_scale(x)) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize(
    ("distribution", "estimate"),
    [
        # the maximum likelihood estimates of the scale of the noise
        ("normal", lambda r: np.sqrt(np.mean(np.square(r)))),
        ("laplace", lambda r: np.mean(np.abs(r))),
    ],
)
def test_the_scale_of_the_noise_is_estimated(
    tmp_path: Path, distribution: str, estimate
) -> None:
    """An estimated parameter of the noise reaches its maximum likelihood estimate.

    The parameters of the model are fixed, so the estimate of `sigma` is the
    root mean square (normal) or the mean absolute value (Laplace) of the
    residuals, which least squares cannot estimate, its cost does not depend
    on `sigma`.
    """
    reader = _reader(tmp_path, "sigma", distribution, sigma=True)
    # the measurements scatter around the simulation by about 0.1
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    scatter = 0.1 * np.array([(-1) ** k * (1 + k / 10) for k in range(10)])
    measurements["measurement"] += scatter
    measurements.to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    reader = PetabReader.from_yaml(tmp_path / "problem.yaml")
    problem = reader.to_optimization_problem()
    problem.initialize(replace(LIKELIHOOD, parameter_scale=ParameterScaleType.LOG10))
    assert problem.pids == ["sigma"]
    (prediction,) = problem.predictions(np.array([1.0])).values()
    residuals = np.asarray(problem.y_references[0]) - prediction
    fits, _ = problem.optimize(size=1, seed=1)
    sigma = float(fits[0].x[0])
    # within the tolerances of the optimizer, far below the uncertainty of the
    # estimate of 10 measurements
    assert sigma == pytest.approx(estimate(residuals), rel=1e-3)
    pset = ParameterSet(sid="fit", values={"sigma": sigma})
    assert fits[0].cost == pytest.approx(-log_likelihood(problem, pset), rel=1e-12)


def test_the_residuals_of_a_likelihood() -> None:
    """The residuals of the data and of the scale give the negative log density."""
    m = np.array([1.0, 2.0, 0.5])
    y = np.array([1.1, 1.8, 0.4])
    s = np.array([0.2, 0.5, 0.1])
    for distribution in NoiseDistribution:
        data, scale = likelihood_residuals(m, y, s, distribution)
        offsets = likelihood_offsets(m, distribution)
        cost = 0.5 * (np.square(data) + np.square(scale)) - 0.5 * offsets
        np.testing.assert_allclose(cost, -log_density(m, y, s, distribution))
        assert np.all(np.sign(data) == np.sign(y - m))


def test_the_objective_of_the_settings() -> None:
    """The objective is stored with the settings, a string is converted."""
    settings = FitSettings(objective="likelihood")  # ty: ignore[invalid-argument-type]
    assert settings.objective is ObjectiveType.LIKELIHOOD
    assert FitSettings.from_dict(settings.to_dict()) == settings
    # settings stored without an objective are least squares
    stored = FitSettings().to_dict()
    del stored["objective"]
    assert FitSettings.from_dict(stored).objective is ObjectiveType.LEAST_SQUARES
    with pytest.raises(ValueError, match="'objective' is one of"):
        FitSettings(objective="maximum")  # ty: ignore[invalid-argument-type]


def test_a_likelihood_of_data_shifted_to_its_baseline_raises(tmp_path: Path) -> None:
    problem = _reader(tmp_path).to_optimization_problem()
    with pytest.raises(ValueError, match="ABSOLUTE_TO_BASELINE"):
        problem.initialize(
            replace(LIKELIHOOD, residual=ResidualType.ABSOLUTE_TO_BASELINE)
        )


def test_a_problem_read_without_the_extension_fits_the_likelihood(
    tmp_path: Path,
) -> None:
    """PEtab defines the objective as the likelihood, which has no noise gaps."""
    reader = _reader(tmp_path, "sigma", sigma=True)
    assert reader.settings.objective is ObjectiveType.LIKELIHOOD
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert not ids & {"noise-model", "noise-parameters"}
    problem.initialize(SETTINGS)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert {"noise-model", "noise-parameters"} <= ids


# ----------------------------------------------------------------------------
# the likelihood in sbmlsim-core (#613)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("distribution", ["normal", "log-normal"])
def test_a_likelihood_of_a_fixed_noise_is_a_fit_of_the_engine(
    tmp_path: Path, distribution: str
) -> None:
    """A likelihood whose noise the fit does not move has no gap (#613).

    The negative log-likelihood of a measurement is `0.5 res**2` plus a term
    of the scale of its noise, which is constant for a noise the fit does not
    move, so the residuals of the data are the weighted residuals of a curve
    of the request.
    """
    reader = _reader(tmp_path, noise_formula="0.05", distribution=distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    assert fit_core.gaps_of(problem) == []


@pytest.mark.parametrize("distribution", ["laplace", "log-laplace"])
def test_a_laplace_likelihood_is_a_gap_of_the_engine(
    tmp_path: Path, distribution: str
) -> None:
    """The residual of a Laplace noise is no difference divided by a deviation."""
    reader = _reader(tmp_path, noise_formula="0.05", distribution=distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    gaps = fit_core.gaps_of(problem)

    assert [gap.code for gap in gaps] == ["objective"]
    assert distribution in gaps[0].message


def test_an_estimated_scale_of_the_noise_is_a_gap_of_the_engine(
    tmp_path: Path,
) -> None:
    """A noise whose scale is a parameter of the fit is no number of a curve."""
    reader = _reader(tmp_path, "sigma", "normal", sigma=True)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert problem.pids == ["sigma"]

    gaps = fit_core.gaps_of(problem)

    assert [gap.code for gap in gaps] == ["objective"]
    assert "sigma" in gaps[0].message
    assert fit_core.fixed_noise(problem, 0) is None


def test_a_noise_which_reads_the_simulation_is_a_gap_of_the_engine(
    tmp_path: Path,
) -> None:
    """A noise whose scale follows the simulation is no number of a curve."""
    reader = _reader(tmp_path, "0.01 + 0.1 * prey_o", "normal")
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    gaps = fit_core.gaps_of(problem)

    assert [gap.code for gap in gaps] == ["objective"]
    assert fit_core.fixed_noise(problem, 0) is None


def test_a_posterior_is_a_gap_of_the_engine(tmp_path: Path) -> None:
    """The residuals of the priors are functions of the parameters alone."""
    reader = _reader(tmp_path, prior="normal;1.3;0.1")
    problem = reader.to_optimization_problem()
    problem.initialize(replace(SETTINGS, objective=ObjectiveType.POSTERIOR))

    gaps = fit_core.gaps_of(problem)

    assert [gap.code for gap in gaps] == ["objective"]
    assert "posterior" in gaps[0].message


@pytest.mark.parametrize("distribution", ["normal", "log-normal"])
def test_the_engine_simulates_the_data_of_a_likelihood_bitwise(
    tmp_path: Path, distribution: str
) -> None:
    """The column a curve of a likelihood reads is what python evaluates, bitwise.

    A log-normal noise reads the logarithm of the observable off a column of
    the model of the fit, whose value is `log` of the column itself; the
    model of the fit integrates with the tolerances of the problem, so its
    simulation is the one of the problem and not one of looser settings.
    """
    pytest.importorskip("sbmlsim_core")
    reader = _reader(tmp_path, noise_formula="0.05", distribution=distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    x = np.asarray(problem.x0, dtype=float)
    expected = np.asarray(
        problem.residuals(problem.to_scale(x), complete_data=True)["y_obsip"][0],
        dtype=float,
    )
    if distribution == "log-normal":
        expected = np.log(expected)

    model = fit_core.fit_model(problem)
    conditions, curves = fit_core._conditions_and_curves(problem, model)
    out = model.batch_executor(1).simulate_many([conditions[0]["request"]])[0]
    columns = list(out.columns)
    values = np.asarray(out.values)
    times = values[:, 0]
    at = [int(np.argmin(np.abs(times - t))) for t in curves[0]["times"]]
    simulated = values[:, columns.index(curves[0]["column"])][at]

    np.testing.assert_array_equal(simulated, expected)


@pytest.mark.parametrize("distribution", ["normal", "log-normal"])
def test_a_fit_of_a_likelihood_in_the_engine_reports_the_log_likelihood(
    tmp_path: Path, distribution: str
) -> None:
    """The cost of a run of the engine is the negative log-likelihood (#613).

    The engine minimizes the half sum of the squares of the residuals of the
    data; `likelihood_constant` is the term of the scales of the noise and
    the offsets, which the run is shifted by, so the cost of a run is the one
    a fit in python reports at the same parameters.
    """
    pytest.importorskip("sbmlsim_core")
    reader = _reader(tmp_path, noise_formula="0.05", distribution=distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    result = fit_core.run(problem, LIKELIHOOD, size=4, seed=1234)

    best = min(result.fits, key=lambda fit: fit.cost)
    x = np.asarray(best.x, dtype=float)
    assert best.cost == pytest.approx(
        problem.cost_least_square(problem.to_scale(x)), abs=1e-9, rel=1e-12
    )
    pset = ParameterSet(sid="fit", values=dict(zip(problem.pids, x, strict=True)))
    assert best.cost == pytest.approx(-log_likelihood(problem, pset), rel=1e-9)


def test_a_jacobian_of_a_likelihood_is_a_gap_of_the_engine(tmp_path: Path) -> None:
    """A jacobian of the engine has the rows of the curves, the ones of the data.

    The residuals of a likelihood are the ones of the data and of the scales
    of the noise, so the rows of the jacobian of the problem are not the ones
    of the engine, although the scales have no derivative.
    """
    reader = _reader(tmp_path, noise_formula="0.05", distribution="normal")
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert fit_core.gaps_of(problem) == []

    gaps = fit_core.gaps_of_a_jacobian(problem)

    assert [gap.code for gap in gaps] == ["objective"]


def test_the_model_of_a_fit_keeps_the_tolerances_of_the_problem(
    tmp_path: Path,
) -> None:
    """The model of a fit with an observable integrates as the problem does (#613).

    `observable_model` loads the model again from the SBML with the
    observables written into it, which gave it the settings of the engine
    instead of the tolerances of the `FitSettings`, so such a fit integrated
    at a relative tolerance of 1e-6 where the fit asked for 1e-12.
    """
    pytest.importorskip("sbmlsim_core")
    path = write_problem(tmp_path, {"prey_o": "2 * prey"}, {"e1": None})
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    of_the_problem = fit_core._models_of(problem)[0]
    of_the_fit = fit_core.fit_model(problem)
    assert of_the_fit is not of_the_problem

    assert of_the_fit.request_settings() == of_the_problem.request_settings()
    assert of_the_fit.integrator_state() == of_the_problem.integrator_state()
