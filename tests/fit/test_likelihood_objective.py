"""The likelihood and the posterior as the objective of a fit (#428)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitSettings, ParameterSet
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
