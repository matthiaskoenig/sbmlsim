"""The likelihood and the posterior as the objective of a fit (#428)."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit import FitSettings, ParameterSet, fit_core
from sbmlsim.fit.objects import MappingKind, NoiseDistribution
from sbmlsim.fit.optimization import OptimizationProblem
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
from tests.engines import require_core_fit
from tests.fit.test_petab_v2_reader import SETTINGS, write_problem

LIKELIHOOD = replace(SETTINGS, objective=ObjectiveType.LIKELIHOOD)

#: how closely the residual of the scale of the noise of the engine follows
#: the one of the problem. It is the one row of a request whose logarithm the
#: engine evaluates itself, `ln` of rust, where python evaluates `numpy.log`:
#: the two agree to within an ulp of each other and not bitwise. The rows of
#: the data stay bitwise, see `fit_core`.
LOGARITHM_RTOL = 1e-15


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


def _scattered(tmp_path: Path, distribution: str = "normal") -> PetabReader:
    """Read the problem whose only estimated parameter is the scale of its noise.

    The measurements scatter around the simulation by about 0.1, so the
    maximum likelihood estimate of `sigma` is a point of its own, see
    `test_the_scale_of_the_noise_is_estimated`.
    """
    # the tables of the problem, which the reader below reads once they are
    # perturbed
    _reader(tmp_path, "sigma", distribution, sigma=True)
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    measurements["measurement"] += 0.1 * np.array(
        [(-1) ** k * (1 + k / 10) for k in range(10)]
    )
    measurements.to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    return PetabReader.from_yaml(tmp_path / "problem.yaml")


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
    # a fixed normal noise sends no `likelihood` at all, its request is the
    # weighted least squares of the scale of its noise
    sigma = fit_core.fixed_noise(problem, 0)
    assert sigma is not None
    curve = fit_core.fit_request(problem)["curves"][0]
    assert "likelihood" not in curve
    assert curve["noise"] == pytest.approx(list(sigma))


@pytest.mark.parametrize("distribution", ["laplace", "log-laplace"])
def test_a_laplace_likelihood_is_a_fit_of_the_engine(
    tmp_path: Path, distribution: str
) -> None:
    """The residual of a Laplace noise is the one of the `likelihood` of its curve.

    The scale of the noise is fixed, so it is the `noise` of the curve, and
    the distribution of the `likelihood` is what makes the residual of a
    measurement `sign(r) sqrt(2 |r|)` and gives the curve its scale rows.
    """
    reader = _reader(tmp_path, noise_formula="0.05", distribution=distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    assert fit_core.gaps_of(problem) == []
    sigma = fit_core.fixed_noise(problem, 0)
    assert sigma is not None
    curve = fit_core.fit_request(problem)["curves"][0]
    assert curve["likelihood"] == {"distribution": distribution}
    assert curve["noise"] == pytest.approx(list(sigma))


def test_an_estimated_scale_of_the_noise_is_a_fit_of_the_engine(
    tmp_path: Path,
) -> None:
    """A noise whose scale is a parameter of the fit is a column of the model.

    The scale of such a noise moves with the search, so it is no number of a
    curve: the model of the fit carries the noise formula as an observable of
    its own and the curve names it as its `noise_column`.
    """
    reader = _reader(tmp_path, "sigma", "normal", sigma=True)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert problem.pids == ["sigma"]
    assert fit_core.fixed_noise(problem, 0) is None

    assert fit_core.gaps_of(problem) == []
    curve = fit_core.fit_request(problem)["curves"][0]
    assert curve["likelihood"] == {"distribution": "normal", "noise_column": ANY}
    assert "noise" not in curve


def test_a_noise_which_reads_the_simulation_is_a_fit_of_the_engine(
    tmp_path: Path,
) -> None:
    """A noise whose scale follows the simulation is a column of the model too.

    Its observable stands for the simulation, which the noise column of the
    model reads off the linear column of the mapping.
    """
    reader = _reader(tmp_path, "0.01 + 0.1 * prey_o", "normal")
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert fit_core.fixed_noise(problem, 0) is None

    assert fit_core.gaps_of(problem) == []
    curve = fit_core.fit_request(problem)["curves"][0]
    assert curve["likelihood"] == {"distribution": "normal", "noise_column": ANY}
    assert "noise" not in curve


def _check_rows_and_cost(problem: OptimizationProblem) -> None:
    """Check the rows and the cost of the request of a problem of one mapping.

    The rows of a curve of a `likelihood` are the residuals of the data and
    then, in the same order, the ones of the scales of its noise, which the
    residuals of the problem have in the same places.

    The cost is a check of its own: the rows fix only `2 c + offset` of the
    constant `c` of the density of the engine and of its `LIKELIHOOD_OFFSET`,
    and they say nothing at all about what `likelihood_constant` takes off the
    cost of a run, which is what makes the cost of the engine the negative
    log-likelihood of the problem.
    """
    assert len(problem.indices(MappingKind.TRAINING)) == 1
    size = len(problem.y_references[0])
    np.testing.assert_array_equal(fit_core._residual_rows(problem), np.arange(2 * size))
    core = fit_core.loaded_model(problem).fit_problem(
        fit_core.fit_request(problem), threads=1
    )
    x = np.asarray(problem.to_scale(problem.xmodel), dtype=float)
    residuals = np.asarray(core.residuals(x), dtype=float)
    reference = np.asarray(problem.residuals(x), dtype=float)

    assert residuals.shape == reference.shape
    np.testing.assert_array_equal(residuals[:size], reference[:size])
    np.testing.assert_allclose(
        residuals[size:], reference[size:], rtol=LOGARITHM_RTOL, atol=0.0
    )
    cost = float(core.cost(x)) + fit_core.likelihood_constant(problem)
    assert cost == pytest.approx(problem.cost_least_square(x), rel=1e-12)


@pytest.mark.parametrize("distribution", ["normal", "log-normal"])
def test_the_engine_computes_the_residuals_of_a_moved_noise_as_the_problem_does(
    tmp_path: Path, distribution: str
) -> None:
    """Every row of a curve whose noise the fit moves is the row of the problem.

    The noise column of the model is the number python computes, as a formula
    observable is, so the rows of the data are bitwise; the row of the scale
    carries the logarithm of that number, which the engine evaluates itself.
    """
    require_core_fit()
    reader = _reader(tmp_path, "sigma", distribution, sigma=True)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    _check_rows_and_cost(problem)


@pytest.mark.parametrize("distribution", ["laplace", "log-laplace"])
def test_the_engine_computes_the_residuals_of_a_laplace_likelihood_as_the_problem_does(
    tmp_path: Path, distribution: str
) -> None:
    """Every row of a Laplace curve is the row of the problem."""
    require_core_fit()
    reader = _reader(tmp_path, "0.05", distribution)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    _check_rows_and_cost(problem)


@pytest.mark.parametrize("distribution", ["normal", "laplace"])
def test_a_fit_of_a_moved_noise_in_the_engine_reports_the_log_likelihood(
    tmp_path: Path, distribution: str
) -> None:
    """The cost of a run of a moved noise is the negative log-likelihood.

    The engine carries the scale rows of such a curve itself, so
    `likelihood_constant` takes only the offsets of those rows off its cost,
    where a curve which sends no `likelihood` has the residuals of its scales
    added as well; the rows of the request say nothing about either, see
    `_check_rows_and_cost`.
    """
    require_core_fit()
    reader = _scattered(tmp_path, distribution)
    problem = reader.to_optimization_problem()
    settings = replace(LIKELIHOOD, parameter_scale=ParameterScaleType.LOG10)
    problem.initialize(settings)
    assert problem.pids == ["sigma"]
    assert fit_core.gaps_of(problem) == []

    result = fit_core.run(problem, settings, size=2, seed=1234)

    best = min(result.fits, key=lambda fit: fit.cost)
    x = np.asarray(best.x, dtype=float)
    assert best.cost == pytest.approx(
        problem.cost_least_square(problem.to_scale(x)), rel=1e-9
    )
    pset = ParameterSet(sid="fit", values=dict(zip(problem.pids, x, strict=True)))
    assert best.cost == pytest.approx(-log_likelihood(problem, pset), rel=1e-9)


def test_a_noise_of_a_value_per_measurement_is_a_gap_of_the_engine(
    tmp_path: Path,
) -> None:
    """A placeholder of the noise which differs between the measurements is no column.

    The model of the fit carries one value of a placeholder, so a noise whose
    placeholder is one parameter of the fit for some measurements and another
    one for the others has no one column of the simulation.
    """
    write_problem(
        tmp_path,
        {"prey_o": "prey"},
        {"e1": None},
        noise_parameters=lambda _, k: "alpha" if k < 5 else "beta",
    )
    problem = PetabReader.from_yaml(tmp_path / "problem.yaml").to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert fit_core.fixed_noise(problem, 0) is None

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"objective"}
    assert "value per measurement" in gaps[0].message


def test_a_measurement_which_is_not_positive_under_a_log_noise_is_a_gap(
    tmp_path: Path,
) -> None:
    """A log kind whose measurement is not positive has no log-likelihood.

    Python reports the residuals of a failure for such a mapping and keeps
    the plain offset of its scale, where the curve of the engine would carry
    the logarithm of the measurement as its value; the mapping is a gap
    rather than another objective.
    """
    _reader(tmp_path, "0.1", "log-normal")
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    measurements.loc[0, "measurement"] = 0.0
    measurements.to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    problem = PetabReader.from_yaml(tmp_path / "problem.yaml").to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"objective"}
    assert "log-normal" in gaps[0].message
    assert "0.0" in gaps[0].message


def test_bindings_before_the_objective_version_keep_a_moved_noise_in_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A curve of a `likelihood` needs bindings which read the field.

    Neither `FitRequest` nor `CurveRequest` denies the fields it does not
    know, so bindings before `OBJECTIVE_VERSION` would ignore the block and
    minimize weighted least squares of the scale of the noise without a word.
    """
    core = pytest.importorskip("sbmlsim_core")
    reader = _reader(tmp_path, "sigma", "normal", sigma=True)
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert fit_core.gaps_of(problem) == []
    monkeypatch.setattr(core, "__version__", "0.22.0")

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"bindings"}
    assert fit_core.OBJECTIVE_VERSION in gaps[0].message


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

    models = fit_core.fit_models(problem)
    model = next(iter(models.values()))
    conditions, curves = fit_core._conditions_and_curves(problem, models)
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


def test_a_jacobian_of_a_fixed_noise_likelihood_comes_from_the_engine(
    tmp_path: Path,
) -> None:
    """The scale of a noise the fit does not move has no derivative, its rows are zero."""
    reader = _reader(tmp_path, noise_formula="0.05", distribution="normal")
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)
    assert fit_core.gaps_of(problem) == []

    assert fit_core.gaps_of_a_jacobian(problem) == []


def test_a_jacobian_of_a_likelihood_whose_noise_the_fit_moves_is_a_gap(
    tmp_path: Path,
) -> None:
    """The scale of a noise which follows the parameters has a derivative of its own."""
    reader = _reader(tmp_path, noise_formula="0.05", distribution="normal")
    problem = reader.to_optimization_problem()
    problem.initialize(LIKELIHOOD)

    with patch.object(fit_core, "fixed_noise", return_value=None):
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
