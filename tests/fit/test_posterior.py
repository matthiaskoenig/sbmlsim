"""The priors of a fit and the Laplace approximation of its posterior (#190).

The model has the parameters `p1` and `p2` and the observables `y = p1 + p2`
and `q = p1 * p2` as assignment rules, so the residuals of `y` are linear in
the parameters on the linear scale and the residuals of `q` on the log scale:
the Gauss-Newton curvature is exact and the Laplace approximation is the
conjugate normal posterior, known in closed form.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitSettings, ParameterSet, fit_core
from sbmlsim.fit.cli import load_fit_records
from sbmlsim.fit.fisher import SENSITIVITIES, FisherInformation, fisher_information
from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
from sbmlsim.fit.methods import ReportContext, methods_section
from sbmlsim.fit.objects import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    Prior,
    PriorDistribution,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    RUN_IN_PYTHON,
    RUN_IN_THE_ENGINE,
    ObjectiveType,
    ParameterScaleType,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.posterior import ParameterPosterior, posterior_laplace
from sbmlsim.fit.priors import JointPrior, prior_distributions, prior_model
from sbmlsim.fit.records import parameter_correlation
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization
from sbmlsim.model import AbstractModel
from sbmlsim.report.document import Document
from sbmlsim.report.render import HTMLRenderer
from sbmlsim.simulation import Simulation
from sbmlsim.simulation.sampling import Truncated, Uniform, posterior_parameters
from sbmlsim.simulation.sampling.distributions import LogNormal, Normal
from sbmlsim.task import Task
from sbmlsim.uncertainty import Covariance
from sbmlsim.units import Quantity
from tests.engines import require_core_fit
from tests.fit.test_likelihood_objective import LOGARITHM_RTOL
from tests.simulator.models import sbml

MODEL = """
model observables
  compartment C = 1;
  species A in C;
  A = 1
  p1 = 1.2; p2 = 2.0
  y := p1 + p2
  q := p1 * p2
end
"""

#: the same model with a unit on `p1`, so that the factor of a parameter in
#: a scaled unit of it into the model is not `1`
UNIT_MODEL = """
model observables
  unit per_time = 1 / second;
  compartment C = 1;
  species A in C;
  A = 1
  p1 = 1.2 per_time; p2 = 2.0
  y := p1 + p2
  q := p1 * p2
end
"""

TIMES = np.array([0.0, 1.0, 2.0, 3.0])
#: the data of y and q, and their standard deviation
Y = np.array([3.1, 3.3, 3.0, 3.4])
Q = np.array([2.3, 2.6, 2.5, 2.2])
SD = 0.1

MODEL_PATH: dict[str, Path] = {}


class ObservablesExperiment(SimulationExperiment):
    """The data of `y` and `q` with their standard deviation."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=MODEL_PATH["path"])}

    def datasets(self) -> dict[str, DataSet]:
        data = pd.DataFrame(
            {
                "time": TIMES,
                "time_unit": "s",
                "y": Y,
                "y_sd": SD,
                "y_unit": "dimensionless",
                "q": Q,
                "q_sd": SD,
                "q_unit": "dimensionless",
            }
        )
        return {"d": DataSet.from_df(data, ureg=self.ureg)}

    def simulations(self) -> dict[str, Simulation]:
        return {"s": Simulation(end=3)}

    def tasks(self) -> dict[str, Task]:
        return {"t": Task(model="m", simulation="s")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            yid: FitMapping(
                self,
                reference=FitData(
                    self, dataset="d", xid="time", yid=yid, yid_sd=f"{yid}_sd"
                ),
                observable=FitData(self, task="t", xid="time", yid=yid),
            )
            for yid in ("y", "q")
        }


def _parameter(
    pid: str,
    prior: Prior | None = None,
    lb: float = 0.1,
    unit: str = "dimensionless",
) -> FitParameter:
    return FitParameter(
        pid=pid,
        lower_bound=lb,
        upper_bound=10.0,
        start_value=1.0,
        unit=unit,
        prior=prior,
    )


def _problem(
    tmp_path: Path,
    parameters: list[FitParameter],
    mappings: list[str],
    correlation: Covariance | None = None,
    objective: ObjectiveType = ObjectiveType.POSTERIOR,
    scale: ParameterScaleType = ParameterScaleType.LINEAR,
    model: str = MODEL,
) -> tuple[OptimizationProblem, FitSettings]:
    MODEL_PATH["path"] = tmp_path / "observables.xml"
    MODEL_PATH["path"].write_text(sbml(model), encoding="utf-8")
    problem = OptimizationProblem(
        opid="observables",
        mapping_collections=[
            FitMappingCollection(experiment=ObservablesExperiment, mappings=mappings)
        ],
        fit_parameters=parameters,
        base_path=tmp_path,
        data_path=tmp_path,
        prior_correlation=correlation,
    )
    settings = FitSettings(
        parameter_scale=scale,
        objective=objective,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
    problem.initialize(settings)
    return problem, settings


def _pset(problem: OptimizationProblem, values: list[float]) -> ParameterSet:
    return ParameterSet(
        sid="x", values=dict(zip(problem.pids, values, strict=True)), units={}
    )


NORMAL_1 = Prior(PriorDistribution.NORMAL, (1.3, 0.2))
NORMAL_2 = Prior(PriorDistribution.NORMAL, (1.9, 0.3))
CORRELATION = Covariance.correlation({("p1", "p2"): 0.5})


def test_the_posterior_of_a_normal_prior_is_the_conjugate_normal(
    tmp_path: Path,
) -> None:
    """One parameter, a normal prior and normal data: the conjugate posterior."""
    problem, settings = _problem(tmp_path, [_parameter("p1", NORMAL_1)], ["y"])
    data = Y - 2.0
    precision = len(data) / SD**2 + 1.0 / 0.2**2
    mean = (data.sum() / SD**2 + 1.3 / 0.2**2) / precision
    posterior = posterior_laplace(problem, settings, _pset(problem, [mean]))
    assert posterior.covariance[0, 0] == pytest.approx(1.0 / precision, rel=1e-6)
    assert posterior.values[0] == pytest.approx(mean)
    assert posterior.objective is ObjectiveType.POSTERIOR
    assert posterior.sigma2 == 1.0
    assert posterior.n == len(TIMES)
    assert posterior.problems == []


def test_correlated_normal_priors_are_whitened(tmp_path: Path) -> None:
    """The cost is the negative log posterior with the multivariate normal prior."""
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y"],
        correlation=CORRELATION,
    )
    covariance = np.array([[0.04, 0.5 * 0.2 * 0.3], [0.5 * 0.2 * 0.3, 0.09]])
    prior = multivariate_normal(mean=[1.3, 1.9], cov=covariance)
    for x in ([1.2, 2.0], [1.5, 1.4]):
        pset = _pset(problem, x)
        expected = -(log_likelihood(problem, pset) + prior.logpdf(x))
        cost = problem.cost_least_square(problem.to_scale(np.array(x)))
        assert cost == pytest.approx(expected, rel=1e-10)
        assert problem.log_prior_density(np.array(x)) == pytest.approx(
            prior.logpdf(x), rel=1e-12
        )

    posterior = posterior_laplace(problem, settings, _pset(problem, [1.2, 2.0]))
    jac = np.ones((len(TIMES), 2)) / SD
    expected_precision = jac.T @ jac + np.linalg.inv(covariance)
    np.testing.assert_allclose(
        posterior.covariance, np.linalg.inv(expected_precision), rtol=1e-5
    )


def test_correlated_log_normal_priors_on_the_log_scale(tmp_path: Path) -> None:
    """The prior adds `ln(10)² Σ⁻¹` of the logarithms to the precision of log10."""
    priors = [
        Prior(PriorDistribution.LOG_NORMAL, (math.log(1.2), 0.3)),
        Prior(PriorDistribution.LOG_NORMAL, (math.log(2.0), 0.4)),
    ]
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", priors[0]), _parameter("p2", priors[1])],
        ["q"],
        correlation=CORRELATION,
        scale=ParameterScaleType.LOG10,
    )
    x = np.array([1.1, 2.2])
    sigma = np.array([[0.09, 0.5 * 0.3 * 0.4], [0.5 * 0.3 * 0.4, 0.16]])
    log_x = np.log(x)
    expected_log_prior = (
        multivariate_normal(mean=[math.log(1.2), math.log(2.0)], cov=sigma).logpdf(
            log_x
        )
        - log_x.sum()
    )
    assert problem.log_prior_density(x) == pytest.approx(expected_log_prior, rel=1e-12)

    posterior = posterior_laplace(problem, settings, _pset(problem, list(x)), step=1e-6)
    jac = np.full((len(TIMES), 2), math.log(10.0) * x.prod() / SD)
    precision = jac.T @ jac + math.log(10.0) ** 2 * np.linalg.inv(sigma)
    np.testing.assert_allclose(
        posterior.covariance, np.linalg.inv(precision), rtol=1e-5
    )
    assert posterior.scales == [ParameterScaleType.LOG10] * 2


def test_the_posterior_of_least_squares_is_the_fisher_covariance(
    tmp_path: Path,
) -> None:
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1"), _parameter("p2")],
        ["y", "q"],
        objective=ObjectiveType.LEAST_SQUARES,
    )
    pset = _pset(problem, [1.2, 2.0])
    fisher = fisher_information(problem, settings, pset)
    posterior = posterior_laplace(problem, settings, pset, fisher=fisher)
    np.testing.assert_allclose(posterior.covariance, fisher.covariance, rtol=1e-8)
    assert posterior.sigma2 == pytest.approx(fisher.sigma2)


def test_the_fisher_information_of_a_likelihood(tmp_path: Path) -> None:
    """The residuals of a likelihood are in units of the noise, without the priors."""
    problem, settings = _problem(
        tmp_path, [_parameter("p1", NORMAL_1)], ["y"], objective=ObjectiveType.POSTERIOR
    )
    fisher = fisher_information(problem, settings, _pset(problem, [1.2]))
    assert not fisher.scaled
    assert fisher.sigma2 == 1.0
    assert fisher.n == len(TIMES)
    # the information of the data alone, the prior is no part of it
    assert fisher.matrix[0, 0] == pytest.approx(len(TIMES) / SD**2, rel=1e-6)
    restored = type(fisher).from_dict(fisher.to_dict())
    assert restored.objective is ObjectiveType.POSTERIOR


def test_an_unconstrained_direction_raises(tmp_path: Path) -> None:
    """The data of `y = p1 + p2` does not constrain `p1 - p2`."""
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1"), _parameter("p2")],
        ["y"],
        objective=ObjectiveType.LIKELIHOOD,
    )
    with pytest.raises(ValueError, match="do not constrain 1 direction"):
        posterior_laplace(problem, settings, _pset(problem, [1.2, 2.0]))


def test_a_parameter_at_a_bound_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    problem, settings = _problem(tmp_path, [_parameter("p1", NORMAL_1, lb=1.0)], ["y"])
    posterior = posterior_laplace(problem, settings, _pset(problem, [1.0]))
    assert "at a bound" in posterior.problems[0]
    assert "at a bound" in caplog.text


@pytest.fixture
def posterior(tmp_path: Path) -> ParameterPosterior:
    """Get the posterior of `p1` (log10) and `p2` (linear) of the data of `q`."""
    p1 = _parameter("p1", Prior(PriorDistribution.LOG_NORMAL, (math.log(1.2), 0.3)))
    p2 = _parameter("p2", NORMAL_2)
    p2.scale = ParameterScaleType.LINEAR
    problem, settings = _problem(
        tmp_path, [p1, p2], ["q"], scale=ParameterScaleType.LOG10
    )
    return posterior_laplace(problem, settings, _pset(problem, [1.2, 2.0]))


def test_a_posterior_is_stored_as_json(
    tmp_path: Path, posterior: ParameterPosterior
) -> None:
    path = posterior.write(tmp_path)
    assert path.name == "posterior.json"
    restored = ParameterPosterior.read(tmp_path)
    np.testing.assert_array_equal(restored.covariance, posterior.covariance)
    assert restored.scales == posterior.scales
    assert restored.to_dict() == posterior.to_dict()


def test_the_draws_of_a_posterior_are_its_normal(posterior: ParameterPosterior) -> None:
    dimension = posterior_parameters(posterior, 40000, seed=1)
    draws = np.column_stack(
        [np.log10(dimension.values["p1"].m), dimension.values["p2"].m]
    )
    np.testing.assert_allclose(draws.mean(axis=0), posterior.mean, atol=0.01)
    np.testing.assert_allclose(
        np.cov(draws.T), posterior.covariance, rtol=0.03, atol=1e-4
    )
    assert dimension.design is not None
    assert dimension.design.method == "posterior_parameters"


def test_a_posterior_is_the_source_of_a_propagation(
    posterior: ParameterPosterior,
) -> None:
    from sbmlsim.uncertainty.draws import draw

    distributions = posterior.distributions()
    assert isinstance(distributions["p1"], LogNormal)
    assert isinstance(distributions["p2"], Normal)
    sd = posterior.standard_deviations
    assert distributions["p1"].cv == pytest.approx(
        math.sqrt(math.expm1((math.log(10.0) * sd[0]) ** 2))
    )
    model = posterior.uncertainty_model()
    assert model.parameters == distributions
    correlation = model.covariance[0]
    assert correlation.kind == "correlation"
    assert correlation.matrix[0, 1] == pytest.approx(posterior.correlation_matrix[0, 1])
    rank = posterior.rank_correlation()
    r = posterior.correlation_matrix[0, 1]
    assert rank[0, 1] == pytest.approx(6.0 / math.pi * math.asin(r / 2.0))

    # the draws of a propagation are the normal of the space of the parameters
    values = draw(model.variables({}), model.covariance, 2**14, design="random", seed=2)
    magnitudes = {
        key: value.m if isinstance(value, Quantity) else value
        for key, value in values.items()
    }
    draws = np.column_stack([np.log10(magnitudes["p1"]), magnitudes["p2"]])
    np.testing.assert_allclose(
        np.cov(draws.T), posterior.covariance, rtol=0.05, atol=1e-4
    )


def test_the_priors_are_distributions(tmp_path: Path) -> None:
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y"],
        correlation=CORRELATION,
    )
    distributions = prior_distributions(problem)
    assert isinstance(distributions["p1"], Truncated)
    assert isinstance(distributions["p1"].distribution, Normal)
    model = prior_model(problem)
    assert model.covariance[0].targets == ("p1", "p2")

    problem, _ = _problem(tmp_path, [_parameter("p1")], ["y"])
    assert isinstance(prior_distributions(problem)["p1"], Uniform)


def test_a_correlation_needs_normal_or_log_normal_priors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes the priors"):
        JointPrior(
            [_parameter("p1", NORMAL_1), _parameter("p2")],
            CORRELATION,
        )
    with pytest.raises(ValueError, match="no parameters of the fit"):
        JointPrior(
            [_parameter("p1", NORMAL_1)],
            Covariance.correlation({("p1", "k"): 0.2}),
        )
    with pytest.raises(ValueError, match="correlation of their normal"):
        JointPrior(
            [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
            Covariance(["p1", "p2"], [[1.0, 0.0], [0.0, 1.0]]),
        )


def test_the_correlation_is_kept_by_a_pickled_problem(tmp_path: Path) -> None:
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y"],
        correlation=CORRELATION,
    )
    assert pickle.loads(pickle.dumps(problem)).prior_correlation == CORRELATION


def test_the_correlation_is_exported_to_petab(tmp_path: Path) -> None:
    from sbmlsim.fit.petab_v2 import gaps_of_problem, to_petab
    from sbmlsim.fit.petab_v2.reader import PetabReader

    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y"],
        correlation=CORRELATION,
    )
    assert "prior-correlation" in {gap.id for gap in gaps_of_problem(problem)}
    path = to_petab(problem, tmp_path / "petab")
    read = PetabReader.from_yaml(path).to_optimization_problem()
    assert read.prior_correlation == CORRELATION


# ----------------------------------------------------------------------------
# the posterior and the priors in the report and its methods
# ----------------------------------------------------------------------------
def _methods_html(
    problem: OptimizationProblem,
    settings: FitSettings,
    fisher: FisherInformation | None = None,
    posterior: ParameterPosterior | None = None,
) -> str:
    """Get the methods of a fit as HTML."""
    section = methods_section(
        problem=problem,
        settings=settings,
        result=None,
        identifiability=None,
        fisher=fisher,
        context=ReportContext(),
        posterior=posterior,
    )
    return HTMLRenderer().render(Document("t", (section,)))


def test_the_report_shows_the_priors_and_the_posterior(tmp_path: Path) -> None:
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y", "q"],
        correlation=CORRELATION,
    )
    pset = _pset(problem, [1.2, 2.0])
    posterior = posterior_laplace(problem, settings, pset)
    report = FitReport(
        problem=problem,
        settings=settings,
        parameter_sets=[pset],
        posterior=posterior,
        mapping_figures=False,
    )
    results_dir = report.create(tmp_path / "report", name="posterior")

    assert ParameterPosterior.read(results_dir).to_dict() == posterior.to_dict()
    table = pd.read_csv(results_dir / "posterior.tsv", sep="\t")
    assert list(table["parameter"]) == ["p1", "p2"]
    records = load_fit_records(results_dir)
    assert records.posterior is not None
    assert records.posterior.to_dict() == posterior.to_dict()
    assert "posterior.json" in records.names

    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "could not be created" not in html
    section = html[html.index('<section id="posterior">') :]
    assert "Laplace approximation" in section
    for pid in ("p1", "p2"):
        assert f"<td>{pid}</td>" in section
    # the priors with their correlation are in the overview
    assert "normal(1.3, 0.2)" in html
    assert "Prior correlation" in html
    # the methods describe the objective, the copula and the posterior
    for text in [
        "maximum a posteriori",
        "Gaussian copula",
        "Laplace approximation",
        "Tierney",
        "Nelsen",
    ]:
        assert text in html, text
    # and the results the table of the posterior
    assert "Value [95% CrI]" in html
    typst = (results_dir / "report.typ").read_text(encoding="utf-8")
    assert "<tab:posterior:observables>" in typst
    assert (results_dir / "report.pdf").exists()


def test_the_methods_of_a_likelihood(tmp_path: Path) -> None:
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2")],
        ["y", "q"],
        objective=ObjectiveType.LIKELIHOOD,
    )
    pset = _pset(problem, [1.2, 2.0])
    fisher = fisher_information(problem, settings, pset)
    posterior = posterior_laplace(problem, settings, pset, fisher=fisher)
    html = _methods_html(problem, settings, fisher, posterior)
    assert "maximum likelihood" in html
    assert "maximum a posteriori" not in html
    # the normal noise of the standard deviations of the data
    assert "a measurement with normal noise is" in html
    assert "log-normal noise" in html  # the default for geometric means
    assert "a measurement with log-normal noise" not in html
    # the weights and the loss do not enter
    assert "do not enter the objective" in html
    assert "loss function" in html
    assert "Gaussian copula" not in html
    # the Fisher information in units of the noise
    assert "4 measurements" not in html  # y and q, 8 measurements
    assert "8 measurements" in html
    assert "quantile of the standard normal" in html
    # the posterior of the uniform priors
    assert "uniform priors of the bounds" in html


def test_the_methods_of_least_squares_are_unchanged(tmp_path: Path) -> None:
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1"), _parameter("p2")],
        ["y", "q"],
        objective=ObjectiveType.LEAST_SQUARES,
    )
    pset = _pset(problem, [1.2, 2.0])
    fisher = fisher_information(problem, settings, pset)
    posterior = posterior_laplace(problem, settings, pset, fisher=fisher)
    html = _methods_html(problem, settings, fisher, posterior)
    assert "maximum likelihood" not in html
    assert "loss function" in html
    assert "covariance is the one of the Fisher information" in html
    assert "quantile of the standard normal" not in html


def test_the_methods_of_independent_priors(tmp_path: Path) -> None:
    problem, settings = _problem(
        tmp_path, [_parameter("p1", NORMAL_1), _parameter("p2")], ["y", "q"]
    )
    html = _methods_html(problem, settings)
    assert "maximum a posteriori" in html
    assert "normal(1.3, 0.2)" in html
    assert "the 1 other parameter has the uniform prior of its bounds" in html
    assert "Gaussian copula" not in html


# ----------------------------------------------------------------------------
# the correlation of the record of a fit
# ----------------------------------------------------------------------------
def test_the_correlation_of_a_record(posterior: ParameterPosterior) -> None:
    correlation = parameter_correlation(("p1", "p2"), posterior=posterior)
    assert correlation.correlated
    assert "Laplace posterior" in correlation.source
    assert correlation.matrix is not None
    np.testing.assert_allclose(correlation.matrix, posterior.correlation_matrix)
    np.testing.assert_allclose(
        correlation.rank_correlation(), posterior.rank_correlation()
    )
    # the order of the parameters is the asked one
    reversed_ = parameter_correlation(("p2", "p1"), posterior=posterior)
    assert reversed_.matrix is not None
    assert reversed_.matrix[0, 1] == pytest.approx(correlation.matrix[0, 1])
    # a posterior of other parameters falls back to independent parameters
    other = parameter_correlation(("p1", "p3"), posterior=posterior)
    assert not other.correlated
    assert np.array_equal(other.rank_correlation(), np.eye(2))
    assert parameter_correlation(("p1",), posterior=posterior).source == (
        "nothing to correlate"
    )


# ----------------------------------------------------------------------------
# the posterior as a fit of the engine (#626)
# ----------------------------------------------------------------------------
#: a prior of `p1` per kind the engine evaluates, each of which the bounds
#: `[0.1, 10]` of `_parameter` cut almost none of the mass of
PRIORS: dict[str, Prior] = {
    "normal": Prior(PriorDistribution.NORMAL, (1.3, 0.2)),
    "log-normal": Prior(PriorDistribution.LOG_NORMAL, (math.log(1.3), 0.3)),
    "uniform": Prior(PriorDistribution.UNIFORM, (0.1, 10.0)),
    "log-uniform": Prior(PriorDistribution.LOG_UNIFORM, (0.1, 10.0)),
    "laplace": Prior(PriorDistribution.LAPLACE, (1.3, 0.2)),
    "log-laplace": Prior(PriorDistribution.LOG_LAPLACE, (math.log(1.3), 0.3)),
}

#: the points the rows of a request are compared at, the values of the model
#: and a point away from them
POINTS: list[list[float]] = [[1.2, 2.0], [0.8, 3.0]]

#: how closely a prior row follows the residual of the problem where the
#: bounds cut the prior far from its median: the normalizer is read at the
#: midpoint of the bounds, where the log density of such a prior is about
#: -136, so an ulp of it is about `1e-14` of the row (measured at `3.9e-15`)
TRUNCATED_RTOL = 1e-14


def _check_the_request(
    problem: OptimizationProblem,
    x: np.ndarray,
    rtol: float = LOGARITHM_RTOL,
) -> None:
    """Check the rows and the cost of the request of a posterior against the problem.

    The rows of the data are bitwise the residuals of the problem and the rows
    of the priors follow them to `rtol`, in the places `_residual_rows` gives
    them.

    The cost is a check of its own: the offset of every prior row cancels in
    it, so the rows say nothing about `JointPrior.offset`, which
    `likelihood_constant` takes off the cost of a run to make it the negative
    log posterior of the problem.
    """
    request = fit_core.fit_request(problem)
    core = fit_core.engine_problem(problem, request, threads=1)
    rows = fit_core._residual_rows(problem)
    residuals = np.asarray(core.residuals(x), dtype=float)
    reference = np.asarray(problem.residuals(x), dtype=float)

    assert residuals.shape == rows.shape
    of_the_data = rows < fit_core._data_rows(problem)
    assert int(np.count_nonzero(~of_the_data)) == len(problem.prior)
    np.testing.assert_array_equal(residuals[of_the_data], reference[rows[of_the_data]])
    np.testing.assert_allclose(
        residuals[~of_the_data], reference[rows[~of_the_data]], rtol=rtol, atol=0.0
    )
    cost = float(core.cost(x)) + fit_core.likelihood_constant(problem)
    assert cost == pytest.approx(problem.cost_least_square(x), rel=1e-12)


def test_a_posterior_is_a_fit_of_the_engine(tmp_path: Path) -> None:
    """The prior of a parameter is one row of the request of the fit.

    The engine evaluates it without a simulation, so a posterior is a fit of
    the engine and no longer falls back to the optimizer in python; the
    `parameters` of a prior are the ones of PEtab and its bounds are the ones
    of the parameter, which the engine refuses a prior narrower than.
    """
    problem, _ = _problem(tmp_path, [_parameter("p1", NORMAL_1)], ["y"])

    assert fit_core.gaps_of(problem) == []
    request = fit_core.fit_request(problem)
    assert [prior["kind"] for prior in request["priors"]] == ["normal"]
    assert request["priors"][0]["parameter"] == 0
    assert request["priors"][0]["parameters"] == [1.3, 0.2]
    assert (request["priors"][0]["lower"], request["priors"][0]["upper"]) == (0.1, 10.0)
    assert "prior_correlation" not in request


def test_a_least_squares_fit_carries_no_prior(tmp_path: Path) -> None:
    """The priors are the residuals of a posterior and of no other objective.

    A parameter of a fit of least squares or of a likelihood carries its prior
    for the report and for the uncertainty analyses, and the fit does not
    minimize it.
    """
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1)],
        ["y"],
        objective=ObjectiveType.LIKELIHOOD,
    )

    request = fit_core.fit_request(problem)

    assert request["priors"] == []
    assert "prior_correlation" not in request


@pytest.mark.parametrize("kind", sorted(PRIORS))
def test_the_engine_computes_the_residuals_of_a_posterior_as_the_problem_does(
    tmp_path: Path, kind: str
) -> None:
    """Every row of the request of a posterior is the residual of the problem.

    The two mappings carry the errors of their data, so the scale of their
    noise is a number of their curve and the engine has no row of it: the rows
    of the data of the second mapping begin after the rows of the scales of
    the first, and the rows of the priors come after every row of the curves.

    A prior row is the other row the engine evaluates a logarithm of: python
    reads the density off the distribution of `petab`, an exponential followed
    by a logarithm, so the two agree to within an ulp and not bitwise.
    """
    require_core_fit()
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", PRIORS[kind]), _parameter("p2", NORMAL_2)],
        ["y", "q"],
    )

    assert fit_core.gaps_of(problem) == []
    np.testing.assert_array_equal(
        fit_core._residual_rows(problem), [0, 1, 2, 3, 8, 9, 10, 11, 16, 17]
    )
    for point in POINTS:
        _check_the_request(problem, problem.to_scale(np.asarray(point)))


@pytest.mark.parametrize("kind", ["normal", "log-normal"])
def test_the_engine_computes_the_residuals_of_a_correlated_posterior(
    tmp_path: Path, kind: str
) -> None:
    """The whitened rows and the logarithmic rows of the correlated priors.

    A whitened score is the product of a row of the inverse of the Cholesky
    factor with the scores, which both sides accumulate from zero with one
    addition per column in ascending order: that order is the contract of
    `_Correlated.whitened`, so the rows of a normal correlation are bitwise
    the ones of the problem. A log-normal prior has the logarithm of the value
    in its score and one row more, the density of that logarithm in the value,
    which carries the logarithm each side evaluates itself.
    """
    require_core_fit()
    priors = (
        [NORMAL_1, NORMAL_2]
        if kind == "normal"
        else [
            Prior(PriorDistribution.LOG_NORMAL, (math.log(1.2), 0.3)),
            Prior(PriorDistribution.LOG_NORMAL, (math.log(2.0), 0.4)),
        ]
    )
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", priors[0]), _parameter("p2", priors[1])],
        ["y"],
        correlation=CORRELATION,
    )

    assert fit_core.gaps_of(problem) == []
    request = fit_core.fit_request(problem)
    correlation = request["prior_correlation"]
    assert request["priors"] == []
    assert correlation["parameters"] == [0, 1]
    assert correlation["log"] == [kind == "log-normal"] * 2
    assert len(correlation["offsets"]) == (2 if kind == "log-normal" else 0)
    rtol = LOGARITHM_RTOL if kind == "log-normal" else 0.0
    for point in POINTS:
        _check_the_request(problem, problem.to_scale(np.asarray(point)), rtol=rtol)


def test_the_normalizer_of_a_truncated_prior_is_the_mass_between_the_bounds(
    tmp_path: Path,
) -> None:
    """The engine evaluates the density of the kind, the host its truncation.

    `Prior.log_density` is the density truncated at the bounds of the
    parameter, which is a distribution function and no term of a density, so
    the request carries their difference: the logarithm of the mass of the
    prior between the bounds. The median of this prior lies below the lower
    bound, so the normalizer is read at the midpoint of the bounds.
    """
    require_core_fit()
    prior = Prior(PriorDistribution.NORMAL, (0.5, 0.3))
    problem, _ = _problem(tmp_path, [_parameter("p1", prior, lb=1.0)], ["y"])
    mass = norm(0.5, 0.3).cdf(10.0) - norm(0.5, 0.3).cdf(1.0)

    request = fit_core.fit_request(problem)

    assert request["priors"][0]["normalizer"] == pytest.approx(math.log(mass), rel=1e-9)
    _check_the_request(problem, problem.to_scale(np.array([1.2])), rtol=TRUNCATED_RTOL)


def test_a_fit_of_a_posterior_in_the_engine_reports_the_log_posterior(
    tmp_path: Path,
) -> None:
    """The cost of a run of a posterior is the negative log posterior.

    `likelihood_constant` carries `JointPrior.offset`, the offsets of the
    prior rows and the normalization of the correlated ones, off the cost of a
    run, as it carries the offsets of a likelihood, so a run of the engine
    reports the cost a fit in python reports at the same point.
    """
    require_core_fit()
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1), _parameter("p2", NORMAL_2)],
        ["y", "q"],
        correlation=CORRELATION,
    )
    assert fit_core.gaps_of(problem) == []

    result = fit_core.run(problem, settings, size=2, seed=1234)

    best = min(result.fits, key=lambda fit: fit.cost)
    x = np.asarray(best.x, dtype=float)
    assert best.cost == pytest.approx(
        problem.cost_least_square(problem.to_scale(x)), rel=1e-9
    )
    expected = -(
        log_likelihood(problem, _pset(problem, list(x))) + problem.log_prior_density(x)
    )
    assert best.cost == pytest.approx(expected, rel=1e-9)


def test_the_profiles_of_a_posterior_in_the_engine_are_the_profiles_of_python(
    tmp_path: Path,
) -> None:
    """The cost of a point of the engine is on the scale of the problem.

    A run and a profile point of the engine both answer the half sum of the
    squares of the rows of their request, which `likelihood_constant` shifts
    onto the cost the problem reports, while the threshold of the analysis is
    `cost_least_square` at the optimum: a point which keeps the scale of the
    engine lies `likelihood_constant` above that threshold from the first
    step, so the crossing is interpolated between the optimum and the first
    point and every parameter comes out `identifiable` with a hairline
    interval.

    The constant is zero for a fit of least squares, which is the objective of
    the comparison of `tests/fit/test_fit_core.py`, so the profiles of an
    objective which carries one are the test of the scale.
    """
    require_core_fit()
    problem, settings = _problem(
        tmp_path, [_parameter("p1", NORMAL_1), _parameter("p2")], ["y"]
    )
    assert fit_core.likelihood_constant(problem) != 0.0
    profile_settings = ProfileSettings(max_points=8)
    # the profiles are walked around the optimum, which the fit of python
    # answers, so the comparison is the walk alone
    fitted = run_optimization(
        problem, settings, size=2, seed=3, show_progress=False, engine_fit=False
    )
    parameter_set = _pset(problem, [float(value) for value in fitted.xopt])
    walked = {
        where: profile_likelihood(
            problem,
            settings,
            parameter_set,
            profile_settings,
            show_progress=False,
            serial=True,
            engine_profiles=where == "engine",
        )
        for where in ("engine", "python")
    }

    assert walked["engine"].profiles_of == RUN_IN_THE_ENGINE
    assert walked["python"].profiles_of == RUN_IN_PYTHON
    assert walked["engine"].cost == pytest.approx(walked["python"].cost, rel=1e-9)
    for pid, profile in walked["engine"].profiles.items():
        expected = walked["python"].profiles[pid]
        np.testing.assert_allclose(profile.values, expected.values, rtol=1e-4)
        np.testing.assert_allclose(profile.costs, expected.costs, rtol=1e-4)
        assert profile.identifiability == expected.identifiability, pid
        for bound, other in (
            (profile.ci_lower, expected.ci_lower),
            (profile.ci_upper, expected.ci_upper),
        ):
            assert (bound is None) == (other is None), pid
            if bound is not None and other is not None:
                assert bound == pytest.approx(other, rel=1e-3), pid
        # the point next to the optimum is one step of the walk above it and
        # not the constant of the objective, which the walk never steps over
        costs = np.asarray(profile.costs, dtype=float)
        k = profile.index_optimum
        assert costs[k + 1] < walked["engine"].threshold, pid
        assert costs[k - 1] < walked["engine"].threshold, pid


def test_the_fisher_information_of_a_posterior_cuts_the_prior_rows(
    tmp_path: Path,
) -> None:
    """The Fisher information of a posterior is the information of its data.

    The jacobian of the engine has a row per prior after the rows of its
    curves, which `residual_jacobian` drops: a `FisherInformation` is the
    information of the data, whose covariance is a confidence interval, where
    the precision of the posterior adds the curvature of the priors and is
    `posterior_laplace`.
    """
    require_core_fit()
    problem, settings = _problem(tmp_path, [_parameter("p1", NORMAL_1)], ["y"])
    assert fit_core.gaps_of_a_jacobian(problem) == []

    information = fisher_information(problem, settings, _pset(problem, [1.2]))

    assert information.jacobian_of == SENSITIVITIES
    rows = fit_core.residual_jacobian(
        problem, np.asarray(problem.to_scale(np.array([1.2]))), threads=1
    )
    assert rows.shape == (fit_core._data_rows(problem), 1)
    # the four measurements of `y = p1 + p2` of the standard deviation `SD`,
    # the prior of `p1` no part of the information, which would add `1/0.2**2`
    assert information.matrix[0, 0] == pytest.approx(len(TIMES) / SD**2, rel=1e-6)


def test_an_exotic_prior_kind_is_a_gap_of_the_engine(tmp_path: Path) -> None:
    """A prior whose log density is no expression of the contract stays in python.

    The engine evaluates the six kinds of `fit_core.ENGINE_PRIORS`; the
    density of another kind of PEtab, e.g. a gamma prior, is no term of a
    request, so the optimizer in python evaluates it.
    """
    problem, _ = _problem(
        tmp_path, [_parameter("p1", Prior(PriorDistribution.GAMMA, (2.0, 1.0)))], ["y"]
    )

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"objective"}
    assert "gamma" in gaps[0].message
    assert "p1" in gaps[0].message


@pytest.mark.parametrize(
    ("prior", "lb"),
    [
        (Prior(PriorDistribution.UNIFORM, (1.0, 2.0)), 0.1),
        (Prior(PriorDistribution.LOG_UNIFORM, (1.0, 2.0)), 0.1),
        (Prior(PriorDistribution.LOG_NORMAL, (0.0, 0.5)), -1.0),
        (Prior(PriorDistribution.LOG_LAPLACE, (0.0, 0.5)), 0.0),
    ],
)
def test_a_prior_whose_support_does_not_cover_the_bounds_is_a_gap(
    tmp_path: Path, prior: Prior, lb: float
) -> None:
    """A zero density between the bounds is a penalty in python and `NaN` in the engine.

    `_prior_residuals` of the problem replaces the `NaN` of a row whose
    density is zero with `LIKELIHOOD_FAILURE_RESIDUAL`, a penalty the
    optimizer of python walks out of, and the row of such a value in a request
    is `NaN`, which gives every run of the engine a cost of `NaN` and the
    values it started from: a uniform prior narrower than the bounds of its
    parameter and a log kind of a parameter whose bounds reach zero are fitted
    in python.
    """
    problem, _ = _problem(tmp_path, [_parameter("p1", prior, lb=lb)], ["y"])

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"objective"}
    assert "p1" in gaps[0].message
    assert "zero" in gaps[0].message


def test_the_optimizer_of_python_walks_out_of_a_zero_prior_density(
    tmp_path: Path,
) -> None:
    """The fit which the gap of the support keeps in python, and what it answers.

    Every run of the fit of the engine has the cost `NaN`, because every one
    of its starts lies outside the support of the prior, so the gap is not a
    tolerance: the gap routes the fit to python, which converges to the
    optimum from the same seed.
    """
    require_core_fit()
    problem, settings = _problem(
        tmp_path,
        [_parameter("p1", Prior(PriorDistribution.UNIFORM, (1.0, 2.0)))],
        ["y"],
    )

    result = run_optimization(problem, settings, size=3, seed=7, show_progress=False)

    assert result.optimizer_of == RUN_IN_PYTHON
    assert min(fit.cost for fit in result.fits) == pytest.approx(-0.5345862, rel=1e-5)
    assert result.xopt[0] == pytest.approx(1.2, rel=1e-6)


def test_a_prior_of_a_parameter_whose_unit_factor_is_not_one_is_a_gap(
    tmp_path: Path,
) -> None:
    """A prior is read at the value of the parameter and not at the one of its target.

    The engine holds the value of the entity of the model, which the factor of
    the unit of a parameter carries its value into, while the parameters and
    the bounds of a prior are in the unit of the parameter: such a prior would
    be read at another value, so the fit stays in python.
    """
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", NORMAL_1, unit="1/minute")],
        ["y"],
        model=UNIT_MODEL,
    )
    assert fit_core._factors_of(problem) == [pytest.approx(1.0 / 60.0)]

    gaps = fit_core.gaps_of(problem)

    assert {gap.code for gap in gaps} == {"objective"}
    assert "unit" in gaps[0].message
    assert "p1" in gaps[0].message


def _prior_slopes(problem: OptimizationProblem, x: np.ndarray) -> np.ndarray:
    """Get the central differences of the prior residuals in the space of the search.

    The step is relative and small enough that the differences are the
    derivative of a smooth row to about `1e-10`, so they stand in for its
    closed form.
    """
    x = np.asarray(x, dtype=float)
    columns = []
    for k in range(x.size):
        step = 1e-6 * max(abs(float(x[k])), 1.0)
        plus, minus = x.copy(), x.copy()
        plus[k] += step
        minus[k] -= step
        up = problem.prior.residuals(problem.from_scale(plus))[0]
        down = problem.prior.residuals(problem.from_scale(minus))[0]
        columns.append((up - down) / (2.0 * step))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("kind", sorted(PRIORS))
def test_the_prior_rows_of_the_jacobian_of_the_engine_are_the_slope_of_the_row(
    tmp_path: Path, kind: str
) -> None:
    """The jacobian of the engine carries the slope of every prior row.

    The rows of the priors lie past the residuals of the data, which is what
    lets `residual_jacobian` drop them, and the engine answers the analytic
    slope of a prior row where its forward sensitivities answer the data: it
    is the derivative of the row python computes, which the central
    differences of `JointPrior.residuals` pin to `1e-10`.

    The difference quotient of the engine is the same jacobian over a fixed
    relative step of `1e-3`, which follows that slope only as closely as the
    curvature of the row allows: `1.4e-3` at most, and `6.6e-3` for a
    log-normal kind, whose row is the steepest of them.
    """
    require_core_fit()
    rtol = 1e-2 if kind == "log-normal" else 2e-3
    problem, _ = _problem(
        tmp_path,
        [_parameter("p1", PRIORS[kind]), _parameter("p2")],
        ["y"],
    )
    x = np.asarray(problem.to_scale(np.array([1.1, 2.3])), dtype=float)
    of_the_priors = fit_core._residual_rows(problem) >= fit_core._data_rows(problem)

    jacobians = {
        where: np.asarray(
            fit_core.engine_problem(
                problem, fit_core.fit_request(problem, jacobian=where), threads=1
            ).jacobian(list(x)),
            dtype=float,
        )[of_the_priors]
        for where in (fit_core.SENSITIVITIES, fit_core.DIFFERENCE_QUOTIENT)
    }

    expected = _prior_slopes(problem, x)
    assert expected.shape == jacobians[fit_core.SENSITIVITIES].shape
    np.testing.assert_allclose(
        jacobians[fit_core.SENSITIVITIES], expected, rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(
        jacobians[fit_core.DIFFERENCE_QUOTIENT],
        jacobians[fit_core.SENSITIVITIES],
        rtol=rtol,
        atol=1e-12,
    )
