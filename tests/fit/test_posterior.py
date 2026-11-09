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
from scipy.stats import multivariate_normal

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitSettings, ParameterSet
from sbmlsim.fit.cli import load_fit_records
from sbmlsim.fit.fisher import FisherInformation, fisher_information
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
from sbmlsim.fit.options import ObjectiveType, ParameterScaleType
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.posterior import ParameterPosterior, posterior_laplace
from sbmlsim.fit.priors import JointPrior, prior_distributions, prior_model
from sbmlsim.fit.records import parameter_correlation
from sbmlsim.fit.report import FitReport
from sbmlsim.model import AbstractModel
from sbmlsim.report.document import Document
from sbmlsim.report.render import HTMLRenderer
from sbmlsim.simulation import Simulation
from sbmlsim.simulation.sampling import Truncated, Uniform, posterior_parameters
from sbmlsim.simulation.sampling.distributions import LogNormal, Normal
from sbmlsim.task import Task
from sbmlsim.uncertainty import Covariance
from sbmlsim.units import Quantity
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


def _parameter(pid: str, prior: Prior | None = None, lb: float = 0.1) -> FitParameter:
    return FitParameter(
        pid=pid,
        lower_bound=lb,
        upper_bound=10.0,
        start_value=1.0,
        unit="dimensionless",
        prior=prior,
    )


def _problem(
    tmp_path: Path,
    parameters: list[FitParameter],
    mappings: list[str],
    correlation: Covariance | None = None,
    objective: ObjectiveType = ObjectiveType.POSTERIOR,
    scale: ParameterScaleType = ParameterScaleType.LINEAR,
) -> tuple[OptimizationProblem, FitSettings]:
    MODEL_PATH["path"] = tmp_path / "observables.xml"
    MODEL_PATH["path"].write_text(sbml(MODEL), encoding="utf-8")
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
