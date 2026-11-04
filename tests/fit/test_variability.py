"""The variability between the subjects of a fit problem (`sbmlsim.fit.variability`).

The decay `A' = -k kx A` from `A(0) = a0`, whose experiment sets `a0 = 2` for
its study group and whose fit fits `k`: a factor of `a0` scales the
prediction, `A = 2 f exp(-k t)`, so the predictions of the subjects are known
exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.objects import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.variability import (
    Variability,
    predictive_check,
    quantile_column,
    subject_predictions,
)
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulation.sampling.distributions import LogNormal, Normal, Uniform
from sbmlsim.task import Task
from sbmlsim.uncertainty import Covariance
from tests.simulator.models import sbml

DECAY = """
model decay
  compartment C = 1;
  species A in C;
  a0 = 1; k = 0.5; kx = 1; kz = 1
  A = a0
  J: A -> ; k*kx*kz*A
end
"""

TIMES = np.array([0.5, 1.0, 2.0, 4.0])

#: the file of the model, written by the fixture
MODEL_PATH: dict[str, Path] = {}


class DecayExperiment(SimulationExperiment):
    """The decay of a study group with `a0 = 2`, the data of six subjects."""

    #: the count of the data points, `None` for individuals
    count: int | None = 6

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=MODEL_PATH["path"])}

    def datasets(self) -> dict[str, DataSet]:
        data = {
            "time": TIMES,
            "time_unit": "s",
            "A": 2.0 * np.exp(-0.5 * TIMES),
            "A_unit": "dimensionless",
        }
        if self.count is not None:
            data["count"] = self.count
        return {"d": DataSet.from_df(pd.DataFrame(data), ureg=self.ureg)}

    def simulations(self) -> dict[str, Simulation]:
        return {
            "s": Simulation(
                end=5,
                preinit_changes={"a0": 2.0},
                changes=[Change(3, {"kz": "kz*1"})],
            )
        }

    def tasks(self) -> dict[str, Task]:
        return {"t": Task(model="m", simulation="s")}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset="d",
                    xid="time",
                    yid="A",
                    count="count" if self.count is not None else None,
                ),
                observable=FitData(self, task="t", xid="time", yid="[A]"),
            )
        }


class IndividualExperiment(DecayExperiment):
    """The decay of one subject, data without a count."""

    count = None


class GeometricExperiment(DecayExperiment):
    """The decay of a study group, geometric means with their GSD."""

    def datasets(self) -> dict[str, DataSet]:
        data = {
            "time": TIMES,
            "time_unit": "s",
            "A": 2.0 * np.exp(-0.5 * TIMES),
            "A_unit": "dimensionless",
            "A_gsd": 1.3,
            "count": 6,
        }
        return {"d": DataSet.from_df(pd.DataFrame(data), ureg=self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset="d",
                    xid="time",
                    yid="A",
                    yid_gsd="A_gsd",
                    count="count",
                ),
                observable=FitData(self, task="t", xid="time", yid="[A]"),
            )
        }


def _problem(tmp_path: Path, experiment: type[DecayExperiment]) -> OptimizationProblem:
    """Get the initialized problem which fits `k`."""
    MODEL_PATH["path"] = tmp_path / "decay.xml"
    MODEL_PATH["path"].write_text(sbml(DECAY))
    problem = OptimizationProblem(
        opid="decay",
        mapping_collections=[
            FitMappingCollection(experiment=experiment, mappings=["fm"])
        ],
        fit_parameters=[
            FitParameter(
                pid="k",
                lower_bound=0.01,
                upper_bound=2.0,
                start_value=0.5,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    return problem


@pytest.fixture
def problem(tmp_path: Path) -> OptimizationProblem:
    """Get the problem of the study group."""
    return _problem(tmp_path, DecayExperiment)


X = np.array([0.5])


def test_the_plan_scales_the_values_of_a_subject(problem: OptimizationProblem) -> None:
    plan = problem.evaluated_plan(0, X, factors={"a0": 1.5, "k": 2.0, "kx": 3.0})
    values = {a.target: a.value for a in plan.preinit}
    # the value of the study group, of the fitted parameter and of the model
    assert values["a0"] == pytest.approx(3.0)
    assert values["k"] == pytest.approx(1.0)
    assert values["kx"] == pytest.approx(3.0)
    assert "kx" in plan.assigned
    assert problem.evaluated_plan(0, X, factors={}) == problem.evaluated_plan(0, X)
    # a target set by a formula has no value to scale
    with pytest.raises(ValueError, match="'kz' by the formula"):
        problem.evaluated_plan(0, X, factors={"kz": 2.0})
    with pytest.raises(ValueError, match="'unknown' has no target"):
        problem.evaluated_plan(0, X, factors={"unknown": 2.0})


def test_the_predictions_of_a_subject(problem: OptimizationProblem) -> None:
    (typical,) = problem.predictions(X, [0]).values()
    np.testing.assert_allclose(typical, 2.0 * np.exp(-0.5 * TIMES), rtol=1e-5)
    (scaled,) = problem.predictions(X, [0], factors={"a0": 1.7}).values()
    np.testing.assert_allclose(scaled, 1.7 * typical, rtol=1e-6)
    (faster,) = problem.predictions(X, [0], factors={"kx": 2.0}).values()
    np.testing.assert_allclose(faster, 2.0 * np.exp(-1.0 * TIMES), rtol=1e-5)


def test_a_variability_is_relative() -> None:
    with pytest.raises(ValueError, match="distributions with a location"):
        Variability({"a0": LogNormal(median=1.0, cv=0.2)})
    with pytest.raises(ValueError, match="no variables"):
        Variability(
            {"a0": LogNormal(cv=0.2)},
            covariance=[Covariance.correlation({("a0", "kx"): 0.5})],
        )


def test_the_factors_of_the_subjects() -> None:
    variability = Variability(
        {"a0": LogNormal(cv=0.3), "kx": LogNormal(cv=0.2), "kz": Uniform(relative=0.5)},
        covariance=[
            Covariance(
                ["a0", "kx"],
                matrix=[
                    [np.log1p(0.09), 0.5 * np.sqrt(np.log1p(0.09) * np.log1p(0.04))],
                    [0.5 * np.sqrt(np.log1p(0.09) * np.log1p(0.04)), np.log1p(0.04)],
                ],
                scale="log",
            )
        ],
    )
    assert variability.targets == ("a0", "kx", "kz")
    factors = variability.draw(4096, seed=1)
    assert factors.shape == (4096, 3)
    logs = np.log(factors[:, :2])
    assert np.median(factors[:, 0]) == pytest.approx(1.0, abs=0.01)
    assert np.std(logs[:, 0]) == pytest.approx(np.sqrt(np.log1p(0.09)), rel=0.02)
    assert np.corrcoef(logs.T)[0, 1] == pytest.approx(0.5, abs=0.02)
    assert factors[:, 2].min() >= 0.5 and factors[:, 2].max() <= 1.5
    # the subjects of a sequence continue
    np.testing.assert_allclose(
        variability.draw(8, seed=1, start=8), variability.draw(16, seed=1)[8:]
    )
    assert Variability({}).draw(3, seed=1).shape == (3, 0)


def test_the_subject_predictions(problem: OptimizationProblem) -> None:
    variability = Variability({"a0": LogNormal(cv=0.2)})
    subjects = subject_predictions(problem, X, variability, 16, seed=3)
    assert subjects.failed == 0
    assert subjects.targets == ("a0",)
    typical = 2.0 * np.exp(-0.5 * TIMES)
    np.testing.assert_allclose(
        subjects.predictions[0], subjects.factors[:, :1] * typical, rtol=1e-5
    )
    with pytest.raises(ValueError, match="must be positive"):
        subject_predictions(problem, X, variability, 0, seed=3)
    with pytest.raises(ValueError, match="'kz' by the formula"):
        subject_predictions(problem, X, Variability({"kz": Normal(cv=0.1)}), 2, seed=3)


def test_the_predictive_check_of_groups(problem: OptimizationProblem) -> None:
    variability = Variability({"a0": LogNormal(cv=0.3)})
    check = predictive_check(problem, X, variability, 256, seed=5, groups=2000)
    points = check.points
    assert list(points["N"]) == [6] * 4
    assert set(points["statistic"]) == {"mean"}
    assert check.coverage == 1.0
    assert (check.n_subjects, check.n_vectors, check.failed) == (256, 1, 0)
    typical = 2.0 * np.exp(-0.5 * TIMES)
    np.testing.assert_allclose(points["PRED"], typical, rtol=1e-5)
    # the mean of six subjects of a CV of 30 %: about 30 %/sqrt(6) around the
    # mean of the log-normal, a factor of sqrt(1 + 0.09) above the median
    mean = np.sqrt(1.09)
    ratio = points[quantile_column(0.5)] / typical
    np.testing.assert_allclose(ratio, mean, rtol=0.03)
    width = (points["q95"] - points["q5"]) / typical
    np.testing.assert_allclose(width, 2 * 1.645 * 0.3 * mean / np.sqrt(6), rtol=0.15)
    by_mapping = check.coverage_by_mapping()
    assert by_mapping.to_dict("records") == [
        {"experiment": "DecayExperiment", "mapping": "fm", "n": 4, "coverage": 1.0}
    ]
    # a data point far from the band is outside
    problem.y_references[0] = problem.y_references[0] * 3.0
    outside = predictive_check(problem, X, variability, 64, seed=5)
    assert outside.coverage == 0.0
    # the uncertainty of the parameters widens the band
    nested = predictive_check(
        problem, np.array([[0.4], [0.5], [0.6]]), variability, 64, seed=5
    )
    assert nested.n_vectors == 3
    assert (nested.points["q95"] - nested.points["q5"]).iloc[-1] > (
        outside.points["q95"] - outside.points["q5"]
    ).iloc[-1]


def test_the_predictive_check_of_individuals(tmp_path: Path) -> None:
    problem = _problem(tmp_path, IndividualExperiment)
    variability = Variability({"a0": LogNormal(cv=0.3)})
    check = predictive_check(
        problem, X, variability, 512, seed=5, quantiles=(0.95, 0.05)
    )
    points = check.points
    assert check.quantiles == (0.05, 0.95)
    assert list(points.columns[-3:]) == ["q5", "q95", "inside"]
    assert set(points["statistic"]) == {"individual"}
    assert list(points["n"]) == [512] * 4
    # the quantiles of the log-normal factor
    typical = 2.0 * np.exp(-0.5 * TIMES)
    sigma = np.sqrt(np.log1p(0.09))
    np.testing.assert_allclose(
        points["q95"] / typical, np.exp(1.645 * sigma), rtol=0.02
    )
    with pytest.raises(ValueError, match="at least two"):
        predictive_check(problem, X, variability, 4, seed=5, quantiles=(0.5,))
    with pytest.raises(ValueError, match="at least one parameter vector"):
        predictive_check(problem, np.empty((0, 1)), variability, 4, seed=5)
    with pytest.raises(ValueError, match="virtual groups must be positive"):
        predictive_check(problem, X, variability, 4, seed=5, groups=0)


def test_the_predictive_check_of_geometric_means(tmp_path: Path) -> None:
    problem = _problem(tmp_path, GeometricExperiment)
    variability = Variability({"a0": LogNormal(cv=0.3)})
    check = predictive_check(problem, X, variability, 256, seed=5, groups=2000)
    points = check.points
    assert set(points["statistic"]) == {"gmean"}
    # the geometric mean of log-normal factors has the median 1
    typical = 2.0 * np.exp(-0.5 * TIMES)
    np.testing.assert_allclose(points["q50"] / typical, 1.0, atol=0.03)
    assert check.coverage == 1.0


def test_the_variability_between_the_studies(problem: OptimizationProblem) -> None:
    between = Variability({"a0": LogNormal(cv=0.3)})
    variability = Variability({"a0": LogNormal(cv=0.01)}, between=between)
    assert variability.all_targets == ("a0",)
    with pytest.raises(ValueError, match="no variability between studies of its own"):
        Variability({}, between=variability)
    check = predictive_check(problem, X, variability, 8, seed=5, studies=256, groups=20)
    assert (check.n_studies, len(check.subjects)) == (256, 256)
    # the subjects of a study share its factor
    first = check.subjects[0]
    assert set(first.study) == {"a0"}
    np.testing.assert_allclose(first.factors[:, 0], first.study["a0"], rtol=0.05)
    # the mean of six subjects keeps the factor of its study
    typical = 2.0 * np.exp(-0.5 * TIMES)
    sigma = np.sqrt(np.log1p(0.09))
    np.testing.assert_allclose(
        check.points["q95"] / typical, np.exp(1.645 * sigma), rtol=0.05
    )
    with pytest.raises(ValueError, match="studies must be positive"):
        predictive_check(problem, X, variability, 4, seed=5, studies=0)
    # without a variability between the studies there is one study
    alone = predictive_check(
        problem, X, Variability({"a0": LogNormal(cv=0.1)}), 4, seed=5, studies=8
    )
    assert alone.n_studies == 1
