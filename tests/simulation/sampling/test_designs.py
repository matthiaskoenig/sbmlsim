"""The designs local, random and lhs."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import (
    LogNormal,
    Normal,
    Uniform,
    lhs,
    local,
    random,
)
from sbmlsim.simulation.sampling.designs import _iman_conover
from sbmlsim.simulation.scan import Design
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml, sbml_minutes


def record(dimension: Dimension) -> Design:
    design = dimension.design
    assert design is not None
    return design


@pytest.fixture(scope="module")
def model() -> RoadrunnerSBMLModel:
    return Simulator().load(sbml())


def test_local_varies_every_target_around_its_reference(
    model: RoadrunnerSBMLModel,
) -> None:
    dimension = local(["k1", "k2"], delta=0.1, model=model)
    assert dimension.labels.tolist() == ["reference", "k1+", "k1-", "k2+", "k2-"]
    np.testing.assert_allclose(dimension.values["k1"], [0.8, 0.88, 0.72, 0.8, 0.8])
    np.testing.assert_allclose(dimension.values["k2"], [0.6, 0.6, 0.6, 0.66, 0.54])
    assert record(dimension).method == "local"
    assert record(dimension).options == {"delta": 0.1, "targets": ["k1", "k2"]}
    assert record(dimension).references["k1"] == {"value": 0.8, "unit": ""}


def test_local_needs_a_delta_below_one(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="delta"):
        local(["k1"], delta=1.0, model=model)


def test_random_draws_from_every_marginal() -> None:
    dimension = random({"a": Uniform(0.0, 1.0), "b": Normal(5.0, 1.0)}, 2000, seed=3)
    assert len(dimension) == 2000 and dimension.labels.tolist()[:3] == [0, 1, 2]
    assert stats.kstest(dimension.values["a"], "uniform").pvalue > 0.01
    assert stats.kstest(dimension.values["b"], stats.norm(5.0, 1.0).cdf).pvalue > 0.01
    assert (
        record(dimension).options["seed"] == 3
        and record(dimension).options["n"] == 2000
    )


def test_random_and_lhs_use_the_draws_of_numpy_and_scipy() -> None:
    u = np.random.default_rng(5).random((4, 2))
    dimension = random({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 4, seed=5)
    np.testing.assert_array_equal(
        np.column_stack([dimension.values["a"], dimension.values["b"]]), u
    )
    from scipy.stats import qmc

    v = qmc.LatinHypercube(d=2, rng=np.random.default_rng(5)).random(n=4)
    dimension = lhs({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 4, seed=5)
    np.testing.assert_array_equal(
        np.column_stack([dimension.values["a"], dimension.values["b"]]), v
    )


def test_lhs_has_one_point_per_stratum() -> None:
    dimension = lhs({"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}, 50, seed=1)
    for target in ("a", "b"):
        strata = np.floor(np.asarray(dimension.values[target]) * 50).astype(int)
        assert sorted(strata.tolist()) == list(range(50))


def test_a_weak_rank_correlation_is_reached() -> None:
    for design in (random, lhs):
        dimension = design(
            {"a": LogNormal(1.0, 0.3), "b": Uniform(0.0, 1.0)},
            2000,
            seed=2,
            correlation=[[1.0, 0.3], [0.3, 1.0]],
        )
        rho = stats.spearmanr(dimension.values["a"], dimension.values["b"]).statistic
        assert rho == pytest.approx(0.3, abs=0.02)


def test_a_correlation_is_reached() -> None:
    correlation = [[1.0, 0.7], [0.7, 1.0]]
    for design in (random, lhs):
        dimension = design(
            {"a": LogNormal(1.0, 0.3), "b": Uniform(0.0, 1.0)},
            2000,
            seed=2,
            correlation=correlation,
        )
        rho = stats.spearmanr(dimension.values["a"], dimension.values["b"]).statistic
        assert rho == pytest.approx(0.7, abs=0.02)
    correlated = lhs(
        {"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)},
        40,
        seed=2,
        correlation=correlation,
    )
    strata = np.floor(np.asarray(correlated.values["a"]) * 40).astype(int)
    assert sorted(strata.tolist()) == list(range(40))


@pytest.mark.parametrize(
    "correlation",
    [
        [[1.0, 0.5], [0.4, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[2.0, 0.0], [0.0, 1.0]],
        [[1.0]],
    ],
)
def test_an_invalid_correlation_raises(correlation: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="correlation"):
        random(
            {"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)},
            10,
            seed=1,
            correlation=correlation,
        )


def test_relative_distributions_need_the_model(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="model="):
        random({"k1": LogNormal(cv=0.1)}, 5, seed=1)
    dimension = random({"k1": LogNormal(cv=0.1)}, 2000, seed=1, model=model)
    assert np.median(dimension.values["k1"]) == pytest.approx(0.8, rel=0.02)
    assert record(dimension).references["k1"] == {"value": 0.8, "unit": ""}


def test_the_seed_of_the_record_reproduces_the_design() -> None:
    first = random({"a": Normal(0.0, 1.0)}, 5)
    seed = record(first).options["seed"]
    assert isinstance(seed, int)
    again = random({"a": Normal(0.0, 1.0)}, 5, seed=seed)
    np.testing.assert_array_equal(first.values["a"], again.values["a"])


def test_a_design_runs_in_a_scan() -> None:
    model = Simulator().load(sbml_minutes())
    design = random({"f": Uniform(Q(1000.0, "ug"), Q(3000.0, "ug"))}, 6, seed=4)
    doses = Dimension("k", values={"k1": [0.5, 1.0]})
    res = Simulator().run(
        model, Scan(Simulation(end=1, steps=2), [design, doses]), [Formula("f_mg", "f")]
    )
    # the model sees f in mg: the values of the design are converted
    np.testing.assert_allclose(
        res["f_mg"].isel(k=0, time=0).values,
        np.asarray(design.values["f"].to("mg").magnitude),
    )
    assert res.ds.attrs["scan"]["dimensions"][0]["design"]["method"] == "random"


def test_a_correlated_lhs_with_few_points_never_raises_a_linalg_error() -> None:
    distributions = {"a": Uniform(0.0, 1.0), "b": Uniform(0.0, 1.0)}
    correlation = [[1.0, 0.5], [0.5, 1.0]]
    dimension = lhs(distributions, 4, seed=7, correlation=correlation)
    assert len(dimension) == 4
    for seed in range(50):
        lhs(distributions, 4, seed=seed, correlation=correlation)


def test_a_collinear_hypercube_raises_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def singular(_: object) -> object:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "cholesky", singular)
    with pytest.raises(ValueError, match="too few"):
        _iman_conover(
            np.random.default_rng(0).random((4, 2)),
            np.eye(2),
            np.random.default_rng(1),
        )


def test_a_negative_seed_raises_a_value_error() -> None:
    with pytest.raises(ValueError, match="seed"):
        random({"a": Uniform(0.0, 1.0)}, 5, seed=-1)


def test_the_cheap_arguments_are_checked_before_the_model_is_read() -> None:
    with pytest.raises(ValueError, match="seed"):
        random({"k1": LogNormal(cv=0.1)}, 5, seed=-1)
    with pytest.raises(ValueError, match="'n'"):
        lhs({"k1": LogNormal(cv=0.1)}, 0, seed=1)
