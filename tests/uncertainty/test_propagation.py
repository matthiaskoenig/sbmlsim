"""The propagation of uncertain values through simulations."""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Custom, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import Uncertain, UncertaintyModel, propagate
from tests.simulator.models import sbml_pk

OBSERVABLES = [PK("pk", "[C]", dose="PODOSE", route="oral")]


def _simulation(dose: object, volume: float = 10.0) -> Simulation:
    return Simulation(
        end=48,
        steps=480,
        preinit_changes={"V": volume},
        changes=[Change(0, {"PODOSE": dose})],  # ty: ignore[invalid-argument-type]
    )


@pytest.fixture(scope="module")
def nominal_cmax() -> float:
    result = Simulator(n_workers=1).run(
        sbml_pk(), _simulation(Q(100, "mg")), OBSERVABLES
    )
    return float(result["pk.cmax"].values)


def test_the_propagation_of_an_uncertain_dose(nominal_cmax: float) -> None:
    """The cmax is linear in the dose: its mean and CV are the ones of the dose."""
    dose = Uncertain(Q(100, "mg"), cv=0.1)
    res = propagate(
        sbml_pk(),
        _simulation(dose),
        UncertaintyModel(),
        OBSERVABLES,
        outputs=["pk.cmax"],
        n=1024,
        seed=1,
        simulator=Simulator(n_workers=1),
    )
    summary = res.summary(["pk.cmax"]).set_index("output").loc["pk.cmax"]
    assert summary["n"] == 1024
    assert summary["n_failed"] == 0
    assert abs(summary["mean"] - nominal_cmax) <= 3 * summary["mean_se"] + 1e-9
    assert summary["sd"] / summary["mean"] == pytest.approx(0.1, rel=0.05)
    assert summary["q0.025_low"] <= summary["q0.025"] <= summary["q0.025_high"]
    assert res.record["seed"] == 1
    assert res.record["variables"][0]["key"] == "PODOSE"


def test_the_adaptive_propagation_stops() -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.2)),
        UncertaintyModel(parameters={"ke": Uncertain(0.2, cv=0.2)}),
        OBSERVABLES,
        outputs=["pk.cmax", "pk.auc_last"],
        rel_tol=0.05,
        n_start=128,
        n_max=1 << 16,
        seed=2,
        simulator=Simulator(n_workers=1),
    )
    history = res.record["history"]
    assert [entry["n"] for entry in history] == [128 << k for k in range(len(history))]
    assert len(history) > 1 and res.record["n"] == history[-1]["n"] < 1 << 16
    assert [entry["converged"] for entry in history] == [False] * (len(history) - 1) + [
        True
    ]
    assert history[-1]["precision"] <= 0.05 < history[-2]["precision"]
    assert res.values("pk.cmax").sizes["draw"] == res.record["n"]


def test_the_adaptive_propagation_stops_at_n_max() -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.2)),
        UncertaintyModel(),
        OBSERVABLES,
        outputs=["pk.cmax"],
        rel_tol=1e-4,
        n_start=256,
        n_max=1000,
        seed=2,
        simulator=Simulator(n_workers=1),
    )
    assert [entry["n"] for entry in res.record["history"]] == [256, 512, 1000]
    assert res.record["n"] == res.values("pk.cmax").sizes["draw"] == 1000
    np.testing.assert_array_equal(res.values("pk.cmax").draw, np.arange(1000))


def test_scenarios_have_common_random_numbers() -> None:
    """The draws of both scenarios are the same, a ratio of a linear model is exact."""
    dose = Uncertain(Q(100, "mg"), cv=0.2)
    res = propagate(
        sbml_pk(),
        {"small": _simulation(dose, 10.0), "large": _simulation(dose, 20.0)},
        UncertaintyModel(),
        OBSERVABLES,
        outputs=["pk.cmax"],
        n=256,
        seed=3,
        simulator=Simulator(n_workers=1),
    )
    assert res.scenarios == ("small", "large")
    ratio = res.contrast("pk.cmax", "large", "small")
    np.testing.assert_allclose(ratio.values, 0.5, rtol=1e-5)
    difference = res.contrast("pk.cmax", "large", "small", kind="difference")
    assert np.all(difference.values < 0)


def fails_for_a_large_dose(time: np.ndarray, values: dict) -> float:
    """Fail for a large concentration, i.e. a large dose."""
    if np.max(values["[C]"]) > 7.0:
        raise ValueError("a large dose")
    return 1.0


def infinite(time: np.ndarray, values: dict) -> float:
    return np.inf


def always_fails(time: np.ndarray, values: dict) -> float:
    raise ValueError("always")


FAILING = [
    *OBSERVABLES,
    Custom("check", fails_for_a_large_dose, "dimensionless", symbols=["[C]"]),
]


def test_failed_draws_are_counted(nominal_cmax: float) -> None:
    assert 6.5 < nominal_cmax < 7.0
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(),
        FAILING,
        outputs=["pk.cmax"],
        n=64,
        seed=4,
        simulator=Simulator(n_workers=1),
    )
    failed = int(res.results["default"].ds["status"].sum())
    assert 0 < failed < 64
    row = res.summary(["pk.cmax"]).iloc[0]
    assert (row["n"], row["n_failed"]) == (64 - failed, failed)
    statement = res.probability("pk.cmax", greater=0.0)
    assert (statement.n, statement.n_failed) == (64 - failed, failed)


def test_the_batches_keep_their_errors_and_draws() -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(),
        FAILING,
        outputs=["pk.cmax"],
        rel_tol=1e-4,
        n_start=32,
        n_max=128,
        seed=4,
        simulator=Simulator(n_workers=1),
    )
    ds = res.results["default"].ds
    (dimension,) = ds.attrs["scan"]["dimensions"]
    assert dimension["labels"] == list(range(128))
    assert len(dimension["values"]["PODOSE"]["value"]) == 128
    assert dimension["design"]["options"] | {"n": 0} == {
        "seed": 4,
        "design": "sobol",
        "start": 0,
        "n": 0,
    }
    assert dimension["design"]["options"]["n"] == 128
    # the errors of every batch, at most MAX_ERRORS of them
    assert len(ds.attrs["errors"]) == min(10, int(ds["status"].sum()))


def test_an_output_which_always_fails_raises() -> None:
    with pytest.raises(ValueError, match="no draw"):
        propagate(
            sbml_pk(),
            _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
            UncertaintyModel(),
            [Custom("check", always_fails, "dimensionless", symbols=["[C]"])],
            outputs=["check"],
            n_start=16,
            seed=4,
            simulator=Simulator(n_workers=1),
        )


def test_an_infinite_output_has_no_adaptive_moments() -> None:
    with pytest.raises(ValueError, match="infinite"):
        propagate(
            sbml_pk(),
            _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
            UncertaintyModel(),
            [Custom("inf", infinite, "dimensionless", symbols=["[C]"])],
            outputs=["inf"],
            n_start=16,
            seed=4,
            simulator=Simulator(n_workers=1),
        )


def test_the_values_are_a_copy() -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(),
        OBSERVABLES,
        n=16,
        seed=4,
        simulator=Simulator(n_workers=1),
    )
    res.values("pk.cmax").attrs["units"] = "changed"
    assert "units" not in res.results["default"].ds["pk.cmax"].attrs
    assert res.values("pk.cmax").attrs["units"] != "changed"


def test_scenarios_with_their_own_marginals_have_common_random_numbers() -> None:
    """A dose of each scenario at the same quantile: the ratio of the cmax is exact."""
    res = propagate(
        sbml_pk(),
        {
            "low": _simulation(Uncertain(Q(100, "mg"), cv=0.2)),
            "high": _simulation(Uncertain(Q(200, "mg"), cv=0.2)),
        },
        UncertaintyModel(),
        OBSERVABLES,
        n=64,
        seed=3,
        simulator=Simulator(n_workers=1),
    )
    np.testing.assert_allclose(
        res.contrast("pk.cmax", "high", "low").values, 2.0, rtol=1e-5
    )
    assert res.record["variables"][0]["by_scenario"].keys() == {"high"}


def test_the_result_does_not_depend_on_the_workers() -> None:
    args = (
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(parameters={"ka": Uncertain(1.0, cv=0.3)}),
        OBSERVABLES,
    )
    serial = propagate(
        *args, outputs=["pk.cmax"], n=512, seed=5, simulator=Simulator(n_workers=1)
    )
    pool = propagate(
        *args, outputs=["pk.cmax"], n=512, seed=5, simulator=Simulator(n_workers=2)
    )
    np.testing.assert_array_equal(
        serial.values("pk.cmax").values, pool.values("pk.cmax").values
    )


def test_a_round_trip_of_the_result(tmp_path: Path) -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(),
        OBSERVABLES,
        outputs=["pk.cmax"],
        n=64,
        seed=6,
        simulator=Simulator(n_workers=1),
    )
    res.to_netcdf(tmp_path / "uq")
    again = type(res).from_netcdf(tmp_path / "uq")
    assert again.record == res.record
    np.testing.assert_array_equal(
        again.values("pk.cmax").values, res.values("pk.cmax").values
    )


def test_an_analysis_without_uncertain_values_raises() -> None:
    with pytest.raises(ValueError, match="no uncertain values"):
        propagate(sbml_pk(), _simulation(Q(100, "mg")), UncertaintyModel(), n=8, seed=1)


def test_the_contributions_rank_the_variables() -> None:
    """The cmax depends on the dose with a CV of 30 % more than on ka with 5 %."""
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.3)),
        UncertaintyModel(parameters={"ka": Uncertain(1.0, cv=0.05)}),
        OBSERVABLES,
        outputs=["pk.cmax"],
        n=1024,
        seed=7,
        simulator=Simulator(n_workers=1),
    )
    contributions = res.contributions("pk.cmax")
    assert list(contributions.columns) == ["variable", "index", "correlated"]
    assert list(contributions.variable) == ["PODOSE", "ka"]
    assert contributions["index"].iloc[0] > 0.8
    assert not contributions.correlated.any()


def test_a_timecourse_is_no_output_of_a_summary() -> None:
    """The draws of a timecourse are not pooled over its time."""

    def run(**kwargs: object) -> object:
        return propagate(
            sbml_pk(),
            _simulation(Uncertain(Q(100, "mg"), cv=0.2)),
            UncertaintyModel(),
            OBSERVABLES,
            keep=["[C]"],
            seed=3,
            simulator=Simulator(n_workers=1),
            **kwargs,  # ty: ignore[invalid-argument-type]
        )

    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.2)),
        UncertaintyModel(),
        OBSERVABLES,
        keep=["[C]"],
        n=32,
        seed=3,
        simulator=Simulator(n_workers=1),
    )
    assert res.values("[C]").dims == ("draw", "time")
    for call in (
        lambda: res.summary(["[C]"]),
        lambda: res.probability("[C]", greater=0.0),
        lambda: res.contributions("[C]"),
    ):
        with pytest.raises(ValueError, match="one value per draw"):
            call()
    with pytest.raises(ValueError, match="one value per draw"):
        run(outputs=["[C]"])


def _sbml_initial_assignment() -> str:
    """Get a model whose rate is an initial assignment of the body weight."""
    import libsbml

    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel()
    model.setId("weight_rate")
    compartment = model.createCompartment()
    compartment.setId("c")
    compartment.setSize(1.0)
    compartment.setConstant(True)
    for pid, value in (("BW", 70.0), ("k", 1.0)):
        parameter = model.createParameter()
        parameter.setId(pid)
        parameter.setValue(value)
        parameter.setConstant(True)
    assignment = model.createInitialAssignment()
    assignment.setSymbol("k")
    assignment.setMath(libsbml.parseL3Formula("0.01 * BW"))
    species = model.createSpecies()
    species.setId("S")
    species.setCompartment("c")
    species.setInitialAmount(1.0)
    species.setHasOnlySubstanceUnits(True)
    species.setBoundaryCondition(False)
    species.setConstant(False)
    rule = model.createRateRule()
    rule.setVariable("S")
    rule.setMath(libsbml.parseL3Formula("-k * S"))
    return libsbml.writeSBMLToString(document)


def test_a_relative_distribution_has_the_reference_of_its_scenario() -> None:
    """The reference of `k` is its initial assignment of the weight of each scenario."""
    from sbmlsim.simulation.sampling import LogNormal

    res = propagate(
        _sbml_initial_assignment(),
        {
            "light": Simulation(end=1, preinit_changes={"BW": 70.0}),
            "heavy": Simulation(end=1, preinit_changes={"BW": 140.0}),
        },
        UncertaintyModel(parameters={"k": LogNormal(cv=0.2)}),
        n=32,
        seed=1,
        simulator=Simulator(n_workers=1),
    )
    light, heavy = (
        np.asarray(res.results[name].ds.coords["draw.k"].values)
        for name in ("light", "heavy")
    )
    np.testing.assert_allclose(heavy, 2.0 * light)
    assert np.median(light) == pytest.approx(0.7, rel=0.1)
