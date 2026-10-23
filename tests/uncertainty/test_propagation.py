"""The propagation of uncertain values through simulations."""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation
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
        n_max=4096,
        seed=2,
        simulator=Simulator(n_workers=1),
    )
    n = res.record["n"]
    assert 256 <= n <= 4096 and n & (n - 1) == 0
    history = res.record["history"]
    assert history[-1]["n"] == n
    assert history[-1]["converged"] == (n < 4096 or history[-1]["converged"])


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


def test_failed_draws_are_counted() -> None:
    res = propagate(
        sbml_pk(),
        _simulation(Uncertain(Q(100, "mg"), cv=0.1)),
        UncertaintyModel(),
        OBSERVABLES,
        outputs=["pk.cmax"],
        n=64,
        seed=4,
        simulator=Simulator(n_workers=1),
    )
    res.results["default"].ds["pk.cmax"][:5] = np.nan
    row = res.summary(["pk.cmax"]).iloc[0]
    assert (row["n"], row["n_failed"]) == (59, 5)


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
