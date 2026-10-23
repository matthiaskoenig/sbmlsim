"""Probability statements about predictions and their reports."""

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import Uncertain, UncertaintyModel, propagate
from sbmlsim.uncertainty.testing import TestReport, probability, wilson
from tests.simulator.models import sbml_pk


def _draws(values: list[float], unit: str = "") -> xr.DataArray:
    data = xr.DataArray(np.asarray(values, dtype=float), dims=("draw",), name="x")
    data.attrs["units"] = unit
    return data


def test_the_wilson_interval() -> None:
    low, high = wilson(950, 1000)
    assert low == pytest.approx(0.9346, abs=1e-3)
    assert high == pytest.approx(0.9626, abs=1e-3)
    # all draws: the interval stays in [0, 1] and is not degenerate
    low, high = wilson(100, 100)
    assert 0.95 < low < 1.0 and high == pytest.approx(1.0)


def test_all_draws_do_not_decide_a_high_threshold() -> None:
    """100 of 100 draws is not enough for a probability of at least 0.97.

    The lower end of the Wilson interval of 100 of 100 is 0.963.
    """
    assert probability(_draws([1.0] * 100), greater=0.0, at_least=0.95).decision == (
        "accepted"
    )
    result = probability(_draws([1.0] * 100), greater=0.0, at_least=0.97)
    assert result.estimate == 1.0
    assert result.decision == "undecided"
    assert result.n_needed is not None and result.n_needed > 100


def test_a_decision() -> None:
    values = [1.0] * 950 + [-1.0] * 50
    accepted = probability(_draws(values), greater=0.0, at_least=0.9)
    assert accepted.decision == "accepted"
    rejected = probability(_draws(values), greater=0.0, at_least=0.99)
    assert rejected.decision == "rejected"
    estimate = probability(_draws(values), greater=0.0)
    assert estimate.decision == "estimate"
    assert estimate.threshold is None
    assert "Wilson" in estimate.method and estimate.references


def test_between_with_units() -> None:
    values = _draws([0.5, 1.5, 2.5, 3.5], unit="mg/l")
    result = probability(values, between=(Q(1000, "ug/l"), Q(3, "mg/l")))
    assert result.estimate == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unit"):
        probability(values, greater=Q(1, "mg"))


def test_failed_draws_are_excluded() -> None:
    result = probability(_draws([1.0, np.nan, 2.0, -1.0]), greater=0.0)
    assert (result.n, result.n_failed) == (3, 1)
    assert result.estimate == pytest.approx(2 / 3)


def test_holm_makes_marginal_decisions_undecided() -> None:
    """Two decisions at the edge of their threshold: the family-wise error counts."""
    marginal = probability(
        _draws([1.0] * 965 + [-1.0] * 35), greater=0.0, at_least=0.95
    )
    other = probability(_draws([1.0] * 965 + [-1.0] * 35), greater=0.0, at_least=0.95)
    assert marginal.decision == other.decision == "accepted"
    holm = TestReport([marginal, other], adjust="holm").table()
    assert list(holm.decision) == ["undecided", "undecided"]
    assert (holm.p_adjusted > holm.p_value).all()
    none = TestReport([marginal, other], adjust="none").table()
    assert list(none.decision) == ["accepted", "accepted"]
    bh = TestReport([marginal, other], adjust="bh").table()
    assert list(bh.decision) == ["accepted", "accepted"]


def test_the_probability_of_a_propagation() -> None:
    dose = Uncertain(Q(100, "mg"), cv=0.2)
    res = propagate(
        sbml_pk(),
        {
            "small": Simulation(
                end=48,
                steps=96,
                preinit_changes={"V": 10.0},
                changes=[Change(0, {"PODOSE": dose})],
            ),
            "large": Simulation(
                end=48,
                steps=96,
                preinit_changes={"V": 20.0},
                changes=[Change(0, {"PODOSE": dose})],
            ),
        },
        UncertaintyModel(),
        [PK("pk", "[C]", dose="PODOSE", route="oral")],
        outputs=["pk.cmax"],
        n=512,
        seed=8,
        simulator=Simulator(n_workers=1),
    )
    ratio = res.contrast("pk.cmax", "large", "small")
    assert res.probability(ratio, less=0.6, at_least=0.99).decision == "accepted"
    nominal = float(res.summary(["pk.cmax"]).set_index("scenario").loc["small", "q0.5"])
    half = res.probability("pk.cmax", scenario="small", greater=nominal)
    assert half.estimate == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("n", [20, 50, 200])
@pytest.mark.parametrize("threshold", [0.564, 0.8, 0.9, 0.95])
def test_a_report_of_one_statement_keeps_its_decision(n: int, threshold: float) -> None:
    """The p-value of a report is dual to the Wilson interval of the statement."""
    for k in range(n + 1):
        single = probability(
            _draws([1.0] * k + [-1.0] * (n - k)), greater=0.0, at_least=threshold
        )
        for adjust in ("none", "holm", "bh"):
            (reported,) = TestReport([single], adjust=adjust).adjusted()
            assert reported.decision == single.decision, (k, adjust)


def test_a_decision_undecided_by_the_adjustment_needs_draws() -> None:
    marginal = probability(
        _draws([1.0] * 965 + [-1.0] * 35), greater=0.0, at_least=0.95
    )
    adjusted = TestReport([marginal, marginal], adjust="holm").adjusted()
    assert [t.decision for t in adjusted] == ["undecided", "undecided"]
    for test in adjusted:
        assert test.n_needed is not None and test.n_needed > 1000
        decided = probability(
            _draws(
                [1.0] * round(0.965 * test.n_needed)
                + [-1.0] * round(0.035 * test.n_needed)
            ),
            greater=0.0,
            at_least=0.95,
        )
        assert [t.decision for t in TestReport([decided, decided]).adjusted()] == [
            "accepted",
            "accepted",
        ]


def test_a_timecourse_is_no_value_per_draw() -> None:
    with pytest.raises(ValueError, match="one value per draw"):
        probability(
            xr.DataArray(np.ones((4, 3)), dims=("draw", "time"), name="[C]"),
            greater=0.0,
        )
