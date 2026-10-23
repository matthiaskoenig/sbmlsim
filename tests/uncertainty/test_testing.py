"""Probability statements about predictions and their reports."""

import itertools

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.simulation import PK, Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import Hypothesis, Uncertain, UncertaintyModel, propagate
from sbmlsim.uncertainty.testing import (
    TestReport,
    confidence_sequence,
    default_margin,
    probability,
)
from tests.simulator.models import sbml_pk


def _draws(values: list[float] | np.ndarray, unit: str = "") -> xr.DataArray:
    data = xr.DataArray(np.asarray(values, dtype=float), dims=("draw",), name="x")
    data.attrs["units"] = unit
    return data


def _spread(k: int, n: int) -> np.ndarray:
    """Get n draws of which k are 1 and n - k are -1, evenly spread."""
    i = np.arange(n)
    ones = np.floor((i + 1) * k / n) - np.floor(i * k / n)
    return np.where(ones > 0, 1.0, -1.0)


def _bernoulli(p: float, n: int, seed: int) -> np.ndarray:
    return np.where(np.random.default_rng(seed).random(n) < p, 1.0, -1.0)


def test_the_confidence_sequence_covers_at_every_number_of_draws() -> None:
    """The true probability is inside the sequence at every n, in 95 % of the runs.

    A Wilson interval looked at after every batch would not be: its coverage
    holds at one fixed n only.
    """
    p, n, runs = 0.3, 2000, 400
    rng = np.random.default_rng(7)
    outside = 0
    for _ in range(runs):
        x = rng.random(n) < p
        low, high = confidence_sequence(x)
        outside += not low <= p <= high
    assert outside / runs <= 0.05


def test_the_sequence_is_nested() -> None:
    x = _bernoulli(0.7, 4096, seed=2) > 0
    bounds = [confidence_sequence(x[:m]) for m in (64, 256, 1024, 4096)]
    for (low_a, high_a), (low_b, high_b) in itertools.pairwise(bounds):
        assert low_a <= low_b <= high_b <= high_a
    assert bounds[-1][1] - bounds[-1][0] < 0.08


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.95, "accepted"), (0.86, "rejected"), (0.9, "equivalent")],
)
def test_a_decision_does_not_change_with_more_draws(p: float, expected: str) -> None:
    """More draws only decide what was undecided, and never change a decision."""
    x = _bernoulli(p, 2**15, seed=11)
    decisions = [
        probability(_draws(x[:m]), greater=0.0, at_least=0.9, margin=0.02).decision
        for m in 2 ** np.arange(6, 16)
    ]
    first = next(k for k, d in enumerate(decisions) if d != "undecided")
    assert set(decisions[:first]) == {"undecided"}
    assert set(decisions[first:]) == {expected}


def test_an_equivalence_becomes_the_direction_shown_later() -> None:
    """More draws refine a decision and never reverse it."""
    values = np.ones(4096)
    decisions = [
        probability(_draws(values[:m]), greater=0.0, at_least=0.99, margin=0.02)
        for m in (256, 4096)
    ]
    assert [d.decision for d in decisions] == ["equivalent", "accepted"]
    first, later = (d.n_decided for d in decisions)
    assert first is not None and later is not None and first < 256 < later
    assert decisions[1].p_greater < 0.05 < decisions[0].p_greater


def test_the_default_margin_is_relative_to_the_threshold() -> None:
    assert default_margin(0.9) == pytest.approx(0.01)
    assert default_margin(0.99) == pytest.approx(0.001)
    assert default_margin(0.3) == pytest.approx(0.03)
    statement = probability(_draws(np.ones(64)), greater=0.0, at_least=0.9)
    assert statement.margin == pytest.approx(0.01)
    assert probability(_draws(np.ones(64)), greater=0.0).margin is None


def test_all_draws_and_a_high_threshold() -> None:
    accepted = probability(_draws([1.0] * 100), greater=0.0, at_least=0.9)
    assert accepted.decision == "accepted"
    result = probability(_draws([1.0] * 100), greater=0.0, at_least=0.95)
    assert result.estimate == 1.0
    assert 0.9 < result.low < 0.95 and result.high == pytest.approx(1.0)
    assert result.decision == "undecided"
    assert result.n_needed is not None and result.n_needed > 100
    more = probability(_draws([1.0] * result.n_needed), greater=0.0, at_least=0.95)
    assert more.decision == "accepted"


def test_a_statement_and_its_margin() -> None:
    values = _spread(950, 1000)
    assert probability(_draws(values), greater=0.0, at_least=0.9).decision == (
        "accepted"
    )
    assert probability(_draws(values), greater=0.0, at_least=0.99).decision == (
        "rejected"
    )
    near = probability(_draws(values), greater=0.0, at_least=0.95, margin=0.05)
    assert near.decision == "equivalent"
    assert near.margin == 0.05
    estimate = probability(_draws(values), greater=0.0)
    assert estimate.decision == "estimate"
    assert estimate.threshold is None
    assert estimate.low < 0.95 < estimate.high
    assert "confidence sequence" in estimate.method and estimate.references


def test_between_with_units() -> None:
    values = _draws([0.5, 1.5, 2.5, 3.5], unit="mg/l")
    result = probability(values, between=(Q(1000, "ug/l"), Q(3, "mg/l")))
    assert result.estimate == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unit"):
        probability(values, greater=Q(1, "mg"))


def test_failed_draws_are_excluded_and_infinite_ones_are_not() -> None:
    result = probability(_draws([1.0, np.nan, 2.0, -1.0, np.inf]), greater=0.0)
    assert (result.n, result.n_failed, result.k) == (4, 1, 3)
    assert result.estimate == pytest.approx(3 / 4)


def _marginal(n: int = 2000, threshold: float = 0.9) -> np.ndarray:
    """Get the draws whose statement is accepted just barely."""
    for k in range(int(threshold * n), n + 1):
        values = _spread(k, n)
        if probability(_draws(values), greater=0.0, at_least=threshold).decision == (
            "accepted"
        ):
            return values
    raise AssertionError


def test_holm_makes_marginal_decisions_undecided() -> None:
    values = _marginal()
    marginal = probability(_draws(values), greater=0.0, at_least=0.9)
    assert marginal.p_value() > 0.025
    holm = TestReport([marginal, marginal], adjust="holm").table()
    assert list(holm.decision) == ["undecided", "undecided"]
    assert (holm.p_adjusted > holm.p_value).all()
    none = TestReport([marginal, marginal], adjust="none").table()
    assert list(none.decision) == ["accepted", "accepted"]
    by = TestReport([marginal, marginal], adjust="by").table()
    assert list(by.decision) == ["undecided", "undecided"]
    with pytest.raises(ValueError, match="adjustment"):
        TestReport([marginal], adjust="bh").table()  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("n", [20, 50, 200])
@pytest.mark.parametrize("threshold", [0.564, 0.8, 0.9, 0.95])
def test_a_report_of_one_statement_keeps_its_decision(n: int, threshold: float) -> None:
    for k in range(n + 1):
        single = probability(_draws(_spread(k, n)), greater=0.0, at_least=threshold)
        for adjust in ("none", "holm", "by"):
            (reported,) = TestReport([single], adjust=adjust).adjusted()
            assert reported.decision == single.decision, (k, adjust)


def test_a_decision_undecided_by_the_adjustment_needs_draws() -> None:
    values = _marginal()
    marginal = probability(_draws(values), greater=0.0, at_least=0.9)
    adjusted = TestReport([marginal, marginal], adjust="holm").adjusted()
    assert [t.decision for t in adjusted] == ["undecided", "undecided"]
    needed = adjusted[0].n_needed
    assert needed is not None and needed > values.size
    more = _spread(round(marginal.estimate * needed), needed)
    decided = probability(_draws(more), greater=0.0, at_least=0.9)
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


def _scenarios(dose: Uncertain) -> dict[str, Simulation]:
    return {
        name: Simulation(
            end=48,
            steps=96,
            preinit_changes={"V": volume},
            changes=[Change(0, {"PODOSE": dose})],
        )
        for name, volume in (("small", 10.0), ("large", 20.0))
    }


def test_the_probability_of_a_propagation() -> None:
    res = propagate(
        sbml_pk(),
        _scenarios(Uncertain(Q(100, "mg"), cv=0.2)),
        UncertaintyModel(),
        [PK("pk", "[C]", dose="PODOSE", route="oral")],
        outputs=["pk.cmax"],
        n=512,
        seed=8,
        simulator=Simulator(n_workers=1),
    )
    ratio = res.contrast("pk.cmax", "large", "small")
    # 512 of 512 draws show a probability above 0.98, not yet above 0.99
    assert res.probability(ratio, less=0.6, at_least=0.98).decision == "accepted"
    undecided = res.probability(ratio, less=0.6, at_least=0.99)
    assert undecided.decision == "undecided" and undecided.n_needed == 1024
    nominal = float(res.summary(["pk.cmax"]).set_index("scenario").loc["small", "q0.5"])
    half = res.probability("pk.cmax", scenario="small", greater=nominal)
    assert half.estimate == pytest.approx(0.5, abs=0.05)
    hypothesis = Hypothesis(
        "pk.cmax", contrast=("large", "small"), less=0.6, at_least=0.98
    )
    (tested,) = res.test([hypothesis]).adjusted()
    assert tested.decision == "accepted"
    assert tested.hypothesis.startswith("P(pk.cmax:large/small < 0.6)")


def test_a_propagation_draws_until_its_hypotheses_are_decided() -> None:
    """The draws double until every hypothesis is decided after the adjustment."""
    hypotheses = [
        Hypothesis("pk.cmax", scenario="small", greater=0.0, at_least=0.9),
        Hypothesis("pk.cmax", contrast=("large", "small"), less=0.55, at_least=0.9),
    ]
    res = propagate(
        sbml_pk(),
        _scenarios(Uncertain(Q(100, "mg"), cv=0.2)),
        UncertaintyModel(),
        [PK("pk", "[C]", dose="PODOSE", route="oral")],
        hypotheses=hypotheses,
        n_start=16,
        seed=8,
        simulator=Simulator(n_workers=1),
    )
    history = res.record["history"]
    assert [entry["n"] for entry in history] == [16 * 2**k for k in range(len(history))]
    assert history[-1]["undecided"] == 0 and history[0]["undecided"] > 0
    assert [t.decision for t in res.test(hypotheses).adjusted()] == [
        "accepted",
        "accepted",
    ]
    assert res.record["hypotheses"][1]["contrast"] == ["large", "small"]


def test_the_sequence_covers_with_quasi_random_draws() -> None:
    """The counts of scrambled Sobol points fluctuate less than of random ones."""
    from scipy.stats import qmc

    truth = 1.2**3 / 6  # the volume of the simplex below a sum of 1.2 in [0, 1]^3
    truth -= 3 * (0.2**3) / 6
    outside = 0
    for seed in range(100):
        u = qmc.Sobol(3, scramble=True, seed=seed).random(1024)
        low, high = confidence_sequence(u.sum(axis=1) < 1.2)
        outside += not low <= truth <= high
    assert outside <= 5
