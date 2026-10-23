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


def _wilson_misses(x: np.ndarray, p: float, looks: list[int]) -> bool:
    """Check whether a Wilson interval looked at after every batch misses p."""
    z = 1.959963984540054
    for m in looks:
        q = x[:m].mean()
        center = (q + z**2 / (2 * m)) / (1 + z**2 / m)
        half = z * np.sqrt(q * (1 - q) / m + z**2 / (4 * m**2)) / (1 + z**2 / m)
        if not center - half <= p <= center + half:
            return True
    return False


def test_the_confidence_sequence_covers_at_every_number_of_draws() -> None:
    """The true probability is inside the sequence at every n, in 95 % of the runs.

    A Wilson interval looked at after every batch is not: its coverage holds
    at one fixed n only.
    """
    p, n, runs = 0.3, 2048, 400
    looks = [2**k for k in range(4, 12)]
    rng = np.random.default_rng(7)
    sequence = wilson = 0
    for _ in range(runs):
        x = rng.random(n) < p
        low, high = confidence_sequence(x)
        sequence += not low <= p <= high
        wilson += _wilson_misses(x, p, looks)
    # at most 5 %, within two standard errors of the runs
    assert sequence / runs <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / runs)
    assert wilson / runs > 0.15


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


def test_bonferroni_makes_marginal_decisions_undecided() -> None:
    values = _marginal()
    marginal = probability(_draws(values), greater=0.0, at_least=0.9)
    assert marginal.p_value() > 0.025
    report = TestReport([marginal, marginal]).table()
    assert list(report.decision) == ["undecided", "undecided"]
    assert list(report.p_adjusted) == pytest.approx(2 * report.p_value)
    # the intervals of the report hold for both statements at once
    assert (report.low < marginal.low).all() and (report.high > marginal.high).all()
    none = TestReport([marginal, marginal], adjust="none").table()
    assert list(none.decision) == ["accepted", "accepted"]
    with pytest.raises(ValueError, match="adjustment"):
        TestReport([marginal], adjust="holm").table()  # ty: ignore[invalid-argument-type]


def test_a_report_controls_the_family_wise_error() -> None:
    """Statements at their threshold: a direction is wrong in at most 5 % of the runs.

    Every statement has a true null, its correct decision is the equivalence:
    a step-down procedure over the decisions, e.g. Holm, would not control the
    error, the decisions at the level of Bonferroni do.
    """
    rng = np.random.default_rng(3)
    runs, wrong = 150, 0
    for _ in range(runs):
        tests = [
            probability(
                _draws(np.where(rng.random(2000) < 0.5, 1.0, -1.0)),
                greater=0.0,
                at_least=0.5,
            )
            for _ in range(10)
        ]
        decisions = [t.decision for t in TestReport(tests).adjusted()]
        wrong += any(d in ("accepted", "rejected") for d in decisions)
    assert wrong / runs <= 0.05


def test_a_report_is_not_reversed_by_more_draws() -> None:
    """More draws refine the decisions of a report and never reverse them."""
    a = _bernoulli(0.515, 2**15, seed=5)
    b = _bernoulli(0.53, 2**15, seed=6)
    allowed = {
        ("undecided", "undecided"),
        ("undecided", "equivalent"),
        ("undecided", "accepted"),
        ("undecided", "rejected"),
        ("equivalent", "equivalent"),
        ("equivalent", "accepted"),
        ("equivalent", "rejected"),
        ("accepted", "accepted"),
        ("rejected", "rejected"),
    }
    before = ["undecided", "undecided"]
    seen = set()
    for m in range(512, 2**15 + 1, 512):
        tests = [
            probability(_draws(x[:m]), greater=0.0, at_least=0.5, margin=0.05)
            for x in (a, b)
        ]
        after = [t.decision for t in TestReport(tests).adjusted()]
        assert all((x, y) in allowed for x, y in zip(before, after, strict=True)), m
        seen |= set(after)
        before = after
    assert {"equivalent", "accepted"} <= seen


@pytest.mark.parametrize("n", [20, 50, 200])
@pytest.mark.parametrize("threshold", [0.564, 0.8, 0.9, 0.95])
def test_a_report_of_one_statement_keeps_its_decision(n: int, threshold: float) -> None:
    for k in range(n + 1):
        single = probability(_draws(_spread(k, n)), greater=0.0, at_least=threshold)
        for adjust in ("none", "bonferroni"):
            (reported,) = TestReport([single], adjust=adjust).adjusted()
            assert reported.decision == single.decision, (k, adjust)


def test_a_decision_undecided_by_the_adjustment_needs_draws() -> None:
    values = _marginal()
    marginal = probability(_draws(values), greater=0.0, at_least=0.9)
    adjusted = TestReport([marginal, marginal]).adjusted()
    assert [t.decision for t in adjusted] == ["undecided", "undecided"]
    needed = adjusted[0].n_needed
    assert needed is not None and needed > values.size
    more = _spread(round(marginal.estimate * needed), needed)
    decided = probability(_draws(more), greater=0.0, at_least=0.9)
    assert [t.decision for t in TestReport([decided, decided]).adjusted()] == [
        "accepted",
        "accepted",
    ]
    assert TestReport([marginal], adjust="none").adjusted()[0].n_needed is None


def test_a_hypothesis_is_checked() -> None:
    with pytest.raises(ValueError, match="threshold"):
        Hypothesis("x", greater=1.0, at_least=1.5)
    with pytest.raises(ValueError, match="margin"):
        Hypothesis("x", greater=1.0, at_least=0.9, margin=0.7)
    with pytest.raises(ValueError, match="exactly one"):
        Hypothesis("x", at_least=0.9)


def test_contradicting_draws_keep_the_sequence_nested() -> None:
    """The draws change their probability: the sets exclude each other at last."""
    x = np.concatenate([np.ones(300), np.zeros(3000)]) > 0
    bounds = [confidence_sequence(x[:m]) for m in (100, 300, 1000, 2000, 3300)]
    for (low_a, high_a), (low_b, high_b) in itertools.pairwise(bounds):
        assert low_a <= low_b <= high_b <= high_a


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
    # independent draws, for which the statements are exactly valid
    assert res.record["design"] == "random"


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


def test_a_report_of_statements_of_other_levels() -> None:
    """Every statement at its own alpha / m: the error is at most the mean alpha."""
    values = _draws(_spread(940, 1000))
    tests = [
        probability(values, greater=0.0, at_least=0.9, level=0.95),
        probability(values, greater=0.0, at_least=0.9, level=0.99),
    ]
    adjusted = TestReport(tests).adjusted()
    assert [t.level for t in adjusted] == pytest.approx([0.975, 0.995])
    assert TestReport(tests).family_error == pytest.approx(0.03)
