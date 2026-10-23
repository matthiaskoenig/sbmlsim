"""Probability statements about predictions, robust to the number of draws.

The question "how probable is it that an output satisfies a hypothesis H,
given everything that is uncertain?" is answered from the draws of a
propagation: the estimate is the fraction of the draws which satisfy H, an
estimate of the posterior predictive probability `P(H)` (Gelman et al. 2013).
`P(H)` is a property of the model and of what is uncertain, not of the number
of draws: more draws estimate it more precisely and change nothing else.

**Anytime-valid inference.** More draws can always be simulated, so a
statement must stay valid whenever one looks at it and adds draws (optional
stopping). A confidence interval of a fixed number of draws, e.g. the Wilson
interval, does not: looked at after every batch, it misses the true
probability far more often than its level says. The statements use the
confidence sequence of the beta-binomial mixture instead (Robbins 1970; Lai
1976; Howard et al. 2021): with the mixture of the likelihood ratios of the
first `m` draws, `M_m(p) = B(a + k_m, b + m - k_m) / B(a, b) / (p^k_m (1 -
p)^(m - k_m))` with the Jeffreys mixture `a = b = 1/2`, the set `{p: M_m(p) <
1 / alpha}` contains the true probability at every `m` at once with
probability at least `1 - alpha` (Ville's inequality, `alpha = 1 - level`).
The sequence is intersected over the numbers of draws, so it is nested: it
only shrinks with more draws. Its anytime-valid p-value of a probability
`p0` is `min(1, 1 / max_m M_m(p0))` over the numbers of draws whose estimate
is on the side of the alternative.

**Decisions.** A statement with a threshold (`at_least=p0`) is decided by the
p-values at `alpha` after every number of draws: `accepted` when `P(H) >= p0`
is shown, `rejected` when `P(H) < p0` is shown, `equivalent` when `P(H)` is
shown to be within the `margin` of the threshold (two one-sided tests,
Schuirmann 1987), `undecided` otherwise, with the number of draws which would
decide it at the current estimate (`n_needed`). More draws only make a
statement more precise: an undecided one becomes decided, an equivalence
becomes the direction which is shown later, and a direction once shown is
kept (the direction shown first, at `n_decided` draws): a decision is never
reversed, and it is wrong with probability at most `alpha`. The margin is the
difference of a probability which does not matter, by default a tenth of the
distance of the threshold to 0 or 1, whichever is nearer (0.01 for 0.9, 0.001
for 0.99); it makes every statement decidable with a bounded number of draws,
also one whose probability is the threshold.

**Several statements.** A `TestReport` adjusts the decisions for the
multiplicity, by Holm (1979, the family-wise error, by default) or by
Benjamini and Yekutieli (2001, the false discovery rate): both are valid for
any dependence between the statements, which share their draws, and both are
valid at any number of draws with anytime-valid p-values. A decision whose
adjusted p-value exceeds `alpha` becomes `undecided`, with the draws which
decide it at the level of Bonferroni, which suffice for both adjustments.

The draws which are `NaN` (failed points) are excluded and counted, infinite
values are values. The validity assumes independent draws, which
`design="random"` of a propagation gives; the counts of quasi-random draws
fluctuate less, for which the sequence is conservative in practice. A
probability statement is about the predictive distribution, i.e. about the
uncertainty of the inputs, not a test of observed data: the draws are not
subjects of a study.

Robbins H (1970). Statistical methods related to the law of the iterated
logarithm. Ann Math Stat 41:1397-1409. Lai TL (1976). On confidence
sequences. Ann Stat 4:265-280. Howard SR, Ramdas A, McAuliffe J, Sekhon J
(2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann
Stat 49:1055-1080. Schuirmann DJ (1987). A comparison of the two one-sided
tests procedure and the power approach for assessing the equivalence of
average bioavailability. J Pharmacokinet Biopharm 15:657-680. Holm S (1979).
A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70.
Benjamini Y, Yekutieli D (2001). The control of the false discovery rate in
multiple testing under dependency. Ann Stat 29:1165-1188. Gelman A, Carlin
JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013). Bayesian Data Analysis,
3rd ed. CRC Press.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from pint.errors import DimensionalityError
from scipy import special

from sbmlsim.units import Quantity

#: the references of a probability statement
REFERENCES: tuple[str, ...] = (
    "Gelman A et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.",
    "Lai TL (1976). On confidence sequences. Ann Stat 4:265-280.",
    "Howard SR et al. (2021). Ann Stat 49:1055-1080.",
)

#: the parameters of the Jeffreys mixture of the confidence sequence
MIXTURE: tuple[float, float] = (0.5, 0.5)

#: the margin of a threshold by default, a fraction of its distance to 0 or 1
MARGIN_FRACTION: float = 0.1


def default_margin(threshold: float) -> float:
    """Get the margin of a threshold by default, see the module."""
    return MARGIN_FRACTION * min(threshold, 1.0 - threshold)


#: the most draws `n_needed` searches
MAX_NEEDED: int = 2**26

#: the iterations of the bisection of the bounds of the sequence
BISECTIONS: int = 60

Decision = Literal["accepted", "rejected", "equivalent", "undecided", "estimate"]


def _log_mixture(k: np.ndarray, m: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    """Get `log M_m(p)`, the log of the mixture of the likelihood ratios."""
    a, b = MIXTURE
    return (
        special.betaln(a + k, b + m - k)
        - special.betaln(a, b)
        - special.xlogy(k, p)
        - special.xlog1py(m - k, -p)
    )


def _counts(satisfied: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Get the counts `k_m` and the numbers `m` of the prefixes of the draws."""
    x = np.asarray(satisfied, dtype=bool).ravel()
    return np.cumsum(x, dtype=float), np.arange(1, x.size + 1, dtype=float)


def _bounds(
    k: np.ndarray, m: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounds of the confidence set of every prefix, by bisection."""
    level = math.log(1.0 / alpha)
    mle = k / m
    low_lo, low_hi = np.zeros_like(mle), mle.copy()
    high_lo, high_hi = mle.copy(), np.ones_like(mle)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(BISECTIONS):
            mid = (low_lo + low_hi) / 2.0
            outside = _log_mixture(k, m, mid) >= level
            low_lo = np.where(outside, mid, low_lo)
            low_hi = np.where(outside, low_hi, mid)
            mid = (high_lo + high_hi) / 2.0
            outside = _log_mixture(k, m, mid) >= level
            high_hi = np.where(outside, mid, high_hi)
            high_lo = np.where(outside, high_lo, mid)
    return low_lo, high_hi


def confidence_sequence(
    satisfied: ArrayLike, level: float = 0.95
) -> tuple[float, float]:
    """Get the confidence sequence of a probability after the draws, see the module.

    Args:
        satisfied: whether every draw satisfies the hypothesis, in the order
            of the draws.
        level: the level of the sequence, its coverage at every number of
            draws at once.

    Returns:
        The lower and the upper end, the intersection of the confidence sets
        of every number of draws; `(0, 1)` without draws.
    """
    k, m = _counts(satisfied)
    if m.size == 0:
        return 0.0, 1.0
    lows, highs = _bounds(k, m, 1.0 - level)
    low, high = float(lows.max()), float(highs.min())
    if low > high:
        # the sets of two numbers of draws exclude each other, which happens
        # with probability at most 1 - level: the set of all the draws
        return float(lows[-1]), float(highs[-1])
    return low, high


def _p_greater(k: np.ndarray, m: np.ndarray, p0: float) -> np.ndarray:
    """Get the anytime-valid p-values of `P > p0` after every number of draws.

    The running minimum of `1 / M_m(p0)` over the numbers of draws whose
    estimate is above `p0`, see the module.
    """
    if p0 <= 0.0:
        return np.zeros_like(m)
    if p0 >= 1.0:
        return np.ones_like(m)
    with np.errstate(over="ignore"):
        p = np.where(k / m > p0, np.exp(-_log_mixture(k, m, p0)), 1.0)
    return np.minimum.accumulate(np.clip(p, 0.0, 1.0))


def _p_less(k: np.ndarray, m: np.ndarray, p0: float) -> np.ndarray:
    """Get the anytime-valid p-values of `P < p0` after every number of draws."""
    return _p_greater(m - k, m, 1.0 - p0)


def _p_values(
    k: np.ndarray, m: np.ndarray, p0: float, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the p-values of `P >= p0`, `P < p0` and of the equivalence with `p0`.

    After every number of draws; the equivalence is shown by both one-sided
    tests of the ends of the margin (Schuirmann 1987).
    """
    equivalent = np.maximum(_p_greater(k, m, p0 - margin), _p_less(k, m, p0 + margin))
    return _p_greater(k, m, p0), _p_less(k, m, p0), equivalent


def _first(shown: np.ndarray) -> int | None:
    """Get the index of the first number of draws at which something is shown."""
    indices = np.flatnonzero(shown)
    return int(indices[0]) if indices.size else None


def _decide(
    k: np.ndarray, m: np.ndarray, p0: float, margin: float, alpha: float
) -> tuple[Decision, int | None]:
    """Decide a statement of the draws, see the module.

    Returns:
        The decision and the number of draws at which it was first shown:
        the direction shown first, else the equivalence, else undecided.
    """
    p_greater, p_less, p_equivalent = _p_values(k, m, p0, margin)
    greater, less = _first(p_greater <= alpha), _first(p_less <= alpha)
    if greater is not None and (less is None or greater <= less):
        return "accepted", int(m[greater])
    if less is not None:
        return "rejected", int(m[less])
    equivalent = _first(p_equivalent <= alpha)
    if equivalent is not None:
        return "equivalent", int(m[equivalent])
    return "undecided", None


def _needed(
    estimate: float, n: int, p0: float, margin: float, alpha: float
) -> int | None:
    """Get the draws which decide a statement at its estimate, `None` beyond `MAX_NEEDED`."""
    m = n
    while m < MAX_NEEDED:
        m *= 2
        k = np.array([round(estimate * m)], dtype=float)
        decision, _ = _decide(k, np.array([m], dtype=float), p0, margin, alpha)
        if decision != "undecided":
            return m
    return None


@dataclass(frozen=True)
class TestResult:
    """A probability statement about a prediction, see the module.

    Attributes:
        hypothesis: the hypothesis in words.
        estimate: the fraction of the draws which satisfy it.
        low: the lower end of the confidence sequence.
        high: the upper end of the confidence sequence.
        n: the number of draws which are not `NaN`.
        n_failed: the number of draws which are `NaN`.
        k: the number of draws which satisfy the hypothesis.
        threshold: the probability of the decision, `None` for an estimate.
        decision: `accepted`, `rejected`, `equivalent`, `undecided` or
            `estimate`.
        n_needed: the draws an undecided statement needs at its estimate,
            `None` otherwise.
        n_decided: the number of draws at which the decision was first shown,
            `None` for an undecided statement.
        level: the level of the confidence sequence and of the decisions.
        margin: the margin of the threshold of an equivalence, `None` for an
            estimate.
        p_greater: the anytime-valid p-value of `P >= threshold`.
        p_less: the anytime-valid p-value of `P < threshold`.
        p_equivalent: the anytime-valid p-value of the equivalence.
        method: the method.
        references: the references of the method.
    """

    __test__ = False  # not a test class of pytest

    hypothesis: str
    estimate: float
    low: float
    high: float
    n: int
    n_failed: int
    k: int
    threshold: float | None
    decision: Decision
    n_needed: int | None = None
    n_decided: int | None = None
    level: float = 0.95
    margin: float | None = None
    p_greater: float = math.nan
    p_less: float = math.nan
    p_equivalent: float = math.nan
    method: str = "fraction of the draws, beta-binomial confidence sequence"
    references: tuple[str, ...] = REFERENCES

    def p_value(self) -> float:
        """Get the p-value of the decision, the smallest one of an undecided statement.

        `nan` for an estimate.
        """
        if self.threshold is None:
            return math.nan
        values = {
            "accepted": self.p_greater,
            "rejected": self.p_less,
            "equivalent": self.p_equivalent,
        }
        if self.decision in values:
            return values[self.decision]
        return min(self.p_greater, self.p_less, self.p_equivalent)


def _magnitude(value: Any, unit: str) -> float:
    """Get a bound in the unit of the values.

    Raises:
        ValueError: for a bound whose unit does not fit.
    """
    if isinstance(value, Quantity):
        try:
            return float(value.to(unit or "dimensionless").magnitude)
        except DimensionalityError as err:
            raise ValueError(
                f"The bound {value} does not fit the unit '{unit}' of the values."
            ) from err
    return float(value)


def probability(
    values: xr.DataArray,
    *,
    greater: Any = None,
    less: Any = None,
    between: tuple[Any, Any] | None = None,
    at_least: float | None = None,
    margin: float | None = None,
    level: float = 0.95,
) -> TestResult:
    """Get the probability that values satisfy a hypothesis, see the module.

    Args:
        values: the draws of an output in their order, one value per draw,
            with its unit in `attrs["units"]`.
        greater: the hypothesis `values > greater`.
        less: the hypothesis `values < less`.
        between: the hypothesis `low <= values <= high`.
        at_least: the threshold of a decision.
        margin: the margin of the threshold of an equivalence, in [0, 0.5);
            `default_margin` of the threshold by default.
        level: the level of the confidence sequence and of the decision.

    Returns:
        The statement.

    Raises:
        ValueError: unless exactly one hypothesis is given, for values which
            are not one value per draw, a threshold outside (0, 1), a margin
            outside [0, 0.5), bounds whose units do not fit, or no draw.
    """
    given = [h for h in (greater, less, between) if h is not None]
    if len(given) != 1:
        raise ValueError(
            "A probability needs exactly one of greater, less and between."
        )
    if at_least is not None and not 0.0 < at_least < 1.0:
        raise ValueError(f"The threshold {at_least} must be in (0, 1).")
    if margin is not None and not 0.0 <= margin < 0.5:
        raise ValueError(f"The margin {margin} must be in [0, 0.5).")
    if values.ndim != 1:
        raise ValueError(
            f"A probability needs one value per draw, '{values.name}' has the "
            f"dimensions {values.dims}; select a time or reduce the timecourse "
            f"with an observable, e.g. `at`, `max` or a `PK` parameter."
        )
    unit = str(values.attrs.get("units", "") or "")
    x = np.asarray(values.values, dtype=float)
    valid = ~np.isnan(x)
    n, n_failed = int(valid.sum()), int(x.size - valid.sum())
    if n == 0:
        raise ValueError(f"The output '{values.name}' has no draw which did not fail.")
    x = x[valid]
    name = str(values.name)
    shown = f" {unit}" if unit else ""
    if greater is not None:
        bound = _magnitude(greater, unit)
        satisfied = x > bound
        hypothesis = f"{name} > {bound:g}{shown}"
    elif less is not None:
        bound = _magnitude(less, unit)
        satisfied = x < bound
        hypothesis = f"{name} < {bound:g}{shown}"
    else:
        assert between is not None
        low_bound, high_bound = (_magnitude(b, unit) for b in between)
        satisfied = (x >= low_bound) & (x <= high_bound)
        hypothesis = f"{low_bound:g} <= {name} <= {high_bound:g}{shown}"
    k = int(satisfied.sum())
    low, high = confidence_sequence(satisfied, level)
    if at_least is None:
        return TestResult(
            hypothesis,
            k / n,
            low,
            high,
            n,
            n_failed,
            k,
            None,
            "estimate",
            level=level,
        )
    if margin is None:
        margin = default_margin(at_least)
    alpha = 1.0 - level
    counts, numbers = _counts(satisfied)
    p_greater, p_less, p_equivalent = (
        float(p[-1]) for p in _p_values(counts, numbers, at_least, margin)
    )
    decision, n_decided = _decide(counts, numbers, at_least, margin, alpha)
    return TestResult(
        f"P({hypothesis}) >= {at_least:g}",
        k / n,
        low,
        high,
        n,
        n_failed,
        k,
        at_least,
        decision,
        n_needed=(
            _needed(k / n, n, at_least, margin, alpha)
            if decision == "undecided"
            else None
        ),
        n_decided=n_decided,
        level=level,
        margin=margin,
        p_greater=p_greater,
        p_less=p_less,
        p_equivalent=p_equivalent,
    )


@dataclass(frozen=True)
class Hypothesis:
    """A hypothesis about an output of a propagation, which it can draw until decided.

    Attributes:
        output: the output, one value per draw, e.g. `pk.auc_last`.
        greater: the hypothesis `output > greater`.
        less: the hypothesis `output < less`.
        between: the hypothesis `low <= output <= high`.
        at_least: the threshold of the decision, `None` for an estimate.
        scenario: the scenario of the output, the only one by default.
        contrast: the scenarios `(a, b)` of a paired contrast of the output
            instead of a scenario.
        kind: the kind of the contrast, `ratio` or `difference`.
        margin: the margin of the threshold of an equivalence, see
            `probability`.
        level: the level of the decision.
    """

    output: str
    greater: Any = None
    less: Any = None
    between: tuple[Any, Any] | None = None
    at_least: float | None = None
    scenario: str | None = None
    contrast: tuple[str, str] | None = None
    kind: Literal["ratio", "difference"] = "ratio"
    margin: float | None = None
    level: float = 0.95

    def __post_init__(self) -> None:
        """Check the hypothesis.

        Raises:
            ValueError: for a scenario and a contrast, or unless exactly one
                hypothesis is given.
        """
        if self.scenario is not None and self.contrast is not None:
            raise ValueError(
                "A hypothesis is about a scenario or a contrast, not both."
            )
        if sum(h is not None for h in (self.greater, self.less, self.between)) != 1:
            raise ValueError(
                "A hypothesis needs exactly one of greater, less and between."
            )
        if self.contrast is not None:
            a, b = self.contrast
            object.__setattr__(self, "contrast", (a, b))

    def evaluate(self, values: xr.DataArray) -> TestResult:
        """Get the statement of the hypothesis about the values of its output."""
        return probability(
            values,
            greater=self.greater,
            less=self.less,
            between=self.between,
            at_least=self.at_least,
            margin=self.margin,
            level=self.level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Get the hypothesis as a dictionary of JSON types."""
        from sbmlsim.simulation.definition import _encode

        return {
            "output": self.output,
            "greater": _encode(self.greater),
            "less": _encode(self.less),
            "between": (
                None if self.between is None else [_encode(b) for b in self.between]
            ),
            "at_least": self.at_least,
            "scenario": self.scenario,
            "contrast": None if self.contrast is None else list(self.contrast),
            "kind": self.kind,
            "margin": self.margin,
            "level": self.level,
        }


def _adjusted(p: np.ndarray, method: Literal["holm", "by", "none"]) -> np.ndarray:
    """Get the adjusted p-values of Holm or Benjamini-Yekutieli."""
    if method not in ("holm", "by", "none"):
        raise ValueError(
            f"The adjustment {method!r} is none of 'holm', 'by' and 'none'."
        )
    m = p.size
    if method == "none" or m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(ranked * (m - np.arange(m)))
    else:
        harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
        adjusted = np.minimum.accumulate(
            (ranked * m * harmonic / np.arange(1, m + 1))[::-1]
        )[::-1]
    result = np.empty(m)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


@dataclass
class TestReport:
    """Probability statements with their decisions adjusted for multiplicity.

    Attributes:
        tests: the statements.
        adjust: `holm` (the family-wise error), `by` (the false discovery
            rate of Benjamini and Yekutieli) or `none`.
    """

    __test__ = False  # not a test class of pytest

    tests: Sequence[TestResult] = field(default_factory=list)
    adjust: Literal["holm", "by", "none"] = "holm"

    def _adjusted_p(self) -> dict[int, float]:
        """Get the adjusted p-values of the statements with a threshold."""
        decided = [k for k, t in enumerate(self.tests) if t.threshold is not None]
        p = np.array([self.tests[k].p_value() for k in decided])
        return dict(zip(decided, _adjusted(p, self.adjust), strict=True))

    def adjusted(self) -> list[TestResult]:
        """Get the statements with the decisions of the adjusted p-values, see the module."""
        adjusted = self._adjusted_p()
        result = list(self.tests)
        for k, value in adjusted.items():
            test = self.tests[k]
            alpha = 1.0 - test.level
            assert test.threshold is not None and test.margin is not None
            demoted = test.decision != "undecided" and value > alpha
            if demoted or (test.decision == "undecided" and len(adjusted) > 1):
                needed = _needed(
                    test.estimate,
                    test.n,
                    test.threshold,
                    test.margin,
                    alpha / len(adjusted),
                )
                result[k] = replace(
                    test, decision="undecided", n_needed=needed, n_decided=None
                )
        return result

    def table(self) -> pd.DataFrame:
        """Get the table of the statements with their adjusted decisions."""
        adjusted = self._adjusted_p()
        rows = []
        for k, (test, result) in enumerate(
            zip(self.tests, self.adjusted(), strict=True)
        ):
            rows.append(
                {
                    "hypothesis": test.hypothesis,
                    "estimate": test.estimate,
                    "low": test.low,
                    "high": test.high,
                    "n": test.n,
                    "n_failed": test.n_failed,
                    "threshold": test.threshold,
                    "p_value": test.p_value(),
                    "p_adjusted": adjusted.get(k, math.nan),
                    "decision": result.decision,
                    "n_decided": result.n_decided,
                    "n_needed": result.n_needed,
                }
            )
        return pd.DataFrame(rows)
