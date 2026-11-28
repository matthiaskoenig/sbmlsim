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
p)^(m - k_m))` with the Jeffreys mixture `a = b = 1/2`, a nonnegative
martingale of mean 1 under the true probability, the set `{p: M_m(p) < 1 /
alpha}` contains the true probability at every `m` at once with probability
at least `1 - alpha` (Ville's inequality, `alpha = 1 - level`). The sequence
is intersected over the numbers of draws, so it is nested: it only shrinks
with more draws (should the sets of two numbers of draws exclude each other,
which happens with probability at most `alpha`, it stays the last set which is
not empty). Its anytime-valid p-value of a probability `p0` is `min(1, 1 /
max_m M_m(p0))` over the numbers of draws whose estimate is on the side of the
alternative, valid for the composite null of that side.

**Decisions.** A statement with a threshold (`at_least=p0`) is decided by the
p-values at `alpha` after every number of draws: `accepted` when `P(H) >= p0`
is shown, `rejected` when `P(H) < p0` is shown, `equivalent` when `P(H)` is
shown to be within the `margin` of the threshold (two one-sided tests,
Schuirmann 1987), `undecided` otherwise, with the number of draws which would
decide it at the current estimate (`n_needed`). More draws only make a
statement more precise: an undecided one becomes decided, an equivalence
becomes the direction which is shown later, and a direction once shown is
kept (the direction shown first, at `n_decided` draws): a decision is never
reversed. Every wrong decision of a statement is an event of the martingale
of its true probability, so a statement is wrong with probability at most
`alpha`. The margin is the difference of a probability which does not matter,
0.05 by default; it makes every statement decidable with a bounded number of
draws, also one whose probability is the threshold.

**Several statements.** A `TestReport` decides `m` statements at the level of
Bonferroni, `alpha / m` each (and gives their simultaneous intervals at that
level), so the probability that any of them is wrong is at most `alpha` (the
mean `alpha` of statements of other levels, `TestReport.family_error`),
whatever the dependence between them, which share their draws: every
statement with a threshold has a true null (both directions at the
threshold, one direction and the equivalence elsewhere), so a correct
decision of one statement must not relax the level of the others. The
decisions of a report are the decisions of single statements at a fixed
level, so they are never reversed
by more draws either; the conservative level costs draws only, about
`log(m)` times the information per statement, which more draws provide.

The draws which are `NaN` (failed points) are excluded and counted, infinite
values are values. The validity assumes independent draws, which
`design="random"` of a propagation gives; the counts of quasi-random draws
fluctuate less, for which the sequence is conservative in practice but not by
a theorem. A probability statement is about the predictive distribution, i.e.
about the uncertainty of the inputs, not a test of observed data: the draws
are not subjects of a study.

Robbins H (1970). Statistical methods related to the law of the iterated
logarithm. Ann Math Stat 41:1397-1409. Lai TL (1976). On confidence
sequences. Ann Stat 4:265-280. Howard SR, Ramdas A, McAuliffe J, Sekhon J
(2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann
Stat 49:1055-1080. Schuirmann DJ (1987). A comparison of the two one-sided
tests procedure and the power approach for assessing the equivalence of
average bioavailability. J Pharmacokinet Biopharm 15:657-680. Gelman A,
Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013). Bayesian Data
Analysis, 3rd ed. CRC Press.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from pint.errors import DimensionalityError

from sbmlsim.engine import import_core
from sbmlsim.units import Quantity, short_unit

#: the references of a probability statement
REFERENCES: tuple[str, ...] = (
    "Gelman A et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.",
    "Lai TL (1976). On confidence sequences. Ann Stat 4:265-280.",
    "Howard SR et al. (2021). Ann Stat 49:1055-1080.",
)

#: the parameters of the Jeffreys mixture of the confidence sequence, the
#: constants are the ones of `sbmlsim_core.stats`, which computes the sequence
MIXTURE: tuple[float, float] = (0.5, 0.5)

#: the margin of a threshold by default, a difference of probability which does not matter
MARGIN: float = 0.05

#: the most draws `n_needed` searches
MAX_NEEDED: int = 2**26

#: the iterations of the bisection of the bounds of the sequence
BISECTIONS: int = 52

Decision = Literal["accepted", "rejected", "equivalent", "undecided", "estimate"]


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
    import_core()
    from sbmlsim_core import stats

    return stats.confidence_sequence(satisfied, level)


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
        satisfied: whether every draw satisfies the hypothesis, in the order
            of the draws, which a statement at another level needs.
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
    satisfied: np.ndarray | None = field(default=None, repr=False, compare=False)
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

    def at_level(self, level: float) -> TestResult:
        """Get the statement at another level, e.g. of a `TestReport`.

        Raises:
            ValueError: for a statement without its draws.
        """
        if self.satisfied is None:
            raise ValueError(
                f"The statement '{self.hypothesis}' has no draws, it was not "
                f"created by `probability`."
            )
        return _statement(
            self.hypothesis,
            self.satisfied,
            self.n_failed,
            self.threshold,
            self.margin,
            level,
        )


def _statement(
    hypothesis: str,
    satisfied: np.ndarray,
    n_failed: int,
    threshold: float | None,
    margin: float | None,
    level: float,
) -> TestResult:
    """Get the statement of the draws which satisfy a hypothesis, see the module."""
    import_core()
    from sbmlsim_core import stats

    n = int(satisfied.size)
    k = int(np.count_nonzero(satisfied))
    alpha = 1.0 - level
    low, high = stats.confidence_sequence(satisfied, level)
    if threshold is None:
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
            satisfied=satisfied,
        )
    if margin is None:
        margin = MARGIN
    # the p-values after every number of draws; the equivalence is shown by
    # both one-sided tests of the ends of the margin (Schuirmann 1987)
    p_greater = stats.p_values(satisfied, threshold, "greater")
    p_less = stats.p_values(satisfied, threshold, "less")
    p_equivalent = stats.p_values(satisfied, threshold, "equivalent", margin)
    decision, n_decided = stats.decision(p_greater, p_less, p_equivalent, alpha)
    return TestResult(
        hypothesis,
        k / n,
        low,
        high,
        n,
        n_failed,
        k,
        threshold,
        decision,
        n_needed=(
            stats.needed(k / n, n, threshold, margin, alpha)
            if decision == "undecided"
            else None
        ),
        n_decided=n_decided,
        level=level,
        margin=margin,
        p_greater=float(p_greater[-1]),
        p_less=float(p_less[-1]),
        p_equivalent=float(p_equivalent[-1]),
        satisfied=satisfied,
    )


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


def _check(
    greater: Any,
    less: Any,
    between: tuple[Any, Any] | None,
    at_least: float | None,
    margin: float | None,
) -> None:
    """Check a hypothesis and its threshold.

    Raises:
        ValueError: unless exactly one hypothesis is given, for a threshold
            outside (0, 1) or a margin outside [0, 0.5).
    """
    if sum(h is not None for h in (greater, less, between)) != 1:
        raise ValueError(
            "A probability needs exactly one of greater, less and between."
        )
    if at_least is not None and not 0.0 < at_least < 1.0:
        raise ValueError(f"The threshold {at_least} must be in (0, 1).")
    if margin is not None and not 0.0 <= margin < 0.5:
        raise ValueError(f"The margin {margin} must be in [0, 0.5).")


def probability(
    values: xr.DataArray,
    *,
    greater: Any = None,
    less: Any = None,
    between: tuple[Any, Any] | None = None,
    at_least: float | None = None,
    margin: float = MARGIN,
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
        margin: the margin of the threshold of an equivalence, in [0, 0.5).
        level: the level of the confidence sequence and of the decision.

    Returns:
        The statement.

    Raises:
        ValueError: unless exactly one hypothesis is given, for values which
            are not one value per draw, a threshold outside (0, 1), a margin
            outside [0, 0.5), bounds whose units do not fit, or no draw.
    """
    _check(greater, less, between, at_least, margin)
    if values.ndim != 1:
        raise ValueError(
            f"A probability needs one value per draw, '{values.name}' has the "
            f"dimensions {values.dims}; select a time or reduce the timecourse "
            f"with an observable, e.g. `at`, `max` or a `PK` parameter."
        )
    unit = str(values.attrs.get("units", "") or "")
    x = np.asarray(values.values, dtype=float)
    valid = ~np.isnan(x)
    n_failed = int(x.size - valid.sum())
    if not valid.any():
        raise ValueError(f"The output '{values.name}' has no draw which did not fail.")
    x = x[valid]
    name = str(values.name)
    short = short_unit(unit)
    shown = f" {short}" if short else ""
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
    if at_least is not None:
        hypothesis = f"P({hypothesis}) >= {at_least:g}"
    return _statement(hypothesis, satisfied, n_failed, at_least, margin, level)


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
    margin: float = MARGIN
    level: float = 0.95

    def __post_init__(self) -> None:
        """Check the hypothesis.

        Raises:
            ValueError: for a scenario and a contrast, unless exactly one
                hypothesis is given, for a threshold outside (0, 1), a margin
                outside [0, 0.5), another kind or a level outside (0, 1).
        """
        if self.scenario is not None and self.contrast is not None:
            raise ValueError(
                "A hypothesis is about a scenario or a contrast, not both."
            )
        _check(self.greater, self.less, self.between, self.at_least, self.margin)
        if self.kind not in ("ratio", "difference"):
            raise ValueError(
                f"The kind {self.kind!r} is none of 'ratio' and 'difference'."
            )
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"The level {self.level} must be in (0, 1).")
        if self.contrast is not None:
            a, b = self.contrast
            object.__setattr__(self, "contrast", (a, b))

    def evaluate(self, values: xr.DataArray, level: float | None = None) -> TestResult:
        """Get the statement of the hypothesis about the values of its output.

        Args:
            values: the values of its output.
            level: the level of the statement, the one of the hypothesis by
                default, e.g. `family_level` of a report.
        """
        return probability(
            values,
            greater=self.greater,
            less=self.less,
            between=self.between,
            at_least=self.at_least,
            margin=self.margin,
            level=self.level if level is None else level,
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


def family_level(
    level: float, m: int, adjust: Literal["bonferroni", "none"] = "bonferroni"
) -> float:
    """Get the level of a statement with a threshold in a report of `m` of them."""
    if adjust == "none":
        return level
    import_core()
    from sbmlsim_core import stats

    return stats.family_level(level, m)


@dataclass
class TestReport:
    """Probability statements with their decisions adjusted for multiplicity.

    Attributes:
        tests: the statements.
        adjust: `bonferroni` (the family-wise error of the decisions, see the
            module) or `none`.
    """

    __test__ = False  # not a test class of pytest

    tests: Sequence[TestResult] = field(default_factory=list)
    adjust: Literal["bonferroni", "none"] = "bonferroni"

    def __post_init__(self) -> None:
        """Check the adjustment.

        Raises:
            ValueError: for another adjustment.
        """
        if self.adjust not in ("bonferroni", "none"):
            raise ValueError(
                f"The adjustment {self.adjust!r} is none of 'bonferroni' and 'none'."
            )

    @property
    def m(self) -> int:
        """Get the number of statements with a threshold, the family of the decisions."""
        return sum(t.threshold is not None for t in self.tests)

    @property
    def family_error(self) -> float:
        """Get the bound of the probability that any decision is wrong.

        The mean `alpha` of the statements with a threshold, each decided at
        its own `alpha / m`; the `alpha` of the statement without adjustment.
        """
        alphas = [1.0 - t.level for t in self.tests if t.threshold is not None]
        if not alphas:
            return 0.0
        return float(np.mean(alphas)) if self.adjust == "bonferroni" else sum(alphas)

    def adjusted(self) -> list[TestResult]:
        """Get the statements at the level of Bonferroni, see the module."""
        m = self.m
        if self.adjust == "none" or m <= 1:
            return list(self.tests)
        return [
            t.at_level(family_level(t.level, m, self.adjust))
            if t.threshold is not None
            else t
            for t in self.tests
        ]

    def table(self) -> pd.DataFrame:
        """Get the table of the statements with their adjusted decisions.

        The interval, the decision and the draws are the ones of the adjusted
        level, `p_adjusted` is the p-value of the decision times `m`.
        """
        m = self.m if self.adjust == "bonferroni" else 1
        rows = []
        for test, result in zip(self.tests, self.adjusted(), strict=True):
            rows.append(
                {
                    "hypothesis": test.hypothesis,
                    "estimate": test.estimate,
                    "low": result.low,
                    "high": result.high,
                    "n": test.n,
                    "n_failed": test.n_failed,
                    "threshold": test.threshold,
                    "level": result.level,
                    "p_value": result.p_value(),
                    "p_adjusted": min(1.0, m * result.p_value()),
                    "decision": result.decision,
                    "n_decided": result.n_decided,
                    "n_needed": result.n_needed,
                }
            )
        return pd.DataFrame(rows)
