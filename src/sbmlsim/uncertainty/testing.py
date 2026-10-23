"""Probability statements about predictions with their Monte Carlo error.

The question "how probable is it that an output satisfies a hypothesis H,
given everything that is uncertain?" is answered from the draws of a
propagation: the estimate is the fraction of the draws which satisfy H (a
posterior predictive probability, Gelman et al. 2013), its uncertainty from
the finite number of draws is the Wilson score interval of a binomial
proportion (Wilson 1927), which stays inside [0, 1] and is not degenerate at
0 or 1.

A decision against a threshold (`at_least=p0`) is made only when the whole
interval is on one side of it: `accepted` when its lower end is at least
`p0`, `rejected` when its upper end is below `p0`, and `undecided` otherwise,
together with the number of draws which would decide it at the current
estimate (`n_needed`), which a propagation with more draws can run. The draws
which are not finite (failed points) are excluded and counted.

A `TestReport` of several statements adjusts their decisions for the
multiplicity: the one-sided p-values of the score tests of the thresholds,
which are dual to the Wilson interval (a statement is decided exactly when its
p-value is at most `(1 - level) / 2`, the one-sided level of the interval),
are adjusted by Holm (1979, the family-wise error, by default) or by Benjamini
and Hochberg (1995, the false discovery rate, for screening many outputs), and
a decision whose adjusted p-value exceeds `(1 - level) / 2` becomes
`undecided`, with the draws which decide it at the level of Bonferroni, which
suffice for both adjustments.

A probability statement is about the predictive distribution, i.e. about the
uncertainty of the inputs, not a test of observed data: the draws are not
subjects of a study, and a frequentist test of a study is a test of simulated
trials of a stated size, see the virtual trials of a later release.

Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB (2013).
Bayesian Data Analysis, 3rd ed. CRC Press. Wilson EB (1927). Probable
inference, the law of succession, and statistical inference. J Am Stat Assoc
22:209-212. Holm S (1979). A simple sequentially rejective multiple test
procedure. Scand J Stat 6:65-70. Benjamini Y, Hochberg Y (1995). Controlling
the false discovery rate: a practical and powerful approach to multiple
testing. J R Stat Soc B 57:289-300.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pint.errors import DimensionalityError
from scipy import stats

from sbmlsim.units import Quantity

#: the references of a probability statement
REFERENCES: tuple[str, ...] = (
    "Gelman A et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.",
    "Wilson EB (1927). J Am Stat Assoc 22:209-212.",
)

#: the most draws `n_needed` searches
MAX_NEEDED: int = 2**26


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Get the Wilson score interval of a binomial proportion.

    Args:
        k: the number of successes.
        n: the number of trials, positive.
        level: the level of the interval.

    Returns:
        The lower and the upper end, inside [0, 1].

    Raises:
        ValueError: without trials.
    """
    if n <= 0:
        raise ValueError("The Wilson interval needs at least one trial.")
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = k / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class TestResult:
    """A probability statement about a prediction, see the module.

    Attributes:
        hypothesis: the hypothesis in words.
        estimate: the fraction of the draws which satisfy it.
        low: the lower end of the Wilson interval.
        high: the upper end of the Wilson interval.
        n: the number of finite draws.
        n_failed: the number of draws which are not finite.
        threshold: the probability of the decision, `None` for an estimate.
        decision: `accepted`, `rejected`, `undecided` or `estimate`.
        n_needed: the draws a decision needs at the estimate, `None` if it is
            decided or cannot be decided (the estimate is the threshold).
        level: the level of the interval.
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
    threshold: float | None
    decision: Literal["accepted", "rejected", "undecided", "estimate"]
    n_needed: int | None = None
    level: float = 0.95
    method: str = "fraction of the draws, Wilson score interval"
    references: tuple[str, ...] = REFERENCES

    @property
    def k(self) -> int:
        """Get the number of draws which satisfy the hypothesis."""
        return round(self.estimate * self.n)

    def p_value(self) -> float:
        """Get the one-sided p-value of the score test of the decision.

        The score `z = (p - p0) / sqrt(p0 (1 - p0) / n)` of the estimate `p`
        against the threshold `p0`, whose test is dual to the Wilson interval:
        `greater` for an estimate at least the threshold, `less` otherwise;
        `nan` for an estimate.
        """
        if self.threshold is None or self.n == 0:
            return math.nan
        p0 = self.threshold
        z = (self.k / self.n - p0) / math.sqrt(p0 * (1.0 - p0) / self.n)
        if self.k / self.n >= p0:
            return float(stats.norm.sf(z))
        return float(stats.norm.cdf(z))


def _decide(
    k: int, n: int, threshold: float, level: float
) -> tuple[Literal["accepted", "rejected", "undecided"], int | None]:
    """Decide a threshold and get the draws an undecided one needs."""
    low, high = wilson(k, n, level)
    if low >= threshold:
        return "accepted", None
    if high < threshold:
        return "rejected", None
    p = k / n
    if p == threshold:
        return "undecided", None
    m = n
    while m < MAX_NEEDED:
        m *= 2
        lo, hi = wilson(round(p * m), m, level)
        if lo >= threshold or hi < threshold:
            return "undecided", m
    return "undecided", None


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
    level: float = 0.95,
) -> TestResult:
    """Get the probability that values satisfy a hypothesis, see the module.

    Args:
        values: the draws of an output, with its unit in `attrs["units"]`.
        greater: the hypothesis `values > greater`.
        less: the hypothesis `values < less`.
        between: the hypothesis `low <= values <= high`.
        at_least: the threshold of a decision.
        level: the level of the interval.

    Returns:
        The statement.

    Raises:
        ValueError: unless exactly one hypothesis is given, for a threshold
            outside (0, 1), bounds whose units do not fit, or no finite draw.
    """
    given = [h for h in (greater, less, between) if h is not None]
    if len(given) != 1:
        raise ValueError(
            "A probability needs exactly one of greater, less and between."
        )
    if at_least is not None and not 0.0 < at_least < 1.0:
        raise ValueError(f"The threshold {at_least} must be in (0, 1).")
    if values.ndim != 1:
        raise ValueError(
            f"A probability needs one value per draw, '{values.name}' has the "
            f"dimensions {values.dims}; select a time or reduce the timecourse "
            f"with an observable, e.g. `at`, `max` or a `PK` parameter."
        )
    unit = str(values.attrs.get("units", "") or "")
    x = np.asarray(values.values, dtype=float)
    finite = np.isfinite(x)
    n, n_failed = int(finite.sum()), int(x.size - finite.sum())
    if n == 0:
        raise ValueError(f"The output '{values.name}' has no finite draw.")
    x = x[finite]
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
    low, high = wilson(k, n, level)
    if at_least is None:
        return TestResult(
            hypothesis, k / n, low, high, n, n_failed, None, "estimate", level=level
        )
    decision, needed = _decide(k, n, at_least, level)
    return TestResult(
        f"P({hypothesis}) >= {at_least:g}",
        k / n,
        low,
        high,
        n,
        n_failed,
        at_least,
        decision,
        needed,
        level,
    )


def _adjusted(p: np.ndarray, method: Literal["holm", "bh", "none"]) -> np.ndarray:
    """Get the adjusted p-values of Holm or Benjamini-Hochberg."""
    m = p.size
    if method == "none" or m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(ranked * (m - np.arange(m)))
    elif method == "bh":
        adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(
            f"The adjustment {method!r} is none of 'holm', 'bh' and 'none'."
        )
    result = np.empty(m)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


@dataclass
class TestReport:
    """Probability statements with their decisions adjusted for multiplicity.

    Attributes:
        tests: the statements.
        adjust: `holm` (the family-wise error), `bh` (the false discovery
            rate) or `none`.
    """

    __test__ = False  # not a test class of pytest

    tests: Sequence[TestResult] = field(default_factory=list)
    adjust: Literal["holm", "bh", "none"] = "holm"

    def adjusted(self) -> list[TestResult]:
        """Get the statements with the decisions of the adjusted p-values, see the module."""
        decided = [k for k, t in enumerate(self.tests) if t.threshold is not None]
        p = np.array([self.tests[k].p_value() for k in decided])
        adjusted = _adjusted(p, self.adjust)
        result = list(self.tests)
        for k, value in zip(decided, adjusted, strict=True):
            test = self.tests[k]
            if (
                test.decision in ("accepted", "rejected")
                and value > (1.0 - test.level) / 2.0
            ):
                assert test.threshold is not None
                bonferroni = 1.0 - (1.0 - test.level) / len(decided)
                _, needed = _decide(test.k, test.n, test.threshold, bonferroni)
                result[k] = replace(test, decision="undecided", n_needed=needed)
        return result

    def table(self) -> pd.DataFrame:
        """Get the table of the statements with their adjusted decisions."""
        decided = {k for k, t in enumerate(self.tests) if t.threshold is not None}
        p = np.array([self.tests[k].p_value() for k in sorted(decided)])
        adjusted = dict(zip(sorted(decided), _adjusted(p, self.adjust), strict=True))
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
                    "n_needed": test.n_needed,
                }
            )
        return pd.DataFrame(rows)
