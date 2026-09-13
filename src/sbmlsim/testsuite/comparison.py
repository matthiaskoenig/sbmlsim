"""Comparison of the results of a case with the expected results.

The SBML Test Suite states when a simulation is correct: a value `u` is within
the tolerances of the expected value `c` if `|c - u| <= abs_tol + rel_tol * |c|`
with the tolerances of the case, see
[the README of the semantic cases](https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/semantic/README.md).
`sbmlsim.comparison.diff.within_tolerance` is that criterion, this module
applies it to a case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sbmlsim.comparison.diff import within_tolerance
from sbmlsim.testsuite.cases import SemanticCase


@dataclass(frozen=True)
class CaseComparison:
    """The comparison of the results of a case with the expected results.

    Attributes:
        cid: identifier of the case.
        valid: whether every point of every variable is within the tolerances.
        missing: variables the simulation did not produce.
        n_points: number of compared points, i.e. rows times variables.
        n_violations: number of points outside the tolerances.
        worst_variable: variable with the largest violation, empty if none.
        worst_expected: expected value at the largest violation.
        worst_observed: simulated value at the largest violation.
        worst_time: time of the largest violation.
    """

    cid: str
    valid: bool
    missing: list[str] = field(default_factory=list)
    n_points: int = 0
    n_violations: int = 0
    worst_variable: str = ""
    worst_expected: float = float("nan")
    worst_observed: float = float("nan")
    worst_time: float = float("nan")

    @property
    def summary(self) -> str:
        """Get a one line description of the comparison."""
        if self.missing:
            return f"variables not simulated: {', '.join(sorted(self.missing))}"
        if self.valid:
            return f"{self.n_points} points within the tolerances"
        return (
            f"{self.n_violations} of {self.n_points} points outside the "
            f"tolerances, worst '{self.worst_variable}' at t={self.worst_time:.6g}: "
            f"expected {self.worst_expected:.6g}, simulated {self.worst_observed:.6g}"
        )


def compare_case(case: SemanticCase, observed: pd.DataFrame) -> CaseComparison:
    """Compare the simulation of a case with its expected results.

    The expected results name the variables of the case, the simulation names
    them by their selection, i.e. a concentration is `[S1]`; the columns are
    matched on the variables of the case.

    Args:
        case: the case which was simulated.
        observed: results of the simulation, with a `time` column.

    Returns:
        The comparison, `valid` if every point is within the tolerances.
    """
    expected = case.expected()

    missing = [
        variable
        for variable, selection in zip(case.variables, case.selections[1:], strict=True)
        if selection not in observed.columns and variable not in observed.columns
    ]
    if missing:
        return CaseComparison(cid=case.cid, valid=False, missing=missing)

    n_points = 0
    n_violations = 0
    worst_variable = ""
    worst_expected = float("nan")
    worst_observed = float("nan")
    worst_time = float("nan")
    worst_excess = -np.inf

    times = np.asarray(expected["time"], dtype=float)
    for variable, selection in zip(case.variables, case.selections[1:], strict=True):
        if variable not in expected.columns:
            # the results do not carry the variable, there is nothing to compare
            continue
        column = selection if selection in observed.columns else variable
        c = np.asarray(expected[variable], dtype=float)
        u = np.asarray(observed[column], dtype=float)[: c.size]
        if u.size != c.size:
            return CaseComparison(
                cid=case.cid, valid=False, missing=[f"{variable} (point count)"]
            )

        # a value which is expected to be undefined and comes out undefined
        # agrees; `within_tolerance` is false for any comparison with a NaN
        undefined = (np.isnan(c) & np.isnan(u)) | (np.isinf(c) & (c == u))
        ok = (
            within_tolerance(
                expected=c,
                observed=u,
                abs_tol=case.absolute_tolerance,
                rel_tol=case.relative_tolerance,
            )
            | undefined
        )
        n_points += int(c.size)
        n_violations += int(np.count_nonzero(~ok))

        # the worst point is the one which exceeds its tolerance by the most,
        # so the report names a variable and not just a number. The points
        # which agree do not compete for it, which also keeps the undefined
        # ones out of the search
        with np.errstate(invalid="ignore"):
            excess = np.where(
                ok,
                -np.inf,
                np.abs(c - u)
                - (case.absolute_tolerance + case.relative_tolerance * np.abs(c)),
            )
        # a violation by a NaN has no magnitude, it is the largest there is
        excess = np.where(np.isnan(excess), np.inf, excess)
        index = int(np.argmax(excess)) if excess.size else 0
        if excess.size and excess[index] > worst_excess:
            worst_excess = float(excess[index])
            worst_variable = variable
            worst_expected = float(c[index])
            worst_observed = float(u[index])
            worst_time = float(times[index]) if index < times.size else float("nan")

    return CaseComparison(
        cid=case.cid,
        valid=n_violations == 0,
        n_points=n_points,
        n_violations=n_violations,
        worst_variable=worst_variable,
        worst_expected=worst_expected,
        worst_observed=worst_observed,
        worst_time=worst_time,
    )
