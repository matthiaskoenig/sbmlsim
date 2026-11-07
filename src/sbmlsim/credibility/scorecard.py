"""The credibility of many models at once: the scorecard and the pooling.

A repository of models assesses every model on its own; the scorecard reads
their records (`credibility.json`) into one table, one row per model and
context of use, and compares it with a stored baseline, so a change of the
shared library or of the data which makes a verdict or a metric worse shows
across all models at once (`compare`, a `Regression` each). The evidence
extensions of a project add columns to the scorecard
(`sbmlsim.credibility.extensions.EvidenceExtension.scorecard_columns`), judge
them against the baseline themselves (`EvidenceExtension.regressions`) and
pool their sections over the models (`pool`), e.g. the ratio pairs of all
models by a value of the metadata of their numerator arm.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from sbmlsim.credibility.assessment import CredibilityAssessment, Verdict
from sbmlsim.credibility.extensions import (
    rule_columns,
    scorecard_columns,
    scorecard_number,
    scorecard_true,
)
from sbmlsim.credibility.report import RECORD

if TYPE_CHECKING:
    from sbmlsim.credibility.extensions import EvidenceExtension
    from sbmlsim.credibility.uncertainty import UncertaintyRecord

__all__ = [
    "SCORECARD_COLUMNS",
    "TOLERANCE",
    "VERDICT_RANK",
    "Regression",
    "compare",
    "load_records",
    "pool",
    "scorecard",
]

#: the columns of the core of the scorecard, before the columns of the
#: extensions and the values of the acceptance rules
SCORECARD_COLUMNS: Final = (
    "model",
    "context",
    "title",
    "level",
    "risk",
    "verdict",
    "n_decisive",
    "n_evaluable",
    "verified",
    "failed_checks",
    "uncertainty_robust",
    "band_fold",
    "vpc_coverage",
    "vpc_agreement",
)
#: the rank of a verdict, a higher rank is worse
VERDICT_RANK: Final[Mapping[Verdict, int]] = {
    Verdict.SUPPORTED: 0,
    Verdict.SUPPORTED_WITH_EXTRAPOLATION: 1,
    Verdict.INSUFFICIENT_EVIDENCE: 2,
    Verdict.NOT_VERIFIED: 3,
    Verdict.NOT_SUPPORTED: 4,
}
#: the change of a metric which is no regression: points of a percentage, the
#: share of a fold error
TOLERANCE: Final = 0.05
#: the rules whose value is a fold error: lower is better, the AFE towards 1
_FOLD_ERRORS: Final = ("GMFE",)
_BIAS: Final = ("AFE",)


def load_records(paths: Iterable[Path]) -> dict[str, CredibilityAssessment]:
    """Read the records of assessments, by the model of their provenance.

    Args:
        paths: records (`credibility.json`) or directories, which are searched
            for records.

    Returns:
        The assessments by model, sorted.

    Raises:
        FileNotFoundError: if a path does not exist or no record is found.
        ValueError: if two records are of one model.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob(RECORD)))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"The record or directory '{path}' does not exist.")
    if not files:
        raise FileNotFoundError(f"No record {RECORD} was found.")
    records: dict[str, CredibilityAssessment] = {}
    sources: dict[str, Path] = {}
    for file in files:
        assessment = CredibilityAssessment.from_json(file)
        model = assessment.provenance.model
        if model in records:
            raise ValueError(
                f"The records '{sources[model]}' and '{file}' are both of the "
                f"model '{model}'."
            )
        records[model] = assessment
        sources[model] = file
    return dict(sorted(records.items()))


def scorecard(
    records: Mapping[str, CredibilityAssessment],
    extensions: Sequence[EvidenceExtension] = (),
) -> pd.DataFrame:
    """Get one row per model and context of use.

    Args:
        records: the assessments by model.
        extensions: the evidence extensions, whose columns follow the columns
            of the core.

    Returns:
        The columns of `SCORECARD_COLUMNS` (`verified` empty without
        verification, `failed_checks` the failed checks of the verification
        comma separated, `uncertainty_robust` whether every rule is met with
        the threshold probability over the parameter draws (False if one is
        shown below it, empty if one is undecided or without the
        propagation), `band_fold` the largest median fold width of the
        prediction bands of the quantities, `vpc_coverage` and
        `vpc_agreement` the coverage of the predictive check and its agreement
        with the expected coverage, empty without the check), the columns of
        the extensions (`EvidenceExtension.scorecard_names`) and per
        acceptance rule its value, the rules of the extensions included.
    """
    rows: list[dict[str, object]] = []
    for model, assessment in records.items():
        for c in assessment.contexts:
            row: dict[str, object] = {
                "model": model,
                "context": c.id,
                "title": c.title,
                "level": c.level.value,
                "risk": c.risk.value,
                "verdict": c.verdict.value,
                "n_decisive": c.n_decisive,
                "n_evaluable": c.n_evaluable,
                "verified": c.verified,
                "failed_checks": ",".join(
                    r["check"] for r in c.verification if not r["passed"]
                ),
                "uncertainty_robust": _robust(c.uncertainty),
                "band_fold": _band_fold(c.uncertainty),
                "vpc_coverage": (
                    None if c.variability is None else c.variability["coverage"]
                ),
                "vpc_agreement": (
                    None if c.variability is None else c.variability["agreement"]
                ),
            }
            for extension in extensions:
                row.update(scorecard_columns(extension, c.section(extension.name)))
            for rule in c.acceptance:
                row[rule["name"]] = rule["value"]
            rows.append(row)
    names = [n for e in extensions for n in e.scorecard_names]
    columns = list(
        dict.fromkeys([*SCORECARD_COLUMNS, *names, *(k for row in rows for k in row)])
    )
    return pd.DataFrame(rows, columns=columns)


def _robust(uncertainty: UncertaintyRecord | None) -> bool | None:
    """Get whether every rule is met robustly over the parameter draws.

    Returns:
        True if every rule is accepted, False if one is rejected, `None`
        without the propagation, a rule or with an undecided one.
    """
    if uncertainty is None or not uncertainty["rules"]:
        return None
    decisions = {r["decision"] for r in uncertainty["rules"]}
    if "rejected" in decisions:
        return False
    return True if decisions == {"accepted"} else None


def _band_fold(uncertainty: UncertaintyRecord | None) -> float | None:
    """Get the largest band fold of the quantities, `None` without."""
    if uncertainty is None:
        return None
    folds = [q["band_fold"] for q in uncertainty["quantities"] if q["band_fold"]]
    return max(folds) if folds else None


@dataclass(frozen=True, kw_only=True)
class Regression:
    """A context of use of a model which got worse than its baseline.

    Attributes:
        model: the model.
        context: the context of use.
        what: what got worse: `missing`, `verdict`, `verified`,
            `uncertainty_robust`, the name of an acceptance rule or a column
            of an extension.
        baseline: the value of the baseline, as text.
        current: the current value, as text.
    """

    model: str
    context: str
    what: str
    baseline: str
    current: str

    def __str__(self) -> str:
        """Describe the regression in one line."""
        return (
            f"{self.model}/{self.context}: {self.what} {self.baseline} -> "
            f"{self.current}"
        )


def _worse(name: str, baseline: float, current: float, tolerance: float) -> bool:
    """Check whether the value of a rule got worse beyond the tolerance."""
    if math.isnan(baseline):
        return False
    if math.isnan(current):
        return True
    if name in _FOLD_ERRORS:
        return current > baseline * (1 + tolerance)
    if name in _BIAS:
        if baseline <= 0 or current <= 0:
            return current != baseline
        return abs(math.log(current)) > abs(math.log(baseline)) + math.log1p(tolerance)
    # a percentage of the points or pairs, higher is better
    return current < baseline - 100 * tolerance


def _row(series: pd.Series) -> dict[str, object]:
    """Get a row of a scorecard as it is, a missing value of a TSV as an empty text."""
    return {
        str(k): ("" if v is None or (isinstance(v, float) and math.isnan(v)) else v)
        for k, v in series.items()
    }


def compare(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    tolerance: float = TOLERANCE,
    extensions: Sequence[EvidenceExtension] = (),
) -> list[Regression]:
    """Get the contexts of use which got worse than the baseline.

    A context of the baseline which is missing is a regression, and so is a
    worse verdict (`VERDICT_RANK`), a verified context which is no longer
    verified, a context whose rules were robust over the parameter draws and
    no longer are, the value of a rule of both which got worse by more than
    the tolerance (a percentage by `100 * tolerance` points, a GMFE by the
    share `tolerance`, an AFE by `tolerance` further from 1) and what every
    extension finds in its own columns (`EvidenceExtension.regressions`). A
    value which is no longer evaluable is worse, a new context or rule is
    none. The scorecards are compared by the names of their columns.

    Args:
        baseline: a scorecard, see `scorecard`, e.g. read from its TSV.
        current: the current scorecard.
        tolerance: the change of a value which is no regression.
        extensions: the evidence extensions, which judge their columns.

    Returns:
        The regressions, in the order of the baseline.
    """
    found: list[Regression] = []
    # a column of an extension of a baseline, e.g. of pkpdsim 0.19, is no
    # rule; without its extension the current scorecard has no such column
    rules = [
        c
        for c in rule_columns(baseline.columns, SCORECARD_COLUMNS, extensions)
        if c in current.columns
    ]
    now = {(str(r["model"]), str(r["context"])): r for _, r in current.iterrows()}
    for _, before in baseline.iterrows():
        key = (str(before["model"]), str(before["context"]))
        after = now.get(key)

        def regression(
            what: str, old: str, new: str, key: tuple[str, str] = key
        ) -> None:
            found.append(
                Regression(
                    model=key[0], context=key[1], what=what, baseline=old, current=new
                )
            )

        if after is None:
            regression("missing", "present", "missing")
            continue
        old, new = Verdict(before["verdict"]), Verdict(after["verdict"])
        if VERDICT_RANK[new] > VERDICT_RANK[old]:
            regression("verdict", old.value, new.value)
        if scorecard_true(before["verified"]) and not scorecard_true(after["verified"]):
            regression("verified", "True", str(after["verified"]))
        # a baseline of pkpdsim 0.18 or earlier has no uncertainty column
        if scorecard_true(before.get("uncertainty_robust")) and not scorecard_true(
            after.get("uncertainty_robust")
        ):
            regression(
                "uncertainty_robust", "True", str(after.get("uncertainty_robust"))
            )
        for rule in rules:
            a, b = scorecard_number(before[rule]), scorecard_number(after[rule])
            if _worse(rule, a, b, tolerance):
                regression(rule, f"{a:.4g}", f"{b:.4g}")
        for extension in extensions:
            for change in extension.regressions(_row(before), _row(after)):
                regression(change.what, change.baseline, change.current)
    return found


def pool(
    records: Mapping[str, CredibilityAssessment],
    extension: EvidenceExtension,
    options: Mapping[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Pool the sections of an extension over the contexts of use of the models.

    Args:
        records: the assessments by model.
        extension: the extension.
        options: the options of the pooling as the caller got them, e.g.
            `{"field": "health"}`; the defaults of the extension without.

    Returns:
        The pooled table of the extension (`EvidenceExtension.pool`), `None`
        for an extension which pools nothing.
    """
    contexts = {
        model: [c.section(extension.name) for c in assessment.contexts]
        for model, assessment in records.items()
    }
    return extension.pool(contexts, dict(options or {}))
