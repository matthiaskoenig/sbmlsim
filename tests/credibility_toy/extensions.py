"""An evidence extension of the toy credibility package: the arms of a ratio.

A ratio quantity with an `Arms` pairing pairs the decisive mappings of a study
which `numerator` selects with the one mapping of the study which
`denominator` selects; a study with two denominators is a problem of the
selection, as the ratios of a PK/PD library check it.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, override

import pandas as pd

from sbmlsim.credibility import (
    ContextOfUse,
    EvidenceExtension,
    MappingLookup,
    Pairing,
    ScorecardChange,
)
from sbmlsim.credibility.extensions import scorecard_number
from sbmlsim.fit.helpers import MappingFilter


@dataclass(frozen=True, kw_only=True)
class Arms:
    """The arms of a ratio: the numerator over the denominator of a study."""

    observable: str
    numerator: MappingFilter
    denominator: MappingFilter


@dataclass(frozen=True)
class MinimumPairs:
    """A setting of the extension for a context of use, checked with the declaration."""

    minimum: int

    def problems(self) -> list[str]:
        """Get the problems: a minimum below 1."""
        return [] if self.minimum >= 1 else [f"the minimum {self.minimum} is below 1"]


class ArmsExtension(EvidenceExtension):
    """Checks that every study of a ratio has one denominator."""

    name: ClassVar[str] = "arms"
    label: ClassVar[str] = "arms of a ratio"
    scorecard_names: ClassVar[tuple[str, ...]] = ("arms",)

    @override
    def claims(self, pairing: Pairing) -> bool:
        return isinstance(pairing, Arms)

    @override
    def selection_problems(
        self, cou: ContextOfUse, decisive: pd.DataFrame, lookup: MappingLookup
    ) -> list[str]:
        problems: list[str] = []
        for q in cou.ratios:
            arms = q.pairing
            if not isinstance(arms, Arms):
                continue
            pairs = decisive[["experiment", "mapping"]].drop_duplicates()
            for experiment, group in pairs.groupby("experiment"):
                denominators = [
                    m
                    for m in group["mapping"]
                    if arms.denominator(m, lookup(str(experiment), m))
                ]
                if len(denominators) > 1:
                    problems.append(
                        f"the ratio '{q.id}' has the denominators "
                        f"{sorted(denominators)} in the study {experiment}"
                    )
        return problems

    @override
    def scorecard_columns(
        self, context: Mapping[str, Any] | None
    ) -> Mapping[str, object]:
        return {"arms": None if context is None else len(context.get("pairs", ()))}

    @override
    def regressions(
        self, baseline: Mapping[str, object], current: Mapping[str, object]
    ) -> list[ScorecardChange]:
        if "arms" not in baseline:
            return []
        before = scorecard_number(baseline["arms"])
        after = scorecard_number(current.get("arms"))
        if not math.isnan(before) and not after >= before:
            return [ScorecardChange("arms", f"{before:g}", f"{after:g}")]
        return []


ARMS = ArmsExtension()
