"""An evidence extension of the toy credibility package: the arms of a ratio.

A ratio quantity with an `Arms` pairing pairs the decisive mappings of a study
which `numerator` selects with the one mapping of the study which
`denominator` selects; a study with two denominators is a problem of the
selection, as the ratios of a PK/PD library check it. A pair is within when
the predicted ratio of the means lies within `FOLD` of the observed one, and
the rule `arms_within` of the extension is met with at least half of the pairs
within. The extension uses every hook of the contract: the acceptance, the
quantities, the factors and the section of a context, the section of the
assessment, the blocks in the card of a context and a section of the report,
a table, the scorecard and the pooling.
"""

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np
import pandas as pd

from sbmlsim.credibility import (
    AcceptanceResult,
    ContextBlock,
    ContextContribution,
    ContextOfUse,
    Credibility,
    EvidenceExtension,
    FactorStatement,
    FitTable,
    MappingLookup,
    Pairing,
    ReportSection,
    ScorecardChange,
    acceptance_result,
)
from sbmlsim.credibility.extensions import scorecard_number
from sbmlsim.fit.helpers import MappingFilter

if TYPE_CHECKING:
    from sbmlsim.project import Project

#: the fold within which a predicted ratio is within its observed one
FOLD = 1.25


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


def _pairs(
    cou: ContextOfUse, decisive: pd.DataFrame, lookup: MappingLookup
) -> list[dict[str, Any]]:
    """Get the pairs of the arms of the ratio quantities of a context."""
    pairs: list[dict[str, Any]] = []
    for q in cou.ratios:
        arms = q.pairing
        if not isinstance(arms, Arms):
            continue
        for experiment, group in decisive.groupby("experiment", sort=True):
            mappings = list(dict.fromkeys(group["mapping"]))
            study = str(experiment)
            numerators = [m for m in mappings if arms.numerator(m, lookup(study, m))]
            denominators = [
                m for m in mappings if arms.denominator(m, lookup(study, m))
            ]
            if len(denominators) != 1:
                continue
            (denominator,) = denominators
            below = group[group["mapping"] == denominator]
            for numerator in numerators:
                above = group[group["mapping"] == numerator]
                observed = float(above["DV"].mean() / below["DV"].mean())
                predicted = float(above["IPRED"].mean() / below["IPRED"].mean())
                ratio = predicted / observed
                within = (
                    bool(abs(math.log(ratio)) <= math.log(FOLD))
                    if math.isfinite(ratio) and ratio > 0
                    else None
                )
                pairs.append(
                    {
                        "quantity": q.id,
                        "experiment": study,
                        "numerator": numerator,
                        "denominator": denominator,
                        "observed": observed if math.isfinite(observed) else None,
                        "predicted": predicted if math.isfinite(predicted) else None,
                        "within": within,
                    }
                )
    return pairs


def _rule(pairs: Sequence[Mapping[str, Any]]) -> AcceptanceResult:
    """Get the rule of the pairs: at least half of the evaluable pairs within."""
    evaluable = [p["within"] for p in pairs if p["within"] is not None]
    value = 100.0 * float(np.mean(evaluable)) if evaluable else math.nan
    return acceptance_result("arms_within", value, ">= 50", value >= 50.0)


class ArmsExtension(EvidenceExtension):
    """The pairs of the arms of the ratio quantities, see the module."""

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
    def prepare(
        self, project: Project, credibility: Credibility, fits: FitTable
    ) -> object:
        return sorted(set(fits.table["experiment"].astype(str)))

    @override
    def assess_context(
        self,
        cou: ContextOfUse,
        decisive: pd.DataFrame,
        fits: FitTable,
        prepared: object,
    ) -> ContextContribution | None:
        if not any(isinstance(q.pairing, Arms) for q in cou.ratios):
            return None
        pairs = _pairs(cou, decisive, fits.lookup)
        within = sum(p["within"] is True for p in pairs)
        return ContextContribution(
            section={"pairs": pairs},
            acceptance=(_rule(pairs),),
            quantities={
                q.id: sum(
                    p["quantity"] == q.id and p["within"] is not None for p in pairs
                )
                for q in cou.ratios
                if isinstance(q.pairing, Arms)
            },
            factors={
                "relevance_of_quantities": FactorStatement(
                    f"{within} of {len(pairs)} pairs of arms within {FOLD}-fold."
                ),
                "output_comparison": FactorStatement("The arms are paired."),
            },
        )

    @override
    def acceptance(
        self, cou: ContextOfUse, decisive: pd.DataFrame, lookup: MappingLookup
    ) -> list[AcceptanceResult]:
        return [_rule(_pairs(cou, decisive, lookup))]

    @override
    def assessment_section(self, prepared: object) -> Mapping[str, Any] | None:
        return {"studies": prepared}

    @override
    def context_report_blocks(
        self, context: Mapping[str, Any], section: Mapping[str, Any] | None
    ) -> Sequence[ContextBlock]:
        if section is None:
            return ()
        rows = "".join(
            f"<tr><td>{html.escape(p['experiment'])}</td>"
            f"<td>{html.escape(p['numerator'])} / {html.escape(p['denominator'])}</td>"
            f"<td>{p['within']}</td></tr>"
            for p in section["pairs"]
        )
        return (
            ContextBlock(
                f'<h4>Arms</h4><table class="arms-pairs">{rows}</table>',
                after="acceptance",
            ),
            ContextBlock(
                f'<p class="arms-note">{len(section["pairs"])} pairs of arms.</p>',
                after="factors",
            ),
        )

    @override
    def report_section(self, record: Mapping[str, Any]) -> ReportSection | None:
        studies = record["extensions"].get(self.name, {}).get("studies", [])
        return ReportSection(
            id="arms",
            title="Arms of the ratios",
            html=f'<p class="arms-studies">{", ".join(studies)}</p>',
        )

    @override
    def report_tables(self, record: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        rows = [
            {"context": c["id"], **p}
            for c in record["contexts"]
            for p in (c["extensions"].get(self.name) or {}).get("pairs", ())
        ]
        return {
            "arms": pd.DataFrame(
                rows,
                columns=[
                    "context",
                    "quantity",
                    "experiment",
                    "numerator",
                    "denominator",
                    "observed",
                    "predicted",
                    "within",
                ],
            )
        }

    @override
    def pool(
        self,
        contexts: Mapping[str, Sequence[Mapping[str, Any] | None]],
        options: Mapping[str, Any],
    ) -> pd.DataFrame | None:
        rows = [
            {"model": model, "n_pairs": len(section["pairs"])}
            for model, sections in contexts.items()
            for section in sections
            if section is not None
        ]
        return pd.DataFrame(rows, columns=["model", "n_pairs"])

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
