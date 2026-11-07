"""The assessment of the contexts of use of a model as plain data.

`assess` evaluates the adopted fits of a project (`evaluate_fits`) and assesses
its declaration on their data points, `assess_table` does the same on a given
table without a project. Per context of use:

- The metrics of every evidence: the points, mappings and studies,
  `criteria_metrics` and the values of the acceptance rules of the context.
- The acceptance: the rules on the union of the points of the decisive
  evidence (`ContextOfUse.decisive_role`, calibration for C1, else
  validation), each row of the table once, followed by the rules of the
  evidence extensions of the project
  (`sbmlsim.credibility.extensions.ContextContribution.acceptance`), each row
  with the name of its extension.
- The coverage of every claimed value of the domain: `validated` with
  applicable points of validation evidence, `calibrated` with applicable points
  of calibration evidence only, `uncovered` without; a C1 context counts
  `calibrated` as covered. A point is applicable when the fold criteria can
  evaluate it, a positive and finite value and prediction (`applicable`): the
  points of a failed simulation or of zeros cover nothing. The values of the
  points outside the claimed domain are reported as unclaimed values, a point
  without the field has none.
- The coverage of every quantity of interest: `covered` with an applicable
  decisive point of its observable inside the claimed domain (every claimed
  field of the point has a claimed value), `uncovered` without; a ratio
  quantity with a pairing by the points its extension derives
  (`ContextContribution.quantities`), a declared ratio without a pairing has
  no coverage.
- The minimum of evaluable points, `ContextOfUse.minimum_points`.
- The verification of the decisive predictions, when the assessment is
  verified (`sbmlsim.credibility.verification`): per check whether it passed,
  the largest relative difference of the alternative predictions and whether
  the acceptance changes with them, and the checks of the battery of the
  variants of the model (`sbmlsim.credibility.battery`), the same for every
  context.
- The parameter uncertainty and the predictive check with the variability of
  the subjects of the decisive predictions, when they ran
  (`sbmlsim.credibility.uncertainty`, `sbmlsim.credibility.variability`).
- The verdict, see `verdict`, and the factors of V&V 40 (`FACTORS`), the
  statements of the extensions combined with the ones of the core
  (`sbmlsim.credibility.factors.combine_factor`).
- The sections of the extensions, which the extensions write and read
  (`ContextResult.extensions`, `CredibilityAssessment.extensions`).

The statement of the domain is recorded and shown, not assessed. The
limitations are reported apart, with their metrics, and enter no acceptance.
The assessment writes and reads JSON (`to_json`, `from_json`), with `NaN` as
`null`; it reads the records of pkpdsim 0.17 to 0.19 as well, whose evidence
of the extensions is read as their sections
(`sbmlsim.credibility.extensions.read_sections`).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import math
import subprocess
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum, unique
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, NotRequired, Self, TypedDict

import numpy as np
import pandas as pd

from sbmlsim.credibility.cross_validation import CrossValidation
from sbmlsim.credibility.declaration import (
    AcceptanceResult,
    AcceptanceRule,
    ContextOfUse,
    CouLevel,
    Credibility,
    EvidenceRole,
    Level,
    Selection,
    Unclaimed,
)
from sbmlsim.credibility.extensions import (
    ACCEPTANCE_EXTENSION,
    EXTENSIONS,
    LEGACY_ASSESSMENT_SECTIONS,
    LEGACY_CONTEXT_SECTIONS,
    ContextContribution,
    EvidenceExtension,
    read_acceptance,
    read_sections,
)
from sbmlsim.credibility.factors import (
    FACTORS,
    FactorRecord,
    FactorStatus,
    combine_factor,
)
from sbmlsim.credibility.records import MappingLookup, evaluate_fits
from sbmlsim.credibility.selection import (
    EvidenceSelection,
    in_domain,
    metadata_values,
    normalize_observable,
    select,
)
from sbmlsim.credibility.uncertainty import (
    FitPropagation,
    FitUncertainty,
    PropagationSettings,
    QuantityBand,
    SensitivityIndex,
    UncertaintyRecord,
    propagate_fits,
    rule_probability,
    sensitivity_indices,
    uncertainty_record_from_dict,
)
from sbmlsim.credibility.variability import (
    VPC_COLUMNS,
    PredictiveChecks,
    VariabilityRecord,
    VariabilitySettings,
    VpcPoint,
    predictive_checks,
    variability_record,
    variability_record_from_dict,
    variability_statement,
)
from sbmlsim.credibility.verification import (
    NUMERICAL_CHECKS,
    PREDICTION_CHECKS,
    SETUP,
    TIGHTENING,
    CheckRecord,
    Verification,
    VerificationSummary,
    check_of,
    check_record_from_dict,
    check_record_to_dict,
    relative_differences,
    verify,
)
from sbmlsim.fit.evaluation import Fold, criteria_metrics
from sbmlsim.fit.filters import MetadataFilter, NamedFilter

if TYPE_CHECKING:
    from pathlib import Path

    from sbmlsim.credibility.records import FitTable
    from sbmlsim.project import Project

__all__ = [
    "COVERED",
    "UNCOVERED",
    "VERSIONED_PACKAGES",
    "AcceptanceRecord",
    "ContextResult",
    "Coverage",
    "CoverageEntry",
    "CredibilityAssessment",
    "EvidenceResult",
    "Provenance",
    "QuantityRecord",
    "UnclaimedRecord",
    "Verdict",
    "applicable",
    "assess",
    "assess_table",
    "describe_selection",
    "provenance",
    "verdict",
]

logger = logging.getLogger(__name__)

#: the packages whose versions the provenance records, besides the packages of
#: the extensions and the battery hook of the project
VERSIONED_PACKAGES: Final = ("sbmlsim", "sbmlutils", "libroadrunner")

_VERDICT_LABELS: Final = {
    "supported": "supported",
    "supported_with_extrapolation": "supported with extrapolation",
    "not_supported": "not supported",
    "insufficient_evidence": "insufficient evidence",
    "not_verified": "not verified",
}


@unique
class Verdict(StrEnum):
    """The verdict of the assessment of a context of use."""

    SUPPORTED = "supported"
    SUPPORTED_WITH_EXTRAPOLATION = "supported_with_extrapolation"
    NOT_SUPPORTED = "not_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_VERIFIED = "not_verified"

    @property
    def label(self) -> str:
        """Get the verdict in words, e.g. `supported with extrapolation`."""
        return _VERDICT_LABELS[self.value]


@unique
class Coverage(StrEnum):
    """The coverage of a claimed value of the domain of a context of use."""

    VALIDATED = "validated"
    CALIBRATED = "calibrated"
    UNCOVERED = "uncovered"


#: a quantity of interest with an applicable decisive point in the claimed domain
COVERED: Final = "covered"
#: a quantity of interest without one
UNCOVERED: Final = "uncovered"


class QuantityRecord(TypedDict):
    """A quantity of interest as plain data, see `QuantityOfInterest`.

    Attributes:
        id: the id of the quantity.
        description: what the quantity is.
        ratio: whether the quantity is a ratio between arms.
        coverage: `covered` with an applicable decisive point of the quantity
            inside the claimed domain, a ratio with a pairing with a point its
            extension derives, else `uncovered`; `None` for a ratio without a
            pairing and in a record of pkpdsim 0.17, which did not assess it.
        n_points: the applicable decisive points of the quantity inside the
            claimed domain, the points the extension derives for a ratio with
            a pairing, 0 for a ratio without one.
    """

    id: str
    description: str
    ratio: bool
    coverage: str | None
    n_points: int


class AcceptanceRecord(TypedDict):
    """The result of an acceptance rule as plain data, see `AcceptanceResult`.

    Attributes:
        name: the name of the rule.
        value: its value, `NaN` if not evaluable.
        limit: its limit in words.
        met: whether it is met, `None` if not evaluable.
        extension: the name of the evidence extension of the rule, absent for
            a rule of the context, see
            `sbmlsim.credibility.extensions.ACCEPTANCE_EXTENSION`.
    """

    name: str
    value: float
    limit: str
    met: bool | None
    extension: NotRequired[str]


class UnclaimedRecord(TypedDict):
    """The unclaimed validation points of a fit mapping.

    Attributes:
        fit: the subset of the fit.
        experiment: the experiment.
        mapping: the key of the mapping.
        n: the points.
    """

    fit: str
    experiment: str
    mapping: str
    n: int


def _write(value: float) -> float | None:
    """Get a number for JSON, `None` for `NaN`."""
    return None if math.isnan(value) else value


def _read(value: float | None) -> float:
    """Get a number from JSON, `NaN` for `None`."""
    return math.nan if value is None else float(value)


def _acceptance_record(
    result: AcceptanceResult, extension: str | None = None
) -> AcceptanceRecord:
    """Get the record of the result of a rule, of an extension with its name."""
    record = AcceptanceRecord(
        name=result.name, value=result.value, limit=result.limit, met=result.met
    )
    if extension is not None:
        record[ACCEPTANCE_EXTENSION] = extension
    return record


@dataclass(frozen=True, kw_only=True)
class EvidenceResult:
    """The metrics of an evidence of a context of use or of a limitation.

    Attributes:
        context: the id of the context of use, `""` for a limitation of the
            model.
        name: the name of the evidence.
        role: the role of the evidence.
        description: what the evidence is.
        selection: what the evidence selects, see `describe_selection`.
        n_points: the data points.
        n_mappings: the fit mappings, by experiment and key.
        n_studies: the studies, the experiments of the mappings.
        outliers_dropped: the outlier points a validation dropped.
        limitation_subtracted: the limitation points subtracted.
        metrics: `criteria_metrics` and the values of the acceptance rules of
            the context, `NaN` if not evaluable.
    """

    context: str
    name: str
    role: EvidenceRole
    description: str
    selection: str
    n_points: int
    n_mappings: int
    n_studies: int
    outliers_dropped: int
    limitation_subtracted: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Get the result as plain data, `NaN` as `None`."""
        return {
            "context": self.context,
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "selection": self.selection,
            "n_points": self.n_points,
            "n_mappings": self.n_mappings,
            "n_studies": self.n_studies,
            "outliers_dropped": self.outliers_dropped,
            "limitation_subtracted": self.limitation_subtracted,
            "metrics": {key: _write(value) for key, value in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the result from plain data, `None` as `NaN`."""
        return cls(
            context=data["context"],
            name=data["name"],
            role=EvidenceRole(data["role"]),
            description=data["description"],
            selection=data["selection"],
            n_points=data["n_points"],
            n_mappings=data["n_mappings"],
            n_studies=data["n_studies"],
            outliers_dropped=data["outliers_dropped"],
            limitation_subtracted=data["limitation_subtracted"],
            metrics={key: _read(value) for key, value in data["metrics"].items()},
        )


@dataclass(frozen=True, kw_only=True)
class CoverageEntry:
    """The coverage of a value of a field of the metadata by a context of use.

    Attributes:
        context: the id of the context of use.
        field: the field of the metadata, e.g. `health`.
        value: the name of the value, e.g. `RENAL_IMPAIRMENT`.
        coverage: the coverage of a claimed value, `None` for a value outside
            the claimed domain.
        claimed: whether the domain claims the value; the values of the points
            of the evidence outside the claim are not claimed.
        n_points: the points with the value: the applicable points of the
            validation evidence for a validated value, of the calibration
            evidence for a calibrated one, every point of the evidence for a
            value outside the claim.
    """

    context: str
    field: str
    value: str
    coverage: Coverage | None
    claimed: bool
    n_points: int

    def to_dict(self) -> dict[str, object]:
        """Get the entry as plain data."""
        return {
            "context": self.context,
            "field": self.field,
            "value": self.value,
            "coverage": None if self.coverage is None else self.coverage.value,
            "claimed": self.claimed,
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the entry from plain data."""
        coverage = data["coverage"]
        return cls(
            context=data["context"],
            field=data["field"],
            value=data["value"],
            coverage=None if coverage is None else Coverage(coverage),
            claimed=data["claimed"],
            n_points=data["n_points"],
        )


@dataclass(frozen=True, kw_only=True)
class ContextResult:
    """The assessment of a context of use.

    Attributes:
        id: the id of the context.
        title: the title.
        question: the question.
        level: the level on the ladder.
        influence: the influence of the model on the decision.
        consequence: the consequence of a wrong decision.
        risk: the model risk.
        rationale: why this level, influence and consequence.
        extrapolation: the argument for a domain without validation evidence.
        domain_statement: what the domain claims beyond the metadata.
        quantities: the quantities of interest (`id`, `description`, `ratio`,
            `coverage`, `n_points`, see `QuantityRecord`).
        acceptance: the results of the acceptance rules (`name`, `value`,
            `limit`, `met`, see `AcceptanceRecord`), the rules of the context
            followed by the rules of the extensions, each with the name of its
            extension (`extension`); a record of pkpdsim 0.19 has its rule of
            the ratio pairs as the last row, of the extension `ratios`.
        verdict: the verdict.
        factors: the factors of V&V 40 (`id`, `name`, `status`, `statement`,
            `step`), in the order of `FACTORS`.
        n_decisive: the points of the decisive evidence, each row once.
        n_evaluable: the decisive points which are applicable, see
            `applicable`.
        minimum_points: the evaluable points the verdict needs, see
            `ContextOfUse.minimum_points`; 0 in a record of pkpdsim 0.17.
        verification: the checks of the verification of the decisive
            predictions, the checks of the predictions followed by the checks
            of the battery; empty if the assessment was not verified.
        uncertainty: the parameter uncertainty of the decisive predictions,
            see `sbmlsim.credibility.uncertainty`; `None` for an assessment
            without the propagation and a record of pkpdsim 0.18 or earlier.
        variability: the predictive check of the decisive points with the
            declared variability of the subjects, see
            `sbmlsim.credibility.variability`; `None` without it and in a
            record of pkpdsim 0.18 or earlier.
        extensions: the sections of the evidence extensions by their name,
            plain data which the extension of that name reads, see
            `sbmlsim.credibility.extensions`.
    """

    id: str
    title: str
    question: str
    level: CouLevel
    influence: Level
    consequence: Level
    risk: Level
    rationale: str
    extrapolation: str
    domain_statement: str
    quantities: list[QuantityRecord]
    acceptance: list[AcceptanceRecord]
    verdict: Verdict
    factors: list[FactorRecord]
    n_decisive: int
    n_evaluable: int
    minimum_points: int
    verification: list[CheckRecord] = dataclasses.field(default_factory=list)
    uncertainty: UncertaintyRecord | None = None
    variability: VariabilityRecord | None = None
    extensions: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def verified(self) -> bool | None:
        """Get whether every check passed, `None` if not verified."""
        if not self.verification:
            return None
        return all(record["passed"] for record in self.verification)

    @property
    def decisive_role(self) -> EvidenceRole:
        """Get the role of the evidence which decides, see `CouLevel.decisive_role`."""
        return self.level.decisive_role

    @property
    def uncovered_quantities(self) -> list[str]:
        """Get the ids of the quantities of interest which are uncovered."""
        return [q["id"] for q in self.quantities if q["coverage"] == UNCOVERED]

    def section(self, name: str) -> Any:  # plain data of an extension
        """Get the section of an extension, `None` without one.

        Args:
            name: the name of the extension.

        Returns:
            The section as plain data.
        """
        return self.extensions.get(name)

    def to_dict(self) -> dict[str, object]:
        """Get the result as plain data, `NaN` as `None`."""
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "level": self.level.value,
            "influence": self.influence.value,
            "consequence": self.consequence.value,
            "risk": self.risk.value,
            "rationale": self.rationale,
            "extrapolation": self.extrapolation,
            "domain_statement": self.domain_statement,
            "quantities": [dict(q) for q in self.quantities],
            "acceptance": [{**a, "value": _write(a["value"])} for a in self.acceptance],
            "verdict": self.verdict.value,
            "factors": [dict(f) for f in self.factors],
            "n_decisive": self.n_decisive,
            "n_evaluable": self.n_evaluable,
            "minimum_points": self.minimum_points,
            "verification": [check_record_to_dict(r) for r in self.verification],
            "uncertainty": (
                None
                if self.uncertainty is None
                else json.loads(json.dumps(self.uncertainty))
            ),
            "variability": (
                None
                if self.variability is None
                else json.loads(json.dumps(self.variability))
            ),
            EXTENSIONS: dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the result from plain data, `None` as `NaN`.

        A record of pkpdsim 0.17 has no minimum of points, read as 0, and no
        coverage of the quantities, read as `None` with 0 points; a record of
        0.18 or earlier has no verification, no uncertainty and no predictive
        check; a record of 0.19 has the evidence of its extensions as fields
        of its own, read as their sections, and the rule of its ratio pairs as
        a field, read as a row of the acceptance, see
        `sbmlsim.credibility.extensions.read_sections` and `read_acceptance`.
        """
        acceptance: list[AcceptanceRecord] = []
        for row in read_acceptance(data):
            record = AcceptanceRecord(
                name=row["name"],
                value=_read(row["value"]),
                limit=row["limit"],
                met=row["met"],
            )
            if ACCEPTANCE_EXTENSION in row:
                record[ACCEPTANCE_EXTENSION] = str(row[ACCEPTANCE_EXTENSION])
            acceptance.append(record)
        return cls(
            id=data["id"],
            title=data["title"],
            question=data["question"],
            level=CouLevel(data["level"]),
            influence=Level(data["influence"]),
            consequence=Level(data["consequence"]),
            risk=Level(data["risk"]),
            rationale=data["rationale"],
            extrapolation=data["extrapolation"],
            domain_statement=data["domain_statement"],
            quantities=[
                QuantityRecord(
                    id=q["id"],
                    description=q["description"],
                    ratio=q["ratio"],
                    coverage=q.get("coverage"),
                    n_points=q.get("n_points", 0),
                )
                for q in data["quantities"]
            ],
            acceptance=acceptance,
            verdict=Verdict(data["verdict"]),
            factors=[
                FactorRecord(
                    id=f["id"],
                    name=f["name"],
                    status=f["status"],
                    statement=f["statement"],
                    step=f["step"],
                )
                for f in data["factors"]
            ],
            n_decisive=data["n_decisive"],
            n_evaluable=data["n_evaluable"],
            minimum_points=data.get("minimum_points", 0),
            verification=[
                check_record_from_dict(r) for r in data.get("verification", ())
            ],
            uncertainty=uncertainty_record_from_dict(data.get("uncertainty")),
            variability=variability_record_from_dict(data.get("variability")),
            extensions=read_sections(data, LEGACY_CONTEXT_SECTIONS),
        )


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """What an assessment was computed from.

    Attributes:
        model: the name of the project of the model.
        title: the title of the model.
        fits: the ids of the adopted fits.
        versions: the versions of the packages, see `VERSIONED_PACKAGES`.
        commit: the commit of the repository of the model, `None` outside one.
        dirty: whether the repository has changes, `None` outside one.
        date: the date of the assessment, ISO 8601 with the time zone.
        lock: whether the evidence matched the lock of the declaration
            (`matched`), the lock was written with the assessment (`written`),
            or `None` without a lock, see `sbmlsim.credibility.lock`.
    """

    model: str
    title: str
    fits: list[str]
    versions: dict[str, str]
    commit: str | None
    dirty: bool | None
    date: str
    lock: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Get the provenance as plain data."""
        return {
            "model": self.model,
            "title": self.title,
            "fits": list(self.fits),
            "versions": dict(self.versions),
            "commit": self.commit,
            "dirty": self.dirty,
            "date": self.date,
            "lock": self.lock,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the provenance from plain data."""
        return cls(
            model=data["model"],
            title=data["title"],
            fits=list(data["fits"]),
            versions=dict(data["versions"]),
            commit=data["commit"],
            dirty=data["dirty"],
            date=data["date"],
            # a record of pkpdsim 0.18 or earlier knew no lock
            lock=data.get("lock"),
        )


@dataclass(frozen=True, kw_only=True)
class CredibilityAssessment:
    """The assessment of the contexts of use of a model.

    Attributes:
        provenance: what the assessment was computed from.
        contexts: the result of every context of use, in the declared order.
        evidence: the metrics of every evidence of every context.
        coverage: the coverage of the domain of every context.
        limitations: the metrics of the limitations, of the model (context
            `""`) and of the contexts; they enter no acceptance.
        unclaimed: the validation points which no evidence and no limitation
            selects, per fit and mapping (`fit`, `experiment`, `mapping`, `n`).
        notes: the notes of the declaration.
        verification: what the verification compared, `None` if the
            assessment was not verified.
        cross_validations: the leave-one-study-out cross-validations of the
            adopted fits, see `sbmlsim.credibility.cross_validation`; empty
            without one and in a record of pkpdsim 0.18 or earlier.
        vpc: every data point of the adopted fits with the band of its
            predictive check, see `sbmlsim.credibility.variability`; empty
            without the check and in a record of pkpdsim 0.18 or earlier.
        extensions: the sections of the evidence extensions in the
            assessment by their name, plain data, e.g. the provenance of the
            data of every study; a record of pkpdsim 0.19 has them as fields of
            its own, read as their sections.
    """

    provenance: Provenance
    contexts: list[ContextResult]
    evidence: list[EvidenceResult]
    coverage: list[CoverageEntry]
    limitations: list[EvidenceResult]
    unclaimed: list[UnclaimedRecord]
    notes: str
    verification: VerificationSummary | None = None
    cross_validations: list[CrossValidation] = dataclasses.field(default_factory=list)
    vpc: list[VpcPoint] = dataclasses.field(default_factory=list)
    extensions: dict[str, Any] = dataclasses.field(default_factory=dict)

    def context(self, context_id: str) -> ContextResult:
        """Get the result of a context of use.

        Args:
            context_id: the id of the context.

        Returns:
            The result.

        Raises:
            KeyError: for an unknown context.
        """
        for result in self.contexts:
            if result.id == context_id:
                return result
        raise KeyError(
            f"The assessment has no context of use '{context_id}', its contexts "
            f"are {[c.id for c in self.contexts]}."
        )

    def to_dict(self) -> dict[str, object]:
        """Get the assessment as plain data, `NaN` as `None`."""
        return {
            "provenance": self.provenance.to_dict(),
            "contexts": [c.to_dict() for c in self.contexts],
            "evidence": [e.to_dict() for e in self.evidence],
            "coverage": [c.to_dict() for c in self.coverage],
            "limitations": [e.to_dict() for e in self.limitations],
            "unclaimed": [dict(u) for u in self.unclaimed],
            "notes": self.notes,
            "verification": (
                None if self.verification is None else self.verification.to_dict()
            ),
            "cross_validations": [c.to_dict() for c in self.cross_validations],
            "vpc": json.loads(json.dumps(self.vpc)),
            EXTENSIONS: dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the assessment from plain data, `None` as `NaN`."""
        return cls(
            provenance=Provenance.from_dict(data["provenance"]),
            contexts=[ContextResult.from_dict(c) for c in data["contexts"]],
            evidence=[EvidenceResult.from_dict(e) for e in data["evidence"]],
            coverage=[CoverageEntry.from_dict(c) for c in data["coverage"]],
            limitations=[EvidenceResult.from_dict(e) for e in data["limitations"]],
            unclaimed=[
                UnclaimedRecord(
                    fit=u["fit"],
                    experiment=u["experiment"],
                    mapping=u["mapping"],
                    n=u["n"],
                )
                for u in data["unclaimed"]
            ],
            notes=data["notes"],
            verification=(
                None
                if data.get("verification") is None
                else VerificationSummary.from_dict(data["verification"])
            ),
            cross_validations=[
                CrossValidation.from_dict(c) for c in data.get("cross_validations", ())
            ],
            vpc=[VpcPoint(**point) for point in data.get("vpc", ())],
            extensions=read_sections(data, LEGACY_ASSESSMENT_SECTIONS),
        )

    def to_json(self, path: Path) -> None:
        """Write the assessment as JSON.

        Args:
            path: the file.
        """
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> CredibilityAssessment:
        """Read an assessment written with `to_json`, or by pkpdsim 0.17 to 0.19.

        Args:
            path: the file.

        Returns:
            The assessment.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _names(values: Iterable[object]) -> str:
    """Show values sorted, the members of an enum by their name."""
    return ", ".join(sorted(v.name if isinstance(v, Enum) else str(v) for v in values))


def describe_selection(selection: Selection) -> str:
    """Describe what a selection of an evidence selects.

    Args:
        selection: the selection.

    Returns:
        The mapping keys by study for a `NamedFilter`, e.g.
        `CredA: fm_dose; CredB: fm_dose, fm_zero`; the clauses of a
        `MetadataFilter`, e.g. `route in [IV, PO] and health not in [HEALTHY]`;
        `unclaimed validation data` for `Unclaimed`, followed by `within` and
        the description of its filter; the qualified name otherwise.
    """
    if isinstance(selection, NamedFilter):
        return "; ".join(
            f"{study}: {_names(keys)}" for study, keys in sorted(selection.mappings)
        )
    if isinstance(selection, MetadataFilter):
        clauses = [f"{name} in [{_names(v)}]" for name, v in selection.allowed]
        clauses += [f"{name} not in [{_names(v)}]" for name, v in selection.excluded]
        return " and ".join(clauses)
    if isinstance(selection, Unclaimed):
        text = "unclaimed validation data"
        if selection.within is not None:
            text += f" within {describe_selection(selection.within)}"
        return text
    return getattr(selection, "__qualname__", type(selection).__qualname__)


def _points(
    table: pd.DataFrame, selections: Iterable[EvidenceSelection]
) -> pd.DataFrame:
    """Get the rows of the union of selections, each row once, in the order of the table."""
    rows = list(chain.from_iterable(s.rows for s in selections))
    return table.loc[table.index.isin(rows)]


def applicable(points: pd.DataFrame) -> pd.DataFrame:
    """Get the points the fold criteria can evaluate.

    The predicate of the ratio criteria of sbmlsim (`Fold`, GMFE, AFE): a
    positive and finite value (`DV`) and prediction (`IPRED`).

    Args:
        points: the data points.

    Returns:
        The applicable points.
    """
    accepted = Fold(2.0).accepted(points).to_numpy()
    return points.loc[np.isfinite(accepted)]


def _metrics(points: pd.DataFrame, rules: Sequence[AcceptanceRule]) -> dict[str, float]:
    """Get `criteria_metrics` and the values of the rules on points."""
    metrics = criteria_metrics(points)
    for rule in rules:
        metrics[rule.name] = rule.evaluate(points).value
    return metrics


def _counts(points: pd.DataFrame) -> tuple[int, int, int]:
    """Get the points, mappings (by experiment and key) and studies."""
    mappings = points[["experiment", "mapping"]].drop_duplicates()
    return len(points), len(mappings), int(points["experiment"].nunique())


def _evidence_result(
    selection: EvidenceSelection, table: pd.DataFrame, rules: Sequence[AcceptanceRule]
) -> EvidenceResult:
    """Get the metrics of the selection of an evidence."""
    points = _points(table, (selection,))
    n_points, n_mappings, n_studies = _counts(points)
    evidence = selection.evidence
    return EvidenceResult(
        context=selection.context,
        name=evidence.name,
        role=evidence.role,
        description=evidence.description,
        selection=describe_selection(evidence.selection),
        n_points=n_points,
        n_mappings=n_mappings,
        n_studies=n_studies,
        outliers_dropped=selection.outliers_dropped,
        limitation_subtracted=selection.limitation_subtracted,
        metrics=_metrics(points, rules),
    )


def _coverage(
    cou: ContextOfUse, table: pd.DataFrame, selections: Sequence[EvidenceSelection]
) -> list[CoverageEntry]:
    """Get the coverage of the claimed values and the values outside the claim."""
    validation = applicable(
        _points(
            table,
            (s for s in selections if s.evidence.role is EvidenceRole.VALIDATION),
        )
    )
    calibration = applicable(
        _points(
            table,
            (s for s in selections if s.evidence.role is EvidenceRole.CALIBRATION),
        )
    )
    evidence = _points(table, selections)
    entries: list[CoverageEntry] = []
    for field, claimed in cou.domain.metadata.items():
        names = [value.name for value in claimed]
        validated = metadata_values(validation, field).value_counts()
        calibrated = metadata_values(calibration, field).value_counts()
        for name in names:
            if validated.get(name, 0):
                coverage, n = Coverage.VALIDATED, int(validated[name])
            elif calibrated.get(name, 0):
                coverage, n = Coverage.CALIBRATED, int(calibrated[name])
            else:
                coverage, n = Coverage.UNCOVERED, 0
            entries.append(
                CoverageEntry(
                    context=cou.id,
                    field=field,
                    value=name,
                    coverage=coverage,
                    claimed=True,
                    n_points=n,
                )
            )
        outside = metadata_values(evidence, field).value_counts()
        entries.extend(
            CoverageEntry(
                context=cou.id,
                field=field,
                value=str(value),
                coverage=None,
                claimed=False,
                n_points=int(outside[value]),
            )
            for value in sorted(outside.index)
            # a point without the field reports no value
            if value and value not in names
        )
    return entries


def _quantities(
    cou: ContextOfUse, decisive: pd.DataFrame, derived: Mapping[str, int]
) -> list[QuantityRecord]:
    """Get the quantities of interest with their coverage by the decisive points.

    A quantity whose points an extension derives, a ratio with a pairing, is
    covered by them; a declared ratio without a pairing has no coverage.
    """
    inside = in_domain(cou, applicable(decisive))
    observables = inside["observable"].astype(str).map(normalize_observable)
    records: list[QuantityRecord] = []
    for q in cou.quantities:
        if q.id in derived or q.pairing is not None:
            n = int(derived.get(q.id, 0))
            coverage: str | None = COVERED if n else UNCOVERED
        elif q.ratio:
            coverage, n = None, 0
        else:
            n = int((observables == normalize_observable(q.id)).sum())
            coverage = COVERED if n else UNCOVERED
        records.append(
            QuantityRecord(
                id=q.id,
                description=q.description,
                ratio=bool(q.ratio),
                coverage=coverage,
                n_points=n,
            )
        )
    return records


def _covered(cou: ContextOfUse, entry: CoverageEntry) -> bool:
    """Check whether a claimed value is covered: validated, for C1 also calibrated."""
    if entry.coverage is Coverage.VALIDATED:
        return True
    return cou.level is CouLevel.C1 and entry.coverage is Coverage.CALIBRATED


def verdict(
    n_decisive: int,
    acceptance: Sequence[AcceptanceResult],
    covered: bool,
    extrapolation: str,
    *,
    level: CouLevel,
    n_evaluable: int,
    minimum: int,
    verified: bool | None = None,
) -> Verdict:
    """Get the verdict of a context of use.

    In this order: no decisive point is insufficient evidence; a failed check
    of the verification is not verified, since the comparison of the
    predictions with the data is not to be trusted; a rule which is not met is
    not supported, also on few points; a rule which is not evaluable
    is insufficient evidence; fewer evaluable points than the minimum is
    insufficient evidence; every claimed value and every quantity with a
    coverage covered is supported, for C4 supported with extrapolation (a C4
    context extrapolates without data of its condition by definition, its
    extrapolation argument is required); an extrapolation argument is
    supported with extrapolation; else insufficient evidence.

    Args:
        n_decisive: the points of the decisive evidence.
        acceptance: the results of the acceptance rules, of the context and of
            the extensions.
        covered: whether every claimed value and every quantity with a
            coverage is covered.
        extrapolation: the extrapolation argument of the context.
        level: the level of the context.
        n_evaluable: the decisive points which are applicable.
        minimum: the evaluable points the verdict needs.
        verified: whether every check of the verification passed, `None` if
            the assessment is not verified.

    Returns:
        The verdict.
    """
    if n_decisive == 0:
        return Verdict.INSUFFICIENT_EVIDENCE
    if verified is False:
        return Verdict.NOT_VERIFIED
    if any(result.met is False for result in acceptance):
        return Verdict.NOT_SUPPORTED
    if any(result.met is None for result in acceptance):
        return Verdict.INSUFFICIENT_EVIDENCE
    if n_evaluable < minimum:
        return Verdict.INSUFFICIENT_EVIDENCE
    if covered and level is not CouLevel.C4:
        return Verdict.SUPPORTED
    if extrapolation:
        return Verdict.SUPPORTED_WITH_EXTRAPOLATION
    return Verdict.INSUFFICIENT_EVIDENCE


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """Get a number of things in words, e.g. `1 point` and `2 points`."""
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def _check_statement(record: CheckRecord, tightening: float) -> str:
    """Get the statement of the factor of a check of the verification."""
    result = "passed" if record["passed"] else "failed"
    if record["check"] not in {c.id for c in PREDICTION_CHECKS}:
        label = check_of(record["check"], record["factor"]).label
        text = f"{label[0].upper()}{label[1:]} of {_count(record['n_points'], 'case')}"
        if not math.isnan(record["max_difference"]):
            text += f", largest deviation {record['max_difference']:.2g}"
            if record["at"]:
                text += f" at {record['at']}"
        problems = "".join(f"; {problem}" for problem in record["problems"])
        return f"{text}{problems}: {result}."
    if record["check"] == SETUP.id:
        text = (
            "The variants are valid SBML with consistent units and the "
            f"{_count(record['n_points'], 'decisive prediction')} are finite"
            if not record["problems"]
            else "; ".join(record["problems"])
        )
        return f"{text}: {result}."
    compared = (
        f"the tolerances of the fits divided by {tightening:g}"
        if record["check"] == NUMERICAL_CHECKS[0].id
        else "the BDF of CVODE of at most order 2 at the tightened tolerances"
    )
    if record["n_points"] == 0:
        difference = "no decisive prediction to compare"
    else:
        difference = (
            f"largest relative difference {record['max_difference']:.2g} of "
            f"{_count(record['n_points'], 'decisive prediction')} with {compared}, "
            f"at {record['at']}, acceptance "
            f"{'changed' if record['acceptance_changed'] else 'unchanged'}"
        )
    problems = "".join(f"; {problem}" for problem in record["problems"])
    return f"{difference[0].upper()}{difference[1:]}{problems}: {result}."


def _factors(
    cou: ContextOfUse,
    decisive: pd.DataFrame,
    acceptance: Sequence[AcceptanceResult],
    coverage: Sequence[CoverageEntry],
    quantities: Sequence[QuantityRecord],
    checks: Sequence[CheckRecord] = (),
    tightening: float = TIGHTENING,
    cross_validations: Sequence[CrossValidation] = (),
    uncertainty: UncertaintyRecord | None = None,
    variability: VariabilityRecord | None = None,
) -> list[FactorRecord]:
    """Get the factors of V&V 40 of a context as the core assesses them.

    The factors of the checks of the verification are assessed when the
    assessment was verified, else they keep their status of `FACTORS`; the
    equivalency of the inputs with the predictive check, the model inputs
    with the parameter uncertainty and the model form when an adopted fit of
    the decisive evidence is cross-validated. The output comparison counts
    the rules of the context and of the extensions.
    """
    role = cou.decisive_role.value
    n_points, n_mappings, n_studies = _counts(decisive)
    claimed = [e for e in coverage if e.claimed]
    outside = [e for e in coverage if not e.claimed]
    met = sum(result.met is True for result in acceptance)
    evaluable = sum(result.met is not None for result in acceptance)
    covered = sum(_covered(cou, e) for e in claimed)
    absolute = [q for q in quantities if not q["ratio"]]
    covered_quantities = sum(q["coverage"] == COVERED for q in absolute)
    statements = {
        "test_samples": (
            f"{_count(n_points, 'point')} of {_count(n_mappings, 'mapping')} of "
            f"{_count(n_studies, 'study', 'studies')} of the {role} evidence "
            f"decide, {len(applicable(decisive))} of them evaluable "
            f"(minimum {cou.minimum_points})."
        ),
        "test_conditions": (
            f"{_count(len(claimed), 'claimed value')} of the metadata, "
            f"{_count(len(outside), 'value')} of the evidence outside the claim."
        ),
        "output_comparison": (
            f"{met} of {_count(len(acceptance), 'acceptance rule')} met on the "
            f"{role} evidence, {len(acceptance) - evaluable} not evaluable."
        ),
        "relevance_of_validation": (
            f"{covered} of {_count(len(claimed), 'claimed value')} and "
            f"{covered_quantities} of {_count(len(absolute), 'quantity', 'quantities')} "
            "of interest covered."
        ),
    }
    verified: dict[str, list[CheckRecord]] = {}
    for record in checks:
        verified.setdefault(record["factor"], []).append(record)
    factors: list[FactorRecord] = []
    for factor in FACTORS:
        data = factor.to_dict()
        if factor.status is FactorStatus.ASSESSED:
            data["statement"] = statements[factor.id]
        elif factor.id in verified:
            # a platform statement stays, the checks of the model follow it
            platform = (
                factor.statement if factor.status is FactorStatus.PLATFORM else ""
            )
            data["status"] = FactorStatus.ASSESSED.value
            data["statement"] = " ".join(
                [
                    *([platform] if platform else []),
                    *(_check_statement(r, tightening) for r in verified[factor.id]),
                ]
            )
            data["step"] = None
        elif factor.id == "equivalency_of_inputs" and variability is not None:
            data["status"] = FactorStatus.ASSESSED.value
            data["statement"] = variability_statement(variability)
            data["step"] = None
        elif factor.id == "model_inputs" and uncertainty is not None:
            data["status"] = FactorStatus.ASSESSED.value
            data["statement"] = _uncertainty_statement(uncertainty)
            data["step"] = None
        elif factor.id == "model_form" and cross_validations:
            data["status"] = FactorStatus.ASSESSED.value
            data["statement"] = " ".join(c.statement() for c in cross_validations)
            data["step"] = None
        factors.append(data)
    return factors


def _fold(value: float | None) -> str:
    """Get a band fold, a share or a probability in words."""
    return "n/a" if value is None else f"{value:.3g}"


def _percent(value: float | None) -> str:
    """Get a share in percent in words."""
    return "n/a" if value is None else f"{100 * value:.0f} %"


def _leading(indices: Sequence[SensitivityIndex], kind: str) -> str:
    """Get the parameter of the largest first-order index of every target."""
    best: dict[str, SensitivityIndex] = {}
    for i in indices:
        if i["kind"] != kind or i["index"] is None:
            continue
        current = best.get(i["target"])
        if current is None or i["index"] > (current["index"] or 0.0):
            best[i["target"]] = i
    return ", ".join(
        f"{target}: {i['fit']}.{i['parameter']} ({i['index']:.2f})"
        for target, i in best.items()
    )


def _uncertainty_statement(uncertainty: UncertaintyRecord) -> str:
    """Get the statement of the factor model inputs from the uncertainty."""
    fits = uncertainty["fits"]
    correlated = sum(f["correlated"] for f in fits)
    failed = sum(f["n_failed"] for f in fits)
    low, high = uncertainty["quantiles"]
    folds = [q["band_fold"] for q in uncertainty["quantities"] if q["band_fold"]]
    parts = [
        f"The parameter uncertainty of {_count(len(fits), 'adopted fit')} "
        f"({correlated} with the correlation of the Fisher information) "
        f"propagated with {uncertainty['n_draws']} draws"
        + (f" ({failed} failed)" if failed else "")
        + ".",
        f"The {100 * (high - low):g} % bands of the predictions have a median "
        f"fold width of up to {_fold(max(folds) if folds else None)} over the "
        f"quantities and hold {_percent(uncertainty['coverage'])} of the "
        f"{_count(uncertainty['n_points'], 'evaluable decisive observation')}.",
    ]
    leading = _leading(uncertainty["sensitivity"], "quantity")
    if leading:
        parts.append(f"The variance of the quantities is carried most by {leading}.")
    rules = uncertainty["rules"]
    if rules:
        decided = ", ".join(
            f"{r['name']} {r['limit']}: P = {_fold(r['estimate'])}, "
            f"{r['decision'].replace('_', ' ')}"
            for r in rules
        )
        parts.append(
            f"The acceptance with probability at least "
            f"{uncertainty['threshold']:g}: {decided}."
        )
    return " ".join(parts)


@dataclass(frozen=True)
class _Contribution:
    """The contribution of an extension to a context of use."""

    extension: EvidenceExtension
    contribution: ContextContribution


def _alternative_acceptance(
    cou: ContextOfUse,
    alternative: pd.DataFrame,
    lookup: MappingLookup,
    contributions: Sequence[_Contribution],
) -> list[AcceptanceResult]:
    """Evaluate the rules of a context and of its extensions on other predictions.

    Raises:
        ValueError: if an extension gives another number of results than its
            contribution.
    """
    results = [rule.evaluate(alternative) for rule in cou.rules]
    for c in contributions:
        if not c.contribution.acceptance:
            continue
        others = list(c.extension.acceptance(cou, alternative, lookup))
        if len(others) != len(c.contribution.acceptance):
            raise ValueError(
                f"The extension '{c.extension.name}' gives {len(others)} results "
                f"of its rules on other predictions of the context of use "
                f"'{cou.id}', not the {len(c.contribution.acceptance)} of its "
                "contribution."
            )
        results.extend(others)
    return results


def _uncertainty(
    cou: ContextOfUse,
    decisive: pd.DataFrame,
    quantities: Sequence[QuantityRecord],
    propagation: FitPropagation | None,
    lookup: MappingLookup,
    acceptance: Sequence[AcceptanceResult],
    contributions: Sequence[_Contribution],
) -> UncertaintyRecord | None:
    """Get the parameter uncertainty of the decisive predictions of a context.

    A draw on which a decisive prediction is not finite where the reference is
    finite failed for every rule; the bands are the quantiles of the finite
    predictions of a point. The rules are the ones of the context and of the
    extensions, `acceptance`.
    """
    if propagation is None:
        return None
    settings = propagation.settings
    predictions = propagation.predictions.loc[decisive.index]
    values = predictions.to_numpy(dtype=float)
    reference = np.isfinite(decisive["IPRED"].to_numpy(dtype=float))
    failed = (~np.isfinite(values) & reference[:, None]).any(axis=0)
    low_q, high_q = settings.quantiles
    finite = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        # a point without a finite prediction has no band
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanquantile(finite, low_q, axis=1) if len(finite) else finite[:, 0]
        high = np.nanquantile(finite, high_q, axis=1) if len(finite) else finite[:, 0]
    bands = pd.DataFrame({"low": low, "high": high}, index=decisive.index)
    evaluable = applicable(decisive)
    band = bands.loc[evaluable.index]
    has_band = (np.isfinite(band["low"]) & np.isfinite(band["high"])).to_numpy()
    banded, band = evaluable.loc[has_band], band.loc[has_band]
    dv = banded["DV"].to_numpy(dtype=float)
    inside = (dv >= band["low"].to_numpy()) & (dv <= band["high"].to_numpy())
    observables = banded["observable"].astype(str).map(normalize_observable)
    fits = sorted(set(decisive["fit"].astype(str))) if "fit" in decisive.columns else []
    drawn = {fit: propagation.draws[fit] for fit in fits if fit in propagation.draws}
    shifts = np.log(
        predictions.loc[banded.index].to_numpy(dtype=float)
        / banded["IPRED"].to_numpy(dtype=float)[:, None]
    )
    sensitivity: list[SensitivityIndex] = []
    bands_of: list[QuantityBand] = []
    for q in quantities:
        if q["ratio"]:
            continue
        mask = (observables == normalize_observable(q["id"])).to_numpy()
        if mask.any():
            with warnings.catch_warnings():
                # a failed draw has no shift
                warnings.simplefilter("ignore", RuntimeWarning)
                shift = np.mean(shifts[mask], axis=0)
            sensitivity.extend(sensitivity_indices(q["id"], "quantity", shift, drawn))
        lows, highs = band["low"].to_numpy()[mask], band["high"].to_numpy()[mask]
        positive = lows > 0
        bands_of.append(
            QuantityBand(
                quantity=q["id"],
                n_points=int(mask.sum()),
                band_fold=(
                    float(np.median(highs[positive] / lows[positive]))
                    if positive.any()
                    else None
                ),
                coverage=float(inside[mask].mean()) if mask.any() else None,
            )
        )
    rules: list[tuple[str, str, list[bool | None]]] = [
        (result.name, result.limit, []) for result in acceptance
    ]
    values_of: list[list[float]] = [[] for _ in rules]
    for k, column in enumerate(predictions.columns):
        if failed[k]:
            for (_, _, met), value in zip(rules, values_of, strict=True):
                met.append(None)
                value.append(math.nan)
            continue
        alternative = decisive.assign(IPRED=predictions[column])
        results = _alternative_acceptance(cou, alternative, lookup, contributions)
        for (_, _, met), value, result in zip(rules, values_of, results, strict=True):
            met.append(result.met)
            value.append(float(result.value))
    for (name, _, _), value in zip(rules, values_of, strict=True):
        sensitivity.extend(
            sensitivity_indices(name, "rule", np.array(value, dtype=float), drawn)
        )
    return UncertaintyRecord(
        n_draws=settings.draws,
        seed=settings.seed,
        quantiles=[low_q, high_q],
        threshold=settings.threshold,
        level=settings.level,
        fits=[
            FitUncertainty(
                fit=fit,
                fit_id=draws.fit_id,
                n_parameters=len(draws.pids),
                correlated=draws.correlated,
                note=draws.note,
                n_failed=int(propagation.failed.get(fit, 0)),
            )
            for fit, draws in drawn.items()
        ],
        n_points=len(banded),
        coverage=float(inside.mean()) if len(banded) else None,
        quantities=bands_of,
        rules=[
            rule_probability(name, limit, met, settings, len(rules))
            for name, limit, met in rules
        ],
        sensitivity=sensitivity,
    )


def _at(point: pd.Series) -> str:
    """Name a data point, `experiment.mapping at x` (without `x` if it has none)."""
    name = f"{point['experiment']}.{point['mapping']}"
    x = point.get("x")
    return name if x is None or pd.isna(x) else f"{name} at {float(x):g}"


def _check_records(
    cou: ContextOfUse,
    table: pd.DataFrame,
    decisive: pd.DataFrame,
    acceptance: Sequence[AcceptanceResult],
    verification: Verification,
    lookup: MappingLookup,
    contributions: Sequence[_Contribution],
) -> list[CheckRecord]:
    """Get the checks of the verification of the decisive predictions of a context.

    A numerical check passes when its evaluation of every fit of the decisive
    points succeeded, the largest relative difference of a decisive prediction
    is at most the limit of the verification and no rule of the acceptance
    (of the context and of the extensions) changes; the setup passes when the
    variants have no issues and every decisive prediction is finite.
    """
    summary = verification.summary
    fits = set(decisive["fit"].astype(str)) if "fit" in decisive.columns else set()
    records: list[CheckRecord] = []
    for check in NUMERICAL_CHECKS:
        predictions = verification.predictions[check]
        differences = relative_differences(table, predictions, summary.scale_floor)
        compared = differences.loc[decisive.index].dropna()
        alternative = decisive.assign(IPRED=predictions.loc[decisive.index])
        alternatives = _alternative_acceptance(cou, alternative, lookup, contributions)
        changed = any(
            other.met != result.met
            for other, result in zip(alternatives, acceptance, strict=True)
        )
        problems = [
            message
            for fit, message in verification.failures[check].items()
            if fit in fits
        ]
        largest = float(compared.max()) if len(compared) else math.nan
        records.append(
            CheckRecord(
                check=check.id,
                factor=check.factor,
                n_points=len(compared),
                max_difference=largest,
                at=_at(decisive.loc[compared.idxmax()]) if len(compared) else "",
                acceptance_changed=changed,
                passed=not problems and not changed and not (largest > summary.limit),
                problems=problems,
            )
        )
    ipred = decisive["IPRED"].to_numpy(dtype=float)
    failed = decisive.loc[~np.isfinite(ipred)]
    problems = list(summary.problems)
    if len(failed):
        mappings = failed[["experiment", "mapping"]].drop_duplicates()
        problems.append(
            f"{len(failed)} of {_count(len(decisive), 'decisive prediction')} "
            f"{'is' if len(failed) == 1 else 'are'} not finite, of "
            + ", ".join(f"{e}.{m}" for e, m in mappings.itertuples(index=False))
        )
    records.append(
        CheckRecord(
            check=SETUP.id,
            factor=SETUP.factor,
            n_points=len(decisive),
            max_difference=math.nan,
            at="",
            acceptance_changed=None,
            passed=not problems,
            problems=problems,
        )
    )
    if summary.battery is not None:
        records.extend(
            CheckRecord(
                check=result.check.id,
                factor=result.check.factor,
                n_points=result.cases,
                max_difference=result.largest,
                at=result.at,
                acceptance_changed=None,
                passed=bool(result.passed),
                problems=list(result.problems),
            )
            for result in summary.battery.results
            if result.passed is not None
        )
    return records


def _contributions(
    cou: ContextOfUse,
    decisive: pd.DataFrame,
    fits: FitTable,
    extensions: Sequence[EvidenceExtension],
    prepared: Mapping[str, object],
) -> list[_Contribution]:
    """Get the contributions of the extensions to a context of use, in their order.

    Raises:
        ValueError: if an extension assesses a factor which is no factor of
            V&V 40.
    """
    known = {factor.id for factor in FACTORS}
    contributions: list[_Contribution] = []
    for extension in extensions:
        contribution = extension.assess_context(
            cou, decisive, fits, prepared.get(extension.name)
        )
        if contribution is None:
            continue
        unknown = sorted(set(contribution.factors) - known)
        if unknown:
            raise ValueError(
                f"The extension '{extension.name}' assesses the factors {unknown}, "
                f"which are no factors of V&V 40: {sorted(known)}."
            )
        contributions.append(_Contribution(extension, contribution))
    return contributions


def _context_result(
    cou: ContextOfUse,
    fits: FitTable,
    selections: Sequence[EvidenceSelection],
    verification: Verification | None,
    cross_validations: Sequence[CrossValidation],
    propagation: FitPropagation | None,
    checks_of_variability: PredictiveChecks | None,
    extensions: Sequence[EvidenceExtension],
    prepared: Mapping[str, object],
) -> tuple[ContextResult, list[CoverageEntry]]:
    """Get the result and the coverage of a context of use."""
    table = fits.table
    decisive = _points(
        table, (s for s in selections if s.evidence.role is cou.decisive_role)
    )
    contributions = _contributions(cou, decisive, fits, extensions, prepared)
    own = [rule.evaluate(decisive) for rule in cou.rules]
    acceptance = [
        *own,
        *(r for c in contributions for r in c.contribution.acceptance),
    ]
    derived: dict[str, int] = {}
    for c in contributions:
        for quantity, n in c.contribution.quantities.items():
            derived[quantity] = derived.get(quantity, 0) + int(n)
    coverage = _coverage(cou, table, selections)
    quantities = _quantities(cou, decisive, derived)
    covered = all(_covered(cou, e) for e in coverage if e.claimed) and all(
        q["coverage"] == COVERED for q in quantities if q["coverage"] is not None
    )
    n_evaluable = len(applicable(decisive))
    checks = (
        []
        if verification is None
        else _check_records(
            cou,
            table,
            decisive,
            acceptance,
            verification,
            fits.lookup,
            contributions,
        )
    )
    verified = None if not checks else all(record["passed"] for record in checks)
    uncertainty = _uncertainty(
        cou, decisive, quantities, propagation, fits.lookup, acceptance, contributions
    )
    variability = (
        None
        if checks_of_variability is None
        else variability_record(
            checks_of_variability, decisive, applicable(decisive).index, cou.quantities
        )
    )
    fitted = set(decisive["fit"].astype(str)) if "fit" in decisive.columns else set()
    core = _factors(
        cou,
        decisive,
        acceptance,
        coverage,
        quantities,
        checks,
        TIGHTENING if verification is None else verification.summary.tightening,
        [c for c in cross_validations if c.subset in fitted],
        uncertainty,
        variability,
    )
    factors = [
        combine_factor(
            record,
            [
                c.contribution.factors[record["id"]]
                for c in contributions
                if record["id"] in c.contribution.factors
            ],
        )
        for record in core
    ]
    result = ContextResult(
        id=cou.id,
        title=cou.title,
        question=cou.question,
        level=cou.level,
        influence=cou.influence,
        consequence=cou.consequence,
        risk=cou.risk,
        rationale=cou.rationale,
        extrapolation=cou.extrapolation,
        domain_statement=cou.domain.statement,
        quantities=quantities,
        acceptance=[
            *(_acceptance_record(a) for a in own),
            *(
                _acceptance_record(a, c.extension.name)
                for c in contributions
                for a in c.contribution.acceptance
            ),
        ],
        verdict=verdict(
            len(decisive),
            acceptance,
            covered,
            cou.extrapolation,
            level=cou.level,
            n_evaluable=n_evaluable,
            minimum=cou.minimum_points,
            verified=verified,
        ),
        factors=factors,
        n_decisive=len(decisive),
        n_evaluable=n_evaluable,
        minimum_points=cou.minimum_points,
        verification=checks,
        uncertainty=uncertainty,
        variability=variability,
        extensions={
            c.extension.name: dict(c.contribution.section)
            for c in contributions
            if c.contribution.section is not None
        },
    )
    return result, coverage


def assess_table(
    credibility: Credibility,
    fits: FitTable,
    provenance: Provenance,
    verification: Verification | None = None,
    cross_validations: Sequence[CrossValidation] = (),
    propagation: FitPropagation | None = None,
    variability: PredictiveChecks | None = None,
    *,
    extensions: Sequence[EvidenceExtension] = (),
    prepared: Mapping[str, object] | None = None,
) -> CredibilityAssessment:
    """Assess the contexts of use of a declaration on the data points of the fits.

    Args:
        credibility: the declaration.
        fits: the data points of the adopted fits with `DV`, `IPRED` and
            `DV_SD`, see `evaluate_fits`.
        provenance: what the table was computed from.
        verification: the alternative predictions of the table and the issues
            of the variants, see `sbmlsim.credibility.verification.verify`;
            the assessment is not verified without.
        cross_validations: the cross-validations of the adopted fits, which
            assess the factor model form.
        propagation: the predictions at the draws of the parameters of the
            fits, see `sbmlsim.credibility.uncertainty.propagate_fits`; no
            context has an uncertainty record without.
        variability: the predictive checks of the fits with the declared
            variability, see `sbmlsim.credibility.variability`; no context
            has a predictive check without.
        extensions: the evidence extensions of the project, in their order.
        prepared: what every extension prepared (`EvidenceExtension.prepare`)
            by its name; `None` for an extension without.

    Returns:
        The assessment.

    Raises:
        SelectionError: if the evidence selects invalid data, see `select`.
    """
    prepared = {} if prepared is None else prepared
    selections = select(credibility, fits, extensions)
    table = fits.table
    evidence: list[EvidenceResult] = []
    contexts: list[ContextResult] = []
    coverage: list[CoverageEntry] = []
    for cou in credibility.contexts:
        own = [s for s in selections.evidence if s.context == cou.id]
        evidence.extend(_evidence_result(s, table, cou.rules) for s in own)
        result, entries = _context_result(
            cou,
            fits,
            own,
            verification,
            cross_validations,
            propagation,
            variability,
            extensions,
            prepared,
        )
        contexts.append(result)
        coverage.extend(entries)
    limitations = [
        _evidence_result(
            s, table, credibility.context(s.context).rules if s.context else ()
        )
        for s in selections.limitation_evidence
    ]
    unclaimed = [
        UnclaimedRecord(
            fit=str(row["fit"]),
            experiment=str(row["experiment"]),
            mapping=str(row["mapping"]),
            n=int(row["n"]),
        )
        for row in selections.unclaimed.to_dict("records")
    ]
    sections: dict[str, Any] = {}
    for extension in extensions:
        section = extension.assessment_section(prepared.get(extension.name))
        if section is not None:
            sections[extension.name] = dict(section)
    return CredibilityAssessment(
        provenance=provenance,
        contexts=contexts,
        evidence=evidence,
        coverage=coverage,
        limitations=limitations,
        unclaimed=unclaimed,
        notes=credibility.notes,
        verification=None if verification is None else verification.summary,
        cross_validations=list(cross_validations),
        vpc=[] if variability is None else _vpc_points(table, variability),
        extensions=sections,
    )


def _vpc_points(table: pd.DataFrame, checks: PredictiveChecks) -> list[VpcPoint]:
    """Get every point of the fits with the band of its predictive check."""
    keys = table.reindex(columns=["fit", "experiment", "mapping", "x", "DV"])
    points = keys.join(checks.points, how="inner")
    return [
        VpcPoint(
            fit=str(row["fit"]),
            experiment=str(row["experiment"]),
            mapping=str(row["mapping"]),
            x=_optional(row["x"]),
            DV=float(row["DV"]),
            N=int(row["N"]),
            statistic=str(row["statistic"]),
            PRED=float(row["PRED"]),
            low=_optional(row["low"]),
            median=_optional(row["median"]),
            high=_optional(row["high"]),
            inside=None if pd.isna(row["inside"]) else bool(row["inside"]),
        )
        for row in points[list(VPC_COLUMNS)].to_dict("records")
    ]


def _optional(value: object) -> float | None:
    """Get a finite number, `None` for a missing or not finite one."""
    number = float(value)  # ty: ignore[invalid-argument-type]
    return number if math.isfinite(number) else None


def _git(project: Project, *args: str) -> str | None:
    """Run git in the root of a project, `None` outside a repository."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _packages(project: Project) -> list[str]:
    """Get the distributions of the extensions and the battery hook of a project.

    A module which is no part of an installed distribution, e.g. of the tests,
    has none.
    """
    try:
        hook = project.battery_hook()
        objects = [*project.evidence_extensions(), *([] if hook is None else [hook])]
    except ValueError:
        return []
    distributions = importlib.metadata.packages_distributions()
    names: list[str] = []
    for obj in objects:
        top = type(obj).__module__.split(".")[0]
        names += [d for d in distributions.get(top, ()) if d not in names]
    return names


def provenance(
    project: Project, source_commit: str | None = None, lock: str | None = None
) -> Provenance:
    """Get the provenance of an assessment of a project now.

    Args:
        project: the project.
        source_commit: the commit of the source repository of the model, given
            by a release, which assesses the model in its release repository:
            the commit is recorded and whether its tree was clean is not known.
        lock: the state of the lock of the evidence, see `Provenance.lock`.

    Returns:
        The versions of `VERSIONED_PACKAGES` and of the distributions of the
        extensions and the battery hook of the project, the commit and
        whether the root of the project has changes
        (`git status --porcelain -- .` in `Project.root`; `None` outside a
        repository, the dirty state `None` with a `source_commit`), the
        adopted fits, the date with the local time zone and the state of the
        lock.
    """
    if source_commit is None:
        commit = _git(project, "rev-parse", "HEAD")
        status = (
            None
            if commit is None
            else _git(project, "status", "--porcelain", "--", ".")
        )
        commit = None if commit is None else commit.strip()
        dirty = None if status is None else bool(status.strip())
    else:
        commit, dirty = source_commit, None
    versions: dict[str, str] = {}
    for package in (*VERSIONED_PACKAGES, *_packages(project)):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return Provenance(
        model=project.name,
        title=project.title or project.name,
        fits=list(project.adopted_fits().values()),
        versions=versions,
        commit=commit,
        dirty=dirty,
        date=datetime.now().astimezone().isoformat(timespec="seconds"),
        lock=lock,
    )


def assess(
    project: Project,
    source_commit: str | None = None,
    *,
    verification: bool = True,
    battery: bool = True,
    lock: Path | None = None,
    uncertainty: PropagationSettings | None = None,
    variability: VariabilitySettings | None = None,
    cross_validations: Mapping[str, CrossValidation] | None = None,
) -> CredibilityAssessment:
    """Assess the contexts of use of a project on its adopted fits.

    The fits are evaluated fresh on the current model and data, see
    `evaluate_fits`, and, with `verification`, again with the alternatives of
    the checks of `sbmlsim.credibility.verification`; the cross-validations of
    the adopted fits are read, and every evidence extension of the project
    prepares what it reads once (`EvidenceExtension.prepare`).

    Args:
        project: the project with a declaration and adopted fits.
        source_commit: the commit of the source repository of a release, see
            `provenance`.
        verification: verify the decisive predictions; the factors of the
            verification are not assessed without.
        battery: run the battery of the checks of the variants with the
            verification, see `sbmlsim.credibility.battery`.
        lock: write the lock of the evidence (`sbmlsim.credibility.lock`) into
            this directory instead of comparing the evidence with the lock of
            the project; the lock of the package is never written.
        uncertainty: propagate the parameter uncertainty of the adopted fits
            to the decisive predictions with these settings, see
            `sbmlsim.credibility.uncertainty`; the factor model inputs is not
            assessed without.
        variability: check the decisive points with the declared variability
            of the subjects (`Credibility.variability`) with these settings,
            nested in the draws of the parameters with `uncertainty`, see
            `sbmlsim.credibility.variability`.
        cross_validations: the cross-validations of the adopted fits by their
            subset, the ones of the records of the project
            (`read_cross_validations`) without.

    Returns:
        The assessment.

    Raises:
        ValueError: if the project has no declaration or no adopted fits, or the
            settings of the variability are given for a declaration without a
            variability.
        KindMismatchError: if a kind differs from the recorded one.
        SelectionError: if the evidence selects invalid data.
        LockMismatchError: if the evidence differs from the lock.
    """
    from sbmlsim.credibility.cross_validation import read_cross_validations
    from sbmlsim.credibility.lock import (
        LockMismatchError,
        lock_differences,
        write_lock,
    )

    credibility = project.load_credibility()
    if not isinstance(credibility, Credibility):
        raise ValueError(
            f"The credibility of '{project.name}' is a "
            f"{type(credibility).__name__}, not a sbmlsim.credibility.Credibility."
        )
    if variability is not None and credibility.variability is None:
        raise ValueError(
            f"The credibility of '{project.name}' declares no variability of the "
            "subjects (`Credibility.variability`), which the predictive check needs."
        )
    extensions = project.evidence_extensions()
    fits = evaluate_fits(project)
    selections = select(credibility, fits, extensions)
    state: str | None
    if lock is not None:
        logger.info("Write the lock '%s'", write_lock(selections, fits, lock))
        state = "written"
    else:
        differences = lock_differences(project, selections, fits)
        if differences:
            raise LockMismatchError(differences)
        state = None if differences is None else "matched"
    prepared = {e.name: e.prepare(project, credibility, fits) for e in extensions}
    checks = verify(project, fits, battery=battery) if verification else None
    propagation = (
        None if uncertainty is None else propagate_fits(project, fits, uncertainty)
    )
    checks_of_variability = (
        None
        if variability is None or credibility.variability is None
        else predictive_checks(
            project, fits, credibility.variability, variability, propagation
        )
    )
    validations = (
        read_cross_validations(project)
        if cross_validations is None
        else cross_validations
    )
    logger.info("Assess the contexts of use of '%s'", project.name)
    return assess_table(
        credibility,
        fits,
        provenance(project, source_commit, state),
        checks,
        list(validations.values()),
        propagation,
        checks_of_variability,
        extensions=extensions,
        prepared=prepared,
    )
