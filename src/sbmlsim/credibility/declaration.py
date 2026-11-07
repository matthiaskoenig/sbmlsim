"""The declaration of the contexts of use of a model.

A `Credibility` lists the `ContextOfUse`s of a model. A context of use asks a
question of the model, names the quantities and the domain it is asked for,
the evidence (a selection of the fit mappings with a role) and, optionally,
the rules by which it is accepted. A context of use does not raise: it reports
`problems()`, and `Credibility` raises one `DeclarationError` with the
problems of every context.

The domain of a context claims values of the fields of the metadata of the
fit mappings, a dataclass derived from `sbmlsim.fit.objects.MappingMetaData`
whose fields are enums, e.g. the route of a dose. The metadata and the fields
a domain can claim belong to the project (`metadata_fields`,
`sbmlsim.project.CredibilitySettings`), so the declaration is checked against
them when the project loads it (`Credibility.domain_problems`,
`sbmlsim.project.Project.load_credibility`).
"""

from __future__ import annotations

import math
import re
import types
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, Union, get_args, get_type_hints

import pandas as pd

from sbmlsim.credibility.extensions import ExtensionSetting, Pairing
from sbmlsim.fit.evaluation import Criterion, afe, criteria_metrics, gmfe
from sbmlsim.fit.helpers import MappingFilter

if TYPE_CHECKING:
    from sbmlsim.fit.variability import Variability

__all__ = [
    "DEFAULT_COVARIATES",
    "UNCLAIMED",
    "Acceptance",
    "AcceptanceResult",
    "AcceptanceRule",
    "AfeRange",
    "Consequence",
    "ContextOfUse",
    "CouLevel",
    "Credibility",
    "DeclarationError",
    "Domain",
    "Evidence",
    "EvidenceRole",
    "Influence",
    "Level",
    "MaxGmfe",
    "QuantityOfInterest",
    "Risk",
    "Selection",
    "Unclaimed",
    "acceptance_result",
    "metadata_fields",
]

#: the covariates of the subjects whose simulated values an extension checks
#: against the reported ones, none in sbmlsim: the extension of a domain which
#: checks them (the inputs of pkpdsim) takes its own default for a declaration
#: without covariates
DEFAULT_COVARIATES: Final[Mapping[str, str]] = MappingProxyType({})

#: the pattern of the id of a context of use
ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")

_COU_LABELS: Final = {
    "C1": "describe",
    "C2": "interpolate",
    "C3": "extrapolate with data",
    "C4": "extrapolate without data",
    "C5": "clinical endpoint",
}


@unique
class CouLevel(StrEnum):
    """The level of a context of use on the ladder from describing to predicting."""

    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"

    @property
    def label(self) -> str:
        """Get the name of the level, e.g. `interpolate`."""
        return _COU_LABELS[self.value]

    @property
    def decisive_role(self) -> EvidenceRole:
        """Get the role of the evidence which decides: calibration for C1, else validation."""
        if self is CouLevel.C1:
            return EvidenceRole.CALIBRATION
        return EvidenceRole.VALIDATION


@unique
class Level(StrEnum):
    """A level of influence, consequence or risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


Influence = Level
Consequence = Level
Risk = Level


@unique
class EvidenceRole(StrEnum):
    """What evidence is used for."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"
    LIMITATION = "limitation"


class DeclarationError(ValueError):
    """The declaration of the credibility of a model is invalid.

    Attributes:
        problems: every problem of the declaration.
    """

    def __init__(self, problems: list[str]) -> None:
        """Create the error.

        Args:
            problems: every problem of the declaration.
        """
        self.problems = list(problems)
        super().__init__(
            "The credibility declaration is invalid:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
        )


@dataclass(frozen=True)
class Unclaimed:
    """The validation points which no other evidence and no limitation selects.

    Attributes:
        within: restricts the points to the ones this filter selects.
    """

    within: MappingFilter | None = None


UNCLAIMED: Final = Unclaimed()

type Selection = MappingFilter | Unclaimed


@dataclass(frozen=True)
class QuantityOfInterest:
    """A quantity a context of use is asked for.

    Attributes:
        id: the observable of the fit mappings, e.g. `Cve` or `pk.cmax`; for a
            ratio its name, it is matched against no observable.
        description: what the quantity is.
        ratio: whether the quantity is a ratio between arms: `True` is
            declared and recorded, not assessed; a `Pairing` derives the ratio
            from the decisive points and is assessed by the evidence extension
            which knows it (`sbmlsim.credibility.extensions`), its observable
            restricts the evidence like the id of an absolute quantity.
    """

    id: str
    description: str
    ratio: bool | Pairing = field(default=False, kw_only=True)

    @property
    def pairing(self) -> Pairing | None:
        """Get the pairing of an assessed ratio, `None` otherwise."""
        return None if isinstance(self.ratio, bool) else self.ratio


def _enum_of(hint: object) -> type[StrEnum] | None:
    """Get the enum of the type hint of a field, of an optional one as well."""
    if isinstance(hint, type) and issubclass(hint, StrEnum):
        return hint
    if isinstance(hint, types.UnionType) or getattr(hint, "__origin__", None) is Union:
        enums = [e for e in map(_enum_of, get_args(hint)) if e is not None]
        if len(enums) == 1:
            return enums[0]
    return None


def metadata_fields(
    metadata_type: type, fields: Collection[str] | None = None
) -> dict[str, type[StrEnum]]:
    """Get the fields of the metadata of the mappings which a domain can claim.

    Args:
        metadata_type: the metadata of the fit mappings of the project, a
            dataclass derived from `sbmlsim.fit.objects.MappingMetaData`.
        fields: the fields a domain can claim, every field whose type is an
            enum (`StrEnum`, optional or not) without.

    Returns:
        The enum of every field, in the order of `fields` or of the class.

    Raises:
        ValueError: if a field of `fields` is not a field of the metadata or
            its type is no enum.
    """
    hints = get_type_hints(metadata_type)
    if fields is None:
        found = {name: _enum_of(hint) for name, hint in hints.items()}
        return {name: enum for name, enum in found.items() if enum is not None}
    enums: dict[str, type[StrEnum]] = {}
    wrong: list[str] = []
    for name in fields:
        enum = _enum_of(hints.get(name))
        if enum is None:
            wrong.append(name)
        else:
            enums[name] = enum
    if wrong:
        raise ValueError(
            f"The domain fields {wrong} are no fields of {metadata_type.__name__} "
            "whose type is an enum."
        )
    return enums


@dataclass(frozen=True)
class Domain:
    """The domain a context of use claims.

    Attributes:
        metadata: the claimed values by field of the metadata of the fit
            mappings, see `metadata_fields`.
        statement: what is claimed beyond the metadata, e.g. the dose range.
    """

    metadata: Mapping[str, Collection[StrEnum]] = field(
        default_factory=dict, kw_only=True
    )
    statement: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        """Store the values of every field as a tuple."""
        object.__setattr__(
            self,
            "metadata",
            {name: tuple(values) for name, values in self.metadata.items()},
        )

    def problems(self, fields: Mapping[str, type[StrEnum]]) -> list[str]:
        """Get the problems of the domain: an unknown field or a value of another vocabulary.

        Args:
            fields: the fields a domain can claim with their enum, see
                `metadata_fields`.

        Returns:
            Every problem, empty if the domain claims values of the fields.
        """
        found: list[str] = []
        for name, values in self.metadata.items():
            enum = fields.get(name)
            if enum is None:
                found.append(
                    f"the domain field '{name}' is not a field of the metadata, "
                    f"use one of {', '.join(fields) or 'none, it has no fields'}"
                )
                continue
            found.extend(
                f"the domain value {value!r} of '{name}' is not a member of "
                f"{enum.__name__}"
                for value in values
                if not isinstance(value, enum)
            )
        return found


@dataclass(frozen=True)
class Evidence:
    """A selection of the data with a role.

    Attributes:
        name: the name of the evidence.
        role: what the evidence is used for.
        selection: the fit mappings it selects.
        description: what the evidence is.
        fit: the subset of the adopted fits it is taken from, `None` for all.
    """

    name: str
    role: EvidenceRole
    selection: Selection
    description: str = ""
    fit: str | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class AcceptanceResult:
    """The result of an acceptance rule.

    Attributes:
        name: the name of the rule.
        value: the value of the rule on the points, `NaN` if not evaluable.
        limit: the limit, e.g. `>= 80`.
        met: whether the rule is met, `None` if the value is `NaN`.
    """

    name: str
    value: float
    limit: str
    met: bool | None


class AcceptanceRule(Protocol):
    """A rule on the data points of the evidence of a context of use."""

    @property
    def name(self) -> str:
        """Get the name of the rule."""
        ...

    @property
    def limit(self) -> str:
        """Get the limit of the rule, e.g. `>= 80`."""
        ...

    def evaluate(self, points: pd.DataFrame) -> AcceptanceResult:
        """Evaluate the rule on the data points (`DV`, `IPRED`, `DV_SD`)."""
        ...


def acceptance_result(
    name: str, value: float, limit: str, met: bool
) -> AcceptanceResult:
    """Get the result of a rule, not evaluable for `NaN`.

    Args:
        name: the name of the rule.
        value: its value on the points.
        limit: its limit.
        met: whether the value meets the limit.

    Returns:
        The result, `met` `None` for a value `NaN`.
    """
    return AcceptanceResult(name, value, limit, None if math.isnan(value) else met)


@dataclass(frozen=True)
class Acceptance:
    """The percentage of the points a criterion accepts is at least `minimum`.

    Attributes:
        criterion: the criterion, e.g. `Fold(2.0)`.
        minimum: the minimum percentage (0 to 100).
    """

    criterion: Criterion
    minimum: float

    @property
    def name(self) -> str:
        """Get the name of the criterion."""
        return self.criterion.name

    @property
    def limit(self) -> str:
        """Get the limit, e.g. `>= 80`."""
        return f">= {self.minimum:g}"

    def evaluate(self, points: pd.DataFrame) -> AcceptanceResult:
        """Evaluate the percentage of the points the criterion accepts."""
        value = criteria_metrics(points, (self.criterion,))[self.criterion.name]
        return acceptance_result(self.name, value, self.limit, value >= self.minimum)


@dataclass(frozen=True)
class MaxGmfe:
    """The geometric mean fold error is at most `maximum`.

    Attributes:
        maximum: the largest accepted GMFE.
    """

    maximum: float

    @property
    def name(self) -> str:
        """Get the name of the rule."""
        return "GMFE"

    @property
    def limit(self) -> str:
        """Get the limit, e.g. `<= 1.5`."""
        return f"<= {self.maximum:g}"

    def evaluate(self, points: pd.DataFrame) -> AcceptanceResult:
        """Evaluate the GMFE of the points."""
        value = gmfe(points)
        return acceptance_result(self.name, value, self.limit, value <= self.maximum)


@dataclass(frozen=True)
class AfeRange:
    """The average fold error, the bias, is inside the range.

    Attributes:
        low: the smallest accepted AFE.
        high: the largest accepted AFE.
    """

    low: float
    high: float

    @property
    def name(self) -> str:
        """Get the name of the rule."""
        return "AFE"

    @property
    def limit(self) -> str:
        """Get the limit, e.g. `0.8 to 1.25`."""
        return f"{self.low:g} to {self.high:g}"

    def evaluate(self, points: pd.DataFrame) -> AcceptanceResult:
        """Evaluate the AFE of the points, the bounds included."""
        value = afe(points)
        return acceptance_result(
            self.name, value, self.limit, self.low <= value <= self.high
        )


@dataclass(frozen=True, kw_only=True)
class ContextOfUse:
    """What a model is used for: the question, the domain, the evidence and the acceptance.

    Attributes:
        id: the slug of the context, unique in the model.
        title: the title.
        question: the question of the context, one sentence.
        level: the level on the ladder.
        influence: the influence of the model on the decision.
        consequence: the consequence of a wrong decision.
        quantities: the quantities asked for, at least one which is no ratio.
        domain: the domain claimed, at least one value of the metadata above
            C1.
        evidence: the evidence.
        acceptance: the rules, empty for the default of the risk.
        limitations: the evidence of what the model does not describe.
        rationale: why this level, influence and consequence.
        extrapolation: the argument for a domain without validation evidence.
        min_points: the evaluable decisive points the verdict needs, at least
            1; `None` for the default of the risk, see `minimum_points`.
        extensions: the settings of the evidence extensions for this context
            by the name of the extension, e.g. the acceptance of the ratio
            pairs; a setting with a method `problems` is checked, see
            `sbmlsim.credibility.extensions`.
    """

    id: str
    title: str
    question: str
    level: CouLevel
    influence: Level
    consequence: Level
    quantities: tuple[QuantityOfInterest, ...]
    domain: Domain
    evidence: tuple[Evidence, ...]
    acceptance: tuple[AcceptanceRule, ...] = ()
    limitations: tuple[Evidence, ...] = ()
    rationale: str = ""
    extrapolation: str = ""
    min_points: int | None = None
    extensions: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Store the settings of the extensions read-only."""
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))

    @property
    def risk(self) -> Level:
        """Get the model risk of the influence and the consequence."""
        from sbmlsim.credibility.risk import model_risk

        return model_risk(self.influence, self.consequence)

    @property
    def rules(self) -> tuple[AcceptanceRule, ...]:
        """Get the acceptance rules: the declared ones or the default of the risk."""
        from sbmlsim.credibility.risk import default_acceptance

        return self.acceptance or default_acceptance(self.risk)

    @property
    def ratios(self) -> tuple[QuantityOfInterest, ...]:
        """Get the assessed ratio quantities, those with a `Pairing`."""
        return tuple(q for q in self.quantities if q.pairing is not None)

    @property
    def minimum_points(self) -> int:
        """Get the evaluable points the verdict needs: the declared or the default of the risk."""
        from sbmlsim.credibility.risk import DEFAULT_MIN_POINTS

        if self.min_points is not None:
            return self.min_points
        return DEFAULT_MIN_POINTS[self.risk]

    @property
    def decisive_role(self) -> EvidenceRole:
        """Get the role of the evidence which decides, see `CouLevel.decisive_role`."""
        return self.level.decisive_role

    def problems(self) -> list[str]:
        """Get the problems of the context of use, each naming the context.

        The domain is checked against the fields of the metadata by
        `domain_problems`.
        """
        who = f"context of use '{self.id}'"
        found: list[str] = []
        if not ID_PATTERN.match(self.id):
            found.append(f"{who}: the id must match {ID_PATTERN.pattern}")
        if not self.question.rstrip().endswith("?"):
            found.append(f"{who}: the question must be a question, ending with '?'")
        if not any(not q.ratio for q in self.quantities):
            # the absolute quantities decide the verdict, a ratio adds to it
            found.append(
                f"{who}: at least one quantity of interest which is no ratio "
                "is required"
            )
        if self.level is not CouLevel.C1 and not any(self.domain.metadata.values()):
            found.append(
                f"{who}: a {self.level} context claims at least one value of the "
                "metadata in its domain"
            )
        roles = {e.role for e in self.evidence}
        if self.level is CouLevel.C1:
            if EvidenceRole.CALIBRATION not in roles:
                found.append(f"{who}: a C1 context needs calibration evidence")
        elif self.level is CouLevel.C4:
            if not self.extrapolation:
                found.append(f"{who}: a C4 context needs an extrapolation argument")
        elif EvidenceRole.VALIDATION not in roles and not self.extrapolation:
            found.append(
                f"{who}: a {self.level} context needs validation evidence "
                "or an extrapolation argument"
            )
        if self.min_points is not None and self.min_points < 1:
            found.append(
                f"{who}: the minimum of points must be at least 1, not "
                f"{self.min_points}"
            )
        unclaimed = [e for e in self.evidence if isinstance(e.selection, Unclaimed)]
        if len(unclaimed) > 1:
            found.append(
                f"{who}: at most one unclaimed evidence, {len(unclaimed)} given"
            )
        found.extend(
            f"{who}: the unclaimed evidence '{e.name}' must have the role "
            f"validation, not {e.role}"
            for e in unclaimed
            if e.role is not EvidenceRole.VALIDATION
        )
        found.extend(
            f"{who}: the limitation '{e.name}' must have the role limitation, "
            f"not {e.role}"
            for e in self.limitations
            if e.role is not EvidenceRole.LIMITATION
        )
        for name, setting in self.extensions.items():
            if isinstance(setting, ExtensionSetting):
                found.extend(
                    f"{who}: the setting of the extension '{name}': {p}"
                    for p in setting.problems()
                )
        found.extend(
            f"{who}: the ratio '{q.id}' needs the observable of its arms"
            for q in self.ratios
            if q.pairing is not None and not q.pairing.observable.strip()
        )
        return found

    def domain_problems(self, fields: Mapping[str, type[StrEnum]]) -> list[str]:
        """Get the problems of the domain, each naming the context.

        Args:
            fields: the fields a domain can claim with their enum, see
                `metadata_fields`.

        Returns:
            Every problem of the domain.
        """
        return [
            f"context of use '{self.id}': {problem}"
            for problem in self.domain.problems(fields)
        ]


@dataclass(frozen=True, kw_only=True)
class Credibility:
    """The contexts of use of a model.

    Attributes:
        contexts: the contexts of use.
        limitations: the model limitations, evidence of what it does not describe.
        notes: notes on the declaration.
        covariates: the covariates of the subjects whose simulated values an
            evidence extension checks against the reported ones (the
            equivalency of the inputs), the target of the model and the
            measurement type of its reported value; none by default
            (`DEFAULT_COVARIATES`), an extension takes its own default then.
        variability: the variability between the subjects, relative
            distributions of targets (`sbmlsim.fit.variability.Variability`),
            the input of the predictive check of the decisive points, see
            `sbmlsim.credibility.variability`; `None` for a model which
            declares none.

    Raises:
        DeclarationError: with every problem of the declaration but the ones
            of the domains, which need the fields of the metadata of the
            project, see `domain_problems`.
    """

    contexts: tuple[ContextOfUse, ...]
    limitations: tuple[Evidence, ...] = ()
    notes: str = ""
    covariates: Mapping[str, str] = field(default=DEFAULT_COVARIATES, hash=False)
    variability: Variability | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        """Check the declaration.

        Raises:
            DeclarationError: with every problem of the declaration.
        """
        found: list[str] = []
        seen: set[str] = set()
        for cou in self.contexts:
            if cou.id in seen:
                found.append(f"context of use '{cou.id}': the id is not unique")
            seen.add(cou.id)
            found.extend(cou.problems())
        found.extend(
            f"model limitation '{e.name}' must have the role limitation, not {e.role}"
            for e in self.limitations
            if e.role is not EvidenceRole.LIMITATION
        )
        found.extend(
            f"covariate '{target}': the target and the measurement type must be "
            "texts which are not empty"
            for target, measurement in self.covariates.items()
            if not (
                isinstance(target, str)
                and target.strip()
                and isinstance(measurement, str)
                and measurement.strip()
            )
        )
        if self.variability is not None:
            from sbmlsim.fit.variability import Variability

            if not isinstance(self.variability, Variability):
                found.append(
                    "the variability must be a sbmlsim.fit.variability.Variability, "
                    f"not {type(self.variability).__name__}"
                )
        if found:
            raise DeclarationError(found)
        object.__setattr__(self, "covariates", MappingProxyType(dict(self.covariates)))

    def domain_problems(self, fields: Mapping[str, type[StrEnum]]) -> list[str]:
        """Get the problems of the domains of the contexts of use.

        Args:
            fields: the fields a domain can claim with their enum, see
                `metadata_fields`.

        Returns:
            Every problem of a domain, empty if every domain claims values of
            the fields.
        """
        return [p for cou in self.contexts for p in cou.domain_problems(fields)]

    def context(self, id: str) -> ContextOfUse:
        """Get a context of use by its id.

        Args:
            id: the id of the context.

        Returns:
            The context.

        Raises:
            KeyError: if there is no such context.
        """
        for cou in self.contexts:
            if cou.id == id:
                return cou
        raise KeyError(
            f"No context of use '{id}'; known: {[c.id for c in self.contexts]}"
        )
