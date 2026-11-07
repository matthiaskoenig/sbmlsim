"""The evidence extensions: evidence of the contexts of use which a project adds.

The core of the credibility assesses what every model has: the data points of
its adopted fits against the acceptance rules, the coverage of the claimed
domain, the verification, the parameter uncertainty and the variability. A
domain adds evidence of its own, e.g. the ratios between the arms of a study,
the equivalency of the simulated and the reported inputs of the subjects, the
chronology of the studies or the provenance of the study data. Such evidence
is an `EvidenceExtension`, which a project registers
(`sbmlsim.project.CredibilitySettings.extensions`) and the core calls at fixed
points. Every method has a default which contributes nothing, so an
extension overrides what it adds. The contract:

The declaration. A setting of an extension for one context of use, e.g. the
acceptance of the ratio pairs, is an entry of `ContextOfUse.extensions` by the
name of the extension; a setting with a method `problems` (`ExtensionSetting`)
is checked with the declaration. A ratio quantity carries a `Pairing`, which
the extension that knows it claims (`claims`). The project checks both when it
loads the declaration (`sbmlsim.project.Project.load_credibility`,
`sbmlsim.credibility.Credibility.extension_problems`): a setting by a name of
no extension of the project and a pairing which no extension claims are
problems, so a typo never falls back to a default silently.

The selection. `selection_problems` checks the decisive points of every
context of use after the selection of the evidence
(`sbmlsim.credibility.selection.select`), e.g. a study of a ratio with two
reference arms.

The assessment. `prepare` reads what the extension needs from the project
once per assessment, e.g. the chronology of the studies; `assess_context`
gives its `ContextContribution` to a context of use:

- `acceptance`: the results of the rules of the extension. They enter the
  verdict of the context and the factor output comparison together with the
  acceptance rules of the context, and they are rows of the acceptance of the
  record of the context: after the rows of the rules of the context, in the
  order of the extensions of the project, each with the name of its
  extension in the field `extension` (`ACCEPTANCE_EXTENSION`; a row of a rule
  of the context has none). `acceptance` evaluates the same rules on other
  predictions, those of a draw of the parameter uncertainty or of an
  alternative of the verification, in the same order.
- `quantities`: the evaluable points of the quantities the extension derives,
  e.g. the pairs of a ratio quantity; a quantity with a point is covered.
- `factors`: a `FactorStatement` per factor of V&V 40 the extension assesses,
  combined with the statement of the core by
  `sbmlsim.credibility.factors.combine_factor`: the statements before the
  core (`before_core`) in the order of the extensions, the statement of the
  core, the statements after it in the order of the extensions; a factor
  with a statement of an extension is assessed and has no step. The
  statement of the core is the one the core gives the factor (an assessed
  factor, a platform statement or the checks of the verification), none for a
  factor the core leaves not assessed.
- `section`: the section of the extension in the record of the context.

`assessment_section` is its section of the record of the assessment, e.g. the
provenance of the study data of every study.

The records. The record of a context of use and of an assessment has the
sections of the extensions by their name in its field `extensions`
(`EXTENSIONS`), which the extension of that name writes and reads; the core
keeps a section as it is. A record of pkpdsim 0.19 has the evidence of its
extensions as fields of its own, which `read_sections` reads as their
sections (`LEGACY_CONTEXT_SECTIONS`, `LEGACY_ASSESSMENT_SECTIONS`) and
`read_acceptance` its rule of the ratio pairs, `ratio_acceptance`, as a row of
the acceptance of the extension `ratios` (`LEGACY_ACCEPTANCE`); a record of
pkpdsim 0.17 or 0.18 has none.

The report. The rows of the acceptance of an extension are rows of the
acceptance table of its context, after the rows of the context, with the name
of the rule followed by the `label` of the extension. `context_report_blocks`
gives the blocks of the extension in the card of a context, each placed after
a block of the core (`ContextBlock.after`, one of `CONTEXT_BLOCKS`), the
blocks after one block of the core in the order of the extensions;
`report_section` a section of the report of its own with an entry in the
navigation, after the core sections and before the provenance; `report_tables`
the tables it writes next to the report.

The scorecard. An extension names its columns statically
(`scorecard_names`) and `scorecard_columns` gives their values for a context,
for every name of `scorecard_names` and no other; they follow the columns of
the core (the order of the columns of pkpdsim changes, the scorecards are
compared by the names of the columns). The rows of the acceptance of an
extension are columns of their rule like the rules of the context, which the
core compares as rules; the columns of `scorecard_names` it never compares
(`rule_columns`): `regressions` of the extension alone judges them, e.g. a
column of pairs which drops from 10 to 4 is no regression of a rule. The core
gives `regressions` the rows as they are: a scorecard of the process has the
values `scorecard_columns` gave, a scorecard read back from its TSV has texts
(an empty text for a missing value), and a baseline of an older record lacks
the columns; `scorecard_number` and `scorecard_true` read a value of either.
`pool` pools the sections of the contexts of several models with the options
of the pooling, which the caller passes through as it got them (e.g.
`{"field": "health"}` of the qualification of the ratio pairs by a value of
the metadata, an option of a command line), or empty for the defaults of the extension.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from sbmlsim.credibility.declaration import (
        AcceptanceResult,
        ContextOfUse,
        Credibility,
    )
    from sbmlsim.credibility.records import FitTable, MappingLookup
    from sbmlsim.project import Project

__all__ = [
    "ACCEPTANCE_EXTENSION",
    "CONTEXT_BLOCKS",
    "EXTENSIONS",
    "LEGACY_ACCEPTANCE",
    "LEGACY_ASSESSMENT_SECTIONS",
    "LEGACY_CONTEXT_SECTIONS",
    "ContextBlock",
    "ContextContribution",
    "EvidenceExtension",
    "ExtensionSetting",
    "FactorStatement",
    "Pairing",
    "ReportSection",
    "ScorecardChange",
    "read_acceptance",
    "read_sections",
    "rule_columns",
    "scorecard_columns",
    "scorecard_number",
    "scorecard_true",
]

#: the field of a record with the sections of the extensions by their name
EXTENSIONS: Final = "extensions"

#: the field of a row of the acceptance of a context with the name of the
#: extension of its rule, absent for a rule of the context
ACCEPTANCE_EXTENSION: Final = "extension"

#: the blocks of the card of a context of use in the report, in their order,
#: which the blocks of an extension follow
CONTEXT_BLOCKS: Final = (
    "acceptance",
    "uncertainty",
    "variability",
    "verification",
    "factors",
)

#: the fields of the record of a context of use of pkpdsim 0.19 which are the
#: sections of its extensions, by the name of the extension
LEGACY_CONTEXT_SECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"ratios": ("ratios",), "temporal": ("temporal",), "inputs": ("inputs",)}
)

#: the fields of the record of a context of use of pkpdsim 0.19 which are the
#: result of a rule of an extension, by the name of the extension
LEGACY_ACCEPTANCE: Final[Mapping[str, str]] = MappingProxyType(
    {"ratios": "ratio_acceptance"}
)

#: the fields of the record of an assessment of pkpdsim 0.19 which are the
#: sections of its extensions, by the name of the extension
LEGACY_ASSESSMENT_SECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"studies": ("studies",), "inputs": ("inputs",)}
)


@runtime_checkable
class Pairing(Protocol):
    """The derivation of a ratio quantity from the decisive points, see `QuantityOfInterest`.

    A pairing is declared with the quantity and assessed by the extension
    which claims it (`EvidenceExtension.claims`), e.g. the arms of a study of
    the ratios of pkpdsim. The core reads its observable only: the evidence of
    a context of use is restricted to the observables of its quantities, a
    ratio quantity to the observable of its pairing.
    """

    @property
    def observable(self) -> str:
        """Get the observable of the points the ratio is derived from, e.g. `[A]`."""
        ...


@runtime_checkable
class ExtensionSetting(Protocol):
    """A setting of an extension for a context of use which checks itself."""

    def problems(self) -> list[str]:
        """Get the problems of the setting, empty if it is valid."""
        ...


@dataclass(frozen=True)
class FactorStatement:
    """The statement of an extension of a factor of V&V 40, see the module.

    Attributes:
        text: the statement.
        before_core: place it before the statement of the core, e.g. the
            inputs before the variability in the equivalency of the inputs;
            after it otherwise, e.g. the temporal validation after the output
            comparison.
    """

    text: str
    before_core: bool = False


@dataclass(frozen=True, kw_only=True)
class ContextContribution:
    """What an extension contributes to the assessment of a context of use.

    Attributes:
        section: the section of the extension in the record of the context,
            plain data which the extension reads back; `None` for none.
        acceptance: the results of the rules of the extension, which enter the
            verdict and the output comparison of the context and are rows of
            the acceptance of its record, see the module.
        quantities: the evaluable points of every quantity of interest the
            extension derives, by its id, e.g. the pairs of a ratio quantity;
            a quantity with a point is covered.
        factors: the statement of every factor of V&V 40 the extension
            assesses, by the id of the factor (`sbmlsim.credibility.FACTORS`),
            combined with the core by `sbmlsim.credibility.combine_factor`.
    """

    section: Mapping[str, Any] | None = None
    acceptance: tuple[AcceptanceResult, ...] = ()
    quantities: Mapping[str, int] = field(default_factory=dict)
    factors: Mapping[str, FactorStatement] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBlock:
    """A block of an extension in the card of a context of use in the report.

    Attributes:
        html: the block, e.g. a heading `h4` and a table.
        after: the block of the core it follows, one of `CONTEXT_BLOCKS`.
    """

    html: str
    after: str = "acceptance"

    def __post_init__(self) -> None:
        """Check the place of the block.

        Raises:
            ValueError: for a place which is no block of the core.
        """
        if self.after not in CONTEXT_BLOCKS:
            raise ValueError(
                f"A block of an extension follows one of {list(CONTEXT_BLOCKS)}, "
                f"not '{self.after}'."
            )


@dataclass(frozen=True)
class ReportSection:
    """A section of an extension in the report, with its entry in the navigation.

    Attributes:
        id: the anchor of the section, e.g. `studies`.
        title: its heading and the text of its entry, e.g. `Study data`.
        html: the content below the heading.
    """

    id: str
    title: str
    html: str


@dataclass(frozen=True)
class ScorecardChange:
    """A column of an extension of a context of use which got worse than its baseline.

    Attributes:
        what: the column, e.g. `temporal_met`.
        baseline: the value of the baseline, as text.
        current: the current value, as text.
    """

    what: str
    baseline: str
    current: str


class EvidenceExtension:
    """Evidence of the contexts of use which a project adds, see the module.

    A subclass names itself (`name`, the key of its sections and settings) and
    overrides the methods of what it adds; every method contributes nothing by
    default.

    Attributes:
        name: the name of the extension, unique among the extensions of a
            project, e.g. `ratios`.
        label: what the rules of the extension evaluate, shown next to their
            rows in the acceptance table, e.g. `ratio pairs`; the name
            without.
        scorecard_names: the columns of the extension in the scorecard, which
            `scorecard_columns` gives and only `regressions` judges.
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    scorecard_names: ClassVar[tuple[str, ...]] = ()

    def claims(self, pairing: Pairing) -> bool:
        """Check whether the extension assesses a pairing of a ratio quantity.

        Args:
            pairing: the pairing of a quantity of interest.

        Returns:
            Whether the extension assesses it; `False` by default.
        """
        return False

    def selection_problems(
        self, cou: ContextOfUse, decisive: pd.DataFrame, lookup: MappingLookup
    ) -> list[str]:
        """Check the decisive points of a context of use after the selection.

        Args:
            cou: the context of use.
            decisive: the rows of the table of the fits of its decisive
                evidence.
            lookup: the fit mapping of an experiment and a key.

        Returns:
            Every problem, without the name of the context, which the
            selection prefixes; empty for none.
        """
        return []

    def prepare(
        self, project: Project, credibility: Credibility, fits: FitTable
    ) -> object:
        """Get what the extension reads from the project once per assessment.

        Args:
            project: the project of the model.
            credibility: its declaration.
            fits: the fresh evaluation of its adopted fits.

        Returns:
            What `assess_context` and `assessment_section` get as `prepared`,
            `None` by default.
        """
        return None

    def assess_context(
        self,
        cou: ContextOfUse,
        decisive: pd.DataFrame,
        fits: FitTable,
        prepared: object,
    ) -> ContextContribution | None:
        """Assess a context of use.

        Args:
            cou: the context of use.
            decisive: the rows of the table of the fits of its decisive
                evidence.
            fits: the table of the fits.
            prepared: what `prepare` returned.

        Returns:
            The contribution, `None` for a context the extension does not
            concern.
        """
        return None

    def acceptance(
        self, cou: ContextOfUse, decisive: pd.DataFrame, lookup: MappingLookup
    ) -> list[AcceptanceResult]:
        """Evaluate the rules of the extension on other predictions of the decisive points.

        The predictions are the column `IPRED` of the points, e.g. those of a
        draw of the parameters; the results are in the order of the ones of
        `assess_context`.

        Args:
            cou: the context of use.
            decisive: its decisive points with the other predictions.
            lookup: the fit mapping of an experiment and a key.

        Returns:
            The results, empty for an extension without rules.
        """
        return []

    def assessment_section(self, prepared: object) -> Mapping[str, Any] | None:
        """Get the section of the extension in the record of the assessment.

        Args:
            prepared: what `prepare` returned.

        Returns:
            Plain data which the extension reads back, `None` for none.
        """
        return None

    def context_report_blocks(
        self, context: Mapping[str, Any], section: Mapping[str, Any] | None
    ) -> Sequence[ContextBlock]:
        """Get the blocks of the extension in the card of a context of use.

        Args:
            context: the record of the context as plain data.
            section: the section of the extension in it, `None` without.

        Returns:
            The blocks, each after a block of the core; empty for none.
        """
        return ()

    def report_section(self, record: Mapping[str, Any]) -> ReportSection | None:
        """Get the section of the extension in the report.

        Args:
            record: the record of the assessment as plain data, with the
                sections of the extension in `extensions` of the assessment
                and of its contexts.

        Returns:
            The section with its entry in the navigation, `None` for none.
        """
        return None

    def report_tables(self, record: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """Get the tables the report writes for the extension.

        Args:
            record: the record of the assessment as plain data.

        Returns:
            The tables by the name of their file without the suffix, e.g.
            `ratios`; empty for none.
        """
        return {}

    def scorecard_columns(
        self, context: Mapping[str, Any] | None
    ) -> Mapping[str, object]:
        """Get the columns of the extension of a context of use in the scorecard.

        Args:
            context: the section of the extension in the record of the
                context, `None` for a context without one, e.g. of an older
                record.

        Returns:
            The values by column, one for every name of `scorecard_names`;
            empty for an extension without columns.
        """
        return {}

    def regressions(
        self, baseline: Mapping[str, object], current: Mapping[str, object]
    ) -> list[ScorecardChange]:
        """Compare the columns of the extension of a context with its baseline.

        Args:
            baseline: the row of the context in the baseline scorecard as it
                is, values of `scorecard_columns` or texts of a TSV, see the
                module; a baseline of an older record lacks the columns.
            current: the row of the context in the current scorecard.

        Returns:
            Every column which got worse, e.g. the rules met on the studies
            collected after the fit and no longer met; empty for none.
        """
        return []

    def pool(
        self,
        contexts: Mapping[str, Sequence[Mapping[str, Any] | None]],
        options: Mapping[str, Any],
    ) -> pd.DataFrame | None:
        """Pool the sections of the contexts of use of several models.

        Args:
            contexts: the sections of the extension of every context of use,
                by model.
            options: the options of the pooling as the caller got them, e.g.
                `{"field": "health"}`; empty for the defaults of the
                extension.

        Returns:
            The pooled table, e.g. the ratio pairs by a value of the metadata;
            `None` for an extension which pools nothing.
        """
        return None


def read_sections(
    data: Mapping[str, Any], legacy: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Get the sections of the extensions of a record.

    The sections are the field `extensions` of the record; a record without
    it, of pkpdsim 0.19 or earlier, has the evidence of the extensions as
    fields of its own, which are read as the sections: every extension of
    `legacy` with at least one of its fields in the record and not `None`
    gets the section of these fields with their values, e.g.
    `{"temporal": {"temporal": ...}}`; `{"temporal": None}` is no section.
    A record of pkpdsim 0.17 or 0.18 has none of them.

    Args:
        data: the record of a context of use or of an assessment.
        legacy: the fields of every extension in a record of pkpdsim 0.19,
            `LEGACY_CONTEXT_SECTIONS` or `LEGACY_ASSESSMENT_SECTIONS`.

    Returns:
        The sections by the name of their extension, a copy.
    """
    if EXTENSIONS in data:
        return {str(name): section for name, section in dict(data[EXTENSIONS]).items()}
    sections: dict[str, Any] = {}
    for name, fields in legacy.items():
        # a field of pkpdsim 0.19 which is `None` is no record of the extension
        present = {f: data[f] for f in fields if data.get(f) is not None}
        if present:
            sections[name] = present
    return sections


def read_acceptance(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Get the rows of the acceptance of the record of a context of use.

    The rows of the rules of the context and of the extensions as the record
    has them; a record of pkpdsim 0.19 has the rule of the ratio pairs as a
    field of its own (`LEGACY_ACCEPTANCE`), which is read as the last row,
    with the extension `ratios`.

    Args:
        data: the record of a context of use.

    Returns:
        The rows as plain data, copies.
    """
    rows = [dict(row) for row in data.get("acceptance", ())]
    if EXTENSIONS in data:
        return rows
    for name, legacy in LEGACY_ACCEPTANCE.items():
        row = data.get(legacy)
        if row is not None:
            rows.append({**row, ACCEPTANCE_EXTENSION: name})
    return rows


def scorecard_columns(
    extension: EvidenceExtension, section: Mapping[str, Any] | None
) -> dict[str, object]:
    """Get the columns of an extension of a context, checked against its names.

    Args:
        extension: the extension.
        section: its section in the record of the context, `None` without.

    Returns:
        The values by column, in the order of `scorecard_names`.

    Raises:
        ValueError: if the columns are not the names of the extension.
    """
    columns = dict(extension.scorecard_columns(section))
    if set(columns) != set(extension.scorecard_names):
        raise ValueError(
            f"The extension '{extension.name}' gives the scorecard columns "
            f"{sorted(columns)}, not its names {list(extension.scorecard_names)}."
        )
    return {name: columns[name] for name in extension.scorecard_names}


def rule_columns(
    columns: Iterable[str],
    core: Collection[str],
    extensions: Sequence[EvidenceExtension],
) -> list[str]:
    """Get the columns of a scorecard which the core compares as rules.

    Every column which is neither a column of the core nor of an extension
    (`EvidenceExtension.scorecard_names`), whose `regressions` alone judges
    them, is the value of an acceptance rule.

    Args:
        columns: the columns of the scorecard, in their order.
        core: the columns of the core, e.g. the model, the context and the
            verdict.
        extensions: the extensions of the project.

    Returns:
        The columns of the rules, in their order.
    """
    names = {n for e in extensions for n in e.scorecard_names}
    return [c for c in columns if c not in core and c not in names]


def scorecard_number(value: object) -> float:
    """Get a value of a scorecard as a number, of the process or read from a TSV.

    Args:
        value: the value, e.g. `4`, `4.0`, `"4"` or `""`.

    Returns:
        The number, `NaN` for `None`, an empty text or a text which is no
        number.
    """
    if value is None or isinstance(value, bool):
        return math.nan if value is None else float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return math.nan


def scorecard_true(value: object) -> bool:
    """Check whether a value of a scorecard is true, of the process or read from a TSV.

    Args:
        value: the value, e.g. `True` or `"True"`.

    Returns:
        Whether it is `True` or the text `True`.
    """
    return value is True or str(value) == "True"
