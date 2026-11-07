"""The evidence extensions: evidence of the contexts of use which a project adds.

The core of the credibility assesses what every model has: the data points of
its adopted fits against the acceptance rules, the coverage of the claimed
domain, the verification, the parameter uncertainty and the variability. A
domain adds evidence of its own, e.g. the ratios between the arms of a study,
the equivalency of the simulated and the reported inputs of the subjects, the
chronology of the studies or the provenance of the study data. Such evidence
is an `EvidenceExtension`, which a project registers
(`sbmlsim.project.CredibilitySettings.extensions`) and the core calls at fixed
points:

- `selection_problems`: the decisive points of a context of use, after the
  selection of the evidence (`sbmlsim.credibility.selection.select`), e.g. a
  study of a ratio with two reference arms.
- `prepare`: what the extension reads from the project once per assessment,
  e.g. the chronology of the studies.
- `assess_context`: the contribution of the extension to a context of use, a
  `ContextContribution` (its section of the record, its acceptance results,
  which enter the verdict, the coverage of the quantities it derives and the
  statements of the factors it assesses).
- `acceptance`: its acceptance results on other predictions of the decisive
  points, those of a draw of the parameter uncertainty or of an alternative
  of the verification.
- `assessment_section`: its section of the record of the assessment, e.g. the
  provenance of the study data of every study.
- `report_tables`, `report_section`, `scorecard_columns` and `pool`: the
  tables and the section of the report, the columns of the scorecard and the
  pooling of its records across models.

Every method has a default which contributes nothing, so an extension
overrides what it adds. The record of a context of use and of an assessment
has the sections of the extensions by their name in its field `extensions`
(`EXTENSIONS`), which the extension of that name writes and reads; the core
keeps a section as it is. A record of pkpdsim 0.19 has the evidence of its
extensions as fields of its own, which `read_sections` reads as their
sections (`LEGACY_CONTEXT_SECTIONS`, `LEGACY_ASSESSMENT_SECTIONS`); a record
of pkpdsim 0.17 or 0.18 has none.

A setting of an extension for one context of use, e.g. the acceptance of the
ratio pairs, is an entry of `ContextOfUse.extensions` by the name of the
extension; a setting with a method `problems` (`ExtensionSetting`) is checked
with the declaration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    "EXTENSIONS",
    "LEGACY_ASSESSMENT_SECTIONS",
    "LEGACY_CONTEXT_SECTIONS",
    "ContextContribution",
    "EvidenceExtension",
    "ExtensionSetting",
    "Pairing",
    "read_sections",
]

#: the field of a record with the sections of the extensions by their name
EXTENSIONS: Final = "extensions"

#: the fields of the record of a context of use of pkpdsim 0.19 which are the
#: sections of its extensions, by the name of the extension
LEGACY_CONTEXT_SECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ratios": ("ratios", "ratio_acceptance"),
        "temporal": ("temporal",),
        "inputs": ("inputs",),
    }
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
    which knows it, e.g. the arms of a study of the ratios of pkpdsim. The
    core reads its observable only: the evidence of a context of use is
    restricted to the observables of its quantities, a ratio quantity to the
    observable of its pairing.
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


@dataclass(frozen=True, kw_only=True)
class ContextContribution:
    """What an extension contributes to the assessment of a context of use.

    Attributes:
        section: the section of the extension in the record of the context,
            plain data which the extension reads back; `None` for none.
        acceptance: the results of the rules of the extension, which enter the
            verdict of the context together with its acceptance rules.
        quantities: the evaluable points of every quantity of interest the
            extension derives, by its id, e.g. the pairs of a ratio quantity;
            a quantity with a point is covered.
        factors: the statement of every factor of V&V 40 the extension
            assesses, by the id of the factor (`sbmlsim.credibility.FACTORS`).
    """

    section: Mapping[str, Any] | None = None
    acceptance: tuple[AcceptanceResult, ...] = ()
    quantities: Mapping[str, int] = field(default_factory=dict)
    factors: Mapping[str, str] = field(default_factory=dict)


class EvidenceExtension:
    """Evidence of the contexts of use which a project adds, see the module.

    A subclass names itself (`name`, the key of its sections in the records)
    and overrides the methods of what it adds; every method contributes
    nothing by default.

    Attributes:
        name: the name of the extension, unique among the extensions of a
            project, e.g. `ratios`.
    """

    name: ClassVar[str] = ""

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

    def report_tables(self, record: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        """Get the tables the report writes for the extension.

        Args:
            record: the record of the assessment as plain data, with the
                sections of the extension in `extensions` of the assessment
                and of its contexts.

        Returns:
            The tables by the name of their file without the suffix, e.g.
            `ratios`; empty for none.
        """
        return {}

    def report_section(self, record: Mapping[str, Any]) -> str:
        """Get the section of the extension in the HTML report.

        Args:
            record: the record of the assessment as plain data.

        Returns:
            The HTML of the section, empty for none.
        """
        return ""

    def scorecard_columns(
        self, context: Mapping[str, Any] | None
    ) -> Mapping[str, object]:
        """Get the columns of the extension of a context of use in the scorecard.

        Args:
            context: the section of the extension in the record of the
                context, `None` for a context without one, e.g. of an older
                record.

        Returns:
            The values by column; empty for none.
        """
        return {}

    def pool(
        self, contexts: Mapping[str, Sequence[Mapping[str, Any] | None]]
    ) -> pd.DataFrame | None:
        """Pool the sections of the contexts of use of several models.

        Args:
            contexts: the sections of the extension of every context of use,
                by model.

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
    `legacy` with at least one of its fields in the record gets the section
    of these fields with their values, e.g. `{"temporal": {"temporal": ...}}`.
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
        present = {f: data[f] for f in fields if f in data}
        if present:
            sections[name] = present
    return sections
