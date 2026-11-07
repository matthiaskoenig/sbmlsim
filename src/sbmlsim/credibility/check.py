"""The static check of the credibility declaration of a project, without a simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sbmlsim.credibility.declaration import Credibility, DeclarationError
from sbmlsim.credibility.lock import LOCK_REMEDY, lock_differences
from sbmlsim.credibility.records import KIND_REMEDY, kind_differences, mapping_table
from sbmlsim.credibility.selection import SelectionError, select

if TYPE_CHECKING:
    from sbmlsim.credibility.extensions import EvidenceExtension
    from sbmlsim.project import Project

__all__ = ["check_declaration"]


def check_declaration(project: Project) -> list[str]:
    """Check the credibility declaration of a project against its adopted fits.

    The declaration must be valid, its domains claim values of the metadata of
    the project, its settings of extensions name extensions of the project and
    every pairing of a ratio is claimed by one (`Project.load_credibility`);
    invalid credibility settings of the project are a problem as well. The kinds of the mappings of the
    current selection of the fit definitions must be the recorded kinds of the
    adopted fits, and the evidence must select valid data (see `select`, with
    the evidence extensions of the project), all from the fit definitions and
    the records, without a simulation, and select the mappings of its lock, if
    the project has one. A problem of the records, e.g. a record without the
    kinds of its mappings, two kinds of one mapping, no adopted fits or a
    subset adopted twice, is a problem as well, after which the kinds and the
    evidence are not checked.

    Args:
        project: the project with a declaration and adopted fits.

    Returns:
        Every problem, empty if the declaration holds.

    Raises:
        ValueError: if the project has no declaration.
    """
    if project.credibility is None:
        raise ValueError(
            f"The project '{project.name}' has no credibility declaration."
        )
    problems: list[str] = []
    credibility: Credibility | None = None
    extensions: list[EvidenceExtension] = []
    try:
        credibility = project.load_credibility()
        extensions = project.evidence_extensions()
    except DeclarationError as err:
        problems.extend(err.problems)
    except ValueError as err:
        # the credibility settings of the project: its metadata, its domain
        # fields or an extension
        problems.append(str(err))
    try:
        fits = mapping_table(project)
    except (FileNotFoundError, ValueError) as err:
        return [*problems, str(err)]
    differences = kind_differences(fits.table, fits.recorded_kinds)
    if differences:
        problems.extend(differences)
        problems.append(
            "The kinds of the fit mappings differ from the kinds the adopted fits "
            f"recorded. {KIND_REMEDY}"
        )
    if credibility is not None:
        try:
            selections = select(credibility, fits, extensions)
        except SelectionError as err:
            problems.extend(err.problems)
        else:
            try:
                locked = lock_differences(project, selections, fits)
            except ValueError as err:
                problems.append(str(err))
            else:
                if locked:
                    problems.extend(locked)
                    problems.append(LOCK_REMEDY)
    return problems
