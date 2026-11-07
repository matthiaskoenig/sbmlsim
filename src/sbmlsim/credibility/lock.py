"""The lock of the evidence of a credibility declaration: splits fixed in advance.

A validation counts only if its data was set apart before the predictions were
seen. The kinds of the mappings (training or validation) are fixed by the
records of the adopted fits (`sbmlsim.credibility.records`); which validation
data answers which question is fixed by the declaration, which a modeller can
change after looking at a verdict. The lock records the mappings every evidence
and every limitation selects, `credibility.lock.json` in the root of the
project (the folder of the model package, `lock_path`), reviewed in the
history of the repository:

- `write_lock` writes the selection of a declaration on the table of the fits
  into an out, never into the package: the new lock is an output which a
  person or an agent reviews and commits as the lock of the package.
- `lock_differences` names every evidence whose mappings differ from the lock;
  `check_declaration` reports them and the assessment raises a
  `LockMismatchError`, so a changed split is renewed explicitly.

A project without a lock is assessed as before; the record says whether the
evidence matched a lock (`Provenance.lock`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from sbmlsim.atomic import write_text_atomic

if TYPE_CHECKING:
    from pathlib import Path

    from sbmlsim.credibility.records import FitTable
    from sbmlsim.credibility.selection import EvidenceSelection, Selections
    from sbmlsim.project import Project

__all__ = [
    "LOCK_FILE",
    "LOCK_REMEDY",
    "LockMismatchError",
    "lock_differences",
    "lock_path",
    "read_lock",
    "selection_lock",
    "write_lock",
]

#: the file of the lock in the root of the project and in the out of a run
LOCK_FILE: Final = "credibility.lock.json"
#: the version of the format of the lock
_VERSION: Final = 1
#: the remedy of a selection which differs from the lock
LOCK_REMEDY: Final = (
    "Review the change of the split, write the lock again (the kind `credibility` "
    f"with `lock` writes {LOCK_FILE} into its out) and commit it into the root of "
    "the project."
)

#: the mappings of the evidence by its key, `fit/experiment/mapping` each
type Lock = dict[str, list[str]]


class LockMismatchError(ValueError):
    """The evidence selects other mappings than the lock of the declaration.

    Attributes:
        problems: every evidence which differs, with its added and removed
            mappings.
    """

    def __init__(self, problems: list[str]) -> None:
        """Create the error.

        Args:
            problems: every evidence which differs.
        """
        self.problems = list(problems)
        super().__init__(
            "The evidence of the credibility declaration differs from its lock:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
            + f"\n{LOCK_REMEDY}"
        )


def lock_path(project: Project) -> Path:
    """Get the file of the lock of a project, `credibility.lock.json` in its root."""
    return project.root / LOCK_FILE


def _key(selection: EvidenceSelection) -> str:
    """Get the key of an evidence: `context/name`, `/name` for one of the model."""
    return f"{selection.context}/{selection.evidence.name}"


def selection_lock(selections: Selections, fits: FitTable) -> Lock:
    """Get the mappings of every evidence and limitation of a selection.

    Args:
        selections: the selection of the declaration, see `select`.
        fits: the table of the fits the selection was made on.

    Returns:
        The sorted mappings, `fit/experiment/mapping`, by the key of the
        evidence, `context/name` (a limitation of the model `/name`).
    """
    table = fits.table
    lock: Lock = {}
    for selection in (*selections.evidence, *selections.limitation_evidence):
        rows = table.loc[selection.rows, ["fit", "experiment", "mapping"]].astype(str)
        mappings = {"/".join(row) for row in rows.itertuples(index=False)}
        lock.setdefault(_key(selection), [])
        lock[_key(selection)] = sorted({*lock[_key(selection)], *mappings})
    return dict(sorted(lock.items()))


def write_lock(selections: Selections, fits: FitTable, out: Path) -> Path:
    """Write the lock of the evidence of a declaration into an out.

    Args:
        selections: the selection of the declaration.
        fits: the table of the fits the selection was made on.
        out: the directory the lock is written into, created if missing;
            never the package, whose lock is the reviewed copy of it.

    Returns:
        The file of the lock, `credibility.lock.json` of the out.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / LOCK_FILE
    data = {"version": _VERSION, "evidence": selection_lock(selections, fits)}
    write_text_atomic(path, json.dumps(data, indent=2) + "\n")
    return path


def read_lock(project: Project) -> Lock | None:
    """Read the lock of a project, `None` without one.

    The format is the one of pkpdsim 0.19, version 1.

    Raises:
        ValueError: if the file is no lock of this version.
    """
    path = lock_path(project)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != _VERSION:
        raise ValueError(f"The file '{path}' is no lock of version {_VERSION}.")
    return {str(k): [str(m) for m in v] for k, v in data["evidence"].items()}


def lock_differences(
    project: Project, selections: Selections, fits: FitTable
) -> list[str] | None:
    """Get every evidence whose mappings differ from the lock of the project.

    Args:
        project: the project.
        selections: the current selection of its declaration.
        fits: the table of the fits the selection was made on.

    Returns:
        A line per evidence which is new, gone or selects other mappings, with
        the added (`+`) and removed (`-`) mappings; empty if the selection
        matches the lock, `None` if the project has no lock.
    """
    locked = read_lock(project)
    if locked is None:
        return None
    current = selection_lock(selections, fits)
    problems: list[str] = []
    for key in sorted(locked.keys() | current.keys()):
        if key not in current:
            problems.append(f"the evidence '{key}' of the lock is gone")
            continue
        if key not in locked:
            problems.append(f"the evidence '{key}' is not in the lock")
            continue
        added = sorted(set(current[key]) - set(locked[key]))
        removed = sorted(set(locked[key]) - set(current[key]))
        if added or removed:
            changes = [*(f"+{m}" for m in added), *(f"-{m}" for m in removed)]
            problems.append(
                f"the evidence '{key}' selects other mappings: {', '.join(changes)}"
            )
    return problems
