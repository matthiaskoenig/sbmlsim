"""The lock of the evidence of a declaration (`sbmlsim.credibility.lock`).

The lock of a project lies in its root; the tests use projects whose root is
their temporary directory, so no test writes into the toy package, and the
lock is written into an out, never into a root.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from sbmlsim.credibility import (
    Credibility,
    Evidence,
    EvidenceRole,
    check_declaration,
    select,
)
from sbmlsim.credibility.lock import (
    LOCK_FILE,
    LOCK_REMEDY,
    lock_differences,
    lock_path,
    read_lock,
    selection_lock,
    write_lock,
)
from sbmlsim.fit.filters import named_filter
from sbmlsim.project import Project
from tests.credibility.tables import fit_table
from tests.credibility_toy import PROJECT
from tests.credibility_toy.credibility import CREDIBILITY

#: the lock of the declaration of the toy package written by pkpdsim 0.19
LOCK_0_19 = (
    Path(__file__).parents[1] / "data" / "credibility" / "credibility.lock-0.19.json"
)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """Get the toy project with its root in the temporary directory."""
    root = tmp_path / "root"
    root.mkdir()
    return PROJECT.replace(root=root)


def _with_dose_subjects() -> Credibility:
    """Get the declaration whose context dose also validates on fm_subject."""
    contexts = []
    for cou in CREDIBILITY.contexts:
        if cou.id == "dose":
            extra = Evidence(
                "subjects",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_subject"}}),
            )
            cou = dataclasses.replace(cou, evidence=(cou.evidence[0], extra))
        contexts.append(cou)
    return dataclasses.replace(CREDIBILITY, contexts=tuple(contexts))


def test_the_lock_of_a_selection() -> None:
    fits = fit_table()
    lock = selection_lock(select(CREDIBILITY, fits), fits)
    assert lock["renal/renal impairment"] == ["DECAY/CredA/fm_renal"]
    assert lock["/scaled data"] == ["DECAY/CredA/fm_limit"]
    assert list(lock) == sorted(lock)


def test_the_lock_of_pkpdsim(project: Project) -> None:
    """The lock of pkpdsim 0.19 is the lock of the same selection."""
    fits = fit_table()
    shutil.copy(LOCK_0_19, lock_path(project))
    assert read_lock(project) == selection_lock(select(CREDIBILITY, fits), fits)
    assert lock_differences(project, select(CREDIBILITY, fits), fits) == []


def test_write_into_the_out_and_compare(project: Project, tmp_path: Path) -> None:
    fits = fit_table()
    selections = select(CREDIBILITY, fits)
    assert lock_path(project) == project.root / LOCK_FILE
    assert lock_differences(project, selections, fits) is None
    written = write_lock(selections, fits, tmp_path / "out")
    assert written == tmp_path / "out" / LOCK_FILE
    assert not lock_path(project).exists()
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["version"] == 1
    # the reviewed lock is committed into the root
    shutil.copy(written, lock_path(project))
    assert read_lock(project) == data["evidence"]
    assert lock_differences(project, selections, fits) == []
    changed = select(_with_dose_subjects(), fits)
    differences = lock_differences(project, changed, fits)
    assert differences is not None
    assert "the evidence 'dose/subjects' is not in the lock" in differences


def test_a_gone_and_a_changed_evidence(project: Project, tmp_path: Path) -> None:
    fits = fit_table()
    write_lock(select(CREDIBILITY, fits), fits, project.root)
    path = lock_path(project)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["evidence"]["renal/renal impairment"] = ["DECAY/CredA/fm_other"]
    data["evidence"]["renal/old"] = ["DECAY/CredA/fm_renal"]
    path.write_text(json.dumps(data), encoding="utf-8")
    differences = lock_differences(project, select(CREDIBILITY, fits), fits)
    assert differences == [
        "the evidence 'renal/old' of the lock is gone",
        "the evidence 'renal/renal impairment' selects other mappings: "
        "+DECAY/CredA/fm_renal, -DECAY/CredA/fm_other",
    ]


def test_a_file_which_is_no_lock(project: Project) -> None:
    lock_path(project).write_text(json.dumps({"version": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="is no lock of version 1"):
        read_lock(project)


def test_check_declaration_compares_the_lock(
    credibility_project: Project, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    project = credibility_project.replace(root=root)
    assert check_declaration(project) == []
    shutil.copy(LOCK_0_19, lock_path(project))
    assert check_declaration(project) == []
    data = json.loads(lock_path(project).read_text(encoding="utf-8"))
    data["evidence"]["renal/renal impairment"] = []
    lock_path(project).write_text(json.dumps(data), encoding="utf-8")
    problems = check_declaration(project)
    assert problems[-1] == LOCK_REMEDY
    assert "selects other mappings: +DECAY/CredA/fm_renal" in problems[0]
    lock_path(project).write_text(json.dumps({"version": 2}), encoding="utf-8")
    (problem,) = check_declaration(project)
    assert "is no lock of version 1" in problem
