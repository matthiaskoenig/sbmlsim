"""The pages of the documentation state what the code has.

The table of the fields of a project in `docs/project.md` names every field of
`sbmlsim.project.Project`, and a block of a page which installs sbmlsim into a
fresh environment and runs it installs the build of libroadrunner from its
index first, since `pip install sbmlsim` alone does not satisfy the
requirement of libroadrunner (`docs/installation.md`).
"""

import re
from pathlib import Path

import pytest

from sbmlsim.project import Project

DOCS = Path(__file__).parents[2] / "docs"
#: the index of the builds of libroadrunner, see `docs/installation.md`
INDEX = "https://matthiaskoenig.github.io/roadrunner-wheels/simple/"
#: a fenced block of shell commands
BLOCK = re.compile(r"^```bash\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _table_fields(page: Path) -> set[str]:
    """Get the fields the first column of the tables of a page names."""
    names: set[str] = set()
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            first = line.split("|")[1]
            names.update(re.findall(r"`([a-z_]+)`", first))
    return names


def test_the_fields_of_a_project_are_documented() -> None:
    documented = _table_fields(DOCS / "project.md")
    missing = sorted(set(Project.model_fields) - documented)
    assert missing == []


def _blocks() -> list[tuple[str, str]]:
    """Get every bash block of the pages with the name of its page."""
    return [
        (page.name, block)
        for page in sorted(DOCS.glob("*.md"))
        for block in BLOCK.findall(page.read_text(encoding="utf-8"))
    ]


@pytest.mark.parametrize(
    ("page", "block"),
    [
        (page, block)
        for page, block in _blocks()
        if re.search(r"pip install [^\n]*sbmlsim", block)
        and re.search(r"python -m sbmlsim", block)
    ],
)
def test_a_block_which_installs_and_runs_sbmlsim_installs_libroadrunner_first(
    page: str, block: str
) -> None:
    """A fresh install which runs at once names the index of libroadrunner."""
    lines = block.splitlines()
    sbmlsim = next(k for k, line in enumerate(lines) if "sbmlsim" in line)
    assert any(INDEX in line for line in lines[:sbmlsim]), page


def test_the_teaching_mode_is_such_a_block() -> None:
    """The one block of the kind today, which pinned the test."""
    assert any(
        page == "serve.md" and "python -m sbmlsim.serve" in block
        for page, block in _blocks()
        if re.search(r"pip install [^\n]*sbmlsim", block)
    )
