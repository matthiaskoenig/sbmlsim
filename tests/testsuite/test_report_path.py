"""Tests that the report of the suite does not collide with its documentation.

`zensical` renders `docs/testsuite.md` into `site/testsuite/index.html`, so a
report written to `site/testsuite` replaces the page which describes it. It
did, and the published page was the report without the theme and without the
navigation of the documentation, while the page itself was unreachable. These
tests pin the two paths against each other.
"""

import re
from pathlib import Path

REPO = Path(__file__).parent.parent.parent

#: the page of the documentation, rendered to `site/<stem>/index.html`
DOCS_PAGE = REPO / "docs" / "testsuite.md"

#: the command line which writes the report, i.e. the default of `--output`
SCRIPT = REPO / "scripts" / "testsuite.py"


def _report_output() -> Path:
    """Get the default output of the report, relative to the site."""
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'default=Path\("site"\)((?:\s*/\s*"[^"]+")+),', source)
    assert match, "the default of `--output` is not a path built from `site`"
    parts = re.findall(r'"([^"]+)"', match.group(1))
    return Path(*parts)


def _report_link() -> Path:
    """Get the link of the documentation to the report, relative to `docs`."""
    page = DOCS_PAGE.read_text(encoding="utf-8")
    match = re.search(r"\[\*\*The report of the current run\*\*\]\(([^)]+)\)", page)
    assert match, "the page does not link the report of the current run"
    return Path(match.group(1))


def test_the_report_does_not_overwrite_its_documentation() -> None:
    """The report is not written over the page rendered from `testsuite.md`."""
    # the page of a source `docs/<stem>.md` is `site/<stem>/index.html`
    page_dir = Path(DOCS_PAGE.stem)
    assert _report_output() != page_dir


def test_the_documentation_links_the_report_where_it_is_written() -> None:
    """The link of the page resolves to the directory the report is written to.

    A relative link of a page is resolved against its source, i.e. against
    `docs/`, which is why the link carries the `testsuite/` of the page itself.
    """
    link = _report_link()
    assert link.name == "index.html"
    assert link.parent == _report_output()
