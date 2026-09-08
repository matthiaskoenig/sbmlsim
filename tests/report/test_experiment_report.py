"""Test the HTML report of the simulation experiments.

The report shares its design with the report of a fit, see
`sbmlsim.report.templates`.
"""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from examples.hctz.experiments.studies import Beermann1976
from examples.hctz.helpers import run_experiments

VOID_TAGS = {"img", "br", "hr", "meta", "link", "input", "source", "col"}


@pytest.fixture(scope="module")
def report_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run a simulation experiment and report it, once for the module."""
    tmp_path = tmp_path_factory.mktemp("experiment_report")
    cwd = Path.cwd()
    import os

    os.chdir(tmp_path)
    try:
        return tmp_path / run_experiments(Beermann1976, output_dir="report")
    finally:
        os.chdir(cwd)


def _html(path: Path) -> str:
    """Read a page of the report."""
    return path.read_text(encoding="utf-8")


def _assert_balanced(html: str) -> None:
    """Check that the tags of a page are balanced."""
    stack: list[str] = []
    errors: list[str] = []

    class Checker(HTMLParser):
        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in VOID_TAGS:
                stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in VOID_TAGS:
                return
            if not stack or stack[-1] != tag:
                errors.append(tag)
            else:
                stack.pop()

    Checker(convert_charrefs=True).feed(html)
    assert errors == []
    assert stack == []


def test_report_pages_exist(report_dir: Path) -> None:
    """The report has an index and a page per experiment."""
    assert (report_dir / "index.html").exists()
    assert (report_dir / "Beermann1976" / "Beermann1976.html").exists()


def test_experiment_page_sections(report_dir: Path) -> None:
    """The page of an experiment has its three sections."""
    html = _html(report_dir / "Beermann1976" / "Beermann1976.html")
    assert re.findall(r'<section id="([a-z]+)"', html) == [
        "overview",
        "figures",
        "code",
    ]
    # the shared chrome of the reports
    for token in ['id="search"', 'id="lightbox"', 'data-filter="card"', "--accent"]:
        assert token in html, token


def test_index_page(report_dir: Path) -> None:
    """The index lists the experiments with their figures."""
    html = _html(report_dir / "index.html")
    assert re.findall(r'<section id="([a-z]+)"', html) == ["experiments"]
    assert "Beermann1976" in html
    assert 'href="Beermann1976/Beermann1976.html"' in html
    assert 'id="search"' in html


@pytest.mark.parametrize("page", ["index.html", "Beermann1976/Beermann1976.html"])
def test_pages_are_offline_and_well_formed(report_dir: Path, page: str) -> None:
    """Every page is balanced, resolves its links and needs no network."""
    path = report_dir / page
    html = _html(path)

    _assert_balanced(html)
    assert "http://" not in html
    assert "https://" not in html

    references = re.findall(r'(?:src|href)="([^"#:]+)"', html)
    assert references
    assert [ref for ref in references if not (path.parent / ref).exists()] == []


def test_code_is_escaped(report_dir: Path) -> None:
    """The source of the experiment is escaped, it is not markup."""
    import html as html_module

    page = _html(report_dir / "Beermann1976" / "Beermann1976.html")
    code = re.search(r'<pre class="code"><code>(.*?)</code></pre>', page, re.S)
    assert code is not None

    # the report reads the source of the experiment, wherever it lives
    module = sys.modules[Beermann1976.__module__]
    assert module.__file__ is not None
    source = Path(module.__file__)
    assert html_module.unescape(code.group(1)) == source.read_text(encoding="utf-8")
