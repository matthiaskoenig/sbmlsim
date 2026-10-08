"""Test the reports of the simulation experiments.

The HTML report shares its design with the report of a fit, see
`sbmlsim.report.templates`; the markdown report is read in the preview of an
editor and the LaTeX report includes the figures in a document.
"""

import html as html_module
import re
import sys
import webbrowser
from html.parser import HTMLParser
from pathlib import Path

import pytest

from examples.hctz_fitting import DATA_PATH, HCTZ_PATH
from examples.hctz_fitting.experiments.studies import Beermann1976
from examples.hctz_fitting.helpers import MODEL_PATH
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.report.experiment_report import ExperimentReport, ReportResults
from sbmlsim.simulator import SimulatorSerial

VOID_TAGS = {"img", "br", "hr", "meta", "link", "input", "source", "col"}

#: the figures of Beermann1976
FIGURES = ["Tab1A", "Fig2", "Fig3"]


@pytest.fixture(scope="module")
def report_and_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ExperimentReport, Path]:
    """Run a simulation experiment and write its reports, once for the module.

    The figures are written as static images and as interactive pages, the
    report is written as HTML, markdown and LaTeX.
    """
    output_path = tmp_path_factory.mktemp("experiment_report")
    runner = ExperimentRunner(
        experiment_classes=[Beermann1976],
        data_path=DATA_PATH,
        base_path=HCTZ_PATH,
        simulator=SimulatorSerial(model=MODEL_PATH),
    )
    results = runner.run_experiments(
        output_path=output_path,
        show_figures=False,
        save_results=False,
        figure_formats=["svg", "png", "html"],
        reduced_selections=True,
    )
    report_results = ReportResults()
    for result in results:
        report_results.add_experiment_result(exp_result=result)
    report = ExperimentReport(report_results)
    for report_type in ExperimentReport.ReportType:
        report.create_report(output_path, report_type=report_type)
    return report, output_path


@pytest.fixture(scope="module")
def report_dir(report_and_dir: tuple[ExperimentReport, Path]) -> Path:
    """Get the directory of the reports."""
    return report_and_dir[1]


def test_the_report_ends_the_output_with_a_link(
    report_and_dir: tuple[ExperimentReport, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report is a section with a link to its index, which can be opened."""
    report, output_path = report_and_dir
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, new=0: opened.append(url))

    report_path = report.create_report(output_path)
    lines = capsys.readouterr().out.splitlines()
    assert any("Report" in line for line in lines)
    uri = report_path.resolve().as_uri()
    experiments = next(line for line in lines if line.startswith("experiments"))
    link = next(line for line in lines if line.startswith("report"))
    assert experiments.split() == ["experiments", "1"]
    assert link.split() == ["report", uri]
    # the link is aligned with the values of the section
    assert link.index(uri) == experiments.index("1")
    assert not opened

    report.create_report(output_path, show_report=True)
    assert opened == [uri]


def _references(path: Path) -> list[str]:
    """Get the local files a page of the report refers to.

    The references of a page are its `src` and `href` attributes and, in
    markdown, the targets of its links and images; anchors and urls are not
    files.
    """
    text = path.read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)="([^"]+)"', text)
    if path.suffix == ".md":
        references += re.findall(r"\]\(([^)\s]+)\)", text)
    return [
        html_module.unescape(ref)
        for ref in references
        if not ref.startswith("#") and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", ref)
    ]


def _unresolved(path: Path) -> list[str]:
    """Get the references of a page which are no file next to it."""
    return [
        ref
        for ref in _references(path)
        if ref.startswith("/") or not (path.parent / ref).exists()
    ]


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

    assert _references(path)
    assert _unresolved(path) == []


def test_code_is_escaped(report_dir: Path) -> None:
    """The source of the experiment is escaped, it is not markup."""
    page = _html(report_dir / "Beermann1976" / "Beermann1976.html")
    code = re.search(r'<pre class="code"><code>(.*?)</code></pre>', page, re.S)
    assert code is not None

    # the report reads the source of the experiment, wherever it lives
    module = sys.modules[Beermann1976.__module__]
    assert module.__file__ is not None
    source = Path(module.__file__)
    assert html_module.unescape(code.group(1)) == source.read_text(encoding="utf-8")


def test_index_shows_every_figure(report_dir: Path) -> None:
    """The index shows the static image of every figure of an experiment."""
    images = re.findall(r'<img[^>]*src="([^"]+)"', _html(report_dir / "index.html"))
    assert images == [f"Beermann1976/Beermann1976_{fig}.svg" for fig in FIGURES]


@pytest.mark.parametrize("page", ["index.md", "Beermann1976/Beermann1976.md"])
def test_markdown_pages_resolve_their_references(report_dir: Path, page: str) -> None:
    """The images and links of the markdown report are relative to the page.

    A path which starts with `/` is relative to the root of the file system or
    of the workspace of an editor, the preview of VS Code does not show it.
    """
    path = report_dir / page
    assert _references(path)
    assert _unresolved(path) == []


@pytest.mark.parametrize("page", ["index.md", "Beermann1976/Beermann1976.md"])
def test_markdown_pages_show_every_figure(report_dir: Path, page: str) -> None:
    """Every figure is an image of the index and of the page of the experiment."""
    text = (report_dir / page).read_text(encoding="utf-8")
    images = [
        Path(target).name for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
    ]
    assert images == [f"Beermann1976_{fig}.svg" for fig in FIGURES]


def test_markdown_experiment_page(report_dir: Path) -> None:
    """The page of an experiment lists its models, datasets, figures and code."""
    text = (report_dir / "Beermann1976" / "Beermann1976.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (.+)$", text, re.M) == [
        "Models",
        "Datasets",
        "Figures",
        "Code",
    ]
    assert "[Beermann1976_Fig3.html](Beermann1976_Fig3.html)" in text
    module = sys.modules[Beermann1976.__module__]
    assert module.__file__ is not None
    assert Path(module.__file__).read_text(encoding="utf-8") in text


def test_latex_report_includes_every_figure(report_dir: Path) -> None:
    """The LaTeX report includes the copied images of the figures."""
    tex = report_dir / "index.tex"
    graphics = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex.read_text())
    assert graphics == [f"index_figures/Beermann1976_{fig}.png" for fig in FIGURES]
    assert [g for g in graphics if not (report_dir / g).exists()] == []
