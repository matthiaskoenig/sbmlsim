"""Tests of the documentation site and the README of a release repository."""

import dataclasses
import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from sbmlsim.credibility import (
    CHECKS,
    Battery,
    CredibilityAssessment,
    Provenance,
    assess_table,
)
from sbmlsim.credibility.assessment import CredibilityAssessment as Assessment
from sbmlsim.fit.records import AdoptedRecord
from sbmlsim.release.build import BuildPaths
from sbmlsim.release.docs import PAGES, pages, write_docs
from sbmlsim.release.hooks import ReleaseHook
from sbmlsim.release.notes import fits_prefix
from sbmlsim.release.project import ProjectContext, write_project_files
from tests.credibility.tables import (
    PROVENANCE,
    battery_of,
    fit_table,
    verification_of,
)
from tests.credibility_toy.credibility import CREDIBILITY
from tests.credibility_toy.extensions import ARMS
from tests.project_toy import PROJECT, TOY_PATH

assert PROJECT.release is not None
SETTINGS = dataclasses.replace(PROJECT.release, archive="toy_model.omex")
PAGE_NAMES = [page for page, _ in PAGES]


@pytest.fixture(autouse=True)
def _distributions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the default variant of the toy the distributions of its fitted rate."""
    monkeypatch.setattr("sbmlsim.release.build.has_distributions", lambda *args: True)


def write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def adopted_ids(context: ProjectContext) -> list[str]:
    return list(context.project.adopted_fits().values())


def without_fits(context: ProjectContext) -> ProjectContext:
    """The context of a release without adopted fits."""
    project = context.project.replace(adopted=None, fits_dir=None)
    return dataclasses.replace(context, project=project)


@pytest.fixture
def release(toy_records: Path, tmp_path: Path) -> tuple[ProjectContext, Path]:
    """A built release repository of the toy model with an adopted fit."""
    model = PROJECT.replace(
        credibility=None,
        adopted=toy_records / "adopted.json",
        fits_dir=toy_records / "fits",
        release=SETTINGS,
    )
    path = tmp_path / "toy-model"
    (fit_id,) = adopted_ids_of(model)
    shutil.copytree(
        toy_records / "fits" / fit_id, path / "src/tests/project_toy/fits" / fit_id
    )
    model = model.replace(fits_dir=TOY_PATH / "fits")
    (path / BuildPaths.MODELS).mkdir(parents=True)
    shutil.copy(TOY_PATH / "models" / "toy.xml", path / BuildPaths.MODELS / "toy.xml")
    write(path / BuildPaths.MODELS / "toy.md", "# toy")
    write(path / BuildPaths.FIGURES / "toy_model.png", b"png")
    write(path / BuildPaths.FIGURES / "toy.png", b"png")
    write(path / "toy_model.omex", b"omex")
    # the simulation report of sbmlsim: `index.html` and `<experiment>/<experiment>.html`
    write(path / BuildPaths.SIMULATIONS / "index.html", "<html/>")
    for experiment in ("Decay", "Slow"):
        write(
            path / BuildPaths.SIMULATIONS / experiment / f"{experiment}.html", "<html/>"
        )
    write(path / BuildPaths.SIMULATIONS / "_figures" / "Decay_fig.svg", "<svg/>")
    for version in ("0.9.0", "0.10.0", "0.2.0"):
        write(path / "release-notes" / f"{version}.md", f"# Notes {version}\n\n## A\n")
    context = ProjectContext(
        project=model,
        reference="tests.project_toy:PROJECT",
        settings=SETTINGS,
        version="0.10.0",
        python="3.14",
        pythons=("3.14", "3.15"),
        dependencies=(),
        constraints=(),
        framework_path=None,
        year=2026,
        requires_python=">=3.14",
    )
    return context, path


def adopted_ids_of(project: Any) -> list[str]:
    return list(project.adopted_fits().values())


def test_docs_pages(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_project_files(context, path)
    written = write_docs(context, path)
    assert pages(context, path) == PAGES
    assert {p.name for p in (path / "docs").glob("*.md")} == {
        f"{p}.md" for p in PAGE_NAMES
    }
    assert path / "README.md" in written
    nav = tomllib.loads((path / "zensical.toml").read_text(encoding="utf-8"))[
        "project"
    ]["nav"]
    in_nav = {file for entry in nav for file in entry.values()}
    assert in_nav == {f"{page}.md" for page in PAGE_NAMES}
    assert path / "zensical.toml" in written
    assert (path / "docs" / "images" / "toy_model.png").is_file()
    assert path / "docs" / "images" / "toy_model.png" in written
    index = (path / "docs" / "index.md").read_text(encoding="utf-8")
    assert "images/toy_model.png" in index
    assert (
        "Doe, J. (2026). *Toy model of the decay of a substance*. Zenodo. "
        "https://doi.org/10.5281/zenodo.0000000" in index
    )
    assert index.startswith("# Toy model of the decay of a substance\n")
    readme = (path / "README.md").read_text(encoding="utf-8")
    assert "# Toy model of the decay of a substance\n" in readme
    assert "*Toy model of the decay of a substance*" in readme
    assert "Toy PK model" not in index + readme


def test_docs_have_unix_line_endings(release: tuple[ProjectContext, Path]) -> None:
    """The pages of the release are the same on every platform."""
    context, path = release
    for written in write_docs(context, path):
        if written.suffix == ".md":
            assert b"\r" not in written.read_bytes(), written


def test_docs_replaced(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write(path / "docs" / "stale.md", "stale")
    write_docs(context, path)
    assert not (path / "docs" / "stale.md").exists()


def test_models_links(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_docs(context, path)
    models = (path / "docs" / "models.md").read_text(encoding="utf-8")
    raw = SETTINGS.raw_url
    # the report of SBML4humans takes the url of the model as query parameter
    report = quote(f"{raw}models/toy.xml", safe="")
    assert f"(https://sbml4humans.de/report?url={report})" in models
    assert f"({raw}models/toy.xml)" in models
    assert "https://github.com/sbmlsim/toy-model/blob/main/models/toy.md" in models
    assert (
        '![toy model diagram](images/toy.png){ style="max-height: 420px; '
        'width: auto; max-width: 100%" }' in models
    )
    assert f"({raw}toy_model.omex)" in models


def test_models_link_the_archive_of_the_model(
    release: tuple[ProjectContext, Path],
) -> None:
    context, path = release
    write(path / "a_stale.omex", b"omex")  # e.g. an ignored file of another build
    write_docs(context, path)
    models = (path / "docs" / "models.md").read_text(encoding="utf-8")
    assert "toy_model.omex" in models
    assert "a_stale.omex" not in models
    settings = dataclasses.replace(SETTINGS, archive=None)
    write_docs(dataclasses.replace(context, settings=settings), path)
    assert ".omex" not in (path / "docs" / "models.md").read_text(encoding="utf-8")


def test_simulations_links(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_docs(context, path)
    page = (path / "docs" / "simulations.md").read_text(encoding="utf-8")
    site = SETTINGS.site_url
    assert f"{site}simulations/index.html" in page
    assert f"{site}simulations_uncertainty/index.html" in page
    assert f"{site}simulations/Decay/Decay.html" in page
    assert f"{site}simulations/Slow/Slow.html" in page
    assert "| Experiment |\n| --- |\n" in page
    assert f"| [Decay]({site}simulations/Decay/Decay.html) |\n" in page
    assert "`Decay`" not in page
    assert "_figures" not in page
    assert "no experiment reports" not in page


def test_fit_links_the_methods_report(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    fit_id = adopted_ids(context)[0]
    write(path / BuildPaths.FITS / fit_id / "report.pdf", b"%PDF")
    write_docs(context, path)
    page = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    pdf = f"[Methods report (PDF)]({SETTINGS.site_url}fit/{fit_id}/report.pdf)"
    assert pdf in page


def test_running_recreates_the_fit_reports(
    release: tuple[ProjectContext, Path],
) -> None:
    context, path = release
    write_docs(context, path)
    page = (path / "docs" / "running.md").read_text(encoding="utf-8")
    command = "uv run python -m sbmlsim.run report specs/report.json --out"
    assert f"{command} results/report\n" in page
    assert "uv run python -m sbmlsim.run list tests.project_toy:PROJECT\n" in page
    write_docs(without_fits(context), path)
    assert "specs/report.json" not in (path / "docs" / "running.md").read_text(
        encoding="utf-8"
    )


def test_without_fits_no_confidence_interval(
    release: tuple[ProjectContext, Path],
) -> None:
    """A model without adopted fits has no simulations with a confidence interval."""
    context, path = release
    write_docs(context, path)
    assert "specs/uncertainty.json" in (path / "docs" / "running.md").read_text(
        encoding="utf-8"
    )
    write_docs(without_fits(context), path)
    assert "specs/uncertainty.json" not in (path / "docs" / "running.md").read_text(
        encoding="utf-8"
    )
    page = (path / "docs" / "simulations.md").read_text(encoding="utf-8")
    assert "simulations_uncertainty" not in page
    assert f"{SETTINGS.site_url}simulations/index.html" in page


def test_fit_table(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_docs(context, path)
    fit_id = adopted_ids(context)[0]
    record = AdoptedRecord.load(path / fits_prefix(context.project) / fit_id)
    page = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    assert f"{SETTINGS.site_url}fit/{fit_id}/index.html" in page
    # the build wrote no methods report
    assert "report.pdf" not in page
    for pid, parameter in record.parameters.items():
        (row,) = [line for line in page.splitlines() if line.startswith(f"| `{pid}`")]
        if parameter.ci_lower is None or parameter.ci_upper is None:
            assert "open" in row
        else:
            assert f"{parameter.ci_lower:.4g}" in row


def test_fit_table_is_a_block(release: tuple[ProjectContext, Path]) -> None:
    """The table of a fit is a block of its own, after the line of its links."""
    context, path = release
    write_docs(context, path)
    page = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    assert "/index.html)\n\n| Parameter |" in page
    fit_id = adopted_ids(context)[0]
    write(path / BuildPaths.FITS / fit_id / "report.pdf", b"%PDF")
    write_docs(context, path)
    page = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    assert "/report.pdf)\n\n| Parameter |" in page


def test_fit_table_open_interval(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    fit_id = adopted_ids(context)[0]
    directory = path / fits_prefix(context.project) / fit_id
    record = AdoptedRecord.load(directory)
    pid = next(iter(record.parameters))
    profiles = directory / "identifiability.json"
    text = profiles.read_text(encoding="utf-8")
    assert pid in text
    # open the interval of the parameter at both ends
    data = json.loads(text)
    entry = data["profiles"][pid]
    for key in ("ci_lower", "ci_upper"):
        entry[key] = None
    profiles.write_text(json.dumps(data), encoding="utf-8")
    reloaded = AdoptedRecord.load(directory).parameters[pid]
    if reloaded.ci_lower is not None:
        pytest.skip("the record format keeps the interval of the profile")
    write_docs(context, path)
    page = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    (row,) = [x for x in page.splitlines() if x.startswith(f"| `{pid}`")]
    assert "open - open" in row


def test_release_notes_sorted(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_docs(context, path)
    page = (path / "docs" / "release-notes.md").read_text(encoding="utf-8")
    order = [page.index(f"Notes {v}") for v in ("0.10.0", "0.9.0", "0.2.0")]
    assert order == sorted(order)
    assert "\n## Notes 0.10.0" in page


def badge_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("[![")]


def test_badges(release: tuple[ProjectContext, Path]) -> None:
    """The README and the documentation show the same badges at their top."""
    context, path = release
    write_docs(context, path)
    readme = (path / "README.md").read_text(encoding="utf-8")
    index = (path / "docs" / "index.md").read_text(encoding="utf-8")
    badges = badge_lines(readme)
    assert badges == badge_lines(index)
    repo = "https://github.com/sbmlsim/toy-model"
    expected = [
        "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)]"
        "(https://doi.org/10.5281/zenodo.0000000)",
        "[![Release](https://img.shields.io/github/v/release/sbmlsim/toy-model)]"
        f"({repo}/releases)",
        f"[![Documentation]({repo}/actions/workflows/docs.yml/badge.svg)]"
        f"({SETTINGS.site_url})",
        f"[![CI]({repo}/actions/workflows/ci.yml/badge.svg)]"
        f"({repo}/actions/workflows/ci.yml)",
        "[![Docker](https://img.shields.io/docker/v/sbmlsim/toy?sort=semver&label=docker)]"
        "(https://hub.docker.com/r/sbmlsim/toy)",
        "[![Python](https://img.shields.io/badge/python-3.14%20%7C%203.15-blue)]"
        "(https://www.python.org)",
        "[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue)]"
        "(https://opensource.org/license/MIT)",
        "[![Model license: CC BY-SA 4.0]"
        "(https://img.shields.io/badge/model%20license-CC%20BY--SA%204.0-lightgrey)]"
        "(https://creativecommons.org/licenses/by-sa/4.0/)",
    ]
    assert badges == expected
    # one paragraph of badges, on top of the README and below the title of the docs
    block = "\n".join(expected) + "\n\n"
    assert readme.startswith(block)
    assert index.startswith(f"# {SETTINGS.title}\n\n{block}")


def test_badges_image_of_another_registry(release: tuple[ProjectContext, Path]) -> None:
    """Only an image of Docker Hub has a badge, the registry of another is unknown."""
    context, path = release
    settings = dataclasses.replace(SETTINGS, docker_image="ghcr.io/sbmlsim/toy")
    write_docs(dataclasses.replace(context, settings=settings), path)
    readme = (path / "README.md").read_text(encoding="utf-8")
    assert "Docker" not in "\n".join(badge_lines(readme))


def test_readme_thin(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_docs(context, path)
    readme = (path / "README.md").read_text(encoding="utf-8")
    assert SETTINGS.site_url in readme
    assert len(readme.splitlines()) < 60
    assert (
        "uv run python -m sbmlsim.run simulate specs/simulate.json --out "
        "results/simulate\n" in readme
    )
    assert "10.5281/zenodo.0000000" in readme


@pytest.mark.slow
def test_zensical_strict(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    write_project_files(context, path)
    write_docs(context, path)
    result = subprocess.run(
        ["zensical", "build", "--clean", "--strict"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    site = path / "site"
    assert (site / "index.html").is_file()

    # the docs workflow copies the reports into the site, they must not
    # replace the pages of the docs
    write(path / BuildPaths.FITS / "fit1" / "index.html", "<html/>")
    for folder in (BuildPaths.SIMULATIONS, BuildPaths.FITS):
        shutil.copytree(path / folder, site / folder, dirs_exist_ok=True)
    # the figure of a model is bounded (attr_list), the overview is not
    models = (site / "models.html").read_text(encoding="utf-8")
    assert 'alt="toy model diagram"' in models
    assert "max-height: 420px" in models
    assert "max-height" not in (site / "index.html").read_text(encoding="utf-8")

    texts = [page.read_text(encoding="utf-8") for page in site.rglob("*.html")]
    assert any("Simulation report with the confidence interval" in t for t in texts)
    assert any("Fit report" in t for t in texts)


def _credibility_context(context: ProjectContext) -> ProjectContext:
    project = context.project.replace(credibility="tests.credibility_toy:CREDIBILITY")
    return dataclasses.replace(context, project=project)


def _write(path: Path, assessment: CredibilityAssessment) -> CredibilityAssessment:
    (path / BuildPaths.CREDIBILITY).mkdir(parents=True, exist_ok=True)
    assessment.to_json(path / BuildPaths.CREDIBILITY / "credibility.json")
    return assessment


def _write_credibility(
    path: Path, *, verified: bool = False, battery: Battery | None = None
) -> CredibilityAssessment:
    provenance = Provenance(
        model="credibility",
        title="Toy",
        fits=["DECAY_fit"],
        versions={"sbmlsim": "0.0.0"},
        commit="0123abc",
        dirty=False,
        date="2026-11-03T12:00:00+01:00",
    )
    fits = fit_table()
    verification = (
        verification_of(
            fits,
            factors={CHECKS["integrator"]: 1.02},
            problems=("the variant 'toy.xml' | has 1 issue",),
            battery=battery,
        )
        if verified
        else None
    )
    assessment = assess_table(
        CREDIBILITY, fits, provenance, verification, extensions=[ARMS]
    )
    return _write(path, assessment)


def _page(context: ProjectContext, path: Path) -> str:
    write_project_files(context, path)
    write_docs(context, path)
    return (path / "docs" / "credibility.md").read_text(encoding="utf-8")


def test_the_credibility_page(release: tuple[ProjectContext, Path]) -> None:
    context, path = release
    context = _credibility_context(context)
    assessment = _write_credibility(path)
    assert pages(context, path)[-2:] == (
        ("credibility", "Credibility"),
        ("release-notes", "Release notes"),
    )
    page = _page(context, path)
    assert page.startswith("# Credibility\n")
    for result in assessment.contexts:
        assert result.id in page
        assert result.verdict.label in page
        assert result.question in page
    assert f"{SETTINGS.site_url}credibility/index.html" in page
    assert "| Context of use | Title | Level | Risk | Verdict |" in page
    assert "within_2fold" in page
    assert f"{SETTINGS.site_url}credibility.html" in (path / "README.md").read_text(
        encoding="utf-8"
    )
    nav = tomllib.loads((path / "zensical.toml").read_text(encoding="utf-8"))[
        "project"
    ]["nav"]
    assert {"Credibility": "credibility.md"} in nav
    assert (
        nav.index({"Credibility": "credibility.md"}) == nav.index({"Fit": "fit.md"}) + 1
    )
    assert "\n\n\n" not in page


def test_the_credibility_page_shows_the_domain(
    release: tuple[ProjectContext, Path],
) -> None:
    """The claimed coverage, the statement and what decides a verdict."""
    context, path = release
    context = _credibility_context(context)
    assessment = _write_credibility(path)
    page = _page(context, path)
    assert "| Context of use | Title | Level | Risk | Verdict | Decided on |" in page
    assert (
        "| `describe` | Describe the decay | C1 | low | supported | calibration |"
        in page
    )
    dose = page[page.index("## dose") : page.index("## renal")]
    assert "| Field | Value | Coverage | Points |" in dose
    claimed = [e for e in assessment.coverage if e.context == "dose" and e.claimed]
    assert claimed
    for entry in claimed:
        assert entry.coverage is not None
        line = (
            f"| `{entry.field}` | `{entry.value}` | {entry.coverage.value} "
            f"| {entry.n_points} |"
        )
        assert line in dose
    renal = page[page.index("## renal") :]
    assert "| Quantity of interest | Description | Kind | Coverage | Points |" in renal
    assert "| `[A]` | the concentration of A | absolute |" in renal


def test_the_credibility_page_names_the_rules_of_an_extension(
    release: tuple[ProjectContext, Path],
) -> None:
    """A rule of an evidence extension is a row of the acceptance with its name."""
    context, path = release
    context = _credibility_context(context)
    _write_credibility(path)
    record = path / BuildPaths.CREDIBILITY / "credibility.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    rule = {"name": "pairs", "value": 2.0, "limit": ">= 1", "met": True}
    data["contexts"][0]["acceptance"].append({**rule, "extension": "arms"})
    record.write_text(json.dumps(data), encoding="utf-8")
    page = _page(context, path)
    assert "| `pairs` (arms) | 2 | >= 1 | yes |" in page


def test_the_credibility_page_shows_the_verification(
    release: tuple[ProjectContext, Path],
) -> None:
    context, path = release
    context = _credibility_context(context)
    _write_credibility(path, verified=True)
    page = _page(context, path)
    dose = page[page.index("## dose") : page.index("## renal")]
    assert "| Verification | Factor | Points | Largest difference | Result |" in dose
    assert "| tolerances | `discretization_error` |" in dose
    assert "| second integration method | `numerical_solver_error` |" in dose
    assert "0.02 | failed |" in dose
    assert (
        "| setup of the simulations | `use_error` | 10 |  | failed: the variant "
        "'toy.xml' \\| has 1 issue |" in dose
    )
    assert "not verified" in page
    assert "## Battery of the variants" not in page


def test_the_credibility_page_shows_the_battery(
    release: tuple[ProjectContext, Path],
) -> None:
    """The battery is a section of its own, not rows of every context."""
    context, path = release
    context = _credibility_context(context)
    _write_credibility(path, verified=True, battery=battery_of(CHECKS["steady_state"]))
    page = _page(context, path)
    contexts, battery = page.split("## Battery of the variants")
    assert "non-negativity" not in contexts
    assert "| Check | Factor | Cases | Largest deviation | Result |" in battery
    assert "| steady state | `numerical_code_verification` | 4 | 1e-09 | " in battery
    assert "failed: broken |" in battery
    assert "not applicable: The check cross_simulator. |" in battery


def test_the_credibility_page_escapes_a_table_cell(
    release: tuple[ProjectContext, Path],
) -> None:
    context, path = release
    context = _credibility_context(context)
    assessment = _write_credibility(path)
    piped = dataclasses.replace(
        assessment,
        contexts=[
            dataclasses.replace(assessment.contexts[0], title="Decay | elimination")
        ],
    )
    _write(path, piped)
    assert "| Decay \\| elimination |" in _page(context, path)


def test_no_credibility_page_without_a_record(
    release: tuple[ProjectContext, Path],
) -> None:
    context, path = release
    write_project_files(context, path)
    assert "credibility" not in [page for page, _ in pages(context, path)]
    write_docs(context, path)
    assert not (path / "docs" / "credibility.md").exists()
    assert "credibility" not in (path / "README.md").read_text(encoding="utf-8").lower()
    assert (
        "credibility"
        not in (path / "zensical.toml").read_text(encoding="utf-8").lower()
    )


class Hook(ReleaseHook):
    """A framework: the evidence of its extension, a page and its templates."""

    def __init__(self, templates: Path) -> None:
        self.directory = templates

    def templates(self) -> list[Path]:
        return [self.directory]

    def pages(self, context: ProjectContext) -> list[tuple[str, str]]:
        return [("library", "The library")]

    def values(self, context: ProjectContext, release_path: Path) -> dict[str, Any]:
        return {"library": "the substances of the library"}

    def credibility(
        self, assessment: Assessment, data: dict[str, Any]
    ) -> dict[str, Any]:
        data["studies"] = ["StudyA", "StudyB"]
        for entry, result in zip(data["contexts"], assessment.contexts, strict=True):
            entry["pairs"] = len((result.section("arms") or {}).get("pairs", ()))
        return data


def test_the_release_hook_adds_to_the_site(
    release: tuple[ProjectContext, Path], tmp_path: Path
) -> None:
    """A hook adds pages, values and the parts of the credibility page."""
    context, path = release
    templates = tmp_path / "templates" / "docs"
    templates.mkdir(parents=True)
    (templates / "library.md.j2").write_text(
        "# Library\n\n{{ library }}\n", encoding="utf-8"
    )
    (templates / "credibility-context.md.j2").write_text(
        "{{ context.pairs }} pairs of arms of `{{ context.id }}`.\n\n", encoding="utf-8"
    )
    (templates / "credibility-sections.md.j2").write_text(
        '## Study data\n\n{{ credibility.studies | join(", ") }}\n\n', encoding="utf-8"
    )
    context = dataclasses.replace(
        _credibility_context(context), hook=Hook(tmp_path / "templates")
    )
    _write_credibility(path)
    assert [page for page, _ in pages(context, path)][-3:] == [
        "credibility",
        "library",
        "release-notes",
    ]
    page = _page(context, path)
    assert "pairs of arms of `renal`." in page
    assert "## Study data\n\nStudyA, StudyB\n" in page
    library = (path / "docs" / "library.md").read_text(encoding="utf-8")
    assert library == "# Library\n\nthe substances of the library\n"
    nav = tomllib.loads((path / "zensical.toml").read_text(encoding="utf-8"))[
        "project"
    ]["nav"]
    assert {"The library": "library.md"} in nav


def test_the_credibility_page_shows_the_parameter_uncertainty(
    release: tuple[ProjectContext, Path],
) -> None:
    from tests.credibility.test_assessment_analyses import _propagation

    context, path = release
    context = _credibility_context(context)
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.01)
    )
    _write(path, assessment)
    page = _page(context, path)
    dose = page[page.index("## dose") : page.index("## renal")]
    assert "Parameter uncertainty, the adopted fits at 256 draws" in dose
    assert "| `GMFE` | <= 1.5 | 1 | accepted |" in dose
    assert "| `GMFE` (rule) | `DECAY.k` " in dose
    assert "\n\n\n" not in page


def test_the_credibility_page_shows_the_variability(
    release: tuple[ProjectContext, Path],
) -> None:
    from tests.credibility.test_variability import _checks

    context, path = release
    context = _credibility_context(context)
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, variability=_checks(0.5)
    )
    _write(path, assessment)
    page = _page(context, path)
    dose = page[page.index("## dose") : page.index("## renal")]
    assert (
        "Variability, the visual predictive check of the decisive points with the "
        "variability of the subjects (`k` LogNormal(cv=0.2)), 16 subjects at the "
        "adopted parameters:"
    ) in dose
    assert "| narrow |" in dose
    assert "| `[A]` |" in dose
    assert "\n\n\n" not in page


def test_the_hook_adds_after_the_acceptance_of_a_context(
    release: tuple[ProjectContext, Path], tmp_path: Path
) -> None:
    """The part of the acceptance follows the acceptance table of every context."""
    context, path = release
    templates = tmp_path / "templates" / "docs"
    templates.mkdir(parents=True)
    (templates / "library.md.j2").write_text("# Library\n", encoding="utf-8")
    (templates / "credibility-acceptance.md.j2").write_text(
        "| Ratio of `{{ context.id }}` |\n| --- |\n\n", encoding="utf-8"
    )
    context = dataclasses.replace(
        _credibility_context(context), hook=Hook(tmp_path / "templates")
    )
    _write_credibility(path)
    page = _page(context, path)
    dose = page[page.index("## dose") : page.index("## renal")]
    acceptance = dose.index("| Acceptance criterion |")
    ratio = dose.index("| Ratio of `dose` |")
    assert acceptance < ratio
    assert "| yes |\n\n| Ratio of `dose` |" in dose
