"""The documentation site and the README of a release repository.

The pages of the Zensical site (`docs/`), its configuration `zensical.toml`
and the thin `README.md` are rendered from the templates in
`sbmlsim/resources/release/docs` and `sbmlsim/resources/release` (and the ones of
the release hook of the project) with the content of the built release
repository: the models, the experiment reports, the records of the adopted fits
and the assessment of the credibility. The reports themselves are copied into
the site by the docs workflow and are linked by the URL of the site.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import quote

from sbmlsim.credibility.assessment import CredibilityAssessment
from sbmlsim.credibility.verification import PREDICTION_CHECKS, check_of
from sbmlsim.fit.records import AdoptedRecord
from sbmlsim.release.build import BuildPaths
from sbmlsim.release.notes import NOTES_DIR, fits_prefix
from sbmlsim.release.project import ProjectContext, render
from sbmlsim.release.source import ReleaseError

__all__ = [
    "PAGES",
    "Badge",
    "FitEntry",
    "FitRow",
    "ModelEntry",
    "badges",
    "citation",
    "pages",
    "write_docs",
]

PAGES: tuple[tuple[str, str], ...] = (
    ("index", "Home"),
    ("running", "Running the model"),
    ("models", "Models"),
    ("simulations", "Simulations"),
    ("fit", "Fit"),
    ("release-notes", "Release notes"),
)
"""The pages of the site with their titles, the entries of its navigation.

The page `credibility` of a model with an assessment of its contexts of use
and the pages of the release hook come before `release-notes`, see `pages`.
"""
_CREDIBILITY = ("credibility", "Credibility")

_IMAGES = "images"
_HEADING = re.compile(r"^(#+ )", re.MULTILINE)
_VERSION = re.compile(r"^\d+(\.\d+)*$")
_MIT = "https://opensource.org/license/MIT"
_CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"


@dataclass(frozen=True)
class ModelEntry:
    """A model of the release repository as the models page shows it.

    Attributes:
        stem: the name of the model file without extension.
        file: the file name of the SBML model.
        equations: if the model has an equations file.
        figure: the name of the figure in the images of the site, if any.
    """

    stem: str
    file: str
    equations: bool
    figure: str | None


@dataclass(frozen=True)
class FitRow:
    """A fitted parameter as the fit page shows it."""

    pid: str
    value: str
    unit: str
    interval: str
    identifiability: str


@dataclass(frozen=True)
class FitEntry:
    """An adopted fit as the fit page shows it."""

    id: str
    rows: list[FitRow]
    #: the build wrote the methods report of the fit, `fit/<fit id>/report.pdf`
    pdf: bool = False


@dataclass(frozen=True)
class Badge:
    """A badge of the README and of the documentation, an image with a link."""

    label: str
    image: str
    link: str


def _shield(label: str, message: str, color: str) -> str:
    """Get the URL of a static badge of shields.io."""

    def escape(text: str) -> str:
        return quote(text.replace("-", "--").replace("_", "__"), safe="")

    return f"https://img.shields.io/badge/{escape(label)}-{escape(message)}-{color}"


def _on_docker_hub(image: str) -> bool:
    """Check whether an image is one of Docker Hub, `owner/name` without a registry."""
    first, _, rest = image.partition("/")
    return bool(rest) and not ("." in first or ":" in first or first == "localhost")


def badges(context: ProjectContext) -> list[Badge]:
    """Get the badges of the release, shown at the top of the README and the docs.

    Args:
        context: the project context.

    Returns:
        The DOI, the latest release, the documentation (the status of its
        workflow, linked to the site), the tests, the Docker image (an image of
        Docker Hub only), the Python versions and the licenses of the code and of
        the models.
    """
    settings = context.settings
    repository = f"https://github.com/{settings.repository}"
    workflows = f"{repository}/actions/workflows"
    image = settings.docker_image
    return [
        Badge(
            "DOI",
            f"https://zenodo.org/badge/DOI/{settings.doi}.svg",
            f"https://doi.org/{settings.doi}",
        ),
        Badge(
            "Release",
            f"https://img.shields.io/github/v/release/{settings.repository}",
            f"{repository}/releases",
        ),
        Badge("Documentation", f"{workflows}/docs.yml/badge.svg", settings.site_url),
        Badge("CI", f"{workflows}/ci.yml/badge.svg", f"{workflows}/ci.yml"),
        *(
            [
                Badge(
                    "Docker",
                    f"https://img.shields.io/docker/v/{image}?sort=semver&label=docker",
                    f"https://hub.docker.com/r/{image}",
                )
            ]
            if _on_docker_hub(image)
            else []
        ),
        Badge(
            "Python",
            _shield("python", " | ".join(context.pythons), "blue"),
            "https://www.python.org",
        ),
        Badge("Code license: MIT", _shield("code license", "MIT", "blue"), _MIT),
        Badge(
            "Model license: CC BY-SA 4.0",
            _shield("model license", "CC BY-SA 4.0", "lightgrey"),
            _CC_BY_SA,
        ),
    ]


def _join(names: list[str], last: str) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + last + names[-1]


def citation(context: ProjectContext) -> str:
    """Get the citation of the release.

    Args:
        context: the project context.

    Returns:
        The authors, the year, the title and the DOI of the Zenodo record.
    """
    settings = context.settings
    authors = _join([a.citation_name for a in settings.authors], " & ")
    return (
        f"{authors} ({context.year}). *{context.settings.title}*. Zenodo. "
        f"https://doi.org/{settings.doi}"
    )


def _number(value: float | None) -> str:
    return "open" if value is None else f"{value:.4g}"


def _fit_entries(context: ProjectContext, release_path: Path) -> list[FitEntry]:
    """Read the records of the adopted fits from the exported package."""
    fit_ids = _adopted_ids(context)
    if not fit_ids:
        return []
    directory = release_path / fits_prefix(context.project)
    entries = []
    for fit_id in fit_ids:
        try:
            record = AdoptedRecord.load(directory / fit_id)
        except (FileNotFoundError, ValueError) as err:
            raise ReleaseError(
                f"The adopted fit '{fit_id}' has no readable record: {err}"
            ) from err
        rows = [
            FitRow(
                pid=pid,
                value=_number(p.value),
                unit=p.unit or "-",
                interval=f"{_number(p.ci_lower)} - {_number(p.ci_upper)}",
                identifiability=str(p.identifiability or "-"),
            )
            for pid, p in sorted(record.parameters.items())
        ]
        pdf = (release_path / BuildPaths.FITS / fit_id / "report.pdf").is_file()
        entries.append(FitEntry(fit_id, rows, pdf=pdf))
    return entries


def _adopted_ids(context: ProjectContext) -> list[str]:
    """Get the ids of the adopted fits of the project of a release."""
    try:
        return list(context.project.adopted_fits().values())
    except ValueError as err:
        raise ReleaseError(str(err)) from err


def pages(context: ProjectContext, release_path: Path) -> tuple[tuple[str, str], ...]:
    """Get the pages of the site of a release with their titles.

    Args:
        context: the project context.
        release_path: the folder of the release repository.

    Returns:
        `PAGES`, with `credibility` if the release has the record of the
        assessment of the contexts of use and then the pages of the release
        hook (`ReleaseHook.pages`) before `release-notes`.
    """
    added = []
    if _credibility_file(release_path).is_file():
        added.append(_CREDIBILITY)
    added.extend(context.hook.pages(context))
    index = [page for page, _ in PAGES].index("release-notes")
    return (*PAGES[:index], *added, *PAGES[index:])


def _credibility_file(release_path: Path) -> Path:
    return release_path / BuildPaths.CREDIBILITY / "credibility.json"


def _credibility(context: ProjectContext, release_path: Path) -> dict[str, Any] | None:
    """Read the assessment of the contexts of use as the page shows it.

    Returns:
        The assessment as plain data with, for every context, the verdict in words
        (`verdict_label`), the role of the evidence which decides it
        (`decided_on`), the coverage of its claimed values (`coverage`), its
        rules of acceptance, those of an evidence extension with the name of
        the extension in `extension`, and its checks of the predictions with
        their names (`label`), and the checks of the battery of the variants
        with their names (`battery`, empty if it did not run), as the release
        hook gives it (`ReleaseHook.credibility`); `None` for a release without
        one.

    Raises:
        ReleaseError: if the record cannot be read.
    """
    file = _credibility_file(release_path)
    if not file.is_file():
        return None
    try:
        assessment = CredibilityAssessment.from_json(file)
    except (ValueError, KeyError, TypeError) as err:
        raise ReleaseError(f"The assessment '{file}' is not readable: {err}") from err
    data = assessment.to_dict()
    prediction_checks = {c.id for c in PREDICTION_CHECKS}
    contexts = cast("list[dict[str, Any]]", data["contexts"])
    for entry, result in zip(contexts, assessment.contexts, strict=True):
        entry["verdict_label"] = result.verdict.label
        entry["decided_on"] = result.decisive_role.value
        entry["coverage"] = [
            coverage.to_dict()
            for coverage in assessment.coverage
            if coverage.context == result.id and coverage.claimed
        ]
        entry["verification"] = [
            {**record, "label": check_of(record["check"], record["factor"]).label}
            for record in cast("list[dict[str, Any]]", entry["verification"])
            if record["check"] in prediction_checks
        ]
    battery = (
        None if assessment.verification is None else assessment.verification.battery
    )
    data["battery"] = [
        {**result.to_dict(), "label": result.check.label, "factor": result.check.factor}
        for result in ([] if battery is None else battery.results)
    ]
    return context.hook.credibility(assessment, data)


def _copy_image(release_path: Path, name: str, images: Path) -> bool:
    """Copy a figure of the release into the images of the site, if it exists."""
    source = release_path / BuildPaths.FIGURES / name
    if not source.is_file():
        return False
    images.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, images / name)
    return True


def _models(release_path: Path, images: Path) -> list[ModelEntry]:
    entries = []
    for path in sorted((release_path / BuildPaths.MODELS).glob("*.xml")):
        figure = f"{path.stem}.png"
        entries.append(
            ModelEntry(
                stem=path.stem,
                file=path.name,
                equations=path.with_suffix(".md").is_file(),
                figure=figure if _copy_image(release_path, figure, images) else None,
            )
        )
    return entries


def _experiments(release_path: Path) -> list[str]:
    """Get the experiments of the simulation report.

    The report of sbmlsim has a folder per experiment with the page
    `<experiment>/<experiment>.html`; `_figures` holds the static figures of
    every experiment, which the build collects
    (`sbmlsim.project.collect_figure_files`), and is no experiment.
    """
    folder = release_path / BuildPaths.SIMULATIONS
    if not folder.is_dir():
        return []
    return sorted(
        path.name
        for path in folder.iterdir()
        if not path.name.startswith("_") and (path / f"{path.name}.html").is_file()
    )


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in path.stem.split("."))


def _release_notes(release_path: Path) -> list[str]:
    """Get the release notes, newest version first, with the headings demoted."""
    files = [
        path
        for path in (release_path / NOTES_DIR).glob("*.md")
        if _VERSION.match(path.stem)
    ]
    return [
        _HEADING.sub(r"#\1", path.read_text(encoding="utf-8").strip())
        for path in sorted(files, key=_version_key, reverse=True)
    ]


def write_docs(context: ProjectContext, release_path: Path) -> list[Path]:
    """Write the documentation site and the README into the release repository.

    The folder `docs` is replaced and `zensical.toml`, the configuration of the
    site with its navigation (`pages`), is written. The release repository
    must be built, the pages are made from its models, figures, reports, fit
    records, assessment of the credibility and release notes, with the values
    of the release hook (`ReleaseHook.values`).

    Args:
        context: the project context.
        release_path: the folder of the release repository.

    Returns:
        The written files: the pages, the configuration of the site, the README
        and the images of the site.

    Raises:
        ReleaseError: if the record of an adopted fit or the assessment of the
            contexts of use cannot be read.
    """
    docs = release_path / "docs"
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir()
    images = docs / _IMAGES

    figure = f"{context.project.name}_model.png"
    archive = context.settings.archive
    site_pages = pages(context, release_path)
    extra = {
        "figure": figure if _copy_image(release_path, figure, images) else None,
        "citation": citation(context),
        "badges": badges(context),
        "authors": context.settings.author_names,
        "models": _models(release_path, images),
        "omex": PurePosixPath(archive).name if archive is not None else None,
        "experiments": _experiments(release_path),
        "fits": _fit_entries(context, release_path),
        "fit_ids": _adopted_ids(context),
        "notes": _release_notes(release_path),
        "credibility": _credibility(context, release_path),
        "pages": site_pages,
    }
    extra.update(context.hook.values(context, release_path))

    written = []
    for page, _ in site_pages:
        path = docs / f"{page}.md"
        path.write_text(
            render(f"docs/{page}.md.j2", context, extra), encoding="utf-8", newline="\n"
        )
        written.append(path)
    configuration = release_path / "zensical.toml"
    configuration.write_text(
        render("zensical.toml.j2", context, extra), encoding="utf-8", newline="\n"
    )
    written.append(configuration)
    readme = release_path / "README.md"
    readme.write_text(
        render("README.md.j2", context, extra), encoding="utf-8", newline="\n"
    )
    written.append(readme)
    written.extend(sorted(images.glob("*")))
    return written
