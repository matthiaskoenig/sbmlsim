"""The project files of a release repository, written from templates.

The files which are the same in structure for every model (`pyproject.toml`,
`Dockerfile`, the workflows, `.zenodo.json`) are rendered from the templates in
`sbmlsim/resources/release` with the settings of the model and the
dependencies of the source project. The templates of the release hook of the
project (`sbmlsim.release.hooks.ReleaseHook.templates`) are searched first, so
a framework replaces a template or adds one.
"""

from __future__ import annotations

import datetime
import html
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from jinja2 import Environment, FunctionLoader, StrictUndefined

from sbmlsim.release.build import BuildStep, build_steps
from sbmlsim.release.hooks import ReleaseHook
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError, SourceRepo

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = [
    "TEMPLATES",
    "UV_VERSION",
    "ProjectContext",
    "project_context",
    "render",
    "write_project_files",
    "zenodo_json",
]

UV_VERSION = "0.13.6"
"""The uv of a release repository: its Docker image and its workflows.

The latest uv when it was set; at least 0.13.4, the first uv whose bundled CA
store lacks the emSign roots (GHSA-hq85-5cpj-mr62).
"""

#: the templates of sbmlsim, below which a template is named by its path
TEMPLATES: Traversable = files("sbmlsim").joinpath("resources", "release")

_PYTHON_FLOOR = re.compile(r">=\s*(\d+\.\d+)")
_PYTHON_VERSION = re.compile(r"^\d+\.\d+")
_PYTHON_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")

_TEMPLATES = {
    "pyproject.toml": "pyproject.toml.j2",
    "Dockerfile": "Dockerfile.j2",
    ".gitignore": "gitignore.j2",
    ".python-version": "python-version.j2",
    ".github/workflows/ci.yml": "workflows/ci.yml.j2",
    ".github/workflows/docs.yml": "workflows/docs.yml.j2",
}
"""The files of the release repository by their template."""


@dataclass(frozen=True)
class ProjectContext:
    """The values the templates of the project files are rendered with.

    Attributes:
        project: the project which is released.
        reference: the project as `module:attribute`, which the specs of the
            build and the commands of the release repository name.
        settings: the release settings of the project.
        version: the version of the release.
        python: the Python version of the source, e.g. `"3.14"`: the Python of
            the Docker image and of the docs.
        pythons: the Python versions the release supports, those of the
            classifiers of the source, e.g. `("3.14", "3.15")`: the tests of the
            release repository run on every one.
        dependencies: the dependencies of the source project.
        constraints: `name==version` for every registry package of the source lock.
        framework_path: local checkout of the framework to use instead of the
            registry (`ReleaseSettings.framework`).
        year: the year of the release.
        requires_python: the `requires-python` of the source project.
        hook: the release hook of the project.
        source_commit: the commit of the source repository.
    """

    project: Project
    reference: str
    settings: ReleaseSettings
    version: str
    python: str
    pythons: tuple[str, ...]
    dependencies: tuple[str, ...]
    constraints: tuple[str, ...]
    framework_path: Path | None
    year: int
    requires_python: str
    hook: ReleaseHook = field(default_factory=ReleaseHook)
    source_commit: str | None = None

    @property
    def repo_name(self) -> str:
        """The name of the release repository."""
        return self.settings.repo_name

    @property
    def package(self) -> str:
        """The dotted name of the model package.

        Raises:
            ReleaseError: for a project without a package.
        """
        if not self.project.package:
            raise ReleaseError(
                f"The project '{self.project.name}' has no package, a release "
                "exports it."
            )
        return self.project.package

    @property
    def top_package(self) -> str:
        """The first component of the model package, the folder below `src`."""
        return self.package.split(".")[0]

    @property
    def copyright_years(self) -> str:
        """The years of the copyright, `2026` or `2024-2026`."""
        first = self.settings.first_year
        return str(first) if first >= self.year else f"{first}-{self.year}"

    @cached_property
    def steps(self) -> list[BuildStep]:
        """The steps of the build, see `sbmlsim.release.build.build_steps`.

        Computed once per context, the check of the distributions of the
        default variant reads its SBML.
        """
        return build_steps(self.project, self.reference, self.source_commit)


def _python_version(source: SourceRepo, requires_python: str) -> str:
    version_file = source.root / ".python-version"
    if version_file.exists():
        match = _PYTHON_VERSION.match(version_file.read_text("utf-8").strip())
        if match:
            return match.group(0)
    floor = _PYTHON_FLOOR.search(requires_python)
    if floor is None:
        raise ReleaseError(
            "The Python version of the source is unknown, add a '.python-version' "
            f"or a '>=X.Y' requires-python (is '{requires_python}')."
        )
    return floor.group(1)


def _python_versions(project: Mapping[str, Any], python: str) -> tuple[str, ...]:
    """Get the Python versions which the classifiers of the source name.

    Args:
        project: the `[project]` table of the source.
        python: the Python version of the source, the versions without a
            classifier of one.

    Returns:
        The versions in their order, `(python,)` without a classifier of one.
    """
    versions = {
        match.group(1)
        for classifier in project.get("classifiers", ())
        if (match := _PYTHON_CLASSIFIER.match(classifier))
    }
    if not versions:
        return (python,)
    return tuple(sorted(versions, key=lambda v: tuple(int(p) for p in v.split("."))))


def _constraints(
    source: SourceRepo, framework: str, framework_path: Path | None
) -> tuple[str, ...]:
    """Get `name==version` of the registry packages of the source lock.

    Packages locked at several versions (resolution forks) are skipped, as
    pins to two versions of one package cannot be satisfied; the framework is
    skipped when it comes from a local checkout.
    """
    lock_path = source.root / "uv.lock"
    if not lock_path.exists():
        raise ReleaseError(f"The source has no 'uv.lock' at {lock_path}.")
    lock = tomllib.loads(lock_path.read_text("utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        if "registry" not in package.get("source", {}):
            continue
        name = package["name"]
        if framework_path is not None and name == framework:
            continue
        versions.setdefault(name, set()).add(package["version"])
    # a package locked at several versions (resolution forks) gets no constraint
    constraints = [
        f"{name}=={next(iter(found))}"
        for name, found in versions.items()
        if len(found) == 1
    ]
    return tuple(constraints)


def project_context(
    project: Project,
    reference: str,
    settings: ReleaseSettings,
    source: SourceRepo,
    version: str,
    framework_path: Path | None = None,
) -> ProjectContext:
    """Read the dependencies of the source project into a project context.

    Args:
        project: the project which is released.
        reference: the project as `module:attribute`.
        settings: the release settings of the project.
        source: the source repository with `pyproject.toml` and `uv.lock`.
        version: the version of the release.
        framework_path: local checkout of the framework, which then replaces
            the registry package in the release project.

    Returns:
        The context of the templates.

    Raises:
        ReleaseError: if the source lacks `pyproject.toml`, its `requires-python`
            or `uv.lock`, or its Python version is unknown, or the release hook
            of the project is not resolved.
    """
    pyproject_path = source.root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ReleaseError(f"The source has no 'pyproject.toml' at {pyproject_path}.")
    table = tomllib.loads(pyproject_path.read_text("utf-8")).get("project", {})
    requires_python = table.get("requires-python")
    if not isinstance(requires_python, str):
        raise ReleaseError(
            f"'{pyproject_path}' has no requires-python in its [project] table."
        )
    python = _python_version(source, requires_python)
    try:
        hook = project.release_hook()
    except (ImportError, AttributeError, ValueError) as err:
        raise ReleaseError(f"The release hook of '{project.name}': {err}") from err
    return ProjectContext(
        project=project,
        reference=reference,
        settings=settings,
        version=version,
        python=python,
        pythons=_python_versions(table, python),
        dependencies=tuple(table.get("dependencies", ())),
        constraints=_constraints(source, settings.framework, framework_path),
        framework_path=framework_path,
        year=datetime.date.today().year,
        requires_python=requires_python,
        hook=hook,
        source_commit=source.commit,
    )


def _toml_string(value: object) -> str:
    """Quote a value as TOML basic string (JSON strings are valid TOML)."""
    return json.dumps(str(value), ensure_ascii=False)


def _query_value(value: object) -> str:
    """Quote a value as the value of a query parameter of a url, `/` included."""
    return quote(str(value), safe="")


def _markdown_cell(value: object) -> str:
    """Get a value as the text of a cell of a markdown table: `|` escaped, one line."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _loader(directories: Sequence[Path | Traversable]) -> Callable[[str], str | None]:
    """Get the loader of the templates of the directories, in their order."""

    def load(name: str) -> str | None:
        for root in directories:
            path = root.joinpath(*name.split("/"))
            if path.is_file():
                return path.read_text("utf-8")
        return None

    return load


def environment(hook: ReleaseHook | None = None) -> Environment:
    """Get the environment of the templates of a release.

    Args:
        hook: the release hook, whose templates are searched before the ones
            of sbmlsim (`TEMPLATES`).

    Returns:
        The environment with the filters `toml`, `query` and `cell`.
    """
    directories = [*(hook.templates() if hook is not None else ()), TEMPLATES]
    env = Environment(
        loader=FunctionLoader(_loader(directories)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["toml"] = _toml_string
    env.filters["query"] = _query_value
    env.filters["cell"] = _markdown_cell
    return env


def render(
    name: str, context: ProjectContext, extra: Mapping[str, Any] | None = None
) -> str:
    """Render a template of a release.

    Args:
        name: the path of the template below the directories of the templates,
            see `environment`.
        context: the values of the template, available by field name.
        extra: further values of the template, e.g. the changes of the release
            notes.

    Returns:
        The rendered text.

    Raises:
        jinja2.UndefinedError: if the template uses a value the context lacks.
    """
    values: dict[str, Any] = {
        "project": context.project,
        "reference": context.reference,
        "settings": context.settings,
        "version": context.version,
        "python": context.python,
        "pythons": context.pythons,
        "uv": UV_VERSION,
        "dependencies": context.dependencies,
        "constraints": context.constraints,
        "framework": context.settings.framework,
        # forward slashes on Windows too: a TOML string and a path which uv accepts
        "framework_path": (
            context.framework_path.as_posix() if context.framework_path else None
        ),
        "year": context.year,
        "requires_python": context.requires_python,
        "repo_name": context.repo_name,
        "package": context.package,
        "top_package": context.top_package,
        "copyright_years": context.copyright_years,
        "steps": {step.name: step for step in context.steps},
        **(extra or {}),
    }
    return environment(context.hook).get_template(name).render(**values)


def zenodo_json(context: ProjectContext) -> dict[str, Any]:
    """Get the metadata of the Zenodo record of the release.

    Args:
        context: the project context.

    Returns:
        The content of `.zenodo.json`.
    """
    settings = context.settings
    creators = []
    for author in settings.authors:
        creator: dict[str, str] = {}
        if author.orcid is not None:
            creator["orcid"] = author.orcid
        if author.affiliation is not None:
            creator["affiliation"] = author.affiliation
        creator["name"] = author.name
        creators.append(creator)
    issues = f"https://github.com/{settings.repository}/issues"
    description = (
        f"<p>{html.escape(settings.description, quote=False)}</p>\n"
        f'<p>If you have any questions or problems please <a href="{issues}">'
        "open an issue</a></p>\n"
        f"<h2>Funding</h2><p>{html.escape(settings.funding, quote=False)}</p>"
    )
    return {
        "upload_type": "software",
        "title": context.settings.title,
        "creators": creators,
        "description": description,
        "access_right": "open",
        "license": "MIT",
        "keywords": list(settings.keywords),
        "contributors": [],
        "communities": [],
        "grants": [],
    }


def write_project_files(context: ProjectContext, release_path: Path) -> list[Path]:
    """Write the project files into the release repository.

    Workflows in `.github/workflows` which are not written are removed. A
    repository without `README.md` gets one with the title of the model, the
    `readme` of `pyproject.toml` which the build installs; the docs stage
    replaces it, and writes the configuration of the site with its pages.

    Args:
        context: the project context.
        release_path: the folder of the release repository.

    Returns:
        The written files.
    """
    written: list[Path] = []
    for target, template in _TEMPLATES.items():
        path = release_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(template, context), encoding="utf-8", newline="\n")
        written.append(path)
    zenodo = release_path / ".zenodo.json"
    zenodo.write_text(
        json.dumps(zenodo_json(context), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    written.append(zenodo)
    for workflow in (release_path / ".github" / "workflows").glob("*"):
        if workflow.is_file() and workflow not in written:
            workflow.unlink()
    readme = release_path / "README.md"
    if not readme.is_file():
        readme.write_text(f"# {context.settings.title}\n", "utf-8", newline="\n")
        written.append(readme)
    return written
