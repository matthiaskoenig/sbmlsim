"""Tests of the project files of a release repository."""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PureWindowsPath

import pytest
import yaml
from jinja2 import UndefinedError

from sbmlsim.release.docs import write_docs
from sbmlsim.release.hooks import ReleaseHook
from sbmlsim.release.project import (
    UV_VERSION,
    PackageIndex,
    ProjectContext,
    project_context,
    render,
    write_project_files,
    zenodo_json,
)
from sbmlsim.release.settings import Author
from sbmlsim.release.source import ReleaseError, SourceRepo
from tests.project_toy import PROJECT

assert PROJECT.release is not None
SETTINGS = PROJECT.release
REFERENCE = "tests.project_toy:PROJECT"

PYPROJECT = """\
[project]
name = "source"
description = "Model of Łódź"
requires-python = ">=3.14"
classifiers = [
    "Programming Language :: Python :: 3.15",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: Python",
]
dependencies = ["sbmlsim>=0.9.7", "sbmlutils>=0.16.0"]
"""

LOCK = """\
version = 1

[[package]]
name = "source"
version = "0.1.0"
source = { editable = "." }

[[package]]
name = "sbmlsim"
version = "0.9.7"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pkpdsim"
version = "0.2.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "gitdep"
version = "1.0.0"
source = { git = "https://github.com/x/gitdep" }
"""


@pytest.fixture
def source(tmp_path: Path) -> SourceRepo:
    root = tmp_path / "source"
    root.mkdir()
    # TOML files are UTF-8, whatever the encoding of the locale
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "uv.lock").write_text(LOCK, encoding="utf-8")
    (root / ".python-version").write_text("3.14\n", encoding="utf-8")
    return SourceRepo(root=root, commit="abc")


@pytest.fixture
def context(source: SourceRepo) -> ProjectContext:
    return project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0")


def zensical(context: ProjectContext, path: Path) -> dict:
    """Write the docs into a release without a build and read `zensical.toml`."""
    write_docs(context, path)
    text = (path / "zensical.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]


def test_constraints_from_lock(context: ProjectContext) -> None:
    assert context.constraints == ("sbmlsim==0.9.7", "pkpdsim==0.2.0")
    assert context.dependencies == ("sbmlsim>=0.9.7", "sbmlutils>=0.16.0")
    assert context.python == "3.14"


def test_python_from_requires_python(source: SourceRepo) -> None:
    (source.root / ".python-version").unlink()
    assert (
        project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0").python == "3.14"
    )


def test_pythons_from_classifiers(context: ProjectContext) -> None:
    assert context.pythons == ("3.14", "3.15")


def test_pythons_without_classifiers(source: SourceRepo) -> None:
    text = "\n".join(
        line for line in PYPROJECT.splitlines() if "Programming Language" not in line
    )
    (source.root / "pyproject.toml").write_text(text, encoding="utf-8")
    assert project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0").pythons == (
        "3.14",
    )


def test_release_supports_the_pythons(context: ProjectContext) -> None:
    classifiers = tomllib.loads(render("pyproject.toml.j2", context))["project"][
        "classifiers"
    ]
    assert classifiers == [
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
    ]
    ci = render("workflows/ci.yml.j2", context)
    assert 'python-version: ["3.14", "3.15"]' in ci
    assert "python-version: ${{ matrix.python-version }}" in ci
    assert "FROM python:3.14-slim" in render("Dockerfile.j2", context)


def test_pyproject_parses(context: ProjectContext) -> None:
    data = tomllib.loads(render("pyproject.toml.j2", context))
    assert data["project"]["version"] == "0.1.0"
    assert data["project"]["name"] == "toy-model"
    assert data["project"]["requires-python"] == ">=3.14"
    assert "scripts" not in data["project"]
    assert "allow-direct-references" not in render("pyproject.toml.j2", context)
    assert "sources" not in data["tool"]["uv"]
    assert data["tool"]["uv"]["constraint-dependencies"] == [
        "sbmlsim==0.9.7",
        "pkpdsim==0.2.0",
    ]
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/tests"
    ]
    assert data["dependency-groups"]["docs"] == ["zensical"]


def test_pyproject_local_framework(source: SourceRepo) -> None:
    context = project_context(
        PROJECT, REFERENCE, SETTINGS, source, "0.1.0", Path("/x/sbmlsim")
    )
    data = tomllib.loads(render("pyproject.toml.j2", context))
    assert data["tool"]["uv"]["sources"]["sbmlsim"] == {
        "path": "/x/sbmlsim",
        "editable": True,
    }
    assert "sbmlsim==0.9.7" not in data["tool"]["uv"]["constraint-dependencies"]


def test_pyproject_local_framework_of_its_settings(source: SourceRepo) -> None:
    """The framework of the settings comes from the local checkout."""
    settings = dataclasses.replace(SETTINGS, framework="pkpdsim")
    context = project_context(
        PROJECT, REFERENCE, settings, source, "0.1.0", Path("/x/pkpdsim")
    )
    data = tomllib.loads(render("pyproject.toml.j2", context))
    assert data["tool"]["uv"]["sources"] == {
        "pkpdsim": {"path": "/x/pkpdsim", "editable": True}
    }
    assert data["tool"]["uv"]["constraint-dependencies"] == ["sbmlsim==0.9.7"]


def test_pyproject_local_framework_on_windows(context: ProjectContext) -> None:
    """A Windows path is a TOML string with forward slashes, which uv accepts."""
    windows = dataclasses.replace(
        context, framework_path=PureWindowsPath(r"C:\Users\me\sbmlsim")
    )
    data = tomllib.loads(render("pyproject.toml.j2", windows))
    assert data["tool"]["uv"]["sources"]["sbmlsim"]["path"] == "C:/Users/me/sbmlsim"


#: the index of the builds of the develop branch of libroadrunner
DEV_INDEX = "https://matthiaskoenig.github.io/roadrunner-wheels/simple"


def test_an_index_of_the_source_lock_is_an_explicit_index_of_the_release(
    source: SourceRepo,
) -> None:
    """libroadrunner of the develop index comes from it in the release project as well."""
    (source.root / "pyproject.toml").write_text(
        PYPROJECT
        + '\n[tool.uv.sources]\nlibroadrunner = { index = "roadrunner-dev" }\n'
        + f'\n[[tool.uv.index]]\nname = "roadrunner-dev"\nurl = "{DEV_INDEX}/"\n'
        + "explicit = true\n",
        encoding="utf-8",
    )
    (source.root / "uv.lock").write_text(
        LOCK
        + '\n[[package]]\nname = "libroadrunner"\nversion = "2.11.1.dev655"\n'
        + f'source = {{ registry = "{DEV_INDEX}/" }}\n',
        encoding="utf-8",
    )
    context = project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0")
    assert context.indexes == (
        PackageIndex("roadrunner-dev", DEV_INDEX, ("libroadrunner",)),
    )
    # a source applies to a direct dependency only
    assert context.dependencies == (
        "sbmlsim>=0.9.7",
        "sbmlutils>=0.16.0",
        "libroadrunner",
    )
    assert "libroadrunner==2.11.1.dev655" in context.constraints
    data = tomllib.loads(render("pyproject.toml.j2", context))
    assert data["tool"]["uv"]["sources"] == {
        "libroadrunner": {"index": "roadrunner-dev"}
    }
    assert data["tool"]["uv"]["index"] == [
        {"name": "roadrunner-dev", "url": DEV_INDEX, "explicit": True}
    ]


def test_a_source_lock_from_pypi_has_no_index(context: ProjectContext) -> None:
    assert context.indexes == ()
    data = tomllib.loads(render("pyproject.toml.j2", context))
    assert "index" not in data["tool"]["uv"]
    assert "sources" not in data["tool"]["uv"]


def test_pyproject_one_entry_per_line(source: SourceRepo) -> None:
    context = project_context(
        PROJECT, REFERENCE, SETTINGS, source, "0.1.0", Path("/x/sbmlsim")
    )
    text = render("pyproject.toml.j2", context)
    assert (
        '\ndependencies = [\n    "sbmlsim>=0.9.7",\n    "sbmlutils>=0.16.0",\n]\n'
        in text
    )
    assert '\nconstraint-dependencies = [\n    "pkpdsim==0.2.0",\n]\n' in text
    assert '\nauthors = [\n    { name = "Doe, Jane" },\n]\n' in text
    assert "]\n\n[tool.uv.sources]\n" in text


def test_zenodo_json(context: ProjectContext) -> None:
    data = zenodo_json(context)
    assert data["creators"] == [
        {"orcid": "0000-0000-0000-0000", "affiliation": "Lab", "name": "Doe, Jane"}
    ]
    assert "No funding." in data["description"]
    assert "https://github.com/sbmlsim/toy-model/issues" in data["description"]
    assert data["license"] == "MIT"
    settings = dataclasses.replace(SETTINGS, authors=(Author("Roe, Rick"),))
    anonymous = dataclasses.replace(context, settings=settings)
    assert zenodo_json(anonymous)["creators"] == [{"name": "Roe, Rick"}]


def test_workflows_replaced(context: ProjectContext, tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "python.yml").write_text("old", encoding="utf-8")
    written = write_project_files(context, tmp_path)
    assert not (workflows / "python.yml").exists()
    assert {p.name for p in workflows.iterdir()} == {"ci.yml", "docs.yml"}
    assert tmp_path / ".zenodo.json" in written
    assert (tmp_path / ".python-version").read_text(encoding="utf-8") == "3.14\n"
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith(
        "# The image is built and pushed by the publishing of the release "
        "(python -m sbmlsim.release publish).\n\nFROM python:3.14-slim\n"
    )
    assert "COPY specs /code/specs\n" in dockerfile
    assert "docker build" not in dockerfile
    assert "${{ steps.deployment.outputs.page_url }}" in (
        workflows / "docs.yml"
    ).read_text(encoding="utf-8")
    assert not (tmp_path / "zensical.toml").exists()
    site = zensical(context, tmp_path)
    assert site["nav"]
    assert site["site_author"] == "Jane Doe"
    assert site["copyright"].endswith(" Jane Doe")


def test_no_em_dash(context: ProjectContext, tmp_path: Path) -> None:
    for path in write_project_files(context, tmp_path):
        assert "\u2014" not in path.read_text(encoding="utf-8"), path


def test_project_files_have_unix_line_endings(
    context: ProjectContext, tmp_path: Path
) -> None:
    """The files of the release are the same on every platform."""
    for path in write_project_files(context, tmp_path):
        assert b"\r" not in path.read_bytes(), path


class TemplatesHook(ReleaseHook):
    """A hook with a directory of templates."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def templates(self) -> list[Path]:
        return [self.directory]


def test_strict_undefined(context: ProjectContext, tmp_path: Path) -> None:
    (tmp_path / "x.j2").write_text("{{ unknown }}", encoding="utf-8")
    hooked = dataclasses.replace(context, hook=TemplatesHook(tmp_path))
    with pytest.raises(UndefinedError):
        render("x.j2", hooked)


def test_a_hook_replaces_and_adds_templates(
    context: ProjectContext, tmp_path: Path
) -> None:
    """The templates of the hook are searched before the ones of sbmlsim."""
    templates = tmp_path / "templates"
    (templates / "workflows").mkdir(parents=True)
    (templates / "gitignore.j2").write_text("/mine/\n", encoding="utf-8")
    (templates / "workflows" / "own.yml.j2").write_text(
        "{{ framework }} {{ package }}\n", encoding="utf-8"
    )
    hooked = dataclasses.replace(context, hook=TemplatesHook(templates))
    assert render("gitignore.j2", hooked) == "/mine/\n"
    assert render("workflows/own.yml.j2", hooked) == "sbmlsim tests.project_toy\n"
    assert render("gitignore.j2", context) != "/mine/\n"


def test_zenodo_file_is_json(context: ProjectContext, tmp_path: Path) -> None:
    write_project_files(context, tmp_path)
    zenodo = json.loads((tmp_path / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["title"] == "Toy model of the decay of a substance"
    site = zensical(context, tmp_path)
    assert site["site_name"] == "Toy model of the decay of a substance"


def test_the_steps_of_the_build(context: ProjectContext) -> None:
    """A project without adopted fits has no reports and no confidence interval."""
    assert [step.name for step in context.steps] == [
        "factory",
        "simulate",
        "credibility",
    ]
    assert context.steps[-1].spec == {"project": REFERENCE, "source_commit": "abc"}
    assert context.package == "tests.project_toy"
    assert context.top_package == "tests"


def test_forked_package_not_constrained(source: SourceRepo) -> None:
    fork = (
        '\n[[package]]\nname = "numpy"\nversion = "1.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        '\n[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    (source.root / "uv.lock").write_text(LOCK + fork, encoding="utf-8")
    constraints = project_context(
        PROJECT, REFERENCE, SETTINGS, source, "0.1.0"
    ).constraints
    assert constraints == ("sbmlsim==0.9.7", "pkpdsim==0.2.0")


def test_workflows_are_yaml(context: ProjectContext) -> None:
    ci = yaml.safe_load(render("workflows/ci.yml.j2", context))
    docs = yaml.safe_load(render("workflows/docs.yml.j2", context))
    runs = [s.get("run", "") for s in ci["jobs"]["simulate"]["steps"]]
    assert (
        "uv run python -m sbmlsim.run factory specs/factory.json --out "
        "results/factory" in runs
    )
    assert f"uv run python -m sbmlsim.run list {REFERENCE} --experiments" in runs
    assert "deploy" in docs["jobs"]
    # a tag push does not run the tests again (YAML 1.1 reads the key `on` as True)
    assert ci[True]["push"] == {"branches": ["main"]}


def test_uv_version(context: ProjectContext) -> None:
    """The Docker image and the workflows use one uv, at least the fix of the
    CA store (GHSA-hq85-5cpj-mr62)."""
    assert UV_VERSION == "0.13.6"
    dockerfile = render("Dockerfile.j2", context)
    assert "COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv\n" in dockerfile
    for name in ("ci.yml.j2", "docs.yml.j2"):
        workflow = yaml.safe_load(render(f"workflows/{name}", context))
        steps = [
            step
            for job in workflow["jobs"].values()
            for step in job["steps"]
            if step.get("uses", "").startswith("astral-sh/setup-uv@")
        ]
        assert steps, name
        assert {step["with"]["version"] for step in steps} == {"0.13.6"}, name


@pytest.mark.skipif(sys.platform == "win32", reason="the workflow runs on linux")
@pytest.mark.skipif(shutil.which("uv") is None, reason="the step runs uv")
def test_ci_workflow_puts_libpython_on_the_library_path(
    context: ProjectContext, tmp_path: Path
) -> None:
    """The extension of libroadrunner links against libpython, which the
    interpreters of uv keep in their `lib`: the step puts it on the library
    search path before the first command which imports roadrunner."""
    ci = yaml.safe_load(render("workflows/ci.yml.j2", context))
    runs = [s.get("run", "") for s in ci["jobs"]["simulate"]["steps"]]
    (index,) = [k for k, run in enumerate(runs) if "LD_LIBRARY_PATH" in run]
    first = next(k for k, run in enumerate(runs) if run.startswith("uv run "))
    assert index < first
    env = tmp_path / "github_env"
    result = subprocess.run(
        ["bash", "-e", "-c", runs[index]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_ENV": str(env)},
    )
    assert result.returncode == 0, result.stderr
    (line,) = env.read_text(encoding="utf-8").splitlines()
    assert line.startswith("LD_LIBRARY_PATH=/")
    assert line.split("=", 1)[1].split(":")[0].endswith("/lib")


def test_ci_workflow_installs_hdf5_before_the_dependencies(
    context: ProjectContext,
) -> None:
    """h5py has no wheel for every python of the matrix, and its build from the
    source distribution needs the library of HDF5 on the runner."""
    ci = yaml.safe_load(render("workflows/ci.yml.j2", context))
    runs = [s.get("run", "") for s in ci["jobs"]["simulate"]["steps"]]
    (index,) = [k for k, run in enumerate(runs) if "libhdf5-dev" in run]
    sync = runs.index("uv sync --locked")
    assert index < sync


@pytest.mark.skipif(sys.platform == "win32", reason="the workflow runs on linux")
@pytest.mark.parametrize(
    "folders",
    [
        ("simulations", "simulations_uncertainty", "fit"),
        ("simulations", "simulations_uncertainty"),
    ],
)
def test_docs_workflow_copies_existing_reports(
    context: ProjectContext, tmp_path: Path, folders: tuple[str, ...]
) -> None:
    """A model without adopted fits has no `fit` folder in the checkout."""
    docs = yaml.safe_load(render("workflows/docs.yml.j2", context))
    (step,) = [s for s in docs["jobs"]["build"]["steps"] if "site/" in s.get("run", "")]
    for folder in folders:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "index.html").write_text("<html/>", encoding="utf-8")
    (tmp_path / "site").mkdir()
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    copied = {p.parent.name for p in (tmp_path / "site").rglob("index.html")}
    assert copied == set(folders)


def test_gitignore(context: ProjectContext, tmp_path: Path) -> None:
    write_project_files(context, tmp_path)
    ignored = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"/results/", "/build/", "/dist/", "/site/", ".venv/", ".idea/"} <= set(
        ignored
    )
    assert not {"results/", "build/", "dist/", "site/"} & set(ignored)


def test_readme_for_the_build(context: ProjectContext, tmp_path: Path) -> None:
    """The build installs the project, whose `readme` must exist before the docs."""
    written = write_project_files(context, tmp_path)
    readme = tmp_path / "README.md"
    assert readme in written
    assert readme.read_text(encoding="utf-8") == f"# {SETTINGS.title}\n"
    readme.write_text("the README of the last release\n", encoding="utf-8")
    assert readme not in write_project_files(context, tmp_path)
    assert readme.read_text(encoding="utf-8") == "the README of the last release\n"


@pytest.mark.parametrize(
    ("first_year", "years"), [(2026, "2026"), (2024, "2024-2026"), (2027, "2027")]
)
def test_copyright_years(
    context: ProjectContext, tmp_path: Path, first_year: int, years: str
) -> None:
    settings = dataclasses.replace(SETTINGS, first_year=first_year)
    context = dataclasses.replace(context, settings=settings, year=2026)
    assert context.copyright_years == years
    assert zensical(context, tmp_path)["copyright"] == (
        f"Copyright &copy; {years} Jane Doe"
    )


def test_source_without_requires_python(source: SourceRepo) -> None:
    pyproject = source.root / "pyproject.toml"
    pyproject.write_text('[project]\nname = "source"\n', encoding="utf-8")
    with pytest.raises(ReleaseError, match="requires-python"):
        project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0")
    pyproject.unlink()
    with pytest.raises(ReleaseError, match=r"no 'pyproject\.toml'"):
        project_context(PROJECT, REFERENCE, SETTINGS, source, "0.1.0")
