"""Tests of the release notes, from the previous release and the new build."""

import dataclasses
import json
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pytest

from sbmlsim.project import Project
from sbmlsim.release.hooks import ReleaseHook
from sbmlsim.release.notes import (
    CoreSummary,
    DataChanges,
    FitChange,
    LibraryChange,
    core_summaries,
    data_changes,
    experiment_changes,
    fit_changes,
    github_releases,
    library_changes,
    model_changes,
    write_release_notes,
)
from sbmlsim.release.project import ProjectContext
from sbmlsim.release.source import ReleaseError, ReleaseRepo, Runner
from tests.project_toy import PROJECT, TOY_PATH
from tests.release.repos import git

OLD_FIT = "PK_20260101_000000__abcde"
NEW_FIT = "PK_20261016_185018__3efe3"
PD_FIT = "PD_20261016_185018__12345"
PACKAGE_DIR = "src/tests/project_toy"
DATA_DIR = f"{PACKAGE_DIR}/data"
DATA = [PurePosixPath("data")]
FITS_DIR = "src/tests/project_toy/fits"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def lock(*packages: tuple[str, str]) -> str:
    text = 'version = 1\n\n[[package]]\nname = "toy-model"\nversion = "0.1.0"\n'
    text += 'source = { editable = "." }\n'
    for name, version in packages:
        text += f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
        text += 'source = { registry = "https://pypi.org/simple" }\n'
    return text


OLD_LOCK = lock(
    ("numpy", "2.0.0"),
    ("pkpdsim", "0.6.0"),
    ("removed-package", "1.0.0"),
    ("sbmlsim", "0.9.6"),
    ("sbmlutils", "0.16.0"),
)
NEW_LOCK = lock(
    ("added-package", "0.1.0"),
    ("numpy", "2.1.0"),
    ("pkpdsim", "0.7.0"),
    ("sbmlsim", "0.9.7"),
    ("sbmlutils", "0.16.0"),
)


pytestmark = pytest.mark.usefixtures("git_env")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parameters(values: dict[str, float]) -> str:
    return json.dumps({"sets": [{"sid": "best", "values": values, "cost": 1.0}]})


def toy_sbml() -> str:
    return (TOY_PATH / "models" / "toy.xml").read_text(encoding="utf-8")


def commit_release(path: Path, tag: str) -> ReleaseRepo:
    git(path, "add", "-A")
    git(path, "commit", "-m", f"release {tag}")
    git(path, "tag", tag)
    return ReleaseRepo(path=path, last_tag=tag)


#: the release settings of a framework on top of sbmlsim, pkpdsim
assert PROJECT.release is not None
FRAMEWORK = dataclasses.replace(
    PROJECT.release,
    framework="pkpdsim",
    core_libraries=("pkpdsim", "sbmlsim", "sbmlutils"),
)


@pytest.fixture
def model(tmp_path: Path) -> Project:
    """The toy project with its data folder and the adopted fit `NEW_FIT`."""
    adopted = tmp_path / "adopted.json"
    adopted.write_text(json.dumps({"fits": {"PK": NEW_FIT}}), encoding="utf-8")
    return PROJECT.replace(data_path=Path("data"), adopted=adopted, release=FRAMEWORK)


@pytest.fixture
def release(tmp_path: Path) -> ReleaseRepo:
    """A previous release 0.1.0 and the new build in the working tree."""
    path = tmp_path / "toy-model"
    path.mkdir()
    git(path, "init", "-b", "main")
    write(path / "uv.lock", OLD_LOCK)
    write(path / DATA_DIR / "drug" / "Same2000" / "Same2000.tsv", "a\n1\n")
    write(path / DATA_DIR / "drug" / "Changed2001" / "Changed2001.tsv", "a\n1\n")
    write(path / DATA_DIR / "drug" / "Removed2002" / "Removed2002.tsv", "a\n1\n")
    write(path / DATA_DIR / "collect_data.py", "print()\n")
    write(path / "simulations" / "A" / "index.html", "<html/>")
    write(path / "simulations" / "B" / "index.html", "<html/>")
    write(path / "simulations" / "_figures" / "a.svg", "<svg/>")
    write(path / "simulations" / "index.html", "<html/>")
    unset = '<parameter id="r" value="NaN" constant="false"/>\n<parameter '
    write(path / "models" / "toy.xml", toy_sbml().replace("<parameter ", unset, 1))
    write(path / "models" / "removed.xml", toy_sbml())
    write(path / "models" / "toy.md", "# toy")
    write(path / FITS_DIR / OLD_FIT / "parameters.json", parameters({"k": 0.2}))
    write(path / FITS_DIR / PD_FIT / "parameters.json", parameters({"e": 1.0}))
    repo = commit_release(path, "0.1.0")

    write(path / "uv.lock", NEW_LOCK)
    write(path / DATA_DIR / "drug" / "Changed2001" / "Changed2001.tsv", "a\n2\n")
    shutil.rmtree(path / DATA_DIR / "drug" / "Removed2002")
    write(path / DATA_DIR / "other" / "Added2003" / "Added2003.tsv", "a\n1\n")
    shutil.rmtree(path / "simulations" / "B")
    write(path / "simulations" / "C" / "index.html", "<html/>")
    changed = toy_sbml().replace('value="0.2"', 'value="0.3"')
    changed = changed.replace(
        "<parameter ",
        '<parameter id="k2" value="1" constant="true"/>\n'
        '<parameter id="r" constant="false"/>\n<parameter ',
        1,
    )
    write(path / "models" / "toy.xml", changed)
    (path / "models" / "removed.xml").unlink()
    write(path / "models" / "added.xml", toy_sbml())
    shutil.rmtree(path / FITS_DIR)
    write(path / FITS_DIR / NEW_FIT / "parameters.json", parameters({"k": 0.25}))
    return repo


@pytest.fixture
def old_layout(tmp_path: Path) -> ReleaseRepo:
    """A previous release of the old layout: no lock file, no fit records."""
    path = tmp_path / "old-model"
    path.mkdir()
    git(path, "init", "-b", "main")
    write(path / DATA_DIR / "drug" / "Same2000" / "Same2000.tsv", "a\n1\n")
    write(path / "simulations" / "A" / "index.html", "<html/>")
    write(path / "simulations" / "_figures" / "a.svg", "<svg/>")
    write(path / "models" / "toy.xml", toy_sbml())
    write(path / "fit" / "20260421_191243__251fc" / "index.html", "<html/>")
    repo = commit_release(path, "0.7.1")

    write(path / "uv.lock", NEW_LOCK)
    write(path / FITS_DIR / NEW_FIT / "parameters.json", parameters({"k": 0.25}))
    return repo


def context(
    model: Project,
    version: str,
    dependencies: tuple[str, ...] = (),
    hook: ReleaseHook | None = None,
) -> ProjectContext:
    assert model.release is not None
    return ProjectContext(
        project=model,
        reference="tests.project_toy:PROJECT",
        settings=model.release,
        version=version,
        python="3.14",
        pythons=("3.14",),
        dependencies=dependencies,
        constraints=(),
        framework_path=None,
        year=2026,
        requires_python=">=3.14",
        hook=hook or ReleaseHook(),
        source_commit=COMMIT,
    )


RELEASES = {
    "pkpdsim": {
        "0.5.0": "- too old. Not shown.\n",
        "0.6.0": "- the old version. Not shown.\n",
        "0.6.1": (
            "# Release notes\n\nIntro text.\n\n## New features\n"
            "- First of 0.6.1. Second sentence.\n"
            "  - nested, not shown\n"
            "- sixth bullet, over the limit.\n"
        ),
        "0.7.0": (
            "## Breaking changes\r\n- third bullet! More.\r\n- fourth bullet\r\n"
            "- fifth bullet? More.\r\n"
            "- `run_report` recreates the report, e.g. of a fit. More.\r\n"
        ),
        "0.8.0": "- newer than the release. Not shown.\n",
    },
    "sbmlsim": {
        "0.9.7": (
            "- **A bold sentence.** More.\n"
            "- **Bold** ([#1](https://x.org/1)). lowercase more.\n"
            "- Versions 1.5 and 2.0, i.e. two. More.\n"
        )
    },
}


def fake_fetch(repository: str) -> dict[str, str]:
    return RELEASES[repository]


def failing_fetch(repository: str) -> dict[str, str]:
    raise ReleaseError("'gh api' could not run")


def test_library_changes() -> None:
    assert library_changes(OLD_LOCK, NEW_LOCK) == [
        LibraryChange("added-package", None, "0.1.0"),
        LibraryChange("numpy", "2.0.0", "2.1.0"),
        LibraryChange("pkpdsim", "0.6.0", "0.7.0"),
        LibraryChange("removed-package", "1.0.0", None),
        LibraryChange("sbmlsim", "0.9.6", "0.9.7"),
    ]
    assert library_changes(None, NEW_LOCK) == []


LOCAL_PKPDSIM_LOCK = """\
version = 1

[[package]]
name = "pkpdsim"
source = { editable = "/home/user/pkpdsim" }

[[package]]
name = "sbmlsim"
version = "0.9.7"
source = { registry = "https://pypi.org/simple" }
"""


def test_library_changes_local_package_without_version() -> None:
    """uv locks a local package with a dynamic version without a version."""
    changes = library_changes(OLD_LOCK, LOCAL_PKPDSIM_LOCK)
    assert LibraryChange("pkpdsim", "0.6.0", "editable /home/user/pkpdsim") in changes


def test_data_changes(release: ReleaseRepo) -> None:
    assert data_changes(release, PACKAGE_DIR, DATA) == DataChanges(
        added=["data/other/Added2003"],
        removed=["data/drug/Removed2002"],
        changed=["data/drug/Changed2001"],
    )


def test_experiment_changes(release: ReleaseRepo) -> None:
    assert experiment_changes(release) == (["C"], ["B"])


def test_model_changes(release: ReleaseRepo) -> None:
    changes = {change.file: change for change in model_changes(release)}
    assert sorted(changes) == ["added.xml", "removed.xml", "toy.xml"]
    toy = changes["toy.xml"]
    assert toy.status == "changed"
    assert toy.added == {"species": 0, "reactions": 0, "parameters": 1}
    assert toy.removed == {"species": 0, "reactions": 0, "parameters": 0}
    assert toy.changed_parameters == [("k", 0.2, 0.3)]
    assert changes["added.xml"].status == "added"
    assert changes["added.xml"].added == {"species": 1, "reactions": 1, "parameters": 1}
    assert changes["removed.xml"].status == "removed"
    assert changes["removed.xml"].removed["reactions"] == 1


def computed_sbml(q: str, p: str) -> str:
    """The toy model with a parameter of an assignment rule and one of an
    initial assignment, whose value attributes are `q` and `p`."""
    computed = (
        f'<parameter id="q" value="{q}" constant="false"/>\n'
        f'<parameter id="p" value="{p}" constant="true"/>\n'
        "</listOfParameters>\n"
        '<listOfInitialAssignments><initialAssignment symbol="p">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>'
        "</initialAssignment></listOfInitialAssignments>\n"
        '<listOfRules><assignmentRule variable="q">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><cn>6.5</cn></math>'
        "</assignmentRule></listOfRules>"
    )
    return toy_sbml().replace("</listOfParameters>", computed, 1)


def test_model_changes_computed_parameters(tmp_path: Path) -> None:
    """The value attribute of a parameter which a rule or an initial assignment
    sets is not the value of the model, its change is no changed value."""
    path = tmp_path / "toy-model"
    path.mkdir()
    git(path, "init", "-b", "main")
    write(path / "models" / "toy.xml", computed_sbml(q="6499800", p="1"))
    release = commit_release(path, "0.1.0")
    write(path / "models" / "toy.xml", computed_sbml(q="6.5", p="2"))
    (change,) = model_changes(release)
    assert change.status == "changed"
    assert change.changed_parameters == []


def test_fit_changes(release: ReleaseRepo, model: Project) -> None:
    assert fit_changes(release, model) == [
        FitChange(PD_FIT, None, []),
        FitChange(OLD_FIT, NEW_FIT, [("k", 0.2, 0.25)]),
    ]


def test_core_summaries() -> None:
    changes = library_changes(OLD_LOCK, NEW_LOCK)
    assert core_summaries(changes, fake_fetch, FRAMEWORK.core_libraries) == [
        CoreSummary(
            "pkpdsim",
            "0.6.0",
            "0.7.0",
            [
                "third bullet!",
                "fourth bullet",
                "fifth bullet?",
                "`run_report` recreates the report, e.g. of a fit.",
                "First of 0.6.1.",
            ],
            total=6,
        ),
        CoreSummary(
            "sbmlsim",
            "0.9.6",
            "0.9.7",
            [
                "**A bold sentence.**",
                "**Bold** ([#1](https://x.org/1)).",
                "Versions 1.5 and 2.0, i.e. two.",
            ],
            total=3,
        ),
    ]


def test_core_summaries_added_library() -> None:
    changes = [LibraryChange("sbmlutils", None, "0.16.0")]
    assert core_summaries(changes, fetch=failing_fetch) == [
        CoreSummary("sbmlutils", None, "0.16.0", [])
    ]


def test_core_summaries_of_sbmlsim() -> None:
    """The core libraries of sbmlsim are sbmlsim and sbmlutils."""
    changes = library_changes(OLD_LOCK, NEW_LOCK)
    assert [s.name for s in core_summaries(changes, fake_fetch)] == ["sbmlsim"]


def test_core_summaries_fetch_fails() -> None:
    changes = library_changes(OLD_LOCK, NEW_LOCK)
    summaries = core_summaries(changes, failing_fetch, FRAMEWORK.core_libraries)
    assert [(s.name, s.fetched, s.bullets) for s in summaries] == [
        ("pkpdsim", False, []),
        ("sbmlsim", False, []),
    ]


def test_github_releases() -> None:
    calls: list[list[str]] = []

    def runner(args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        calls.append([str(a) for a in args])
        return '{"tag":"0.7.0","body":"- a"}\n{"tag":"v0.6.0","body":null}\n'

    assert github_releases("pkpdsim", runner) == {"0.7.0": "- a", "0.6.0": ""}
    assert calls[0][:3] == ["gh", "api", "repos/matthiaskoenig/pkpdsim/releases"]


def test_write_release_notes(release: ReleaseRepo, model: Project) -> None:
    path = write_release_notes(context(model, "0.2.0"), release, fetch=fake_fetch)
    assert path == release.path / "release-notes" / "0.2.0.md"
    # UTF-8 with Unix line endings on every platform
    text = path.read_bytes().decode("utf-8")
    assert "\r" not in text
    assert text.startswith("# Release notes for toy-model 0.2.0\n")
    assert "The changes since 0.1.0.\n" in text
    # the source package and the pkpdsim of the release lock
    assert f"Made from tests {COMMIT[:10]} with pkpdsim 0.7.0." in text
    for heading in ("Core libraries", "Libraries", "Data", "Analysis"):
        assert f"## {heading}\n" in text
    assert "pkpdsim 0.6.0 → 0.7.0 (latest 5 of 6):\n" in text
    assert "sbmlsim 0.9.6 → 0.9.7:\n" in text
    assert "third bullet!" in text
    assert "sixth bullet" not in text
    assert "| `numpy` | 2.0.0 | 2.1.0 |" in text
    assert "`data/other/Added2003`" in text
    assert "`data/drug/Removed2002`" in text
    assert "`data/drug/Changed2001`" in text
    assert "`k` 0.2 → 0.3" in text
    assert "`k` 0.2 → 0.25" in text
    assert "\u2014" not in text
    assert "\n\n\n" not in text


def test_write_release_notes_fetch_fails(release: ReleaseRepo, model: Project) -> None:
    text = write_release_notes(
        context(model, "0.2.0"), release, fetch=failing_fetch
    ).read_text(encoding="utf-8")
    assert "pkpdsim 0.6.0 → 0.7.0: the release notes could not be fetched." in text


def minimal_release(tmp_path: Path, old_lock: str) -> ReleaseRepo:
    """A previous release 0.1.0 with a lock file, a model and a fit record."""
    path = tmp_path / "toy-model"
    path.mkdir()
    git(path, "init", "-b", "main")
    write(path / "uv.lock", old_lock)
    write(path / "models" / "toy.xml", toy_sbml())
    write(path / FITS_DIR / NEW_FIT / "parameters.json", parameters({"k": 0.25}))
    return commit_release(path, "0.1.0")


def test_notes_no_changes(tmp_path: Path, model: Project) -> None:
    release = minimal_release(tmp_path, OLD_LOCK)
    text = write_release_notes(
        context(model, "0.1.1"), release, fetch=fake_fetch
    ).read_text(encoding="utf-8")
    assert text.count("No changes.") == 6


def test_notes_direct_dependencies(tmp_path: Path, model: Project) -> None:
    release = minimal_release(
        tmp_path, lock(("numpy", "2.0.0"), ("pint", "0.25"), ("idna", "3.11"))
    )
    write(
        release.path / "uv.lock",
        lock(("numpy", "2.1.0"), ("pint", "0.26"), ("idna", "3.20")),
    )
    dependencies = ("NumPy>=2.0", "pkpdsim[docs]>=0.7")
    text = write_release_notes(
        context(model, "0.1.1", dependencies), release, fetch=fake_fetch
    ).read_text(encoding="utf-8")
    libraries = text.split("## Libraries\n\n")[1].split("\n## Data")[0]
    assert libraries == (
        "| Package | Previous | New |\n"
        "| --- | --- | --- |\n"
        "| `numpy` | 2.0.0 | 2.1.0 |\n"
        "\n"
        "2 other packages changed.\n"
        "\n"
        "<details>\n"
        "<summary>All changed packages</summary>\n"
        "\n"
        "| Package | Previous | New |\n"
        "| --- | --- | --- |\n"
        "| `idna` | 3.11 | 3.20 |\n"
        "| `numpy` | 2.0.0 | 2.1.0 |\n"
        "| `pint` | 0.25 | 0.26 |\n"
        "\n"
        "</details>\n"
    )


@pytest.mark.parametrize(
    "content", ["{", '{"sets": []}', '{"sets": [{"sid": "best"}]}']
)
@pytest.mark.parametrize("where", ["tag", "worktree"])
def test_fit_changes_malformed_record(
    release: ReleaseRepo, model: Project, content: str, where: str
) -> None:
    if where == "tag":
        shutil.rmtree(release.path / FITS_DIR)
        write(release.path / FITS_DIR / OLD_FIT / "parameters.json", content)
        git(release.path, "add", "-A")
        git(release.path, "commit", "-m", "broken record")
        git(release.path, "tag", "0.1.1")
        release = ReleaseRepo(path=release.path, last_tag="0.1.1")
        write(release.path / FITS_DIR / NEW_FIT / "parameters.json", parameters({}))
        name = f"0.1.1:{FITS_DIR}/{OLD_FIT}/parameters.json"
    else:
        write(release.path / FITS_DIR / NEW_FIT / "parameters.json", content)
        name = str(release.path / FITS_DIR / NEW_FIT / "parameters.json")
    with pytest.raises(ReleaseError, match="malformed") as err:
        fit_changes(release, model)
    assert name in str(err.value)


def test_notes_old_layout(old_layout: ReleaseRepo, model: Project) -> None:
    text = write_release_notes(
        context(model, "0.8.0"), old_layout, fetch=failing_fetch
    ).read_text(encoding="utf-8")
    assert "no lock file in the previous release" in text
    assert "no fit record in the previous release" in text
    assert f"`{NEW_FIT}`" in text


def test_notes_no_previous_release(release: ReleaseRepo, model: Project) -> None:
    first = ReleaseRepo(path=release.path, last_tag=None)
    text = write_release_notes(
        context(model, "0.1.0"), first, fetch=fake_fetch
    ).read_text(encoding="utf-8")
    assert "First release." in text
    for item in (
        "data/drug/Same2000",
        "data/other/Added2003",
        "`A`",
        "`C`",
        "toy.xml",
        NEW_FIT,
    ):
        assert item in text
    assert "data/drug/Removed2002" not in text
    assert "no fit record in the previous release" not in text


def test_write_release_notes_of_sbmlsim(release: ReleaseRepo, model: Project) -> None:
    """A release of sbmlsim names sbmlsim and summarizes sbmlsim and sbmlutils."""
    project = model.replace(release=PROJECT.release)
    text = write_release_notes(
        context(project, "0.2.0"), release, fetch=fake_fetch
    ).read_text(encoding="utf-8")
    assert f"Made from tests {COMMIT[:10]} with sbmlsim 0.9.7." in text
    assert "sbmlsim 0.9.6 → 0.9.7:\n" in text
    assert "pkpdsim 0.6.0" not in text


class Hook(ReleaseHook):
    """A hook with the studies of PK-DB, `<substance>/<study>`, and a section."""

    def data_study(self, data_dir: PurePosixPath, path: PurePosixPath) -> str | None:
        return "/".join(path.parts[:2]) if len(path.parts) >= 3 else None

    def notes_sections(
        self, context: ProjectContext, release: ReleaseRepo, runner: Runner
    ) -> list[str]:
        return [f"## Framework\n\nThe notes of {context.settings.framework}.\n"]


def test_the_release_hook_adds_to_the_notes(
    release: ReleaseRepo, model: Project
) -> None:
    write(release.path / DATA_DIR / "drug" / "Same2000" / "deep" / "x.tsv", "a\n")
    plain = data_changes(release, PACKAGE_DIR, DATA)
    assert plain.added == ["data/drug/Same2000/deep", "data/other/Added2003"]
    hooked = data_changes(release, PACKAGE_DIR, DATA, study=Hook().data_study)
    assert hooked.changed == [
        "drug/Changed2001",
        "drug/Same2000",
    ]
    text = write_release_notes(
        context(model, "0.2.0", hook=Hook()), release, fetch=fake_fetch
    ).read_text(encoding="utf-8")
    assert text.endswith("\n\n## Framework\n\nThe notes of pkpdsim.\n")
    assert "\n\n\n" not in text


def test_the_data_of_the_project(model: Project) -> None:
    from sbmlsim.release.notes import data_dirs

    assert data_dirs(model) == DATA
    assert data_dirs(model.replace(data_path=None)) == []
    outside = model.replace(data_path=[Path("data"), TOY_PATH.parent / "data"])
    assert data_dirs(outside) == DATA


def test_the_studies_of_two_data_folders_do_not_merge(tmp_path: Path) -> None:
    """Two data folders with a study of one name are two studies."""
    path = tmp_path / "toy-model"
    path.mkdir()
    git(path, "init", "-b", "main")
    for folder in ("data/omeprazole", "data/ph"):
        write(path / PACKAGE_DIR / folder / "Cederberg1992" / "a.tsv", "a\n1\n")
    release = commit_release(path, "0.1.0")
    write(path / PACKAGE_DIR / "data/ph" / "Cederberg1992" / "a.tsv", "a\n2\n")
    folders = [PurePosixPath("data/omeprazole"), PurePosixPath("data/ph")]
    assert data_changes(release, PACKAGE_DIR, folders) == DataChanges(
        changed=["data/ph/Cederberg1992"]
    )

    class ByStudy(ReleaseHook):
        """Give a study of every data folder one name, as a hook may."""

        def data_study(
            self, data_dir: PurePosixPath, path: PurePosixPath
        ) -> str | None:
            return path.parts[0] if len(path.parts) > 1 else None

    merged = data_changes(release, PACKAGE_DIR, folders, study=ByStudy().data_study)
    assert merged == DataChanges(changed=["Cederberg1992"])
