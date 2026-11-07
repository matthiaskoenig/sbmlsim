"""The repositories of the tests of the release (`repos`).

`toy_records` fits the toy project once per process, `source_project` copies the
toy into a committed source repository with the records of the fit, and
`remote_and_clone` creates the release repository with its bare remote, which
`release_settings` makes the repository the release clones from.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from sbmlsim.fit.records import AdoptedRecords
from sbmlsim.project import Project
from sbmlsim.release import source as source_module
from sbmlsim.release.settings import ReleaseSettings
from tests.project_toy.records import adopt_toy_fit, toy_model_file
from tests.release.repos import (
    GIT_ENV,
    PACKAGE,
    SOURCE_LOCK,
    adopt_records,
    copy_toy,
    forget_package,
    git,
    init_repo,
    load_toy,
)


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run git without the configuration of the user, see `repos.GIT_ENV`."""
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(scope="session")
def toy_records(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of the adopted fit of the toy, fitted once per process."""
    out = tmp_path_factory.mktemp("toy_fit") / "out"
    adopt_toy_fit(out)
    return out


def _source(root: Path, records: Path | None, *, credibility: bool) -> Path:
    """Write the source repository with the toy package and commit it."""
    init_repo(root)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "toy"\nrequires-python = ">=3.14"\n', encoding="utf-8"
    )
    shutil.copy(SOURCE_LOCK, root / "uv.lock")
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    package = copy_toy(root / "pkg" / PACKAGE, credibility=credibility)
    if records is not None:
        fit_id = adopt_records(package, records)
        # the default variant with the value of the fitted rate and its
        # uncertainty, as a factory takes it from the records
        toy_model_file(
            package / "models" / "toy.xml",
            AdoptedRecords(package / "fits", [fit_id]),
        )
    (package / "TODO.md").write_text("todo\n", encoding="utf-8")
    (package / "literature").mkdir()
    (package / "literature" / "x.txt").write_text("x\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-m", "init")
    return package


@pytest.fixture
def source_project(
    tmp_path: Path,
    git_env: None,
    toy_records: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Project]:
    """Get the toy project with its adopted fit, committed in a monorepo."""
    package = _source(tmp_path / "monorepo", toy_records, credibility=True)
    monkeypatch.syspath_prepend(str(package.parent))
    yield load_toy()
    forget_package()


@pytest.fixture
def source_without_credibility(
    tmp_path: Path,
    git_env: None,
    toy_records: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Project]:
    """Get the toy project with its adopted fit and without its declaration of
    the credibility, a plain object which the kind `credibility` refuses."""
    package = _source(tmp_path / "monorepo", toy_records, credibility=False)
    monkeypatch.syspath_prepend(str(package.parent))
    yield load_toy()
    forget_package()


@pytest.fixture
def source_without_fits(
    tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Project]:
    """Get the toy project without adopted fits, committed in a monorepo."""
    package = _source(tmp_path / "monorepo", None, credibility=False)
    monkeypatch.syspath_prepend(str(package.parent))
    yield load_toy()
    forget_package()


@pytest.fixture
def remote_and_clone(tmp_path: Path, git_env: None) -> tuple[Path, Path]:
    """Get a bare remote and a clone of the release repository, released 0.1.0."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-b", "main")
    clone = tmp_path / "toy-model"
    git(tmp_path, "clone", f"file://{remote}", str(clone))
    git(clone, "checkout", "-B", "main")
    (clone / "README.md").write_text("readme\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "initial")
    git(clone, "tag", "0.1.0")
    git(clone, "push", "origin", "main", "--tags")
    return remote, clone


@pytest.fixture
def release_settings(
    monkeypatch: pytest.MonkeyPatch, remote_and_clone: tuple[Path, Path]
) -> ReleaseSettings:
    """Get the release settings of the toy, its repository is the bare remote."""
    remote, _ = remote_and_clone
    monkeypatch.setattr(
        source_module, "clone_url", lambda repository: f"file://{remote}"
    )
    from tests.project_toy import PROJECT

    assert PROJECT.release is not None
    return PROJECT.release
