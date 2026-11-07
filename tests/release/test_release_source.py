"""Tests of the source repository, the release repository and the export."""

import sys
from pathlib import Path

import pytest

from sbmlsim.project import Project
from sbmlsim.release import source as source_module
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import (
    ReleaseError,
    check_version,
    export_package,
    release_repo,
    run,
    source_framework,
    source_repo,
)
from tests.project_toy import PROJECT
from tests.release.repos import PACKAGE, SBMLSIM, git

assert PROJECT.release is not None
SETTINGS: ReleaseSettings = PROJECT.release
pytestmark = pytest.mark.usefixtures("git_env")


def test_source_dirty_refused(source_without_fits: Project) -> None:
    (source_without_fits.root / "new.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="commit the changes of"):
        source_repo(source_without_fits)


def test_source_commit(source_without_fits: Project) -> None:
    source = source_repo(source_without_fits)
    assert source.commit == git(source.root, "rev-parse", "HEAD").strip()
    assert source.root == source_without_fits.root.parent.parent.resolve()


def test_source_framework(source_without_fits: Project) -> None:
    """The framework of the release is the one the source locks."""
    source = source_repo(source_without_fits)
    assert source_framework(source, "sbmlsim") == SBMLSIM
    assert source_framework(source, "pkpdsim") is None


def test_release_repo_cloned_when_missing(
    tmp_path: Path, release_settings: ReleaseSettings
) -> None:
    path = tmp_path / "elsewhere" / "clone"
    repo = release_repo(release_settings, path)
    assert (repo.path / "README.md").exists()
    assert repo.last_tag == "0.1.0"
    assert repo.ahead == 0
    (path / "x.txt").write_text("x\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-m", "local")
    assert release_repo(release_settings, path).ahead == 1


def test_release_repo_cloned_from_a_checkout(
    tmp_path: Path,
    remote_and_clone: tuple[Path, Path],
    release_settings: ReleaseSettings,
) -> None:
    """A checkout is cloned with its tags, the clone fetches from its origin."""
    remote, clone = remote_and_clone
    path = tmp_path / "out" / "repository"
    repo = release_repo(release_settings, path, checkout=clone)
    assert repo.last_tag == "0.1.0"
    assert repo.ahead == 0
    origin = git(path, "remote", "get-url", "origin").strip()
    assert origin == f"file://{remote}"
    with pytest.raises(ReleaseError, match="does not exist"):
        release_repo(release_settings, tmp_path / "other", checkout=tmp_path / "none")


def test_release_repo_other_origin_refused(
    remote_and_clone: tuple[Path, Path], release_settings: ReleaseSettings
) -> None:
    _, clone = remote_and_clone
    git(clone, "remote", "set-url", "origin", "https://github.com/other/thing.git")
    with pytest.raises(ReleaseError) as err:
        release_repo(release_settings, clone)
    assert "other/thing" in str(err.value)
    assert "sbmlsim/toy-model" in str(err.value)


def test_release_repo_wrong_branch_refused(
    remote_and_clone: tuple[Path, Path], release_settings: ReleaseSettings
) -> None:
    _, clone = remote_and_clone
    git(clone, "checkout", "-b", "feature")
    with pytest.raises(ReleaseError, match="not 'main', which no build") as err:
        release_repo(release_settings, clone)
    assert f"remove the out {clone.parent} or give another one" in str(err.value)


def push_commit(tmp_path: Path, remote: Path, name: str = "new.txt") -> str:
    """Push a commit to the release repository from another clone."""
    other = tmp_path / f"other-{name}"
    git(tmp_path, "clone", str(remote), str(other))
    (other / name).write_text("new\n", encoding="utf-8")
    git(other, "add", "-A")
    git(other, "commit", "-m", "more")
    git(other, "push", "origin", "main")
    return git(other, "rev-parse", "HEAD").strip()


def test_a_reused_clone_behind_is_brought_to_origin(
    tmp_path: Path,
    remote_and_clone: tuple[Path, Path],
    release_settings: ReleaseSettings,
) -> None:
    """A clone of an earlier run is fast-forwarded to the release repository."""
    remote, clone = remote_and_clone
    head = push_commit(tmp_path, remote)
    repo = release_repo(release_settings, clone)
    assert git(clone, "rev-parse", "HEAD").strip() == head
    assert repo.ahead == 0


def test_a_fresh_clone_is_the_release_repository_whatever_the_checkout(
    tmp_path: Path,
    remote_and_clone: tuple[Path, Path],
    release_settings: ReleaseSettings,
) -> None:
    """The checkout is a cache: its branch, its commits and its tags are not cloned."""
    remote, clone = remote_and_clone
    head = push_commit(tmp_path, remote)  # the checkout is behind
    git(clone, "checkout", "-q", "-b", "feature")
    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "local")
    git(clone, "tag", "9.9.9")
    path = tmp_path / "out" / "repository"
    repo = release_repo(release_settings, path, checkout=clone)
    assert git(path, "branch", "--show-current").strip() == "main"
    assert git(path, "rev-parse", "HEAD").strip() == head
    assert (repo.ahead, repo.last_tag) == (0, "0.1.0")
    assert not (path / "local.txt").exists()
    assert git(clone, "rev-parse", "HEAD").strip() != head


def test_a_checkout_with_commits_on_main_which_are_not_pushed_is_refused(
    tmp_path: Path,
    remote_and_clone: tuple[Path, Path],
    release_settings: ReleaseSettings,
) -> None:
    remote, clone = remote_and_clone
    push_commit(tmp_path, remote)  # behind and ahead: the commit is still missing
    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "local")
    git(clone, "commit", "-q", "--allow-empty", "-m", "another")
    path = tmp_path / "out" / "repository"
    with pytest.raises(ReleaseError, match=r"has 2 commit\(s\) on main which"):
        release_repo(release_settings, path, checkout=clone)


def test_a_diverged_clone_is_refused(
    tmp_path: Path,
    remote_and_clone: tuple[Path, Path],
    release_settings: ReleaseSettings,
) -> None:
    remote, clone = remote_and_clone
    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "local")
    push_commit(tmp_path, remote)
    with pytest.raises(ReleaseError) as err:
        release_repo(release_settings, clone)
    message = str(err.value)
    assert f"{clone} has 1 local commit(s) and is behind origin/main by 1" in message
    assert f"remove the out {clone.parent} or give another one" in message


def test_check_version() -> None:
    check_version("0.2.0", "0.1.0")
    check_version("0.10.0", "0.9.0")
    check_version("0.1.0", None)
    with pytest.raises(ReleaseError):
        check_version("0.1.0", "0.1.0")
    with pytest.raises(ReleaseError):
        check_version("0.0.9", "0.1.0")
    with pytest.raises(ReleaseError, match="not a version"):
        check_version("1.0", None)


def test_run_failure_message(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="git") as err:
        run(["git", "rev-parse", "--verify", "nothing"], tmp_path)
    assert "git rev-parse --verify nothing" in str(err.value)


def test_run_stream(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """A streamed command writes to the terminal, its output is not returned."""
    script = "import sys; print('to stdout'); print('to stderr', file=sys.stderr)"
    assert run([sys.executable, "-c", script], tmp_path, stream=True) == ""
    captured = capfd.readouterr()
    assert "to stdout" in captured.out
    assert "to stderr" in captured.err
    with pytest.raises(ReleaseError, match="exit code 3"):
        run([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, stream=True)


def test_run_without_the_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A command does not inherit the virtual environment of the source, which
    the release runs in: uv warns about it in the release repository."""
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    script = "import os; print(os.environ.get('VIRTUAL_ENV'))"
    assert run([sys.executable, "-c", script], tmp_path).strip() == "None"
    run([sys.executable, "-c", script], tmp_path, stream=True)
    assert capfd.readouterr().out.strip() == "None"


def export(tmp_path: Path, project: Project) -> tuple[Path, list[Path]]:
    release = tmp_path / "release"
    release.mkdir()
    written = export_package(project, SETTINGS, source_repo(project), release)
    return release, written


def test_export_excludes(tmp_path: Path, source_without_fits: Project) -> None:
    release, written = export(tmp_path, source_without_fits)
    target = release / "src" / PACKAGE
    assert (target / "__init__.py").exists()
    assert not (target / "TODO.md").exists()
    assert not (target / "literature").exists()
    assert Path(f"src/{PACKAGE}/__init__.py") in written
    assert all(not p.is_absolute() for p in written)
    assert not (release / "src" / "__init__.py").exists()


def test_export_namespace(tmp_path: Path, source_without_fits: Project) -> None:
    project = source_without_fits.replace(package=f"pkg.{PACKAGE}")
    release = tmp_path / "release"
    (release / "src" / "pkg").mkdir(parents=True)
    (release / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (release / "src" / "__init__.py").write_text("", encoding="utf-8")
    export_package(project, SETTINGS, source_repo(project), release)
    assert not (release / "src" / "pkg" / "__init__.py").exists()
    assert not (release / "src" / "__init__.py").exists()
    assert (release / "src" / "pkg" / PACKAGE / "__init__.py").exists()


def test_export_replaces(tmp_path: Path, source_without_fits: Project) -> None:
    release = tmp_path / "release"
    stale = release / "src" / PACKAGE / "stale.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n", encoding="utf-8")
    export_package(
        source_without_fits, SETTINGS, source_repo(source_without_fits), release
    )
    assert not stale.exists()


def test_export_needs_a_package(tmp_path: Path, source_without_fits: Project) -> None:
    project = Project(name="toy", root=source_without_fits.root)
    with pytest.raises(ReleaseError, match="has no package"):
        export_package(project, SETTINGS, source_repo(project), tmp_path)


def test_release_repo_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new repository on GitHub has no commit, it has no release yet."""
    remote = tmp_path / "empty.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-b", "main")
    monkeypatch.setattr(
        source_module, "clone_url", lambda repository: f"file://{remote}"
    )
    path = tmp_path / "toy-model"
    repo = release_repo(SETTINGS, path)
    assert (repo.last_tag, repo.ahead) == (None, 0)
    (path / "x.txt").write_text("x\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-m", "the release commit, not pushed")
    assert release_repo(SETTINGS, path).ahead == 1


def test_release_repo_detached_refused(
    remote_and_clone: tuple[Path, Path], release_settings: ReleaseSettings
) -> None:
    _, clone = remote_and_clone
    git(clone, "checkout", "--detach")
    with pytest.raises(ReleaseError, match="on branch 'HEAD', not 'main'"):
        release_repo(release_settings, clone)
