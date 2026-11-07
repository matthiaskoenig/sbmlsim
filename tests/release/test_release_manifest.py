"""Tests of the release manifest and the pruning of the release repository."""

import json
from pathlib import Path

import pytest

from sbmlsim.release import manifest as manifest_module
from sbmlsim.release.manifest import (
    BUILD_MARKER,
    LEGACY_FRAMEWORK,
    Manifest,
    changed_by_build,
    check_file_sizes,
    check_not_ignored,
    finish_build,
    prune,
    start_build,
    stop_build,
)
from sbmlsim.release.source import ReleaseError, SourceRepo, run

LOCK = """version = 1

[[package]]
name = "sbmlsim"
version = "0.8.0"

[[package]]
name = "sbmlutils"
version = "0.9.2"
"""
SOURCE = SourceRepo(root=Path("/unused"), commit="abc123")
CORE = ("sbmlsim", "sbmlutils")


def write_manifest(
    root: Path, version: str, source: SourceRepo, written: list[Path]
) -> Manifest:
    """Write the manifest of a release of sbmlsim with its core libraries."""
    return manifest_module.write_manifest(
        root, version, source, written, framework="sbmlsim", core_libraries=CORE
    )


def matches(
    root: Path, version: str, commit: str, framework_version: str | None
) -> bool:
    """Tell whether the release holds the build of a release of sbmlsim."""
    return manifest_module.matches(root, version, commit, "sbmlsim", framework_version)


pytestmark = pytest.mark.usefixtures("git_env")


def write(root: Path, relative: str, text: str = "x\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, list[Path]]:
    """A release repository with an older commit and an uncommitted build."""
    root = tmp_path / "release"
    root.mkdir()
    run(["git", "init", "-b", "main"], root)
    write(root, ".gitignore", "results/\n.venv/\n")
    write(root, "release-notes/0.7.1.md", "old\n")
    write(root, "docs/RELEASE.md", "old\n")
    write(root, "src/pkg/run_hctz.py", "old\n")
    run(["git", "add", "-A"], root)
    run(["git", "commit", "-m", "old"], root)
    write(root, "results/x", "mine\n")
    write(root, ".venv/lib/y", "env\n")
    write(root, "simulations/a/index.html", "a\n")
    written = [
        write(root, "README.md", "readme\n"),
        write(root, "uv.lock", LOCK),
        write(root, ".gitignore", "results/\n.venv/\n"),
        write(root, "release-notes/0.8.0.md", "new\n"),
        root / "simulations",  # a stage may return a folder
    ]
    return root, written


def test_manifest_roundtrip(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    manifest = write_manifest(root, "0.8.0", SOURCE, written)
    assert (manifest.framework, manifest.framework_version) == ("sbmlsim", "0.8.0")
    assert manifest.libraries == {"sbmlsim": "0.8.0", "sbmlutils": "0.9.2"}
    assert "simulations/a/index.html" in manifest.files
    assert "release.json" not in manifest.files
    assert Manifest.read(root / "release.json") == manifest
    assert Manifest.read(root / "missing.json") is None
    assert b"\r" not in (root / "release.json").read_bytes()


def test_manifest_local_framework(release: tuple[Path, list[Path]]) -> None:
    """A local framework (`framework_path`) has no version in the lock."""
    root, written = release
    lock = LOCK.replace(
        'version = "0.8.0"', 'source = { editable = "/home/user/sbmlsim" }'
    )
    write(root, "uv.lock", lock)
    manifest = write_manifest(root, "0.8.0", SOURCE, written)
    assert manifest.framework_version == "editable /home/user/sbmlsim"
    assert not matches(root, "0.8.0", "abc123", "0.8.0")


def test_manifest_lacks_a_library(release: tuple[Path, list[Path]]) -> None:
    """The lock of a release has the framework and every core library."""
    root, written = release
    with pytest.raises(ReleaseError, match="lacks pkpdsim, other"):
        manifest_module.write_manifest(
            root,
            "0.8.0",
            SOURCE,
            written,
            framework="pkpdsim",
            core_libraries=("pkpdsim", "sbmlsim", "other"),
        )


def test_manifest_of_pkpdsim(release: tuple[Path, list[Path]]) -> None:
    """A manifest of pkpdsim 0.19 has the versions of its libraries as fields."""
    root, _ = release
    legacy = {
        "version": "0.8.0",
        "source_commit": "abc123",
        "pkpdsim": "0.19.0",
        "sbmlsim": "0.16.0",
        "sbmlutils": "0.17.1",
        "files": {"README.md": "0" * 64},
    }
    (root / "release.json").write_text(json.dumps(legacy), encoding="utf-8")
    manifest = Manifest.read(root / "release.json")
    assert manifest is not None
    assert (manifest.framework, manifest.framework_version) == (
        LEGACY_FRAMEWORK,
        "0.19.0",
    )
    assert manifest.libraries == {
        "pkpdsim": "0.19.0",
        "sbmlsim": "0.16.0",
        "sbmlutils": "0.17.1",
    }
    assert manifest.files == {"README.md": "0" * 64}
    # another framework, and the hash of the README differs
    assert not matches(root, "0.8.0", "abc123", "0.19.0")
    assert not manifest_module.matches(root, "0.8.0", "abc123", "pkpdsim", "0.19.0")
    (root / "release.json").write_text('{"version": "0.8.0"}', encoding="utf-8")
    with pytest.raises(ReleaseError, match="not a release manifest"):
        Manifest.read(root / "release.json")


def test_matches_after_build(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    write_manifest(root, "0.8.0", SOURCE, [p.relative_to(root) for p in written])
    assert matches(root, "0.8.0", "abc123", "0.8.0")


def test_hand_edit_breaks_match(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    write_manifest(root, "0.8.0", SOURCE, written)
    with (root / "README.md").open("a", encoding="utf-8") as f:
        f.write("x")
    assert not matches(root, "0.8.0", "abc123", "0.8.0")


def test_extra_file_breaks_match(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    write_manifest(root, "0.8.0", SOURCE, written)
    write(root, "extra.txt")
    assert not matches(root, "0.8.0", "abc123", "0.8.0")


def test_other_commit_breaks_match(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    write_manifest(root, "0.8.0", SOURCE, written)
    assert not matches(root, "0.8.0", "other", "0.8.0")
    assert not matches(root, "0.9.0", "abc123", "0.8.0")
    assert not matches(root, "0.8.0", "abc123", "0.8.1")
    assert not matches(root, "0.8.0", "abc123", None)


def test_prune_keeps_release_notes_and_ignored(
    release: tuple[Path, list[Path]],
) -> None:
    root, written = release
    removed = prune(root, written)
    assert (root / "release-notes" / "0.7.1.md").exists()
    assert (root / "results" / "x").exists()
    assert (root / ".venv" / "lib" / "y").exists()
    assert (root / ".git").is_dir()
    assert (root / "README.md").exists()
    assert not (root / "docs").exists()
    assert not (root / "src").exists()
    assert root / "docs" / "RELEASE.md" in removed
    assert root / "src" / "pkg" / "run_hctz.py" in removed


def test_empty_folder_of_ignored_directory_survives(
    release: tuple[Path, list[Path]],
) -> None:
    root, written = release
    (root / ".venv" / "include").mkdir()
    (root / "results" / "empty").mkdir()
    prune(root, written)
    assert (root / ".venv" / "include").is_dir()
    assert (root / "results" / "empty").is_dir()


def test_matches_after_prune_of_tracked_files(
    release: tuple[Path, list[Path]],
) -> None:
    root, written = release
    prune(root, written)
    write_manifest(root, "0.8.0", SOURCE, written)
    assert matches(root, "0.8.0", "abc123", "0.8.0")
    with (root / "README.md").open("a", encoding="utf-8") as f:
        f.write("x")
    assert not matches(root, "0.8.0", "abc123", "0.8.0")


def test_missing_written_path_is_a_release_error(
    release: tuple[Path, list[Path]],
) -> None:
    root, written = release
    with pytest.raises(ReleaseError, match=r"nope\.txt"):
        write_manifest(root, "0.8.0", SOURCE, [*written, Path("nope.txt")])


def test_changed_by_build(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    assert not changed_by_build(root, "0.8.0")
    write_manifest(root, "0.8.0", SOURCE, written)
    assert changed_by_build(root, "0.8.0")
    with (root / "README.md").open("a", encoding="utf-8") as f:
        f.write("edited by hand")
    assert changed_by_build(root, "0.8.0")
    assert not changed_by_build(root, "0.9.0")
    write(root, "extra.txt")
    assert not changed_by_build(root, "0.8.0")


def test_changed_by_build_clean(release: tuple[Path, list[Path]]) -> None:
    root, _ = release
    run(["git", "add", "-A"], root)
    run(["git", "commit", "-m", "all"], root)
    assert changed_by_build(root, "0.8.0")
    (root / "docs" / "RELEASE.md").unlink()
    assert changed_by_build(root, "0.8.0")


def test_unfinished_build_accepts_its_changes(
    release: tuple[Path, list[Path]],
) -> None:
    root, written = release
    write_manifest(root, "0.8.0", SOURCE, written)
    write(root, "extra.txt")
    assert not changed_by_build(root, "0.8.0")
    marker = root / ".git" / BUILD_MARKER
    start_build(root, "0.9.0")
    assert json.loads(marker.read_text(encoding="utf-8")) == {"version": "0.9.0"}
    # a killed build recorded no paths, every change may be its change
    assert changed_by_build(root, "0.8.0")
    assert changed_by_build(root, "1.0.0")
    stop_build(root)
    assert "extra.txt" in json.loads(marker.read_text(encoding="utf-8"))["changed"]
    assert changed_by_build(root, "1.0.0")
    write(root, "added_by_hand.txt")
    assert not changed_by_build(root, "1.0.0")
    (root / "added_by_hand.txt").unlink()
    finish_build(root)
    assert not marker.exists()
    assert not changed_by_build(root, "0.8.0")


def test_stop_build_without_marker(release: tuple[Path, list[Path]]) -> None:
    root, _ = release
    stop_build(root)
    assert not (root / ".git" / BUILD_MARKER).exists()


def test_marker_of_a_worktree(release: tuple[Path, list[Path]], tmp_path: Path) -> None:
    """In a worktree `.git` is a file, the marker lies in its git folder."""
    root, _ = release
    run(["git", "add", "-A"], root)
    run(["git", "commit", "-m", "all"], root)
    worktree = tmp_path / "worktree"
    run(["git", "worktree", "add", "-b", "other", str(worktree)], root)
    assert (worktree / ".git").is_file()
    start_build(worktree, "0.9.0")
    git_dir = Path(run(["git", "rev-parse", "--absolute-git-dir"], worktree).strip())
    assert (git_dir / BUILD_MARKER).is_file()
    write(worktree, "extra.txt")
    removed = prune(worktree, [])
    assert (worktree / ".git").is_file()
    assert worktree / "extra.txt" in removed
    finish_build(worktree)
    assert not (git_dir / BUILD_MARKER).exists()


def test_check_not_ignored(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    check_not_ignored(root, written)
    tracked = write(root, "src/pkg/fitting/results/data.xlsx")
    write(root, ".gitignore", "/results/\n.venv/\n")
    check_not_ignored(root, [*written, tracked])
    write(root, ".gitignore", "results/\n.venv/\n*.log\n")
    log = write(root, "simulations/a/run.log")
    with pytest.raises(ReleaseError) as err:
        check_not_ignored(root, [*written, tracked])
    assert "src/pkg/fitting/results/data.xlsx" in str(err.value)
    assert "simulations/a/run.log" in str(err.value)
    assert "README.md" not in str(err.value)
    assert log.is_file()


def test_check_file_sizes(release: tuple[Path, list[Path]]) -> None:
    root, written = release
    check_file_sizes(root, written, max_size=1000)
    large = write(root, "simulations/a/a_task.nc", "x" * 2000)
    with pytest.raises(ReleaseError) as err:
        check_file_sizes(root, [*written, large], max_size=1000)
    assert "simulations/a/a_task.nc (0 MB)" in str(err.value)
    assert "README.md" not in str(err.value)
