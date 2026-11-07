"""Tests of the release of a model repository: the build into an out and its publishing.

The source and the release repository are real git repositories (`repos`); the
build tools (`uv`, the analyses of the run layer), docker and `gh` are replaced
by a fake runner which writes what they would write, the publishing steps by a
fake publisher.
"""

import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import pytest

from sbmlsim.fit.result import fit_subset
from sbmlsim.project import Project
from sbmlsim.release import source as source_module
from sbmlsim.release.manifest import BUILD_MARKER, MANIFEST_FILE, Manifest
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError
from sbmlsim.release.workflow import (
    REPOSITORY_DIR,
    ReleaseBuild,
    build_release,
    build_summary,
    check_build,
    prepare_release,
    publish_release,
    publishable,
)
from tests.project_toy import TOY_PATH
from tests.release.fakes import (
    PYPI,
    SBMLSIM,
    VERSION,
    FakePublisher,
    FakeRunner,
)
from tests.release.repos import PACKAGE, REFERENCE, git

pytestmark = pytest.mark.usefixtures("git_env")


@pytest.fixture
def model(source_project: Project, release_settings: ReleaseSettings) -> Project:
    """The toy project with its adopted fit and an archive, committed in a monorepo."""
    assert source_project.release is not None
    return source_project.replace(
        release=dataclasses.replace(source_project.release, archive="toy.omex")
    )


@pytest.fixture
def clone(remote_and_clone: tuple[Path, Path]) -> Path:
    """The checkout of the release repository with a file of an old layout."""
    _, clone = remote_and_clone
    (clone / "run_toy.py").write_text("print('old')\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "old layout")
    git(clone, "push", "origin", "main")
    return clone


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "out"


def build(
    model: Project,
    out: Path,
    clone: Path | None,
    runner: FakeRunner | None = None,
    *,
    version: str = VERSION,
    framework_path: Path | None = None,
) -> Path:
    """Build the release into the out, the repository cloned from the checkout."""
    return built(
        model, out, clone, runner, version=version, framework_path=framework_path
    )[0]


def built(
    model: Project,
    out: Path,
    clone: Path | None,
    runner: FakeRunner | None = None,
    *,
    version: str = VERSION,
    framework_path: Path | None = None,
) -> tuple[Path, ReleaseBuild]:
    """Build the release into the out, with what the build wrote and ran."""
    runner = runner or FakeRunner()
    prepared = prepare_release(model, version, out, checkout=clone, runner=runner)
    release = build_release(
        model,
        REFERENCE,
        version,
        prepared,
        framework_path=framework_path,
        workers=2,
        runner=runner,
    )
    return out / REPOSITORY_DIR, release


def publish(
    model: Project,
    out: Path,
    runner: FakeRunner | None = None,
    publisher: FakePublisher | None = None,
) -> list[str]:
    return publish_release(
        model,
        VERSION,
        out,
        runner=runner or FakeRunner(),
        publisher=publisher or FakePublisher(),
    )


def adopted_id(model: Project) -> str:
    (fit_id,) = model.adopted_fits().values()
    return fit_id


def test_missing_release_settings(model: Project, out: Path, clone: Path) -> None:
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match="has no release settings"):
        build(model.replace(release=None), out, clone, runner)
    assert model.release is not None
    incomplete = dataclasses.replace(model.release, doi="", funding="")
    with pytest.raises(ReleaseError, match="lack doi, funding"):
        build(model.replace(release=incomplete), out, clone, runner)
    assert runner.commands == []


def test_the_build_writes_into_the_out_only(
    model: Project, out: Path, clone: Path
) -> None:
    """The build clones the checkout and commits nothing, the checkout is read."""
    head = git(clone, "rev-parse", "HEAD")
    runner = FakeRunner()
    path = build(model, out, clone, runner)
    assert path == out / "repository"
    assert not runner.ran("docker") and not runner.ran("gh")
    assert git(path, "rev-parse", "HEAD") == head
    assert git(clone, "rev-parse", "HEAD") == head
    assert not git(clone, "status", "--porcelain").strip()
    assert not (clone / MANIFEST_FILE).exists()
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["version"] == VERSION
    assert manifest["framework"] == "sbmlsim"
    assert manifest["framework_version"] == SBMLSIM
    assert "uv.lock" in manifest["files"]
    assert "docs/images/toy_model.png" in manifest["files"]
    assert "specs/simulate.json" in manifest["files"]
    assert (path / "uv.lock").is_file()
    assert (path / "zensical.toml").is_file()
    assert (path / "docs" / "images" / "toy_model.png").is_file()
    assert (path / "release-notes" / f"{VERSION}.md").is_file()
    fit_id = adopted_id(model)
    assert (path / "fit" / fit_id / "index.html").is_file()
    assert (path / "fit" / fit_id / "report.pdf").is_file()
    assert (path / "credibility" / "index.html").is_file()
    assert (path / "src" / PACKAGE / "fits" / fit_id / "parameters.json").is_file()
    assert not (path / "run_toy.py").exists()
    assert not (path / "simulations" / "run.json").exists()


def test_the_build_clones_the_repository_on_github(
    model: Project, out: Path, clone: Path
) -> None:
    """Without a checkout the release repository is cloned from its URL."""
    path = build(model, out, None)
    assert (path / MANIFEST_FILE).is_file()


def test_the_build_knows_the_source_commit(
    model: Project, out: Path, clone: Path
) -> None:
    """The assessment of the credibility records the commit of the source."""
    path = build(model, out, clone)
    spec = json.loads((path / "specs" / "credibility.json").read_text(encoding="utf-8"))
    assert spec == {
        "project": REFERENCE,
        "source_commit": git(model.root, "rev-parse", "HEAD").strip(),
    }


def test_stage_lines(
    model: Project, out: Path, clone: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="sbmlsim.release"):
        build(model, out, clone)
    messages = "\n".join(caplog.messages)
    for line in (
        f"Export of {PACKAGE}",
        "Project files",
        "Lock and sync of the environment",
        "Simulations with the confidence interval",
        "Release notes",
        "Docs and README",
        "Prune and manifest",
    ):
        assert line in messages


def test_the_summary_of_the_build(model: Project, out: Path, clone: Path) -> None:
    path, release = built(model, out, clone)
    manifest = Manifest.read(path / MANIFEST_FILE)
    assert manifest == release.manifest
    summary = build_summary(release, path)
    assert summary["repository"] == "repository"
    assert summary["version"] == VERSION
    assert summary["publishable"] is True
    assert summary["files"] == len(release.manifest.files)
    assert summary["steps"] == [
        "factory",
        "simulate",
        "uncertainty",
        "report",
        "credibility",
    ]
    assert summary["changed"] == []
    assert summary["skipped"] == []


def test_the_release_has_the_models_of_the_factory(
    model: Project, out: Path, clone: Path
) -> None:
    """The factory runs first, its rebuild replaces the models of the package."""
    rebuilt = (TOY_PATH / "models" / "toy_slow.xml").read_text(encoding="utf-8")
    rebuilt = rebuilt.replace('value="0.1"', 'value="0.05"')
    runner = FakeRunner(factory_changes={"models/toy_slow.xml": rebuilt})
    path, release = built(model, out, clone, runner)
    package = path / "src" / PACKAGE
    assert (package / "models" / "toy_slow.xml").read_text(encoding="utf-8") == rebuilt
    assert (path / "models" / "toy_slow.xml").read_text(encoding="utf-8") == rebuilt
    # an output the package does not track is released at the top only
    assert (path / "toy.omex").is_file()
    assert not (package / "toy.omex").exists()
    manifest = release.manifest.files
    assert f"src/{PACKAGE}/models/toy_slow.xml" in manifest
    assert build_summary(release, path)["changed"] == ["models/toy_slow.xml"]
    factory = runner.commands.index(next(c for c in runner.commands if "factory" in c))
    simulate = next(c for c in runner.commands if "specs/simulate.json" in c)
    assert factory < runner.commands.index(simulate)


def test_the_draws_are_decided_on_the_models_of_the_factory(
    model: Project, out: Path, clone: Path
) -> None:
    """A factory whose default variant has no distributions skips the draws."""
    plain = (TOY_PATH / "models" / "toy.xml").read_text(encoding="utf-8")
    runner = FakeRunner(factory_changes={"models/toy.xml": plain})
    path, release = built(model, out, clone, runner)
    summary = build_summary(release, path)
    assert "uncertainty" not in summary["steps"]
    assert summary["skipped"] == [
        {
            "step": "uncertainty",
            "reason": "the default variant has no distributions of its fitted "
            "parameters",
        }
    ]
    assert not runner.ran("specs/uncertainty.json")
    assert not (path / "simulations_uncertainty").exists()
    assert not (path / "specs" / "uncertainty.json").exists()
    page = (path / "docs" / "simulations.md").read_text(encoding="utf-8")
    assert "simulations_uncertainty" not in page


def test_publish_after_review(model: Project, out: Path, clone: Path) -> None:
    path = build(model, out, clone)
    runner, publisher = FakeRunner(), FakePublisher()
    publish(model, out, runner, publisher)
    assert not runner.ran("sbmlsim.run")
    assert runner.ran("docker") and runner.ran("gh")
    assert model.release is not None
    assert publisher.calls == [(model.release, path, VERSION, path / "toy.omex")]


def test_publish_needs_a_build(model: Project, out: Path) -> None:
    with pytest.raises(ReleaseError, match="holds no build of a release"):
        publish(model, out)


def test_hand_edit_after_review_needs_a_rebuild(
    model: Project, out: Path, clone: Path
) -> None:
    path = build(model, out, clone)
    (path / "docs" / "index.md").write_text("edited by hand\n", encoding="utf-8")
    publisher = FakePublisher()
    with pytest.raises(ReleaseError, match="no unchanged build"):
        publish(model, out, publisher=publisher)
    assert publisher.calls == []
    build(model, out, clone)
    assert (path / "docs" / "index.md").read_text(
        encoding="utf-8"
    ) != "edited by hand\n"
    publish(model, out, publisher=publisher)
    assert len(publisher.calls) == 1


def test_a_moved_source_needs_a_rebuild(model: Project, out: Path, clone: Path) -> None:
    build(model, out, clone)
    commit_source(model)
    with pytest.raises(ReleaseError, match="no unchanged build"):
        publish(model, out)


def commit_source(model: Project) -> None:
    """Move the source repository on by a commit which changes the package."""
    (model.root / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
    root = model.root.parent.parent
    git(root, "add", "-A")
    git(root, "commit", "-m", "more")


def commit_release(path: Path, tag: bool) -> None:
    """Commit the build as the publisher does, with or without the tag."""
    git(path, "add", "-A")
    git(path, "commit", "-m", f"Release {VERSION}")
    if tag:
        git(path, "tag", VERSION)


def test_a_committed_release_is_published_not_built(
    model: Project, out: Path, clone: Path
) -> None:
    path = build(model, out, clone)
    commit_release(path, tag=True)
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match="is committed in"):
        build(model, out, clone, runner)
    assert not runner.ran("sbmlsim.run")
    publisher = FakePublisher()
    publish(model, out, runner, publisher)
    assert not runner.ran("sbmlsim.run")
    assert len(publisher.calls) == 1


def test_dirty_unrelated_changes_refused(
    model: Project, out: Path, clone: Path
) -> None:
    path = build(model, out, clone)
    (path / "notes.txt").write_text("not from a release\n", encoding="utf-8")
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match="changes which no release made"):
        build(model, out, clone, runner)
    assert not runner.ran("sbmlsim.run")
    with pytest.raises(ReleaseError, match="no unchanged build"):
        publish(model, out)


def test_missing_tools_fail_before_publishing(
    model: Project, out: Path, clone: Path
) -> None:
    build(model, out, clone)
    publisher = FakePublisher()
    with pytest.raises(ReleaseError, match="docker"):
        publish(model, out, FakeRunner(fail_on="docker info"), publisher)
    assert publisher.calls == []


def test_a_local_lock_is_not_published(model: Project, out: Path, clone: Path) -> None:
    path = build(model, out, clone)
    lock = path / "uv.lock"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(PYPI, '{ directory = "../sbmlsim" }'),
        encoding="utf-8",
    )
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["files"]["uv.lock"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    (path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    publisher = FakePublisher()
    with pytest.raises(ReleaseError, match="not from PyPI"):
        publish(model, out, publisher=publisher)
    assert publisher.calls == []


def test_a_build_with_a_local_framework_is_not_published(
    model: Project, out: Path, clone: Path, tmp_path: Path
) -> None:
    """The lock of the release has the local framework, not its version."""
    path, release = built(model, out, clone, framework_path=tmp_path / "sbmlsim")
    pyproject = (path / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv.sources]\nsbmlsim = { path = " in pyproject
    assert build_summary(release, path)["publishable"] is False


def test_fit_without_record_refused(model: Project, out: Path, clone: Path) -> None:
    fit_id = f"{fit_subset(adopted_id(model))}_19990101_000000__00000"
    adopted = model.root.parent.parent.parent / "adopted.json"
    adopted.write_text(json.dumps({"fits": {"DECAY": fit_id}}), encoding="utf-8")
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match=fit_id):
        build(model.replace(adopted=adopted), out, clone, runner)
    assert not runner.ran("sbmlsim.run")


def test_publish_needs_archive(model: Project, out: Path, clone: Path) -> None:
    build(model, out, clone)
    assert model.release is not None
    settings = dataclasses.replace(model.release, archive=None)
    with pytest.raises(ReleaseError, match="archive"):
        publish(model.replace(release=settings), out)


def test_failed_build_rerun(model: Project, out: Path, clone: Path) -> None:
    with pytest.raises(ReleaseError, match="uncertainty"):
        build(model, out, clone, FakeRunner(fail_on="specs/uncertainty.json"))
    path = build(model, out, clone)
    assert (path / MANIFEST_FILE).is_file()
    assert not (path / ".git" / BUILD_MARKER).exists()


def test_file_added_after_a_failed_build_refused(
    model: Project, out: Path, clone: Path
) -> None:
    """A failed build records its changes, a file added by hand is not one."""
    with pytest.raises(ReleaseError, match="uncertainty"):
        build(model, out, clone, FakeRunner(fail_on="specs/uncertainty.json"))
    path = out / REPOSITORY_DIR
    (path / "mine.txt").write_text("added by hand\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="changes which no release made"):
        build(model, out, clone)
    assert (path / "mine.txt").is_file()
    (path / "mine.txt").unlink()
    build(model, out, clone)
    assert (path / MANIFEST_FILE).is_file()


def test_interrupted_build_is_rebuilt(model: Project, out: Path, clone: Path) -> None:
    """An interrupt (Ctrl+C) records the changes of the build like a failure."""
    with pytest.raises(KeyboardInterrupt):
        build(model, out, clone, FakeRunner(interrupt_on="specs/uncertainty.json"))
    path = out / REPOSITORY_DIR
    marker = json.loads((path / ".git" / BUILD_MARKER).read_text(encoding="utf-8"))
    assert f"src/{PACKAGE}/__init__.py" in marker["changed"]
    build(model, out, clone)
    assert not (path / ".git" / BUILD_MARKER).exists()


def test_stale_results_are_not_reused(model: Project, out: Path, clone: Path) -> None:
    path = build(model, out, clone)
    (path / "results" / "x").write_text("stale\n", encoding="utf-8")
    build(model, out, clone)
    assert not (path / "results" / "x").exists()
    assert (path / "results" / "simulate" / "Decay" / "Decay.html").is_file()


def test_resume_tag_with_moved_source(model: Project, out: Path, clone: Path) -> None:
    path = build(model, out, clone)
    commit_release(path, tag=True)
    commit_source(model)
    runner, publisher = FakeRunner(), FakePublisher()
    publish(model, out, runner, publisher)
    assert not runner.ran("sbmlsim.run")
    assert len(publisher.calls) == 1


def test_resume_release_commit_without_tag(
    model: Project, out: Path, clone: Path
) -> None:
    path = build(model, out, clone)
    commit_release(path, tag=False)
    head = git(path, "rev-parse", "HEAD")
    commit_source(model)
    publisher = FakePublisher()
    publish(model, out, publisher=publisher)
    assert len(publisher.calls) == 1
    assert git(path, "rev-parse", "HEAD") == head
    assert not git(path, "status", "--porcelain").strip()


def test_invalid_version_refused(model: Project, out: Path, clone: Path) -> None:
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match="not a version"):
        build(model, out, clone, runner, version="1.0")
    with pytest.raises(ReleaseError, match="not a version"):
        publish_release(model, "1.0", out, runner=runner)
    assert runner.commands == []


def test_a_version_not_after_the_last_release_refused(
    model: Project, out: Path, clone: Path
) -> None:
    with pytest.raises(ReleaseError, match=r"greater than the last release 0\.1\.0"):
        build(model, out, clone, version="0.1.0")


def commit_package_file(model: Project, relative: str) -> None:
    """Track a file in the model package of the source repository."""
    (model.root / relative).parent.mkdir(parents=True, exist_ok=True)
    (model.root / relative).write_text("x\n", encoding="utf-8")
    root = model.root.parent.parent
    git(root, "add", "-f", str(model.root / relative))
    git(root, "commit", "-m", f"track {relative}")


def test_tracked_results_of_the_package_are_released(
    model: Project, out: Path, clone: Path
) -> None:
    """Only the results folder at the top of the release repository is ignored."""
    commit_package_file(model, "fitting/results/data.xlsx")
    path = build(model, out, clone)
    released = f"src/{PACKAGE}/fitting/results/data.xlsx"
    assert (
        released
        in json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))["files"]
    )
    untracked = git(path, "ls-files", "--others", "--exclude-standard")
    assert released in untracked.splitlines()


def test_ignored_export_refused(model: Project, out: Path, clone: Path) -> None:
    commit_package_file(model, "cache/old.pyc")
    with pytest.raises(ReleaseError, match=rf"src/{PACKAGE}/cache/old\.pyc"):
        build(model, out, clone)
    assert not (out / REPOSITORY_DIR / MANIFEST_FILE).exists()


def test_publish_removes_a_stale_marker(model: Project, out: Path, clone: Path) -> None:
    path = build(model, out, clone)
    marker = path / ".git" / BUILD_MARKER
    # a rebuild killed before it changed a file
    marker.write_text(json.dumps({"version": VERSION}) + "\n", encoding="utf-8")
    publish(model, out)
    assert not marker.exists()


def test_local_commit_ahead_refused(model: Project, out: Path, clone: Path) -> None:
    """A local commit of the checkout which the release repository lacks."""
    (clone / "notes.txt").write_text("not from a release\n", encoding="utf-8")
    git(clone, "add", "-A")
    git(clone, "commit", "-m", "local")
    runner = FakeRunner()
    with pytest.raises(ReleaseError, match="ahead of origin/main"):
        build(model, out, clone, runner)
    assert not runner.ran("sbmlsim.run")


def test_local_commit_before_the_release_commit_refused(
    model: Project, out: Path, clone: Path
) -> None:
    """Only the release commit itself may wait for the push."""
    path = build(model, out, clone)
    git(path, "commit", "-q", "--allow-empty", "-m", "local")
    commit_release(path, tag=False)
    publisher = FakePublisher()
    with pytest.raises(ReleaseError, match="ahead of origin/main by 2 commits"):
        publish(model, out, publisher=publisher)
    assert publisher.calls == []


def test_first_release_into_an_empty_repository(
    model: Project, out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "empty.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-b", "main")
    monkeypatch.setattr(
        source_module, "clone_url", lambda repository: f"file://{remote}"
    )
    path = build(model, out, None)
    assert (path / MANIFEST_FILE).is_file()
    assert (path / "README.md").is_file()
    notes = (path / "release-notes" / f"{VERSION}.md").read_text(encoding="utf-8")
    assert "First release." in notes


def test_check_build_tells_whether_an_out_can_be_published(
    model: Project, out: Path, clone: Path
) -> None:
    """The check which `publish_release` makes before it publishes."""
    runner = FakeRunner()
    assert publishable(model, VERSION, out, runner=runner) == (
        f"{out} holds no build of a release, build it first with the kind "
        "release (python -m sbmlsim.release build)."
    )
    path = build(model, out, clone)
    checked = check_build(model, VERSION, out, runner=runner)
    assert checked.path == path
    assert checked.archive == path / "toy.omex"
    assert checked.resumed is False
    assert model.release is not None
    assert checked.settings == model.release
    assert publishable(model, VERSION, out, runner=runner) is None
    assert not runner.ran("docker") and not runner.ran("gh")
    (path / "docs" / "index.md").write_text("edited by hand\n", encoding="utf-8")
    reason = publishable(model, VERSION, out, runner=runner)
    assert reason is not None
    assert "no unchanged build" in reason
    with pytest.raises(ReleaseError, match="no unchanged build"):
        check_build(model, VERSION, out, runner=runner)


def test_check_build_of_a_resumed_publishing(
    model: Project, out: Path, clone: Path
) -> None:
    path = build(model, out, clone)
    commit_release(path, tag=True)
    commit_source(model)
    assert check_build(model, VERSION, out, runner=FakeRunner()).resumed is True
