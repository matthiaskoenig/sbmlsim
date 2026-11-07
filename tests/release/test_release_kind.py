"""Tests of the kind `release` and of the command line `python -m sbmlsim.release`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sbmlsim import run
from sbmlsim.project import Project
from sbmlsim.release import cli
from sbmlsim.release import source as source_module
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError
from sbmlsim.run.context import RUN_FILE, SUMMARY_FILE
from sbmlsim.run.kinds import ConfigurationError, get_kind
from sbmlsim.run.release_kinds import ReleaseSpec
from tests.release.fakes import VERSION, FakeRunner
from tests.release.repos import REFERENCE, git

pytestmark = pytest.mark.usefixtures("git_env")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    """The fake runner of the commands of the kind."""
    fake = FakeRunner()
    monkeypatch.setattr(source_module, "run", fake)
    return fake


def test_the_kind_is_registered() -> None:
    kind = get_kind("release")
    assert kind.spec is ReleaseSpec
    assert kind.uses_cores
    assert not kind.splittable


def test_the_version_of_a_spec() -> None:
    assert ReleaseSpec(project=REFERENCE, version="1.2.3").version == "1.2.3"
    with pytest.raises(ValueError, match="not a version"):
        ReleaseSpec(project=REFERENCE, version="1.2")


def test_the_plan_checks_the_release_settings(source_without_fits: Project) -> None:
    run.plan("release", {"project": REFERENCE, "version": VERSION})
    with pytest.raises(ConfigurationError, match="has no release settings"):
        run.plan(
            "release", {"project": "tests.credibility_toy:PROJECT", "version": "1.0.0"}
        )


def test_the_kind_builds_the_release_into_the_out(
    source_project: Project,
    release_settings: ReleaseSettings,
    remote_and_clone: tuple[Path, Path],
    runner: FakeRunner,
    tmp_path: Path,
) -> None:
    _, clone = remote_and_clone
    head = git(clone, "rev-parse", "HEAD")
    out = tmp_path / "out"
    spec = {"project": REFERENCE, "version": VERSION, "repository": str(clone)}
    summary = run.execute("release", spec, out, workers=2)
    assert summary["repository"] == "repository"
    assert summary["version"] == VERSION
    assert summary["framework"] == "sbmlsim"
    assert summary["publishable"] is True
    assert json.loads((out / SUMMARY_FILE).read_text(encoding="utf-8")) == summary
    assert json.loads((out / RUN_FILE).read_text(encoding="utf-8"))["status"] == "done"
    assert (out / "repository" / "release.json").is_file()
    assert git(clone, "rev-parse", "HEAD") == head
    simulate = next(c for c in runner.commands if "specs/simulate.json" in c)
    assert simulate[-2:] == ["--workers", "2"]
    # a resume builds again into the repository of the out
    again = run.execute("release", spec, out, workers=2, resume=True)
    assert again["files"] == summary["files"]


def test_a_refused_release_is_a_configuration_error(
    source_project: Project,
    release_settings: ReleaseSettings,
    runner: FakeRunner,
    tmp_path: Path,
) -> None:
    (source_project.root / "new.txt").write_text("untracked\n", encoding="utf-8")
    spec = {"project": REFERENCE, "version": VERSION}
    with pytest.raises(ConfigurationError, match="refused: commit the changes of"):
        run.execute("release", spec, tmp_path / "out")
    assert not runner.ran("sbmlsim.run")


def test_build_runs_the_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[tuple[str, dict, Path, dict]] = []

    def execute(kind: str, spec: dict, out: Path, **kwargs: object) -> dict:
        calls.append((kind, spec, out, kwargs))
        return {"repository": "repository"}

    monkeypatch.setattr("sbmlsim.run.execute", execute)
    out = tmp_path / "out"
    options = ["--repository", ".", "--framework-path", "x", "--workers", "3"]
    command = ["build", REFERENCE, "--version", VERSION, "--out", str(out)]
    code = cli.main([*command, *options, "--resume"])
    assert code == 0
    ((kind, spec, given, kwargs),) = calls
    assert kind == "release"
    assert spec == {
        "project": REFERENCE,
        "version": VERSION,
        "repository": str(Path.cwd()),
        "framework_path": str(Path.cwd() / "x"),
    }
    assert given == out
    assert kwargs == {"workers": 3, "resume": True}
    assert capsys.readouterr().out == f"{out / 'repository'}\nreview, then publish it\n"


def test_build_reports_a_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def refuse(*args: object, **kwargs: object) -> dict:
        raise ConfigurationError("The release is refused: no")

    monkeypatch.setattr("sbmlsim.run.execute", refuse)
    args = ["build", REFERENCE, "--version", VERSION, "--out", str(tmp_path)]
    assert cli.main(args) == 4
    assert "The release is refused: no" in capsys.readouterr().err

    def fail(*args: object, **kwargs: object) -> dict:
        raise ReleaseError("Build step 'uv lock' failed")

    monkeypatch.setattr("sbmlsim.run.execute", fail)
    assert cli.main(args) == 3


def test_publish_publishes_the_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[tuple[Project, str, Path]] = []

    def publish(project: Project, version: str, out: Path) -> list[str]:
        calls.append((project, version, out))
        return []

    monkeypatch.setattr("sbmlsim.release.workflow.publish_release", publish)
    args = ["publish", "tests.project_toy:PROJECT", "--version", VERSION]
    assert cli.main([*args, "--out", str(tmp_path)]) == 0
    ((project, version, out),) = calls
    assert (project.name, version, out) == ("toy", VERSION, tmp_path)
    assert capsys.readouterr().out == f"Released {VERSION} from {tmp_path}.\n"

    def refuse(*args: object) -> list[str]:
        raise ReleaseError("the build is stale")

    monkeypatch.setattr("sbmlsim.release.workflow.publish_release", refuse)
    assert cli.main([*args, "--out", str(tmp_path)]) == cli.EXIT_REFUSED
    assert "error: the build is stale" in capsys.readouterr().err
    missing = ["publish", "tests.nothing:PROJECT", "--version", VERSION]
    assert cli.main([*missing, "--out", str(tmp_path)]) == 4


@pytest.mark.parametrize("workers", ["0", "-2", "many"])
def test_workers_validated(workers: str, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["build", REFERENCE, "--version", VERSION, "--out", "x"]
    assert cli.main([*args, "--workers", workers]) == 2
    assert "--workers" in capsys.readouterr().err


def test_the_module_runs_the_command_line(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sbmlsim.release", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "build" in result.stdout
    assert "publish" in result.stdout
