"""The build of a release of the toy project with the analyses of the run layer.

Only `uv lock` and `uv sync` are faked: the steps of the build run `python -m
sbmlsim.run` with their specs in the release repository, with the exported
package on the path, as `uv run` runs them in the environment of the release.
"""

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from sbmlsim.project import Project
from sbmlsim.release.docs import PAGES
from sbmlsim.release.manifest import MANIFEST_FILE, Manifest
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import run
from sbmlsim.release.workflow import build_release, prepare_release
from tests.release.fakes import VERSION, lock_text
from tests.release.repos import REFERENCE, git

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("git_env")]


class AnalysesRunner:
    """Runs git and the analyses of the run layer; fakes uv lock and uv sync."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        command = [str(a) for a in args]
        self.commands.append(command)
        if command[0] == "git":
            return run(command, cwd)
        if command[:2] == ["uv", "lock"]:
            (cwd / "uv.lock").write_text(lock_text(), encoding="utf-8")
        elif command[:2] == ["uv", "run"]:
            env = {**os.environ, "PYTHONPATH": str(cwd / "src")}
            result = subprocess.run(
                [sys.executable, *command[3:]],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        return ""


def test_the_release_of_the_toy(
    source_without_credibility: Project,
    release_settings: ReleaseSettings,
    remote_and_clone: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    project = source_without_credibility
    _, clone = remote_and_clone
    runner = AnalysesRunner()
    out = tmp_path / "out"
    prepared = prepare_release(project, VERSION, out, checkout=clone, runner=runner)
    manifest = build_release(
        project, REFERENCE, VERSION, prepared, workers=1, runner=runner
    )
    path = prepared.target.path
    assert Manifest.read(path / MANIFEST_FILE) == manifest
    steps = [c[5] for c in runner.commands if c[:2] == ["uv", "run"]]
    assert steps == ["factory", "simulate", "simulate", "report"]

    assert sorted(p.name for p in (path / "models").glob("*.xml")) == [
        "toy.xml",
        "toy_slow.xml",
    ]
    experiments = sorted(
        p.name
        for p in (path / "simulations").iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    assert experiments
    for experiment in experiments:
        assert (path / "simulations" / experiment / f"{experiment}.html").is_file()
        assert (path / "simulations_uncertainty" / experiment).is_dir()
    assert not (path / "simulations" / "run.json").exists()
    (fit_id,) = project.adopted_fits().values()
    assert (path / "fit" / fit_id / "index.html").is_file()

    simulations = (path / "docs" / "simulations.md").read_text(encoding="utf-8")
    for experiment in experiments:
        assert f"simulations/{experiment}/{experiment}.html" in simulations
    fit = (path / "docs" / "fit.md").read_text(encoding="utf-8")
    assert f"## {fit_id}" in fit
    assert "| `k` |" in fit
    assert {p.stem for p in (path / "docs").glob("*.md")} == {p for p, _ in PAGES}
    notes = (path / "release-notes" / f"{VERSION}.md").read_text(encoding="utf-8")
    assert f"`{fit_id}`" in notes
    assert "`toy.xml`" in notes
    assert git(path, "status", "--porcelain").strip()
    assert git(clone, "status", "--porcelain").strip() == ""

    # the site of the release builds without a warning
    result = subprocess.run(
        ["zensical", "build", "--clean", "--strict"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
