"""Tests of the build of a release repository."""

import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from sbmlsim.project import Project
from sbmlsim.release.build import _replace, build, build_steps, has_distributions
from sbmlsim.release.source import ReleaseError
from tests.project_toy import PROJECT

FIT_ID = "DECAY_20260101_000000__abcde"
OTHER_FIT_ID = "OTHER_20260102_000000__bcdef"
FIT_IDS = [FIT_ID, OTHER_FIT_ID]
REFERENCE = "tests.project_toy:PROJECT"
assert PROJECT.release is not None
SETTINGS = dataclasses.replace(PROJECT.release, archive="toy.omex")


def project_with(
    tmp_path: Path,
    fit_ids: Sequence[str] = tuple(FIT_IDS),
    *,
    credibility: bool = False,
    factory: bool = True,
) -> Project:
    """The toy project with adopted fits and an archive."""
    adopted = tmp_path / "adopted.json"
    adopted.write_text(
        json.dumps({"fits": {f"subset{k}": f for k, f in enumerate(fit_ids)}}),
        encoding="utf-8",
    )
    changes: dict = {"adopted": adopted, "release": SETTINGS}
    if not credibility:
        changes["credibility"] = None
    if not factory:
        changes["factory"] = None
    return PROJECT.replace(**changes)


@pytest.fixture(autouse=True)
def _distributions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the default variant of the toy the distributions of its fitted rate."""
    monkeypatch.setattr("sbmlsim.release.build.has_distributions", lambda *args: True)


class FakeRunner:
    """Records the commands and creates the files a real run would."""

    def __init__(
        self,
        fail_on: str | None = None,
        no_archive: bool = False,
        no_sim: bool = False,
        no_credibility: bool = False,
        no_figures: bool = False,
        reported: Sequence[str] = tuple(FIT_IDS),
    ) -> None:
        self.reported = reported
        self.no_figures = no_figures
        self.no_archive = no_archive
        self.no_sim = no_sim
        self.no_credibility = no_credibility
        self.commands: list[list[str]] = []
        self.streamed: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        command = [str(a) for a in args]
        self.commands.append(command)
        if stream:
            self.streamed.append(command)
        joined = " ".join(command)
        if self.fail_on and self.fail_on in joined:
            raise ReleaseError(f"'{joined}' failed in {cwd}")
        if "sbmlsim.run" not in command:
            return ""
        kind, out = command[5], cwd / command[8]
        if kind == "factory":
            models = out / "models"
            models.mkdir(parents=True)
            (models / "toy.xml").write_text("<sbml/>", encoding="utf-8")
            (models / "toy.md").write_text("# toy", encoding="utf-8")
            (models / "other.txt").write_text("not copied", encoding="utf-8")
            (models / "toy.svg").write_text("<svg/>", encoding="utf-8")
            (out / "mit.txt").write_text("MIT", encoding="utf-8")
            (out / "cc-by-sa-4.0.txt").write_text("CC", encoding="utf-8")
            if not self.no_figures:
                (out / "figures").mkdir()
                (out / "figures" / "model.svg").write_text("<svg/>", encoding="utf-8")
            if not self.no_archive:
                (out / "toy.omex").write_text("omex", encoding="utf-8")
        elif kind == "simulate" and ("uncertainty" in joined or not self.no_sim):
            self._write(out / "Exp/index.html")
        elif kind == "credibility" and not self.no_credibility:
            self._write(out / "credibility/credibility.json")
        elif kind == "report":
            # the kind `report` with `adopted` writes a report per adopted fit
            for fit_id in self.reported:
                self._write(out / fit_id / "index.html")
                self._write(out / fit_id / "report.pdf")
        if out.is_dir():
            # the records of the run layer, which the release leaves out
            (out / "run.json").write_text("{}", encoding="utf-8")
        return ""

    @staticmethod
    def _write(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html/>", encoding="utf-8")


def run_command(kind: str, step: str, *workers: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "sbmlsim.run",
        kind,
        f"specs/{step}.json",
        "--out",
        f"results/{step}",
        *workers,
    ]


def test_build_commands_in_order(tmp_path: Path) -> None:
    runner = FakeRunner()
    release = tmp_path / "release"
    release.mkdir()
    build(project_with(tmp_path), REFERENCE, release, runner=runner, workers=2)
    assert runner.commands == [
        ["uv", "lock"],
        ["uv", "sync", "--locked"],
        run_command("factory", "factory"),
        run_command("simulate", "simulate", "--workers", "2"),
        run_command("simulate", "uncertainty", "--workers", "2"),
        run_command("report", "report"),
    ]


def test_build_writes_the_specs_of_the_steps(tmp_path: Path) -> None:
    """The specs are files of the release, its users run the same analyses."""
    release = tmp_path / "release"
    release.mkdir()
    (release / "specs").mkdir()
    (release / "specs" / "stale.json").write_text("{}", encoding="utf-8")
    written = build(
        project_with(tmp_path), REFERENCE, release, runner=FakeRunner()
    ).written
    specs = {p.name for p in (release / "specs").iterdir()}
    assert specs == {"factory.json", "simulate.json", "uncertainty.json", "report.json"}
    assert Path("specs/report.json") in written
    report = json.loads((release / "specs" / "report.json").read_text(encoding="utf-8"))
    assert report == {"adopted": True, "project": REFERENCE, "subset": "all"}
    uncertainty = json.loads(
        (release / "specs" / "uncertainty.json").read_text(encoding="utf-8")
    )
    assert uncertainty == {
        "project": REFERENCE,
        "uncertainty": {"draws": 100, "quantiles": [0.05, 0.95], "seed": 1234},
    }
    assert b"\r" not in (release / "specs" / "report.json").read_bytes()


def test_build_logs_the_stages(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The steps stream their output and every stage logs a line."""
    runner = FakeRunner()
    with caplog.at_level(logging.INFO, logger="sbmlsim.release"):
        build(project_with(tmp_path), REFERENCE, tmp_path, runner=runner)
    assert runner.streamed == runner.commands
    for line in (
        "Lock and sync of the environment",
        "Models",
        "Simulations",
        "Simulations with the confidence interval",
        "Fit reports",
    ):
        assert line in caplog.messages


def test_build_copies_outputs(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    written = build(
        project_with(tmp_path), REFERENCE, release, runner=FakeRunner()
    ).written
    assert (release / "models/toy.xml").is_file()
    assert (release / "models/toy.md").is_file()
    assert not (release / "models/other.txt").exists()
    # the diagram of the model which the factory draws beside it
    assert (release / "models/toy.svg").is_file()
    assert (release / "simulations/Exp/index.html").is_file()
    assert not (release / "simulations/run.json").exists()
    assert (release / "simulations_uncertainty/Exp/index.html").is_file()
    for fit_id in FIT_IDS:
        assert (release / "fit" / fit_id / "index.html").is_file()
        assert (release / "fit" / fit_id / "report.pdf").is_file()
    assert (release / "toy.omex").is_file()
    assert (release / "mit.txt").is_file()
    assert (release / "cc-by-sa-4.0.txt").is_file()
    assert (release / "figures/model.svg").is_file()
    assert Path("models/toy.xml") in written
    assert Path("uv.lock") in written
    assert Path("simulations") in written
    assert Path("fit") / FIT_ID in written
    assert Path("fit") / OTHER_FIT_ID in written


def test_build_removes_the_figures_of_an_earlier_release(tmp_path: Path) -> None:
    """A factory without figures leaves no figure of an earlier release behind."""
    release = tmp_path / "release"
    (release / "figures").mkdir(parents=True)
    (release / "figures" / "toy_model.png").write_bytes(b"png")
    build(
        project_with(tmp_path), REFERENCE, release, runner=FakeRunner(no_figures=True)
    )
    assert not (release / "figures").exists()


def test_build_replaces_stale_reports(tmp_path: Path) -> None:
    old = tmp_path / "fit" / "OLD_20200101_000000__aaaaa"
    old.mkdir(parents=True)
    (old / "index.html").write_text("old", encoding="utf-8")
    build(project_with(tmp_path), REFERENCE, tmp_path, runner=FakeRunner())
    assert not old.exists()
    assert (tmp_path / "fit" / FIT_ID).is_dir()


def test_build_failure_names_step(tmp_path: Path) -> None:
    runner = FakeRunner(fail_on="specs/uncertainty.json")
    with pytest.raises(ReleaseError, match=r"specs/uncertainty\.json"):
        build(project_with(tmp_path), REFERENCE, tmp_path, runner=runner)


def test_build_missing_declared_archive_raises(tmp_path: Path) -> None:
    runner = FakeRunner(no_archive=True)
    with pytest.raises(ReleaseError, match=r"toy\.omex"):
        build(project_with(tmp_path), REFERENCE, tmp_path, runner=runner)


def test_build_removes_stale_models(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models/old.xml").write_text("old", encoding="utf-8")
    build(project_with(tmp_path), REFERENCE, tmp_path, runner=FakeRunner())
    assert not (tmp_path / "models/old.xml").exists()


def test_build_does_not_reuse_stale_results(tmp_path: Path) -> None:
    stale = tmp_path / "results/simulate/Exp"
    stale.mkdir(parents=True)
    (stale / "index.html").write_text("stale", encoding="utf-8")
    with pytest.raises(ReleaseError, match="simulate"):
        build(
            project_with(tmp_path), REFERENCE, tmp_path, runner=FakeRunner(no_sim=True)
        )


def test_build_without_fits_writes_no_reports(tmp_path: Path) -> None:
    """Without adopted fits there are no fit reports and no confidence interval."""
    stale = tmp_path / "simulations_uncertainty"
    stale.mkdir()
    (stale / "index.html").write_text("stale", encoding="utf-8")
    runner = FakeRunner()
    written = build(project_with(tmp_path, []), REFERENCE, tmp_path, runner).written
    assert not any("report" in command for command in runner.commands)
    assert not any("specs/uncertainty.json" in c for c in runner.commands)
    assert not stale.exists()
    assert Path("simulations_uncertainty") not in written
    assert (tmp_path / "fit").is_dir()
    assert not any((tmp_path / "fit").iterdir())
    assert not any(path.parts[0] == "fit" for path in written)


def test_build_requires_the_report_of_every_fit(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match=OTHER_FIT_ID):
        build(
            project_with(tmp_path),
            REFERENCE,
            tmp_path,
            runner=FakeRunner(reported=[FIT_ID]),
        )


def test_build_runs_the_credibility_of_a_declared_project(tmp_path: Path) -> None:
    runner = FakeRunner()
    project = project_with(tmp_path, credibility=True)
    written = build(project, REFERENCE, tmp_path, runner=runner).written
    assert runner.commands[-1] == run_command("credibility", "credibility")
    assert (tmp_path / "credibility/credibility.json").is_file()
    assert Path("credibility") in written


def test_build_passes_the_source_commit_to_the_credibility(tmp_path: Path) -> None:
    """The assessment runs in the release repository, its provenance is the source."""
    project = project_with(tmp_path, credibility=True)
    build(project, REFERENCE, tmp_path, runner=FakeRunner(), source_commit="0123abc")
    spec = json.loads(
        (tmp_path / "specs" / "credibility.json").read_text(encoding="utf-8")
    )
    assert spec == {"project": REFERENCE, "source_commit": "0123abc"}


def test_build_without_a_declaration_removes_credibility(tmp_path: Path) -> None:
    stale = tmp_path / "credibility"
    stale.mkdir()
    (stale / "credibility.json").write_text("{}", encoding="utf-8")
    runner = FakeRunner()
    written = build(project_with(tmp_path), REFERENCE, tmp_path, runner=runner).written
    assert not any("credibility" in " ".join(c) for c in runner.commands)
    assert not stale.exists()
    assert Path("credibility") not in written


def test_build_requires_the_credibility_folder(tmp_path: Path) -> None:
    project = project_with(tmp_path, credibility=True)
    with pytest.raises(ReleaseError, match=r"specs/credibility\.json"):
        build(project, REFERENCE, tmp_path, runner=FakeRunner(no_credibility=True))


def test_build_without_a_factory_releases_the_models_of_the_package(
    tmp_path: Path,
) -> None:
    """The models of a project without a factory are the ones of its package."""
    project = project_with(tmp_path, factory=False)
    project = project.replace(release=PROJECT.release)
    release = tmp_path / "release"
    package = release / "src" / "tests" / "project_toy" / "models"
    package.mkdir(parents=True)
    (package / "toy.xml").write_text("<sbml/>", encoding="utf-8")
    runner = FakeRunner()
    build(project, REFERENCE, release, runner=runner)
    assert not any("factory" in " ".join(c) for c in runner.commands)
    assert (release / "models" / "toy.xml").read_text(encoding="utf-8") == "<sbml/>"
    with pytest.raises(ReleaseError, match="no SBML model"):
        (package / "toy.xml").unlink()
        build(project, REFERENCE, release, runner=runner)


def test_the_steps_of_a_project_without_experiments(tmp_path: Path) -> None:
    project = project_with(tmp_path).replace(experiments=None)
    assert [s.name for s in build_steps(project, REFERENCE)] == ["factory", "report"]


def test_the_copies_leave_out_the_records_of_the_runs(tmp_path: Path) -> None:
    """The records of a run name the host, local paths and the git state."""
    source = tmp_path / "results" / "all"
    (source / "Exp" / "progress").mkdir(parents=True)
    for name in ("run.json", "spec.json", "summary.json", "progress.jsonl"):
        (source / name).write_text("{}", encoding="utf-8")
        # a file of an experiment of the same name is a result
        (source / "Exp" / name).write_text("{}", encoding="utf-8")
    (source / "progress").mkdir()
    (source / "progress" / "unit-0.jsonl").write_text("", encoding="utf-8")
    (source / "index.html").write_text("report", encoding="utf-8")
    target = tmp_path / "simulations"
    _replace(source, target)
    assert sorted(p.name for p in target.iterdir()) == ["Exp", "index.html"]
    assert (target / "Exp" / "run.json").is_file()
    assert (target / "Exp" / "progress").is_dir()


@pytest.mark.usefixtures("git_env")
def test_has_distributions(source_project: Project) -> None:
    """The draws need the distributions of the fitted parameters in the model."""
    assert has_distributions(source_project)
    assert not has_distributions(PROJECT)
    assert not has_distributions(Project(name="empty", root=PROJECT.root))


def test_a_changed_output_of_the_factory_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The analyses run on the models of the package, the release has the rebuilt."""

    class Changed(FakeRunner):
        def __call__(
            self, args: Sequence[str], cwd: Path, *, stream: bool = False
        ) -> str:
            result = super().__call__(args, cwd, stream=stream)
            if "specs/factory.json" in args:
                summary = cwd / "results" / "factory" / "summary.json"
                summary.write_text(
                    json.dumps({"changed": ["models/toy.xml"]}), encoding="utf-8"
                )
            return result

    with caplog.at_level(logging.WARNING, logger="sbmlsim.release"):
        build(project_with(tmp_path), REFERENCE, tmp_path, runner=Changed())
    (message,) = caplog.messages
    assert "'models/toy.xml' of the factory of 'toy' differs" in message
