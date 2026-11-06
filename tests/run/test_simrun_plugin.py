"""The kinds of the run layer as task types of simrun, see `sbmlsim.simrun`.

A fit is computed by the worker of simrun and merged by its merge, and the
report of it runs as the command of an unsplit task, both into the directories
of a task of a run store in `tmp_path`. The tests of the worker and of the
command run on POSIX only: simrun's worker, which `simrun.podspec` and
`simrun.client` import, uses the signals of POSIX and a forkserver.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("simrun")

from simrun.plugins import ENTRY_POINT_GROUP, TaskPaths, load_task_types, resolve
from simrun.workflow import Workflow

from sbmlsim import run
from sbmlsim import simrun as plugin
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.run.fit_kinds import FitSpec
from sbmlsim.run.kinds import get_kind

if TYPE_CHECKING:
    from simrun.store import RunRecord

#: the tests which import simrun's worker, which runs on POSIX
posix = pytest.mark.skipif(
    sys.platform == "win32", reason="simrun's worker runs on POSIX"
)

PROJECT = "examples.hctz_fitting.project:PROJECT"
ROOT = Path(__file__).parents[2]
KINDS = (
    "simulate",
    "fit",
    "identifiability",
    "fisher",
    "report",
    "petab",
    "scan",
    "sensitivity",
    "uncertainty",
    "adopt",
)

#: a fit of three repeats which stop after a few steps
FIT: dict[str, Any] = {
    "project": PROJECT,
    "subset": "IV",
    "repeats": 3,
    "optimizer": {"max_nfev": 3},
}

RUN_ID = "20261106-120000-abc123"


@pytest.fixture(autouse=True)
def _no_simrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of the tests is no task of simrun, whatever the environment."""
    for name in list(os.environ):
        if name.startswith("SIMRUN_"):
            monkeypatch.delenv(name)


def _task_paths(store: Path, task: str, inputs: dict[str, Path]) -> TaskPaths:
    """Get the paths of a task in the run store `store`, as simrun has them."""
    run_dir = store / "runs" / RUN_ID
    task_dir = run_dir / "tasks" / task
    return TaskPaths(
        store=store,
        run_dir=run_dir,
        task_dir=task_dir,
        out=task_dir / "out",
        inputs=inputs,
        scratch=store / "scratch",
    )


def _result(out: Path) -> OptimizationResult:
    return OptimizationResult.from_json(out / "IV" / "optimization_result.json")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get a run store in which simrun's worker computed the fit and merged it."""
    from simrun.worker import run_merge, run_worker

    root = tmp_path_factory.mktemp("store")
    planned = plugin.fit_plan(FitSpec.model_validate(FIT))
    paths = _task_paths(root, "fit", {})
    run_worker(
        plugin.FIT,
        planned,
        paths,
        "w0",
        cores=2,
        memory_bytes=None,
        base_gb=plugin.FIT.hint.memory_base_gb,
        per_worker_gb=plugin.FIT.hint.memory_per_worker_gb,
    )
    run_merge(plugin.FIT, planned, paths)
    return root


def test_entry_points_register_every_kind() -> None:
    types = load_task_types()
    for name in KINDS:
        task_type = types[f"sbmlsim.{name}"]
        kind = get_kind(name)
        assert task_type is plugin.TASK_TYPES[name]
        assert task_type.settings is kind.spec
        assert task_type.splittable == kind.splittable, name
        assert task_type.inputs == kind.inputs
        assert task_type.uses_cores == kind.uses_cores
        assert task_type.cost_setting == kind.cost_setting
        assert task_type.checkpoints
        assert task_type.plan == f"sbmlsim.simrun:{name}_plan"
        assert (task_type.command is None) == kind.splittable


def test_the_entry_points_are_named_as_their_task_types() -> None:
    entries = [
        entry
        for entry in entry_points(group=ENTRY_POINT_GROUP)
        if entry.name.startswith("sbmlsim.")
    ]
    assert len(entries) == len(KINDS)
    for entry in entries:
        assert entry.name == entry.load().name


def test_the_functions_of_the_task_types_resolve() -> None:
    for task_type in plugin.TASK_TYPES.values():
        refs = (
            task_type.plan,
            task_type.count_units,
            task_type.compute_unit,
            task_type.merge,
        )
        for ref in refs:
            if ref is not None:
                assert callable(resolve(ref)), ref


@posix
def test_fit_through_simrun_worker_equals_execute(store: Path, tmp_path: Path) -> None:
    paths = _task_paths(store, "fit", {})
    out = paths.out / plugin.OUTPUT
    # the plan drew a seed, which the units share
    planned = FitSpec.model_validate_json(
        (out / "spec.json").read_text(encoding="utf-8")
    )
    assert planned.seed is not None
    whole = tmp_path / "whole"
    summary = run.execute("fit", planned, whole, workers=1)

    split, reference = _result(out), _result(whole)
    np.testing.assert_array_equal(
        split.df_fits.cost.to_numpy(), reference.df_fits.cost.to_numpy()
    )
    np.testing.assert_array_equal(
        np.stack(split.df_fits.x), np.stack(reference.df_fits.x)
    )
    # the summary of the task, which `simrun compare` reads, is the one of the run
    stored = json.loads((paths.out / "summary.json").read_text(encoding="utf-8"))
    assert stored == summary
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    assert (out / "run.json").exists()


def test_plan_hook_fills_units() -> None:
    planned = plugin.fit_plan(FitSpec(project=PROJECT, subset="IV", repeats=130))
    assert isinstance(planned, FitSpec)
    assert planned.units == 44
    assert planned.seed is not None
    assert plugin.fit_count_units(planned) == 44
    # simrun submits the fields which were given or which the plan set
    submitted = planned.model_dump(mode="json", exclude_unset=True)
    assert {"units", "seed", "opid"} <= submitted.keys()
    assert FitSpec.model_validate(submitted) == planned


@posix
def test_plan_tasks_of_a_planned_workflow_keeps_its_settings() -> None:
    """simrun's client plans a planned workflow again into the same settings."""
    from simrun.client import plan_tasks

    workflow = Workflow(
        name="fit", tasks=[{"id": "fit", "type": plugin.FIT.name, "settings": FIT}]
    )
    types = load_task_types()
    once = plan_tasks(workflow, types)
    twice = plan_tasks(once, types)
    settings = once.task("fit").settings
    assert settings["seed"] is not None
    assert settings["units"] == 3
    assert twice.task("fit").settings == settings


def _record(task: str, settings: dict[str, Any]) -> RunRecord:
    """Get a run of the fit and a report which reads it."""
    from simrun.snapshot import SourceProvenance
    from simrun.store import RunRecord, RunState, TaskState
    from simrun.workflow import Priority

    workflow = Workflow(
        name="fit",
        tasks=[
            {"id": "fit", "type": plugin.FIT.name, "settings": FIT},
            {
                "id": task,
                "type": plugin.TASK_TYPES[task].name,
                "settings": settings,
                "inputs": {"fit": "fit.out"},
            },
        ],
    )
    return RunRecord(
        id=RUN_ID,
        submitted_by="u",
        owner="o",
        agent=None,
        priority=Priority.OWNER,
        group="g",
        workflow=workflow,
        sources=[
            SourceProvenance(
                name="sbmlsim",
                branch="develop",
                commit="abc",
                dirty=False,
                changed_files=[],
                diff_sha256="00",
                snapshot="snapshots/sbmlsim/00",
                patch=None,
            )
        ],
        image="localhost:5000/env:1",
        reserve=None,
        state=RunState.QUEUED,
        tasks={"fit": TaskState.WAITING, task: TaskState.WAITING},
        plans={},
        backend_name=None,
        created=datetime.now(UTC),
        finished=None,
        message=None,
    )


@posix
def test_command_of_report_has_input_placeholder(store: Path) -> None:
    from simrun.podspec import unsplit_command

    command = plugin.REPORT.command
    assert command is not None
    assert "--input={input}" in command.split()
    settings = {"project": PROJECT, "subset": "IV", "mapping_figures": False}
    record = _record("report", settings)
    _, argv = unsplit_command(
        record,
        record.workflow.task("report"),
        plugin.REPORT,
        str(store),
        str(store / "scratch"),
    )
    task_dir = store / "runs" / RUN_ID / "tasks" / "report"
    fit_out = store / "runs" / RUN_ID / "tasks" / "fit" / "out" / plugin.OUTPUT
    assert argv[:4] == ["simrun-exec", "python", "-m", "sbmlsim.run"]
    assert f"--input=fit={fit_out}" in argv
    assert "--resume" in argv
    out = task_dir / "out"
    assert argv[argv.index("--summary") + 1] == f"{out}/summary.json"

    # the command of the pod, run here: the report of the fit of the worker
    task_dir.mkdir(parents=True)
    (task_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("OMP_NUM_THREADS", "MPLBACKEND")
        and not name.startswith("SIMRUN_")
    }
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, *argv[2:]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (out / plugin.OUTPUT / "report" / "index.html").exists()
    # the summary of the task, which `simrun compare` reads, is the one of the run
    summary = (out / "summary.json").read_bytes()
    assert summary == (out / plugin.OUTPUT / "summary.json").read_bytes()
