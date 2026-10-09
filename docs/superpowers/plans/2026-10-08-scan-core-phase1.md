# The scan core, phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One way to run many similar simulations: `Scan(simulation, dimensions)` run by `Simulator.run(model, scan)` serially or in a pool of processes, answered with a `ScanResult`, replacing `ScanSim`, `XResult` and `SimulatorSerial` in every caller.

**Architecture:** `sbmlsim/parallel.py` owns every pool of the package. A scan is compiled once per combination of its simulations and models into plans (`compile_simulation`); a point is a plan with other values (`Plan.with_values`, with `at=` for a change at a time), never a compiled simulation. The points are cut into chunks which share a plan; `simulator/worker.py` runs a chunk with `execute` the same way serially and in a worker, and the parent writes the arrays of the chunks into one `xarray.Dataset`, the `ScanResult` (`result/scan.py`), whose coordinates are the labels and the changed values of the dimensions.

**Tech Stack:** python 3.13/3.14, uv, libroadrunner (CVODE), numpy, xarray (netCDF through scipy), pint, rich, pytest with xdist, ruff, ty, zensical.

**Spec:** `docs/superpowers/specs/2026-10-08-scan-core-design.md` (phase 1 of its "Phases"; the observables `Formula`, `PK`, `Custom`, `keep`, `nca` and `to_timecourses` are phase 2 and not part of this plan).

## Global Constraints

- Phase 1 starts from `develop` after `fix/pkdb-models-084` is merged and released as 0.8.5 (`sbmlsim.model.tolerances.AbsoluteTolerance`, `RoadrunnerSBMLModel.set_integrator_settings` as a method of the model, restarts in local time). Branch `scan-core-phase1`, one pull request to `develop`, one commit per task.
- No API compatibility, release 0.9.0: `ScanSim`, `XResult`, `SimulatorSerial`, `Dimension(index=...)`, `Dimension.changes`, `simulation/range.py` and `indices_from_dimensions` are removed; `XResult.from_timecourses`, `dim_mean`, `dim_std`, `dim_min`, `dim_max`, `to_dataframe`, `to_tsv`, `to_mean_dataframe` and the `_dfs` dimension go with `XResult`. `TimecourseResult` stays.
- `Dimension(id, *, values=None, simulations=None, models=None, at=None, labels=None)`, exactly one of `values`, `simulations` and `models`; values are copied into read-only numpy arrays at construction; `labels` are `0..n-1` for `values` by default and the keys for `simulations` and `models`.
- Several dimensions span their cartesian product; the points are enumerated in C order of the dimensions, the last dimension fastest.
- A point is the plan of its simulation and model, compiled once, with the values of the point applied by `Plan.with_values`; no compile per point.
- Chunks share one plan and one model and hold at most `ceil(n_points / (4 * n_workers))` and at most 1000 points.
- `resolve_workers(n_workers, n_tasks)`: `1` is serial in the calling process, a number is taken as given, `None` is `os.process_cpu_count()` for 64 points or more and serial below.
- Integrator settings are the ones of 0.8.5: every setting of roadrunner by its name, `absolute_tolerance` a float or an `AbsoluteTolerance`, default `AbsoluteTolerance()` (`1e-10` for every kind) and `relative_tolerance=1e-10`; applied to every model the simulator runs, in the parent and in every worker; a name the integrator does not have raises before a pool starts.
- Output grid: ragged native time points by default (dimension `_point`, variable `time` over `(*dims, _point)`); a dimension `time` when every plan has the same output times; `time=` interpolates linearly in the worker, the value at the time of a change is the value after it.
- Result layout `(*dims, time)` or `(*dims, _point)`; `attrs["units"]` maps every variable and coordinate to its unit; the scan, the dimension order and the integrator settings are stored in `attrs`.
- `on_error="raise"` (default) raises the error of the first failing point with its labels and values; `"flag"` gives `NaN`, the variable `status` over the scan dimensions (`0` ok, `1` failed), one warning with the count and the first ten messages in `attrs["errors"]`.
- Never use the em dash character, use a plain dash `-`.
- No agent attribution anywhere: no `Co-Authored-By` trailer, no "Generated with Claude Code" line in commits, the pull request, docs or code.
- Commit messages are full sentences which describe the outcome, in the style of `git log` (e.g. "The fit runs its repeats in the pools of sbmlsim.parallel"), with a body that explains what and why; no conventional commit prefixes.
- Never edit `CHANGELOG.md` or auto-generated files; no release notes (they belong to the release commit).
- Markdown has no hard line wraps: a paragraph, list item or table row is one line.
- Every module, class and function of the package has full type annotations and a google style docstring (ruff `D`); `tests/` and `examples/` are exempt from docstrings. A subclass marks overrides with `typing.override`.
- ty stays at zero diagnostics (`uv run ty check`); suppress only with a rule specific `# ty: ignore[rule-name]`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting, it never prints.
- Every pool comes from `sbmlsim.parallel`; never `multiprocessing.Pool` or a `ProcessPoolExecutor` of the default context.
- Every commit passes `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest -q` (the default suite runs in parallel with xdist). Use `uv run` for every command.

## Decisions this plan takes where the spec is silent

- A dimension of `simulations` replaces the simulation of the scan, `Scan(simulation, ...)` still takes one.
- The models of a `models` dimension share the selections of the first model and the units of the time and of every selection; another model raises a `ValueError` which names the model.
- The selections of a run are the selections of its (first) model, `RoadrunnerSBMLModel.selections`, set with the new `RoadrunnerSBMLModel.set_selections`; `Simulator.run` takes no selections (phase 2 adds the observables).
- A changed target is a coordinate along its dimension unless the result has a variable of the same name, i.e. a selection: then the variable stays and the coordinate is left out (`xarray` cannot hold both). A dimension id which is a selection raises.
- `status` is reserved like `time`, `_point` and `statistic`.
- `Simulator.compile` applies the changes of a model as defaults of the `preinit_changes` (`Simulation.with_preinit_defaults`), so a model of a `models` dimension carries its own changes; the experiment no longer does it.
- Inside a worker process `n_workers=None` resolves to `1`; an explicit pool in a worker raises with the guard message.
- The fit starts its own pool per fit (`parallel.start_pool`) and stops it after the fit, so the workers do not keep the initialized problems; the repeats are collected as they finish (`concurrent.futures.wait`) and ordered by their index.
- A `Data` of type TASK returns, as every `Data`, a `Quantity`: the values of the variable in the layout of the result, `(*dims, time)`; the `DataArray` with coordinates is part of sub-project 4, where the plots draw the scan dimensions. A curve of a scan draws its first point (`plot/padding.first_curve`).
- `DataSetsComparison` keeps taking data frames; `examples/comparison/diff_example.py` passes `res.ds.to_dataframe()`.

## Review Focus

- A scan of a target which is also a selection (e.g. `k1` with the default selections, which include every parameter) runs, keeps the variable `k1` over `(d, time)` and has no coordinate `k1`; covered in Task 7.
- A pooled scan with `on_error="flag"` in which one point fails in the integrator: that point is `NaN` with `status == 1`, every other point equals the serial run; covered in Task 8.
- A ragged scan whose chunks have different numbers of rows (chunks of one point each) is padded to the longest simulation, not to the longest of the first chunk; covered in Task 7.
- `time=` with a grid time which is the time of a change gives the value after the change, and a grid time outside of the simulation gives `NaN`; covered in Task 7.
- Ctrl-C during a pooled scan stops the workers, drops the pool and the next run starts a new one; covered in Task 8.

---

## File Structure

- Create `src/sbmlsim/parallel.py`: `process_context` (moved from `utils.py`), `in_worker`, `resolve_workers`, `start_pool`, `pool`, `stop`, `shutdown`, `worker_cache`, `GUARD_MESSAGE`.
- Create `src/sbmlsim/simulator/worker.py`: `ModelSpec`, `Chunk`, `ChunkResult`, `ScanPointError`, `run_chunk`, `run_chunk_in_worker`. No pint, no xarray.
- Create `src/sbmlsim/simulator/simulator.py`: `Simulator`, `ScanError`, the compile step `_Compiled` and the assembly.
- Create `src/sbmlsim/result/scan.py`: `ScanResult`.
- Rewrite `src/sbmlsim/simulation/scan.py`: `DimensionKind`, `Dimension`, `Scan`, `RESERVED`; the legacy `ScanSim` stays in it, on the new `Dimension`, until Task 11 deletes it. `simulation/range.py` is deleted in Task 4.
- Modify `src/sbmlsim/simulator/plan.py`: `Plan.with_values(values, at=None)`, `Plan.output_times`, `target_values`, `model_time`.
- Modify `src/sbmlsim/result/timecourse.py`: `interpolate`.
- Modify `src/sbmlsim/model/model_roadrunner.py`: `set_selections`.
- Modify callers: `fit/runner.py`, `fit/identifiability.py`, `fit/optimization.py`, `testsuite/runner.py`, `testsuite/submission.py`, `sensitivity/analysis.py`, `scripts/petab_benchmark.py`, `utils.py`, `experiment/experiment.py`, `experiment/runner.py`, `data.py`, `plot/padding.py`, `simulation/sensitivity.py`, the package `__init__` files.
- Delete in Task 11: `result/xresult.py`, `simulator/simulation_serial.py`, `tests/result/test_xresult.py`, `tests/simulator/test_simulator_serial.py`, `docs/api/simulation.range.md`, `docs/api/result.xresult.md`, `docs/api/simulator.simulation_serial.md`.
- Tests: create `tests/test_parallel.py`, `tests/simulator/test_plan_values.py`, `tests/simulation/test_scan_definition.py` (rewritten), `tests/result/test_scan_result.py`, `tests/simulator/test_worker.py`, `tests/simulator/test_simulator.py`, `tests/simulator/test_simulator_pool.py`, `tests/simulator/test_benchmark.py`, `tests/simulator/test_scan_regression.py`, `tests/data/scan_regression.json`; modify the tests listed per task.
- Examples: `scan.py`, `model_sensitivity.py`, `units.py`, `timecourse.py`, `demo/demo.py`, `repressilator/repressilator_scans.py`, `glucose/glucose.py`, `glucose/experiments/dose_response.py`, `initial_assignment/initial_assignment.py`, `curve_types/experiment.py`, `hctz_fitting/helpers.py`, `comparison/diff_example.py`, `README.md`.
- Docs: `scans.md` (rewritten), `simulation.md`, `units.md`, `experiments.md`, `data.md`, `index.md`, `models.md`, `references.md`, `docs/api/` (new `parallel.md`, `simulator.simulator.md`, `simulator.worker.md`, `result.scan.md`), `docs/api/index.md`, `zensical.toml`, `CLAUDE.md`, `pyproject.toml` (marker `benchmark`).

Notation: `$SCRATCH` is the scratchpad directory of the session (a temporary directory outside the repository); `$REPO` is the root of the repository. The code of 0.8.5 may differ in details from the snippets which quote existing code; match by the quoted text, not by line numbers.

---

### Task 0: The branch and the values of the scan examples before the change

**Files:**
- Create: `tests/data/scan_regression.json`

**Interfaces:**
- Consumes: the API of 0.8.5 (`ScanSim`, `SimulatorSerial`, `XResult`, `ExperimentRunner`).
- Produces: `tests/data/scan_regression.json`, a JSON object `{"<example>.<run>": {"<variable>": nested lists}}` in the layout `(*dims, time)`, read by Task 10. Keys: `scan.run_scan0d`, `scan.run_scan1d`, `scan.run_scan2d` (variables `time`, `X`, `Y`, `Z`, `PX`, `PY`, `PZ`, every time point), `repressilator_scans.task_model1_scan1d`, `repressilator_scans.task_model2_scan1d`, `repressilator_scans.task_model1_scan2d`, `repressilator_scans.task_model2_scan2d` (the same variables, every 200th of 4001 time points), `dose_response.task_glc_scan` (`time`, `[glc_ext]`, `glu`, `epi`, `ins`, `gamma`).

- [ ] **Step 1: Create the branch from the released 0.8.5**

```bash
cd $REPO
git fetch origin
git switch -c scan-core-phase1 origin/develop
grep -n "__version__" src/sbmlsim/__init__.py
test -f src/sbmlsim/model/tolerances.py && echo "0.8.5 present"
```
Expected: the version is `0.8.5` and `0.8.5 present` is printed. If not, stop: the tolerance design has not landed yet.

- [ ] **Step 2: Bring the spec and this plan onto the branch**

```bash
git log --oneline origin/develop..design/scan-core
git cherry-pick $(git rev-list --reverse origin/develop..design/scan-core -- docs/superpowers)
```
Expected: the commits which add `docs/superpowers/specs/2026-10-08-scan-core-design.md` and this plan apply cleanly.

- [ ] **Step 3: Write the recorder**

Create `$SCRATCH/record_scans.py` (not part of the repository):

```python
"""Record the values of the scan examples with the API of 0.8.5.

Run from the root of the repository, before any change of the scan core:

    PYTHONPATH=. uv run python $SCRATCH/record_scans.py tests/data/scan_regression.json

The values are stored in the layout of a `ScanResult`, `(*dims, time)`.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from examples import scan as example_scan
from examples.glucose.experiments.dose_response import DoseResponseExperiment
from examples.repressilator.repressilator_scans import RepressilatorScanExperiment
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.simulator import SimulatorSerial

#: the variables of the repressilator which are compared
REPRESSILATOR = ["time", "X", "Y", "Z", "PX", "PY", "PZ"]
#: every 200th of the 4001 time points of the repressilator scans
STEP = 200


def layout(xres: Any, key: str, step: int = 1) -> list[Any]:
    """Get a variable in the layout `(*dims, time)`, every `step`-th time point."""
    values = np.moveaxis(np.asarray(xres[key].values, dtype=float), 0, -1)
    return values[..., ::step].tolist()


def experiment_results(
    experiment_class: Any, base_path: Path, data_path: Path, reduced: bool
) -> dict[str, Any]:
    """Run an experiment and get its results by task."""
    runner = ExperimentRunner(
        [experiment_class],
        simulator=SimulatorSerial(),
        base_path=base_path,
        data_path=data_path,
    )
    with tempfile.TemporaryDirectory() as tmp:
        runner.run_experiments(
            output_path=Path(tmp), show_figures=False, reduced_selections=reduced
        )
    return next(iter(runner.experiments.values())).results


def main(path: Path) -> None:
    """Record every scan and write the JSON."""
    record: dict[str, dict[str, list[Any]]] = {}
    for name in ["run_scan0d", "run_scan1d", "run_scan2d"]:
        xres = getattr(example_scan, name)()
        record[f"scan.{name}"] = {key: layout(xres, key) for key in REPRESSILATOR}

    base = Path("examples/repressilator")
    results = experiment_results(RepressilatorScanExperiment, base, base, False)
    for model in ["model1", "model2"]:
        for simulation in ["scan1d", "scan2d"]:
            task = f"task_{model}_{simulation}"
            record[f"repressilator_scans.{task}"] = {
                key: layout(results[task], key, STEP) for key in REPRESSILATOR
            }

    glucose = Path("examples/glucose")
    results = experiment_results(
        DoseResponseExperiment, glucose, glucose / "data", True
    )
    record["dose_response.task_glc_scan"] = {
        key: layout(results["task_glc_scan"], key)
        for key in ["time", "[glc_ext]", "glu", "epi", "ins", "gamma"]
    }

    path.write_text(json.dumps(record), encoding="utf-8")
    for key, values in record.items():
        print(key, {k: np.shape(v) for k, v in values.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

- [ ] **Step 4: Record the values**

Run: `cd $REPO && PYTHONPATH=. uv run python $SCRATCH/record_scans.py tests/data/scan_regression.json`
Expected: eight lines; the shapes are `(101,)` for `scan.run_scan0d`, `(8, 101)` for `scan.run_scan1d`, `(8, 4, 101)` for `scan.run_scan2d`, `(11, 21)` for the two `scan1d` tasks, `(10, 10, 21)` for the two `scan2d` tasks and `(30, 2)` for `dose_response.task_glc_scan`.

- [ ] **Step 5: Commit**

```bash
git add tests/data/scan_regression.json
git commit -m "The values of the scan examples before the scan core are recorded" -m "The examples scan.py, repressilator_scans.py and glucose/dose_response.py are simulated with the API of 0.8.5 and their values are stored in the layout of the new result, (*dims, time). The regression test of the scan core compares the migrated examples with them."
```

---

### Task 1: `sbmlsim.parallel`, one module for every pool

**Files:**
- Create: `src/sbmlsim/parallel.py`
- Create: `tests/test_parallel.py`
- Modify: `src/sbmlsim/utils.py` (remove `process_context` and its imports `multiprocessing`, `BaseContext`)
- Modify: `src/sbmlsim/testsuite/runner.py` (`map_cases`, the import)
- Modify: `src/sbmlsim/sensitivity/analysis.py`, `scripts/petab_benchmark.py`, `src/sbmlsim/fit/runner.py` (the import of `process_context` only)
- Modify: `tests/test_utils.py` (the tests of `process_context` move), `tests/conftest.py` (fixture `_no_pool_left`, docstring of `_no_fork_of_threads`)

**Interfaces:**
- Consumes: nothing new.
- Produces (all in `sbmlsim.parallel`):
  - `POOL_THRESHOLD: int = 64`, `WORKER_STARTUP_TIMEOUT: float = 300.0`, `WORKER_CACHE_SIZE: int = 16`, `PRELOAD: tuple[str, ...] = ("sbmlsim.simulator.executor",)`, `GUARD_MESSAGE: str`.
  - `process_context() -> BaseContext`
  - `in_worker() -> bool`
  - `resolve_workers(n_workers: int | None, n_tasks: int) -> int`
  - `start_pool(n_workers: int, preload: Sequence[str] = ()) -> ProcessPoolExecutor` (a new pool, the caller stops it)
  - `pool(n_workers: int) -> ProcessPoolExecutor` (kept per process and size)
  - `stop(executor: ProcessPoolExecutor) -> None`
  - `shutdown() -> None`
  - `worker_cache[T](key: Hashable, factory: Callable[[], T]) -> T`

- [ ] **Step 1: Move the tests of `process_context`**

Create `tests/test_parallel.py` with the module docstring `"""The pools of sbmlsim."""`. Move the six tests `test_process_context_is_not_fork_by_default`, `test_process_context_keeps_the_start_method_of_the_user`, `test_process_context_does_not_take_the_default_for_a_choice`, `test_process_context_keeps_another_default`, `test_process_context_does_not_fix_the_start_method` and `test_pools_of_the_process_context_do_not_fork_a_process_with_threads` from `tests/test_utils.py` into it unchanged, with their decorators and the imports they use. Replace `from sbmlsim.utils import process_context` by `from sbmlsim.parallel import process_context`, also inside the string of the subprocess of `test_process_context_does_not_fix_the_start_method`. In `tests/test_utils.py` keep `test_paths_text` and import only `paths_text`; remove the imports which only the moved tests used (`subprocess`, `sys`, `threading`, `Callable`).

- [ ] **Step 2: Write the failing tests of the new functions**

Append to `tests/test_parallel.py` (merge the imports into the header):

```python
import multiprocessing
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

from sbmlsim import parallel

REPO = Path(__file__).parents[1]


def test_one_worker_is_serial() -> None:
    assert parallel.resolve_workers(1, 10_000) == 1


def test_a_number_of_workers_is_taken_as_given() -> None:
    assert parallel.resolve_workers(3, 2) == 3


def test_less_than_one_worker_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        parallel.resolve_workers(0, 10)


def test_none_is_serial_below_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8)
    assert parallel.resolve_workers(None, parallel.POOL_THRESHOLD - 1) == 1
    assert parallel.resolve_workers(None, parallel.POOL_THRESHOLD) == 8
    assert parallel.resolve_workers(None, 10**6) == 8


def test_none_is_serial_in_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    assert parallel.in_worker()
    assert parallel.resolve_workers(None, 10**6) == 1


def test_a_worker_starts_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    with pytest.raises(RuntimeError, match="__main__"):
        parallel.start_pool(2)


def test_the_pool_is_kept_per_size() -> None:
    first = parallel.pool(2)
    assert parallel.pool(2) is first
    assert first.submit(os.getpid).result() != os.getpid()


def test_a_stopped_pool_is_replaced() -> None:
    first = parallel.pool(2)
    parallel.stop(first)
    second = parallel.pool(2)
    assert second is not first
    assert second.submit(os.getpid).result() != os.getpid()


def test_shutdown_stops_every_pool() -> None:
    parallel.pool(2)
    parallel.shutdown()
    assert parallel._POOLS == {}


def test_the_cache_builds_an_object_once() -> None:
    built: list[int] = []

    def factory() -> int:
        built.append(1)
        return len(built)

    assert parallel.worker_cache(("test", "once"), factory) == 1
    assert parallel.worker_cache(("test", "once"), factory) == 1
    assert built == [1]


def test_the_cache_keeps_the_newest_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "WORKER_CACHE_SIZE", 2)
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    for k in range(3):
        parallel.worker_cache(("test", k), lambda k=k: k)
    assert list(parallel._CACHE) == [("test", 1), ("test", 2)]


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_the_forkserver_preloads_the_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context("forkserver")
    preloaded: list[list[str]] = []
    monkeypatch.setattr(parallel, "process_context", lambda: context)
    monkeypatch.setattr(context, "set_forkserver_preload", preloaded.append)

    assert parallel._context(["examples.demo.demo"]) is context
    assert preloaded == [sorted({*parallel.PRELOAD, "examples.demo.demo"})]


def test_another_start_method_is_used_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context("spawn")
    monkeypatch.setattr(parallel, "process_context", lambda: context)
    monkeypatch.setattr(
        context,
        "set_forkserver_preload",
        lambda modules: pytest.fail("only the forkserver preloads"),
    )
    assert parallel._context(["examples.demo.demo"]) is context


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_a_script_without_the_guard_is_reported(tmp_path: Path) -> None:
    """The workers import the script again and die, the parent names the guard."""
    script = tmp_path / "unguarded.py"
    script.write_text("from sbmlsim import parallel\n\nparallel.pool(2)\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert result.returncode != 0
    assert 'if __name__ == "__main__":' in result.stderr
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest -q -n 0 tests/test_parallel.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.parallel'`.

- [ ] **Step 4: Write `src/sbmlsim/parallel.py`**

```python
"""The process pools of sbmlsim.

Every pool of the package comes from here:

- `process_context` is the start method of every pool, never `fork` of a
  process which runs threads;
- `pool(n)` is a pool kept per process and number of workers, so that the
  start of the workers (the imports, roadrunner) is paid once per process and
  not once per run; `start_pool(n)` is a new pool which the caller stops, e.g.
  the pool of a fit, whose workers keep the initialized problem;
- `resolve_workers` turns `n_workers` into the number of processes;
- `worker_cache` keeps objects in a worker, e.g. the models of a scan or the
  initialized problem of a fit, and builds one only when its key is new.

A pool starts worker processes which import the main module again (the start
methods `forkserver` and `spawn`), so a script which starts a pool must do it
behind the guard `if __name__ == "__main__":`. A worker which starts a pool
raises, the worker dies while it starts and the parent reports the guard, see
`GUARD_MESSAGE`.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from multiprocessing.context import BaseContext
from typing import cast

logger = logging.getLogger(__name__)

#: the smallest number of tasks which `resolve_workers` runs in a pool for
#: `n_workers=None`; below it the start of the workers costs more than it saves
POOL_THRESHOLD: int = 64

#: seconds the workers of a new pool may take to start
WORKER_STARTUP_TIMEOUT: float = 300.0

#: the most objects a worker keeps, see `worker_cache`
WORKER_CACHE_SIZE: int = 16

#: the modules the forkserver imports once for all workers
PRELOAD: tuple[str, ...] = ("sbmlsim.simulator.executor",)

#: what to do about workers which do not start
GUARD_MESSAGE = (
    "A parallel run starts worker processes which import the script again, so "
    "the run must be behind a guard:\n\n"
    '    if __name__ == "__main__":\n        main()\n\n'
    "Use one worker to run without worker processes."
)

#: the pools of the process by their number of workers, see `pool`
_POOLS: dict[int, ProcessPoolExecutor] = {}

#: the objects of a worker process, see `worker_cache`
_CACHE: OrderedDict[Hashable, object] = OrderedDict()


def process_context() -> BaseContext:
    ...  # moved verbatim from `sbmlsim.utils.process_context`, docstring included


def in_worker() -> bool:
    """Check whether this process is a worker of a pool."""
    return multiprocessing.parent_process() is not None


def resolve_workers(n_workers: int | None, n_tasks: int) -> int:
    """Get the number of processes of a run.

    Args:
        n_workers: `1` runs serially in the calling process, a number is taken
            as given, `None` is the number of CPUs for `POOL_THRESHOLD` tasks
            or more and serial below, and serial in a worker process.
        n_tasks: the number of tasks of the run, e.g. the points of a scan.

    Returns:
        The number of processes, `1` for a serial run.

    Raises:
        ValueError: if `n_workers` is less than 1.
    """
    if n_workers is not None:
        if n_workers < 1:
            raise ValueError(f"The number of workers must be at least 1, not {n_workers}.")
        return n_workers
    if in_worker() or n_tasks < POOL_THRESHOLD:
        return 1
    return max(1, os.process_cpu_count() or 1)


def _context(preload: Sequence[str]) -> BaseContext:
    """Get the context of a pool, the forkserver preloads the modules.

    The forkserver imports the modules once and the workers inherit them; the
    main module is never preloaded, so a worker imports it and a script without
    the guard is found, see the module.
    """
    context = process_context()
    if context.get_start_method() == "forkserver":
        with suppress(Exception):
            # preloading is an optimization, a module which does not import
            # must not end the run
            context.set_forkserver_preload(sorted({*PRELOAD, *preload}))
    return context


def _alive() -> int:
    """Probe of a worker, which answers when it started."""
    return os.getpid()


def start_pool(n_workers: int, preload: Sequence[str] = ()) -> ProcessPoolExecutor:
    """Start a pool whose workers answer, see `pool` for one which is kept.

    Args:
        n_workers: the number of worker processes.
        preload: modules the forkserver imports once for all workers.

    Returns:
        The pool, the caller stops it with `stop`.

    Raises:
        RuntimeError: in a worker process, or if the workers die while they
            start or do not start within `WORKER_STARTUP_TIMEOUT`; both are
            what a script without the guard does, see `GUARD_MESSAGE`.
    """
    if in_worker():
        raise RuntimeError(
            f"A worker process started a pool, i.e. the script ran again when "
            f"it was imported. {GUARD_MESSAGE}"
        )
    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=_context(preload))
    try:
        executor.submit(_alive).result(timeout=WORKER_STARTUP_TIMEOUT)
    except BrokenProcessPool as err:
        stop(executor)
        raise RuntimeError(f"The workers died while they started. {GUARD_MESSAGE}") from err
    except TimeoutError as err:
        stop(executor)
        raise RuntimeError(
            f"No worker started within {WORKER_STARTUP_TIMEOUT:.0f} s. {GUARD_MESSAGE}"
        ) from err
    return executor


def pool(n_workers: int) -> ProcessPoolExecutor:
    """Get the pool of the process with a number of workers.

    The pool is kept, so the workers start once per process; a pool which
    broke, e.g. because a worker died, is replaced. Every pool is stopped when
    the process ends, see `shutdown`.

    Args:
        n_workers: the number of worker processes.

    Returns:
        The pool.

    Raises:
        RuntimeError: see `start_pool`.
    """
    executor = _POOLS.get(n_workers)
    if executor is not None and not getattr(executor, "_broken", False):
        return executor
    if executor is not None:
        stop(executor)
    executor = start_pool(n_workers)
    _POOLS[n_workers] = executor
    return executor


def stop(executor: ProcessPoolExecutor) -> None:
    """Stop a pool: the pending tasks are cancelled and the workers end.

    The workers are terminated, so a pool whose workers still run, e.g. after
    Ctrl-C or a timeout, stops at once. A kept pool is dropped, the next
    `pool` starts a new one.
    """
    for n_workers, kept in list(_POOLS.items()):
        if kept is executor:
            del _POOLS[n_workers]
    processes = list((executor._processes or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def shutdown() -> None:
    """Stop every kept pool of the process."""
    for executor in list(_POOLS.values()):
        stop(executor)


atexit.register(shutdown)


def worker_cache[T](key: Hashable, factory: Callable[[], T]) -> T:
    """Get an object of the worker process, built once per key.

    The newest `WORKER_CACHE_SIZE` objects are kept.

    Args:
        key: identifies the object, e.g. a model with the settings of its
            integrator.
        factory: builds the object when the key is new.

    Returns:
        The object of the key.
    """
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return cast(T, _CACHE[key])
    value = factory()
    _CACHE[key] = value
    while len(_CACHE) > WORKER_CACHE_SIZE:
        _CACHE.popitem(last=False)
    return value
```

Replace the `...` of `process_context` by the function moved verbatim from `utils.py`; in its docstring replace "every pool of sbmlsim must use `process_context().Pool(...)` or `ProcessPoolExecutor(mp_context=process_context())`" by "every pool of sbmlsim comes from this module". If ty reports `executor._processes` (a private attribute of the stubs), keep the access and suppress exactly the reported rule on that line.

- [ ] **Step 5: Remove `process_context` from `utils.py` and point the importers at `parallel`**

In `src/sbmlsim/utils.py` delete `process_context` and the imports only it used (`import multiprocessing`, `from multiprocessing.context import BaseContext`). Replace `from sbmlsim.utils import process_context` by `from sbmlsim.parallel import process_context` in `src/sbmlsim/fit/runner.py`, `src/sbmlsim/sensitivity/analysis.py` and `scripts/petab_benchmark.py` (the fit moves to the pools in Task 2, the sensitivity analyses in sub-project 3). Check that nothing else imports it:

Run: `rg -n "utils import .*process_context|utils\.process_context" src tests scripts examples`
Expected: no output.

- [ ] **Step 6: `map_cases` uses the kept pool**

In `src/sbmlsim/testsuite/runner.py` replace the imports `from concurrent.futures import ProcessPoolExecutor` and of `process_context` by `from sbmlsim import parallel`, and the pool of `map_cases` by:

```python
    workers = workers or os.process_cpu_count() or 1
    if workers == 1 or len(cases) <= 1:
        return [function(case) for case in cases]
    chunksize = max(1, min(8, len(cases) // (4 * workers)))
    executor = parallel.pool(min(workers, len(cases)))
    return list(executor.map(function, cases, chunksize=chunksize))
```

Keep its docstring and say in it that the pool is the kept pool of `sbmlsim.parallel`.

- [ ] **Step 7: No test leaves a pool behind**

In `tests/conftest.py` add, next to `_no_fork_of_threads`:

```python
from sbmlsim import parallel


@pytest.fixture(autouse=True)
def _no_pool_left() -> Iterator[None]:
    """Stop the pools a test started, so that no test leaves workers behind.

    A kept pool lives as long as its process; a worker of pytest-xdist runs
    many tests, whose pools would add up.
    """
    yield
    parallel.shutdown()
```

In the docstring of `_no_fork_of_threads` replace `sbmlsim.utils.process_context` by `sbmlsim.parallel.process_context`.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q -n 0 tests/test_parallel.py tests/test_utils.py && uv run pytest -q tests/testsuite`
Expected: PASS.

- [ ] **Step 9: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add src/sbmlsim/parallel.py src/sbmlsim/utils.py src/sbmlsim/testsuite/runner.py src/sbmlsim/sensitivity/analysis.py src/sbmlsim/fit/runner.py scripts/petab_benchmark.py tests/test_parallel.py tests/test_utils.py tests/conftest.py
git commit -m "Every pool of sbmlsim comes from sbmlsim.parallel" -m "process_context moves from utils.py into the new module sbmlsim.parallel, which also resolves the number of workers, keeps a pool per process and size whose workers start once, starts a pool for a caller which stops it, keeps objects in a worker and reports a script without the __main__ guard when its workers die while they start. The test suite runs its cases in the kept pool; every test stops the pools it started."
```

---

### Task 2: The fit runs its repeats and profiles in the pools of `sbmlsim.parallel`

**Files:**
- Modify: `src/sbmlsim/fit/runner.py`
- Modify: `src/sbmlsim/fit/identifiability.py`
- Modify: `tests/fit/test_robustness.py`, `tests/fit/test_fit.py`

**Interfaces:**
- Consumes: `parallel.start_pool`, `parallel.stop`, `parallel.worker_cache`, `parallel.in_worker`, `parallel.GUARD_MESSAGE` (Task 1).
- Produces (in `sbmlsim.fit.runner`):
  - `worker_problem(token: str, problem: OptimizationProblem, settings: FitSettings) -> OptimizationProblem` (raises `RuntimeError` when the worker could not initialize the problem)
  - `FitPool` (frozen dataclass: `executor`, `token`, `problem`, `settings`) with `submit(function, task) -> Future[Any]`, which calls `function(token, problem, settings, task)` in a worker
  - `worker_pool(problem, settings, n_cores) -> Generator[FitPool]` (context manager)
  - `_preload(problem: OptimizationProblem) -> list[str]`
  - removed: `WORKER_STARTUP_TIMEOUT`, `GUARD_MESSAGE`, `_WORKER_PROBLEM`, `_WORKER_ERROR`, `_worker_initialize`, `_worker_alive`, `_pool_context`, `_wait_for_workers`.

- [ ] **Step 1: Rewrite the tests of the workers**

In `tests/fit/test_robustness.py` replace `test_a_worker_without_a_problem_reports_it`, `test_the_forkserver_preloads_the_modules_of_the_fit` and `test_another_start_method_is_used_as_it_is` by:

```python
class _Broken:
    """A problem whose data cannot be resolved."""

    calls: int = 0

    def initialize(self, settings: Any) -> None:
        _Broken.calls += 1
        raise ValueError("no data")


def test_a_worker_without_a_problem_reports_it() -> None:
    """A worker which could not initialize reports it for every repeat, once."""
    _Broken.calls = 0
    token = f"broken-{uuid4().hex}"
    problem, task = _Broken(), {"run": 3, "x0": None}
    for _ in range(2):
        run, fit, trajectory = runner._worker_run(token, problem, None, task)  # ty: ignore[invalid-argument-type]
        assert run == 3
        assert fit.success is False
        assert "no data" in fit.message
        assert trajectory == []
    # the error of the initialization is kept like an initialized problem
    assert _Broken.calls == 1
    assert task == {"run": 3, "x0": None}


def test_the_fit_preloads_its_modules(op_hctz_pk: OptimizationProblem) -> None:
    """The forkserver imports the optimization and the experiments once."""
    experiments = {
        mapping_collection.experiment_class.__module__
        for mapping_collection in op_hctz_pk.mapping_collections
    }
    assert runner._preload(op_hctz_pk) == sorted(
        {"sbmlsim.fit.optimization", *experiments}
    )
```

Add `from uuid import uuid4` and `from typing import Any` if missing; remove the imports which are no longer used (`multiprocessing`, `sys` if nothing else uses them).

In `tests/fit/test_fit.py` replace `test_an_interrupted_parallel_fit_keeps_the_repeats_which_finished` by the version below and remove the import of `ApplyResult`:

```python
@pytest.mark.parametrize("interrupted", [0, 1])
def test_an_interrupted_parallel_fit_keeps_the_repeats_which_finished(
    interrupted: int,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C while the runner waits for repeats ends the parallel fit.

    A fit interrupted after repeats finished returns them; one interrupted
    before is interrupted and not reported as a fit in which every repeat
    failed.
    """
    from sbmlsim.fit import runner

    wait = runner.wait
    returned: list[int] = []

    def interrupting(fs: Any, timeout: float | None = None, return_when: str = "ALL_COMPLETED") -> Any:
        if len(returned) == interrupted:
            raise KeyboardInterrupt
        done, pending = wait(fs, timeout=timeout, return_when=return_when)
        returned.append(len(done))
        return done, pending

    monkeypatch.setattr(runner, "wait", interrupting)
    fit = partial(
        run_optimization,
        problem=op_hctz_pk,
        settings=fit_settings,
        size=2,
        seed=1234,
        n_cores=2,
        show_progress=False,
        **short_fit,
    )
    if interrupted == 0:
        with pytest.raises(KeyboardInterrupt):
            fit()
        assert returned == []
    else:
        # the first wait may return both repeats, then there is no second one
        assert fit().size == sum(returned)
        assert len(returned) == 1
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/fit/test_robustness.py -k "worker_without or preloads" tests/fit/test_fit.py -k interrupted_parallel`
Expected: FAIL (`_worker_run` takes one argument, `runner` has no `_preload` and no `wait`).

- [ ] **Step 3: Replace the pool of the fit**

In `src/sbmlsim/fit/runner.py`:

- remove `WORKER_STARTUP_TIMEOUT`, `GUARD_MESSAGE`, `_WORKER_PROBLEM`, `_WORKER_ERROR`, `_worker_initialize`, `_worker_alive`, `worker_problem()`, `_worker_run(task)`, `_pool_context`, `worker_pool`, `_wait_for_workers` and the imports only they used (`multiprocessing`, `BaseContext`, `Pool`, `suppress`, the import of `process_context`);
- add the imports `from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait`, `from dataclasses import dataclass`, `from uuid import uuid4` and `from sbmlsim import parallel`;
- in `run_optimization` replace the check `if multiprocessing.parent_process() is not None:` by `if parallel.in_worker():` and `{GUARD_MESSAGE}` in its message by `{parallel.GUARD_MESSAGE}`;
- add, where the removed worker functions were:

```python
def _initialized(
    problem: OptimizationProblem, settings: FitSettings
) -> OptimizationProblem | str:
    """Initialize the problem of a fit in a worker process, see `worker_problem`.

    Returns:
        The initialized problem, or why it could not be initialized: the error
        is kept like a problem, so a broken problem fails every repeat at once
        instead of initializing again for each.
    """
    logging.getLogger(PACKAGE_LOGGER).setLevel(logging.ERROR)
    logger.debug("worker <%s> initializing problem ...", os.getpid())
    try:
        problem.initialize(settings)
    except Exception as err:
        return f"{type(err).__name__}: {err}"
    return problem


def worker_problem(
    token: str, problem: OptimizationProblem, settings: FitSettings
) -> OptimizationProblem:
    """Get the initialized problem of a fit in a worker process.

    A worker resolves the data of the problem of a fit once and keeps it under
    the token of the fit, see `sbmlsim.parallel.worker_cache`, so the repeats
    and the scans of a profile a worker runs share one initialization. The
    workers would report the same messages about the data, once per core, so
    only their errors are shown; the runner reports the problem itself.

    Args:
        token: identifies the fit, see `FitPool`.
        problem: the problem, its definition as it is pickled into a task.
        settings: the settings of the fit.

    Returns:
        The initialized problem.

    Raises:
        RuntimeError: if the worker could not initialize the problem, with the
            error of the initialization.
    """
    initialized = parallel.worker_cache(
        ("fit", token), lambda: _initialized(problem, settings)
    )
    if isinstance(initialized, str):
        raise RuntimeError(
            f"the worker could not initialize the problem: {initialized}"
        )
    return initialized


def _worker_run(
    token: str,
    problem: OptimizationProblem,
    settings: FitSettings,
    task: dict[str, Any],
) -> tuple[int, OptimizeResult, list[float]]:
    """Run a single optimization in a worker process.

    Args:
        token: identifies the fit, see `FitPool`.
        problem: the problem of the fit.
        settings: the settings of the fit.
        task: index `run` of the repeat and the arguments of `optimize_run`.

    Returns:
        The index of the repeat, its fit and its trajectory.
    """
    arguments = dict(task)
    run: int = arguments.pop("run")
    try:
        initialized = worker_problem(token, problem, settings)
    except RuntimeError as err:
        return (
            run,
            RuntimeErrorOptimizeResult(x0=arguments.get("x0"), message=str(err)),
            [],
        )
    fit, trajectory = initialized.optimize_run(run=run, **arguments)
    return run, fit, trajectory


@dataclass(frozen=True)
class FitPool:
    """The workers of a fit: a pool and the problem its tasks run on.

    Attributes:
        executor: the pool, see `sbmlsim.parallel.start_pool`.
        token: identifies the fit in the caches of the workers.
        problem: the problem, pickled into every task as its definition.
        settings: the settings the workers initialize the problem with.
    """

    executor: ProcessPoolExecutor
    token: str
    problem: OptimizationProblem
    settings: FitSettings

    def submit(
        self, function: Callable[..., Any], task: dict[str, Any]
    ) -> Future[Any]:
        """Run `function(token, problem, settings, task)` in a worker."""
        return self.executor.submit(
            function, self.token, self.problem, self.settings, task
        )


def _preload(problem: OptimizationProblem) -> list[str]:
    """Get the modules the forkserver imports once for the workers of a fit.

    Under the `forkserver` start method every worker imports sbmlsim and the
    module of the experiments again, which costs more than a short
    optimization. The forkserver imports them once and the workers inherit
    them, which matters for a fit of several problems: the forkserver outlives
    the pool, so only the first pool pays the imports. The experiments of a
    script (`__main__`) are imported by every worker.
    """
    modules = {"sbmlsim.fit.optimization"}
    for mapping_collection in problem.mapping_collections:
        module = getattr(mapping_collection.experiment_class, "__module__", None)
        if module and module != "__main__":
            modules.add(module)
    return sorted(modules)


@contextmanager
def worker_pool(
    problem: OptimizationProblem, settings: FitSettings, n_cores: int
) -> Generator[FitPool]:
    """Start the workers of a fit and stop them after it.

    The pool belongs to the fit and is not the kept pool of
    `sbmlsim.parallel.pool`: its workers keep the initialized problem, which a
    kept pool would keep after the fit.

    Args:
        problem: the problem of the fit.
        settings: the settings of the fit.
        n_cores: the number of workers.

    Yields:
        The workers of the fit.

    Raises:
        RuntimeError: if the workers do not start, e.g. in a script without
            the guard, see `sbmlsim.parallel.start_pool`.
    """
    executor = parallel.start_pool(n_cores, preload=_preload(problem))
    try:
        yield FitPool(
            executor=executor, token=uuid4().hex, problem=problem, settings=settings
        )
    finally:
        parallel.stop(executor)
```

- [ ] **Step 4: Collect the repeats as they finish**

In `_run_optimization_parallel` remove `on_result` and replace the block from `with (` to the end of the loop by:

```python
    collected: dict[int, tuple[OptimizeResult, list[float]]] = {}
    with (
        optimization_progress(
            "optimizing", size, show_progress, workers=n_cores
        ) as progress,
        worker_pool(problem, settings, n_cores) as pool,
    ):
        # one task per repeat, so that a worker which dies loses one repeat
        futures = {pool.submit(_worker_run, task): task["run"] for task in tasks}
        pending = set(futures)
        while pending:
            # a fit which stopped making progress is a worker which is gone
            remaining = (
                None
                if timeout is None
                else max(1.0, finished + timeout + 60.0 - time.monotonic())
            )
            try:
                done, pending = wait(
                    pending, timeout=remaining, return_when=FIRST_COMPLETED
                )
            except KeyboardInterrupt:
                interrupted = True
                if collected:
                    logger.warning(
                        "'%s': the fit was interrupted, it keeps the %s repeats "
                        "which finished.",
                        problem.opid,
                        len(collected),
                    )
                break
            if not done:
                for future in pending:
                    message = (
                        f"repeat {futures[future]}: no result within "
                        f"{timeout} s and 60 s more"
                    )
                    failures.append(message)
                    logger.error("'%s': %s", problem.opid, message)
                break
            for future in sorted(done, key=futures.__getitem__):
                k = futures[future]
                try:
                    _, fit, trajectory = future.result()
                except Exception as err:
                    message = f"repeat {k}: {type(err).__name__}: {err}"
                    failures.append(message)
                    logger.error("'%s': %s", problem.opid, message)
                    continue
                finished = time.monotonic()
                _store_run(
                    problem=problem,
                    settings=settings,
                    runs_dir=runs_dir,
                    fit=fit,
                    trajectory=trajectory,
                    sid=f"{problem.opid}_run_{k}",
                )
                _advance(progress)
                collected[k] = (fit, trajectory)

    fits = [collected[k][0] for k in sorted(collected)]
    trajectories = [collected[k][1] for k in sorted(collected)]
```

Remove the earlier initializations `fits: list[OptimizeResult] = []` and `trajectories: list[list[float]] = []`; keep `failures`, `interrupted` and `finished`. Update the docstring: "Every repeat is a task of the pool, which hands the next repeat to the worker which is free; the repeats are collected as they finish and ordered by their index, so a fit with a seed gives the same result for any number of workers."

- [ ] **Step 5: The profiles use the same pool**

In `src/sbmlsim/fit/identifiability.py` replace `_worker_scan` by:

```python
def _worker_scan(
    token: str,
    problem: OptimizationProblem,
    settings: FitSettings,
    task: dict[str, Any],
) -> tuple[int, int, list[ProfilePoint]]:
    """Run a scan in a worker process of the pool of the fit.

    Returns:
        The index of the parameter, the direction and the points of the scan.
    """
    initialized = runner.worker_problem(token, problem, settings)
    return int(task["index"]), int(task["direction"]), _scan_task(initialized, task)
```

and the parallel branch of `profile_likelihood` by:

```python
        if parallel:
            with runner.worker_pool(problem, settings, n_cores) as pool:
                futures = [pool.submit(_worker_scan, task) for task in tasks]
                for future in futures:
                    index, direction, points = future.result()
                    scans[index][direction] = points
                    runner._advance(progress)
```

Import `FitSettings` there if the module does not yet.

- [ ] **Step 6: Run the fit tests**

Run: `uv run pytest -q tests/fit/test_robustness.py tests/fit/test_fit.py tests/fit/test_identifiability.py`
Expected: PASS, the parallel fits and profiles included.

- [ ] **Step 7: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add src/sbmlsim/fit/runner.py src/sbmlsim/fit/identifiability.py tests/fit/test_robustness.py tests/fit/test_fit.py
git commit -m "The fit runs its repeats and profiles in the pools of sbmlsim.parallel" -m "worker_pool starts a pool of sbmlsim.parallel for the fit and stops it after; a worker initializes the problem of the fit once and keeps it, or the error of its initialization, under a token of the fit. The repeats are collected as they finish and ordered by their index. The probe of the workers and the guard message move to sbmlsim.parallel."
```

### Task 3: A point is a plan with other values, also at a time

**Files:**
- Modify: `src/sbmlsim/simulator/plan.py`
- Create: `tests/simulator/test_plan_values.py`

**Interfaces:**
- Consumes: `Plan`, `Assignment`, `PlanEvent`, `OutputMode`, `_Converter` of `plan.py`.
- Produces (in `sbmlsim.simulator.plan`):
  - `Plan.with_values(self, values: Mapping[str, float], at: float | None = None) -> Plan`: without `at` unchanged; with `at` the values are a change at that time in the time unit of the model, added to the event of that time (replacing the assignment of a target the event has) or a new event, the events stay sorted. Raises `ValueError` for a time outside of `[start, end]` or a target of no model entity.
  - `Plan.output_times(self) -> tuple[float, ...] | None`: the output times shifted by `time_shift`, with `inf` for the steady state after the end; `None` for the steps of the integrator.
  - `target_values(target: str, values: Any, symbols: ModelSymbols, uinfo: UnitsInformation) -> np.ndarray`: a `Quantity` or numbers as floats in the unit of the target in the model.
  - `model_time(simulation: Simulation, time: Time, symbols: ModelSymbols, uinfo: UnitsInformation) -> float`: a time of a simulation in the time unit of the model.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulator/test_plan_values.py`:

```python
"""A point of a scan is a plan with other values, see `Plan.with_values`."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation, model_time, target_values
from tests.simulator.models import sbml, sbml_minutes

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(source=sbml())


def _plan(model: RoadrunnerSBMLModel, simulation: Simulation) -> Plan:
    return compile_simulation(simulation, model.symbols, model.uinfo)


def _same(model: RoadrunnerSBMLModel, plan: Plan, simulation: Simulation) -> None:
    """The plan gives the values of the compiled simulation."""
    expected = execute(_plan(model, simulation), model, SEL).values
    np.testing.assert_allclose(execute(plan, model, SEL).values, expected, rtol=1e-12)


def test_a_value_before_the_initialization(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"a0": 3.0})
    _same(model, plan, Simulation(end=2, steps=4, preinit_changes={"a0": 3.0}))


def test_a_value_replaces_every_change_of_its_target(model: RoadrunnerSBMLModel) -> None:
    base = Simulation(end=2, steps=4, changes=[Change([0.5, 1.5], {"k1": 0.1})])
    plan = _plan(model, base).with_values({"k1": 2.0})
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change([0.5, 1.5], {"k1": 2.0})]))


def test_a_value_at_a_time_is_a_change(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"k1": 2.0}, at=1.0)
    assert [event.time for event in plan.events] == [1.0]
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 2.0})]))


def test_a_value_at_the_time_of_a_change_merges_into_it(model: RoadrunnerSBMLModel) -> None:
    base = Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 0.5, "k2": 0.1})])
    plan = _plan(model, base).with_values({"k1": 2.0}, at=1.0)
    assert len(plan.events) == 1
    assert {a.target: a.value for a in plan.events[0].assignments} == {"k1": 2.0, "k2": 0.1}
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": 2.0, "k2": 0.1})]))


def test_the_events_stay_sorted(model: RoadrunnerSBMLModel) -> None:
    base = Simulation(end=2, changes=[Change(0.5, {"k2": 0.1}), Change(1.5, {"k2": 0.2})])
    plan = _plan(model, base).with_values({"k1": 2.0}, at=1.0)
    assert [event.time for event in plan.events] == [0.5, 1.0, 1.5]


def test_a_value_of_the_presimulation(model: RoadrunnerSBMLModel) -> None:
    def simulation(k1: float) -> Simulation:
        return Simulation(end=2, steps=4, presimulation=SteadyState(preinit_changes={"k1": k1}))

    _same(model, _plan(model, simulation(0.5)).with_values({"k1": 2.0}), simulation(2.0))


def test_a_compartment_at_a_time(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2, steps=4)).with_values({"C": 4.0}, at=1.0)
    _same(model, plan, Simulation(end=2, steps=4, changes=[Change(1.0, {"C": 4.0})]))


def test_a_time_outside_of_the_simulation_is_an_error(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        _plan(model, Simulation(end=2)).with_values({"k1": 2.0}, at=3.0)


def test_a_value_at_a_time_of_no_target_is_an_error(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="nope"):
        _plan(model, Simulation(end=2)).with_values({"nope": 2.0}, at=1.0)


def test_no_values_are_the_plan(model: RoadrunnerSBMLModel) -> None:
    plan = _plan(model, Simulation(end=2))
    assert plan.with_values({}, at=1.0) is plan


def test_the_output_times(model: RoadrunnerSBMLModel) -> None:
    assert _plan(model, Simulation(end=2)).output_times() is None
    shifted = Simulation(end=2, steps=2, time_shift=1.0)
    assert _plan(model, shifted).output_times() == (1.0, 2.0, 3.0)
    steady = Simulation(end=2, times=[0, 2, np.inf])
    assert _plan(model, steady).output_times() == (0.0, 2.0, np.inf)


def test_the_values_of_a_target_in_the_unit_of_the_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml_minutes())
    np.testing.assert_allclose(
        target_values("f", Q([1.0, 2.0], "g"), model.symbols, model.uinfo), [1000.0, 2000.0]
    )
    np.testing.assert_array_equal(
        target_values("f", [1.0, 2.0], model.symbols, model.uinfo), [1.0, 2.0]
    )
    with pytest.raises(ValueError, match="cannot be converted"):
        target_values("f", Q([1.0], "mmol"), model.symbols, model.uinfo)
    with pytest.raises(ValueError, match="nope"):
        target_values("nope", [1.0], model.symbols, model.uinfo)


def test_a_time_in_the_time_unit_of_the_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml_minutes())
    hours = Simulation(time_unit="hr", end=2)
    assert model_time(hours, 1.0, model.symbols, model.uinfo) == pytest.approx(60.0)
    assert model_time(hours, Q(30, "s"), model.symbols, model.uinfo) == pytest.approx(0.5)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_plan_values.py`
Expected: FAIL with `ImportError: cannot import name 'model_time'`.

- [ ] **Step 3: Implement**

In `src/sbmlsim/simulator/plan.py`, change the signature and the docstring of `Plan.with_values` and dispatch at its start:

```python
    def with_values(
        self, values: Mapping[str, float], at: float | None = None
    ) -> Plan:
        """Get the plan with other values of targets.

        A value replaces the assignment of its target wherever the plan has
        one, i.e. before the initialization, in the steady state and at every
        time, and is added to the assignments before the initialization of a
        target which the plan does not set. This is the rule of
        `Simulation.with_values` on numbers in the units of the model, which
        is what a fit and a scan set.

        With `at` the values are a change at that time instead: they join the
        event of that time, where they replace the assignment of a target the
        event already has, or are a new event. This is a dimension of a scan
        with `at`.

        Args:
            values: target -> value in the unit of the target in the model.
            at: the time of the change in the time unit of the model, `None`
                for the rule above.

        Returns:
            The new plan, this one is not changed.

        Raises:
            ValueError: if a target is not a target of the model, or `at` is
                outside of the simulation.
        """
        if not values:
            return self
        if at is not None:
            return self._with_change(values, at)
```

Keep the rest of the body as it is. Add after `with_values`:

```python
    def _with_change(self, values: Mapping[str, float], at: float) -> Plan:
        """Get the plan with the values as a change at a time, see `with_values`."""
        if not self.start <= at <= self.end:
            raise ValueError(
                f"The change of {sorted(values)} at the time {at} is outside of "
                f"the simulation [{self.start}, {self.end}] in the time unit of "
                f"the model."
            )
        added = tuple(
            Assignment(target, self.symbols.kind(target), value=float(value))
            for target, value in values.items()
        )
        events: list[PlanEvent] = []
        merged = False
        for event in self.events:
            if event.time == at:
                kept = tuple(a for a in event.assignments if a.target not in values)
                events.append(PlanEvent(at, kept + added))
                merged = True
            else:
                events.append(event)
        if not merged:
            events.append(PlanEvent(at, added))
            events.sort(key=lambda event: event.time)
        return replace(self, events=tuple(events))

    def output_times(self) -> tuple[float, ...] | None:
        """Get the times of the result of the plan.

        Returns:
            The output times shifted by `time_shift`, with `inf` last for the
            steady state after the end; `None` for the steps of the
            integrator, which differ between the values of a scan.
        """
        if self.output is not OutputMode.TIMES:
            return None
        times = tuple(t + self.time_shift for t in self.times)
        if self.steady_state_output is not None:
            times = (*times, float(np.inf))
        return times
```

Add the two module functions after `compile_simulation`:

```python
def target_values(
    target: str, values: Any, symbols: ModelSymbols, uinfo: UnitsInformation
) -> np.ndarray:
    """Get values of a target as numbers in the unit of the target in the model.

    Args:
        target: the target, `S`, `[S]` or the id of a parameter or a
            compartment.
        values: a quantity, or numbers in the unit of the target in the model.
        symbols: the symbols of the model.
        uinfo: the units of the model.

    Returns:
        The values as floats.

    Raises:
        ValueError: if the target is not a target of the model, or a quantity
            cannot be converted into the unit of the target.
    """
    symbols.kind(target)
    if not isinstance(values, Quantity):
        return np.asarray(values, dtype=float)
    unit = uinfo.get(target)
    if unit is None:
        raise ValueError(
            f"'{target}' has no unit in the model, its values '{values}' cannot "
            f"be converted: give numbers in the unit the model means."
        )
    try:
        return np.asarray(values.to(unit or "dimensionless").magnitude, dtype=float)
    except (DimensionalityError, UndefinedUnitError) as err:
        raise ValueError(
            f"The values of '{target}' in '{values.units}' cannot be converted "
            f"into the unit '{unit}' of the model: {err}"
        ) from err


def model_time(
    simulation: Simulation, time: Time, symbols: ModelSymbols, uinfo: UnitsInformation
) -> float:
    """Get a time of a simulation in the time unit of the model.

    Args:
        simulation: the simulation, a number is in its `time_unit`.
        time: a number or a quantity.
        symbols: the symbols of the model.
        uinfo: the units of the model.

    Returns:
        The time in the time unit of the model.

    Raises:
        ValueError: if the time cannot be converted.
    """
    return _Converter(simulation, symbols, uinfo).time(time)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q -n 0 tests/simulator/test_plan_values.py tests/simulator/test_plan.py tests/simulator/test_executor.py`
Expected: PASS.

- [ ] **Step 5: Lint, types and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: no diagnostics.

```bash
git add src/sbmlsim/simulator/plan.py tests/simulator/test_plan_values.py
git commit -m "A plan takes values at a time, which is a point of a scan with a change" -m "Plan.with_values(values, at=time) adds the values as a change at that time, merged into the event of the plan at that time; output_times gives the times of the result of a plan, so the simulator finds a common grid without simulating. target_values and model_time convert the values and the times of a dimension once per model."
```

---

### Task 4: `Scan` and `Dimension`

**Files:**
- Rewrite: `src/sbmlsim/simulation/scan.py` (new `DimensionKind`, `Dimension`, `Scan`, `RESERVED`; the legacy `ScanSim` stays in it on the new `Dimension` until Task 11)
- Delete: `src/sbmlsim/simulation/range.py`
- Modify: `src/sbmlsim/simulation/__init__.py`, `src/sbmlsim/result/xresult.py` (`from_timecourses`), `src/sbmlsim/simulation/sensitivity.py` (the two `Dimension(...)` calls)
- Modify every call `Dimension(..., changes=..., index=...)`: `tests/simulator/test_simulator_serial.py`, `tests/result/test_xresult.py`, `tests/result/test_timecourse.py`, `tests/experiment/test_model_changes_merge.py`, `examples/scan.py`, `examples/units.py`, `examples/demo/demo.py`, `examples/repressilator/repressilator_scans.py`, `examples/glucose/experiments/dose_response.py`, `docs/scans.md`, `docs/units.md`
- Rewrite: `tests/simulation/test_scan_definition.py`

**Interfaces:**
- Consumes: `Simulation`, `Time`, `_encode` of `simulation/definition.py`.
- Produces (in `sbmlsim.simulation.scan`, `Dimension` and `Scan` exported from `sbmlsim.simulation`):
  - `RESERVED: frozenset[str] = frozenset({"time", "_point", "statistic", "status"})`
  - `class DimensionKind(StrEnum)`: `VALUES = "values"`, `SIMULATIONS = "simulations"`, `MODELS = "models"`.
  - `Dimension(id: str, *, values: Mapping[str, Any] | None = None, simulations: Mapping[str, Simulation] | None = None, models: Mapping[str, Any] | None = None, at: Time | None = None, labels: Sequence[Any] | np.ndarray | None = None)`, frozen, attributes `id`, `kind`, `values: dict[str, np.ndarray | Quantity]`, `simulations: dict[str, Simulation]`, `models: dict[str, Any]`, `at`, `labels: np.ndarray`; `__len__`, `to_dict() -> dict[str, Any]`.
  - `Scan(simulation: Simulation, dimensions: Sequence[Dimension] = ())`, frozen, attributes `simulation`, `dimensions: tuple[Dimension, ...]`; `of(scan: Scan | Simulation) -> Scan` (classmethod), `shape: tuple[int, ...]`, `size: int`, `dims: tuple[str, ...]` (properties), `__len__`, `dimension(kind: DimensionKind) -> Dimension | None`, `simulations() -> list[Simulation]`, `points() -> Iterator[tuple[int, ...]]`, `to_dict() -> dict[str, Any]`.
  - Legacy, removed in Task 11: `ScanSim(simulation, dimensions)` with `indices()`, `to_dict()`, `to_simulations()` on the new `Dimension` (values only).

- [ ] **Step 1: Write the failing tests**

Replace `tests/simulation/test_scan_definition.py` by:

```python
"""A scan and its dimensions are validated when they are created and immutable."""

import json
import pickle
from typing import Any

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulation.scan import DimensionKind

SIM = Simulation(end=10, steps=10)


def test_the_values_are_read_only_copies() -> None:
    values = np.array([1.0, 2.0])
    dimension = Dimension("d", values={"k1": values})
    values[0] = 5.0
    assert dimension.values["k1"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        dimension.values["k1"][0] = 3.0


def test_a_quantity_keeps_its_unit_and_is_read_only() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([5, 10], "mg")}).values["PODOSE"]
    assert str(dose.units) == "milligram"
    assert dose.magnitude.tolist() == [5.0, 10.0]
    assert not dose.magnitude.flags.writeable


def test_a_list_is_an_array() -> None:
    assert len(Dimension("d", values={"k1": [1, 2, 3]})) == 3


@pytest.mark.parametrize(
    "values", [1.0, Q(5, "mg"), "k1*2", ["a", "b"], [[1.0, 2.0]], []]
)
def test_values_which_are_no_array_of_numbers_are_an_error(values: Any) -> None:
    with pytest.raises(ValueError, match="'k1'"):
        Dimension("d", values={"k1": values})


def test_the_values_of_a_dimension_have_one_length() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        Dimension("d", values={"k1": [1.0, 2.0], "k2": [1.0]})


def test_the_labels_of_values_are_their_positions() -> None:
    assert Dimension("d", values={"k1": [5.0, 6.0]}).labels.tolist() == [0, 1]


def test_the_labels_are_given() -> None:
    dimension = Dimension("d", values={"k1": [5.0, 6.0]}, labels=["low", "high"])
    assert dimension.labels.tolist() == ["low", "high"]
    assert not dimension.labels.flags.writeable


@pytest.mark.parametrize("labels", [["a"], ["a", "a"]])
def test_labels_which_do_not_fit_are_an_error(labels: list[str]) -> None:
    with pytest.raises(ValueError, match="labels"):
        Dimension("d", values={"k1": [5.0, 6.0]}, labels=labels)


def test_the_labels_of_simulations_and_models_are_their_keys() -> None:
    simulations = Dimension("regimen", simulations={"single": SIM, "multiple": SIM})
    assert simulations.kind is DimensionKind.SIMULATIONS
    assert simulations.labels.tolist() == ["single", "multiple"]
    models = Dimension("genotype", models={"wt": "wt.xml", "pm": "pm.xml"})
    assert models.kind is DimensionKind.MODELS
    assert len(models) == 2


@pytest.mark.parametrize(
    "kwargs", [{}, {"values": {"k1": [1.0]}, "simulations": {"a": SIM}}]
)
def test_a_dimension_varies_one_thing(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Dimension("d", **kwargs)


def test_only_values_have_a_time() -> None:
    with pytest.raises(ValueError, match="'at'"):
        Dimension("d", simulations={"a": SIM}, at=1.0)


def test_a_simulation_of_a_dimension_is_a_simulation() -> None:
    simulations: dict[str, Any] = {"a": "sim"}
    with pytest.raises(ValueError, match="Simulation"):
        Dimension("d", simulations=simulations)


def test_the_points_are_in_c_order() -> None:
    scan = Scan(
        SIM,
        [
            Dimension("a", values={"k1": [1.0, 2.0]}),
            Dimension("b", values={"k2": [1.0, 2.0, 3.0]}),
        ],
    )
    assert scan.shape == (2, 3)
    assert scan.size == len(scan) == 6
    assert scan.dims == ("a", "b")
    assert list(scan.points())[:4] == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_a_simulation_is_a_scan_of_one_point() -> None:
    scan = Scan.of(SIM)
    assert scan.shape == ()
    assert scan.size == 1
    assert list(scan.points()) == [()]
    assert Scan.of(scan) is scan
    assert scan.simulations() == [SIM]


@pytest.mark.parametrize("sid", ["time", "_point", "statistic", "status"])
def test_a_reserved_id_is_an_error(sid: str) -> None:
    with pytest.raises(ValueError, match="names of the result"):
        Scan(SIM, [Dimension(sid, values={"k1": [1.0]})])


def test_two_dimensions_of_one_id_are_an_error() -> None:
    with pytest.raises(ValueError, match="more than once"):
        Scan(
            SIM,
            [Dimension("d", values={"k1": [1.0]}), Dimension("d", values={"k2": [1.0]})],
        )


def test_a_dimension_named_as_a_target_is_an_error() -> None:
    with pytest.raises(ValueError, match="changed targets"):
        Scan(
            SIM,
            [Dimension("k1", values={"k2": [1.0]}), Dimension("d", values={"k1": [1.0]})],
        )


def test_a_target_is_set_once_before_the_initialization() -> None:
    with pytest.raises(ValueError, match="'k1'"):
        Scan(
            SIM,
            [Dimension("a", values={"k1": [1.0]}), Dimension("b", values={"k1": [2.0]})],
        )


def test_a_target_is_set_once_at_a_time() -> None:
    Scan(
        SIM,
        [Dimension("a", values={"k1": [1.0]}), Dimension("b", values={"k1": [2.0]}, at=5)],
    )
    with pytest.raises(ValueError, match="'k1'"):
        Scan(
            SIM,
            [
                Dimension("a", values={"k1": [1.0]}, at=5),
                Dimension("b", values={"k1": [2.0]}, at=5),
            ],
        )


def test_one_dimension_of_simulations_and_one_of_models() -> None:
    with pytest.raises(ValueError, match="at most one"):
        Scan(
            SIM,
            [Dimension("a", simulations={"x": SIM}), Dimension("b", simulations={"y": SIM})],
        )
    with pytest.raises(ValueError, match="at most one"):
        Scan(
            SIM,
            [Dimension("a", models={"x": "x.xml"}), Dimension("b", models={"y": "y.xml"})],
        )


def test_a_time_outside_of_the_simulation_is_an_error() -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(SIM, [Dimension("d", values={"k1": [1.0]}, at=11)])


def test_a_time_outside_of_a_simulation_of_a_dimension_is_an_error() -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(
            SIM,
            [
                Dimension("sim", simulations={"short": Simulation(end=1), "long": SIM}),
                Dimension("d", values={"k1": [1.0]}, at=5),
            ],
        )


def test_a_time_with_a_unit() -> None:
    hours = Simulation(time_unit="hr", end=2)
    Scan(hours, [Dimension("d", values={"k1": [1.0]}, at=Q(90, "min"))])
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(hours, [Dimension("d", values={"k1": [1.0]}, at=Q(3, "hr"))])


def test_the_scan_is_stored_as_json() -> None:
    scan = Scan(
        Simulation(end=10, changes=[Change(1, {"k1": 2.0})]),
        [
            Dimension("dose", values={"PODOSE": Q([5, 10], "mg")}, at=1),
            Dimension("regimen", simulations={"single": SIM}),
            Dimension("genotype", models={"wt": "wt.xml"}),
        ],
    )
    d = json.loads(json.dumps(scan.to_dict()))
    assert [dimension["id"] for dimension in d["dimensions"]] == [
        "dose",
        "regimen",
        "genotype",
    ]
    assert d["dimensions"][0]["values"]["PODOSE"] == {
        "value": [5.0, 10.0],
        "unit": "milligram",
    }
    assert d["dimensions"][2]["models"] == {"wt": "wt.xml"}


def test_a_scan_pickles() -> None:
    scan = Scan(SIM, [Dimension("d", values={"k1": [1.0, 2.0]})])
    again = pickle.loads(pickle.dumps(scan))
    assert again.dims == ("d",)
    assert again.dimensions[0].values["k1"].tolist() == [1.0, 2.0]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/simulation/test_scan_definition.py`
Expected: FAIL with `ImportError: cannot import name 'Scan'`.

- [ ] **Step 3: Write the new `simulation/scan.py`**

Replace `src/sbmlsim/simulation/scan.py` by:

```python
"""A scan: a simulation and the dimensions of its changes.

A `Scan` runs its `Simulation` for every point of its dimensions; several
dimensions span their cartesian product, whose points are enumerated in C
order of the dimensions, the last dimension fastest. A `Dimension` varies one
of three things:

- `values`: targets to arrays of values. The arrays of a dimension have one
  length and are coupled, i.e. point `k` sets the `k`-th value of every
  target. A value replaces its target wherever the simulation sets it and is
  a pre-initialization change otherwise, see `Simulation.with_values`; with
  `at` the values are a `Change` at that time instead. A sampled design, e.g.
  a Latin hypercube or a virtual population, is one dimension with coupled
  values. A value is a number in the unit of its target in the model or a
  quantity; a formula per point is a dimension of simulations.
- `simulations`: labels to simulations. Every point simulates its own
  simulation, which replaces the simulation of the scan.
- `models`: labels to models (an `AbstractModel`, a `RoadrunnerSBMLModel` or
  a path). Every point simulates its own model, which replaces the model of
  the run.

A dimension and a scan are validated when they are created and never change:
the values are copied into read-only arrays, and running a scan never
changes the objects of the user. `sbmlsim.simulator.Simulator.run` runs a
scan.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from sbmlsim.simulation.definition import Change, Simulation, Time, _encode
from sbmlsim.units import Quantity, ureg

#: the names of the dimensions and variables of a result, which no dimension of
#: a scan takes
RESERVED = frozenset({"time", "_point", "statistic", "status"})


class DimensionKind(StrEnum):
    """What a dimension varies."""

    VALUES = "values"
    SIMULATIONS = "simulations"
    MODELS = "models"


def _array(target: str, values: Any) -> np.ndarray | Quantity:
    """Copy the values of a target into a read-only array.

    Raises:
        ValueError: if the values are a string, a scalar, empty or not one
            dimensional numbers.
    """
    if isinstance(values, str):
        raise ValueError(
            f"The values of '{target}' are the string '{values}': the values of a "
            f"dimension are numbers, a formula per point is a dimension of "
            f"simulations."
        )
    magnitude = values.magnitude if isinstance(values, Quantity) else values
    try:
        array = np.array(magnitude, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"The values of '{target}' are not numbers: {err}") from err
    if array.ndim != 1:
        raise ValueError(
            f"The values of '{target}' must be an array of one value per point, "
            f"not of the shape {array.shape}; a scalar is a change of the "
            f"simulation."
        )
    if array.size == 0:
        raise ValueError(f"The values of '{target}' are empty.")
    array.setflags(write=False)
    if isinstance(values, Quantity):
        return ureg.Quantity(array, values.units)
    return array


def _model_text(model: Any) -> str:
    """Get the path of a model, or its id, for the provenance of a scan."""
    source = getattr(model, "source", None)
    if source is None:
        return str(model)
    if source.path is not None:
        return str(source.path)
    return str(getattr(model, "sid", None) or "<sbml>")


@dataclass(frozen=True, init=False, eq=False)
class Dimension:
    """A dimension of a scan, see the module.

    Attributes:
        id: the id of the dimension, its name in the result.
        kind: what the dimension varies.
        values: target -> read-only array, a quantity or numbers in the unit
            of the target in the model; empty unless `kind` is `VALUES`.
        simulations: label -> simulation; empty unless `kind` is
            `SIMULATIONS`.
        models: label -> model; empty unless `kind` is `MODELS`.
        at: the time of the values, `None` for the rule of
            `Simulation.with_values`.
        labels: the read-only coordinate of the dimension.
    """

    id: str
    kind: DimensionKind
    values: dict[str, np.ndarray | Quantity]
    simulations: dict[str, Simulation]
    models: dict[str, Any]
    at: Time | None
    labels: np.ndarray

    def __init__(
        self,
        id: str,
        *,
        values: Mapping[str, Any] | None = None,
        simulations: Mapping[str, Simulation] | None = None,
        models: Mapping[str, Any] | None = None,
        at: Time | None = None,
        labels: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        """Create a dimension, see the class.

        Raises:
            ValueError: if not exactly one of `values`, `simulations` and
                `models` is given, if it is empty, if the arrays of the values
                differ in their length or are no arrays of numbers, if a
                dimension which is no dimension of values has `at`, or if the
                labels do not fit.
        """
        given = [
            name
            for name, mapping in (
                ("values", values),
                ("simulations", simulations),
                ("models", models),
            )
            if mapping is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"The dimension '{id}' needs exactly one of 'values', "
                f"'simulations' and 'models', it has {given or 'none'}."
            )
        kind = DimensionKind(given[0])
        arrays: dict[str, np.ndarray | Quantity] = {}
        sims: dict[str, Simulation] = {}
        mods: dict[str, Any] = {}
        if kind is DimensionKind.VALUES:
            if not values:
                raise ValueError(f"The dimension '{id}' has no values.")
            arrays = {target: _array(target, v) for target, v in values.items()}
            lengths = {target: len(array) for target, array in arrays.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"The values of the dimension '{id}' have different lengths "
                    f"{lengths}: point k sets the k-th value of every target."
                )
            keys: list[Any] = list(range(next(iter(lengths.values()))))
        else:
            if at is not None:
                raise ValueError(
                    f"The dimension '{id}' of {kind} has a time 'at', which only "
                    f"a dimension of values has."
                )
            if kind is DimensionKind.SIMULATIONS:
                sims = dict(simulations or {})
                for label, simulation in sims.items():
                    if not isinstance(simulation, Simulation):
                        raise ValueError(
                            f"'{label}' of the dimension '{id}' is no Simulation: "
                            f"{simulation!r}."
                        )
            else:
                mods = dict(models or {})
            keys = list(sims or mods)
            if not keys:
                raise ValueError(f"The dimension '{id}' has no {kind}.")
        coordinate = np.array(keys if labels is None else list(labels))
        if coordinate.ndim != 1 or coordinate.size != len(keys):
            raise ValueError(
                f"The dimension '{id}' has {len(keys)} points, its labels "
                f"{coordinate.tolist()} do not fit."
            )
        if len(set(coordinate.tolist())) != coordinate.size:
            raise ValueError(
                f"The labels {coordinate.tolist()} of the dimension '{id}' are "
                f"not unique."
            )
        coordinate.setflags(write=False)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", arrays)
        object.__setattr__(self, "simulations", sims)
        object.__setattr__(self, "models", mods)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "labels", coordinate)

    def __len__(self) -> int:
        """Get the number of points."""
        return int(self.labels.size)

    def __repr__(self) -> str:
        """Get the representation."""
        what = list(self.values or self.simulations or self.models)
        at = "" if self.at is None else f", at={self.at}"
        return f"Dimension({self.id}[{len(self)}], {self.kind}={what}{at})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "id": self.id,
            "kind": str(self.kind),
            "labels": self.labels.tolist(),
            "values": {
                target: _encode(values) if isinstance(values, Quantity) else values.tolist()
                for target, values in self.values.items()
            },
            "simulations": {
                label: simulation.to_dict()
                for label, simulation in self.simulations.items()
            },
            "models": {label: _model_text(model) for label, model in self.models.items()},
            "at": _encode(self.at),
        }


def _at_key(at: Time | None) -> Any:
    """Get a key of a time which tells two times of dimensions apart."""
    if isinstance(at, Quantity):
        return (float(at.magnitude), str(at.units))
    return None if at is None else float(at)


def _validate(simulation: Simulation, dimensions: tuple[Dimension, ...]) -> None:
    """Check the dimensions of a scan, see `Scan`.

    Raises:
        ValueError: see `Scan`.
    """
    ids = [dimension.id for dimension in dimensions]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"The dimensions {duplicates} appear more than once in the scan.")
    reserved = sorted(set(ids) & RESERVED)
    if reserved:
        raise ValueError(
            f"The dimension ids {reserved} are names of the result "
            f"({sorted(RESERVED)}), choose other ids."
        )
    targets = {target for dimension in dimensions for target in dimension.values}
    clash = sorted(set(ids) & targets)
    if clash:
        raise ValueError(
            f"The dimension ids {clash} are changed targets, which are "
            f"coordinates of the result, choose other ids."
        )
    for kind in (DimensionKind.SIMULATIONS, DimensionKind.MODELS):
        count = sum(dimension.kind is kind for dimension in dimensions)
        if count > 1:
            raise ValueError(
                f"A scan has at most one dimension of {kind}, this one has {count}."
            )
    seen: dict[tuple[str, Any], str] = {}
    for dimension in dimensions:
        key = _at_key(dimension.at)
        for target in dimension.values:
            other = seen.get((target, key))
            if other is not None:
                when = (
                    "before the initialization"
                    if dimension.at is None
                    else f"at {dimension.at}"
                )
                raise ValueError(
                    f"The target '{target}' is set {when} by the dimensions "
                    f"'{other}' and '{dimension.id}': a point sets a target once."
                )
            seen[(target, key)] = dimension.id
    simulations = next(
        (
            list(dimension.simulations.values())
            for dimension in dimensions
            if dimension.kind is DimensionKind.SIMULATIONS
        ),
        [simulation],
    )
    for dimension in dimensions:
        if dimension.at is None:
            continue
        for sim in simulations:
            at = sim._magnitude(dimension.at)
            if not sim._magnitude(sim.start) <= at <= sim._magnitude(sim.end):
                raise ValueError(
                    f"The time {dimension.at} of the dimension '{dimension.id}' is "
                    f"outside of the simulation [{sim.start}, {sim.end}]."
                )


@dataclass(frozen=True, init=False, eq=False)
class Scan:
    """A simulation over the points of its dimensions, see the module.

    Attributes:
        simulation: the simulation, which a dimension of simulations replaces.
        dimensions: the dimensions, the last one fastest.
    """

    simulation: Simulation
    dimensions: tuple[Dimension, ...]

    def __init__(
        self, simulation: Simulation, dimensions: Sequence[Dimension] = ()
    ) -> None:
        """Create a scan, see the class.

        Raises:
            ValueError: if two dimensions have one id; if an id is in
                `RESERVED` or a changed target; if a target is set by two
                dimensions without `at` or by two with the same `at`; if there
                is more than one dimension of simulations or of models; or if
                the `at` of a dimension is outside of a simulation it applies
                to.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError(f"A scan needs a Simulation, not {simulation!r}.")
        dims = tuple(dimensions)
        _validate(simulation, dims)
        object.__setattr__(self, "simulation", simulation)
        object.__setattr__(self, "dimensions", dims)

    @classmethod
    def of(cls, scan: Scan | Simulation) -> Scan:
        """Get a scan, a simulation is a scan without dimensions."""
        return scan if isinstance(scan, Scan) else cls(scan)

    def __repr__(self) -> str:
        """Get the representation."""
        return f"Scan({self.simulation!r}, {list(self.dimensions)})"

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the number of points of every dimension."""
        return tuple(len(dimension) for dimension in self.dimensions)

    @property
    def size(self) -> int:
        """Get the number of points of the scan."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        """Get the number of points of the scan."""
        return self.size

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the ids of the dimensions."""
        return tuple(dimension.id for dimension in self.dimensions)

    def dimension(self, kind: DimensionKind) -> Dimension | None:
        """Get the dimension of simulations or of models, `None` without one."""
        return next((d for d in self.dimensions if d.kind is kind), None)

    def simulations(self) -> list[Simulation]:
        """Get the simulations of the points, those of a dimension of simulations."""
        dimension = self.dimension(DimensionKind.SIMULATIONS)
        if dimension is None:
            return [self.simulation]
        return list(dimension.simulations.values())

    def points(self) -> Iterator[tuple[int, ...]]:
        """Get the index of every point along every dimension, in C order."""
        return iter(np.ndindex(*self.shape))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types, the provenance of a result."""
        return {
            "type": self.__class__.__name__,
            "simulation": self.simulation.to_dict(),
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
        }


class ScanSim:
    """A scan of a simulation over dimensions of values, see `SimulatorSerial`.

    Superseded by `Scan` and `sbmlsim.simulator.Simulator`, it is removed with
    `SimulatorSerial`.
    """

    def __init__(
        self,
        simulation: Simulation,
        dimensions: list[Dimension] | None = None,
    ):
        """Scan a simulation.

        Raises:
            ValueError: if two dimensions have the same id or a dimension is
                no dimension of values.
        """
        self.simulation: Simulation = simulation
        self.dimensions: list[Dimension] = list(dimensions or [])
        ids = [dimension.id for dimension in self.dimensions]
        if len(ids) > len(set(ids)):
            raise ValueError(f"duplicate dimension keys in scan: {ids}")
        for dimension in self.dimensions:
            if dimension.kind is not DimensionKind.VALUES:
                raise ValueError(
                    f"A ScanSim scans values, '{dimension.id}' is a dimension of "
                    f"{dimension.kind}: use Scan."
                )

    def __repr__(self) -> str:
        """Get representation."""
        return f"ScanSim({self.simulation!r}: {self.dimensions})"

    def indices(self) -> list[tuple[int, ...]]:
        """Get the indices of all combinations of the dimensions."""
        return list(itertools.product(*(range(len(d)) for d in self.dimensions)))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "type": self.__class__.__name__,
            "simulation": self.simulation.to_dict(),
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
        }

    def to_simulations(self) -> tuple[list[tuple[int, ...]], list[Simulation]]:
        """Get the indices of every combination of the dimensions and its simulation."""
        indices = self.indices()
        simulations: list[Simulation] = []
        for index_list in indices:
            values: dict[str, Any] = {}
            timed: list[Change] = []
            for k_dim, k_index in enumerate(index_list):
                dimension = self.dimensions[k_dim]
                dim_values = {
                    key: dimension.values[key][k_index] for key in dimension.values
                }
                if dimension.at is None:
                    values.update(dim_values)
                else:
                    timed.append(Change(dimension.at, dim_values))
            simulation = self.simulation.with_values(values)
            simulation.changes.extend(timed)
            simulations.append(simulation)
        return indices, simulations
```

The parameter `id` of `Dimension` shadows the builtin on purpose (the spec names it so); ruff does not select the rules of flake8-builtins. If ty reports the private `Simulation._magnitude` or `_encode`, they are module level helpers of the same package: keep them and add nothing.

- [ ] **Step 4: Delete `range.py`, export the new classes, adapt the legacy code**

```bash
git rm src/sbmlsim/simulation/range.py
```

`src/sbmlsim/simulation/__init__.py`:

```python
"""Package for simulation."""

from .definition import Change, Simulation, SteadyState
from .scan import Dimension, Scan, ScanSim

__all__ = [
    "Change",
    "Dimension",
    "Scan",
    "ScanSim",
    "Simulation",
    "SteadyState",
]
```

In `src/sbmlsim/result/xresult.py` import only `ScanSim` (`from sbmlsim.simulation import ScanSim`) and replace the block from `# Additional dimensions` to the line `indices = Dimension.indices_from_dimensions(...)` of `from_timecourses` by:

```python
        # the dimensions of the scan, or `_dfs` for several results without one
        shape = [n_point]
        dims = ["_point"]
        coords: dict[str, np.ndarray] = {"_point": np.arange(n_point)}
        indices: list[tuple[int, ...]] = [()]
        if scan is not None and scan.dimensions:
            for dimension in scan.dimensions:
                shape.append(len(dimension))
                dims.append(dimension.id)
                coords[dimension.id] = dimension.labels
            indices = scan.indices()
        elif scan is None and len(results) > 1:
            shape.append(len(results))
            dims.append("_dfs")
            coords["_dfs"] = np.arange(len(results))
            indices = [(k,) for k in range(len(results))]
```

In `src/sbmlsim/simulation/sensitivity.py` change the import to `from sbmlsim.simulation.definition import Simulation` and `from sbmlsim.simulation.scan import Dimension, ScanSim`, and both `return Dimension("dim_sens", changes=changes)` to `return Dimension("dim_sens", values=changes)`.

- [ ] **Step 5: Every `Dimension` takes `values`**

Run: `rg -n "Dimension\(" src tests examples docs scripts --glob '!docs/superpowers/**'`
In every call listed which passes `changes=`, rename the keyword to `values=`; remove every `index=` argument (`examples/units.py`, `tests/result/test_xresult.py`). The files are `tests/simulator/test_simulator_serial.py`, `tests/result/test_xresult.py`, `tests/result/test_timecourse.py`, `tests/experiment/test_model_changes_merge.py`, `tests/test_sensitivity.py` (it reads `scan.dimensions[0].changes["n"]`: read `.values["n"]`), `examples/scan.py`, `examples/units.py`, `examples/demo/demo.py`, `examples/repressilator/repressilator_scans.py`, `examples/glucose/experiments/dose_response.py`, `docs/scans.md`, `docs/units.md`. Then:

Run: `rg -n "changes=\{|index=np" src tests examples docs --glob '!docs/superpowers/**' | rg -i "dimension"`
Expected: no output.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -q tests/simulation tests/simulator tests/result tests/experiment tests/test_sensitivity.py tests/docs tests/examples`
Expected: PASS.

- [ ] **Step 7: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add -A src/sbmlsim/simulation src/sbmlsim/result/xresult.py tests examples docs/scans.md docs/units.md
git commit -m "A Scan of dimensions of values, simulations or models replaces range.Dimension" -m "Dimension(id, values=..., simulations=..., models=..., at=..., labels=...) copies its values into read-only arrays and validates them; Scan(simulation, dimensions) validates the ids, the targets, the times and the kinds of its dimensions and enumerates its points in C order. The legacy ScanSim and XResult run on the new Dimension until the simulator replaces them; simulation/range.py and Dimension(index=, changes=) are gone."
```

---

### Task 5: `ScanResult`

**Files:**
- Create: `src/sbmlsim/result/scan.py`
- Modify: `src/sbmlsim/result/timecourse.py` (`interpolate`, the module docstring), `src/sbmlsim/result/__init__.py`
- Create: `tests/result/test_scan_result.py`

**Interfaces:**
- Consumes: nothing of earlier tasks.
- Produces:
  - `sbmlsim.result.timecourse.interpolate(time: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray`
  - `sbmlsim.result.scan`: `TIME = "time"`, `POINT = "_point"`, `STATISTIC = "statistic"`, `STATISTICS: tuple[str, ...] = ("mean", "sd", "cv", "min", "max")`, `NETCDF_ATTRS = "sbmlsim"`, and `ScanResult(ds: xr.Dataset)` with `ds`, `units: dict[str, str]`, `dims: tuple[str, ...]`, `ragged: bool`, `variables: list[str]` (properties), `__getitem__(key) -> xr.DataArray`, `__contains__`, `quantity(key) -> Quantity`, `sel(**indexers) -> ScanResult`, `isel(**indexers) -> ScanResult`, `time_points() -> np.ndarray`, `interpolate(times) -> ScanResult`, `summary(dims=None, statistics=STATISTICS, quantiles=()) -> ScanResult`, `to_netcdf(path) -> None`, `from_netcdf(path) -> ScanResult` (classmethod).
  - `ScanResult` exported from `sbmlsim.result`.
  - The dataset convention every producer follows: `attrs["dims"]` the scan dimensions in order, `attrs["units"]` variable/coordinate -> unit, optional `attrs["scan"]`, `attrs["integrator_settings"]`, `attrs["errors"]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/result/test_scan_result.py`:

```python
"""The result of a scan wraps a dataset with the units of its variables."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.result.timecourse import interpolate


def _grid() -> ScanResult:
    """Two points of a dimension `d` on the grid 0, 1, 2 with a coordinate `k1`."""
    ds = xr.Dataset(
        {"y": (("d", "time"), np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]))},
        coords={"d": [0, 1], "k1": ("d", [1.0, 2.0]), "time": [0.0, 1.0, 2.0]},
        attrs={"dims": ["d"], "units": {"y": "mM", "k1": "1/min", "time": "min"}},
    )
    return ScanResult(ds)


def _ragged() -> ScanResult:
    """Two points with their own time points, the second padded."""
    time = np.array([[0.0, 1.0, 3.0], [0.0, 2.0, np.nan]])
    y = np.array([[0.0, 1.0, 3.0], [0.0, 4.0, np.nan]])
    ds = xr.Dataset(
        {"time": (("d", "_point"), time), "y": (("d", "_point"), y)},
        coords={"d": [0, 1]},
        attrs={"dims": ["d"], "units": {"y": "mM", "time": "min"}},
    )
    return ScanResult(ds)


def test_interpolate_ignores_the_padding_and_the_steady_state() -> None:
    time = np.array([0.0, 1.0, np.inf, np.nan])
    values = np.array([[0.0, 1.0], [2.0, 3.0], [9.0, 9.0], [np.nan, np.nan]])
    out = interpolate(time, values, np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(out, [[1.0, 2.0], [2.0, 3.0], [np.nan, np.nan]])
    assert np.isnan(interpolate(np.array([np.nan]), np.array([np.nan]), np.array([0.0]))).all()


def test_the_dimensions_and_the_layout() -> None:
    assert _grid().dims == ("d",)
    assert not _grid().ragged
    assert _ragged().ragged
    assert _grid().variables == ["y"]
    assert _ragged().variables == ["y"]
    assert "k1" in _grid()


def test_a_variable_is_a_quantity_with_its_unit() -> None:
    y = _grid().quantity("y")
    assert str(y.units) == "millimolar"
    assert y.magnitude.shape == (2, 3)
    k1 = _grid().quantity("k1").to("1/s").magnitude
    np.testing.assert_allclose(k1, [1 / 60, 2 / 60])


def test_an_unknown_variable_names_the_variables() -> None:
    with pytest.raises(KeyError, match="'y'"):
        _grid()["nope"]


def test_a_selection_keeps_the_units() -> None:
    one = _grid().sel(d=1)
    assert one.dims == ()
    assert one["y"].values.tolist() == [0.0, 2.0, 4.0]
    assert one.units["y"] == "mM"
    assert _grid().isel(d=0)["k1"].item() == 1.0


def test_the_time_points() -> None:
    assert _grid().time_points().tolist() == [0.0, 1.0, 2.0]
    assert _ragged().time_points().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_ragged_result_is_interpolated_onto_a_grid() -> None:
    grid = _ragged().interpolate([0.0, 2.0, 3.0])
    assert not grid.ragged
    assert grid["y"].dims == ("d", "time")
    np.testing.assert_allclose(grid["y"].values, [[0.0, 2.0, 3.0], [0.0, 4.0, np.nan]])
    assert grid["time"].values.tolist() == [0.0, 2.0, 3.0]
    assert grid.units == _ragged().units


def test_a_grid_is_interpolated_and_a_quantity_is_converted() -> None:
    grid = _grid().interpolate(Q([30, 90], "s"))
    np.testing.assert_allclose(grid["y"].values, [[0.5, 1.5], [1.0, 3.0]])
    assert grid["k1"].values.tolist() == [1.0, 2.0]


def test_the_summary_over_a_dimension() -> None:
    summary = _grid().summary("d", quantiles=[0.5])
    assert summary["y"].dims == ("statistic", "time")
    assert summary.ds["statistic"].values.tolist() == [
        "mean",
        "sd",
        "cv",
        "min",
        "max",
        "q0.5",
    ]
    y = summary["y"]
    np.testing.assert_allclose(y.sel(statistic="mean").values, [0.0, 1.5, 3.0])
    np.testing.assert_allclose(
        y.sel(statistic="sd").values, [0.0, np.sqrt(0.5), np.sqrt(2.0)]
    )
    np.testing.assert_allclose(y.sel(statistic="max").values, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(y.sel(statistic="q0.5").values, [0.0, 1.5, 3.0])
    assert summary.dims == ()
    assert summary.units["y"] == "mM"


def test_a_ragged_summary_is_on_the_union_of_the_time_points() -> None:
    summary = _ragged().summary(statistics=["mean"])
    assert summary["y"].dims == ("statistic", "time")
    assert summary.ds["time"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_allclose(
        summary["y"].sel(statistic="mean").values, [0.0, 1.5, 3.0, 3.0]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dims": "x"}, "no dimensions"),
        ({"statistics": ["median"]}, "Unknown statistics"),
        ({"quantiles": [1.5]}, "quantile"),
    ],
)
def test_a_wrong_summary_is_an_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _grid().summary(**kwargs)


def test_the_netcdf_round_trip(tmp_path: Path) -> None:
    for result in (_grid(), _ragged()):
        path = tmp_path / "result.nc"
        result.to_netcdf(path)
        again = ScanResult.from_netcdf(path)
        xr.testing.assert_allclose(again.ds, result.ds)
        assert again.units == result.units
        assert again.dims == result.dims


def test_a_result_pickles() -> None:
    again = pickle.loads(pickle.dumps(_grid()))
    xr.testing.assert_identical(again.ds, _grid().ds)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/result/test_scan_result.py`
Expected: FAIL with `ImportError: cannot import name 'ScanResult'`.

- [ ] **Step 3: Write `interpolate`**

Append to `src/sbmlsim/result/timecourse.py` (add `import numpy as np` if missing, it is there):

```python
def interpolate(time: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a timecourse linearly onto a grid of times.

    The time points of a simulation increase and a time of a change appears
    once, with the state after the change, so the value at the time of a
    change is the value after it. The padding (`NaN`) and the steady state
    after the end (`inf`) are no time points of the interpolation.

    Args:
        time: the time points.
        values: the values, a row per time point, one or two dimensional.
        grid: the times of the result.

    Returns:
        The values at the times of the grid, a row per time; `NaN` outside of
        the finite time points and for a timecourse without any.
    """
    grid = np.asarray(grid, dtype=float)
    mask = np.isfinite(time)
    out = np.full((grid.size, *values.shape[1:]), np.nan)
    if not mask.any():
        return out
    t, v = time[mask], values[mask]
    if v.ndim == 1:
        return np.interp(grid, t, v, left=np.nan, right=np.nan)
    for j in range(v.shape[1]):
        out[:, j] = np.interp(grid, t, v[:, j], left=np.nan, right=np.nan)
    return out
```

In the module docstring of `timecourse.py` replace the sentence "`XResult.from_timecourses` places the results of the simulations of a scan into one `xarray.Dataset`." by "`sbmlsim.simulator.Simulator.run` places the results of the simulations of a scan into a `ScanResult`."

- [ ] **Step 4: Write `src/sbmlsim/result/scan.py`**

```python
"""The result of a scan, see `sbmlsim.simulator.Simulator.run`.

A `ScanResult` wraps one `xarray.Dataset`:

- dimensions: the dimensions of the scan in their order, then `time` when
  every simulation has the same output times and `_point` otherwise, i.e.
  `(*dims, time)` or `(*dims, _point)`, the layout of the timecourses of
  pkpdutils;
- variables: one per selection over `(*dims, time)` or `(*dims, _point)`; in
  the ragged layout of `_point` every simulation keeps its own time points
  and the variable `time` over `(*dims, _point)` holds them, padded with
  `NaN`; `status` over the dimensions of the scan for a run with
  `on_error="flag"`;
- coordinates: the labels of every dimension, every changed target of a
  dimension of values along its dimension, and `time` on a grid;
- `attrs`: `dims`, the dimensions of the scan in their order, `units`, the
  unit of every variable and coordinate, `scan` and `integrator_settings`,
  the provenance, and `errors` for a run with `on_error="flag"`.

`xarray` does the rest: `res["[X]"].sel(dose=10)`, `res.ds.to_dataframe()`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from sbmlsim.result.timecourse import interpolate
from sbmlsim.units import Quantity, ureg

#: the dimension of the time of a grid
TIME = "time"
#: the dimension of the output points of the ragged layout
POINT = "_point"
#: the dimension of `ScanResult.summary`
STATISTIC = "statistic"
#: the statistics of `ScanResult.summary`
STATISTICS: tuple[str, ...] = ("mean", "sd", "cv", "min", "max")
#: the attribute of a netCDF file which holds the attributes of the result
NETCDF_ATTRS = "sbmlsim"


class ScanResult:
    """The result of a scan, see the module.

    Attributes:
        ds: the dataset.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset which follows the layout of the module."""
        self.ds = ds

    def __repr__(self) -> str:
        """Get the representation."""
        sizes = ", ".join(f"{dim}: {size}" for dim, size in self.ds.sizes.items())
        return f"ScanResult({sizes}; {', '.join(self.variables)})"

    @property
    def units(self) -> dict[str, str]:
        """Get the unit of every variable and coordinate."""
        return self.ds.attrs.get("units", {})

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the dimensions of the scan which the result has, in their order."""
        return tuple(d for d in self.ds.attrs.get("dims", []) if d in self.ds.dims)

    @property
    def ragged(self) -> bool:
        """Check whether every simulation keeps its own time points."""
        return POINT in self.ds.dims

    @property
    def variables(self) -> list[str]:
        """Get the variables of the selections, without `time` and `status`."""
        return [str(name) for name in self.ds.data_vars if name not in (TIME, "status")]

    def __getitem__(self, key: str) -> xr.DataArray:
        """Get a variable or a coordinate.

        Raises:
            KeyError: if the result has no variable or coordinate of the name.
        """
        try:
            return self.ds[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is no variable or coordinate of the result, its "
                f"variables are {self.variables}."
            ) from None

    def __contains__(self, key: object) -> bool:
        """Check whether the result has a variable or coordinate."""
        return key in self.ds.variables

    def quantity(self, key: str) -> Quantity:
        """Get the values of a variable or coordinate with its unit."""
        values = np.asarray(self[key].values, dtype=float)
        return ureg.Quantity(values, self.units.get(key, ""))

    def sel(self, **indexers: Any) -> ScanResult:
        """Select by labels, see `xarray.Dataset.sel`."""
        return ScanResult(self.ds.sel(indexers))

    def isel(self, **indexers: Any) -> ScanResult:
        """Select by positions, see `xarray.Dataset.isel`."""
        return ScanResult(self.ds.isel(indexers))

    def time_points(self) -> np.ndarray:
        """Get the times of the grid, or the union of the time points of the simulations."""
        times = np.asarray(self.ds[TIME].values, dtype=float).ravel()
        if not self.ragged:
            return times
        return np.unique(times[np.isfinite(times)])

    def _grid(self, times: ArrayLike | Quantity) -> np.ndarray:
        """Get times as numbers in the time unit of the result."""
        if isinstance(times, Quantity):
            unit = self.units.get(TIME) or "dimensionless"
            return np.asarray(times.to(unit).magnitude, dtype=float).ravel()
        return np.asarray(times, dtype=float).ravel()

    def interpolate(self, times: ArrayLike | Quantity) -> ScanResult:
        """Get the result on a grid of times.

        Every simulation is interpolated linearly onto the times, see
        `sbmlsim.result.timecourse.interpolate`; a time outside of a
        simulation is `NaN`. The variables without a time stay as they are.

        Args:
            times: the times of the grid, numbers in the time unit of the
                result or a quantity.

        Returns:
            The result with the dimension `time`.
        """
        grid = self._grid(times)
        tdim = POINT if self.ragged else TIME
        time_dims = self.ds[TIME].dims
        variables: dict[str, Any] = {}
        for name, array in self.ds.data_vars.items():
            if name == TIME:
                continue
            if tdim not in array.dims:
                variables[str(name)] = array
                continue
            order = time_dims if self.ragged else (*[d for d in array.dims if d != TIME], TIME)
            values = np.asarray(array.transpose(*order).values, dtype=float)
            times_of = (
                np.asarray(self.ds[TIME].values, dtype=float)
                if self.ragged
                else np.broadcast_to(np.asarray(self.ds[TIME].values, dtype=float), values.shape)
            )
            out = np.full((*values.shape[:-1], grid.size), np.nan)
            for index in np.ndindex(*values.shape[:-1]):
                out[index] = interpolate(times_of[index], values[index], grid)
            variables[str(name)] = ([*order[:-1], TIME], out)
        coords = {
            name: coord
            for name, coord in self.ds.coords.items()
            if name != TIME and tdim not in coord.dims
        }
        coords[TIME] = grid
        return ScanResult(xr.Dataset(variables, coords=coords, attrs=dict(self.ds.attrs)))

    def summary(
        self,
        dims: str | Sequence[str] | None = None,
        statistics: Sequence[str] = STATISTICS,
        quantiles: Sequence[float] = (),
    ) -> ScanResult:
        """Get statistics of the variables over dimensions of the scan.

        A result in the ragged layout is interpolated onto the union of its
        time points first. `sd` is the sample standard deviation and `cv` the
        ratio of `sd` and `mean`; a quantile `q` is the statistic `q<q>`,
        e.g. `q0.05`. `NaN`, e.g. of a failed point, is skipped. The unit of a
        variable is the unit of its statistics, except of `cv`, a ratio.

        Args:
            dims: the dimensions to reduce, every dimension of the scan by
                default.
            statistics: statistics of `STATISTICS`.
            quantiles: quantiles between 0 and 1.

        Returns:
            The result with the dimension `statistic` instead of the reduced
            dimensions.

        Raises:
            ValueError: if a dimension is no dimension of the scan, a
                statistic is unknown or a quantile is outside of [0, 1].
        """
        reduced = (
            list(self.dims)
            if dims is None
            else [dims]
            if isinstance(dims, str)
            else list(dims)
        )
        unknown = sorted(set(reduced) - set(self.dims))
        if unknown:
            raise ValueError(
                f"{unknown} are no dimensions of the scan, its dimensions are "
                f"{list(self.dims)}."
            )
        wrong = sorted(set(statistics) - set(STATISTICS))
        if wrong:
            raise ValueError(
                f"Unknown statistics {wrong}, the statistics are {list(STATISTICS)}."
            )
        outside = [q for q in quantiles if not 0.0 <= q <= 1.0]
        if outside:
            raise ValueError(f"The quantiles {outside} are outside of [0, 1].")
        source = self.interpolate(self.time_points()) if self.ragged else self
        ds = source.ds.drop_vars("status", errors="ignore")
        parts: list[xr.Dataset] = []
        labels: list[str] = []
        for statistic in statistics:
            if statistic == "mean":
                part = ds.mean(dim=reduced, skipna=True)
            elif statistic == "sd":
                part = ds.std(dim=reduced, skipna=True, ddof=1)
            elif statistic == "cv":
                part = ds.std(dim=reduced, skipna=True, ddof=1) / ds.mean(
                    dim=reduced, skipna=True
                )
            elif statistic == "min":
                part = ds.min(dim=reduced, skipna=True)
            else:
                part = ds.max(dim=reduced, skipna=True)
            parts.append(part)
            labels.append(statistic)
        for q in quantiles:
            parts.append(ds.quantile(q, dim=reduced, skipna=True).drop_vars("quantile"))
            labels.append(f"q{q:g}")
        summary = xr.concat(parts, dim=pd.Index(labels, name=STATISTIC))
        summary.attrs = dict(self.ds.attrs)
        return ScanResult(summary)

    def to_netcdf(self, path: str | Path) -> None:
        """Write the result as netCDF, the attributes as JSON."""
        ds = self.ds.copy()
        ds.attrs = {NETCDF_ATTRS: json.dumps(self.ds.attrs)}
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ScanResult:
        """Read a result written by `to_netcdf`."""
        ds = xr.load_dataset(path)
        ds.attrs = json.loads(ds.attrs.get(NETCDF_ATTRS, "{}"))
        return cls(ds)
```

`src/sbmlsim/result/__init__.py`:

```python
"""Results of simulations and simulation experiments."""

from .scan import ScanResult
from .timecourse import TimecourseResult
from .xresult import XResult

__all__ = ["ScanResult", "TimecourseResult", "XResult"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -n 0 tests/result`
Expected: PASS. If `xr.concat` warns about the coordinates of the parts, pass `coords="minimal", compat="override"`.

- [ ] **Step 6: Lint, types and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: no diagnostics.

```bash
git add src/sbmlsim/result tests/result/test_scan_result.py
git commit -m "ScanResult is the result of a scan, an xarray dataset with units" -m "The dimensions of the scan come first and the time last, (*dims, time) on a common grid or (*dims, _point) for the native time points; the units of the variables and coordinates, the order of the dimensions and the provenance are attributes of the dataset. summary reduces over dimensions into a dimension statistic, interpolate puts a ragged result on a grid, netCDF stores the attributes as JSON, and a result pickles."
```

---

### Task 6: The worker of a scan

**Files:**
- Create: `src/sbmlsim/simulator/worker.py`
- Modify: `tests/simulator/models.py` (`BLOWUP`)
- Create: `tests/simulator/test_worker.py`

**Interfaces:**
- Consumes: `Plan.with_values(values, at=)` (Task 3), `interpolate` (Task 5), `parallel.worker_cache` (Task 1), `execute`, `RoadrunnerSBMLModel.set_integrator_settings` (0.8.5).
- Produces (in `sbmlsim.simulator.worker`):
  - `OnError = Literal["raise", "flag"]`, `MAX_ERRORS: int = 10`
  - `class ScanPointError(RuntimeError)` with `index: int`, `message: str`; `ScanPointError(index, message)` pickles.
  - `@dataclass(frozen=True) class ModelSpec(key: str, source: str, base_path: Path | None, parameters: tuple[tuple[str, float], ...], settings: tuple[tuple[str, Any], ...])` with `of(model: RoadrunnerSBMLModel, settings: Mapping[str, Any]) -> ModelSpec` (classmethod) and `load() -> RoadrunnerSBMLModel`.
  - `@dataclass(frozen=True) class Chunk(indices: np.ndarray, plan: Plan, model: int, selections: tuple[str, ...], values: dict[str, np.ndarray], timed: dict[float, dict[str, np.ndarray]], time: np.ndarray | None, on_error: OnError = "raise")` with `plan_of(k: int) -> Plan`.
  - `@dataclass(frozen=True) class ChunkResult(indices: np.ndarray, values: np.ndarray, status: np.ndarray, errors: tuple[tuple[int, str], ...])`; `values` is `(point, row, column)`, the columns those of `selections`, the time first.
  - `run_chunk(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult`
  - `run_chunk_in_worker(spec: ModelSpec, chunk: Chunk) -> ChunkResult`
  - `tests.simulator.models.BLOWUP` (antimony): `S' = k S^2`, the integration up to the time 1 fails for `k = 2`.

- [ ] **Step 1: Add the model which fails**

Append to `tests/simulator/models.py`:

```python
#: a species which grows as `S' = k S^2` and goes to infinity at the time
#: `1 / (k S0)`: the integration up to the time 1 fails for `k = 2` and works
#: for `k = 0.1`
BLOWUP = """
model blowup
  compartment C = 1
  species S in C = 1
  k = 0.1
  J: -> S; k*S^2
end
"""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/simulator/test_worker.py`:

```python
"""A chunk of a scan runs its points on one plan and one model."""

import pickle
from collections import OrderedDict

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import compile_simulation
from sbmlsim.simulator.worker import (
    Chunk,
    ModelSpec,
    OnError,
    ScanPointError,
    run_chunk,
    run_chunk_in_worker,
)
from tests.simulator.models import BLOWUP, sbml

SEL = ("time", "[A]", "[B]", "k1")


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(source=sbml())


def _chunk(
    model: RoadrunnerSBMLModel,
    simulation: Simulation,
    values: dict[str, np.ndarray] | None = None,
    timed: dict[float, dict[str, np.ndarray]] | None = None,
    time: np.ndarray | None = None,
    selections: tuple[str, ...] = SEL,
    on_error: OnError = "raise",
    indices: np.ndarray | None = None,
) -> Chunk:
    return Chunk(
        indices=np.arange(2) if indices is None else indices,
        plan=compile_simulation(simulation, model.symbols, model.uinfo),
        model=0,
        selections=selections,
        values=values or {},
        timed=timed or {},
        time=time,
        on_error=on_error,
    )


def test_every_point_is_its_simulation(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(
        model,
        Simulation(end=2, steps=4),
        values={"b0": np.array([0.0, 2.0])},
        timed={1.0: {"k1": np.array([0.1, 3.0])}},
    )
    result = run_chunk(chunk, model)
    assert result.values.shape == (2, 5, 4)
    assert result.status.tolist() == [0, 0]
    for k, (b0, k1) in enumerate([(0.0, 0.1), (2.0, 3.0)]):
        simulation = Simulation(
            end=2, steps=4, preinit_changes={"b0": b0}, changes=[Change(1.0, {"k1": k1})]
        )
        plan = compile_simulation(simulation, model.symbols, model.uinfo)
        np.testing.assert_allclose(result.values[k], execute(plan, model, SEL).values, rtol=1e-12)


def test_the_points_are_padded_to_the_longest(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(model, Simulation(end=2), values={"k1": np.array([0.1, 30.0])})
    result = run_chunk(chunk, model)
    rows = [int(np.isfinite(result.values[k, :, 0]).sum()) for k in range(2)]
    assert rows[0] < rows[1] == result.values.shape[1]
    assert np.isnan(result.values[0, rows[0] :]).all()


def test_a_grid_of_times_is_interpolated(model: RoadrunnerSBMLModel) -> None:
    grid = np.array([0.0, 0.5, 1.0, 3.0])
    chunk = _chunk(
        model,
        Simulation(end=2, changes=[Change(1.0, {"[A]": 5.0})]),
        values={"k1": np.array([0.1, 0.2])},
        time=grid,
    )
    result = run_chunk(chunk, model)
    assert result.values.shape == (2, 4, 4)
    np.testing.assert_array_equal(result.values[0, :, 0], grid)
    # the value after the change and none after the end
    assert result.values[0, 2, 1] == pytest.approx(5.0)
    assert np.isnan(result.values[0, 3, 1])


def _blowup_chunk(on_error: OnError) -> tuple[Chunk, RoadrunnerSBMLModel]:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    chunk = _chunk(
        blowup,
        Simulation(end=1, steps=4),
        values={"k": np.array([0.1, 2.0, 0.1])},
        selections=("time", "S"),
        on_error=on_error,
        indices=np.array([4, 5, 6]),
    )
    return chunk, blowup


def test_a_failed_point_is_flagged() -> None:
    chunk, blowup = _blowup_chunk("flag")
    result = run_chunk(chunk, blowup)
    assert result.status.tolist() == [0, 1, 0]
    assert np.isnan(result.values[1]).all()
    assert np.isfinite(result.values[[0, 2]]).all()
    assert result.errors[0][0] == 5
    assert "CVODE" in result.errors[0][1]


def test_a_failed_point_raises_with_its_index() -> None:
    chunk, blowup = _blowup_chunk("raise")
    with pytest.raises(ScanPointError) as info:
        run_chunk(chunk, blowup)
    assert info.value.index == 5
    again = pickle.loads(pickle.dumps(info.value))
    assert (again.index, again.message) == (5, info.value.message)


def test_the_spec_of_a_model_loads_it_with_the_settings(model: RoadrunnerSBMLModel) -> None:
    settings: dict[str, float | AbsoluteTolerance] = {
        "absolute_tolerance": AbsoluteTolerance(
            amount=1e-12, concentration=1e-9, ids={"A": 1e-14}
        ),
        "relative_tolerance": 1e-8,
    }
    model.set_integrator_settings(**settings)
    spec = ModelSpec.of(model, settings)
    loaded = spec.load()
    np.testing.assert_allclose(
        loaded.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
        model.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
    )
    assert loaded.r_loaded.getIntegrator().getValue("relative_tolerance") == 1e-8
    assert ModelSpec.of(model, settings).key == spec.key
    assert ModelSpec.of(model, {**settings, "relative_tolerance": 1e-6}).key != spec.key
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_a_worker_loads_a_model_once(
    model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    spec = ModelSpec.of(model, {})
    loads: list[str] = []
    load = ModelSpec.load

    def counting(self: ModelSpec) -> RoadrunnerSBMLModel:
        loads.append(self.key)
        return load(self)

    monkeypatch.setattr(ModelSpec, "load", counting)
    chunk = _chunk(model, Simulation(end=1, steps=2), values={"k1": np.array([0.1, 0.2])})
    first = run_chunk_in_worker(spec, chunk)
    second = run_chunk_in_worker(spec, chunk)
    np.testing.assert_array_equal(first.values, second.values)
    assert loads == [spec.key]
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_worker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbmlsim.simulator.worker'`.

- [ ] **Step 4: Write `src/sbmlsim/simulator/worker.py`**

```python
"""The worker of a scan: the points of a chunk on one plan and one model.

`run_chunk` is the same function serially and in a worker process: for every
point of a chunk it applies the values of the point to the plan of the chunk,
`Plan.with_values`, runs the plan with `execute` and stacks the native
solutions into one array, padded with `NaN`; with a grid of times every
solution is interpolated onto it first. Nothing in here uses pint or xarray,
and a chunk and its result are numbers, strings and a plan, so they pickle.

In a worker process the model of a chunk is loaded once from its `ModelSpec`
and kept, see `sbmlsim.parallel.worker_cache`, with the settings of the
integrator of the parent, so every worker integrates with the tolerances of
the parent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sbmlsim import parallel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.result.timecourse import interpolate
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan

#: what a run does about a point which fails: raise its error or flag it
OnError = Literal["raise", "flag"]

#: the most messages of failed points a chunk reports
MAX_ERRORS: int = 10


class ScanPointError(RuntimeError):
    """A point of a scan failed and the run raises, see `Chunk.on_error`.

    Attributes:
        index: the flat index of the point in the scan.
        message: the error of the point.
    """

    def __init__(self, index: int, message: str) -> None:
        """Create the error of a point, the arguments pickle."""
        super().__init__(index, message)
        self.index = index
        self.message = message

    def __str__(self) -> str:
        """Get the message."""
        return f"point {self.index}: {self.message}"


@dataclass(frozen=True)
class ModelSpec:
    """What a worker needs to load a model as the parent loaded it.

    Attributes:
        key: identifies the model and the settings of its integrator, the key
            of the model in the cache of a worker.
        source: the path of the model or the SBML.
        base_path: the directory a relative path is resolved against.
        parameters: the parameters added to the model, see `AbstractModel`.
        settings: the settings of the integrator.
    """

    key: str
    source: str
    base_path: Path | None
    parameters: tuple[tuple[str, float], ...]
    settings: tuple[tuple[str, Any], ...]

    @classmethod
    def of(cls, model: RoadrunnerSBMLModel, settings: Mapping[str, Any]) -> ModelSpec:
        """Get the spec of a loaded model and the settings of its integrator."""
        source = (
            model.source.content
            if model.source.content is not None
            else str(model.source.path)
        )
        parameters = tuple(
            sorted((str(k), float(v)) for k, v in (model.parameters or {}).items())
        )
        items = tuple(sorted(settings.items()))
        digest = hashlib.sha256(
            repr((source, model.base_path, parameters, items)).encode("utf-8")
        ).hexdigest()
        return cls(
            key=digest,
            source=source,
            base_path=model.base_path,
            parameters=parameters,
            settings=items,
        )

    def load(self) -> RoadrunnerSBMLModel:
        """Load the model and set the settings of its integrator."""
        model = RoadrunnerSBMLModel(
            source=self.source,
            base_path=self.base_path,
            parameters=dict(self.parameters) or None,
        )
        model.set_integrator_settings(**dict(self.settings))
        return model


@dataclass(frozen=True)
class Chunk:
    """Points of a scan which share a plan and a model.

    Attributes:
        indices: the flat indices of the points in the scan, in C order.
        plan: the plan of the points.
        model: the index of the model of the points among the models of the
            run.
        selections: the columns of the result, `time` first.
        values: target -> value of every point, which replaces the target
            wherever the plan sets it and is a change before the
            initialization otherwise.
        timed: time -> target -> value of every point, a change at that time.
        time: the grid of times to interpolate onto, `None` for the time
            points of the simulation.
        on_error: what to do about a point which fails.
    """

    indices: np.ndarray
    plan: Plan
    model: int
    selections: tuple[str, ...]
    values: dict[str, np.ndarray]
    timed: dict[float, dict[str, np.ndarray]]
    time: np.ndarray | None
    on_error: OnError = "raise"

    def plan_of(self, k: int) -> Plan:
        """Get the plan of the `k`-th point of the chunk."""
        plan = self.plan.with_values({t: float(v[k]) for t, v in self.values.items()})
        for at, values in self.timed.items():
            plan = plan.with_values({t: float(v[k]) for t, v in values.items()}, at=at)
        return plan


@dataclass(frozen=True)
class ChunkResult:
    """The answer of a chunk.

    Attributes:
        indices: the flat indices of the points, those of the chunk.
        values: the values, `(point, row, column)` with the columns of the
            selections, the time first, padded with `NaN`.
        status: `0` for a point which ran, `1` for one which failed.
        errors: the flat index and the error of the first `MAX_ERRORS`
            points which failed.
    """

    indices: np.ndarray
    values: np.ndarray
    status: np.ndarray
    errors: tuple[tuple[int, str], ...]


def run_chunk(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
    """Run the points of a chunk on a loaded model, see the module.

    Args:
        chunk: the points.
        model: the loaded model of the chunk.

    Returns:
        The values of every point.

    Raises:
        ScanPointError: for the first point which fails, with
            `on_error="raise"`.
    """
    n = len(chunk.indices)
    rows: list[np.ndarray | None] = []
    status = np.zeros(n, dtype=np.int8)
    errors: list[tuple[int, str]] = []
    for k in range(n):
        index = int(chunk.indices[k])
        try:
            result = execute(chunk.plan_of(k), model, chunk.selections)
        except Exception as err:
            message = f"{type(err).__name__}: {err}"
            if chunk.on_error == "raise":
                raise ScanPointError(index, message) from err
            status[k] = 1
            if len(errors) < MAX_ERRORS:
                errors.append((index, message))
            rows.append(None)
            continue
        values = result.values
        if chunk.time is not None:
            values = interpolate(values[:, 0], values, chunk.time)
            values[:, 0] = chunk.time
        rows.append(values)
    n_rows = (
        chunk.time.size
        if chunk.time is not None
        else max((r.shape[0] for r in rows if r is not None), default=0)
    )
    out = np.full((n, n_rows, len(chunk.selections)), np.nan)
    for k, values in enumerate(rows):
        if values is not None:
            out[k, : values.shape[0]] = values
    return ChunkResult(
        indices=chunk.indices, values=out, status=status, errors=tuple(errors)
    )


def run_chunk_in_worker(spec: ModelSpec, chunk: Chunk) -> ChunkResult:
    """Run a chunk in a worker process, on the model the worker keeps.

    Args:
        spec: the model of the chunk and the settings of its integrator.
        chunk: the points.

    Returns:
        The values of every point, see `run_chunk`.
    """
    model = parallel.worker_cache(("model", spec.key), spec.load)
    return run_chunk(chunk, model)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -n 0 tests/simulator/test_worker.py`
Expected: PASS (CVODE prints its warnings of the failing point to stderr, which is expected).

- [ ] **Step 6: Lint, types and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check`
Expected: no diagnostics.

```bash
git add src/sbmlsim/simulator/worker.py tests/simulator/models.py tests/simulator/test_worker.py
git commit -m "A chunk of a scan runs its points on one plan, serially or in a worker" -m "run_chunk applies the values of every point to the plan of the chunk, executes it and stacks the native solutions into one padded array, interpolated onto a grid of times if one is given; a failed point raises with its index or is flagged. A worker loads the model of a chunk once from its ModelSpec, with the integrator settings of the parent."
```

---

### Task 7: `Simulator`, the serial run

**Files:**
- Create: `src/sbmlsim/simulator/simulator.py`
- Modify: `src/sbmlsim/simulator/__init__.py`, `src/sbmlsim/model/model_roadrunner.py` (`set_selections`)
- Create: `tests/simulator/test_simulator.py`

**Interfaces:**
- Consumes: `Scan`, `Dimension`, `DimensionKind` (Task 4), `Plan.with_values`, `Plan.output_times`, `target_values`, `model_time` (Task 3), `ScanResult`, `POINT`, `TIME` (Task 5), `Chunk`, `ChunkResult`, `ModelSpec`, `OnError`, `ScanPointError`, `MAX_ERRORS`, `run_chunk`, `run_chunk_in_worker` (Task 6), `parallel.resolve_workers`, `parallel.pool`, `parallel.stop` (Task 1).
- Produces:
  - `RoadrunnerSBMLModel.set_selections(self, selections: Sequence[str] | None) -> None`
  - In `sbmlsim.simulator.simulator`, exported from `sbmlsim.simulator`: `MAX_CHUNK: int = 1000`, `class ScanError(RuntimeError)`, `class Simulator` with `__init__(self, n_workers: int | None = None, **integrator_settings: float | int | bool | AbsoluteTolerance)`, attributes `n_workers`, `integrator_settings: dict[str, ...]`, methods `set_integrator_settings(**integrator_settings) -> None`, `load(model) -> RoadrunnerSBMLModel`, `compile(model: RoadrunnerSBMLModel, simulation: Simulation) -> Plan`, `simulate(model, simulation: Simulation | Plan) -> TimecourseResult`, `run(model, scan, *, time=None, on_error="raise", progress=None) -> ScanResult`. `model` is a `RoadrunnerSBMLModel`, an `AbstractModel`, a `str` or a `Path`; `run` takes `None` with a dimension of models.
  - The pooled branch `Simulator._run_pool` is written here and tested in Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulator/test_simulator.py`:

```python
"""The simulator runs a scan and answers with a ScanResult, here serially."""

from pathlib import Path

import numpy as np
import pytest

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import Q
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.result import ScanResult, TimecourseResult
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import ScanError, Simulator
from tests.simulator.models import BLOWUP, PROBE, sbml, sbml_minutes

SEL = ["time", "[A]", "[B]", "X", "k1"]


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=sbml())
    model.set_selections(SEL)
    return model


@pytest.fixture
def simulator() -> Simulator:
    return Simulator(n_workers=1)


def test_the_selections_of_a_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    every = list(model.selections)
    model.set_selections(["time", "X"])
    assert model.selections == ["time", "X"]
    model.set_selections(None)
    assert model.selections == every


def test_simulate_is_one_timecourse(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    result = simulator.simulate(model, Simulation(end=1, steps=2))
    assert isinstance(result, TimecourseResult)
    assert result.columns == tuple(SEL)


def test_a_simulation_is_a_scan_without_dimensions(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    res = simulator.run(model, Simulation(end=1, steps=10))
    assert isinstance(res, ScanResult)
    assert res.dims == ()
    assert res["[B]"].dims == ("time",)
    np.testing.assert_allclose(res["time"].values, np.linspace(0, 1, 11))
    assert res.variables == ["[A]", "[B]", "X", "k1"]
    assert res.ds.attrs["integrator_settings"]["absolute_tolerance"] == AbsoluteTolerance().to_dict()


def test_the_steps_of_the_integrator_are_ragged(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    res = simulator.run(model, Simulation(end=1))
    assert res.ragged
    assert res["time"].dims == ("_point",)
    expected = simulator.simulate(model, Simulation(end=1))
    np.testing.assert_allclose(res["[A]"].values, expected["[A]"], rtol=1e-12)


def test_a_dimension_of_values(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    scan = Scan(Simulation(end=1, steps=10), [Dimension("d", values={"b0": [0.0, 2.0]})])
    res = simulator.run(model, scan)
    assert res["[B]"].dims == ("d", "time")
    np.testing.assert_allclose(res["[B]"].values[:, 0], [0.0, 2.0])
    # a changed target is a coordinate
    assert res["b0"].dims == ("d",)
    assert res["b0"].values.tolist() == [0.0, 2.0]
    assert res.units["b0"] == ""
    for k, b0 in enumerate([0.0, 2.0]):
        expected = simulator.simulate(
            model, Simulation(end=1, steps=10, preinit_changes={"b0": b0})
        )
        np.testing.assert_allclose(res["[B]"].values[k], expected["[B]"], rtol=1e-12)


def test_a_quantity_is_a_coordinate_in_its_unit() -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    minutes.set_selections(["time", "X"])
    scan = Scan(Simulation(end=1, steps=2), [Dimension("dose", values={"f": Q([1.0, 2.0], "g")})])
    res = Simulator(n_workers=1).run(minutes, scan)
    assert res["f"].values.tolist() == [1.0, 2.0]
    assert res.units["f"] == "gram"
    # X = 3 * pinit = 6 * f, f in mg in the model
    np.testing.assert_allclose(res["X"].values[:, 0], [6000.0, 12000.0])


def test_a_change_at_a_time(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    scan = Scan(
        Simulation(end=2, steps=4), [Dimension("d", values={"k1": [0.1, 3.0]}, at=1.0)]
    )
    res = simulator.run(model, scan)
    for k, k1 in enumerate([0.1, 3.0]):
        expected = simulator.simulate(
            model, Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": k1})])
        )
        np.testing.assert_allclose(res["[A]"].values[k], expected["[A]"], rtol=1e-12)


def test_the_points_are_in_c_order(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    scan = Scan(
        Simulation(end=1, steps=2),
        [
            Dimension("a", values={"b0": [1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.3]}),
        ],
    )
    res = simulator.run(model, scan)
    assert res["[B]"].shape == (2, 3, 3)
    expected = simulator.simulate(
        model, Simulation(end=1, steps=2, preinit_changes={"b0": 2.0, "k2": 0.3})
    )
    np.testing.assert_allclose(res["[B]"].values[1, 2], expected["[B]"], rtol=1e-12)
    assert res["k2"].values.tolist() == [0.1, 0.2, 0.3]


def test_a_dimension_of_simulations(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    simulations = {"short": Simulation(end=1, steps=2), "long": Simulation(end=2, steps=4)}
    scan = Scan(Simulation(end=1), [Dimension("sim", simulations=simulations)])
    res = simulator.run(model, scan)
    assert res.ragged
    short = res["time"].sel(sim="short").values
    assert short[:3].tolist() == [0.0, 0.5, 1.0]
    assert np.isnan(short[3:]).all()
    assert res["time"].sel(sim="long").values.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_simulations_with_one_output_share_a_grid(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulations = {
        "a": Simulation(end=1, steps=2),
        "b": Simulation(end=1, steps=2, preinit_changes={"b0": 2.0}),
    }
    res = simulator.run(model, Scan(Simulation(end=1), [Dimension("sim", simulations=simulations)]))
    assert not res.ragged
    assert res["[B]"].sel(sim="b").values[0] == pytest.approx(2.0)


def test_a_dimension_of_models(
    simulator: Simulator, model: RoadrunnerSBMLModel, tmp_path: Path
) -> None:
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    scan = Scan(
        Simulation(end=1, steps=2), [Dimension("model", models={"fast": model, "slow": slow})]
    )
    res = simulator.run(None, scan)
    assert res["[A]"].dims == ("model", "time")
    assert res["k1"].sel(model="slow").values[0] == pytest.approx(0.1)
    assert res["k1"].sel(model="fast").values[0] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="replaces the model"):
        simulator.run(model, scan)
    with pytest.raises(ValueError, match="needs a model"):
        simulator.run(None, Simulation(end=1))


def test_a_model_without_a_selection_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=2), [Dimension("model", models={"probe": model, "blowup": blowup})]
    )
    with pytest.raises(ValueError, match="'blowup'"):
        simulator.run(None, scan)


def test_models_with_other_units_are_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"seconds": model, "minutes": minutes})],
    )
    with pytest.raises(ValueError, match="units of 'time'"):
        simulator.run(None, scan)


def test_a_grid_of_times(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    simulation = Simulation(end=2, changes=[Change(1.0, {"[A]": 5.0})])
    res = simulator.run(model, simulation, time=[0.0, 0.5, 1.0, 3.0])
    assert res["[A]"].dims == ("time",)
    assert res["time"].values.tolist() == [0.0, 0.5, 1.0, 3.0]
    # the value after the change, none after the end
    assert res["[A]"].sel(time=1.0).item() == pytest.approx(5.0)
    assert np.isnan(res["[A]"].sel(time=3.0).item())


def test_a_grid_of_times_with_a_unit() -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    res = Simulator(n_workers=1).run(minutes, Simulation(end=2), time=Q([0, 60], "s"))
    assert res["time"].values.tolist() == [0.0, 1.0]


def test_a_failed_point_raises_with_its_labels_and_values(simulator: Simulator) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})])
    with pytest.raises(ScanError, match=r"rate=1, k=2\.0"):
        simulator.run(blowup, scan)


def test_a_failed_point_is_flagged(simulator: Simulator, caplog: pytest.LogCaptureFixture) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0, 0.2]})])
    res = simulator.run(blowup, scan, on_error="flag")
    assert res["status"].values.tolist() == [0, 1, 0]
    assert np.isnan(res["S"].values[1]).all()
    assert np.isfinite(res["S"].values[[0, 2]]).all()
    assert res.ds.attrs["errors"][0].startswith("rate=1, k=2.0: RuntimeError")
    assert "1 of 3 points of the scan failed" in caplog.text


def test_no_compile_per_point(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    compile_simulation = simulator_module.compile_simulation

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return compile_simulation(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(simulator_module, "compile_simulation", counting)
    values = Scan(
        Simulation(end=1, steps=2),
        [
            Dimension("a", values={"b0": [1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.3]}, at=0.5),
        ],
    )
    simulator.run(model, values)
    assert len(calls) == 1
    simulations = {"a": Simulation(end=1, steps=2), "b": Simulation(end=2, steps=2)}
    calls.clear()
    simulator.run(
        model,
        Scan(
            Simulation(end=1),
            [Dimension("sim", simulations=simulations), Dimension("b", values={"k2": [0.1, 0.2]})],
        ),
    )
    assert len(calls) == 2


def test_the_objects_of_the_user_are_not_changed(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulation = Simulation(
        end=1, steps=2, preinit_changes={"a0": 2.0}, changes=[Change(0.5, {"k1": 0.1})]
    )
    dimension = Dimension("d", values={"k1": [0.3, 0.4]}, at=0.5)
    before = (simulation.to_dict(), dimension.to_dict(), list(model.selections))
    simulator.run(model, Scan(simulation, [dimension, Dimension("e", values={"a0": [1.0, 5.0]})]))
    assert (simulation.to_dict(), dimension.to_dict(), list(model.selections)) == before


def test_a_dimension_named_as_a_selection_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match=r"\['X'\]"):
        simulator.run(model, Scan(Simulation(end=1), [Dimension("X", values={"b0": [1.0]})]))


def test_a_scanned_selection_is_a_variable_and_no_coordinate(simulator: Simulator) -> None:
    model = RoadrunnerSBMLModel(source=sbml())  # every entity is selected
    scan = Scan(Simulation(end=1, steps=2), [Dimension("d", values={"k1": [0.5, 1.0]})])
    res = simulator.run(model, scan)
    assert res["k1"].dims == ("d", "time")
    np.testing.assert_allclose(res["k1"].values[:, 0], [0.5, 1.0])
    assert "k1" not in res.ds.coords


def test_ragged_chunks_are_padded_to_the_longest(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    # the first chunk is the short one
    scan = Scan(Simulation(end=2), [Dimension("d", values={"k1": [0.1, 30.0]})])
    res = simulator.run(model, scan)
    lengths = []
    for k, k1 in enumerate([0.1, 30.0]):
        expected = simulator.simulate(model, Simulation(end=2, preinit_changes={"k1": k1}))
        n = len(expected)
        lengths.append(n)
        np.testing.assert_allclose(res["[A]"].values[k, :n], expected["[A]"], rtol=1e-12)
        assert np.isnan(res["time"].values[k, n:]).all()
    assert lengths[0] < lengths[1] == res.ds.sizes["_point"]


def test_the_changes_of_a_model_are_defaults(simulator: Simulator, tmp_path: Path) -> None:
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    abstract = AbstractModel(source=path, changes={"b0": 3.0})
    res = simulator.run(abstract, Simulation(end=1, steps=2))
    assert res["[B]"].values[0] == pytest.approx(3.0)
    scan = Scan(Simulation(end=1, steps=2), [Dimension("d", values={"b0": [1.0, 4.0]})])
    res = simulator.run(abstract, scan)
    np.testing.assert_allclose(res["[B]"].values[:, 0], [1.0, 4.0])


def test_the_integrator_settings_reach_the_model(model: RoadrunnerSBMLModel) -> None:
    tolerance = AbsoluteTolerance(ids={"A": 1e-14})
    simulator = Simulator(n_workers=1, absolute_tolerance=tolerance, relative_tolerance=1e-8)
    assert simulator.load(model) is model
    assert model.absolute_tolerance == tolerance
    assert model.r_loaded.getIntegrator().getValue("relative_tolerance") == 1e-8


def test_an_unknown_setting_is_an_error(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="no settings"):
        Simulator(n_workers=1, nope=1.0).run(model, Simulation(end=1))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/simulator/test_simulator.py`
Expected: FAIL with `ImportError: cannot import name 'ScanError' from 'sbmlsim.simulator'`.

- [ ] **Step 3: `RoadrunnerSBMLModel.set_selections`**

Add to `RoadrunnerSBMLModel` in `src/sbmlsim/model/model_roadrunner.py`, after `set_timecourse_selections` (import `Sequence` from `collections.abc` if missing):

```python
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, every entity of the model for `None`,
                see `set_timecourse_selections`; the parameters added to the
                model are not selected by default.

        Raises:
            RuntimeError: if roadrunner has no selection of a name.
        """
        self.selections = self.set_timecourse_selections(
            self.r_loaded,
            selections=None if selections is None else list(selections),
            exclude=set(self.parameters),
        )
```

- [ ] **Step 4: Write `src/sbmlsim/simulator/simulator.py`**

```python
"""The simulator: simulations and scans of models, serially or in a pool.

`Simulator.run(model, scan)` runs every point of a scan in four steps:

1. Compile: a plan per combination of the simulations and the models of the
   scan (one plan without a dimension of simulations or of models), with
   `compile_simulation`; the values of the dimensions of values in the units
   of their targets in every model, once; the times of the dimensions with
   `at` in the time unit of every model; the selections and the grid of the
   output.
2. Points: a point is a tuple of indices, nothing is built per point in the
   parent. The points are cut into chunks which share a plan, at most
   `ceil(n_points / (4 * n_workers))` and at most `MAX_CHUNK` points each.
3. Worker: a chunk applies the values of each point to its plan and runs it,
   see `sbmlsim.simulator.worker`; the same function runs serially in the
   calling process and in a worker of a pool of `sbmlsim.parallel`.
4. Assembly: the arrays of the chunks are written into the arrays of the
   result, a `ScanResult`, and reshaped into the dimensions of the scan.

The result does not depend on the number of workers or the size of the
chunks: the values of a scan are fixed before the run.

The integrator settings of a simulator apply to every model it runs: the model
of a run, every model of a dimension of models and every model a worker
loads, each with its own tolerance per state.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from sbmlsim import parallel
from sbmlsim.console import console
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.result.scan import POINT, TIME, ScanResult
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.scan import DimensionKind, Scan
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import Plan, compile_simulation, model_time, target_values
from sbmlsim.simulator.worker import (
    MAX_ERRORS,
    Chunk,
    ChunkResult,
    ModelSpec,
    OnError,
    ScanPointError,
    run_chunk,
    run_chunk_in_worker,
)
from sbmlsim.units import Quantity

logger = logging.getLogger(__name__)

#: the most points of a chunk
MAX_CHUNK: int = 1000

#: a setting of the integrator, see `RoadrunnerSBMLModel.set_integrator_settings`
IntegratorSetting = float | int | bool | AbsoluteTolerance

#: a model the simulator runs: a loaded model, its description or its path
ModelLike = RoadrunnerSBMLModel | AbstractModel | str | Path


class ScanError(RuntimeError):
    """A point of a scan failed; the message names its labels and values."""


class Simulator:
    """Runs simulations and scans of models, see the module.

    Attributes:
        n_workers: the processes of a run, see
            `sbmlsim.parallel.resolve_workers`.
        integrator_settings: the settings of the integrator of every model.
    """

    def __init__(
        self, n_workers: int | None = None, **integrator_settings: IntegratorSetting
    ) -> None:
        """Create a simulator.

        Args:
            n_workers: `1` runs serially, a number is the number of processes,
                `None` uses every CPU for a scan of
                `sbmlsim.parallel.POOL_THRESHOLD` points or more and runs a
                smaller one serially.
            **integrator_settings: every setting of the integrator of
                roadrunner by its name, e.g. `relative_tolerance` or
                `variable_step_size`; `absolute_tolerance` is a float or an
                `AbsoluteTolerance`. The defaults are `AbsoluteTolerance()`
                and `relative_tolerance=1e-10`.

        Raises:
            ValueError: if `n_workers` is less than 1.
        """
        if n_workers is not None and n_workers < 1:
            raise ValueError(
                f"The number of workers must be at least 1, not {n_workers}."
            )
        self.n_workers = n_workers
        self.integrator_settings: dict[str, IntegratorSetting] = {
            "absolute_tolerance": AbsoluteTolerance(),
            "relative_tolerance": 1e-10,
            **integrator_settings,
        }

    def __repr__(self) -> str:
        """Get the representation."""
        return f"Simulator(n_workers={self.n_workers}, {self.integrator_settings})"

    def set_integrator_settings(self, **integrator_settings: IntegratorSetting) -> None:
        """Set settings of the integrator for every model run after.

        A name the integrator does not have raises when a model is loaded,
        i.e. before a pool starts.
        """
        self.integrator_settings.update(integrator_settings)

    def load(self, model: ModelLike) -> RoadrunnerSBMLModel:
        """Get a loaded model with the integrator settings of the simulator.

        A `RoadrunnerSBMLModel` is used as it is and gets the settings; an
        `AbstractModel`, with its selections, or a path is loaded.

        Raises:
            ValueError: if the model is of no supported type, or the
                integrator has no setting of a name.
        """
        if isinstance(model, RoadrunnerSBMLModel):
            loaded = model
        elif isinstance(model, AbstractModel):
            loaded = RoadrunnerSBMLModel.from_abstract_model(
                abstract_model=model, selections=model.selections
            )
        elif isinstance(model, (str, Path)):
            loaded = RoadrunnerSBMLModel(source=model)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        loaded.set_integrator_settings(**self.integrator_settings)
        return loaded

    def compile(self, model: RoadrunnerSBMLModel, simulation: Simulation) -> Plan:
        """Compile a simulation against a model.

        The changes of the model are defaults of the pre-initialization
        changes, see `Simulation.with_preinit_defaults`: a change of the model
        applies unless the simulation sets the target.
        """
        if model.changes:
            simulation = simulation.with_preinit_defaults(model.changes)
        return compile_simulation(simulation, model.symbols, model.uinfo)

    def simulate(
        self, model: ModelLike, simulation: Simulation | Plan
    ) -> TimecourseResult:
        """Run one simulation with the selections of the model.

        The native solution of one simulation, which the fit and the test
        suites use; a scan is `run`.
        """
        loaded = self.load(model)
        plan = (
            simulation if isinstance(simulation, Plan) else self.compile(loaded, simulation)
        )
        return execute(plan, loaded, loaded.selections or [TIME])

    def run(
        self,
        model: ModelLike | None,
        scan: Scan | Simulation,
        *,
        time: ArrayLike | Quantity | None = None,
        on_error: OnError = "raise",
        progress: bool | None = None,
    ) -> ScanResult:
        """Run every point of a scan, see the module.

        The variables of the result are the selections of the model, of the
        first model of a dimension of models.

        Args:
            model: the model of the run; `None` for a scan with a dimension
                of models, which gives the model of every point.
            scan: the scan, a simulation is a scan without dimensions.
            time: a grid of times every timecourse is interpolated onto,
                numbers in the time unit of the model or a quantity; without
                it the result keeps the output times of the simulations, a
                dimension `time` if they agree and `_point` otherwise.
            on_error: `"raise"` raises a `ScanError` for the first point which
                fails; `"flag"` sets every value of a point which fails to
                `NaN` and records it in the variable `status`.
            progress: show a progress bar; `None` shows it for a run in a pool.

        Returns:
            The result, see `sbmlsim.result.scan`.

        Raises:
            ValueError: if the scan does not fit its models, see `_compile`,
                or the integrator has no setting of a name.
            ScanError: for a point which fails with `on_error="raise"`.
            RuntimeError: if a worker of the pool dies.
        """
        if on_error not in ("raise", "flag"):
            raise ValueError(f"'on_error' is 'raise' or 'flag', not '{on_error}'.")
        compiled = self._compile(model, Scan.of(scan), time)
        workers = parallel.resolve_workers(self.n_workers, compiled.size)
        chunks = compiled.chunks(workers, on_error)
        show = workers > 1 if progress is None else progress
        with _progress(show, compiled.size) as advance:
            if workers == 1:
                results = self._run_serial(compiled, chunks, advance)
            else:
                results = self._run_pool(compiled, chunks, workers, advance)
        return compiled.assemble(results, on_error, self.integrator_settings)

    def _models(
        self, model: ModelLike | None, scan: Scan
    ) -> tuple[list[RoadrunnerSBMLModel], list[str]]:
        """Get the loaded models of a run and their labels."""
        dimension = scan.dimension(DimensionKind.MODELS)
        if dimension is None:
            if model is None:
                raise ValueError(
                    "The run needs a model: give the model of the run or a "
                    "dimension of models."
                )
            return [self.load(model)], [""]
        if model is not None:
            raise ValueError(
                f"The dimension of models '{dimension.id}' replaces the model of "
                f"the run: give `None` as the model."
            )
        models = [self.load(m) for m in dimension.models.values()]
        return models, [str(label) for label in dimension.labels]

    def _compile(
        self, model: ModelLike | None, scan: Scan, time: ArrayLike | Quantity | None
    ) -> _Compiled:
        """Compile a scan against its models, see the module.

        Raises:
            ValueError: if neither the run nor a dimension of models gives a
                model, or both do; if a model of a dimension of models has not
                the selections or the units of the first one; if a dimension
                id is a selection; if a simulation, a value or a time does not
                fit a model; or if two dimensions set one target at one time.
        """
        models, labels = self._models(model, scan)
        first = models[0]
        selections = tuple(first.selections or [TIME])
        if selections[0] != TIME:
            selections = (TIME, *selections)
        for loaded, label in zip(models[1:], labels[1:], strict=True):
            _check_model(loaded, label, first, selections)
        clash = sorted(set(scan.dims) & set(selections))
        if clash:
            raise ValueError(
                f"The dimension ids {clash} are selections of the model: a "
                f"dimension and a variable of the result share no name, choose "
                f"other ids."
            )
        plans: dict[tuple[int, int], Plan] = {}
        at_times: dict[tuple[int, int], list[float | None]] = {}
        for s, simulation in enumerate(scan.simulations()):
            for m, loaded in enumerate(models):
                try:
                    plan = self.compile(loaded, simulation)
                    at_times[(s, m)] = _at_times(scan, simulation, loaded, plan)
                except ValueError as err:
                    raise ValueError(f"{_where(scan, s, labels[m])}: {err}") from err
                plans[(s, m)] = plan
        vectors = [
            _vectors(scan, loaded, label)
            for loaded, label in zip(models, labels, strict=True)
        ]
        grid, interpolate = _grid(plans, time, first)
        return _Compiled(
            scan=scan,
            models=models,
            plans=plans,
            vectors=vectors,
            at_times=at_times,
            selections=selections,
            grid=grid,
            interpolate=interpolate,
        )

    def _run_serial(
        self,
        compiled: _Compiled,
        chunks: Sequence[Chunk],
        advance: Callable[[int], None],
    ) -> list[ChunkResult]:
        """Run the chunks in the calling process."""
        results: list[ChunkResult] = []
        for chunk in chunks:
            try:
                results.append(run_chunk(chunk, compiled.models[chunk.model]))
            except ScanPointError as err:
                raise compiled.error(err) from err
            advance(len(chunk.indices))
        return results

    def _run_pool(
        self,
        compiled: _Compiled,
        chunks: Sequence[Chunk],
        workers: int,
        advance: Callable[[int], None],
    ) -> list[ChunkResult]:
        """Run the chunks in the kept pool of `sbmlsim.parallel`."""
        specs = [ModelSpec.of(model, self.integrator_settings) for model in compiled.models]
        executor = parallel.pool(workers)
        futures = [
            executor.submit(run_chunk_in_worker, specs[chunk.model], chunk)
            for chunk in chunks
        ]
        results: list[ChunkResult] = []
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                advance(len(result.indices))
        except ScanPointError as err:
            for future in futures:
                future.cancel()
            raise compiled.error(err) from err
        except BrokenProcessPool as err:
            parallel.stop(executor)
            raise RuntimeError(
                f"A worker of the pool died while it ran the scan: {err}"
            ) from err
        except BaseException:
            # e.g. Ctrl-C: the workers still run their chunks, the pool stops
            parallel.stop(executor)
            raise
        return results


@dataclass
class _Compiled:
    """A scan compiled against its models, see `Simulator._compile`.

    Attributes:
        scan: the scan.
        models: the loaded models, the one of the run or one per label of the
            dimension of models.
        plans: (index of the simulation, index of the model) -> plan.
        vectors: model -> dimension -> target -> the values of the dimension
            in the unit of the target in the model.
        at_times: plan -> dimension -> the time of its values in the time unit
            of the model, `None` for a dimension without `at`.
        selections: the columns of the result, `time` first.
        grid: the times of a grid, `None` for the ragged layout.
        interpolate: whether the workers interpolate onto the grid.
    """

    scan: Scan
    models: list[RoadrunnerSBMLModel]
    plans: dict[tuple[int, int], Plan]
    vectors: list[list[dict[str, np.ndarray]]]
    at_times: dict[tuple[int, int], list[float | None]]
    selections: tuple[str, ...]
    grid: np.ndarray | None
    interpolate: bool

    @property
    def size(self) -> int:
        """Get the number of points."""
        return self.scan.size

    def _positions(self) -> np.ndarray:
        """Get the index of every point along every dimension, a row per point."""
        if not self.scan.dimensions:
            return np.zeros((1, 0), dtype=int)
        return np.stack(np.unravel_index(np.arange(self.size), self.scan.shape), axis=1)

    def _axis(self, kind: DimensionKind) -> int | None:
        """Get the position of the dimension of a kind, `None` without one."""
        return next(
            (k for k, d in enumerate(self.scan.dimensions) if d.kind is kind), None
        )

    def chunks(self, workers: int, on_error: OnError) -> list[Chunk]:
        """Cut the points into chunks which share a plan, see the module."""
        size = max(1, min(MAX_CHUNK, math.ceil(self.size / (4 * workers))))
        positions = self._positions()
        sim_axis = self._axis(DimensionKind.SIMULATIONS)
        model_axis = self._axis(DimensionKind.MODELS)
        chunks: list[Chunk] = []
        for (s, m), plan in self.plans.items():
            mask = np.ones(self.size, dtype=bool)
            if sim_axis is not None:
                mask &= positions[:, sim_axis] == s
            if model_axis is not None:
                mask &= positions[:, model_axis] == m
            indices = np.flatnonzero(mask)
            for start in range(0, indices.size, size):
                part = indices[start : start + size]
                values: dict[str, np.ndarray] = {}
                timed: dict[float, dict[str, np.ndarray]] = {}
                for i in range(len(self.scan.dimensions)):
                    at = self.at_times[(s, m)][i]
                    for target, vector in self.vectors[m][i].items():
                        point_values = vector[positions[part, i]]
                        if at is None:
                            values[target] = point_values
                        else:
                            timed.setdefault(at, {})[target] = point_values
                chunks.append(
                    Chunk(
                        indices=part,
                        plan=plan,
                        model=m,
                        selections=self.selections,
                        values=values,
                        timed=timed,
                        time=self.grid if self.interpolate else None,
                        on_error=on_error,
                    )
                )
        return chunks

    def point_text(self, index: int) -> str:
        """Get the labels and the values of a point, for a message."""
        if not self.scan.dimensions:
            return ""
        parts: list[str] = []
        position = np.unravel_index(index, self.scan.shape)
        for dimension, k in zip(self.scan.dimensions, position, strict=True):
            parts.append(f"{dimension.id}={dimension.labels[k]}")
            parts.extend(f"{t}={v[k]}" for t, v in dimension.values.items())
        return ", ".join(parts)

    def error(self, err: ScanPointError) -> ScanError:
        """Get the error of a failed point with its labels and values."""
        text = self.point_text(err.index)
        where = f"The point {text} of the scan" if text else "The simulation"
        return ScanError(f"{where} failed: {err.message}")

    def assemble(
        self,
        results: Sequence[ChunkResult],
        on_error: OnError,
        settings: Mapping[str, Any],
    ) -> ScanResult:
        """Write the arrays of the chunks into the result, see `ScanResult`."""
        n, shape = self.size, self.scan.shape
        n_rows = (
            self.grid.size
            if self.grid is not None
            else max((r.values.shape[1] for r in results), default=0)
        )
        cube = np.full((len(self.selections), n, n_rows), np.nan)
        status = np.zeros(n, dtype=np.int8)
        errors: list[tuple[int, str]] = []
        for result in results:
            rows = result.values.shape[1]
            cube[:, result.indices, :rows] = np.moveaxis(result.values, 2, 0)
            status[result.indices] = result.status
            errors.extend(result.errors)

        first = self.models[0]
        dims = list(self.scan.dims)
        tdim = TIME if self.grid is not None else POINT
        units: dict[str, str] = {TIME: first.uinfo.get(TIME, "") or ""}
        data_vars: dict[str, Any] = {}
        for name, j in _columns(self.selections).items():
            values = cube[j].reshape(*shape, n_rows)
            if name == TIME:
                if self.grid is None:
                    data_vars[TIME] = ([*dims, POINT], values)
                continue
            data_vars[name] = ([*dims, tdim], values)
            units[name] = first.uinfo.get(name, "") or ""
        coords: dict[str, Any] = {}
        for dimension in self.scan.dimensions:
            coords[dimension.id] = dimension.labels
            for target, values in dimension.values.items():
                if target in data_vars:
                    # a selection of the same name, the variable stays
                    continue
                if isinstance(values, Quantity):
                    coords[target] = (dimension.id, np.asarray(values.magnitude))
                    units[target] = str(values.units)
                else:
                    coords[target] = (dimension.id, np.asarray(values))
                    units[target] = first.uinfo.get(target, "") or ""
        if self.grid is not None:
            coords[TIME] = self.grid
        attrs: dict[str, Any] = {
            "dims": dims,
            "units": units,
            "scan": self.scan.to_dict(),
            "integrator_settings": _settings(settings),
        }
        if on_error == "flag":
            data_vars["status"] = (dims, status.reshape(shape))
            attrs["errors"] = [
                f"{self.point_text(i)}: {message}"
                for i, message in sorted(errors)[:MAX_ERRORS]
            ]
            failed = int(status.sum())
            if failed:
                logger.warning(
                    "%s of %s points of the scan failed, their values are NaN, see "
                    "the variable 'status': %s",
                    failed,
                    n,
                    attrs["errors"][0],
                )
        return ScanResult(xr.Dataset(data_vars, coords=coords, attrs=attrs))


def _columns(selections: Sequence[str]) -> dict[str, int]:
    """Get the column of every name, a name which appears twice is its first column."""
    return {name: selections.index(name) for name in dict.fromkeys(selections)}


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Get the integrator settings as JSON types, the provenance of a result."""
    return {
        key: value.to_dict() if isinstance(value, AbsoluteTolerance) else value
        for key, value in settings.items()
    }


def _check_model(
    model: RoadrunnerSBMLModel,
    label: str,
    first: RoadrunnerSBMLModel,
    selections: Sequence[str],
) -> None:
    """Check that a model of a dimension has the selections and units of the first.

    Raises:
        ValueError: if roadrunner has no selection of a name in the model, or
            the model has another unit of the time or of a selection.
    """
    try:
        model.r_loaded.timeCourseSelections = list(selections)
    except RuntimeError as err:
        raise ValueError(
            f"The model '{label}' has not every selection of the first model of "
            f"the dimension: {err}"
        ) from err
    for name in dict.fromkeys(selections):
        unit = model.uinfo.get(name, "") or ""
        expected = first.uinfo.get(name, "") or ""
        if unit != expected:
            raise ValueError(
                f"The models of the dimension have different units of '{name}': "
                f"'{expected}' and '{unit}' of the model '{label}'."
            )


def _where(scan: Scan, s: int, label: str) -> str:
    """Name a simulation and a model of a scan, for a message."""
    dimension = scan.dimension(DimensionKind.SIMULATIONS)
    simulation = "" if dimension is None else f" '{dimension.labels[s]}'"
    model = f" on the model '{label}'" if label else ""
    return f"The simulation{simulation}{model}"


def _at_times(
    scan: Scan, simulation: Simulation, model: RoadrunnerSBMLModel, plan: Plan
) -> list[float | None]:
    """Get the time of every dimension with `at` in the time unit of a model.

    Raises:
        ValueError: if a time is outside of the simulation, or two dimensions
            set one target at one time.
    """
    times: list[float | None] = []
    seen: dict[tuple[float, str], str] = {}
    for dimension in scan.dimensions:
        if dimension.at is None:
            times.append(None)
            continue
        at = model_time(simulation, dimension.at, model.symbols, model.uinfo)
        if not plan.start <= at <= plan.end:
            raise ValueError(
                f"The time {dimension.at} of the dimension '{dimension.id}' is "
                f"outside of the simulation [{plan.start}, {plan.end}] in the time "
                f"unit of the model."
            )
        for target in dimension.values:
            other = seen.get((at, target))
            if other is not None:
                raise ValueError(
                    f"The dimensions '{other}' and '{dimension.id}' set '{target}' "
                    f"at the same time {at}."
                )
            seen[(at, target)] = dimension.id
        times.append(at)
    return times


def _vectors(
    scan: Scan, model: RoadrunnerSBMLModel, label: str
) -> list[dict[str, np.ndarray]]:
    """Get the values of every dimension in the units of a model.

    Raises:
        ValueError: if a target is no target of the model or a value cannot be
            converted.
    """
    vectors: list[dict[str, np.ndarray]] = []
    for dimension in scan.dimensions:
        try:
            vectors.append(
                {
                    target: target_values(target, values, model.symbols, model.uinfo)
                    for target, values in dimension.values.items()
                }
            )
        except ValueError as err:
            on = f"the model '{label}'" if label else "the model"
            raise ValueError(
                f"The dimension '{dimension.id}' does not fit {on}: {err}"
            ) from err
    return vectors


def _grid(
    plans: Mapping[tuple[int, int], Plan],
    time: ArrayLike | Quantity | None,
    model: RoadrunnerSBMLModel,
) -> tuple[np.ndarray | None, bool]:
    """Get the grid of the result and whether the workers interpolate onto it.

    Raises:
        ValueError: if `time` is empty.
    """
    if time is not None:
        if isinstance(time, Quantity):
            unit = model.uinfo.get(TIME, "") or "dimensionless"
            values: Any = time.to(unit).magnitude
        else:
            values = time
        grid = np.asarray(values, dtype=float).ravel()
        if grid.size == 0:
            raise ValueError("The grid of times 'time' is empty.")
        return grid, True
    outputs = {plan.output_times() for plan in plans.values()}
    if len(outputs) == 1:
        (times,) = outputs
        if times is not None:
            return np.asarray(times, dtype=float), False
    return None, False


@contextmanager
def _progress(show: bool, total: int) -> Iterator[Callable[[int], None]]:
    """Show the progress of a run, yield the function which advances it."""
    if not show:
        yield lambda n: None
        return
    with Progress(
        TextColumn("scan"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("scan", total=total)
        yield lambda n: progress.advance(task, n)
```

`src/sbmlsim/simulator/__init__.py`:

```python
"""Package for simulator."""

from .simulation_serial import SimulatorSerial
from .simulator import ScanError, Simulator

__all__ = ["ScanError", "Simulator", "SimulatorSerial"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q -n 0 tests/simulator/test_simulator.py`
Expected: PASS (the failing points of `BLOWUP` print CVODE warnings to stderr).

- [ ] **Step 6: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add src/sbmlsim/simulator src/sbmlsim/model/model_roadrunner.py tests/simulator/test_simulator.py
git commit -m "Simulator runs a scan from one plan per simulation and model" -m "Simulator(n_workers, **integrator_settings).run(model, scan) compiles one plan per combination of the simulations and models of a scan, converts the values of its dimensions once per model, cuts the points into chunks and writes their arrays into a ScanResult whose coordinates are the labels and the changed values. A dimension of models replaces the model of the run, time= interpolates in the worker, on_error='flag' keeps the points which ran. simulate runs one simulation; a model sets its selections with set_selections."
```

---

### Task 8: The pooled run, its equivalence and its speed

**Files:**
- Create: `tests/simulator/test_simulator_pool.py`, `tests/simulator/test_benchmark.py`
- Modify: `pyproject.toml` (marker `benchmark`, deselected by `addopts`)
- Modify: `src/sbmlsim/simulator/simulator.py` only if a test of this task fails

**Interfaces:**
- Consumes: `Simulator`, `simulator_module.MAX_CHUNK`, `simulator_module.as_completed` (Task 7), `ModelSpec` (Task 6), `parallel` (Task 1).
- Produces: the marker `benchmark` (`pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py`).

- [ ] **Step 1: Write the tests of the pool**

Create `tests/simulator/test_simulator_pool.py`:

```python
"""A scan gives the same result serially and in the pool."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.simulator.worker import ModelSpec
from tests.simulator.models import BLOWUP, PROBE, sbml

SEL = ["time", "[A]", "[B]", "X", "k1"]


def _model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=sbml())
    model.set_selections(SEL)
    return model


def _scan() -> Scan:
    """Twelve points with the steps of the integrator, a change and a value at a time."""
    return Scan(
        Simulation(end=2, changes=[Change(1.0, {"[A]": 2.0})]),
        [
            Dimension("a", values={"b0": [0.5, 1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.4, 0.8]}, at=0.5),
        ],
    )


@pytest.mark.parametrize("n_workers", [2, 4])
@pytest.mark.parametrize("max_chunk", [1, 1000])
def test_the_result_does_not_depend_on_the_workers(
    n_workers: int, max_chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial = Simulator(n_workers=1).run(_model(), _scan())
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", max_chunk)
    pooled = Simulator(n_workers=n_workers).run(_model(), _scan())
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_a_scan_over_models_in_the_pool(tmp_path: Path) -> None:
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    scan = Scan(
        Simulation(end=1, steps=4),
        [
            Dimension("model", models={"fast": _model(), "slow": slow}),
            Dimension("d", values={"b0": [1.0, 2.0]}),
        ],
    )
    serial = Simulator(n_workers=1).run(None, scan)
    pooled = Simulator(n_workers=2).run(None, scan)
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_a_failed_point_is_flagged_in_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0, 0.2, 0.3]})]
    )
    serial = Simulator(n_workers=1).run(blowup, scan, on_error="flag")
    pooled = Simulator(n_workers=2).run(blowup, scan, on_error="flag")
    assert pooled["status"].values.tolist() == [0, 1, 0, 0]
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_a_failed_point_raises_in_the_pool() -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})])
    with pytest.raises(ScanError, match=r"rate=1, k=2\.0"):
        Simulator(n_workers=2).run(blowup, scan)


def _worker_tolerances(spec: ModelSpec) -> list[float]:
    """Get the absolute tolerances of the model a worker keeps."""
    model = parallel.worker_cache(("model", spec.key), spec.load)
    vector = model.r_loaded.getIntegrator().getAbsoluteToleranceVector()
    return [float(v) for v in vector]


def test_the_tolerances_reach_the_workers() -> None:
    tolerance = AbsoluteTolerance(amount=1e-12, concentration=1e-9, other=1e-8, ids={"A": 1e-14})
    simulator = Simulator(n_workers=2, absolute_tolerance=tolerance)
    model = simulator.load(_model())
    expected = [float(v) for v in model.r_loaded.getIntegrator().getAbsoluteToleranceVector()]
    spec = ModelSpec.of(model, simulator.integrator_settings)
    assert parallel.pool(2).submit(_worker_tolerances, spec).result() == expected


def test_every_model_of_a_dimension_gets_the_tolerances(tmp_path: Path) -> None:
    """Each model of a dimension of models has its own vector, the same in a worker."""
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    tolerance = AbsoluteTolerance(amount=1e-12, concentration=1e-9, ids={"A": 1e-14})
    simulator = Simulator(n_workers=2, absolute_tolerance=tolerance)
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"fast": _model(), "slow": slow})],
    )
    models, _ = simulator._models(None, scan)
    for model in models:
        assert model.absolute_tolerance == tolerance
        ids = model.state_ids()
        vector = [float(v) for v in model.r_loaded.getIntegrator().getAbsoluteToleranceVector()]
        spec = ModelSpec.of(model, simulator.integrator_settings)
        assert parallel.pool(2).submit(_worker_tolerances, spec).result() == vector
        assert len(ids) == len(vector)


def test_an_unknown_setting_raises_before_a_pool_starts() -> None:
    with pytest.raises(ValueError, match="no settings"):
        Simulator(n_workers=2, nope=1.0).run(_model(), _scan())
    assert parallel._POOLS == {}


def test_a_late_change_agrees_in_the_pool_and_does_not_warn(
    capfd: pytest.CaptureFixture[str],
) -> None:
    scan = Scan(
        Simulation(end=2e5, changes=[Change(1.5e5, {"[A]": 2.0})], steps=20),
        [Dimension("d", values={"k2": [0.1, 0.2, 0.4, 0.8]})],
    )
    serial = Simulator(n_workers=1).run(_model(), scan)
    assert "t + h = t" not in capfd.readouterr().err
    pooled = Simulator(n_workers=4).run(_model(), scan)
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_an_interrupted_run_stops_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(simulator_module, "as_completed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Simulator(n_workers=2).run(_model(), _scan())
    assert parallel._POOLS == {}
    monkeypatch.undo()
    serial = Simulator(n_workers=1).run(_model(), _scan())
    xr.testing.assert_identical(Simulator(n_workers=2).run(_model(), _scan()).ds, serial.ds)


def test_none_runs_a_small_scan_serially(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pool(n_workers: int) -> Any:
        raise AssertionError("a scan of 12 points started a pool")

    monkeypatch.setattr(parallel, "pool", no_pool)
    res = Simulator().run(_model(), _scan())
    assert res["[A]"].shape[:2] == (3, 4)
    assert np.isfinite(res["[A]"].values[..., 0]).all()
```

- [ ] **Step 2: Run them**

Run: `uv run pytest -q tests/simulator/test_simulator_pool.py`
Expected: PASS. A difference between the serial and the pooled result is a bug of the worker (`ModelSpec.load`, the settings, the chunks), not a tolerance to widen: find it with `pytest -n 0 --pdb`.

- [ ] **Step 3: The benchmarks**

Add to the `markers` of `[tool.pytest.ini_options]` in `pyproject.toml`:

```toml
    "benchmark: the speed of the scan core, run on demand with `-n 0 -s`",
```

and append ` and not benchmark` inside the quotes of the `-m` expression of `addopts`, i.e. `-m 'not testsuite and not sciml_testsuite and not petab_testsuite and not petab_benchmark and not benchmark'`.

Create `tests/simulator/test_benchmark.py`:

```python
"""The speed of the scan core: `pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py`."""

import time

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

pytestmark = pytest.mark.benchmark


def _model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
    model.set_selections(["time", "PX", "PY", "PZ"])
    return model


def _scan(n: int) -> Scan:
    return Scan(
        Simulation(end=100, steps=100),
        [Dimension("n", values={"n": np.linspace(1.5, 4.0, n)})],
    )


def test_a_simulation() -> None:
    simulator, model = Simulator(n_workers=1), _model()
    simulation = Simulation(end=100, steps=100)
    simulator.simulate(model, simulation)
    start = time.perf_counter()
    for _ in range(200):
        simulator.simulate(model, simulation)
    elapsed = (time.perf_counter() - start) / 200
    print(f"Simulator.simulate: {elapsed * 1e3:.3f} ms per simulation")


def test_the_serial_time_of_a_scan_of_1e3_points() -> None:
    model = _model()
    start = time.perf_counter()
    Simulator(n_workers=1).run(model, _scan(1000))
    print(f"a serial scan of 1e3 points: {time.perf_counter() - start:.2f} s")


def test_a_scan_of_1e4_points_is_faster_on_4_workers() -> None:
    model, scan = _model(), _scan(10_000)
    start = time.perf_counter()
    serial = Simulator(n_workers=1).run(model, scan)
    t1 = time.perf_counter() - start
    # the start of the workers is paid once per process
    parallel.pool(4)
    start = time.perf_counter()
    pooled = Simulator(n_workers=4).run(model, scan)
    t4 = time.perf_counter() - start
    print(f"1e4 points: {t1:.2f} s on 1 worker, {t4:.2f} s on 4, {t1 / t4:.2f}x")
    np.testing.assert_array_equal(pooled["PX"].values, serial["PX"].values)
    assert t1 / t4 >= 2.5
```

- [ ] **Step 4: Run the benchmarks once**

Run: `uv run pytest -m benchmark -n 0 -s -q tests/simulator/test_benchmark.py`
Expected: PASS; note the three printed lines, the pull request reports them (Task 12). If the speedup is below 2.5, profile the parent (`python -X importtime`, `cProfile` of `Simulator.run`): the compile step and the assembly are once per scan, a chunk is one pickle of a plan, so the pool should scale.

Run: `uv run pytest -q --co tests/simulator/test_benchmark.py | tail -1`
Expected: `3 deselected` (the default run does not collect the benchmarks).

- [ ] **Step 5: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add pyproject.toml tests/simulator/test_simulator_pool.py tests/simulator/test_benchmark.py src/sbmlsim/simulator/simulator.py
git commit -m "A scan gives the same result in the pool as serially" -m "The equivalence tests run scans of values, of a change at a time, of models and with failing points on 1, 2 and 4 workers and with chunks of 1 and 1000 points; the workers integrate with the tolerances of the parent, an unknown setting raises before a pool starts and Ctrl-C stops the pool. The benchmarks, deselected by default, report the speed of a simulation and of a serial scan and require a speedup of 2.5 for 1e4 points on 4 workers."
```

---

### Task 9: Experiments, data, plots and the fit run on `Simulator`

**Files:**
- Modify: `src/sbmlsim/experiment/experiment.py`, `src/sbmlsim/experiment/runner.py`, `src/sbmlsim/data.py`, `src/sbmlsim/plot/padding.py`, `src/sbmlsim/fit/optimization.py`
- Modify examples: `examples/demo/demo.py`, `examples/repressilator/repressilator_scans.py`, `examples/glucose/glucose.py`, `examples/glucose/experiments/dose_response.py`, `examples/initial_assignment/initial_assignment.py`, `examples/curve_types/experiment.py`, `examples/hctz_fitting/helpers.py`
- Modify tests: `tests/experiment/test_experiment_run.py`, `tests/experiment/test_model_changes_merge.py`, `tests/report/test_experiment_report.py`, `tests/plot/test_serialization_matplotlib.py`, `tests/plot/test_padding.py`
- Modify docs: `docs/experiments.md`, `docs/data.md`

**Interfaces:**
- Consumes: `Simulator`, `Simulator.run`, `RoadrunnerSBMLModel.set_selections` (Task 7), `Scan` (Task 4), `ScanResult` (Task 5).
- Produces:
  - `SimulationExperiment.simulator: Simulator | None`, `SimulationExperiment.results: dict[str, ScanResult]`, `SimulationExperiment.run(simulator: Simulator | None, ...)`, `simulations() -> Mapping[str, Simulation | Scan]`; a run writes every result as netCDF (`<sid>_<task>.nc`), no TSV.
  - `ExperimentRunner(..., simulator: Simulator | None = None, ...)`, `ExperimentRunner.set_simulator(simulator: Simulator | None)`.
  - `Data.get_data` of a TASK: `result.quantity(selection)`, the values in the layout `(*dims, time)` or `(*dims, _point)`, a coordinate (a changed target) over its dimension.
  - `first_curve(values)`: the first point of the scan dimensions, the time last.
  - `OptimizationProblem` uses `Simulator(n_workers=1, ...)`.

- [ ] **Step 1: Write the failing tests**

`tests/plot/test_padding.py`, replace `test_first_curve_of_a_scan` by:

```python
def test_first_curve_of_a_scan() -> None:
    """A scan has the time last, the first point of its dimensions is drawn."""
    np.testing.assert_array_equal(first_curve(np.array([[1, 2], [3, 4]])), [1, 2])
    np.testing.assert_array_equal(first_curve(np.arange(12).reshape(2, 2, 3)), [0, 1, 2])
    np.testing.assert_array_equal(first_curve(np.array([1, 2])), [1, 2])
    assert first_curve(None) is None
```

`tests/experiment/test_model_changes_merge.py`: import `from sbmlsim.simulation import Change, Dimension, Scan, Simulation` and `from sbmlsim.simulator import Simulator`; in the experiment the annotation is `dict[str, Simulation | Scan]` and the scan is `Scan(Simulation(end=1, steps=2), [Dimension("d", values={"b0": np.array([1.0, 4.0])})])`; run with `exp.run(Simulator(), reduced_selections=False)`; replace `xres` by `res` and the last two assertions by (the scan has the time last):

```python
    res = exp.results["t_scan"]
    np.testing.assert_allclose(res["[B]"].values[:, 0], [1.0, 4.0])
    np.testing.assert_allclose(res["[A]"].values[:, 0], [2.0, 2.0])
```

`tests/experiment/test_experiment_run.py`: import `from sbmlsim.simulator import Simulator` instead of `SimulatorSerial`; `simulator=Simulator()` in `_runner` and `_runner_of`; `experiment.results["task"].xds` becomes `experiment.results["task"].ds` (twice). Add:

```python
def test_the_results_are_written_as_netcdf(tmp_path: Path) -> None:
    """A run which saves its results writes every task as netCDF, no TSV."""
    runner = _runner(FitMappingExperiment)
    experiment = runner.experiments["FitMappingExperiment"]
    experiment.run(runner.simulator, output_path=tmp_path, save_results=True)
    path = tmp_path / "FitMappingExperiment_task.nc"
    assert ScanResult.from_netcdf(path)["[X]"].size > 0
    assert not list(tmp_path.glob("FitMappingExperiment_task.tsv"))
```

with `from sbmlsim.result import ScanResult` (`SimulationExperiment.run` saves the results into `output_path` itself).

`tests/report/test_experiment_report.py` and `tests/plot/test_serialization_matplotlib.py`: import `Simulator` from `sbmlsim.simulator` and pass `simulator=Simulator()` where they pass `SimulatorSerial(...)`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q -n 0 tests/plot/test_padding.py tests/experiment`
Expected: FAIL (`first_curve` takes the first column, `SimulationExperiment` calls `set_model` on a `Simulator`).

- [ ] **Step 3: The experiment runs its tasks with `Simulator.run`**

In `src/sbmlsim/experiment/experiment.py`:

- imports: `from sbmlsim.result import ScanResult` (instead of `XResult`), `from sbmlsim.simulation import Scan, Simulation` (instead of `ScanSim`), `from sbmlsim.simulator import Simulator` (instead of `SimulatorSerial`);
- `self.simulator: Simulator | None = None`, `self._simulations: dict[str, Simulation | Scan] = {}`, `self._results: dict[str, ScanResult] = {}`, `def simulations(self) -> Mapping[str, Simulation | Scan]:`, `def results(self) -> dict[str, ScanResult]:`;
- in `_check_types`: `if not isinstance(sim, Simulation | Scan):` and the message "simulations must be of type Simulation or Scan, ...";
- `run(self, simulator: Simulator | None, output_path: ...)`;
- `save_results`: delete the line `result.to_tsv(results_path / f"{self.sid}_{rkey}.tsv")` and say in the docstring "Save the result of every task as netCDF, see `ScanResult.to_netcdf`.";
- delete the function `_with_model_changes` (the simulator applies the changes of a model, see `Simulator.compile`);
- replace `_run_tasks` by:

```python
    @timeit
    def _run_tasks(
        self, simulator: Simulator | None, reduced_selections: bool = True
    ) -> None:
        """Run the tasks of the experiment, the tasks of a model one after another.

        The selections of a model are the variables its data refers to, every
        variable of the model without `reduced_selections`. The changes of a
        model are defaults of the pre-initialization changes of every
        simulation of it, see `Simulator.compile`.

        Raises:
            ValueError: without a simulator.
        """
        if simulator is None:
            raise ValueError(
                f"The experiment '{self.sid}' has no simulator: run it with a "
                f"Simulator or through an ExperimentRunner with one."
            )
        if self._results is None:
            self._results = {}
        model_tasks: dict[str, list[str]] = defaultdict(list)
        for task_key, task in self._tasks.items():
            model_tasks[task.model_id].append(task_key)
        for model_id, task_keys in model_tasks.items():
            model = self._models[model_id]
            model.set_selections(
                sorted(self._selections_of_model(model_id))
                if reduced_selections
                else None
            )
            for task_key in task_keys:
                task = self._tasks[task_key]
                self._results[task_key] = simulator.run(
                    model, self._simulations[task.simulation_id]
                )
```

Remove the imports which are no longer used (`Any` if only `_with_model_changes` used it, `AbstractModel` if only the old `_run_tasks` used it).

Run: `rg -n "ScanSim|XResult|SimulatorSerial|to_tsv|_with_model_changes|set_timecourse_selections" src/sbmlsim/experiment`
Expected: no output.

In `src/sbmlsim/experiment/runner.py`: `from sbmlsim.simulator import Simulator`; `simulator: Simulator | None = None` in `__init__`, `self.simulator: Simulator | None = None`, `def set_simulator(self, simulator: Simulator | None) -> None:` and `self.simulator = simulator` in its body (drop the annotation on the assignment); in the module function `run_experiments`, `simulator = Simulator()`.

- [ ] **Step 4: A `Data` of a task is the variable of the result**

In `src/sbmlsim/data.py` replace `from sbmlsim.result import XResult` by `from sbmlsim.result import ScanResult` and the TASK branch by:

```python
        elif self.dtype == Data.Types.TASK:
            result = experiment.results[self.task_id]
            if not isinstance(result, ScanResult):
                raise ValueError(
                    f"The result of the task '{self.task_id}' is no ScanResult: "
                    f"{type(result)}."
                )
            if self.selection not in result:
                raise KeyError(
                    f"'{self.selection}' is not in the result of the task "
                    f"'{self.task_id}', its variables are {result.variables}: add "
                    f"it to the selections of the experiment."
                )
            # the values in the layout of the result, the time last
            x = result.quantity(self.selection)
            self.unit = result.units.get(self.selection, "")
```

Keep the code after the branch (the conversion `to_units` and the return) as it is.

- [ ] **Step 5: A figure draws the first point of a scan**

In `src/sbmlsim/plot/padding.py` replace the module docstring and `first_curve` by:

```python
"""The curves of a result of a scan, without the padding of ragged results.

The values of a result of a scan have the dimensions of the scan first and
the time last, see `sbmlsim.result.scan`; in the ragged layout every
simulation keeps its own time points and one with fewer points is padded with
`NaN`. A figure draws the first point of a scan, and the points whose x is
`NaN` are the padding; a `NaN` of y is a gap of the data and is kept.
"""
```

```python
def first_curve(values: np.ndarray | None) -> np.ndarray | None:
    """Get the values of the first point of a scan.

    Args:
        values: the values, a dimension per dimension of the scan and the
            time last.

    Returns:
        The values of the first point over the time, `None` without values.
    """
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim <= 1:
        return array
    return array.reshape(-1, array.shape[-1])[0]
```

- [ ] **Step 6: The fit hands its settings to a `Simulator`**

In `src/sbmlsim/fit/optimization.py` replace `from sbmlsim.simulator import SimulatorSerial` by `from sbmlsim.simulator import Simulator`, every annotation `SimulatorSerial` by `Simulator` (`set_simulator`, `_simulate_groups`, `_simulator_and_quantities`), and in `initialize`:

```python
        simulator = Simulator(
            n_workers=1,
            absolute_tolerance=settings.absolute_tolerance,
            relative_tolerance=settings.relative_tolerance,
            variable_step_size=settings.variable_step_size,
            initial_time_step=settings.initial_time_step,
        )
```

Keep the lines after it (`self.set_simulator(simulator)`, the loop which sets `simulator.integrator_settings` on every model, `self._compile_plans()`). `n_workers=1`: the fit runs its own plans in its own pool, the experiments it initializes run serially, also inside a worker of the fit.

Run: `rg -n "SimulatorSerial" src/sbmlsim/fit`
Expected: no output.

- [ ] **Step 7: The examples of experiments**

- `examples/glucose/glucose.py`, `examples/initial_assignment/initial_assignment.py`, `examples/curve_types/experiment.py`: import `Simulator` from `sbmlsim.simulator` instead of `SimulatorSerial` (from `sbmlsim.simulator` or `sbmlsim.simulator.simulation_serial`) and pass `simulator=Simulator()`.
- `examples/hctz_fitting/helpers.py`: `from sbmlsim.simulator import Simulator`, `simulator = Simulator()` (the model is the one of the experiments, the runner loads it).
- `examples/repressilator/repressilator_scans.py`: import `Dimension, Scan` instead of `ScanSim`, `Simulator` instead of `SimulatorSerial`; `simulations(self) -> dict[str, Simulation | Scan]`; `ScanSim(` becomes `Scan(`; `simulator=Simulator()`.
- `examples/demo/demo.py`: the same imports; `simulations(self) -> dict[str, Simulation | Scan]`; `"scan_init": Scan(`; `simulator=Simulator()`. `ModelSensitivity.create_difference_dimension` returns a `Dimension` already (Task 4).
- `examples/glucose/experiments/dose_response.py`: import `Scan` instead of `ScanSim` and `ScanResult` instead of `XResult`; the glucose is a coordinate of the result, so it is no selection:

```python
#: selections of the scan: the hormones, which are assignment rules of the
#: glucose; the glucose of the scan is a coordinate of the result
SELECTIONS = ["glu", "epi", "ins", "gamma"]
```

`simulations(self) -> dict[str, Scan]` with `glc_scan = Scan(`, and in `figures_mpl` replace the block from the comment "# the hormones are assignment rules ..." to the end of `DataSet.from_df(...)` by:

```python
        # the hormones are assignment rules of the glucose, the first time point
        # of every simulation of the scan is the dose response; the glucose of
        # the scan is the coordinate of the dimension
        res: ScanResult = self.results["task_glc_scan"]
        initial = res.ds.isel(time=0)
        columns = ["[glc_ext]", *SELECTIONS]
        dset = DataSet.from_df(
            pd.DataFrame({sid: np.asarray(initial[sid].values) for sid in columns}),
            udict={sid: res.units[sid] for sid in columns},
            ureg=self.ureg,
        )
```

Run: `rg -n "SimulatorSerial|ScanSim|XResult|\.xds\b" examples/demo examples/repressilator/repressilator_scans.py examples/glucose examples/initial_assignment examples/curve_types examples/hctz_fitting`
Expected: no output.

- [ ] **Step 8: The documentation of experiments and data**

`docs/experiments.md`: in the list item of the simulations write "**simulations** are `Simulation` or `Scan` objects, see [Simulations](simulation.md) and [Parameter scans](scans.md); the changes of a model are changes before the initialization of every simulation of it, unless the simulation sets the target itself."; in the code block of "Running an experiment" use `from sbmlsim.simulator import Simulator` and `simulator=Simulator()`; the sentence "The `results` of an experiment are the `XResult` of every task:" becomes "The `results` of an experiment are the `ScanResult` of every task, written as netCDF with `save_results=True`:" and its block:

```python
experiment = results[0].experiment
res = experiment.results["task_tc"]
print(res["[X]"].values[-3:])
```

`docs/data.md`: `from sbmlsim.simulator import Simulator` and `simulator=Simulator()` in the block of the runner.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest -q tests/experiment tests/plot tests/report tests/fit tests/test_data.py tests/docs tests/examples`
Expected: PASS.

- [ ] **Step 10: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add src/sbmlsim/experiment src/sbmlsim/data.py src/sbmlsim/plot/padding.py src/sbmlsim/fit/optimization.py examples tests docs/experiments.md docs/data.md
git commit -m "Simulation experiments run their tasks with Simulator and keep ScanResults" -m "A task pairs a model with a Simulation or a Scan; the experiment sets the selections of the model and calls Simulator.run, which applies the changes of the model, and writes the results as netCDF. A Data of a task is the variable of the result with the time last, a figure draws the first point of a scan, the dose response of the glucose example reads the glucose from the coordinate of its dimension. The fit hands its integrator settings to a Simulator."
```

---

### Task 10: The test suite, the sensitivity scans, the scan examples and the page of scans

**Files:**
- Modify: `src/sbmlsim/testsuite/runner.py`, `src/sbmlsim/testsuite/submission.py`, `src/sbmlsim/simulation/sensitivity.py`
- Modify examples: `examples/scan.py`, `examples/model_sensitivity.py`, `examples/units.py`, `examples/timecourse.py`, `examples/comparison/diff_example.py`
- Modify tests: `tests/simulation/test_scan.py`, `tests/simulation/test_simulation.py`, `tests/test_sensitivity.py`, `tests/fit/test_observable_model.py`, `tests/result/test_timecourse.py`
- Create: `tests/simulator/test_scan_regression.py`
- Rewrite: `docs/scans.md`

**Interfaces:**
- Consumes: everything of Tasks 1 to 9; `tests/data/scan_regression.json` (Task 0).
- Produces:
  - `simulate_case(case: SemanticCase, simulator: Simulator, model: RoadrunnerSBMLModel) -> TimecourseResult` in `sbmlsim.testsuite.runner`.
  - `ModelSensitivity.difference_sensitivity_scan(...) -> Scan`, `ModelSensitivity.distribution_sensitivity_scan(...) -> Scan`; `create_sampling_dimension` and `create_difference_dimension` return a `Dimension` (Task 4).
  - `examples.scan.run_scan0d/run_scan1d/run_scan2d/run_scan1d_distribution() -> ScanResult`.

- [ ] **Step 1: The test suite simulates with `Simulator`**

In `src/sbmlsim/testsuite/runner.py` import `from sbmlsim.simulator import Simulator` and `from sbmlsim.model import RoadrunnerSBMLModel` (next to `AbstractModel`) instead of `SimulatorSerial`; replace `simulate_case` by:

```python
def simulate_case(
    case: SemanticCase, simulator: Simulator, model: RoadrunnerSBMLModel
) -> TimecourseResult:
    """Simulate a case with the selections and the output of its settings.

    Args:
        case: the case.
        simulator: the simulator with the tolerances of the test suite.
        model: the model of the case, loaded by the simulator.

    Returns:
        The values of the selections of the case at its output times.
    """
    model.set_selections(case.selections)
    simulation = Simulation(
        start=case.start, end=case.start + case.duration, steps=case.steps
    )
    return simulator.simulate(model, simulation)
```

and in `run_case` the simulator and the model:

```python
    simulator = Simulator(
        n_workers=1,
        absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
        relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
        variable_step_size=False,
    )
    try:
        model = simulator.load(AbstractModel(source=case.model_path))
    except Exception as err:
        return result(CaseStatus.NOT_READ, str(err).strip().splitlines()[0][:300])
    try:
        observed = simulate_case(case, simulator, model)
```

Keep the rest of `run_case` (the messages and the comparison). In `src/sbmlsim/testsuite/submission.py` the same in `case_csv`:

```python
    simulator = Simulator(
        n_workers=1,
        absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
        relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
        variable_step_size=False,
    )
    try:
        model = simulator.load(AbstractModel(source=case.model_path))
        observed = simulate_case(case, simulator, model)
```

Run: `uv run pytest -q tests/testsuite && uv run pytest -q -m testsuite -n auto tests/testsuite 2>&1 | tail -3`
Expected: PASS; the second command runs the semantic cases if the cache of the suite exists (`~/.cache/sbmlsim/test-suite/`, else `tox r -e testsuite` downloads it) and has no regression against `tests/data/testsuite_baseline.json`.

- [ ] **Step 2: The sensitivity scans are `Scan`s**

In `src/sbmlsim/simulation/sensitivity.py` import `from sbmlsim.simulation.scan import Dimension, Scan`; `difference_sensitivity_scan` and `distribution_sensitivity_scan` return `Scan` (annotation, docstring) with `return Scan(simulation=simulation, dimensions=[dim])`. Their behavior is otherwise unchanged until sub-project 3.

`tests/test_sensitivity.py`, `test_difference_scan_of_a_simulation`:

```python
def test_difference_scan_of_a_simulation() -> None:
    """The reference values are the ones of the model with the simulation's changes."""
    from sbmlsim.simulation import Simulation
    from sbmlsim.simulator import Simulator

    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulation = Simulation(end=10, steps=10, preinit_changes={"n": 3.0})
    scan = ModelSensitivity.difference_sensitivity_scan(
        model=model, simulation=simulation, difference=0.1
    )
    assert scan.simulation is simulation
    values = scan.dimensions[0].values["n"].magnitude
    assert any(v == pytest.approx(3.0 * 1.1) for v in values)
    model.set_selections(["time", "PX"])
    res = Simulator(n_workers=1).run(model, scan)
    # the changed parameters are coordinates of the dimension
    assert res["n"].values.max() == pytest.approx(3.3)
    assert res["PX"].dims == ("dim_sens", "time")
```

- [ ] **Step 3: The scan examples**

`examples/scan.py`: import `ScanResult` from `sbmlsim.result`, `Dimension, Scan` from `sbmlsim.simulation`, `Simulator` from `sbmlsim.simulator`; every `run_*` returns a `ScanResult` and is:

```python
def run_scan0d() -> ScanResult:
    """Perform a parameter 0D scan, i.e., simple simulation."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    return Simulator().run(model, Scan(_simulation()))


def run_scan1d() -> ScanResult:
    """Perform a 1D parameter scan.

    Scanning a single parameter.
    """
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    scan1d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": np.linspace(start=2, stop=10, num=8)}),
        ],
    )
    return Simulator().run(model, scan1d)


def run_scan2d() -> ScanResult:
    """Perform a 2D parameter scan."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    scan2d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": np.linspace(start=2, stop=10, num=8)}),
            Dimension("dim2", values={"Y": np.logspace(start=2, stop=2.5, num=4)}),
        ],
    )
    return Simulator().run(model, scan2d)


def run_scan1d_distribution() -> ScanResult:
    """Perform a parameter scan by sampling from a distribution."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    rng = np.random.default_rng(seed=1234)
    scan1d = Scan(
        simulation=_simulation(),
        dimensions=[
            Dimension("dim1", values={"n": rng.normal(loc=5.0, scale=0.2, size=50)}),
        ],
    )
    return Simulator().run(model, scan1d)
```

In its `__main__` block: `print(res.ds)` instead of `print(xres.xds)`, `res.ds.sizes["dim1"]` instead of `xres.sizes["dim1"]`, rename `xres` to `res`; `res["time"]` is the grid of the scan (`steps=100`), so the plots stay as they are.

`examples/model_sensitivity.py`:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Simulation
from sbmlsim.simulation.sensitivity import ModelSensitivity
from sbmlsim.simulator import Simulator


def plot_results(res: ScanResult, filename: str) -> None:
    """Plot the mean and the range of the simulations of a scan."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = (ax1, ax2, ax3, ax4)

    summary = res.summary(statistics=["mean", "min", "max"])
    times = summary["time"].values
    ax: plt.Axes
    for ax in (ax1, ax3):
        for sid, color in [
            ("[X]", "tab:blue"),
            ("[Y]", "tab:red"),
            ("[Z]", "tab:green"),
        ]:
            # range of the simulations
            ax.fill_between(
                times,
                summary[sid].sel(statistic="min").values,
                summary[sid].sel(statistic="max").values,
                color=color,
                alpha=0.3,
            )
            # mean line
            ax.plot(times, summary[sid].sel(statistic="mean").values, color=color, label=sid)

    for ax in (ax2, ax4):
        ax.plot(
            summary["[X]"].sel(statistic="mean").values,
            summary["[Y]"].sel(statistic="mean").values,
            color="black",
            label="Y~X",
        )
```

Keep the rest of `plot_results` (the scales, the labels, saving the figure); `run_sensitivity` becomes:

```python
def run_sensitivity() -> None:
    """Parameter sensitivity simulations."""
    simulator = Simulator()
    model = simulator.load(REPRESSILATOR_SBML)
    model.set_selections(["time", "[X]", "[Y]", "[Z]"])

    # parameter sensitivity
    tcsim = Simulation(end=200, steps=2000)

    distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
        model=model, simulation=tcsim, cv=0.03, size=50
    )
    res_distrib_scan = simulator.run(model, distrib_scan)

    diff_scan = ModelSensitivity.difference_sensitivity_scan(
        model=model, simulation=tcsim, difference=0.1
    )
    res_diff_scan = simulator.run(model, diff_scan)

    # create figures
    plot_results(res_distrib_scan, "model_sensitivity_distribution.png")
    plot_results(res_diff_scan, "model_sensitivity_difference.png")
```

`examples/units.py`: import `Dimension, Scan, Simulation` and `Simulator`, drop `XResult` and `UnitsInformation`; the scan is `tc_scan = Scan(simulation=Simulation(...), dimensions=[Dimension("dim1", values={"[e__A]": Q(np.linspace(5, 15, num=20), "mM")})])` with the simulation unchanged; `res = Simulator().run(DEMO_SBML, tc_scan)`, `console.log(res)`; the plot reads the units of the result:

```python
        for key in ["[e__A]", "[e__B]", "[e__C]", "[c__A]", "[c__B]", "[c__C]"]:
            ax.plot(
                res.quantity("time").to(xunit).m,
                res.quantity(key).to(yunit).m.T,
                label=f"{key} [{yunit}]",
            )
```

(the values are `(dim1, time)`, `.T` draws one line per point).

`examples/timecourse.py`: `from sbmlsim.result import ScanResult`, `from sbmlsim.simulator import Simulator`; `simulator = Simulator()`, `model = simulator.load(REPRESSILATOR_SBML)` and `xrN: ScanResult = simulator.run(model, simulation)` for the three simulations (rename to `res1`, `res2`, `res3`).

`examples/comparison/diff_example.py`, `simulate_examples`:

```python
    simulator = Simulator(absolute_tolerance=1e-16, relative_tolerance=1e-13)
    model = simulator.load(REPRESSILATOR_SBML)

    dfs: dict[str, pd.DataFrame] = {}
    for key, json_path in sorted(get_files_by_extension(DIFF_DIR).items()):
        simulation = Simulation.from_json(json_path)
        res = simulator.run(model, simulation)
        df = res.ds.to_dataframe().reset_index()
        dfs[key] = df.drop(columns=["_point"], errors="ignore")
    return dfs
```

with `from sbmlsim.simulator import Simulator`.

`tests/simulation/test_scan.py` keeps running the examples and checks their results (the examples select every entity of the model, so the scanned `n` is a variable, see the decisions above):

```python
"""Test scans."""

import numpy as np

from examples import scan as example_scan


def test_scan0d() -> None:
    res = example_scan.run_scan0d()
    assert res.dims == ()
    assert res["PX"].dims == ("time",)


def test_scan1d() -> None:
    res = example_scan.run_scan1d()
    assert res["PX"].dims == ("dim1", "time")
    np.testing.assert_allclose(res["n"].values[:, 0], np.linspace(2, 10, 8))


def test_scan2d() -> None:
    res = example_scan.run_scan2d()
    assert res["PX"].shape == (8, 4, 101)


def test_scan1d_distribution() -> None:
    """The sample has a seed, the scan is the same every time."""
    res = example_scan.run_scan1d_distribution()
    assert res.ds.sizes["dim1"] == 50
    again = example_scan.run_scan1d_distribution()
    np.testing.assert_array_equal(res["n"].values, again["n"].values)
```

- [ ] **Step 4: The remaining tests of the old simulator**

Replace `tests/simulation/test_simulation.py` by:

```python
"""Test simulations of the repressilator."""

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator import Simulator

SIMULATOR = Simulator(n_workers=1)


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    """Get the repressilator."""
    return RoadrunnerSBMLModel(REPRESSILATOR_SBML)


def test_simulation(model: RoadrunnerSBMLModel) -> None:
    """A simulation with a grid has its points, a change before the start is there."""
    res = SIMULATOR.run(model, Simulation(end=100, steps=100))
    assert len(res["time"]) == 101

    res = SIMULATOR.run(model, Simulation(end=100, steps=100, preinit_changes={"PX": 10.0}))
    assert res["time"].values[-1] == 100.0
    assert res["[PX]"].values[0] == 10.0

    res = SIMULATOR.run(model, Simulation(end=100, steps=100, preinit_changes={"[X]": 10.0}))
    assert res["[X]"].values[0] == 10.0


def test_simulation_with_the_steps_of_the_integrator(model: RoadrunnerSBMLModel) -> None:
    """Without times or steps the output are the steps of the integrator."""
    res = SIMULATOR.run(model, Simulation(end=100))
    assert res.ragged
    time = res["time"].values
    assert time[0] == 0.0
    assert time[-1] == pytest.approx(100.0)
    assert np.all(np.diff(time) > 0)


def test_changes_at_times(model: RoadrunnerSBMLModel) -> None:
    """A change at several times sets its value at each of them."""
    res = SIMULATOR.run(
        model,
        Simulation(
            end=150,
            changes=[Change([0, 50, 100], {"X": 10})],
            times=[0, 50, 100, 150],
        ),
    )
    assert res["time"].values[-1] == 150.0
    assert res["X"].values[:3].tolist() == [10.0, 10.0, 10.0]


def test_presimulation_to_steady_state(model: RoadrunnerSBMLModel) -> None:
    """A steady state presimulation starts the simulation where nothing changes."""
    # the oscillation of the repressilator is damped for a small `n`
    res = SIMULATOR.run(
        model,
        Simulation(
            end=100,
            preinit_changes={"n": 1.0},
            presimulation=SteadyState(),
            times=[0, 100],
        ),
    )
    assert res["[X]"].values[0] == pytest.approx(res["[X]"].values[-1], rel=1e-4)
```

`tests/fit/test_observable_model.py`, `test_fit_evaluates_the_observable_model`:

```python
    simulator = Simulator(n_workers=1)
    model = simulator.load(MODEL_PATH["path"])
    model.set_selections(["time", "[A]"])
    result = simulator.simulate(
        model, Simulation(end=2, times=[0.0, 1.5, 2.0], preinit_changes={"f": 3.0})
    )
```

`tests/result/test_timecourse.py`: delete `test_scan_places_every_simulation` (covered by `tests/simulator/test_simulator.py`) and rewrite `test_simulator_returns_arrays` with `simulator = Simulator(n_workers=1)`, `model = simulator.load(REPRESSILATOR_SBML)`, `model.set_selections(["time", "[X]", "Y"])` and `result = simulator.simulate(model, simulation)`. The three `from_timecourses` tests stay until Task 11 deletes `XResult`.

- [ ] **Step 5: The regression of the scan examples**

Create `tests/simulator/test_scan_regression.py`:

```python
"""The scan examples give the values they gave before the scan core.

The values were recorded with the API of 0.8.5 in the layout `(*dims, time)`,
see `tests/data/scan_regression.json`.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples import scan as example_scan
from examples.glucose.experiments.dose_response import DoseResponseExperiment
from examples.repressilator.repressilator_scans import RepressilatorScanExperiment
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.result import ScanResult
from sbmlsim.simulator import Simulator

DATA = Path(__file__).parents[1] / "data" / "scan_regression.json"
RECORD: dict[str, dict[str, Any]] = json.loads(DATA.read_text(encoding="utf-8"))
EXAMPLES = Path(__file__).parents[2] / "examples"


def _compare(res: ScanResult, recorded: dict[str, Any], step: int = 1) -> None:
    for key, values in recorded.items():
        expected = np.asarray(values, dtype=float)
        if key == "time" and not res.ragged:
            actual = np.broadcast_to(res["time"].values[::step], expected.shape)
        else:
            actual = res[key].values[..., ::step]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, err_msg=key)


def _results(experiment_class: Any, base: Path, data: Path, reduced: bool, tmp_path: Path) -> dict[str, ScanResult]:
    runner = ExperimentRunner([experiment_class], simulator=Simulator(), base_path=base, data_path=data)
    runner.run_experiments(output_path=tmp_path, show_figures=False, reduced_selections=reduced)
    return next(iter(runner.experiments.values())).results


@pytest.mark.parametrize("name", ["run_scan0d", "run_scan1d", "run_scan2d"])
def test_the_scans_of_the_scan_example(name: str) -> None:
    _compare(getattr(example_scan, name)(), RECORD[f"scan.{name}"])


def test_the_scans_of_the_repressilator(tmp_path: Path) -> None:
    base = EXAMPLES / "repressilator"
    results = _results(RepressilatorScanExperiment, base, base, False, tmp_path)
    for key, recorded in RECORD.items():
        if key.startswith("repressilator_scans."):
            _compare(results[key.split(".", 1)[1]], recorded, step=200)


def test_the_dose_response_of_the_glucose(tmp_path: Path) -> None:
    glucose = EXAMPLES / "glucose"
    res = _results(DoseResponseExperiment, glucose, glucose / "data", True, tmp_path)["task_glc_scan"]
    recorded = dict(RECORD["dose_response.task_glc_scan"])
    # the glucose of the scan is the coordinate of its dimension
    glucose_values = np.asarray(recorded.pop("[glc_ext]"), dtype=float)
    assert "[glc_ext]" in res.ds.coords
    np.testing.assert_allclose(res["[glc_ext]"].values, glucose_values[:, 0], rtol=1e-9)
    _compare(res, recorded)
```

Run: `uv run pytest -q -n 0 tests/simulator/test_scan_regression.py`
Expected: PASS. A difference is a regression of the scan core: find the point and the variable from the message (`err_msg` names the variable) and compare the plan of the point with the plan `Simulation.with_values` compiles for it; do not loosen the tolerances.

- [ ] **Step 6: The page of scans**

Replace `docs/scans.md` by:

````markdown
# Parameter scans

A scan runs a simulation for every point of its dimensions. A `Scan` combines a `Simulation` with `Dimension` objects; `Simulator.run` runs it serially or in a pool of processes and answers with a `ScanResult`, a labeled array whose coordinates are the labels of the dimensions and the values they change.

## A one dimensional scan

A `Dimension` maps targets to arrays of values. The arrays of a dimension have one length and are coupled: point `k` sets the `k`-th value of every target. The values are numbers in the unit of their target in the model or quantities.

```python
import numpy as np

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
model.set_selections(["time", "PX", "PY", "PZ"])

scan = Scan(
    simulation=Simulation(end=100, steps=100),
    dimensions=[Dimension("dim_n", values={"n": np.linspace(2, 4, num=5)})],
)
res = simulator.run(model, scan)
print(res)
print(res["PX"].dims, res["PX"].shape)
print(res["n"].values)
```

The result has the dimensions of the scan first and the time last, `(dim_n, time)`. A changed target is a coordinate along its dimension, here `n`, unless it is also a selection of the model: then the result keeps its timecourse as a variable. The labels of a dimension of values are `0..n-1` by default and can be given with `labels=`.

## Where the values of a scan go

A value replaces its target wherever the simulation sets it, i.e. in the `preinit_changes` and in every `Change`; a target which the simulation does not set is a change before the initialization. Scanning the dose of a multiple dosing therefore scans every dose. A dimension with `at` sets its values as a `Change` at that time instead:

```python
dosing = Simulation(end=150, changes=[Change([0, 50, 100], {"X": 10.0})], steps=150)
res = simulator.run(model, Scan(dosing, [Dimension("dim_dose", values={"X": [5.0, 20.0]})]))

scan = Scan(
    Simulation(end=200, steps=200, changes=[Change(100, {"X": 10.0})]),
    [Dimension("dim_X0", values={"X": [1.0, 50.0]}, at=0)],
)
res = simulator.run(model, scan)
print(res["PX"].sel(dim_X0=1).values[:3])
```

## Several dimensions, simulations and models

Several dimensions span their cartesian product; the points are enumerated in C order of the dimensions, the last one fastest. A sampled design, e.g. values from a distribution, is one dimension with coupled values:

```python
rng = np.random.default_rng(seed=1234)
scan = Scan(
    simulation=Simulation(end=100, steps=100),
    dimensions=[
        Dimension("sample", values={"n": rng.normal(3.0, 0.2, size=20), "Y": rng.normal(10, 1, size=20)}),
        Dimension("dim_X", values={"X": [1.0, 10.0, 100.0]}),
    ],
)
res = simulator.run(model, scan)
print(res["PX"].sizes)
```

A dimension of `simulations` gives every point its own simulation, a dimension of `models` its own model; the model of the run is then `None`:

```python
scan = Scan(
    Simulation(end=100, steps=100),
    [
        Dimension("regimen", simulations={
            "single": Simulation(end=100, steps=100, changes=[Change(0, {"X": 10.0})]),
            "multiple": Simulation(end=100, steps=100, changes=[Change([0, 50], {"X": 10.0})]),
        }),
    ],
)
res = simulator.run(model, scan)
print(res["PX"].sel(regimen="multiple").values[-1])
```

## The output and the run

With `times` or `steps` every simulation has the same output times and the result has the dimension `time`. With the steps of the integrator every simulation keeps its own time points: the result has the dimension `_point` and the variable `time`, padded with `NaN`. `time=` interpolates every timecourse onto a grid; the value at the time of a change is the value after it:

```python
res = simulator.run(model, Scan(Simulation(end=100), [Dimension("dim_n", values={"n": [2.0, 3.0]})]))
print(res.ragged, res["time"].dims)
res = simulator.run(model, Scan(Simulation(end=100), [Dimension("dim_n", values={"n": [2.0, 3.0]})]), time=np.linspace(0, 100, 11))
print(res.ragged, res["time"].values)
```

`Simulator(n_workers=...)` sets the processes: `1` runs in the calling process, a number is the size of the pool, and `None` (the default) uses every CPU for a scan of 64 points or more. The result does not depend on the number of workers. A script which runs a pool must do it behind `if __name__ == "__main__":`, because the workers import the script again. `on_error="flag"` keeps the points which ran when a point fails in the integrator: its values are `NaN` and the variable `status` is `1`.

## Working with scan results

A `ScanResult` is an `xarray.Dataset` with units, `res.ds`; `res.quantity(key)` gives the values with their unit, `res.summary(dims)` the statistics over dimensions and `res.interpolate(times)` puts a ragged result on a grid. `res.ds.to_dataframe()` is the table, `res.to_netcdf(path)` and `ScanResult.from_netcdf(path)` store the result with its units:

```python
from pathlib import Path

from sbmlsim.result import ScanResult

res = simulator.run(model, Scan(Simulation(end=100, steps=100), [Dimension("dim_n", values={"n": np.linspace(2, 4, num=5)})]))
print(res["PX"].isel(dim_n=0).values[:3])  # a single simulation

summary = res.summary("dim_n", quantiles=[0.05, 0.95])
print(summary["PX"].sel(statistic="mean").values[:3])

res.to_netcdf(Path("scan.nc"))
print(ScanResult.from_netcdf(Path("scan.nc")).units["PX"])
```

## Sensitivity scans

`ModelSensitivity` creates scans of all parameters of a model, either by relative differences or by sampling from distributions, see `sbmlsim.simulation.sensitivity`. The reference values are the ones of the model with the `preinit_changes` of the simulation:

```python
from sbmlsim.simulation.sensitivity import ModelSensitivity

simulation = Simulation(end=100, steps=100)
diff_scan = ModelSensitivity.difference_sensitivity_scan(
    model=model, simulation=simulation, difference=0.1
)
res = simulator.run(model, diff_scan)
print(res["PX"].sizes)

distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
    model=model, simulation=simulation, cv=0.05, size=10
)
res = simulator.run(model, distrib_scan)
print(res["PX"].sizes)
```

The difference scan varies every constant parameter up and down by the relative `difference` (two simulations per parameter); the distribution scan samples `size` values of every parameter from a normal distribution with the coefficient of variation `cv`. The global sensitivity methods of `sbmlsim.sensitivity` build on scans like these, see [Sensitivity analysis](sensitivity.md).
````

Check the page: `uv run pytest -q -n 0 tests/docs -k scans`. Expected: PASS. A selection of a model which is a changed target of a scan (`X`, `n` and `Y` here are not selected) would be a variable instead of a coordinate; the page selects only `PX`, `PY`, `PZ` and `time`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q tests/testsuite tests/test_sensitivity.py tests/simulation tests/simulator tests/fit/test_observable_model.py tests/result tests/docs tests/examples`
Expected: PASS.

Run: `rg -n "SimulatorSerial|ScanSim|run_simulation|run_scan|\.xds\b|XResult|set_timecourse_selections\(" src examples tests docs scripts --glob '!docs/superpowers/**'`
Expected: only `src/sbmlsim/simulator/simulation_serial.py`, `src/sbmlsim/simulator/__init__.py`, `src/sbmlsim/result/xresult.py`, `src/sbmlsim/result/__init__.py`, `src/sbmlsim/simulation/scan.py` (`ScanSim`), `src/sbmlsim/simulation/__init__.py`, `src/sbmlsim/model/model_roadrunner.py` (`set_timecourse_selections` itself), `tests/result/test_xresult.py`, `tests/result/test_timecourse.py` (the `from_timecourses` tests), `tests/simulator/test_simulator_serial.py` and the pages `docs/index.md`, `docs/simulation.md`, `docs/units.md`, `docs/models.md`, which Task 11 migrates.

- [ ] **Step 8: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

```bash
git add src/sbmlsim/testsuite src/sbmlsim/simulation/sensitivity.py examples tests docs/scans.md
git commit -m "The test suite, the sensitivity scans and the scan examples run on Simulator" -m "The SBML Test Suite simulates its cases with Simulator.simulate, ModelSensitivity returns Scans, the scan, units, timecourse, sensitivity and comparison examples use Simulator and ScanResult, and the page of scans describes Scan, Dimension, the parallel run and the result. The values of the scan examples agree with the values recorded with 0.8.5."
```

---

### Task 11: `ScanSim`, `XResult` and `SimulatorSerial` are removed, the documentation describes the core

**Files:**
- Delete: `src/sbmlsim/result/xresult.py`, `src/sbmlsim/simulator/simulation_serial.py`, `tests/result/test_xresult.py`, `tests/simulator/test_simulator_serial.py`, `docs/api/simulation.range.md`, `docs/api/result.xresult.md`, `docs/api/simulator.simulation_serial.md`
- Modify: `src/sbmlsim/simulation/scan.py` (remove `ScanSim`), `src/sbmlsim/simulation/__init__.py`, `src/sbmlsim/result/__init__.py`, `src/sbmlsim/simulator/__init__.py`, `tests/result/test_timecourse.py`
- Modify docs: `docs/index.md`, `docs/simulation.md`, `docs/units.md`, `docs/models.md`, `docs/references.md`, `docs/api/index.md`, `zensical.toml`, `CLAUDE.md`
- Create: `docs/api/parallel.md`, `docs/api/simulator.simulator.md`, `docs/api/simulator.worker.md`, `docs/api/result.scan.md`

**Interfaces:**
- Consumes: everything before.
- Produces: the public API of 0.9.0: `sbmlsim.simulation` exports `Change`, `Dimension`, `Scan`, `Simulation`, `SteadyState`; `sbmlsim.simulator` exports `ScanError`, `Simulator`; `sbmlsim.result` exports `ScanResult`, `TimecourseResult`.

- [ ] **Step 1: Remove the legacy code**

```bash
git rm src/sbmlsim/result/xresult.py src/sbmlsim/simulator/simulation_serial.py tests/result/test_xresult.py tests/simulator/test_simulator_serial.py
```

- In `src/sbmlsim/simulation/scan.py` delete the class `ScanSim` and the imports only it used (`itertools`, `Change`).
- `src/sbmlsim/simulation/__init__.py`: `from .scan import Dimension, Scan` and `"ScanSim"` out of `__all__`.
- `src/sbmlsim/result/__init__.py`: export `ScanResult` and `TimecourseResult` only.
- `src/sbmlsim/simulator/__init__.py`: `from .simulator import ScanError, Simulator` and `__all__ = ["ScanError", "Simulator"]`.
- `tests/result/test_timecourse.py`: delete `test_from_timecourses_without_scan`, `test_from_timecourses_without_time` and `test_from_timecourses_of_different_lengths` and the imports they used (`XResult`, `Dimension`, `ScanSim`).

Run: `rg -n "SimulatorSerial|ScanSim|XResult|xresult|simulation_serial|run_simulation\(|run_scan\(|dim_mean|to_mean_dataframe|indices_from_dimensions|simulation\.range|\.xds\b" src tests examples scripts docs CLAUDE.md zensical.toml --glob '!docs/superpowers/**'`
Expected: only the documentation pages and `CLAUDE.md`, which the next steps rewrite, and `zensical.toml`.

- [ ] **Step 2: The pages of the documentation**

`docs/index.md`:

- In "Background", the sentence beginning with "`sbmlsim` is the layer above the simulator" reads: "`sbmlsim` is the layer above the simulator which describes these experiments. A `Simulation` is a simulation with its changes before the initialization and at times, e.g. the doses of a dosing protocol, a `Scan` runs a simulation over dimensions of values, simulations and models, serially or in a pool of processes, and a `SimulationExperiment` collects models, datasets, simulations, tasks, data and figures into one python object which is executed and reported by an `ExperimentRunner`. Results are `ScanResult` objects, labeled N-dimensional arrays with units, so the statistics over a scan dimension or the conversion to the units of a dataset is one call."
- The feature: "- **[Parameter scans](scans.md)** - `Scan` and `Dimension` run a simulation over dimensions of values, simulations and models, serially or in a pool of processes; the result is a `ScanResult` with the changed values as coordinates."
- The quickstart: "A model is simulated with a `Simulation`, the result is a `ScanResult`:" and

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
simulation = Simulation(end=200, changes=[Change(100, {"X": 10})], steps=200)
res = simulator.run(REPRESSILATOR_SBML, simulation)
print(res["X"])
```

`docs/simulation.md` (keep the text 0.8.5 added about the tolerances, replace only what names the old API):

- "the simulator returns an `XResult`:" becomes "`Simulator.run` returns a `ScanResult`:" and the first block:

```python
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)

res = simulator.run(model, Simulation(end=100))
print(res["time"].values[:5])
print(res["[X]"].values[:5])
```

- every later `xres = simulator.run_simulation(<simulation>)` becomes `res = simulator.run(model, <simulation>)` and every `xres[` becomes `res[`;
- the section "Selections and integrator settings":

````markdown
The variables recorded in a simulation are the selections of the model. By default all species (amounts and concentrations), parameters, reactions and compartments are recorded; a smaller selection speeds up the simulation:

```python
model.set_selections(["time", "[X]", "[Y]", "[Z]"])
res = simulator.run(model, Simulation(end=10, steps=10))
print(res.variables)
```

The integrator settings of roadrunner are passed to the simulator or set afterwards; they apply to every model the simulator runs, also in the workers of a pool:

```python
simulator = Simulator(absolute_tolerance=1e-10, relative_tolerance=1e-10)
simulator.set_integrator_settings(stiff=True)
```
````

- the section "Results":

````markdown
The result of a simulation is a `ScanResult`, an `xarray.Dataset` with the units of its variables, see [Parameter scans](scans.md#working-with-scan-results). With `times` or `steps` it has the dimension `time`; with the steps of the integrator it has the dimension `_point` and the time as a variable. `simulator.simulate(model, simulation)` gives the native solution of one simulation, a `TimecourseResult`. A result is converted to pandas for further processing and stored as netCDF with its units:

```python
from pathlib import Path

df = res.ds.to_dataframe()
print(df.head())

res.to_netcdf(Path("repressilator.nc"))
```
````

`docs/units.md`:

- "Every `RoadrunnerSBMLModel` and every `SimulatorSerial` carry the units of their model as `uinfo`." becomes "Every `RoadrunnerSBMLModel` carries the units of its model as `uinfo`, and every `ScanResult` the units of its variables and coordinates as `units`."
- the block of "Changes with units":

```python
import numpy as np

from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
scan = Scan(
    simulation=Simulation(
        end=10,
        steps=100,
        preinit_changes={
            "[e__A]": Q(10, "mM"),
            "[e__B]": Q(1, "mmole/litre"),
            "[e__C]": Q(1, "mole/m**3"),
            "c__A": Q(1e-5, "mole"),
            "c__B": Q(10, "µmole"),
            "Vmax_bA": Q(300.0, "mole/min"),
        },
    ),
    dimensions=[
        Dimension("dim1", values={"[e__A]": Q(np.linspace(5, 15, num=5), "mM")}),
    ],
)
res = simulator.run(DEMO_SBML, scan)
print(res["[e__A]"].values[:, 0])
```

- "The result of a simulation knows the units of its variables, so reductions return quantities:" becomes "The result of a simulation knows the units of its variables, so its values and their statistics are quantities:" and its block:

```python
print(res.units["[e__A]"])
mean = res.summary("dim1", statistics=["mean"]).quantity("[e__A]")
print(mean.units)
print(mean.to("mole/litre").magnitude[0, :3])
```

`docs/models.md`, the block of "Loading a model" and the sentence before it:

```markdown
The simulator loads a model from a path or an SBML string, with the integrator settings of the simulator:
```

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = simulator.load(REPRESSILATOR_SBML)
print(model)
```

Replace "Behind the simulator is a `RoadrunnerSBMLModel`" by "The model is a `RoadrunnerSBMLModel`". Check the rest of the page for `simulator.model` and replace it by `model`.

`docs/references.md`: "see `sbmlsim.result.xresult`" becomes "see `sbmlsim.result.scan`".

Run: `uv run pytest -q -n 0 tests/docs`
Expected: PASS.

- [ ] **Step 3: The API pages and the navigation**

```bash
git rm docs/api/simulation.range.md docs/api/result.xresult.md docs/api/simulator.simulation_serial.md
printf '# parallel\n\n::: sbmlsim.parallel\n' > docs/api/parallel.md
printf '# simulator.simulator\n\n::: sbmlsim.simulator.simulator\n' > docs/api/simulator.simulator.md
printf '# simulator.worker\n\n::: sbmlsim.simulator.worker\n' > docs/api/simulator.worker.md
printf '# result.scan\n\n::: sbmlsim.result.scan\n' > docs/api/result.scan.md
```

`docs/api/index.md`:

- after the row of `utils`: `| [parallel](parallel.md) | the process pools of sbmlsim: the start method, the kept pools and the caches of the workers |`
- the row of `simulation.scan`: `| [simulation.scan](simulation.scan.md) | `Scan` and `Dimension`, a simulation over dimensions of values, simulations and models |`; delete the row of `simulation.range`;
- replace the row of `simulator.simulation_serial` by `| [simulator.simulator](simulator.simulator.md) | `Simulator`, simulations and scans of models, serially or in a pool of processes |` and add after it `| [simulator.worker](simulator.worker.md) | the chunks of a scan, run on one plan and one model in a worker |`;
- replace the row of `result.xresult` by `| [result.scan](result.scan.md) | `ScanResult`, the result of a scan as an xarray dataset with units |`.

`zensical.toml`: add `{ "parallel" = "api/parallel.md" },` after `{ "utils" = "api/utils.md" },`; delete `{ "range" = "api/simulation.range.md" },`; replace `{ "simulation_serial" = "api/simulator.simulation_serial.md" },` by `{ "simulator" = "api/simulator.simulator.md" },` and add `{ "worker" = "api/simulator.worker.md" },` after the entry of `formula`; replace `{ "xresult" = "api/result.xresult.md" },` by `{ "scan" = "api/result.scan.md" },`.

Run: `uv run zensical build --clean 2>&1 | tail -5`
Expected: the site builds without a warning about a missing page or reference.

- [ ] **Step 4: `CLAUDE.md`**

- Commands: `pytest tests/simulation/test_scan.py::test_scan1d  # single test` stays valid; add after `pytest -rs` the line `pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py  # the speed of the scan core, on demand`.
- Architecture, the paragraph "**`simulation/`, `simulator/`, `result/` - the simulation core.**": replace the sentences about `ScanSim`, `SimulatorSerial` and `XResult` by: "`Scan(simulation, dimensions)` (`simulation/scan.py`) runs it over `Dimension` objects, each of `values` (targets to arrays, coupled within a dimension; a value replaces its target wherever the simulation sets it, `Simulation.with_values`, `at=` makes it a `Change` at that time), `simulations` or `models`; several dimensions span their product in C order. `simulator/plan.py` compiles a simulation against a model (`ModelSymbols`, `UnitsInformation`) into a frozen, picklable `Plan` without units, and a point of a scan is that plan with other values (`Plan.with_values(values, at=None)`), never a compiled simulation; `simulator/executor.py` `execute(plan, model, selections)` runs it on roadrunner (no pint, no pandas, no xarray). `Simulator(n_workers=None, **integrator_settings)` (`simulator/simulator.py`) has `load`, `compile`, `simulate` (one `TimecourseResult`) and `run(model, scan, *, time, on_error, progress)`: it compiles one plan per combination of the simulations and models of a scan, cuts the points into chunks of one plan and runs them with `simulator/worker.py` serially or in the kept pool of `sbmlsim/parallel.py` (the same function, a worker loads a model once from its `ModelSpec`), and assembles a `ScanResult` (`result/scan.py`), an `xarray.Dataset` with the scan dimensions first and the time last, `(*dims, time)` on a common grid or `(*dims, _point)` with the native time points padded with `NaN`, the labels and the changed values as coordinates, `attrs["units"]`, `summary`, `interpolate` and netCDF. The selections of a run are the ones of the model (`RoadrunnerSBMLModel.set_selections`); the changes of a model are defaults of the pre-initialization changes (`Simulator.compile`)." Keep the sentences about `Simulation`, `simulation/sensitivity.py` (it builds the `Scan`s of `ModelSensitivity`) and `TimecourseResult`.
- Conventions, the bullet of the pools: "A pool comes from `sbmlsim.parallel`: `parallel.pool(n)` is kept per process and size, `parallel.start_pool(n)` is stopped by its caller (the fit), both use `parallel.process_context()`, never `fork`: ..." and keep the rest of the bullet (the start method, the fixture `_no_fork_of_threads`); add "a test stops the pools it started (`_no_pool_left` in `tests/conftest.py`)."
- Conventions, the bullet of libsbml and roadrunner: "`RoadrunnerSBMLModel.r` is `None` until a model is loaded; use `r_loaded` or check for `None` before using it."
- The paragraph of `experiment/`: "`Task` pairs a model and a simulation" becomes "`Task` pairs a model and a `Simulation` or a `Scan`; the results are `ScanResult`s, written as netCDF".

Run: `rg -n "SimulatorSerial|ScanSim|XResult|range\.py" CLAUDE.md`
Expected: no output.

- [ ] **Step 5: Lint, types, the whole suite and commit**

Run: `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q`
Expected: no diagnostics, every test passes.

Run: `rg -n "SimulatorSerial|ScanSim|XResult|xresult|simulation_serial|run_simulation\(|run_scan\(|dim_mean|to_mean_dataframe|indices_from_dimensions|simulation\.range|\.xds\b" src tests examples scripts docs CLAUDE.md zensical.toml --glob '!docs/superpowers/**'`
Expected: no output.

```bash
git add -A src tests docs CLAUDE.md zensical.toml
git commit -m "ScanSim, XResult and SimulatorSerial are removed" -m "Scan, Simulator and ScanResult are the one way to run many simulations. The pages of simulations, units, models and the index describe them, the API reference has the pages of parallel, simulator.simulator, simulator.worker and result.scan instead of simulation.range, simulator.simulation_serial and result.xresult, and CLAUDE.md describes the core."
```

---

### Task 12: Verification, the speed against 0.8.5 and the pull request

**Files:** none new; the pull request.

**Interfaces:**
- Consumes: the branch.
- Produces: a pull request to `develop`.

- [ ] **Step 1: The whole suite, the examples and the docs**

```bash
cd $REPO
uv run ruff check && uv run ruff format --check && uv run ty check
uv run pytest -q
uv run pytest -q -m testsuite tests/testsuite 2>&1 | tail -3
uv run zensical build --clean 2>&1 | tail -3
```
Expected: no diagnostics, every test passes, the semantic cases of the SBML Test Suite have no regression against `tests/data/testsuite_baseline.json`, the site builds.

- [ ] **Step 2: The speed against 0.8.5**

Measure `SimulatorSerial` of the base in a worktree of the commit the branch started from:

```bash
cd $REPO
git worktree add $SCRATCH/base $(git merge-base HEAD origin/develop)
cat > $SCRATCH/speed_base.py <<'EOF'
"""The speed of 0.8.5: one simulation and a serial scan of 1e3 points."""

import time

import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial

model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
simulator = SimulatorSerial(model)
simulator.set_timecourse_selections(["time", "PX", "PY", "PZ"])
simulation = Simulation(end=100, steps=100)
simulator.simulate(simulation)
start = time.perf_counter()
for _ in range(200):
    simulator.simulate(simulation)
print(f"SimulatorSerial.simulate: {(time.perf_counter() - start) / 200 * 1e3:.3f} ms")
scan = ScanSim(simulation, [Dimension("n", changes={"n": np.linspace(1.5, 4.0, 1000)})])
start = time.perf_counter()
simulator.run_scan(scan)
print(f"run_scan of 1e3 points: {time.perf_counter() - start:.2f} s")
EOF
(cd $SCRATCH/base && uv run python $SCRATCH/speed_base.py)
uv run pytest -m benchmark -n 0 -s -q tests/simulator/test_benchmark.py
git worktree remove $SCRATCH/base
```
Expected: `Simulator.simulate` is not slower than `SimulatorSerial.simulate` (within 10 %), the serial scan of 1e3 points is faster than `run_scan` (no compile per point), the speedup of 1e4 points on 4 workers is at least 2.5. Note the five numbers for the pull request.

- [ ] **Step 3: The review of the branch**

Use superpowers:requesting-code-review on the whole branch against `origin/develop`, with the spec and this plan; fix what it finds in a commit per finding, each with the whole suite green.

- [ ] **Step 4: The pull request**

```bash
git push -u origin scan-core-phase1
```

Create the pull request with `gh-axi` (never `gh`), base `develop`, title "The scan core: Scan, Simulator and ScanResult (#249, phase 1)". The body describes, without any agent attribution: the new API (`Scan`, `Dimension`, `Simulator.run`, `ScanResult`, `sbmlsim.parallel`), what was removed (`ScanSim`, `XResult`, `SimulatorSerial`, `Dimension(index=, changes=)`, `simulation/range.py`), the decisions of this plan where the spec is silent (the list "Decisions this plan takes where the spec is silent"), the regression of the scan examples against 0.8.5, the five numbers of the speed, and that the observables (`Formula`, `PK`, `Custom`, `keep`) are phase 2. Wait for the checks `tests`, `ruff`, `ty` and `docs`.
