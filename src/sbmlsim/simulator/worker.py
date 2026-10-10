"""The worker of a scan: the points of a chunk on one plan and one model.

`run_chunk` is the same function serially and in a worker process: for every
point of a chunk it applies the values of the point to the plan of the chunk,
`Plan.with_values`, and runs the plan with `execute`; the native solutions of
the points which ran are stacked into arrays padded with `NaN`, the observables
of the run are evaluated on them, see `sbmlsim.simulator.observables`, and the
kept timecourses are interpolated onto the grid of times of the chunk, if it
has one. A run without observables needs no native solution: the kept
selections of every point are interpolated right after its simulation, so the
memory of a chunk is bounded by its grid. A point fails exactly when its own
simulation or observables fail, not because of the other points of its chunk.
Nothing in here uses pint or xarray, and a chunk and its result are numbers,
strings, a plan and the graph of the observables, so they pickle.

In a worker process the model of a chunk is loaded once from its `ModelSpec`
and kept, see `sbmlsim.parallel.worker_cache`, with the integrator of the
model of the parent and every one of its settings, those of the simulator and
those set on the model, so a worker integrates as the parent does and the
result does not depend on the number of workers.

A point which fails is reported once, by its error: roadrunner logs the error
of CVODE as well, so its log level is lowered while a chunk runs. SUNDIALS
writes its own messages, a dozen lines or more per failing point, which a
worker process silences for the models it loads, see `quiet_sundials`; a
serial run keeps them, as the model of the user is not changed.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import roadrunner

from sbmlsim import parallel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.result.scan import TIME
from sbmlsim.result.timecourse import apply_weights, grid_weights
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.observables import TIMECOURSE, ObservableError, ObservableGraph
from sbmlsim.simulator.plan import Plan

#: what a run does about a point which fails: raise its error or flag it
OnError = Literal["raise", "flag"]

#: the most messages of failed points a chunk reports
MAX_ERRORS: int = 10

#: the variables of the environment which name the files of the messages of
#: SUNDIALS, read when roadrunner creates the integrator of a model
SUNDIALS_LOGS: tuple[str, ...] = (
    "SUNLOGGER_ERROR_FILENAME",
    "SUNLOGGER_WARNING_FILENAME",
)


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
        key: identifies the model, its integrator and the settings of the
            integrator, the key of the model in the cache of a worker.
        source: the path of the model or the SBML.
        base_path: the directory a relative path is resolved against.
        parameters: the parameters added to the model, see `AbstractModel`.
        integrator: the name of the integrator, e.g. `cvode`.
        settings: every setting of the integrator by its name, the absolute
            tolerance as the `AbsoluteTolerance` of the model.
    """

    key: str
    source: str
    base_path: Path | None
    parameters: tuple[tuple[str, float], ...]
    integrator: str
    settings: tuple[tuple[str, Any], ...]

    @classmethod
    def of(cls, model: RoadrunnerSBMLModel) -> ModelSpec:
        """Get the spec of a loaded model and of the state of its integrator.

        The integrator and its settings are read from the model after the
        simulator set its own, see `integrator_state`, so a setting of the
        model which the simulator does not set, e.g. one of
        `RoadrunnerSBMLModel(settings=...)` or of `set_integrator_settings`,
        holds in a worker as well.
        """
        # the key is the content of the model and not its path, a worker
        # lives as long as its pool and the file may be rewritten meanwhile;
        # the files a comp model includes are not part of it
        if model.source.content is not None:
            source = model.source.content
            text = source
        else:
            source = str(model.source.path)
            text = Path(source).read_text(encoding="utf-8")
        parameters = tuple(
            sorted((str(k), float(v)) for k, v in (model.parameters or {}).items())
        )
        integrator, settings = integrator_state(model)
        digest = hashlib.sha256(
            repr((text, model.base_path, parameters, integrator, settings)).encode(
                "utf-8"
            )
        ).hexdigest()
        return cls(
            key=digest,
            source=source,
            base_path=model.base_path,
            parameters=parameters,
            integrator=integrator,
            settings=settings,
        )

    def load(self) -> RoadrunnerSBMLModel:
        """Load the model with the integrator and its settings."""
        model = RoadrunnerSBMLModel(
            source=self.source,
            base_path=self.base_path,
            parameters=dict(self.parameters) or None,
        )
        r = model.r_loaded
        if r.getIntegrator().getName() != self.integrator:
            r.setIntegrator(self.integrator)
        model.set_integrator_settings(**dict(self.settings))
        return model


def integrator_state(
    model: RoadrunnerSBMLModel,
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Get the name of the integrator of a model and every one of its settings.

    The absolute tolerance is the `AbsoluteTolerance` of the model, from which
    `RoadrunnerSBMLModel.set_integrator_settings` computes the tolerance of
    every state, and not the vector of the integrator.

    Args:
        model: the loaded model.

    Returns:
        The name of the integrator and its settings, sorted by name.
    """
    integrator: roadrunner.Integrator = model.r_loaded.getIntegrator()
    settings: list[tuple[str, Any]] = []
    for name in sorted(str(name) for name in integrator.getSettings()):
        if name == "absolute_tolerance":
            settings.append((name, model.absolute_tolerance))
        else:
            settings.append((name, integrator.getValue(name)))
    return str(integrator.getName()), tuple(settings)


def point_plan(
    plan: Plan,
    values: Mapping[str, np.ndarray],
    timed: Mapping[float, Mapping[str, np.ndarray]],
    k: int,
) -> Plan:
    """Get the plan of the `k`-th point of values.

    The values of the dimensions without a time are applied first, then the
    changes of the dimensions with one in time order, so a target in both has
    the value of the change at its time.

    Args:
        plan: the plan of the points.
        values: target -> value of every point.
        timed: time -> target -> value of every point.
        k: the point.

    Returns:
        The plan with the values of the point.
    """
    plan = plan.with_values({t: float(v[k]) for t, v in values.items()})
    for at in sorted(timed):
        changes = timed[at]
        plan = plan.with_values({t: float(v[k]) for t, v in changes.items()}, at=at)
    return plan


@dataclass(frozen=True)
class Chunk:
    """Points of a scan which share a plan and a model.

    Attributes:
        indices: the flat indices of the points in the scan, in C order.
        plan: the plan of the points.
        model: the index of the model of the points among the models of the
            run.
        graph: the observables of the run, which give the selections.
        values: target -> value of every point, which replaces the target
            wherever the plan sets it and is a change before the
            initialization otherwise.
        timed: time -> target -> value of every point, a change at that time.
        time: the grid of times to interpolate the kept timecourses onto,
            `None` for the time points of the simulation.
        on_error: what to do about a point which fails.
    """

    indices: np.ndarray
    plan: Plan
    model: int
    graph: ObservableGraph
    values: dict[str, np.ndarray]
    timed: dict[float, dict[str, np.ndarray]]
    time: np.ndarray | None
    on_error: OnError = "raise"

    @property
    def selections(self) -> tuple[str, ...]:
        """Get the selections of roadrunner, `time` first."""
        return (TIME, *self.graph.selections)

    def plan_of(self, k: int) -> Plan:
        """Get the plan of the `k`-th point of the chunk, see `point_plan`."""
        return point_plan(self.plan, self.values, self.timed, k)


@dataclass(frozen=True)
class ChunkResult:
    """The answer of a chunk.

    Attributes:
        indices: the flat indices of the points, those of the chunk.
        values: the kept timecourses, `(point, row, column)` with the time
            first and the timecourses of the graph, padded with `NaN`.
        scalars: the kept values per simulation, `(point, column)` with the
            scalars of the graph.
        status: `0` for a point which ran, `1` for one which failed.
        errors: the flat index and the error of the first `MAX_ERRORS`
            points which failed, in the order of the scan.
    """

    indices: np.ndarray
    values: np.ndarray
    scalars: np.ndarray
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
    with quiet_roadrunner():
        return _run_chunk(chunk, model)


def _run_chunk(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
    """Run the points of a chunk, see `run_chunk`.

    With `on_error="raise"` the first point which fails in the order of the
    scan is raised, whatever the chunking: a failed simulation stops the
    chunk, the observables of the points before it are evaluated, and a point
    among them whose observable fails comes before it.

    Without observables (a graph without nodes) the kept selections of a
    solution are its outputs, which are interpolated onto the grid right after
    the simulation: such a chunk never holds the native solutions of its
    points, which only the observables need.
    """
    n = len(chunk.indices)
    status = np.zeros(n, dtype=np.int8)
    failures: list[tuple[int, str, BaseException]] = []
    points: list[int] = []
    solutions: list[np.ndarray] = []
    plans: list[Plan] = []
    observed = bool(chunk.graph.nodes)
    columns = (
        []
        if observed
        else [chunk.selections.index(s) for s in (TIME, *chunk.graph.timecourses)]
    )
    for k in range(n):
        # a definition which does not fit the plan is no failed point
        plan = chunk.plan_of(k)
        try:
            result = execute(plan, model, chunk.selections)
        except Exception as err:
            failures.append((k, f"{type(err).__name__}: {err}", err))
            if chunk.on_error == "raise":
                break
            continue
        points.append(k)
        if observed:
            solutions.append(result.values)
            plans.append(plan)
        else:
            solutions.append(_on_grid(chunk.time, result.values[:, columns].T))
    if observed:
        points, time, outputs, failed = _observe(chunk, points, solutions, plans)
        failures.extend(failed)
        n_rows = time.shape[1]
        rows: Iterable[np.ndarray] = _timecourses(chunk, time, outputs)
        scalars = np.empty((len(points), len(chunk.graph.scalars)))
        if points:
            for c, name in enumerate(chunk.graph.scalars):
                scalars[:, c] = outputs[name]
    else:
        n_rows = max((solution.shape[1] for solution in solutions), default=0)
        rows = solutions
        scalars = np.empty((len(points), 0))
    failures.sort(key=lambda failure: failure[0])
    errors: list[tuple[int, str]] = []
    for k, message, cause in failures:
        index = int(chunk.indices[k])
        if chunk.on_error == "raise":
            raise ScanPointError(index, message) from cause
        status[k] = 1
        errors.append((index, message))
    return _pack(chunk, points, n_rows, rows, scalars, status, errors)


def _on_grid(grid: np.ndarray | None, rows: np.ndarray) -> np.ndarray:
    """Interpolate timecourses `(column, row)` with the time first onto a grid.

    Without a grid the timecourses are returned as they are.
    """
    if grid is None:
        return rows
    rows = apply_weights(grid_weights(rows[0], grid), rows)
    rows[0] = grid
    return rows


def _stack(
    selections: Sequence[str], solutions: Sequence[np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stack native solutions into the time and the columns `(point, row)`.

    The solutions are padded with `NaN` to the longest of them.
    """
    rows = max(solution.shape[0] for solution in solutions)
    stacked = np.full((len(solutions), rows, len(selections)), np.nan)
    for r, solution in enumerate(solutions):
        stacked[r, : solution.shape[0]] = solution
    columns = {name: stacked[:, :, j] for j, name in enumerate(selections) if j}
    return stacked[:, :, 0], columns


def observe(
    graph: ObservableGraph,
    selections: Sequence[str],
    solutions: Sequence[np.ndarray],
    plans: Sequence[Plan],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[tuple[int, str, BaseException]]]:
    """Evaluate the observables on the native solutions of points.

    The graph is evaluated on all solutions at once. If an observable fails
    the points are evaluated one at a time, each alone on its native solution
    without padding, so a point fails exactly when its own evaluation fails,
    whatever the other points are; the outputs of the points which succeed are
    stacked again, padded with `NaN` to the longest of them.

    Args:
        graph: the observables.
        selections: the columns of the solutions, the time first.
        solutions: the native solutions of the points.
        plans: their plans.

    Returns:
        The time points `(n_points, n_rows)` padded with `NaN`, the kept
        outputs (a timecourse `(n_points, n_rows)`, a value per simulation
        `(n_points,)`) of the points which are left, in their order, and the
        points whose observable failed (the position in `solutions`) with
        their message and error.
    """
    if not solutions:
        return np.empty((0, 0)), {}, []
    time, columns = _stack(selections, solutions)
    try:
        return time, graph.evaluate(time, columns, plans), []
    except ObservableError:
        pass
    singles: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    failed: list[tuple[int, str, BaseException]] = []
    for k, (solution, plan) in enumerate(zip(solutions, plans, strict=True)):
        single_time, single_columns = _stack(selections, [solution])
        try:
            outputs = graph.evaluate(single_time, single_columns, [plan])
        except ObservableError as err:
            failed.append((k, err.message, err))
            continue
        singles.append((single_time, outputs))
    if not singles:
        return np.empty((0, 0)), {}, failed
    n_rows = max(single_time.shape[1] for single_time, _ in singles)
    time = np.full((len(singles), n_rows), np.nan)
    stacked = {
        name: np.full(
            (len(singles), n_rows) if graph.kinds[name] is TIMECOURSE else len(singles),
            np.nan,
        )
        for name in graph.keep
    }
    for r, (single_time, outputs) in enumerate(singles):
        rows = single_time.shape[1]
        time[r, :rows] = single_time[0]
        for name, value in outputs.items():
            if value.ndim == 2:
                stacked[name][r, :rows] = value[0]
            else:
                stacked[name][r] = value[0]
    return time, stacked, failed


def _observe(
    chunk: Chunk,
    points: Sequence[int],
    solutions: Sequence[np.ndarray],
    plans: Sequence[Plan],
) -> tuple[
    list[int], np.ndarray, dict[str, np.ndarray], list[tuple[int, str, BaseException]]
]:
    """Evaluate the observables on the native solutions of the points which ran.

    See `observe`, which does the work.

    Args:
        chunk: the chunk.
        points: the positions in the chunk of the points which ran.
        solutions: their native solutions.
        plans: their plans.

    Returns:
        The positions of the points which are left, their time points
        `(n_points, n_rows)` padded with `NaN`, the kept outputs and the
        points whose observable failed with their message and error.
    """
    time, outputs, failures = observe(chunk.graph, chunk.selections, solutions, plans)
    lost = {k for k, _, _ in failures}
    kept = [p for k, p in enumerate(points) if k not in lost]
    failed = [(points[k], message, err) for k, message, err in failures]
    return kept, time, outputs, failed


def _timecourses(
    chunk: Chunk, time: np.ndarray, outputs: Mapping[str, np.ndarray]
) -> Iterator[np.ndarray]:
    """Get the time and the kept timecourses of every point, see `_pack`.

    The points are interpolated onto the grid of the chunk one at a time, as
    `_pack` writes them, so the interpolated outputs are not held twice.
    """
    names = chunk.graph.timecourses
    for r in range(time.shape[0]):
        rows = np.stack([time[r], *(outputs[name][r] for name in names)])
        yield _on_grid(chunk.time, rows)


def _pack(
    chunk: Chunk,
    points: Sequence[int],
    n_rows: int,
    rows: Iterable[np.ndarray],
    scalars: np.ndarray,
    status: np.ndarray,
    errors: list[tuple[int, str]],
) -> ChunkResult:
    """Write the outputs of the points which ran into the arrays of a chunk.

    A failed point is `NaN`, with the time of the grid if the chunk has one.

    Args:
        chunk: the chunk.
        points: the positions in the chunk of the points which ran.
        n_rows: the number of rows of their native timecourses, padded to the
            longest, which a chunk without a grid has.
        rows: the time and the kept timecourses of every point which ran,
            `(column, row)`, on the grid of the chunk if it has one.
        scalars: the kept values per simulation of every point which ran,
            `(point, column)`.
        status: `0` for a point which ran, `1` for one which failed.
        errors: the flat index and the error of every point which failed.
    """
    n = len(chunk.indices)
    if chunk.time is not None:
        n_rows = chunk.time.size
    values = np.full((n, n_rows, 1 + len(chunk.graph.timecourses)), np.nan)
    if chunk.time is not None:
        values[:, :, 0] = chunk.time
    table = np.full((n, len(chunk.graph.scalars)), np.nan)
    for k, point_rows, point_scalars in zip(points, rows, scalars, strict=True):
        values[k, : point_rows.shape[1]] = point_rows.T
        table[k] = point_scalars
    return ChunkResult(
        indices=chunk.indices,
        values=values,
        scalars=table,
        status=status,
        errors=tuple(sorted(errors)[:MAX_ERRORS]),
    )


def run_chunk_in_worker(spec: ModelSpec, chunk: Chunk) -> ChunkResult:
    """Run a chunk in a worker process, on the model the worker keeps.

    Args:
        spec: the model of the chunk and the settings of its integrator.
        chunk: the points.

    Returns:
        The values of every point, see `run_chunk`.
    """
    quiet_sundials()
    model = parallel.worker_cache(("model", spec.key), spec.load)
    return run_chunk(chunk, model)


@contextmanager
def quiet_roadrunner() -> Generator[None]:
    """Lower the log level of roadrunner to critical messages, then restore it.

    roadrunner logs the error of CVODE of a point which fails, which the run
    reports as the error of the point.
    """
    level = roadrunner.Logger.getLevel()
    roadrunner.Logger.setLevel(min(level, roadrunner.Logger.LOG_CRITICAL))
    try:
        yield
    finally:
        roadrunner.Logger.setLevel(level)


def quiet_sundials() -> None:
    """Silence the messages of SUNDIALS of the models a worker process loads.

    SUNDIALS reads the files of its messages from the environment when
    roadrunner creates the integrator of a model, so this applies to the
    models loaded after it and is no setting of a loaded model. Only a worker
    process of a pool is changed, never the process of the user, and a file
    the environment names already stays.

    The change is permanent for the worker: a kept worker of
    `sbmlsim.parallel.pool` never prints a message of SUNDIALS again, for
    every model it loads later, e.g. the cases of `testsuite.map_cases`, also
    for a scan which raises and for a point which succeeds. A failure is
    reported through its exception or the variable `status`.
    """
    if not parallel.in_worker():
        return
    for name in SUNDIALS_LOGS:
        os.environ.setdefault(name, os.devnull)
