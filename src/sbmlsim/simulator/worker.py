"""The worker of a scan: the points of a chunk on one plan and one model.

`run_chunk` is the same function serially and in a worker process: for every
point of a chunk it applies the values of the point to the plan of the chunk,
`Plan.with_values`, and runs the plan with `execute`; the native solutions of
the points which ran are stacked into arrays padded with `NaN`, the observables
of the run are evaluated on them, see `sbmlsim.simulator.observables`, and the
kept timecourses are interpolated onto the grid of times of the chunk, if it
has one. Nothing in here uses pint or xarray, and a chunk and its result are
numbers, strings, a plan and the graph of the observables, so they pickle.

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
from collections.abc import Callable, Generator, Mapping
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
from sbmlsim.simulator.observables import ObservableError, ObservableGraph
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
    """Run the points of a chunk, see `run_chunk`."""
    n = len(chunk.indices)
    status = np.zeros(n, dtype=np.int8)
    errors: list[tuple[int, str]] = []

    def fail(k: int, message: str, cause: BaseException) -> None:
        index = int(chunk.indices[k])
        if chunk.on_error == "raise":
            raise ScanPointError(index, message) from cause
        status[k] = 1
        errors.append((index, message))

    points: list[int] = []
    solutions: list[np.ndarray] = []
    plans: list[Plan] = []
    for k in range(n):
        # a definition which does not fit the plan is no failed point
        plan = chunk.plan_of(k)
        try:
            result = execute(plan, model, chunk.selections)
        except Exception as err:
            fail(k, f"{type(err).__name__}: {err}", err)
            continue
        points.append(k)
        solutions.append(result.values)
        plans.append(plan)
    time, outputs = _observe(chunk, points, solutions, plans, fail)
    return _pack(chunk, n, points, time, outputs, status, errors)


def _observe(
    chunk: Chunk,
    points: list[int],
    solutions: list[np.ndarray],
    plans: list[Plan],
    fail: Callable[[int, str, BaseException], None],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the observables on the native solutions of the points which ran.

    A point whose observable fails fails and is dropped, and the observables
    are evaluated again on the others; an observable which fails for every
    point fails them all. `points`, `solutions` and `plans` keep the points
    which are left.

    Returns:
        The time points `(n_points, n_rows)` padded with `NaN` and the kept
        outputs of the points which are left.
    """
    while points:
        rows = max(solution.shape[0] for solution in solutions)
        stacked = np.full((len(points), rows, len(chunk.selections)), np.nan)
        for r, solution in enumerate(solutions):
            stacked[r, : solution.shape[0]] = solution
        time = stacked[:, :, 0]
        columns = {
            name: stacked[:, :, j] for j, name in enumerate(chunk.selections) if j
        }
        try:
            return time, chunk.graph.evaluate(time, columns, plans)
        except ObservableError as err:
            failing = list(range(len(points))) if err.row is None else [err.row]
            for r in failing:
                fail(points[r], err.message, err)
            for r in reversed(failing):
                del points[r], solutions[r], plans[r]
    return np.empty((0, 0)), {}


def _pack(
    chunk: Chunk,
    n: int,
    points: list[int],
    time: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    status: np.ndarray,
    errors: list[tuple[int, str]],
) -> ChunkResult:
    """Write the kept outputs of the points which ran into the arrays of a chunk.

    The timecourses are interpolated onto the grid of the chunk, if it has
    one; a failed point is `NaN`, with the time of the grid.
    """
    names = chunk.graph.timecourses
    scalars = chunk.graph.scalars
    if chunk.time is not None:
        n_rows = chunk.time.size
    else:
        n_rows = time.shape[1] if points else 0
    values = np.full((n, n_rows, 1 + len(names)), np.nan)
    if chunk.time is not None:
        values[:, :, 0] = chunk.time
    table = np.full((n, len(scalars)), np.nan)
    for r, k in enumerate(points):
        rows = np.stack([time[r], *(outputs[name][r] for name in names)])
        if chunk.time is not None:
            rows = apply_weights(grid_weights(time[r], chunk.time), rows)
            rows[0] = chunk.time
        values[k] = rows.T
        for c, name in enumerate(scalars):
            table[k, c] = outputs[name][r]
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
