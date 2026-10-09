"""The worker of a scan: the points of a chunk on one plan and one model.

`run_chunk` is the same function serially and in a worker process: for every
point of a chunk it applies the values of the point to the plan of the chunk,
`Plan.with_values`, runs the plan with `execute` and stacks the native
solutions into one array, padded with `NaN`; with a grid of times every
solution is interpolated onto it first. Nothing in here uses pint or xarray,
and a chunk and its result are numbers, strings and a plan, so they pickle.

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
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import roadrunner

from sbmlsim import parallel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.result.timecourse import apply_weights, grid_weights
from sbmlsim.simulator.executor import execute
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
        """Get the plan of the `k`-th point of the chunk.

        The values of the dimensions without a time are applied first, then
        the changes of the dimensions with one in time order, so a target in
        both has the value of the change at its time.
        """
        plan = self.plan.with_values({t: float(v[k]) for t, v in self.values.items()})
        for at in sorted(self.timed):
            values = self.timed[at]
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
    with quiet_roadrunner():
        return _run_chunk(chunk, model)


def _run_chunk(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
    """Run the points of a chunk, see `run_chunk`."""
    n = len(chunk.indices)
    rows: list[np.ndarray | None] = []
    status = np.zeros(n, dtype=np.int8)
    errors: list[tuple[int, str]] = []
    for k in range(n):
        index = int(chunk.indices[k])
        # a definition which does not fit the plan is no failed point
        plan = chunk.plan_of(k)
        try:
            result = execute(plan, model, chunk.selections)
        except Exception as err:
            message = f"{type(err).__name__}: {err}"
            if chunk.on_error == "raise":
                raise ScanPointError(index, message) from err
            status[k] = 1
            if len(errors) < MAX_ERRORS:
                errors.append((index, message))
            values = None
        else:
            values = result.values
            if chunk.time is not None:
                weights = grid_weights(values[:, 0], chunk.time)
                values = apply_weights(weights, values.T).T
        if chunk.time is not None:
            if values is None:
                values = np.full((chunk.time.size, len(chunk.selections)), np.nan)
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
