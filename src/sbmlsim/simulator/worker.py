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
