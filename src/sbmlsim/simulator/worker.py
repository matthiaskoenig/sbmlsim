"""The worker of a scan: the points of a chunk on one plan and one model.

`run_chunk` is the same function serially and in a worker process: for every
point of a chunk it applies the values of the point to the plan of the chunk,
`Plan.with_values`, and runs the plan with the executor of the engine of the
model, `sbmlsim.simulator.engines.execute_plan`; the native solutions of the
points which ran are stacked into arrays padded with `NaN`, the observables of
the run are evaluated on them, see `sbmlsim.simulator.observables`, and the
kept timecourses are interpolated onto the grid of times of the chunk, if it
has one. A run without observables needs no native solution: the kept
selections of every point are interpolated right after its simulation, so a
chunk of the loop holds the grid of one point at a time. A point fails exactly
when its own simulation or observables fail, not because of the other points of
its chunk. Nothing in here uses pint or xarray, and a chunk and its result are
numbers, strings, a plan and the graph of the observables, so they pickle.

A chunk of a model of the engine core runs its points as one batch of the
engine on its threads instead (`_run_chunk_core`), which is why a scan of
that engine runs in one python process and starts no pool, see
`sbmlsim.simulator.simulator`. The two paths share everything after the
simulations, `_finish`. A batch answers with the table of every one of its
points at once, so the memory of such a chunk grows with its points also
without observables, which is why they are capped at `CORE_MAX_CHUNK`, well
below the `MAX_CHUNK` of a chunk of a pool.

In a worker process the model of a chunk is loaded once from its `ModelSpec`
and kept, see `sbmlsim.parallel.worker_cache`, in the engine of the model of
the parent, with its integrator and every one of its settings, those of the
simulator and those set on the model, so a worker integrates as the parent
does and the result does not depend on the number of workers (bitwise with
roadrunner and AMICI, within the integration with COPASI, see
`sbmlsim.simulator.simulator`).

In a worker of a pool a large array of the result of a chunk is written into
shared memory instead of being pickled through the pipe of the pool, see
`SharedChunkResult` and `sbmlsim.simulator.shared`, on POSIX only.

A point which fails is reported once, by its error: roadrunner logs the error
of CVODE as well, so its log level is lowered while a chunk of a model of
roadrunner runs. A feature which the engine does not simulate, the
`NotImplementedError` of an executor, is the same for every point of a model
and is no failed point: it is raised also with `on_error="flag"`. The
messages of SUNDIALS follow the log level of roadrunner (libroadrunner 2.11).
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import roadrunner

from sbmlsim import cgroup, parallel
from sbmlsim.engine import Engine, amici_cache, import_core, using_amici_cache
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_roadrunner import JitOptimization, RoadrunnerSBMLModel
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.result.scan import MAX_ERRORS, TIME
from sbmlsim.result.timecourse import apply_weights, grid_weights
from sbmlsim.simulator import shared
from sbmlsim.simulator.engines import core_model, execute_plan
from sbmlsim.simulator.executor import output_columns, output_time
from sbmlsim.simulator.observables import TIMECOURSE, ObservableError, ObservableGraph
from sbmlsim.simulator.plan import Plan, time_tolerance
from sbmlsim.simulator.shared import SharedArray

if TYPE_CHECKING:
    from sbmlsim.model.model_core import CoreSBMLModel

#: what a run does about a point which fails: raise its error or flag it
OnError = Literal["raise", "flag"]

#: the values of a chunk from which `_pack` writes its points on threads; a
#: smaller chunk is written by the calling thread, which costs no pool
PACK_THREAD_BYTES = 16 * 1024 * 1024

#: the threads `_pack` writes the points of a chunk with at the most. The
#: copy is bound by the memory of the machine and not by the threads: on the
#: HCTZ model, a chunk of 250 points of 401 rows and 190 selections (152 MB),
#: 18.4 ms with one thread, 12.8 with four, 10.9 with six and 11.1 with eight
PACK_MAX_THREADS = 6


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

    A model of every engine has a spec, the engine core included, although a
    scan of that engine takes no pool and so loads no model in a worker, see
    `_run_chunk_core`: a spec is what a model of it round trips through, and
    a chunk of it would run in a worker as every other chunk does.

    Attributes:
        key: identifies the model, its engine, its optimization level, its
            integrator and the settings of the integrator, the key of the
            model in the cache of a worker.
        engine: the engine which simulates the model.
        source: the SBML of the model as the parent loaded it, see
            `SBMLModel.sbml`, or the path of the file
            `parallel.shared_payloads` shares it in. It is flattened and the
            path is absolute, so nothing is resolved against a directory.
        parameters: the parameters added to the model, see `AbstractModel`.
        jit_optimization: the optimization level the model is compiled with,
            see `RoadrunnerSBMLModel.jit_optimization`: a worker compiles it
            at the level of the parent, whatever its environment says;
            `None` for a model of COPASI or AMICI, which roadrunner does not
            compile.
        integrator: the name of the integrator, e.g. `cvode` or `lsoda`.
        settings: every setting of the integrator by its name, the absolute
            tolerance as the `AbsoluteTolerance` of the model.
        cache: the directory of the compiled models a model of AMICI was
            loaded from, which a worker loads it from as well, see
            `sbmlsim.model.model_amici.using_cache`; `None` for the other
            engines.
    """

    key: str
    engine: Engine
    source: str
    parameters: tuple[tuple[str, float], ...]
    jit_optimization: JitOptimization | None
    integrator: str
    settings: tuple[tuple[str, Any], ...]
    cache: str | None = None

    @classmethod
    def of(cls, model: SBMLModel) -> ModelSpec:
        """Get the spec of a loaded model and of the state of its integrator.

        The integrator and its settings are read from the model after the
        simulator set its own, see `SBMLModel.integrator_state`, so a setting
        of the model which the simulator does not set, e.g. one of
        `RoadrunnerSBMLModel(settings=...)` or of `set_integrator_settings`,
        holds in a worker as well.

        Args:
            model: the loaded model, of any engine.
        """
        # the key is the SBML the parent loaded and not the path of its file:
        # a worker lives as long as its pool and the file, or a file which a
        # hierarchical model includes, may be rewritten meanwhile
        source = model.sbml
        parameters = tuple(
            sorted((str(k), float(v)) for k, v in (model.parameters or {}).items())
        )
        jit_optimization = (
            model.jit_optimization if isinstance(model, RoadrunnerSBMLModel) else None
        )
        integrator, settings = model.integrator_state()
        cache = amici_cache(model.engine, (model,))
        digest = hashlib.sha256(
            repr(
                (
                    str(model.engine),
                    source,
                    parameters,
                    str(jit_optimization),
                    integrator,
                    settings,
                )
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            key=digest,
            engine=model.engine,
            source=source,
            parameters=parameters,
            jit_optimization=jit_optimization,
            integrator=integrator,
            settings=settings,
            cache=cache,
        )

    def load(self) -> SBMLModel:
        """Load the model in its engine with the integrator and its settings."""
        with using_amici_cache(self.cache):
            model = load_model(
                AbstractModel(
                    source=self.source, parameters=dict(self.parameters) or None
                ),
                self.engine,
                jit_optimization=self.jit_optimization,
            )
        model.set_integrator_state((self.integrator, self.settings))
        return model


def core_threads(n_workers: int | None, n_points: int) -> int:
    """Get the number of threads the engine core batches the chunks of a run with.

    The threads of the batches of a run are counted like the processes of a
    run of another engine, `parallel.resolve_workers`: a number is taken as
    given, `None` is every CPU of the quota for a run of
    `parallel.POOL_THRESHOLD` points or more, and a run of `n_workers=1`, a
    smaller run and a run inside a worker of a pool, which has its cores
    already, use one thread.

    Args:
        n_workers: the workers of the run, see `parallel.resolve_workers`.
        n_points: the points of the run.

    Returns:
        The number of threads of a batch, which `parallel.resolve_workers`
        decides.
    """
    return parallel.resolve_workers(n_workers, n_points)


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
        """Get the selections of the engine, `time` first."""
        return (TIME, *self.graph.selections)

    def plan_of(self, k: int) -> Plan:
        """Get the plan of the `k`-th point of the chunk, see `point_plan`."""
        return point_plan(self.plan, self.values, self.timed, k)


@dataclass(frozen=True)
class ChunkResult:
    """The answer of a chunk.

    Attributes:
        indices: the flat indices of the points, those of the chunk.
        values: the kept timecourses, `(column, point, row)` with the time
            first and the timecourses of the graph, padded with `NaN`. It is
            the layout of the cube of the run, so a chunk is written into it
            without moving an axis, see `sbmlsim.simulator.simulator`.
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


def run_chunk(
    chunk: Chunk,
    model: SBMLModel,
    threads: int = 1,
    out: np.ndarray | None = None,
) -> ChunkResult:
    """Run the points of a chunk on a loaded model, see the module.

    Args:
        chunk: the points.
        model: the loaded model of the chunk, of any engine.
        threads: the threads of the engine core a chunk of a model of that
            engine is batched on, see `_run_chunk_core`; every other engine
            runs the points of a chunk one after the other in this process.
        out: where to write the values of the chunk, see `_pack`; only a run
            in the calling process has the cube of the run to offer.

    Returns:
        The values of every point.

    Raises:
        ScanPointError: for the first point which fails, with
            `on_error="raise"`.
        NotImplementedError: for a feature the engine does not simulate, also
            with `on_error="flag"`.
    """
    chunk.plan.check_model(model.symbols, model.uinfo)
    if model.engine is Engine.CORE:
        return _run_chunk_core(chunk, core_model(model), threads, out)
    quiet: AbstractContextManager[None] = (
        quiet_roadrunner() if model.engine is Engine.ROADRUNNER else nullcontext()
    )
    with quiet:
        return _run_chunk(chunk, model, out)


def _run_chunk(
    chunk: Chunk, model: SBMLModel, out: np.ndarray | None = None
) -> ChunkResult:
    """Run the points of a chunk one after the other, see `run_chunk`.

    A chunk of a model of the engine core runs as one batch of its threads
    instead, see `_run_chunk_core`, which shares the tail of this, `_finish`.

    With `on_error="raise"` the first point which fails in the order of the
    scan is raised, whatever the chunking: a failed simulation stops the
    chunk, the observables of the points before it are evaluated, and a point
    among them whose observable fails comes before it.

    Without observables (a graph without nodes) the kept selections of a
    solution are its outputs, which are interpolated onto the grid right after
    the simulation: such a chunk never holds the native solutions of its
    points, which only the observables need.

    The `NotImplementedError` of an executor, a feature the engine does not
    simulate, is raised as it is, see the module.
    """
    n = len(chunk.indices)
    failures: list[tuple[int, str, BaseException]] = []
    points: list[int] = []
    solutions: list[np.ndarray] = []
    plans: list[Plan] = []
    observed = bool(chunk.graph.nodes)
    selections = chunk.selections
    columns = _kept_columns(chunk)
    for k in range(n):
        # a definition which does not fit the plan is no failed point
        plan = chunk.plan_of(k)
        try:
            result = execute_plan(plan, model, selections)
        except NotImplementedError:
            raise
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
            solutions.append(
                _on_grid(chunk.time, _timecourses_of(result.values, columns), plan)
            )
    return _finish(chunk, points, solutions, plans, failures, out)


def _run_chunk_core(
    chunk: Chunk, model: CoreSBMLModel, threads: int, out: np.ndarray | None = None
) -> ChunkResult:
    """Run the points of a chunk of the engine core as one batch, see the module.

    The engine simulates a batch of requests on threads of its own and
    answers with the output of every point or with the error which failed it
    (`PointError`), so the points of a chunk run at once in this process
    rather than one after the other, which is why a scan of the engine starts
    no pool: `Simulator.run` gives a run of it one python worker and the
    threads of the batch, see `sbmlsim.simulator.simulator`.

    The result is the one of the points run one by one, bitwise: every thread
    of the batch simulates on an instance of its own and the engine holds no
    state between two requests, so neither the threads nor the chunking of
    the points reach the numbers (measured, see
    `tests/simulator/test_scan_core.py`). The failures of the points are
    sorted and reported as the ones of the loop are, so a run raises the
    error of the first point in the order of the scan whatever the threads
    ran first, and a point the engine does not support raises for the whole
    scan, also with `on_error="flag"`. A point whose request cannot be built
    fails alone as well, as one whose simulation fails: the points of the
    batch are the ones whose request was built, and `on_error` decides about
    the others.

    The requests of the points are built together, their folds batched over
    the points of the chunk (`executor_core.requests_of`, #550): a relative
    protocol costs one segment of a simulation per change and point, which
    run on the threads of the engine as the points themselves do and not one
    after the other in this process. Every point folds its own state, so the
    chunk is the one of the points folded one by one, bitwise.

    The model records no pre-initialization here, unlike `executor_core.execute`
    of a single simulation: the engine is stateless and every point of the
    batch carries its own values in its request.

    The engine answers with the table of every point of the batch at once, so
    a chunk of it holds the tables of its points, and not the grid of one
    point at a time as the loop does: the points of such a chunk are capped
    at `CORE_MAX_CHUNK` for that. A chunk without observables keeps the kept
    timecourses of a table as a view of it (`_timecourses_of`), so the tables
    are held until `_pack` copies them and the chunk copies the values once
    rather than twice; the peak is the same, the tables and the values of the
    chunk, where it was the copies and the values before. Measured
    on the midazolam model with a grid of 200 output times and all 268
    selections, the widest scan of the package (2000 points, a result of 815
    MiB): the process peaks at 1.45 GiB in the chunks of 250 points of the
    cap against 1.87 GiB in chunks of 500, and the same scan of roadrunner,
    which holds the grid of one point at a time, at 1.66 GiB. The cap costs
    no measurable time: the pool of threads a batch builds is a few
    milliseconds against the hundreds a batch of 250 points of such a model
    integrates.

    Args:
        chunk: the points.
        model: the loaded model of the chunk.
        threads: the threads of the engine the batch runs on.
        out: where to write the values of the chunk, see `_pack`.

    Returns:
        The values of every point.

    Raises:
        ScanPointError: for the first point which fails, with
            `on_error="raise"`.
        NotImplementedError: for a feature the engine does not simulate, also
            with `on_error="flag"`.
        ValueError: if the engine refuses the batch, see `_batch`.
    """
    # the module of the executor of the engine core, imported here as
    # `sbmlsim.simulator.engines` imports it, so that a run of another engine
    # does not import it; the bindings are imported by `import_core`
    from sbmlsim.simulator import executor_core

    core = import_core()
    observed = bool(chunk.graph.nodes)
    kept = _kept_columns(chunk)
    columns = output_columns(chunk.selections)
    # the tolerance of every state is computed once for the whole chunk
    settings = model.request_settings()
    failures: list[tuple[int, str, BaseException]] = []
    # a definition which does not fit the plan is no failed point
    plans: list[Plan] = [chunk.plan_of(k) for k in range(len(chunk.indices))]
    batched: list[int] = []
    requests: list[dict[str, Any]] = []
    # the requests of the points of the chunk, the k-th segment of the folds
    # of all of them simulated as one batch (#550)
    for k, request in enumerate(
        executor_core.requests_of(
            [(plan, columns) for plan in plans],
            model,
            lambda segments: _batch(model, segments, threads),
            settings,
        )
    ):
        if isinstance(request, Exception):
            failures.append((k, f"{type(request).__name__}: {request}", request))
            continue
        requests.append(request)
        batched.append(k)
    outputs = _batch(model, requests, threads)
    # the requests of the batch are released with it, the engine has read them
    del requests
    points: list[int] = []
    solutions: list[np.ndarray] = []
    ran: list[Plan] = []
    for position, k in enumerate(batched):
        plan = plans[k]
        output = outputs[position]
        # the table of the point is released once its values are read: the
        # array of a point of a chunk without observables is dropped with it
        outputs[position] = None
        if isinstance(output, core.PointError):
            err = executor_core.point_error(output, model)
            if isinstance(err, NotImplementedError):
                raise err
            failures.append((k, f"{type(err).__name__}: {err}", err))
            continue
        # the time is shifted in the array the bindings own, which
        # `output.values` answers with as it is, the same array on every
        # access, so it is read once and the chunk holds it until `_pack`
        values = output_time(output.values, columns, plan, model.symbols)
        points.append(k)
        ran.append(plan)
        solutions.append(
            values
            if observed
            else _on_grid(chunk.time, _timecourses_of(values, kept), plan)
        )
    return _finish(chunk, points, solutions, ran, failures, out)


def _batch(
    model: CoreSBMLModel, requests: list[dict[str, Any]], threads: int
) -> list[Any]:
    """Simulate the requests of a chunk as one batch of the engine core.

    The batch runs on the executor of the model (`CoreSBMLModel.batch_executor`,
    one per number of threads), which keeps the pool of its threads and the
    instance of every one of them across the chunks of a run and across the
    runs of a model, so a chunk pays for neither.

    Args:
        model: the loaded model of the chunk.
        requests: the request of every point of the batch, in their order.
        threads: the threads of the engine the batch runs on, the workers of
            the run.

    Returns:
        The output of every request in the order of the requests, the
        `PointError` of a point which failed among them.

    Raises:
        ValueError: if the engine refuses the batch, e.g. for more threads
            than it runs on; the message names the threads, which the
            `n_workers` of the simulator says.
    """
    try:
        return model.batch_executor(threads).simulate_many(requests)
    except ValueError as err:
        raise ValueError(
            f"The engine 'core' refused the batch of a chunk on {threads} "
            f"threads, the 'n_workers' of the run: {err}"
        ) from err


def _kept_columns(chunk: Chunk) -> list[int]:
    """Get the columns of the time and of the kept timecourses of a chunk.

    The first column of every selection, without a search per selection; a
    chunk with observables keeps the native solutions of its points and has
    none of them.
    """
    if chunk.graph.nodes:
        return []
    position: dict[str, int] = {}
    for j, selection in enumerate(chunk.selections):
        position.setdefault(selection, j)
    return [position[s] for s in (TIME, *chunk.graph.timecourses)]


def _finish(
    chunk: Chunk,
    points: list[int],
    solutions: list[np.ndarray],
    plans: Sequence[Plan],
    failures: list[tuple[int, str, BaseException]],
    out: np.ndarray | None = None,
) -> ChunkResult:
    """Build the result of a chunk from the points which ran, see `_run_chunk`.

    The observables are evaluated on the native solutions, the kept
    timecourses are interpolated onto the grid of the chunk and the points
    which failed are reported in the order of the scan: the first of them is
    raised with `on_error="raise"` and every one of them is flagged
    otherwise. Every engine shares it, the loop over the points of
    `_run_chunk` and the batch of `_run_chunk_core`.

    Args:
        chunk: the points.
        points: the positions in the chunk of the points which ran.
        solutions: their native solutions, or their timecourses on the grid
            of the chunk when it has no observables.
        plans: the plans of the points which ran; a chunk without observables
            does not need them.
        failures: the points which failed with their message and error, which
            the observables of the chunk extend and which is sorted here.
        out: where to write the values, see `_pack`.

    Returns:
        The values of every point.

    Raises:
        ScanPointError: for the first point which fails, with
            `on_error="raise"`.
    """
    status = np.zeros(len(chunk.indices), dtype=np.int8)
    if chunk.graph.nodes:
        plan_of = dict(zip(points, plans, strict=True))
        points, time, outputs, failed = _observe(chunk, points, solutions, plans)
        failures.extend(failed)
        n_rows = time.shape[1]
        rows: Iterable[np.ndarray] = _timecourses(
            chunk, time, outputs, [plan_of[k] for k in points]
        )
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
    return _pack(chunk, points, n_rows, rows, scalars, status, errors, out)


def _timecourses_of(values: np.ndarray, columns: list[int]) -> np.ndarray:
    """Get the kept columns of a solution as `(column, row)`, the time first.

    `values[:, columns].T` would copy the whole solution to select from it,
    also when `columns` is every column of it in order, which is what a run
    without observables keeps of a model whose selections are its outputs.
    The transpose of the solution is a view of it, so the common case copies
    nothing at all and the chunk copies the values exactly once, in `_pack`.
    Measured on the HCTZ model with 401 output times and 191 selections, a
    chunk of 250 points: 47.7 ms for the selecting copy against 0.0 ms for
    the view.

    Args:
        values: the solution, `(row, column)`.
        columns: the kept columns, the time first.

    Returns:
        The kept timecourses, `(column, row)`; a view where it can be one.
    """
    if len(columns) == values.shape[1] and columns == list(range(values.shape[1])):
        return values.T
    return values[:, columns].T


def _on_grid(grid: np.ndarray | None, rows: np.ndarray, plan: Plan) -> np.ndarray:
    """Interpolate timecourses `(column, row)` with the time first onto a grid.

    The output has the states before and after a change at its time, and an
    event of the model; time points closer than the resolution of the time of
    the plan are one jump. A time of the grid before a change is interpolated
    toward the state before it, the time of a change sees the state after it,
    as the data of a fit does, see `grid_weights`. Without a grid the
    timecourses are returned as they are.

    Args:
        grid: the times of the grid, `None` for none.
        rows: the timecourses, the time first.
        plan: the plan of the point, with its changes.
    """
    if grid is None:
        return rows
    tolerance = time_tolerance(plan.start, plan.end)
    changes = plan.output_change_times
    rows = apply_weights(grid_weights(rows[0], grid, tolerance, changes), rows)
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
    chunk: Chunk,
    time: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    plans: Sequence[Plan],
) -> Iterator[np.ndarray]:
    """Get the time and the kept timecourses of every point, see `_pack`.

    The points are interpolated onto the grid of the chunk one at a time, as
    `_pack` writes them, so the interpolated outputs are not held twice.
    `plans` are the plans of the points, with their changes.
    """
    names = chunk.graph.timecourses
    if not names:
        return
    for r in range(time.shape[0]):
        rows = np.stack([time[r], *(outputs[name][r] for name in names)])
        yield _on_grid(chunk.time, rows, plans[r])


def _pack_threads(nbytes: int, n_points: int) -> int:
    """Get the threads `_pack` writes the points of a chunk with.

    The copy of a point is bound by the memory of the machine, so the threads
    help a chunk which is large enough to pay for them and saturate well
    below one per CPU (`PACK_MAX_THREADS`). A worker of a pool writes with the
    calling thread: it has the cores of one worker and the other workers are
    copying beside it, as a run of the engine core, which writes in the
    calling process, has no pool at all.

    Args:
        nbytes: the values of the chunk.
        n_points: the points which ran.

    Returns:
        The number of threads, `1` to write with the calling thread.
    """
    if nbytes < PACK_THREAD_BYTES or n_points < 2 or parallel.in_worker():
        return 1
    return max(1, min(PACK_MAX_THREADS, cgroup.available_cpus(), n_points))


def _write_point(
    values: np.ndarray, chunk: Chunk, n_rows: int, k: int, rows: np.ndarray
) -> None:
    """Write the rows of one point of a chunk into its values, see `_pack`.

    `rows` is `(column, row)`, the layout of the cube, so the rows of a point
    are written without transposing them. A point shorter than the values,
    which a ragged output has, is padded with `NaN` and keeps the time of the
    grid of the chunk where it has one.

    Args:
        values: the values of the chunk, `(column, point, row)`.
        chunk: the chunk.
        n_rows: the rows of the values.
        k: the position of the point in the chunk.
        rows: the time and the kept timecourses of the point.
    """
    kept = rows.shape[1]
    values[:, k, :kept] = rows
    if kept < n_rows:
        values[:, k, kept:] = np.nan
        if chunk.time is not None:
            values[0, k, kept:] = chunk.time[kept:]


def _pack(
    chunk: Chunk,
    points: Sequence[int],
    n_rows: int,
    rows: Iterable[np.ndarray],
    scalars: np.ndarray,
    status: np.ndarray,
    errors: list[tuple[int, str]],
    out: np.ndarray | None = None,
) -> ChunkResult:
    """Write the outputs of the points which ran into the arrays of a chunk.

    A failed point is `NaN`, with the time of the grid if the chunk has one.

    With `out` the values are written where the run keeps them, the slice of
    its cube which holds the points of the chunk, instead of into an array of
    the chunk which the run then copies into that slice: a run in the calling
    process copies the values of a point exactly once, from the solution into
    the result. The chunk answers with that slice as its values, and the run
    knows not to write them again, see `sbmlsim.simulator.simulator`.

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
        out: where to write the values, a view of the cube of the run; `None`
            allocates the values of the chunk, which a run in a pool sends
            back and a run of a ragged output keeps until its rows are known.

    Raises:
        ValueError: if `out` is not the shape the chunk writes.
    """
    n = len(chunk.indices)
    timed = bool(chunk.graph.timecourses) or not chunk.graph.nodes
    if not timed:
        table = np.full((n, len(chunk.graph.scalars)), np.nan)
        for k, point_scalars in zip(points, scalars, strict=True):
            table[k] = point_scalars
        return ChunkResult(
            indices=chunk.indices,
            values=np.empty((0, n, 0)),
            scalars=table,
            status=status,
            errors=tuple(sorted(errors)[:MAX_ERRORS]),
        )
    if out is not None:
        # the destination carries the rows of the run, which a chunk whose
        # points all failed does not know: its own count would be 0
        n_rows = out.shape[2]
    elif chunk.time is not None:
        n_rows = chunk.time.size
    # the values are filled in full below, point by point: a point which ran
    # writes its rows and the `NaN` of the rows it does not have, and the
    # points which failed are filled after them, so a chunk whose points all
    # ran and are all as long as the grid, the common case, never pays for a
    # pass of `NaN` over the whole array it then overwrites (measured on the
    # HCTZ model, 1000 points of 401 rows and 191 selections: 66 ms of a run
    # of 485 ms)
    shape = (1 + len(chunk.graph.timecourses), n, n_rows)
    if out is None:
        values = np.empty(shape)
    elif out.shape != shape:
        raise ValueError(
            f"The run gave the chunk a destination of the shape {out.shape} "
            f"for values of the shape {shape}."
        )
    else:
        values = out
    table = np.full((n, len(chunk.graph.scalars)), np.nan)
    written = np.zeros(n, dtype=bool)
    # the points of a chunk with observables are interpolated one at a time
    # as they are written (`_timecourses`), so they are not held twice and
    # there is nothing to spread over threads
    held = rows if isinstance(rows, Sequence) else None
    threads = 1 if held is None else _pack_threads(values.nbytes, len(points))
    if held is not None and threads > 1:

        def write(part: Iterable[int]) -> None:
            """Write the points of one part, which no other thread writes."""
            for i in part:
                _write_point(values, chunk, n_rows, points[i], held[i])

        # a part writes the points `i`, `i + threads`, ... and the points of
        # two parts are disjoint, so no two threads write into one cell and
        # the values are the ones of a write by the calling thread, bitwise
        parts = [range(i, len(points), threads) for i in range(threads)]
        with ThreadPoolExecutor(threads) as pool:
            for _ in pool.map(write, parts):
                pass
        for i, k in enumerate(points):
            written[k] = True
            table[k] = scalars[i]
    else:
        for k, point_rows, point_scalars in zip(points, rows, scalars, strict=True):
            _write_point(values, chunk, n_rows, k, point_rows)
            written[k] = True
            table[k] = point_scalars
    if not written.all():
        # a point which failed is `NaN`, with the time of the grid if there is one
        values[:, ~written] = np.nan
        if chunk.time is not None:
            values[0, ~written] = chunk.time
    return ChunkResult(
        indices=chunk.indices,
        values=values,
        scalars=table,
        status=status,
        errors=tuple(sorted(errors)[:MAX_ERRORS]),
    )


@dataclass(frozen=True)
class SharedChunkResult:
    """The answer of a chunk of a worker whose large arrays are in shared memory.

    The fields of `ChunkResult`, of which `values` and `scalars` are a
    `SharedArray` when the worker wrote them into a segment, see
    `sbmlsim.simulator.shared`.
    """

    indices: np.ndarray
    values: np.ndarray | SharedArray
    scalars: np.ndarray | SharedArray
    status: np.ndarray
    errors: tuple[tuple[int, str], ...]

    def received(self) -> ChunkResult:
        """Get the `ChunkResult`, its arrays read from their segments.

        Such an array is a read-only view on its segment, whose name is
        removed, so this is called once, in the parent, see
        `sbmlsim.simulator.shared.receive`.
        """
        return ChunkResult(
            indices=self.indices,
            values=_received(self.values),
            scalars=_received(self.scalars),
            status=self.status,
            errors=self.errors,
        )


def _received(array: np.ndarray | SharedArray) -> np.ndarray:
    """Get an array, read from its segment if it is in one."""
    return shared.receive(array) if isinstance(array, SharedArray) else array


def run_chunk_in_worker(
    spec: ModelSpec, chunk: Chunk, segment: str | None = None
) -> ChunkResult | SharedChunkResult:
    """Run a chunk in a worker process, on the model the worker keeps.

    Args:
        spec: the model of the chunk and the settings of its integrator.
        chunk: the points.
        segment: the stem of the names of the segments of shared memory the
            arrays of the result are written into, `values` and `scalars`
            with the `SUFFIXES` of `sbmlsim.simulator.shared`;
            `None` returns the arrays through the pipe of the pool.

    Returns:
        The values of every point, see `run_chunk`, with its large arrays in
        shared memory when `segment` is given.
    """
    model = parallel.worker_cache(("model", spec.key), spec.load)
    result = run_chunk(chunk, model)
    if segment is None:
        return result
    values, scalars = shared.segment_names(segment)
    return SharedChunkResult(
        indices=result.indices,
        values=shared.share(result.values, values),
        scalars=shared.share(result.scalars, scalars),
        status=result.status,
        errors=result.errors,
    )


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
