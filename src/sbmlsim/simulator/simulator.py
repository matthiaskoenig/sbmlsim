"""The simulator: simulations and scans of models, serially or in a pool.

`Simulator.run(model, scan)` runs every point of a scan in four steps:

1. Compile: a plan per combination of the simulations and the models of the
   scan (one plan without a dimension of simulations or of models), with
   `compile_simulation`; the values of the dimensions of values in the units
   of their targets in every model, once; the times of the dimensions with
   `at` in the time unit of every model; the selections and the grid of the
   output; the observables of the run, ordered, checked against the first
   model and with their kinds and units, see `sbmlsim.simulator.observables`.
2. Points: a point is a tuple of indices, nothing is built per point in the
   parent. The points are cut into chunks which share a plan, at most
   `ceil(n_points / (4 * n_workers))` and at most `MAX_CHUNK` points each.
3. Worker: a chunk applies the values of each point to its plan and runs it,
   see `sbmlsim.simulator.worker`; the same function runs serially in the
   calling process and in a worker of a pool of `sbmlsim.parallel`; the
   observables are evaluated on the native solutions of the points of a chunk
   before the kept timecourses are interpolated onto a grid.
4. Assembly: the arrays of the chunks are written into the arrays of the
   result, a `ScanResult`, and reshaped into the dimensions of the scan.

The result does not depend on the number of workers or the size of the
chunks: the values of a scan are fixed before the run. A run which raises
raises the error of the first point in the order of the scan which fails, and
a run which flags the points which fail keeps their errors in that order.

The integrator settings of a simulator apply to every model it runs: the model
of a run, every model of a dimension of models and every model a worker
loads, each with its own tolerance per state. A setting of a model which the
simulator does not set stays, and a worker loads the model with the
integrator and the settings the model has in the calling process.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
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
from sbmlsim.result.scan import POINT, STATUS, TIME, ScanResult, time_magnitudes
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.observables import Observable
from sbmlsim.simulation.scan import RESERVED, DimensionKind, Scan
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.observables import ObservableGraph, compile_observables
from sbmlsim.simulator.plan import Plan, compile_simulation, model_time, target_values
from sbmlsim.simulator.worker import (
    MAX_ERRORS,
    Chunk,
    ChunkResult,
    ModelSpec,
    OnError,
    ScanPointError,
    point_plan,
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
        parallel.check_workers(n_workers)
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

        Raises:
            ValueError: if the simulation does not fit the model, see
                `compile_simulation`.
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

        Raises:
            ValueError: if the simulation does not fit the model.
            RuntimeError: if roadrunner fails to integrate.
        """
        loaded = self.load(model)
        plan = (
            simulation
            if isinstance(simulation, Plan)
            else self.compile(loaded, simulation)
        )
        return execute(plan, loaded, loaded.selections or [TIME])

    def run(
        self,
        model: ModelLike | None,
        scan: Scan | Simulation,
        observables: Sequence[Observable] | None = None,
        *,
        time: ArrayLike | Quantity | None = None,
        keep: Sequence[str] | None = None,
        on_error: OnError = "raise",
        progress: bool | None = None,
    ) -> ScanResult:
        """Run every point of a scan, see the module.

        The variables of the result are the kept observables; without
        observables they are the selections of the model, of the first model of
        a dimension of models, each a timecourse of its own name.

        Args:
            model: the model of the run; `None` for a scan with a dimension
                of models, which gives the model of every point.
            scan: the scan, a simulation is a scan without dimensions.
            observables: what the run computes from every simulation, see
                `sbmlsim.simulation.observables`; they are evaluated on the
                native solution of every simulation, also with `time`.
            time: a grid of times every timecourse is interpolated onto,
                numbers in the time unit of the model or a quantity; without
                it the result keeps the output times of the simulations, a
                dimension `time` if they agree and `_point` otherwise.
            keep: the observables of the result, every one by default; the id
                of a PK observable keeps all its parameters; without
                observables the selections of the result. The others are
                evaluated where a kept one needs them and dropped.
            on_error: `"raise"` raises a `ScanError` for the first point in
                the order of the scan which fails; `"flag"` sets every value of
                a point which fails to `NaN` and records it in the variable
                `status`. A failing point is reported by its error, not by the
                log of roadrunner; a run in a pool prints no messages of the
                integrator (SUNDIALS), while a serial run, which integrates the
                model of the user in the calling process, still shows them, a
                dozen lines or more per failing point.
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
        compiled = self._compile(model, Scan.of(scan), time, observables, keep)
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
        """Get the loaded models of a run and their labels.

        Raises:
            ValueError: if neither the run nor a dimension of models gives a
                model, or both do.
        """
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
        self,
        model: ModelLike | None,
        scan: Scan,
        time: ArrayLike | Quantity | None,
        observables: Sequence[Observable] | None = None,
        keep: Sequence[str] | None = None,
    ) -> _Compiled:
        """Compile a scan against its models, see the module.

        Args:
            model: the model of the run, or `None`.
            scan: the scan.
            time: the grid of times, see `run`.
            observables: the observables of the run.
            keep: the ids of the observables to keep.

        Raises:
            ValueError: if neither the run nor a dimension of models gives a
                model, or both do; an observable which does not fit the first
                model, see `compile_observables`; an observable id which is a
                dimension id or a target the scan changes; if a model of a
                dimension of models has not the selections or the units of the
                first one; if a selection is a name the result reserves, e.g.
                `status`; if a dimension id is a selection; if a simulation, a
                value or a time does not fit a model, e.g. a target which is no
                target of a model; if two dimensions set one target at one
                time; if a coordinate of a dimension has the name of an output,
                a dimension id, a changed target or a coordinate of another
                dimension; or if the grid of times is empty or a quantity which
                the unit of time of the first model cannot take.
        """
        models, labels = self._models(model, scan)
        first = models[0]
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
        positions = _positions(scan)
        # the plan of the first point of every simulation and model carries the
        # assignments of the values dimensions and of those with `at`, which
        # give the dose times of the PK observables, see `compile_pk`
        point_plans: list[Plan] = []
        for (s, m), plan in plans.items():
            indices = _plan_points(scan, positions, s, m)
            if indices.size:
                values, timed = _values_of(
                    scan, positions, vectors[m], at_times[(s, m)], indices[:1]
                )
                point_plans.append(point_plan(plan, values, timed, 0))
        graph = compile_observables(observables, first, keep=keep, plans=point_plans)
        # the time is the first column, also where the model selects it
        # elsewhere, e.g. last in the sorted selections of an experiment
        selections = (TIME, *graph.selections)
        if not observables:
            reserved = sorted((set(selections[1:]) & RESERVED) - {TIME})
            if reserved:
                raise ValueError(
                    f"The selections {reserved} are names of the result "
                    f"({sorted(RESERVED)}), select other entities, see "
                    f"`RoadrunnerSBMLModel.set_selections`."
                )
        for loaded, label in zip(models[1:], labels[1:], strict=True):
            _check_model(loaded, label, first, (*selections, *graph.doses))
        clash = sorted(set(scan.dims) & set(graph.outputs))
        if clash:
            raise ValueError(
                f"The dimension ids {clash} are selections of the model or "
                f"observables of the run: a dimension and a variable of the "
                f"result share no name, choose other ids."
            )
        # an observable id is no changed target (its values would be read as
        # the observable) and no dimension id (a PK observable `<id>` has the
        # outputs `<id>.<parameter>`, which would meet the qualified values
        # `<dimension>.<target>` of the dimension)
        ids = {o.id for o in observables or ()}
        targets = {t for dimension in scan.dimensions for t in dimension.values}
        changed = sorted(targets & ids)
        if changed:
            raise ValueError(
                f"The observables {changed} are targets the scan changes; "
                f"choose other ids."
            )
        same = sorted(set(scan.dims) & ids)
        if same:
            raise ValueError(
                f"The observables {same} have the id of a dimension of the "
                f"scan; choose other ids."
            )
        _check_coordinates(scan, set(graph.outputs))
        grid, interpolate = _grid(plans, time, first)
        return _Compiled(
            scan=scan,
            models=models,
            plans=plans,
            vectors=vectors,
            at_times=at_times,
            graph=graph,
            grid=grid,
            interpolate=interpolate,
            observables=tuple(observables or ()),
        )

    def _run_serial(
        self,
        compiled: _Compiled,
        chunks: Sequence[Chunk],
        advance: Callable[[int], None],
    ) -> list[ChunkResult]:
        """Run the chunks in the calling process.

        The chunks are in the order of their first point. After a point
        failed, only the chunks which start before it still run: one of them
        may hold a point before it which fails as well.

        Raises:
            ScanError: for the first point in the order of the scan which
                fails, with `on_error="raise"`.
        """
        results: list[ChunkResult] = []
        failed: ScanPointError | None = None
        for chunk in chunks:
            if failed is not None and int(chunk.indices[0]) > failed.index:
                break
            try:
                results.append(run_chunk(chunk, compiled.models[chunk.model]))
            except ScanPointError as err:
                failed = _first(failed, err)
                continue
            advance(len(chunk.indices))
        if failed is not None:
            raise compiled.error(failed) from failed
        return results

    def _run_pool(
        self,
        compiled: _Compiled,
        chunks: Sequence[Chunk],
        workers: int,
        advance: Callable[[int], None],
    ) -> list[ChunkResult]:
        """Run the chunks in the kept pool of `sbmlsim.parallel`.

        After a point failed, the chunks which start after it and did not
        start yet are cancelled and the ones before it still run, see
        `_run_serial`. The pool stops when a worker dies or the run is
        interrupted, e.g. by Ctrl-C, and the next run starts a new one.

        Raises:
            ScanError: for the first point in the order of the scan which
                fails, with `on_error="raise"`.
            RuntimeError: if a worker of the pool dies.
        """
        specs = [ModelSpec.of(model) for model in compiled.models]
        executor = parallel.pool(workers)
        results: list[ChunkResult] = []
        failed: ScanPointError | None = None
        futures: dict[Future[ChunkResult], Chunk] = {}
        try:
            for chunk in chunks:
                future = executor.submit(run_chunk_in_worker, specs[chunk.model], chunk)
                futures[future] = chunk
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except ScanPointError as err:
                        failed = _first(failed, err)
                        # `wait` reports a cancelled future only once the pool
                        # reaches it, so it leaves pending here
                        pending = {
                            other
                            for other in pending
                            if int(futures[other].indices[0]) < failed.index
                            or not other.cancel()
                        }
                        continue
                    results.append(result)
                    advance(len(result.indices))
        except BrokenProcessPool as err:
            parallel.stop(executor)
            raise RuntimeError(
                f"A worker of the pool died while it ran the scan: {err}"
            ) from err
        except Exception:
            # the pool is fine, the chunks which did not start are dropped
            for future in futures:
                future.cancel()
            raise
        except BaseException:
            # e.g. Ctrl-C: the workers still run their chunks, the pool stops
            parallel.stop(executor)
            raise
        if failed is not None:
            raise compiled.error(failed) from failed
        return results


def _check_coordinates(scan: Scan, outputs: set[str]) -> None:
    """Check that no coordinate hides or is hidden by another name of the result.

    Raises:
        ValueError: if a coordinate has the name of an output, a dimension id, a
            changed target or a coordinate of another dimension.
    """
    taken: dict[str, str] = {}
    for dimension in scan.dimensions:
        for target in dimension.values:
            taken.setdefault(target, "a changed target")
    owner: dict[str, str] = {}
    for dimension in scan.dimensions:
        for name in dimension.coordinates:
            if name in RESERVED:
                what = "a name of the result"
            elif name in outputs:
                what = "an output of the result"
            elif name in scan.dims:
                what = "a dimension id"
            elif name in taken:
                what = "a changed target"
            elif name in owner:
                what = f"a coordinate of the dimension '{owner[name]}'"
            else:
                owner[name] = dimension.id
                continue
            raise ValueError(
                f"The coordinate '{name}' of the dimension '{dimension.id}' would "
                f"hide or be hidden by {what} of the same name; rename the "
                f"covariate."
            )


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
        graph: the observables of the run.
        grid: the times of a grid, `None` for the ragged layout.
        interpolate: whether the workers interpolate onto the grid.
        observables: the observables of the run, their definitions are the
            provenance of the result.
    """

    scan: Scan
    models: list[RoadrunnerSBMLModel]
    plans: dict[tuple[int, int], Plan]
    vectors: list[list[dict[str, np.ndarray]]]
    at_times: dict[tuple[int, int], list[float | None]]
    graph: ObservableGraph
    grid: np.ndarray | None
    interpolate: bool
    observables: tuple[Observable, ...] = ()

    @property
    def size(self) -> int:
        """Get the number of points."""
        return self.scan.size

    def chunks(self, workers: int, on_error: OnError) -> list[Chunk]:
        """Cut the points into chunks which share a plan, see the module.

        Returns:
            The chunks in the order of their first point.
        """
        size = _chunk_size(self.size, workers)
        positions = _positions(self.scan)
        chunks: list[Chunk] = []
        for (s, m), plan in self.plans.items():
            indices = _plan_points(self.scan, positions, s, m)
            for start in range(0, indices.size, size):
                part = indices[start : start + size]
                values, timed = _values_of(
                    self.scan, positions, self.vectors[m], self.at_times[(s, m)], part
                )
                chunks.append(
                    Chunk(
                        indices=part,
                        plan=plan,
                        model=m,
                        graph=self.graph,
                        values=values,
                        timed=timed,
                        time=self.grid if self.interpolate else None,
                        on_error=on_error,
                    )
                )
        chunks.sort(key=lambda chunk: int(chunk.indices[0]))
        return chunks

    def point_text(self, index: int) -> str:
        """Get the labels and the values of a point, for a message."""
        if not self.scan.dimensions:
            return ""
        parts: list[str] = []
        position = np.unravel_index(index, self.scan.shape)
        for dimension, k in zip(self.scan.dimensions, position, strict=True):
            parts.append(f"{dimension.id}={dimension.labels[k]}")
            for target, values in dimension.values.items():
                value = np.asarray(getattr(values, "magnitude", values))[k]
                unit = f" {values.units}" if isinstance(values, Quantity) else ""
                parts.append(f"{target}={value}{unit}")
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
        """Write the arrays of the chunks into the result, see `ScanResult`.

        A changed target is a coordinate along the first dimension which
        changes it, unless a selection of the same name is a variable: the
        variable keeps the name (its timecourse) and the values of the
        dimension are stored as `<dimension>.<target>`.
        """
        n, shape = self.size, self.scan.shape
        names = self.graph.timecourses
        scalars = self.graph.scalars
        n_rows = (
            self.grid.size
            if self.grid is not None
            else max((r.values.shape[1] for r in results), default=0)
        )
        cube = np.full((1 + len(names), n, n_rows), np.nan)
        table = np.full((len(scalars), n), np.nan)
        status = np.zeros(n, dtype=np.int8)
        errors: list[tuple[int, str]] = []
        for result in results:
            rows = result.values.shape[1]
            cube[:, result.indices, :rows] = np.moveaxis(result.values, 2, 0)
            table[:, result.indices] = result.scalars.T
            status[result.indices] = result.status
            errors.extend(result.errors)

        first = self.models[0]
        dims = list(self.scan.dims)
        tdim = TIME if self.grid is not None else POINT
        units: dict[str, str] = {}
        data_vars: dict[str, Any] = {}
        # a run without observables keeps the time also without a selection
        timed = bool(names) or not self.observables
        if timed:
            units[TIME] = first.uinfo.get(TIME, "") or ""
            if self.grid is None:
                data_vars[TIME] = ([*dims, POINT], cube[0].reshape(*shape, n_rows))
        for j, name in enumerate(names, 1):
            data_vars[name] = ([*dims, tdim], cube[j].reshape(*shape, n_rows))
            units[name] = self.graph.units[name]
        for j, name in enumerate(scalars):
            data_vars[name] = (dims, table[j].reshape(shape))
            units[name] = self.graph.units[name]
        coords: dict[str, Any] = {}
        for dimension in self.scan.dimensions:
            coords[dimension.id] = np.array(dimension.labels)
            # labels carry no unit
            units[dimension.id] = ""
            for target, values in dimension.values.items():
                if target in data_vars:
                    # a selection of the same name: the variable keeps the
                    # name, the values are qualified by the dimension
                    name = f"{dimension.id}.{target}"
                elif target in coords:
                    # a target an earlier dimension changes
                    continue
                else:
                    name = target
                if isinstance(values, Quantity):
                    coords[name] = (dimension.id, np.array(values.magnitude))
                    units[name] = str(values.units)
                else:
                    coords[name] = (dimension.id, np.array(values))
                    units[name] = first.uinfo.get(target, "") or ""
            for name, values in dimension.coordinates.items():
                if name in data_vars or name in coords:
                    continue
                if isinstance(values, Quantity):
                    coords[name] = (dimension.id, np.array(values.magnitude))
                    units[name] = str(values.units)
                else:
                    coords[name] = (dimension.id, np.array(values))
                    units[name] = ""
        if self.grid is not None and timed:
            coords[TIME] = self.grid
        attrs: dict[str, Any] = {
            "dims": dims,
            "units": units,
            "scan": self.scan.to_dict(),
            "integrator_settings": _settings(settings),
        }
        if self.observables:
            attrs["observables"] = [o.to_dict() for o in self.observables]
        if on_error == "flag":
            data_vars[STATUS] = (dims, status.reshape(shape))
            units[STATUS] = ""
            attrs["errors"] = [
                f"{self.point_text(i)}: {message}" if self.scan.dimensions else message
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


def scan_point_plans(
    scan: Scan, model: RoadrunnerSBMLModel, plan: Plan, positions: np.ndarray
) -> list[Plan]:
    """Get the plans of points of a scan of value dimensions on a compiled plan.

    The fit simulates the points of a scan with the values it fits applied to
    the plan first, so the values of a dimension win over a fitted value of
    the same target, as in a run of the scan.

    Args:
        scan: the scan, whose dimensions all set values.
        model: the loaded model, whose units the values are converted to.
        plan: the compiled plan of `scan.simulation` on the model, with any
            values applied already.
        positions: the index of every point along every dimension, a row per
            point.

    Returns:
        The plan of every point, in the order of `positions`.

    Raises:
        ValueError: for a dimension of simulations or models, or values which
            do not fit the model.
    """
    for dimension in scan.dimensions:
        if dimension.kind is not DimensionKind.VALUES:
            raise ValueError(
                f"The dimension '{dimension.id}' of the scan is a dimension of "
                f"{dimension.kind.value}; the points of a fit come from a scan whose "
                f"dimensions all set values."
            )
    positions = np.asarray(positions, dtype=int).reshape(-1, len(scan.dimensions))
    vectors = _vectors(scan, model, "")
    at_times = _at_times(scan, scan.simulation, model, plan)
    values, timed = _values_of(
        scan, positions, vectors, at_times, np.arange(len(positions))
    )
    return [point_plan(plan, values, timed, k) for k in range(len(positions))]


def _positions(scan: Scan) -> np.ndarray:
    """Get the index of every point along every dimension, a row per point."""
    if not scan.dimensions:
        return np.zeros((1, 0), dtype=int)
    return np.stack(np.unravel_index(np.arange(scan.size), scan.shape), axis=1)


def _axis(scan: Scan, kind: DimensionKind) -> int | None:
    """Get the position of the dimension of a kind, `None` without one."""
    return next((k for k, d in enumerate(scan.dimensions) if d.kind is kind), None)


def _plan_points(scan: Scan, positions: np.ndarray, s: int, m: int) -> np.ndarray:
    """Get the flat indices of the points of a simulation and a model."""
    mask = np.ones(scan.size, dtype=bool)
    sim_axis = _axis(scan, DimensionKind.SIMULATIONS)
    model_axis = _axis(scan, DimensionKind.MODELS)
    if sim_axis is not None:
        mask &= positions[:, sim_axis] == s
    if model_axis is not None:
        mask &= positions[:, model_axis] == m
    return np.flatnonzero(mask)


def _values_of(
    scan: Scan,
    positions: np.ndarray,
    vectors: Sequence[Mapping[str, np.ndarray]],
    at_times: Sequence[float | None],
    part: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[float, dict[str, np.ndarray]]]:
    """Get the values of points: target -> values, and time -> target -> values."""
    values: dict[str, np.ndarray] = {}
    timed: dict[float, dict[str, np.ndarray]] = {}
    for i in range(len(scan.dimensions)):
        at = at_times[i]
        for target, vector in vectors[i].items():
            point_values = vector[positions[part, i]]
            if at is None:
                values[target] = point_values
            else:
                timed.setdefault(at, {})[target] = point_values
    return values, timed


def _chunk_size(n_points: int, workers: int) -> int:
    """Get the most points of a chunk.

    Four chunks per worker, so that a worker which finishes early takes
    another one, of at most `MAX_CHUNK` points and of at least one.

    Args:
        n_points: the points of the scan.
        workers: the number of processes of the run.

    Returns:
        The size of a chunk.
    """
    return max(1, min(MAX_CHUNK, math.ceil(n_points / (4 * workers))))


def _first(failed: ScanPointError | None, err: ScanPointError) -> ScanPointError:
    """Get the error of the point which is first in the order of the scan."""
    return err if failed is None or err.index < failed.index else failed


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
    r = model.r_loaded
    current = list(r.timeCourseSelections)
    try:
        r.timeCourseSelections = list(selections)
    except RuntimeError as err:
        raise ValueError(
            f"The model '{label}' has not every selection of the first model of "
            f"the dimension: {err}"
        ) from err
    finally:
        r.timeCourseSelections = current
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
        ValueError: if `time` is empty, or a quantity and the model has no unit
            of time or another one, see `time_magnitudes`.
    """
    if time is not None:
        grid = time_magnitudes(time, model.uinfo.get(TIME))
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
def _progress(show: bool, total: int) -> Generator[Callable[[int], None]]:
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
