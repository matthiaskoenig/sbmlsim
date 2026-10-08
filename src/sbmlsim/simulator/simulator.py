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
chunks: the values of a scan are fixed before the run. A run which raises
raises the error of the first point in the order of the scan which fails, and
a run which flags the points which fail keeps their errors in that order.

The integrator settings of a simulator apply to every model it runs: the model
of a run, every model of a dimension of models and every model a worker
loads, each with its own tolerance per state.
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
from sbmlsim.result.scan import POINT, STATUS, TIME, ScanResult
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
            on_error: `"raise"` raises a `ScanError` for the first point in
                the order of the scan which fails; `"flag"` sets every value of
                a point which fails to `NaN` and records it in the variable
                `status`.
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
        self, model: ModelLike | None, scan: Scan, time: ArrayLike | Quantity | None
    ) -> _Compiled:
        """Compile a scan against its models, see the module.

        Raises:
            ValueError: if neither the run nor a dimension of models gives a
                model, or both do; if a model of a dimension of models has not
                the selections or the units of the first one; if a dimension
                id is a selection; if a simulation, a value or a time does not
                fit a model, e.g. a target which is no target of a model; or
                if two dimensions set one target at one time.
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
        specs = [
            ModelSpec.of(model, self.integrator_settings) for model in compiled.models
        ]
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
        """Cut the points into chunks which share a plan, see the module.

        Returns:
            The chunks in the order of their first point.
        """
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
        changes it, unless a selection of the same name is a variable.
        """
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
            coords[dimension.id] = np.array(dimension.labels)
            for target, values in dimension.values.items():
                if target in data_vars or target in coords:
                    # a selection of the same name, the variable stays; or a
                    # target an earlier dimension changes
                    continue
                if isinstance(values, Quantity):
                    coords[target] = (dimension.id, np.array(values.magnitude))
                    units[target] = str(values.units)
                else:
                    coords[target] = (dimension.id, np.array(values))
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
            data_vars[STATUS] = (dims, status.reshape(shape))
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


def _first(failed: ScanPointError | None, err: ScanPointError) -> ScanPointError:
    """Get the error of the point which is first in the order of the scan."""
    return err if failed is None or err.index < failed.index else failed


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
