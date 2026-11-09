"""Module for running parameter optimizations.

The optimization runs either serial or in parallel. The parallel optimization
runs in a pool of `sbmlsim.parallel` which belongs to the fit (`worker_pool`):
one worker process per core, and every repeat of the fit goes to the worker
which is free. The pool is stopped after the fit, so the workers do not keep the
initialized problem, and it is started again, a few times, when a worker dies.

A worker cannot start workers of its own, so the workers of a fit go either to
its repeats or to the population of differential evolution, see `split_cores`:
a fit of differential evolution with fewer repeats than workers runs them one
after another, and every repeat evaluates the members of its population on all
workers (`PopulationPool`), so a single global optimization uses every core.

The `OptimizationProblem` is pickled and sent to the workers, so it must be
picklable: every worker initializes it once and runs repeats on it. The start
values are created by the runner, so a fit with a seed gives the same result
for any number of workers.

The runner only optimizes. Its result carries the fitted parameters and the
settings of the fit, and `sbmlsim.fit.report.FitReport` turns them into figures
and reports, see `sbmlsim.fit.parameters`.
"""

import datetime
import logging
import os
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from scipy.optimize import OptimizeResult

from sbmlsim import cgroup, parallel
from sbmlsim.console import console
from sbmlsim.events import emit
from sbmlsim.fit import display
from sbmlsim.fit.derived import hook_summaries
from sbmlsim.fit.optimization import (
    FitTimeout,
    OptimizationProblem,
    PoolError,
    RuntimeErrorOptimizeResult,
    run_indices,
)
from sbmlsim.fit.options import FitSettings, OptimizationAlgorithmType
from sbmlsim.fit.provenance import FitProvenance
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.log import PACKAGE_LOGGER

logger = logging.getLogger(__name__)


def resolve_n_cores(n_cores: int | None) -> int:
    """Resolve the number of worker processes.

    Args:
        n_cores: requested number of workers, `None` uses all available cores
            but one.

    Returns:
        Number of workers, at least one and at most the number of available cores.
    """
    cpu_count = cgroup.available_cpus()
    if n_cores is None:
        return max(1, cpu_count - 1)
    if n_cores > cpu_count:
        logger.error(
            "More cores '%s' then cpus '%s' requested, reducing cores.",
            n_cores,
            cpu_count,
        )
        return max(1, cpu_count - 1)
    return max(1, n_cores)


def split_cores(
    algorithm: OptimizationAlgorithmType,
    size: int,
    n_cores: int,
    kwargs: Mapping[str, Any],
) -> tuple[int, int]:
    """Split the workers of a fit between its runs and a population.

    The workers of a fit are the processes of one pool, and a worker cannot
    start workers of its own, so a run is either a task of the pool or it
    evaluates the members of its population on the pool, never both. The runs
    are the tasks as long as there are as many of them as workers or more:
    every worker optimizes a run of its own, without waiting for the others.
    Differential evolution with fewer runs than workers runs its runs one after
    another instead, and every run evaluates the members of its population on
    all workers (`PopulationPool`), so a single global optimization uses every
    core; the polishing of the best member at the end evaluates its steps on
    the workers one after another. Least squares runs its runs in parallel only.

    A population is evaluated on the workers only if it is updated once per
    generation, `updating="deferred"` (the default of the fit, see
    `OptimizationProblem.optimizer_arguments`), and its objective is not
    `vectorized`. The result of a seed is the same either way: a run evaluates
    every generation together, in its worker or on all workers.

    Args:
        algorithm: optimization algorithm.
        size: number of runs.
        n_cores: number of workers, see `resolve_n_cores`.
        kwargs: additional arguments of the optimizer.

    Returns:
        The workers of the runs and the workers of the population of a run;
        one of them is one, both are one for a serial fit.
    """
    n_cores = max(int(n_cores), 1)
    if (
        algorithm == OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION
        and size < n_cores
        and kwargs.get("updating", "deferred") == "deferred"
        and not kwargs.get("vectorized", False)
    ):
        return 1, n_cores
    return max(min(n_cores, size), 1), 1


def estimate_total_time(
    elapsed: float, completed: float, total: float, workers: int = 1
) -> float | None:
    """Estimate the total runtime from the runs which are done.

    The runs are handed to the workers in batches of `workers`, so the time of
    a batch is what the elapsed time measures: the estimate is the time per
    batch times the number of batches. With a single worker this is the mean
    time per run times the number of runs. The rate of the progress bar is
    not used, it is measured over a window of seconds and a run takes minutes.

    Args:
        elapsed: seconds since the start.
        completed: runs which are done.
        total: runs in total.
        workers: runs which are processed at once.

    Returns:
        The estimated total runtime in seconds, `None` before the first run
        is done.
    """
    if completed <= 0 or total <= 0 or elapsed <= 0:
        return None
    workers = max(int(workers), 1)
    batches_done = -(-completed // workers)
    batches_total = -(-total // workers)
    return elapsed / batches_done * batches_total


class TotalTimeColumn(ProgressColumn):
    """The estimated total runtime, `~ H:MM:SS` behind the elapsed time.

    The estimate is `estimate_total_time` with the `workers` field of the
    task, so it is corrected for the runs a pool processes at once.
    """

    def render(self, task: Task) -> Text:
        """Render the estimate of a task."""
        elapsed = task.elapsed
        estimate = (
            None
            if elapsed is None or task.total is None
            else estimate_total_time(
                elapsed=elapsed,
                completed=task.completed,
                total=task.total,
                workers=int(task.fields.get("workers", 1)),
            )
        )
        if estimate is None:
            return Text("~ -:--:-- total", style="progress.remaining")
        return Text(
            f"~ {datetime.timedelta(seconds=round(estimate))} total",
            style="progress.remaining",
        )


@contextmanager
def optimization_progress(
    description: str,
    size: int,
    enabled: bool = True,
    unit: str = "runs",
    workers: int = 1,
) -> Generator[Progress | None]:
    """Show the progress of the optimization runs on the console.

    The bar shows the count, the elapsed time and the estimated total runtime,
    see `TotalTimeColumn`.

    Args:
        description: text in front of the progress bar.
        size: total number of optimization runs.
        enabled: show the progress, a plain context without display if `False`.
        unit: what is counted, behind the count.
        workers: runs which are processed at once, for the estimate.

    Yields:
        The progress with a single task, or `None` if it is disabled.
    """
    if not enabled:
        yield None
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(unit),
        TimeElapsedColumn(),
        TotalTimeColumn(),
        console=console,
        transient=False,
    )
    with progress:
        progress.add_task(description, total=size, workers=workers)
        yield progress


def _advance(progress: Progress | None, advance: int = 1) -> None:
    """Advance the single task of the progress, if there is one."""
    if progress is not None and progress.task_ids:
        progress.advance(progress.task_ids[0], advance=advance)


def run_optimization(
    problem: OptimizationProblem,
    settings: FitSettings | None = None,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    seed: int | None = None,
    n_cores: int | None = 1,
    serial: bool = False,
    show_progress: bool = True,
    timeout: float | None = None,
    runs_dir: Path | None = None,
    sampling: SamplingType = SamplingType.UNIFORM,
    provenance: FitProvenance | None = None,
    indices: Sequence[int] | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Run the optimization of the problem.

    The runner executes the given `OptimizationProblem` `size` times, every
    repeat starting from its own sample of the parameters, and returns the
    `OptimizationResult` with the fitted parameters and the settings of the fit.

    Args:
        problem: problem to optimize (picklable); it does not have to be
            initialized, `run_optimization` initializes it to show the
            parameters and their coverage before the runs start.
        settings: settings of the fit, the defaults of `FitSettings` are used
            if none are given.
        size: number of optimizations.
        algorithm: optimization algorithm to use.
        seed: random seed (for sampling of the start values).
        n_cores: number of workers, `None` uses all available cores but one;
            one worker fits without starting a process. The workers optimize
            the runs in parallel, or, for differential evolution with fewer
            runs than workers, the runs run one after another and every run
            evaluates its population on all workers, see `split_cores`. The
            result of a seed does not depend on the number of workers.
        serial: run the optimization in a serial fashion (debugging), whatever
            `n_cores` says.
        show_progress: show the progress of the runs on the console.
        timeout: seconds a single optimization may run, no limit if `None`. A
            run which is out of time keeps the parameters it reached.
        runs_dir: directory the single runs are written to while the fit runs,
            so a fit which is interrupted or crashes leaves the runs which
            finished; they are read back with
            `OptimizationResult.from_directory`.
        sampling: sampling of the start values, see `sbmlsim.fit.sampling`.
        provenance: what the fit runs with, see `sbmlsim.fit.provenance`; it is
            stored with every run and with the result.
        indices: the repeats which run, all `size` repeats if `None`. The start
            values and the seeds are drawn for all `size` repeats and every
            given repeat keeps its index, i.e. its file `<opid>_run_<k>.json`,
            so the repeats of pieces of a fit are the repeats of the whole fit
            and `OptimizationResult.from_directory` merges them.
        kwargs: additional arguments for the optimizer, i.e., for
            `scipy.optimize.least_squares` or
            `scipy.optimize.differential_evolution`, e.g. xtol.

    Returns:
        OptimizationResult with the fits of the repeats. A repeat which failed
        is part of the result and carries its message. A fit which is
        interrupted (Ctrl-C) after some repeats finished returns these.

    Raises:
        ValueError: if a bound or a start value does not suit the scale of
            its parameter or the algorithm, if every worker of a parallel
            fit failed, or if the indices are invalid, see `run_indices`.
        TypeError: if `kwargs` has an argument which the optimizer of the
            algorithm does not accept, e.g. one of the other optimizer.
        KeyboardInterrupt: if the fit is interrupted (Ctrl-C) before a repeat
            finished, there is no result to keep.
    """
    if settings is None:
        settings = FitSettings()
    # checked before anything runs, the pieces of a fit are its repeats
    selected = run_indices(size, indices)

    display.section("Optimization", icon=display.ICON_OPTIMIZATION)
    # initialized here rather than left to the serial/parallel helpers, so
    # that the parameters and their coverage are shown, and a versioned
    # parameter which covers no simulation is warned about, before the
    # workers of a parallel fit start; `initialize` is a no-op when the
    # helpers initialize the same problem with the same settings again
    problem.initialize(settings)
    # the bounds the algorithm needs are checked once, not in every run
    problem._validate_parameters(algorithm)
    # an argument which the optimizer does not accept would fail every repeat
    problem.check_optimizer_arguments(algorithm, kwargs)
    display.print_parameters(
        problem.parameters,
        coverage=(
            problem.parameter_mapping.coverage()
            if problem.parameter_mapping is not None
            else None
        ),
        hooks=hook_summaries(problem.hybridizations),
    )

    # a worker without a repeat only costs the start of a process, and one
    # worker is a serial fit without the start of a process
    workers, population_workers = split_cores(
        algorithm=algorithm,
        size=len(selected),
        n_cores=1 if serial else resolve_n_cores(n_cores),
        kwargs=kwargs,
    )
    if max(workers, population_workers) > 1 and parallel.in_worker():
        # a worker which runs the fit again starts workers of its own
        raise RuntimeError(
            f"'{problem.opid}': a worker process started a parallel fit, "
            f"i.e., the script ran again when it was imported. "
            f"{parallel.GUARD_MESSAGE}"
        )
    opt_result: OptimizationResult
    if population_workers > 1:
        display.key_values(
            {
                "runs": _runs_label(size, selected),
                "workers": f"{population_workers} (population of a run)",
            }
        )
        with (
            optimization_progress(
                "optimizing", len(selected), show_progress
            ) as progress,
            worker_pool(problem, settings, population_workers) as pool,
        ):
            opt_result = _run_optimization_serial(
                problem=problem,
                settings=settings,
                size=size,
                algorithm=algorithm,
                seed=seed,
                sampling=sampling,
                timeout=timeout,
                runs_dir=runs_dir,
                provenance=provenance,
                on_progress=lambda: _advance(progress),
                population=PopulationPool(pool),
                indices=selected,
                **kwargs,
            )
    elif workers <= 1:
        display.key_values(
            {"runs": _runs_label(size, selected), "workers": "1 (serial)"}
        )
        with optimization_progress(
            "optimizing", len(selected), show_progress
        ) as progress:
            opt_result = _run_optimization_serial(
                problem=problem,
                settings=settings,
                size=size,
                algorithm=algorithm,
                seed=seed,
                sampling=sampling,
                timeout=timeout,
                runs_dir=runs_dir,
                provenance=provenance,
                on_progress=lambda: _advance(progress),
                indices=selected,
                **kwargs,
            )
    else:
        display.key_values({"runs": _runs_label(size, selected), "workers": workers})
        opt_result = _run_optimization_parallel(
            problem=problem,
            settings=settings,
            size=size,
            algorithm=algorithm,
            seed=seed,
            n_cores=workers,
            show_progress=show_progress,
            sampling=sampling,
            timeout=timeout,
            runs_dir=runs_dir,
            provenance=provenance,
            indices=selected,
            **kwargs,
        )

    _print_summary(opt_result)
    return opt_result


def _runs_label(size: int, selected: Sequence[int]) -> str | int:
    """Get the runs of a fit for the console, the given ones of a piece."""
    if len(selected) == size:
        return size
    return f"{len(selected)} of {size} ({', '.join(str(k) for k in selected)})"


def _print_summary(opt_result: OptimizationResult) -> None:
    """Print the outcome of the optimization on the console."""
    successful = sum(1 for fit in opt_result.fits if fit.success)
    style = "green" if successful == opt_result.size else "orange3"
    info: dict[str, Any] = {
        "converged": f"[{style}]{successful}/{opt_result.size} runs[/{style}]"
    }
    if opt_result.size:
        info["best cost"] = f"{opt_result.df_fits.cost.iloc[0]:.6g}"
        # the sum over the runs, which is more than the wall time in parallel
        info["optimizer time"] = f"{opt_result.df_fits.duration.sum():.1f} s"
    display.key_values(info)


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
        PoolError: if the worker could not initialize the problem, with the
            error of the initialization; a `RuntimeError`.
    """
    initialized = parallel.worker_cache(
        ("fit", token), lambda: _initialized(problem, settings)
    )
    if isinstance(initialized, str):
        raise PoolError(f"the worker could not initialize the problem: {initialized}")
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


def _worker_cost(
    token: str,
    problem: OptimizationProblem,
    settings: FitSettings,
    task: dict[str, Any],
) -> float:
    """Evaluate the cost of a member of a population in a worker process.

    Args:
        token: identifies the fit, see `FitPool`.
        problem: the problem of the fit.
        settings: the settings of the fit.
        task: the member `x`, in the space the optimizer searches.

    Returns:
        The cost of the member, see `OptimizationProblem.evaluate_cost`.

    Raises:
        PoolError: if the worker could not initialize the problem.
    """
    return worker_problem(token, problem, settings).evaluate_cost(task["x"])


def _worker_warm(
    token: str,
    problem: OptimizationProblem,
    settings: FitSettings,
    task: dict[str, Any],
) -> None:
    """Initialize the problem of a fit in a worker process, see `PopulationPool`.

    Args:
        token: identifies the fit, see `FitPool`.
        problem: the problem of the fit.
        settings: the settings of the fit.
        task: nothing, one task per worker.

    Raises:
        PoolError: if the worker could not initialize the problem.
    """
    worker_problem(token, problem, settings)


class WorkerDied(PoolError):
    """A task failed because its worker died while it ran, see `FitPool.run`.

    Attributes:
        alone: whether the task ran alone on the pool when its worker died, so
            it killed the worker itself; otherwise it was one of the tasks in
            flight at a crash after the restarts were spent, and which of them
            killed the worker is not known.
    """

    def __init__(self, message: str, alone: bool = False) -> None:
        """Initialize the failure of a task.

        Args:
            message: what happened.
            alone: whether the task ran alone on the pool.
        """
        super().__init__(message)
        self.alone = alone


#: seconds added to the timeout of a fit before its pending repeats are given up
START_GRACE = 60.0

#: the number of restarts of the pool of a run after which the tasks which were
#: running at a crash are no longer run again alone to find the one which killed
#: its worker: they fail at once. `FitPool.restarts` counts them over the calls
#: of `FitPool.run`, i.e. over a fit of runs as tasks or a profile; the
#: population of differential evolution counts them per generation, see
#: `PopulationPool`. A worker which dies (a crash in the integrator,
#: the kernel killing it for memory) breaks the whole pool and with it every task
#: in flight. The pool is still started again, so the tasks which did not run
#: complete, and every crash after the cap fails the tasks which were running.
#: A pool which is found broken with no task running is started again as well,
#: but after this many such restarts in a row without a result the run ends and
#: the tasks which did not run fail.
MAX_POOL_RESTARTS = 3


@dataclass
class FitPool:
    """The workers of a fit: a pool and the problem its tasks run on.

    Attributes:
        executor: the pool, see `sbmlsim.parallel.start_pool`; replaced when
            the pool broke and was started again, see `run`.
        token: identifies the fit in the caches of the workers.
        problem: the problem, pickled into every task as its definition.
        settings: the settings the workers initialize the problem with.
        n_cores: the number of workers.
        preload: the modules the forkserver imports, see `_preload`.
        restarts: how often the pool was started again after a crash, for the
            cap `MAX_POOL_RESTARTS`.
        stopped: whether the workers are stopped, e.g. after a crash with no
            task left or by `stop`; the next `run` starts new ones.
    """

    executor: ProcessPoolExecutor
    token: str
    problem: OptimizationProblem
    settings: FitSettings
    n_cores: int = 1
    preload: Sequence[str] = ()
    restarts: int = 0
    stopped: bool = False

    def stop(self) -> None:
        """Stop the workers with the tasks they run, the next `run` starts new ones."""
        parallel.stop(self.executor)
        self.stopped = True

    def start(self) -> None:
        """Start new workers, if they are stopped; no restart after a crash.

        Raises:
            RuntimeError: if the workers do not start, see
                `sbmlsim.parallel.start_pool`.
        """
        if self.stopped:
            self.executor = parallel.start_pool(self.n_cores, preload=self.preload)
            self.stopped = False

    def submit(self, function: Callable[..., Any], task: dict[str, Any]) -> Future[Any]:
        """Run `function(token, problem, settings, task)` in a worker."""
        return self.executor.submit(
            function, self.token, self.problem, self.settings, task
        )

    def run(
        self,
        function: Callable[..., Any],
        tasks: Mapping[int, dict[str, Any]],
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> Generator[tuple[int, Any]]:
        """Run the tasks and hand out their results as they finish.

        At most `n_cores` tasks are in flight, the next is submitted when one
        finishes, so a crash implicates only the tasks which were running. A
        worker which dies breaks the whole pool; it is started again and the
        tasks which were running, the suspects, are run again one at a time, so
        a task fails only if it breaks the pool while it runs alone. Then the
        remaining tasks run again at full width. After `MAX_POOL_RESTARTS`
        restarts the tasks which were running at a crash fail without a second
        run. A pool which is found broken when a task is submitted is started
        again the same way, with no suspect if nothing was running; after
        `MAX_POOL_RESTARTS` restarts with no suspect and no result in between
        the run ends there. If the pool cannot be started again, or the run
        ends, the tasks which did not run fail with a `PoolError`. A task which
        fails because its worker died fails with `WorkerDied`, which tells
        whether it ran alone. A crash which leaves no task to run stops the
        pool and marks it `stopped`, so the next call starts new workers
        without counting a crash; stopped workers are started at the start of
        a call.

        Args:
            function: called in the worker as
                `function(token, problem, settings, task)`.
            tasks: the tasks by their key.
            timeout: seconds without any result after which the tasks which are
                running or waiting fail, `START_GRACE` s are added for the
                start; no limit if `None`.
            deadline: the time of `time.monotonic` at which the tasks which did
                not finish fail with a `TimeoutError`, without a grace; the
                tasks which are running are left to the workers, which the
                caller stops. No limit if `None`.

        Yields:
            The key of a task and its result, or the exception which failed it.
        """
        queue = deque(tasks)
        alone: deque[int] = deque()
        running: dict[Future[Any], int] = {}
        solo = False
        # the restarts with no suspect since the last result, the run ends after
        # MAX_POOL_RESTARTS of them; a result shows the pool makes progress
        idle_restarts = 0
        if self.stopped and queue:
            try:
                self.start()
            except RuntimeError as err:
                logger.error(
                    "'%s': the pool could not be started (%s), the tasks %s are "
                    "given up.",
                    self.problem.opid,
                    err,
                    sorted(queue),
                )
                for key in sorted(queue):
                    yield key, PoolError(str(err))
                return
        last = time.monotonic()
        while queue or alone or running:
            if deadline is not None and time.monotonic() >= deadline:
                lost = sorted([*running.values(), *alone, *queue])
                logger.debug(
                    "'%s': the deadline passed, the tasks %s are given up.",
                    self.problem.opid,
                    lost,
                )
                for key in lost:
                    yield key, TimeoutError("the deadline passed")
                return
            submit_broken = False
            try:
                if alone:
                    if not running:
                        key = alone[0]
                        running[self.submit(function, tasks[key])] = key
                        alone.popleft()
                        solo = True
                else:
                    solo = False
                    while queue and len(running) < self.n_cores:
                        key = queue[0]
                        running[self.submit(function, tasks[key])] = key
                        queue.popleft()
            except BrokenProcessPool:
                # the pool broke after the last result: the task did not run
                # and is no suspect, the tasks which do are handled below
                submit_broken = True
            if running:
                remaining = (
                    None
                    if timeout is None
                    else max(1.0, last + timeout + START_GRACE - time.monotonic())
                )
                if deadline is not None:
                    left = max(0.0, deadline - time.monotonic())
                    remaining = left if remaining is None else min(remaining, left)
                done, _ = wait(running, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done and deadline is not None and time.monotonic() >= deadline:
                    # handed out at the start of the next round
                    continue
                if not done:
                    lost = sorted([*running.values(), *alone, *queue])
                    logger.error(
                        "'%s': no result within %s s and %g s more, the tasks %s "
                        "are given up.",
                        self.problem.opid,
                        timeout,
                        START_GRACE,
                        lost,
                    )
                    for key in lost:
                        yield (
                            key,
                            TimeoutError(
                                f"no result within {timeout} s and "
                                f"{START_GRACE:g} s more"
                            ),
                        )
                    return
            else:
                done = set()
            broken = submit_broken
            for future in sorted(done, key=running.__getitem__):
                key = running.pop(future)
                try:
                    result = future.result()
                except BrokenProcessPool:
                    broken = True
                    running[future] = key
                    continue
                except Exception as err:
                    idle_restarts = 0
                    yield key, err
                    continue
                idle_restarts = 0
                last = time.monotonic()
                yield key, result
            if not broken:
                continue
            # the tasks which were in flight are the suspects
            suspects = sorted(running.values())
            running.clear()
            self.stop()
            if not suspects and idle_restarts >= MAX_POOL_RESTARTS:
                lost = sorted([*alone, *queue])
                logger.error(
                    "'%s': the pool keeps breaking, the tasks %s are given up.",
                    self.problem.opid,
                    lost,
                )
                for key in lost:
                    yield key, PoolError("the pool keeps breaking")
                return
            if suspects and (solo or self.restarts >= MAX_POOL_RESTARTS):
                if solo:
                    logger.error(
                        "'%s': the task %s killed its worker and fails.",
                        self.problem.opid,
                        suspects[0],
                    )
                    failure = "the task killed its worker"
                else:
                    logger.error(
                        "'%s': a worker died, the tasks %s were running and fail.",
                        self.problem.opid,
                        suspects,
                    )
                    failure = "a worker died while the task was running"
                for key in suspects:
                    yield key, WorkerDied(failure, alone=solo)
            elif suspects:
                logger.error(
                    "'%s': a worker died; restarting the pool, the tasks %s which "
                    "were running run again one at a time.",
                    self.problem.opid,
                    suspects,
                )
                alone.extend(suspects)
            else:
                idle_restarts += 1
                logger.error(
                    "'%s': a worker died; restarting the pool.", self.problem.opid
                )
            if not (queue or alone):
                # stopped, the next call starts new workers
                return
            self.restarts += 1
            try:
                self.start()
            except RuntimeError as err:
                lost = sorted([*alone, *queue])
                logger.error(
                    "'%s': the pool could not be started again (%s), the tasks %s "
                    "are given up.",
                    self.problem.opid,
                    err,
                    lost,
                )
                for key in lost:
                    yield key, PoolError(str(err))
                return
            last = time.monotonic()


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
    preload = _preload(problem)
    pool = FitPool(
        executor=parallel.start_pool(n_cores, preload=preload),
        token=uuid4().hex,
        problem=problem,
        settings=settings,
        n_cores=n_cores,
        preload=preload,
    )
    try:
        yield pool
    finally:
        # the pool of the fit, which may be another one than the one it started
        pool.stop()


@dataclass
class PopulationPool:
    """Evaluates the population of differential evolution on the workers of a fit.

    It is the `PopulationEvaluator` of a run, see
    `OptimizationProblem.optimize_run`: every member is a task of the pool,
    `_worker_cost`, so at most one member per worker is in flight and a worker
    takes the next member when it is free; the run gets the costs as they
    finish and returns them to scipy in the order of the members. The workers
    initialize the problem once per fit, see `worker_problem`, so the runs of a
    fit share them; `start` initializes them before the budget of a run.

    A member whose worker dies runs again alone on a new pool, see
    `FitPool.run`, and the cap `MAX_POOL_RESTARTS` counts per generation, so
    crashes spread over a fit lose no member. A member which kills its worker
    when it runs alone is a failed simulation, its cost is infinite. Members
    which fail together, once the restarts of a generation are spent, end the
    run with a `PoolError`: which of them killed the worker is not known, and
    an infinite cost for all of them would make the result depend on the
    timing of the crashes. Any other error of a worker ends the run with that
    error, as in a serial run. The members which are not evaluated at the
    deadline of the run are given up. After a deadline, an error or an
    interrupt the workers are stopped, so no member which is still in flight
    holds them, and the next generation, or the next run, starts new ones.

    Attributes:
        pool: the workers of the fit.
        warmed: the pool whose workers initialized the problem, see `start`.
    """

    pool: FitPool
    warmed: ProcessPoolExecutor | None = None

    def start(self) -> None:
        """Start the workers and initialize the problem in every one of them.

        One task per worker initializes the problem, which the members of the
        run then find initialized; it runs before the budget of a run, see
        `OptimizationProblem._optimize_single`. A worker which takes a second
        task before another started initializes the problem with its first
        member instead. A worker which fails to initialize the problem fails
        the members of the run with that error, so the failure is left to them.

        Raises:
            RuntimeError: if the workers do not start.
        """
        self.pool.start()
        if self.warmed is self.pool.executor:
            return
        for _ in self.pool.run(_worker_warm, {k: {} for k in range(self.pool.n_cores)}):
            pass
        self.warmed = self.pool.executor

    def __call__(
        self, members: Sequence[np.ndarray], deadline: float | None
    ) -> Iterator[tuple[int, float]]:
        """Evaluate the cost of every member on the workers.

        Args:
            members: the parameter vectors in the space the optimizer searches.
            deadline: the time of `time.monotonic` at which the members which
                are not evaluated are given up, no limit if `None`.

        Yields:
            The index of a member and its cost, as they finish.

        Raises:
            FitTimeout: if the deadline passed before every member was evaluated.
            PoolError: if the workers failed, see the class.
        """
        # the cap of the restarts counts per generation
        self.pool.restarts = 0
        outcomes = self.pool.run(
            _worker_cost,
            {k: {"x": member, "member": k} for k, member in enumerate(members)},
            deadline=deadline,
        )
        timed_out = finished = False
        try:
            for k, outcome in outcomes:
                if isinstance(outcome, TimeoutError):
                    timed_out = True
                    break
                if isinstance(outcome, WorkerDied) and outcome.alone:
                    logger.error(
                        "'%s': the member %s of the population killed its worker, "
                        "its cost is infinite.",
                        self.pool.problem.opid,
                        k,
                    )
                    yield k, float("inf")
                    continue
                if isinstance(outcome, WorkerDied):
                    raise PoolError(
                        f"the member {k} of the population failed with the "
                        f"members in flight after {MAX_POOL_RESTARTS} restarts "
                        f"of the pool in one generation: {outcome}"
                    )
                if isinstance(outcome, Exception):
                    raise outcome
                yield k, float(outcome)
            else:
                finished = True
        finally:
            outcomes.close()
            if not finished:
                # the members still in flight, after a deadline, an error or an
                # interrupt, would hold the workers of the next generation
                self.pool.stop()
        if timed_out:
            raise FitTimeout(
                f"'{self.pool.problem.opid}': the optimization did not finish in "
                f"its budget."
            )


def _run_optimization_parallel(
    problem: OptimizationProblem,
    settings: FitSettings,
    size: int,
    algorithm: OptimizationAlgorithmType,
    seed: int | None,
    n_cores: int,
    show_progress: bool,
    sampling: SamplingType = SamplingType.UNIFORM,
    timeout: float | None = None,
    runs_dir: Path | None = None,
    provenance: FitProvenance | None = None,
    indices: Sequence[int] | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Run the optimizations in a pool of worker processes.

    Every repeat is a task of the pool, which hands the next repeat to the
    worker which is free, so that repeats of different duration do not leave
    workers idle. The repeats are collected as they finish and ordered by their
    index, and the start values are created here and not in the workers, so a
    fit with a seed gives the same result for any number of workers. Only the
    repeats of `indices` are tasks, all `size` repeats if `None`.
    """
    selected = run_indices(size, indices)
    starts = problem.start_values(
        size=size, algorithm=algorithm, sampling=sampling, seed=seed
    )
    seeds = problem.run_seeds(size=size, algorithm=algorithm, seed=seed)
    tasks: list[dict[str, Any]] = [
        {
            "run": k,
            "size": size,
            "x0": starts[k],
            "run_seed": seeds[k],
            "algorithm": algorithm,
            "timeout": timeout,
            **kwargs,
        }
        for k in selected
    ]

    failures: list[str] = []
    interrupted = False

    collected: dict[int, tuple[OptimizeResult, list[float]]] = {}
    with (
        optimization_progress(
            "optimizing", len(selected), show_progress, workers=n_cores
        ) as progress,
        worker_pool(problem, settings, n_cores) as pool,
    ):
        try:
            # at most one repeat per worker is in flight, see `FitPool.run`
            for k, outcome in pool.run(
                _worker_run, {task["run"]: task for task in tasks}, timeout
            ):
                if isinstance(outcome, Exception):
                    message = f"repeat {k}: {type(outcome).__name__}: {outcome}"
                    failures.append(message)
                    logger.debug("'%s': %s", problem.opid, message)
                    continue
                _, fit, trajectory = outcome
                _store_run(
                    problem=problem,
                    settings=settings,
                    runs_dir=runs_dir,
                    fit=fit,
                    trajectory=trajectory,
                    sid=f"{problem.opid}_run_{k}",
                    provenance=provenance,
                    run=k,
                )
                _advance(progress)
                collected[k] = (fit, trajectory)
        except KeyboardInterrupt:
            interrupted = True
            if collected:
                logger.warning(
                    "'%s': the fit was interrupted, it keeps the %s repeats "
                    "which finished.",
                    problem.opid,
                    len(collected),
                )

    fits = [collected[k][0] for k in sorted(collected)]
    trajectories = [collected[k][1] for k in sorted(collected)]

    if failures:
        logger.warning(
            "'%s': %s of %s repeats were lost, the fit continues with the others: %s",
            problem.opid,
            len(failures),
            len(selected),
            failures,
        )
    if not fits:
        if interrupted:
            # nothing to keep, like the serial fit the interrupt is not an error
            raise KeyboardInterrupt
        stored = f" The runs which finished are in '{runs_dir}'." if runs_dir else ""
        raise ValueError(
            f"'{problem.opid}': every repeat failed, there is no result.{stored} "
            f"{failures}"
        )
    result = OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        trajectories=trajectories,
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
        units=problem.parameter_units,
        provenance=provenance,
        prior_correlation=problem.prior_correlation,
    )
    result.interrupted = interrupted
    return result


def _store_run(
    problem: OptimizationProblem,
    settings: FitSettings,
    runs_dir: Path | None,
    fit: OptimizeResult,
    trajectory: list[float],
    sid: str,
    provenance: FitProvenance | None = None,
    run: int | None = None,
) -> None:
    """Store a finished repeat, which must never end the fit.

    Args:
        problem: problem of the fit.
        settings: settings of the fit.
        runs_dir: directory of the repeats, nothing is stored if `None`.
        fit: result of the repeat.
        trajectory: cost of every step of the repeat.
        sid: id of the repeat, the name of its file.
        provenance: what the fit runs with.
        run: index of the repeat, reported as the event `fit.repeat`.
    """
    emit("fit.repeat", opid=problem.opid, run=run)
    if runs_dir is None:
        return
    try:
        OptimizationResult.write_run(
            directory=Path(runs_dir),
            parameters=problem.parameters,
            fit=fit,
            trajectory=trajectory,
            sid=sid,
            opid=problem.opid,
            settings=settings,
            units=problem.parameter_units,
            provenance=provenance,
            prior_correlation=problem.prior_correlation,
        )
    except Exception as err:
        logger.error(
            "'%s': the run '%s' could not be stored: %s: %s",
            problem.opid,
            sid,
            type(err).__name__,
            err,
        )


def _run_optimization_serial(
    problem: OptimizationProblem,
    settings: FitSettings,
    size: int = 5,
    algorithm: OptimizationAlgorithmType = OptimizationAlgorithmType.LEAST_SQUARE,
    seed: int | None = None,
    timeout: float | None = None,
    runs_dir: Path | None = None,
    on_progress: Callable[[], None] | None = None,
    run_prefix: str = "run",
    provenance: FitProvenance | None = None,
    population: PopulationPool | None = None,
    indices: Sequence[int] | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Run the given optimization problem in a serial fashion.

    This function should not be called directly, `run_optimization` executes the
    optimizations. See `run_optimization` for the arguments; `population`
    evaluates the population of differential evolution on the workers of the
    fit, the runs run one after another; only the runs of `indices` run, all
    `size` runs if `None`.
    """
    # initialize problem, which resolves the data and calculates the weights
    problem.initialize(settings)

    # collected as the repeats finish, so that an interrupt keeps them
    fits: list[OptimizeResult] = []
    trajectories: list[list[float]] = []

    def on_run_finished(k: int, fit: OptimizeResult, trajectory: list[float]) -> None:
        """Keep the finished run, store it and report the progress."""
        fits.append(fit)
        trajectories.append(trajectory)
        _store_run(
            problem=problem,
            settings=settings,
            runs_dir=runs_dir,
            fit=fit,
            trajectory=trajectory,
            sid=f"{problem.opid}_{run_prefix}_{k}",
            provenance=provenance,
            run=k,
        )
        if on_progress is not None:
            on_progress()

    interrupted = False
    try:
        problem.optimize(
            size=size,
            seed=seed,
            algorithm=algorithm,
            timeout=timeout,
            on_run_finished=on_run_finished,
            population=population,
            indices=indices,
            **kwargs,
        )
    except KeyboardInterrupt:
        # like the pool, which keeps the repeats which came back
        if not fits:
            raise
        interrupted = True
        logger.warning(
            "'%s': the fit was interrupted, it keeps the %s repeats which finished.",
            problem.opid,
            len(fits),
        )

    result = OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        trajectories=trajectories,
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
        units=problem.parameter_units,
        provenance=provenance,
        prior_correlation=problem.prior_correlation,
    )
    result.interrupted = interrupted
    return result
