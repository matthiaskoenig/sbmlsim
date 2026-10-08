"""Module for running parameter optimizations.

The optimization runs either serial or in parallel. The parallel optimization
runs in a pool of `sbmlsim.parallel` which belongs to the fit (`worker_pool`):
one worker process per core, and every repeat of the fit goes to the worker
which is free. The pool is stopped after the fit, so the workers do not keep the
initialized problem, and it is started again, a few times, when a worker dies.

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
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

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

from sbmlsim import parallel
from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.derived import hook_summaries
from sbmlsim.fit.optimization import OptimizationProblem, RuntimeErrorOptimizeResult
from sbmlsim.fit.options import FitSettings, OptimizationAlgorithmType
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
    cpu_count = os.process_cpu_count() or 1
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
            one worker fits without starting a process.
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
        kwargs: additional arguments for the optimizer, i.e., for
            `scipy.optimize.least_squares` or
            `scipy.optimize.differential_evolution`, e.g. xtol.

    Returns:
        OptimizationResult with the fits of all repeats. A repeat which failed
        is part of the result and carries its message. A fit which is
        interrupted (Ctrl-C) after some repeats finished returns these.

    Raises:
        ValueError: if a bound or a start value does not suit the scale of
            its parameter or the algorithm, or if every worker of a parallel
            fit failed.
        TypeError: if `kwargs` has an argument which the optimizer of the
            algorithm does not accept, e.g. one of the other optimizer.
        KeyboardInterrupt: if the fit is interrupted (Ctrl-C) before a repeat
            finished, there is no result to keep.
    """
    if settings is None:
        settings = FitSettings()

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
    workers = 1 if serial else min(resolve_n_cores(n_cores), size)
    opt_result: OptimizationResult
    if workers <= 1:
        display.key_values({"runs": size, "workers": "1 (serial)"})
        with optimization_progress("optimizing", size, show_progress) as progress:
            opt_result = _run_optimization_serial(
                problem=problem,
                settings=settings,
                size=size,
                algorithm=algorithm,
                seed=seed,
                sampling=sampling,
                timeout=timeout,
                runs_dir=runs_dir,
                on_progress=lambda: _advance(progress),
                **kwargs,
            )
    else:
        if parallel.in_worker():
            # a worker which runs the fit again starts workers of its own
            raise RuntimeError(
                f"'{problem.opid}': a worker process started a parallel fit, "
                f"i.e., the script ran again when it was imported. "
                f"{parallel.GUARD_MESSAGE}"
            )
        display.key_values({"runs": size, "workers": workers})
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
            **kwargs,
        )

    _print_summary(opt_result)
    return opt_result


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


#: seconds added to the timeout of a fit before its pending repeats are given up
START_GRACE = 60.0

#: how often the pool of a fit is started again after a worker died; a worker
#: which dies (a crash in the integrator, the kernel killing it for memory) breaks
#: the whole pool, so the repeats which did not finish are submitted again. A
#: repeat which kills its worker every time is given up after this many restarts.
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
        restarts: how often the pool was started again.
    """

    executor: ProcessPoolExecutor
    token: str
    problem: OptimizationProblem
    settings: FitSettings
    n_cores: int = 1
    preload: Sequence[str] = ()
    restarts: int = 0

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
    ) -> Iterator[tuple[int, Any]]:
        """Run the tasks and hand out their results as they finish.

        A worker which dies breaks the whole pool and with it every task which
        did not finish. The pool is started again and these tasks are submitted
        again, at most `MAX_POOL_RESTARTS` times; the tasks which are still
        unfinished after that are failed.

        Args:
            function: called in the worker as
                `function(token, problem, settings, task)`.
            tasks: the tasks by their key.
            timeout: seconds without any result after which the tasks which are
                pending are failed, 60 s are added for the start; no limit if
                `None`.

        Yields:
            The key of a task and its result, or the exception which failed it.
        """
        pending: dict[Future[Any], int] = {
            self.submit(function, task): key for key, task in tasks.items()
        }
        last = time.monotonic()
        while pending:
            remaining = (
                None
                if timeout is None
                else max(1.0, last + timeout + START_GRACE - time.monotonic())
            )
            done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                for key in pending.values():
                    yield (
                        key,
                        TimeoutError(
                            f"no result within {timeout} s and {START_GRACE:g} s more"
                        ),
                    )
                return
            broken = False
            for future in sorted(done, key=pending.__getitem__):
                key = pending.pop(future)
                try:
                    result = future.result()
                except BrokenProcessPool:
                    broken = True
                    pending[future] = key
                    continue
                except Exception as err:
                    yield key, err
                    continue
                last = time.monotonic()
                yield key, result
            if not broken:
                continue
            unfinished = sorted(pending.values())
            if self.restarts >= MAX_POOL_RESTARTS:
                logger.error(
                    "'%s': a worker died again, the pool was restarted %s times; "
                    "the %s unfinished repeats are given up.",
                    self.problem.opid,
                    self.restarts,
                    len(unfinished),
                )
                for key in unfinished:
                    yield key, RuntimeError("the workers died, the pool broke")
                return
            self.restarts += 1
            logger.error(
                "'%s': a worker died; restarting the pool, %s unfinished repeats "
                "are submitted again.",
                self.problem.opid,
                len(unfinished),
            )
            parallel.stop(self.executor)
            self.executor = parallel.start_pool(self.n_cores, preload=self.preload)
            pending = {self.submit(function, tasks[key]): key for key in unfinished}
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
        parallel.stop(pool.executor)


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
    **kwargs: Any,
) -> OptimizationResult:
    """Run the optimizations in a pool of worker processes.

    Every repeat is a task of the pool, which hands the next repeat to the
    worker which is free, so that repeats of different duration do not leave
    workers idle. The repeats are collected as they finish and ordered by their
    index, and the start values are created here and not in the workers, so a
    fit with a seed gives the same result for any number of workers.
    """
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
        for k in range(size)
    ]

    failures: list[str] = []
    interrupted = False

    collected: dict[int, tuple[OptimizeResult, list[float]]] = {}
    with (
        optimization_progress(
            "optimizing", size, show_progress, workers=n_cores
        ) as progress,
        worker_pool(problem, settings, n_cores) as pool,
    ):
        try:
            # one task per repeat, so that a worker which dies loses one repeat
            for k, outcome in pool.run(
                _worker_run, {task["run"]: task for task in tasks}, timeout
            ):
                if isinstance(outcome, Exception):
                    message = f"repeat {k}: {type(outcome).__name__}: {outcome}"
                    failures.append(message)
                    logger.error("'%s': %s", problem.opid, message)
                    continue
                _, fit, trajectory = outcome
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
            "'%s': %s of %s repeats were lost, the fit continues with the others.",
            problem.opid,
            len(failures),
            size,
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
    return OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        trajectories=trajectories,
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
    )


def _store_run(
    problem: OptimizationProblem,
    settings: FitSettings,
    runs_dir: Path | None,
    fit: OptimizeResult,
    trajectory: list[float],
    sid: str,
) -> None:
    """Store a finished repeat, which must never end the fit.

    Args:
        problem: problem of the fit.
        settings: settings of the fit.
        runs_dir: directory of the repeats, nothing is stored if `None`.
        fit: result of the repeat.
        trajectory: cost of every step of the repeat.
        sid: id of the repeat, the name of its file.
    """
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
    **kwargs: Any,
) -> OptimizationResult:
    """Run the given optimization problem in a serial fashion.

    This function should not be called directly, `run_optimization` executes the
    optimizations. See `run_optimization` for the arguments.
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
        )
        if on_progress is not None:
            on_progress()

    try:
        problem.optimize(
            size=size,
            seed=seed,
            algorithm=algorithm,
            timeout=timeout,
            on_run_finished=on_run_finished,
            **kwargs,
        )
    except KeyboardInterrupt:
        # like the pool, which keeps the repeats which came back
        if not fits:
            raise
        logger.warning(
            "'%s': the fit was interrupted, it keeps the %s repeats which finished.",
            problem.opid,
            len(fits),
        )

    return OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        trajectories=trajectories,
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
    )
