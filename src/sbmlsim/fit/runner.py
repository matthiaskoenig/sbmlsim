"""Module for running parameter optimizations.

The optimization runs either serial or in parallel. The parallel optimization
uses multiprocessing, i.e., the runner starts one worker process per core and
hands every repeat of the fit to the worker which is free.

The `OptimizationProblem` is pickled and sent to the workers, so it must be
picklable: every worker initializes it once and runs repeats on it. The start
values are created by the runner, so a fit with a seed gives the same result
for any number of workers.

The runner only optimizes. Its result carries the fitted parameters and the
settings of the fit, and `sbmlsim.fit.report.FitReport` turns them into figures
and reports, see `sbmlsim.fit.parameters`.
"""

import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from multiprocessing.context import BaseContext
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from scipy.optimize import OptimizeResult

from sbmlsim.console import console
from sbmlsim.fit import display
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


@contextmanager
def optimization_progress(
    description: str, size: int, enabled: bool = True
) -> Iterator[Progress | None]:
    """Show the progress of the optimization runs on the console.

    Args:
        description: text in front of the progress bar.
        size: total number of optimization runs.
        enabled: show the progress, a plain context without display if `False`.

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
        TextColumn("runs"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        progress.add_task(description, total=size)
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
    **kwargs: Any,
) -> OptimizationResult:
    """Run the optimization of the problem.

    The runner executes the given `OptimizationProblem` `size` times, every
    repeat starting from its own sample of the parameters, and returns the
    `OptimizationResult` with the fitted parameters and the settings of the fit.

    Args:
        problem: uninitialized problem to optimize (picklable).
        settings: settings of the fit, the defaults of `FitSettings` are used
            if none are given.
        size: number of optimizations.
        algorithm: optimization algorithm to use.
        seed: random seed (for sampling of the start values).
        n_cores: number of workers, `None` uses all available cores but one.
        serial: run the optimization in a serial fashion (debugging).
        show_progress: show the progress of the runs on the console.
        timeout: seconds a single optimization may run, no limit if `None`. A
            run which is out of time keeps the parameters it reached.
        runs_dir: directory the single runs are written to while the fit runs,
            so a fit which is interrupted or crashes leaves the runs which
            finished; they are read back with
            `OptimizationResult.from_directory`.
        kwargs: additional arguments for the optimizer, e.g. xtol.

    Returns:
        OptimizationResult with the fits of all repeats. A repeat which failed
        is part of the result and carries its message.

    Raises:
        ValueError: for the removed parameters `fitting_type` and
            `weighting_local`, or if every worker of a parallel fit failed.
    """
    for deprecated, replacement in [
        ("fitting_type", "fitting_strategy"),
        ("weighting_local", "weighting_points"),
    ]:
        if deprecated in kwargs:
            raise ValueError(
                f"Deprecated parameter '{deprecated}', use '{replacement}' instead."
            )

    if settings is None:
        settings = FitSettings()

    display.section("Optimization", icon=display.ICON_OPTIMIZATION)

    opt_result: OptimizationResult
    if serial:
        display.key_values({"runs": size, "workers": "1 (serial)"})
        with optimization_progress("optimizing", size, show_progress) as progress:
            opt_result = _run_optimization_serial(
                problem=problem,
                settings=settings,
                size=size,
                algorithm=algorithm,
                seed=seed,
                timeout=timeout,
                runs_dir=runs_dir,
                on_progress=lambda: _advance(progress),
                **kwargs,
            )
    else:
        if multiprocessing.parent_process() is not None:
            # a worker which runs the fit again starts workers of its own
            raise RuntimeError(
                f"'{problem.opid}': a worker process started a parallel fit, "
                f"i.e., the script ran again when it was imported. "
                f"{GUARD_MESSAGE}"
            )
        # a worker without a repeat only costs the start of a process
        n_cores = min(resolve_n_cores(n_cores), size)
        display.key_values({"runs": size, "workers": n_cores})
        opt_result = _run_optimization_parallel(
            problem=problem,
            settings=settings,
            size=size,
            algorithm=algorithm,
            seed=seed,
            n_cores=n_cores,
            show_progress=show_progress,
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


#: seconds the workers of a parallel fit may take to start and to initialize the
#: problem. Workers which die while they start, which is what a script without
#: the `if __name__ == "__main__":` guard does, are replaced by the pool over and
#: over, so a fit which has no worker after this time is stopped
WORKER_STARTUP_TIMEOUT: float = 300.0

#: what to do about workers which do not start
GUARD_MESSAGE = (
    "A parallel fit starts worker processes which import the script again, so "
    "the fit must run behind a guard:\n\n"
    '    if __name__ == "__main__":\n        main()\n\n'
    "Use 'serial=True' or 'n_cores=1' to fit without worker processes."
)


#: the problem of the worker process, initialized once by `_worker_initialize`
_WORKER_PROBLEM: OptimizationProblem | None = None
#: why the initialization of the worker failed, `None` if it worked
_WORKER_ERROR: str | None = None


def _worker_initialize(problem: OptimizationProblem, settings: FitSettings) -> None:
    """Initialize the problem of a worker process.

    Every worker resolves the data of the same problem once and runs its share
    of the repeats on it. The workers would report the same messages about the
    data, once per core, so only their errors are shown; the runner reports the
    problem itself.

    An error is stored instead of raised: a pool whose initializer raises
    replaces its workers over and over, which turns a broken problem into a
    fit that does not end.
    """
    global _WORKER_PROBLEM, _WORKER_ERROR
    logging.getLogger(PACKAGE_LOGGER).setLevel(logging.ERROR)
    logger.debug("worker <%s> initializing problem ...", os.getpid())
    try:
        problem.initialize(settings)
    except Exception as err:
        _WORKER_PROBLEM, _WORKER_ERROR = None, f"{type(err).__name__}: {err}"
        return
    _WORKER_PROBLEM, _WORKER_ERROR = problem, None


def _worker_alive() -> bool:
    """Probe of a worker, which answers when it initialized the problem."""
    return _WORKER_PROBLEM is not None


def _worker_run(task: dict[str, Any]) -> tuple[int, OptimizeResult, list[float]]:
    """Run a single optimization in a worker process.

    Args:
        task: index `run` of the repeat and the arguments of `optimize_run`.

    Returns:
        The index of the repeat, its fit and its trajectory.
    """
    run: int = task.pop("run")
    if _WORKER_PROBLEM is None:
        return (
            run,
            RuntimeErrorOptimizeResult(
                x0=task.get("x0"),
                message=f"the worker could not initialize the problem: {_WORKER_ERROR}",
            ),
            [],
        )
    fit, trajectory = _WORKER_PROBLEM.optimize_run(run=run, **task)
    return run, fit, trajectory


def _pool_context(problem: OptimizationProblem) -> BaseContext:
    """Get the multiprocessing context of a fit.

    Under the `forkserver` start method, the default on linux since python
    3.14, every worker imports sbmlsim and the module of the experiments again,
    which costs more than a short optimization. The forkserver imports them once
    and the workers inherit them, which matters for a fit of several problems:
    the forkserver of the context outlives the pool, so only the first pool
    pays the imports.

    Args:
        problem: problem of the fit, its experiments name the modules to import.

    Returns:
        The context the pool is created from.
    """
    ctx = multiprocessing.get_context()
    if ctx.get_start_method() != "forkserver":
        return ctx
    modules = {"sbmlsim.fit.optimization"}
    for fit_experiment in problem.fit_experiments:
        module = getattr(fit_experiment.experiment_class, "__module__", None)
        # the experiments of a script are re-imported, only importable modules
        if module and module != "__main__":
            modules.add(module)
    with suppress(Exception):
        # preloading is an optimization, a module which does not import must
        # not end the fit
        ctx.set_forkserver_preload(sorted(modules))
    return ctx


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
    workers idle. The start values are created here and not in the workers, so
    a fit with a seed gives the same result for any number of workers.
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

    fits: list[OptimizeResult] = []
    trajectories: list[list[float]] = []
    failures: list[str] = []

    # when a repeat came back, i.e., when the workers made progress last
    finished = time.monotonic()

    def on_result(result: tuple[int, OptimizeResult, list[float]]) -> None:
        """Store a repeat as soon as it is done, in the thread of the pool."""
        nonlocal finished
        finished = time.monotonic()
        run, fit, trajectory = result
        _store_run(
            problem=problem,
            settings=settings,
            runs_dir=runs_dir,
            fit=fit,
            trajectory=trajectory,
            sid=f"{problem.opid}_run_{run}",
        )
        _advance(progress)

    with (
        optimization_progress("optimizing", size, show_progress) as progress,
        _worker_pool(problem, settings, n_cores) as pool,
    ):
        # one task per repeat, so that a worker which dies loses one repeat
        async_results = [
            pool.apply_async(_worker_run, (task,), callback=on_result) for task in tasks
        ]
        for k, async_result in enumerate(async_results):
            try:
                # the repeats are collected in the order they were given out,
                # a fit which stopped making progress is a worker which is gone
                remaining = (
                    None
                    if timeout is None
                    else max(1.0, finished + timeout + 60.0 - time.monotonic())
                )
                _, fit, trajectory = async_result.get(timeout=remaining)
            except KeyboardInterrupt:
                logger.warning(
                    "'%s': the fit was interrupted, it keeps the %s repeats which "
                    "finished.",
                    problem.opid,
                    len(fits),
                )
                break
            except Exception as err:
                message = f"repeat {k}: {type(err).__name__}: {err}"
                failures.append(message)
                logger.error("'%s': %s", problem.opid, message)
                continue
            fits.append(fit)
            trajectories.append(trajectory)

    if failures:
        logger.warning(
            "'%s': %s of %s repeats were lost, the fit continues with the others.",
            problem.opid,
            len(failures),
            size,
        )
    if not fits:
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


@contextmanager
def _worker_pool(
    problem: OptimizationProblem, settings: FitSettings, n_cores: int
) -> Iterator[Pool]:
    """Create the pool of workers of a parallel fit.

    Raises:
        RuntimeError: if the workers cannot be started, which is what a script
            without the `if __name__ == "__main__":` guard runs into.
    """
    context = _pool_context(problem)
    try:
        pool = context.Pool(
            processes=n_cores,
            initializer=_worker_initialize,
            initargs=(problem, settings),
        )
    except Exception as err:
        raise RuntimeError(
            f"the workers of the fit could not be started "
            f"({type(err).__name__}: {err}). {GUARD_MESSAGE}"
        ) from err
    try:
        with pool:
            _wait_for_workers(pool)
            yield pool
    finally:
        pool.terminate()


def _wait_for_workers(pool: Pool) -> None:
    """Wait until a worker of the pool has the problem.

    A worker which dies while it starts is replaced by the pool, over and over,
    which is what a script without the `if __name__ == "__main__":` guard does:
    the workers import the script, the script fits again and the fit never ends.
    The workers of a fit are not replaced, so a pool which has none of the
    workers it started and still did not answer is not going to work.

    Args:
        pool: pool of workers of the fit.

    Raises:
        RuntimeError: if no worker started.
    """
    probe = pool.apply_async(_worker_alive)
    started = {process.pid for process in multiprocessing.active_children()}
    deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT
    while not probe.ready():
        alive = {process.pid for process in multiprocessing.active_children()}
        if started and not (started & alive):
            raise RuntimeError(
                f"every worker of the fit died while it started. {GUARD_MESSAGE}"
            )
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"no worker of the fit started within "
                f"{WORKER_STARTUP_TIMEOUT:.0f} s. {GUARD_MESSAGE}"
            )
        probe.wait(timeout=0.5)


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
    if "n_cores" in kwargs:
        # remove parallel arguments
        logger.warning(
            "Parameter 'n_cores' does not have any effect in serial optimization."
        )
        kwargs.pop("n_cores")

    # initialize problem, which resolves the data and calculates the weights
    problem.initialize(settings)

    def on_run_finished(k: int, fit: OptimizeResult, trajectory: list[float]) -> None:
        """Store the finished run and report the progress."""
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

    fits, trajectories = problem.optimize(
        size=size,
        seed=seed,
        algorithm=algorithm,
        timeout=timeout,
        on_run_finished=on_run_finished,
        **kwargs,
    )

    return OptimizationResult(
        parameters=problem.parameters,
        fits=fits,
        trajectories=trajectories,
        sid=problem.opid,
        opid=problem.opid,
        settings=settings,
    )
