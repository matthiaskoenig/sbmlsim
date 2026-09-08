"""Module for running parameter optimizations.

The optimization runs either serial or in parallel. The parallel optimization
uses multiprocessing, i.e., the runner starts one worker process per core and
every worker runs a part of the repeats of the problem.

The `OptimizationProblem` is pickled and sent to the workers, so it must be
picklable: it is initialized in the worker, not before. The workers report every
finished run through a queue, which drives the progress display of the runner.

The runner only optimizes. Its result carries the fitted parameters and the
settings of the fit, and `sbmlsim.fit.report.FitReport` turns them into figures
and reports, see `sbmlsim.fit.parameters`.
"""

import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
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
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, OptimizationAlgorithmType
from sbmlsim.fit.result import OptimizationResult
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
        n_cores = resolve_n_cores(n_cores)
        if size < n_cores:
            logger.warning(
                "Less optimizations then cores '%s < %s', running '%s' optimizations.",
                size,
                n_cores,
                n_cores,
            )
            size = n_cores
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


def _run_optimization_parallel(
    problem: OptimizationProblem,
    settings: FitSettings,
    size: int,
    algorithm: OptimizationAlgorithmType,
    seed: int | None,
    n_cores: int,
    show_progress: bool,
    runs_dir: Path | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Run the optimizations in a pool of worker processes.

    The runs are distributed over the workers, which report every finished run
    through a managed queue so that the progress can be shown while they work.
    """
    # distribute the repeats over the workers
    sizes = [len(c) for c in np.array_split(range(size), n_cores)]

    # every worker needs its own seed to get different start values
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(n_cores)]

    with multiprocessing.Manager() as manager:
        queue: Queue = manager.Queue()
        args_list = [
            {
                "problem": problem,
                "settings": settings,
                "size": sizes[k],
                "algorithm": algorithm,
                "seed": seeds[k],
                "queue": queue,
                "worker_index": k,
                "runs_dir": runs_dir,
                **kwargs,
            }
            for k in range(n_cores)
        ]

        opt_results: list[OptimizationResult] = []
        failures: list[str] = []
        with (
            optimization_progress("optimizing", size, show_progress) as progress,
            multiprocessing.Pool(processes=n_cores) as pool,
        ):
            # one task per worker, so that a worker which dies does not take
            # the results of the other workers with it
            async_results = [pool.apply_async(worker, (args,)) for args in args_list]
            while not all(result.ready() for result in async_results):
                _advance(progress, _drain(queue))
                async_results[0].wait(timeout=0.2)
            _advance(progress, _drain(queue))

            for k, async_result in enumerate(async_results):
                try:
                    opt_results.append(async_result.get())
                except Exception as err:
                    message = f"worker {k}: {type(err).__name__}: {err}"
                    failures.append(message)
                    logger.error("'%s': %s", problem.opid, message)

    if failures:
        logger.warning(
            "'%s': %s of %s workers failed, the fit continues with the results of "
            "the others.",
            problem.opid,
            len(failures),
            n_cores,
        )
    if not opt_results:
        stored = f" The runs which finished are in '{runs_dir}'." if runs_dir else ""
        raise ValueError(
            f"'{problem.opid}': every worker failed, there is no result.{stored} "
            f"{failures}"
        )
    return OptimizationResult.combine(opt_results)


def _drain(queue: Queue) -> int:
    """Get the number of runs the workers finished since the last call."""
    finished = 0
    while True:
        try:
            queue.get_nowait()
        except Empty:
            return finished
        finished += 1


def worker(kwargs: dict[str, Any]) -> OptimizationResult:
    """Run a part of the optimizations in a worker process.

    Every worker initializes the same problem and would report the same
    messages about the data, once per core. Only errors of a worker are shown,
    the runner reports the problem itself.
    """
    logging.getLogger(PACKAGE_LOGGER).setLevel(logging.ERROR)
    logger.debug("worker <%s> running optimization ...", os.getpid())
    queue: Queue | None = kwargs.pop("queue", None)
    worker_index: int = kwargs.pop("worker_index", 0)

    def on_progress() -> None:
        """Report a finished run to the runner."""
        if queue is not None:
            queue.put(1)

    return _run_optimization_serial(
        on_progress=on_progress, run_prefix=f"w{worker_index}", **kwargs
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
        if runs_dir is not None:
            try:
                OptimizationResult.write_run(
                    directory=Path(runs_dir),
                    parameters=problem.parameters,
                    fit=fit,
                    trajectory=trajectory,
                    sid=f"{problem.opid}_{run_prefix}_{k}",
                    opid=problem.opid,
                    settings=settings,
                )
            except Exception as err:
                # storing a run must never end the fit
                logger.error(
                    "'%s': the run '%s' could not be stored: %s: %s",
                    problem.opid,
                    k,
                    type(err).__name__,
                    err,
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
