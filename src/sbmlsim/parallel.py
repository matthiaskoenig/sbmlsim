"""The process pools of sbmlsim.

Every pool of the package comes from here:

- `process_context` is the start method of every pool, never `fork` of a
  process which runs threads;
- `pool(n)` is a pool kept per process and number of workers, so that the
  start of the workers (the imports, roadrunner) is paid once per process and
  not once per run; `start_pool(n)` is a new pool which the caller stops, e.g.
  the pool of a fit, whose workers keep the initialized problem;
- `resolve_workers` turns `n_workers` into the number of processes;
- `worker_cache` keeps objects in a worker, e.g. the models of a scan or the
  initialized problem of a fit, and builds one only when its key is new.

A pool starts worker processes which import the main module again (the start
methods `forkserver` and `spawn`), so a script which starts a pool must do it
behind the guard `if __name__ == "__main__":`. Without it the workers die while
they start and the parent reports the guard, see `GUARD_MESSAGE`. A worker
itself never starts a pool, a task which runs in a worker runs serially.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import BaseContext
from typing import cast

logger = logging.getLogger(__name__)

#: the smallest number of tasks which `resolve_workers` runs in a pool for
#: `n_workers=None`; below it the start of the workers costs more than it saves
POOL_THRESHOLD: int = 64

#: seconds the workers of a new pool may take to start
WORKER_STARTUP_TIMEOUT: float = 300.0

#: the most objects a worker keeps, see `worker_cache`
WORKER_CACHE_SIZE: int = 16

#: the modules the forkserver imports once for all workers
PRELOAD: tuple[str, ...] = ("sbmlsim.simulator.executor",)

#: what to do about workers which do not start
GUARD_MESSAGE = (
    "A parallel run starts worker processes which import the script again, so "
    "the run must be behind a guard:\n\n"
    '    if __name__ == "__main__":\n        main()\n\n'
    "Use one worker to run without worker processes."
)

#: the pools of the process by their number of workers, see `pool`
_POOLS: dict[int, ProcessPoolExecutor] = {}

#: the objects of a worker process, see `worker_cache`
_CACHE: OrderedDict[Hashable, object] = OrderedDict()


def process_context() -> BaseContext:
    """Get the multiprocessing context a pool of sbmlsim is created from.

    A start method set with `multiprocessing.set_start_method` is used, else
    the default of the platform, except `fork`: the process of a simulation
    runs the threads of roadrunner and of the linear algebra, and a fork of a
    process with threads may deadlock in the child. Python 3.14 made
    `forkserver` the default on linux for this reason, a pool takes it on
    python 3.13 as well, where it warns about the fork of a process with
    threads with a `DeprecationWarning`. A start method which equals the
    default of the platform counts as no choice, so on python 3.13 on linux a
    `fork` which is set is replaced by `forkserver` as well, see below.

    Every pool of sbmlsim comes from this module, see `pool` and `start_pool`,
    and the start method of the process is not fixed by the call
    (`get_context()` without a method would), because a start method which is
    set is read as one the user chose.

    Python 3.13 fixes the start method of the process to the default of the
    platform when a process is started by `spawn` or `forkserver`, python 3.14
    does not. On linux the first pool of this context, or of any other code
    which does not fork, would make every later call read `fork` as the choice
    of the user. A start method which equals the default of the platform is
    therefore no choice, so on python 3.13 on linux `fork` cannot be asked for,
    while it can wherever `fork` is not the default (python 3.14, macOS).

    Returns:
        The context.
    """
    method = multiprocessing.get_start_method(allow_none=True)
    # the first of the supported methods is the default of the platform
    methods = multiprocessing.get_all_start_methods()
    if method is None or method == methods[0]:
        method = methods[0]
        if method == "fork" and "forkserver" in methods:
            method = "forkserver"
    return multiprocessing.get_context(method)


def in_worker() -> bool:
    """Check whether this process is a child process of multiprocessing."""
    return multiprocessing.parent_process() is not None


def check_workers(n_workers: int | None) -> None:
    """Check a number of workers.

    Args:
        n_workers: `None` or the number of processes.

    Raises:
        ValueError: if `n_workers` is less than 1.
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"The number of workers must be at least 1, not {n_workers}.")


def resolve_workers(n_workers: int | None, n_tasks: int) -> int:
    """Get the number of processes of a run.

    Args:
        n_workers: `1` runs serially in the calling process, a number is taken
            as given, `None` is the number of CPUs for `POOL_THRESHOLD` tasks
            or more and serial below, and serial in a worker process.
        n_tasks: the number of tasks of the run, e.g. the points of a scan.

    Returns:
        The number of processes, `1` for a serial run.

    Raises:
        ValueError: if `n_workers` is less than 1.
    """
    check_workers(n_workers)
    if n_workers is not None:
        return n_workers
    if in_worker() or n_tasks < POOL_THRESHOLD:
        return 1
    return max(1, os.process_cpu_count() or 1)


def _context(preload: Sequence[str]) -> BaseContext:
    """Get the context of a pool, the forkserver preloads the modules.

    The forkserver imports the modules once and the workers inherit them; the
    main module is never preloaded, so a worker imports it and a script without
    the guard is found, see the module.
    """
    context = process_context()
    if context.get_start_method() == "forkserver":
        context.set_forkserver_preload(sorted({*PRELOAD, *preload}))
    return context


def _alive() -> int:
    """Probe of a worker, which answers when it started."""
    return os.getpid()


def start_pool(n_workers: int, preload: Sequence[str] = ()) -> ProcessPoolExecutor:
    """Start a pool whose workers answer, see `pool` for one which is kept.

    Args:
        n_workers: the number of worker processes.
        preload: modules the forkserver imports once for all workers; it
            takes effect only when the forkserver starts for the first time in
            this process.

    Returns:
        The pool, the caller stops it with `stop`.

    Raises:
        RuntimeError: in a worker process (no nested pools), or if the workers die while they
            start or do not start within `WORKER_STARTUP_TIMEOUT`; both are
            what a script without the guard does, see `GUARD_MESSAGE`.
    """
    if in_worker():
        raise RuntimeError(
            "A pool was started inside a worker process; run the inner part "
            "serially (n_workers=1)."
        )
    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=_context(preload))
    logger.debug("Starting a pool of %s workers", n_workers)
    try:
        executor.submit(_alive).result(timeout=WORKER_STARTUP_TIMEOUT)
    except BrokenProcessPool as err:
        stop(executor)
        raise RuntimeError(
            f"The workers died while they started. {GUARD_MESSAGE}"
        ) from err
    except TimeoutError as err:
        stop(executor)
        raise RuntimeError(
            f"No worker started within {WORKER_STARTUP_TIMEOUT:.0f} s. {GUARD_MESSAGE}"
        ) from err
    except BaseException:
        # e.g. Ctrl-C while the workers start: nobody else can stop the pool
        stop(executor)
        raise
    return executor


def _dead(executor: ProcessPoolExecutor) -> bool:
    """Check whether a pool broke or was shut down and takes no more tasks.

    Reads attributes of the executor which the standard library does not
    document; they are read directly, so a python which renames them fails
    here and not by handing out a dead pool.
    """
    return bool(executor._broken or executor._shutdown_thread)


def pool(n_workers: int) -> ProcessPoolExecutor:
    """Get the pool of the process with a number of workers.

    The pool is kept, so the workers start once per process; a pool which
    broke, e.g. because a worker died, is replaced. Every pool is stopped when
    the process ends, see `shutdown`.

    Args:
        n_workers: the number of worker processes.

    Returns:
        The pool.

    Raises:
        RuntimeError: see `start_pool`.
    """
    executor = _POOLS.get(n_workers)
    if executor is not None and not _dead(executor):
        return executor
    if executor is not None:
        logger.debug("Replacing the pool of %s workers", n_workers)
        stop(executor)
    executor = start_pool(n_workers)
    _POOLS[n_workers] = executor
    return executor


def stop(executor: ProcessPoolExecutor) -> None:
    """Stop a pool: the pending tasks are cancelled and the workers end.

    The workers are terminated, so a pool whose workers still run, e.g. after
    Ctrl-C or a timeout, stops at once. A kept pool is dropped, the next
    `pool` starts a new one.
    """
    for n_workers, kept in list(_POOLS.items()):
        if kept is executor:
            del _POOLS[n_workers]
    logger.debug("Stopping a pool")
    processes = list((executor._processes or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def shutdown() -> None:
    """Stop every kept pool of the process."""
    for executor in list(_POOLS.values()):
        stop(executor)


atexit.register(shutdown)


def worker_cache[T](key: Hashable, factory: Callable[[], T]) -> T:
    """Get an object of the worker process, built once per key.

    The newest `WORKER_CACHE_SIZE` objects are kept.

    Args:
        key: identifies the object, e.g. a model with the settings of its
            integrator.
        factory: builds the object when the key is new.

    Returns:
        The object of the key.
    """
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return cast(T, _CACHE[key])
    value = factory()
    _CACHE[key] = value
    while len(_CACHE) > WORKER_CACHE_SIZE:
        _CACHE.popitem(last=False)
    return value
