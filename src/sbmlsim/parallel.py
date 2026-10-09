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

Ctrl-C belongs to the process of the user: a terminal sends SIGINT to every
process of the job, the workers ignore it and the parent stops a pool whose
run it interrupts, see `start_pool` and `stop`. On POSIX a worker starts with
SIGINT blocked, so Ctrl-C does not reach it while it still imports, see
`_SigintBlocked`.

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
import signal
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import cast, override

if sys.platform != "win32":
    from multiprocessing.context import (
        ForkContext,
        ForkProcess,
        ForkServerContext,
        ForkServerProcess,
        SpawnContext,
        SpawnProcess,
    )

logger = logging.getLogger(__name__)

#: the smallest number of tasks which `resolve_workers` runs in a pool for
#: `n_workers=None`, set by the benchmark `test_the_pool_pays_from_the_threshold`
#: on linux with the start method `forkserver` and 20 CPUs. The start of the
#: pool and its workers is not counted, it is paid once per process (the first
#: pooled scan of 256 points of the repressilator, 0.8 ms per point, takes
#: about 870 ms against 200 ms serially and 30 ms in the next pooled run);
#: the load of the model in every worker is counted, it is paid per model
#: (about 150 ms on 20 workers). With it the pool is as fast as serially at
#: about 200 points and 1.2 times faster at 256; with the model already in the
#: workers it pays from fewer than 32 points. A short script with a single
#: scan of a cheap model can therefore be faster with `n_workers=1`.
POOL_THRESHOLD: int = 256

#: seconds the workers of a new pool may take to start
WORKER_STARTUP_TIMEOUT: float = 300.0

#: seconds `stop` waits for the pool to end its workers
STOP_TIMEOUT: float = 30.0

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


#: the contexts of the pools by their start method, see `_context`
_WORKER_CONTEXTS: dict[str, BaseContext] = {}

if sys.platform != "win32":

    class _SigintBlocked(BaseProcess):
        """A worker process which starts with SIGINT blocked.

        A worker imports the main module and sbmlsim before its initializer
        ignores SIGINT, which takes seconds under `spawn`. Ctrl-C in that time
        would end it with a traceback and break the pool. The signal mask of
        the starting thread is inherited by the process through fork and
        exec, so SIGINT stays pending until `_initialize_worker` ignores it,
        which discards it.
        """

        @override
        def start(self) -> None:
            """Start the process with SIGINT blocked."""
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                super().start()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    class _SpawnWorker(_SigintBlocked, SpawnProcess):
        """A worker started by `spawn`, see `_SigintBlocked`."""

    class _ForkServerWorker(_SigintBlocked, ForkServerProcess):
        """A worker started by `forkserver`, see `_SigintBlocked`.

        The worker is forked by the forkserver and has its mask, i.e. the mask
        of the thread which started the forkserver, the first worker.
        """

    class _ForkWorker(_SigintBlocked, ForkProcess):
        """A worker started by `fork`, see `_SigintBlocked`."""

    class _SpawnContext(SpawnContext):
        """The context of a pool started by `spawn`."""

        Process = _SpawnWorker

    class _ForkServerContext(ForkServerContext):
        """The context of a pool started by `forkserver`."""

        Process = _ForkServerWorker

    class _ForkContext(ForkContext):
        """The context of a pool started by `fork`."""

        Process = _ForkWorker

    _WORKER_CONTEXTS.update(
        spawn=_SpawnContext(), forkserver=_ForkServerContext(), fork=_ForkContext()
    )


def _context(preload: Sequence[str]) -> BaseContext:
    """Get the context of a pool, the forkserver preloads the modules.

    The forkserver imports the modules once and the workers inherit them; the
    main module is never preloaded, so a worker imports it and a script without
    the guard is found, see the module. On POSIX the workers start with SIGINT
    blocked, see `_SigintBlocked`.
    """
    context = process_context()
    method = context.get_start_method()
    if method == "forkserver":
        context.set_forkserver_preload(sorted({*PRELOAD, *preload}))
    return _WORKER_CONTEXTS.get(method, context)


def _alive() -> int:
    """Probe of a worker, which answers when it started."""
    return os.getpid()


def _initialize_worker() -> None:
    """Initialize a worker: SIGINT is ignored.

    A terminal sends Ctrl-C to every process of the job: a worker which waits
    for a task would die with a traceback and break the pool. The parent
    handles it and stops a pool whose run it interrupts, see `stop`. On POSIX
    the worker started with SIGINT blocked, see `_SigintBlocked`: ignoring it
    discards a SIGINT which arrived since, and it is unblocked again.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform != "win32":
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})


def start_pool(n_workers: int, preload: Sequence[str] = ()) -> ProcessPoolExecutor:
    """Start a pool whose workers answer, see `pool` for one which is kept.

    The workers ignore SIGINT, Ctrl-C interrupts the parent only, see the
    module.

    Args:
        n_workers: the number of worker processes.
        preload: modules the forkserver imports once for all workers; it
            takes effect only when the forkserver starts for the first time in
            this process.

    Returns:
        The pool, the caller stops it with `stop`.

    Raises:
        RuntimeError: in a worker process (no nested pools), or if the
            workers die while they start or do not start within
            `WORKER_STARTUP_TIMEOUT`; both are what a script without the
            guard does, see `GUARD_MESSAGE`.
    """
    if in_worker():
        raise RuntimeError(
            "A pool was started inside a worker process; run the inner part "
            "serially (n_workers=1)."
        )
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=_context(preload),
        initializer=_initialize_worker,
    )
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

    A worker which ran a chunk of a scan prints no messages of the integrator
    (SUNDIALS) for the rest of its life, see
    `sbmlsim.simulator.worker.quiet_sundials`: the workers report a failure
    through its exception or the variable `status` of a scan.

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

    The thread of the pool which manages its workers is the only one which
    waits for them to end, and `stop` waits for that thread: when it returns,
    the workers have ended and the pool knows it. A second thread which
    waited for a worker would race with it under the start method `spawn` on
    POSIX: the one which loses finds no process to wait for, and the worker
    looks alive for a moment after it ended.
    """
    for n_workers, kept in list(_POOLS.items()):
        if kept is executor:
            del _POOLS[n_workers]
    logger.debug("Stopping a pool")
    processes = list((executor._processes or {}).values())
    # typeshed declares the thread as the object which wakes it up
    manager = cast(threading.Thread | None, executor._executor_manager_thread)
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        # a worker which ended already is not signalled again
        process.terminate()
    if manager is not None:
        manager.join(timeout=STOP_TIMEOUT)
        if manager.is_alive():
            logger.warning(
                "The workers of a pool did not end within %s s.", STOP_TIMEOUT
            )


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
