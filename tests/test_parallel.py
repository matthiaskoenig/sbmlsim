"""The pools of sbmlsim."""

import multiprocessing
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

from sbmlsim import parallel
from sbmlsim.parallel import process_context

REPO = Path(__file__).parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_process_context_is_not_fork_by_default(
    start_methods: Callable[[str | None, str], None],
) -> None:
    """The default `fork` of python 3.13 on linux is replaced by `forkserver`."""
    start_methods(None, "fork")
    assert process_context().get_start_method() == "forkserver"


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no fork")
def test_process_context_keeps_the_start_method_of_the_user(
    start_methods: Callable[[str | None, str], None],
) -> None:
    """A start method set with `multiprocessing.set_start_method` is used."""
    start_methods("fork", "forkserver")
    assert process_context().get_start_method() == "fork"
    start_methods("spawn", "fork")
    assert process_context().get_start_method() == "spawn"


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_process_context_does_not_take_the_default_for_a_choice(
    start_methods: Callable[[str | None, str], None],
) -> None:
    """A start method which is the default of the platform is no choice.

    Python 3.13 fixes the start method of the process to the default when a
    process is started by `spawn` or `forkserver`, i.e. by the first pool of
    this context, which must not make the second pool fork.
    """
    start_methods("fork", "fork")
    assert process_context().get_start_method() == "forkserver"


def test_process_context_keeps_another_default(
    start_methods: Callable[[str | None, str], None],
) -> None:
    """The default `spawn` of macos and windows is used."""
    start_methods(None, "spawn")
    assert process_context().get_start_method() == "spawn"


def test_process_context_does_not_fix_the_start_method() -> None:
    """The start method of the process stays unset, so it is not read as a choice.

    A start method which is set cannot be told from one the user chose, a pool
    of the default context on python 3.13 would fix it to `fork`.
    """
    code = (
        "import multiprocessing\n"
        "from sbmlsim.parallel import process_context\n"
        "method = process_context().get_start_method()\n"
        "print(method, multiprocessing.get_start_method(allow_none=True))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    method, fixed = out.stdout.split()
    assert method != "fork"
    assert fixed == "None"


def test_pools_of_the_process_context_do_not_fork_a_process_with_threads() -> None:
    """The workers of a pool are not forked from a process which runs threads.

    A second pool is created after the first one has started its workers, which
    fixes the start method of the process on python 3.13, the fixture
    `_no_fork_of_threads` of the tests fails the test if one of them forks.
    """
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    try:
        for _ in range(2):
            context = process_context()
            assert context.get_start_method() != "fork"
            with context.Pool(processes=2) as pool:
                assert pool.map(abs, [-1, -2, -3]) == [1, 2, 3]
    finally:
        stop.set()
        thread.join()


def test_one_worker_is_serial() -> None:
    assert parallel.resolve_workers(1, 10_000) == 1


def test_a_number_of_workers_is_taken_as_given() -> None:
    assert parallel.resolve_workers(3, 2) == 3


def test_less_than_one_worker_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        parallel.resolve_workers(0, 10)
    with pytest.raises(ValueError, match="at least 1"):
        parallel.check_workers(0)
    parallel.check_workers(None)
    parallel.check_workers(1)


def test_none_is_serial_below_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8)
    assert parallel.resolve_workers(None, parallel.POOL_THRESHOLD - 1) == 1
    assert parallel.resolve_workers(None, parallel.POOL_THRESHOLD) == 8
    assert parallel.resolve_workers(None, 10**6) == 8


def test_none_is_serial_in_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    assert parallel.in_worker()
    assert parallel.resolve_workers(None, 10**6) == 1


def test_a_worker_starts_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    with pytest.raises(RuntimeError, match="inside a worker"):
        parallel.start_pool(2)


def test_the_pool_is_kept_per_size() -> None:
    first = parallel.pool(2)
    assert parallel.pool(2) is first
    assert first.submit(os.getpid).result() != os.getpid()


def test_a_stopped_pool_is_replaced() -> None:
    first = parallel.pool(2)
    parallel.stop(first)
    second = parallel.pool(2)
    assert second is not first
    assert second.submit(os.getpid).result() != os.getpid()


def test_shutdown_stops_every_pool() -> None:
    parallel.pool(2)
    parallel.shutdown()
    assert parallel._POOLS == {}


def test_the_cache_builds_an_object_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    built: list[int] = []

    def factory() -> int:
        built.append(1)
        return len(built)

    assert parallel.worker_cache(("test", "once"), factory) == 1
    assert parallel.worker_cache(("test", "once"), factory) == 1
    assert built == [1]


def test_the_cache_keeps_the_newest_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "WORKER_CACHE_SIZE", 2)
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    for k in range(3):
        parallel.worker_cache(("test", k), lambda k=k: k)
    assert list(parallel._CACHE) == [("test", 1), ("test", 2)]


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_the_forkserver_preloads_the_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context("forkserver")
    preloaded: list[list[str]] = []
    monkeypatch.setattr(parallel, "process_context", lambda: context)
    monkeypatch.setattr(context, "set_forkserver_preload", preloaded.append)

    assert parallel._context(["examples.demo.demo"]) is context
    assert preloaded == [sorted({*parallel.PRELOAD, "examples.demo.demo"})]


def test_another_start_method_is_used_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context("spawn")
    monkeypatch.setattr(parallel, "process_context", lambda: context)
    monkeypatch.setattr(
        context,
        "set_forkserver_preload",
        lambda modules: pytest.fail("only the forkserver preloads"),
    )
    assert parallel._context(["examples.demo.demo"]) is context


@pytest.mark.skipif(sys.platform == "win32", reason="windows has no forkserver")
def test_a_script_without_the_guard_is_reported(tmp_path: Path) -> None:
    """The workers import the script again and die, the parent names the guard."""
    script = tmp_path / "unguarded.py"
    script.write_text("from sbmlsim import parallel\n\nparallel.pool(2)\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert result.returncode != 0
    assert 'if __name__ == "__main__":' in result.stderr


def test_a_pool_which_broke_is_replaced() -> None:
    first = parallel.pool(2)
    with pytest.raises(BrokenProcessPool):
        first.submit(os._exit, 1).result()
    second = parallel.pool(2)
    assert second is not first
    assert second.submit(os.getpid).result() != os.getpid()


def test_a_pool_which_was_shut_down_is_replaced() -> None:
    with parallel.pool(2) as first:
        pass
    second = parallel.pool(2)
    assert second is not first
    assert second.submit(os.getpid).result() != os.getpid()


def test_an_interrupted_start_stops_the_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[ProcessPoolExecutor] = []
    stopped: list[ProcessPoolExecutor] = []
    real_stop = parallel.stop

    def stop(executor: ProcessPoolExecutor) -> None:
        stopped.append(executor)
        real_stop(executor)

    def interrupted(self: Future, timeout: float | None = None) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel, "stop", stop)
    monkeypatch.setattr(Future, "result", interrupted)

    class Recording(ProcessPoolExecutor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Recording)
    with pytest.raises(KeyboardInterrupt):
        parallel.start_pool(2)
    assert stopped == started and len(started) == 1
    assert parallel._POOLS == {}
    assert not any(p.is_alive() for p in (started[0]._processes or {}).values())
