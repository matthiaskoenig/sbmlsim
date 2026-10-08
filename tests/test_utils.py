"""Test the utility functions."""

import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from sbmlsim.utils import paths_text, process_context


@pytest.mark.parametrize(
    "paths, text",
    [
        (None, "none"),
        ("data", "data"),
        (Path("data"), "data"),
        ([Path("a"), Path("b")], "a\nb"),
        ((p for p in ["a", Path("b")]), "a\nb"),
        ([], ""),
    ],
)
def test_paths_text(paths: str | Path | Iterable[str | Path] | None, text: str) -> None:
    """A path is its text, several paths are one per line."""
    assert paths_text(paths) == text


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
        "from sbmlsim.utils import process_context\n"
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
