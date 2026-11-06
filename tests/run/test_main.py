"""The process of `python -m sbmlsim.run` runs itself again with one thread."""

import os
import subprocess
import sys

import pytest

from sbmlsim.run import __main__ as entry


class _Replaced(Exception):
    """`os.execv`, which replaces the process."""


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*entry.THREAD_VARIABLES, "MPLBACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [sys.executable, "-X", "dev", "-W", "error", "-m", "sbmlsim.run", "list", "p"],
    )


def test_posix_replaces_the_process_with_the_flags_of_python(
    unset: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def execv(path: str, argv: list[str]) -> None:
        calls.append((path, argv))
        raise _Replaced

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "execv", execv)
    with pytest.raises(_Replaced):
        entry._single_threaded()
    assert calls == [(sys.executable, sys.orig_argv)]
    for name in entry.THREAD_VARIABLES:
        assert os.environ[name] == "1"
    assert os.environ["MPLBACKEND"] == "Agg"


def test_windows_exits_with_the_code_of_the_new_process(
    unset: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert argv == sys.orig_argv
        assert not check
        return subprocess.CompletedProcess(argv, 4)

    def execv(path: str, argv: list[str]) -> None:
        raise AssertionError("os.execv ends the process with 0 on windows")

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "execv", execv)
    with pytest.raises(SystemExit) as info:
        entry._single_threaded()
    assert info.value.code == 4


def test_set_variables_are_kept_and_the_process_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in entry.THREAD_VARIABLES:
        monkeypatch.setenv(name, "3")
    monkeypatch.setenv("MPLBACKEND", "svg")

    def execv(path: str, argv: list[str]) -> None:
        raise AssertionError("the process runs again once only")

    monkeypatch.setattr(os, "execv", execv)
    entry._single_threaded()
    assert [os.environ[name] for name in entry.THREAD_VARIABLES] == ["3"] * 3
    assert os.environ["MPLBACKEND"] == "svg"
