"""The large arrays of a chunk result in shared memory."""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.simulator import shared
from sbmlsim.simulator.shared import SHARED_MEMORY_THRESHOLD, SharedArray

posix = pytest.mark.skipif(os.name != "posix", reason="shared memory on POSIX only")


def _large() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((3, SHARED_MEMORY_THRESHOLD // 8 // 3 + 1, 2))


def _name() -> str:
    return shared.segment_stem(shared.run_prefix(), 7) + shared.SUFFIXES[0]


def test_the_names_are_short_enough_for_macos() -> None:
    """macOS limits a name to 31 characters with its leading slash."""
    stem = shared.segment_stem(shared.run_prefix(), 10**6)
    assert all(len(f"/{name}") <= 31 for name in shared.segment_names(stem))
    assert shared.run_prefix() != shared.run_prefix()


def test_a_small_array_is_not_shared() -> None:
    array = np.ones((4, 5))
    assert shared.share(array, _name()) is array


@posix
def test_a_large_array_is_received_bitwise_and_its_segment_removed() -> None:
    array, name = _large(), _name()
    handle = shared.share(array, name)
    assert isinstance(handle, SharedArray)
    if sys.platform == "linux":
        assert Path("/dev/shm", name).exists()
    received = shared.receive(handle)
    if sys.platform == "linux":
        assert not Path("/dev/shm", name).exists()
    np.testing.assert_array_equal(received, array)
    assert received.dtype == array.dtype
    assert not received.flags.writeable
    # the memory stays mapped as long as the array lives
    del handle
    assert float(received.sum()) == float(array.sum())


def test_windows_keeps_pickling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "supported", lambda: False)
    array = _large()
    assert shared.share(array, _name()) is array
    shared.unlink([_name()])


@posix
def test_a_segment_which_cannot_be_created_keeps_the_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def full(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shared, "SharedMemory", full)
    array = _large()
    assert shared.share(array, _name()) is array


@posix
def test_memory_which_cannot_be_reserved_keeps_the_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `/dev/shm` too small: the segment is removed, the array returned."""

    def full(segment: Any, size: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shared, "_reserve", full)
    array, name = _large(), _name()
    assert shared.share(array, name) is array
    if sys.platform == "linux":
        assert not Path("/dev/shm", name).exists()


@posix
def test_unlink_removes_the_segments_which_exist() -> None:
    names = [_name(), _name()]
    handle = shared.share(_large(), names[0])
    assert isinstance(handle, SharedArray)
    shared.unlink(names)
    with pytest.raises(FileNotFoundError):
        shared.receive(handle)
