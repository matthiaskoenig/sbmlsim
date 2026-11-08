"""The large arrays of a chunk result in shared memory, from a worker to the parent.

A worker of a pool returns the result of a chunk through a pipe of the pool,
which pickles its arrays in the worker and unpickles them in the parent: 8
results of 214 MB take 1.44 s on linux with 4 workers, 0.11 s written into
shared memory. A worker therefore writes an array of at least
`SHARED_MEMORY_THRESHOLD` bytes into a segment of `multiprocessing.shared_memory`
which it creates (`share`) and returns its `SharedArray`, the name, the shape
and the dtype, instead of the array; the parent maps the segment, removes its
name and reads the array from the mapping, without a copy, which is released
with the array (`receive`). A smaller array
keeps pickling, a segment costs more than it saves.

The parent owns the segments of a run: their names are a prefix of the run
(`run_prefix`) and the position of the chunk (`segment_stem`), and a worker
creates a segment without the resource tracker (`track=False`), which would
remove it when the worker ends, before the parent read it. The parent removes
the name of every segment it reads and, at the end of the run, also after an
error, Ctrl-C or a dead worker, every name of the run which may exist
(`unlink`), so no segment outlives its run. A run in the calling process never
uses segments.

Shared memory is used on POSIX only (linux, macOS): on windows a named
mapping disappears when its last handle closes, so a worker cannot hand one
over after it returned, and the arrays keep pickling there. An array keeps
pickling as well when its segment cannot be created or its memory reserved,
e.g. a `/dev/shm` too small for it.
"""

from __future__ import annotations

import importlib
import logging
import math
import mmap
import os
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: the smallest array in bytes a worker hands over in shared memory
SHARED_MEMORY_THRESHOLD: int = 1 << 20

#: the suffixes of the segments of the arrays of a chunk, `values` and
#: `scalars` of `sbmlsim.simulator.worker.ChunkResult`
SUFFIXES: tuple[str, str] = ("v", "s")


@dataclass(frozen=True)
class SharedArray:
    """An array in a segment of shared memory, see `share`.

    Attributes:
        name: the name of the segment.
        shape: the shape of the array.
        dtype: the dtype of the array, as `numpy.dtype.str`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str


def supported() -> bool:
    """Whether arrays are handed over in shared memory, on POSIX only."""
    return os.name == "posix"


def run_prefix() -> str:
    """Get a new prefix of the names of the segments of a run.

    It is short, since macOS limits the name of a segment to 31 characters
    with its leading slash.
    """
    return f"sbs{secrets.token_hex(4)}_"


def segment_stem(prefix: str, chunk: int) -> str:
    """Get the stem of the names of the segments of a chunk of a run.

    The segment of an array of the chunk is the stem and one of `SUFFIXES`.

    Args:
        prefix: the prefix of the run, see `run_prefix`.
        chunk: the position of the chunk in the run.
    """
    return f"{prefix}{chunk}"


def segment_names(stem: str) -> list[str]:
    """Get the name of every segment a chunk of this stem may create."""
    return [f"{stem}{suffix}" for suffix in SUFFIXES]


def share(array: np.ndarray, name: str) -> np.ndarray | SharedArray:
    """Write an array into a new segment of shared memory, if it is worth it.

    Args:
        array: the array.
        name: the name of the segment.

    Returns:
        The `SharedArray` of the segment, or the array itself when it is
        smaller than `SHARED_MEMORY_THRESHOLD`, shared memory is not
        `supported` or the segment cannot be created.
    """
    if not supported() or array.nbytes < SHARED_MEMORY_THRESHOLD:
        return array
    try:
        segment = SharedMemory(name=name, create=True, size=array.nbytes, track=False)
    except OSError as err:
        logger.debug("No shared memory for %s: %s", name, err)
        return array
    try:
        _reserve(segment, array.nbytes)
        np.ndarray(array.shape, array.dtype, buffer=segment.buf)[...] = array
    except OSError as err:
        logger.debug("No shared memory for %s: %s", name, err)
        segment.close()
        segment.unlink()
        return array
    segment.close()
    return SharedArray(name=name, shape=array.shape, dtype=array.dtype.str)


def _reserve(segment: SharedMemory, size: int) -> None:
    """Reserve the memory of a new segment, so that writing it cannot fail.

    On linux a segment is a file of `/dev/shm`, which `ftruncate` sizes
    without reserving its memory: writing more than the file system holds
    kills the process with SIGBUS. `posix_fallocate` reserves it or raises.

    Raises:
        OSError: if the memory cannot be reserved.
    """
    fallocate = getattr(os, "posix_fallocate", None)
    descriptor = getattr(segment, "_fd", -1)
    if fallocate is not None and descriptor >= 0:
        fallocate(descriptor, 0, size)


def receive(shared: SharedArray) -> np.ndarray:
    """Get an array from its segment, whose name is removed.

    The array is a read-only view on the memory of the segment, mapped into
    the process, which is released with the last reference to the array: no
    copy, and nothing to close. The segment is mapped without
    `SharedMemory`, which must be closed explicitly and cannot be closed
    while a view on it exists.

    Args:
        shared: the array in its segment.

    Returns:
        The array.
    """
    dtype = np.dtype(shared.dtype)
    size = math.prod(shared.shape) * dtype.itemsize
    posixshmem = _posixshmem()
    descriptor = posixshmem.shm_open(f"/{shared.name}", os.O_RDONLY, 0)
    try:
        # the name goes first: the memory stays mapped until it is released
        posixshmem.shm_unlink(f"/{shared.name}")
        memory = mmap.mmap(descriptor, size, access=mmap.ACCESS_READ)
    finally:
        os.close(descriptor)
    return np.frombuffer(memory, dtype=dtype).reshape(shared.shape)


def _posixshmem() -> Any:
    """Get the module of the stdlib behind `SharedMemory` on POSIX.

    It opens and removes a segment by its name, see `receive` and `unlink`.
    """
    return importlib.import_module("_posixshmem")


def unlink(names: Iterable[str]) -> None:
    """Remove the segments of these names which exist.

    Args:
        names: the names of segments which may exist.
    """
    if not supported():
        return
    # by name, without mapping a segment which a worker may have left empty
    posixshmem = _posixshmem()
    for name in names:
        try:
            posixshmem.shm_unlink(f"/{name}")
        except FileNotFoundError:
            continue
