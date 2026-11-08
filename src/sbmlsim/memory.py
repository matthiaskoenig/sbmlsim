"""Give the memory which the C allocator holds free back to the system.

glibc keeps memory which a program freed in its heap for later allocations
and returns it to the system only from the top of the heap. The compilation of
a model by LLVM makes many small allocations which it frees again, about 0.9 GB
for a large model (the creatinine model of pkdb_models at `O0`), and a run of
experiments frees the arrays of its results, which glibc allocates in its heap
once their size is below its dynamic threshold of `mmap`. A process which
loads several models or runs several experiments held this memory until it
ended. `release_freed_memory` returns it (`malloc_trim`) after a model was
compiled and after the results of an experiment were released. Its cost grows
with the free memory of the heap, from microseconds for a heap without freed
memory to about 0.3 s for 10 GB of freed small allocations, small against the
compilation and the experiment which freed it.

The module imports the standard library only.
"""

import ctypes
import functools
import sys
from collections.abc import Callable

__all__ = ["release_freed_memory"]


def release_freed_memory() -> None:
    """Return the memory which the C allocator holds free to the system.

    Nothing happens on a platform without glibc (macOS, windows, musl).
    """
    trim = _malloc_trim()
    if trim is not None:
        trim(0)


@functools.cache
def _malloc_trim() -> Callable[[int], int] | None:
    """Get `malloc_trim` of glibc, `None` without it."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL(None).malloc_trim
    except OSError, AttributeError:
        return None
