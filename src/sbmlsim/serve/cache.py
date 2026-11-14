"""The cache of the answers of the simulations, by the hash of their request.

A response of the service is addressed by its `spec_hash`, the hash of the
resolved request (`sbmlsim.serve.app`), which covers everything which changes
its numbers: the model, the declaration of the panel, the settings of the
simulator and the values of the knobs. A body of that hash is therefore the
one answer of that request forever, which is what a `ResponseCache` keeps: a
presentation of many viewers who turn the same knob to the same value costs
one simulation, and the stored body is served as it is, its provenance
included, so a reader who asks twice is told of the one run which computed it.

The cache is the one of an application (`sbmlsim.serve.app.create_app`) and
never a global of the module, so two applications in one process never serve
each other's bodies.

A service runs its sync routes in a thread pool, so two requests are genuinely
concurrent: every operation which reads or writes the entries holds the lock
of the cache (`get`, `put`, `len`), and none of them holds it over anything
slow, a simulation least of all - a run happens between the `get` which missed
and the `put` which stores its answer. Two concurrent requests of one hash may
therefore both simulate and both store the same body, which costs a second run
and never a wrong answer.

`BoundedCache` is the bounded mapping both caches of the service are: the
responses here and the runs which the panels of one simulation share
(`sbmlsim.serve.runner.PanelRunner`), each of its own bound. The bound is one
of bytes and not of entries, since the entries differ by orders of magnitude:
the answer of one point of a panel of one curve is kilobytes, the one of a grid
at the limits of a declaration tens of megabytes. The entry which was used
least recently is evicted until the entries fit, and an entry larger than the
whole bound is not kept at all, so its request is served and simulated again
the next time.

The bytes of a response are the ones of `response_bytes`, measured with
`tracemalloc` on the objects of a `SimulateResponse`: its numbers are a list of
python floats each, 24 bytes of the float and 8 of its place in the list, so a
response is about 32 bytes per number. The four curves of the glimepiride page
of pkdb_models over its largest grid, 64 points of 501 times, are 4.1 MB, and
an answer at the limits a declaration may state (`sbmlsim.panel.MAX_CURVES`
curves over `MAX_GRID` points of `MAX_POINTS + 1` times) is 65.7 MB. A cache
of `MAX_BYTES` (256 MiB) thus keeps 64 such answers of the glimepiride page or
4 at the limits, and thousands of the answers of one point.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from sbmlsim.serve.payload import SimulateResponse

logger = logging.getLogger(__name__)

__all__ = ["MAX_BYTES", "BoundedCache", "ResponseCache", "response_bytes"]

#: the bytes of the responses a cache keeps by default, 256 MiB, see the module
MAX_BYTES: Final = 256 * 2**20

#: the bytes of a number of a response: a python float of 24 bytes and its
#: place of 8 bytes in the list of its row, see `response_bytes`
NUMBER_BYTES: Final = 32
#: the bytes of a list of a response, a row of a curve or the rows of one: the
#: 56 bytes of the object of the list and a margin of its over-allocation
LIST_BYTES: Final = 64
#: the bytes of the rest of a response: the hash, the units, the axes of the
#: grid, the provenance and the status, measured as at most about 6 KB
BASE_BYTES: Final = 8192


def response_bytes(response: SimulateResponse) -> int:
    """Get the bytes a response holds, the measure of `ResponseCache`.

    It is an estimate from the numbers of the response which is at least what
    `tracemalloc` measures of it and within a few percent of it for a response
    of more than a few thousand numbers, see the module.

    Args:
        response: the response.

    Returns:
        The bytes.
    """
    rows = [row for curve in response.curves.values() for row in curve]
    numbers = len(response.x) + sum(len(row) for row in rows)
    lists = 1 + len(response.curves) + len(rows)
    return BASE_BYTES + NUMBER_BYTES * numbers + LIST_BYTES * lists


class BoundedCache[V]:
    """A mapping of at most a number of bytes of entries by their key, see the module.

    The entry which was used least recently is the one which is evicted when
    the entries are more bytes than the bound, and a `get` of an entry is a
    use of it; an entry of more bytes than the bound is not kept. Every
    operation holds the lock of the cache and none of them holds it over
    anything slow, so the cache is safe under the thread pool of a server.
    """

    def __init__(
        self, max_bytes: int, size: Callable[[V], int], what: str = "entry"
    ) -> None:
        """Create an empty cache of at most a number of bytes.

        Args:
            max_bytes: the bytes the entries of the cache hold at most; the
                ones which were used least recently are evicted when an entry
                is stored above it.
            size: the bytes of an entry, computed once when it is stored.
            what: what an entry is, e.g. `response`, which the refusal of a
                bound and the log of an eviction name.

        Raises:
            ValueError: for a cache of fewer than one byte.
        """
        if max_bytes < 1:
            raise ValueError(
                f"A cache of {what}s keeps at least one byte, not {max_bytes}."
            )
        self._max_bytes = max_bytes
        self._size = size
        self._what = what
        #: the entries by their key, the least recently used one first
        self._entries: OrderedDict[str, V] = OrderedDict()
        #: the bytes of every entry by its key
        self._sizes: dict[str, int] = {}
        #: the bytes of all entries
        self._bytes = 0
        #: the lock of every read and write of the entries, see the module
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        """Get the representation, which carries no entry."""
        return f"{type(self).__name__}(max_bytes={self._max_bytes})"

    def __len__(self) -> int:
        """Get the number of entries the cache holds."""
        with self._lock:
            return len(self._entries)

    @property
    def max_bytes(self) -> int:
        """The bytes the entries of the cache hold at most, see `__init__`."""
        return self._max_bytes

    @property
    def nbytes(self) -> int:
        """The bytes the entries of the cache hold now."""
        with self._lock:
            return self._bytes

    def get(self, key: str) -> V | None:
        """Get the entry which is stored under a key, and use it.

        Args:
            key: the key of the entry.

        Returns:
            The entry as it was stored, or `None` for a key the cache does not
            hold.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    def store(self, key: str, entry: V) -> bool:
        """Store an entry under a key as the most recently used one.

        A key which is stored already is replaced by the given entry and stays
        one entry. An entry of more bytes than the bound of the cache is not
        kept, and an entry it would replace is removed, so the cache never
        answers a key with an entry older than the last one stored.

        Args:
            key: the key of the entry.
            entry: the entry.

        Returns:
            Whether the entry is kept.
        """
        size = self._size(entry)
        evicted: list[str] = []
        with self._lock:
            self._remove(key)
            kept = size <= self._max_bytes
            if kept:
                self._entries[key] = entry
                self._sizes[key] = size
                self._bytes += size
                while self._bytes > self._max_bytes:
                    oldest = next(iter(self._entries))
                    self._remove(oldest)
                    evicted.append(oldest)
        if not kept:
            logger.debug(
                "The %s '%s' of %d bytes is larger than the cache of %d bytes and "
                "is not kept.",
                self._what,
                key,
                size,
                self._max_bytes,
            )
        for one in evicted:
            logger.debug("The %s '%s' was evicted from the cache.", self._what, one)
        return kept

    def _remove(self, key: str) -> None:
        """Remove the entry of a key, if there is one; the caller holds the lock."""
        if self._entries.pop(key, None) is not None:
            self._bytes -= self._sizes.pop(key)


class ResponseCache(BoundedCache["SimulateResponse"]):
    """The responses of the simulations by their hash, see the module.

    The response which was used least recently is the one which is evicted
    when the responses are more bytes than the bound (`response_bytes`), and a
    `get` of an entry is a use of it, so the knobs a presentation turns to
    again and again stay in it.
    """

    def __init__(self, max_bytes: int = MAX_BYTES) -> None:
        """Create an empty cache of at most a number of bytes of responses.

        Args:
            max_bytes: the bytes the responses of the cache hold at most, see
                the module.

        Raises:
            ValueError: for a cache of fewer than one byte.
        """
        super().__init__(max_bytes, response_bytes, what="response")

    def put(self, response: SimulateResponse) -> bool:
        """Store a response under its own hash as the most recently used one.

        A hash which is stored already is replaced by the given response,
        which is the same body of the same request, and stays one entry; a
        response of more bytes than the cache is not kept.

        Args:
            response: the response, stored under its `spec_hash`.

        Returns:
            Whether the response is kept.
        """
        return self.store(response.spec_hash, response)
