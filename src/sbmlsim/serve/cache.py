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
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from sbmlsim.serve.payload import SimulateResponse

logger = logging.getLogger(__name__)

__all__ = ["MAX_ENTRIES", "ResponseCache"]

#: the responses a cache keeps by default, see `ResponseCache`
MAX_ENTRIES: Final = 256


class ResponseCache:
    """The responses of the simulations by their hash, see the module.

    The entry which was used least recently is the one which is evicted when
    the cache is full, and a `get` of an entry is a use of it, so the knobs a
    presentation turns to again and again stay in it.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        """Create an empty cache of at most a number of responses.

        Args:
            max_entries: the responses the cache keeps; the one which was used
                least recently is evicted when one more is stored.

        Raises:
            ValueError: for a cache of fewer than one entry.
        """
        if max_entries < 1:
            raise ValueError(f"A cache keeps at least one response, not {max_entries}.")
        self._max_entries = max_entries
        #: the responses by their hash, the least recently used one first
        self._entries: OrderedDict[str, SimulateResponse] = OrderedDict()
        #: the lock of every read and write of the entries, see the module
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        """Get the representation, which carries no body."""
        return f"ResponseCache(max_entries={self._max_entries})"

    def __len__(self) -> int:
        """Get the number of responses the cache holds."""
        with self._lock:
            return len(self._entries)

    @property
    def max_entries(self) -> int:
        """The responses the cache keeps, see `__init__`."""
        return self._max_entries

    def get(self, spec_hash: str) -> SimulateResponse | None:
        """Get the response which is stored under a hash, and use it.

        Args:
            spec_hash: the hash of the resolved request.

        Returns:
            The response as it was stored, provenance included, or `None` for
            a hash the cache does not hold.
        """
        with self._lock:
            response = self._entries.get(spec_hash)
            if response is None:
                return None
            self._entries.move_to_end(spec_hash)
            return response

    def put(self, response: SimulateResponse) -> None:
        """Store a response under its own hash as the most recently used one.

        A hash which is stored already is replaced by the given response,
        which is the same body of the same request, and stays one entry.

        Args:
            response: the response, stored under its `spec_hash`.
        """
        evicted: list[str] = []
        with self._lock:
            self._entries[response.spec_hash] = response
            self._entries.move_to_end(response.spec_hash)
            while len(self._entries) > self._max_entries:
                evicted.append(self._entries.popitem(last=False)[0])
        for spec_hash in evicted:
            logger.debug("The response '%s' was evicted from the cache.", spec_hash)
