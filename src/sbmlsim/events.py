"""Events of a run: the steps of the analyses as JSON lines.

A run records the steps of its analyses (a repeat of a fit, a chunk of a scan,
a block of a propagation, a profile) as one JSON line each, `{"time": ...,
"event": ..., **fields}`, into the file of the recording, so that something
outside of the process can follow the run. The recording is one per process,
not one per context, so a thread which does not copy the context (the thread
which forwards the progress of the experiments) reaches it as well. Outside of
a recording `emit` does nothing and costs a lookup. An event never aborts an
analysis: a value JSON has no number for is written as `null` and a failing
write is logged. This module imports the standard library only.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sink: IO[str] | None = None


def _clean(value: Any) -> Any:
    """Replace the numbers JSON has no value for (NaN, infinity) by `None`."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def emit(event: str, **fields: Any) -> None:
    """Record an event in the recording of the process, if there is one.

    Args:
        event: the name of the event, e.g. `fit.repeat`.
        **fields: JSON values of the event.
    """
    if _sink is None:
        return
    record = {
        "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": event,
        **_clean(fields),
    }
    try:
        line = json.dumps(record, default=str, allow_nan=False) + "\n"
        with _lock:
            sink = _sink
            if sink is not None:
                sink.write(line)
                sink.flush()
    except (OSError, ValueError) as err:
        logger.warning("The event '%s' could not be recorded: %s", event, err)


@contextmanager
def recording(path: Path) -> Generator[None]:
    """Record the events of the process into a file, appended line by line.

    A recording inside another one replaces it until it ends.

    Args:
        path: the file, created if it does not exist.
    """
    global _sink
    with Path(path).open("a", encoding="utf-8", buffering=1) as sink:
        with _lock:
            previous, _sink = _sink, sink
        try:
            yield
        finally:
            with _lock:
                _sink = previous
