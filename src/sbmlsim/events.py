"""Events of a run: the steps of the analyses as JSON lines.

A run records the steps of its analyses (a repeat of a fit, a chunk of a scan,
a block of a propagation, a profile) as one JSON line each, `{"time": ...,
"event": ..., **fields}`, into the file of the recording of its context, so
that something outside of the process can follow the run. Outside of a
recording `emit` does nothing and costs a lookup of the context. This module
imports the standard library only.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

_lock = threading.Lock()
_sink: ContextVar[IO[str] | None] = ContextVar("sbmlsim_events_sink", default=None)


def emit(event: str, **fields: Any) -> None:
    """Record an event in the recording of the context, if there is one.

    Args:
        event: the name of the event, e.g. `fit.repeat`.
        **fields: JSON values of the event.
    """
    sink = _sink.get()
    if sink is None:
        return
    record = {
        "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }
    line = json.dumps(record, default=str) + "\n"
    with _lock:
        sink.write(line)
        sink.flush()


@contextmanager
def recording(path: Path) -> Generator[None]:
    """Record the events of the context into a file, appended line by line.

    Args:
        path: the file, created if it does not exist.
    """
    with Path(path).open("a", encoding="utf-8", buffering=1) as sink:
        token = _sink.set(sink)
        try:
            yield
        finally:
            _sink.reset(token)
