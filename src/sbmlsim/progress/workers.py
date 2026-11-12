"""The progress of the activities which run in a worker process.

A run which hands its work to a pool (the experiments of a runner, the runs of
a fit, the scans of an identifiability) sees the progress of its workers
through a queue: the worker puts a `Snapshot` of every activity on it, see
`worker_reporting`, and a thread of the parent replays the snapshots on its
reporter, see `forwarding`. The activities of a worker nest under the activity
of the parent, so the bar of a scan of a worker is drawn below the bar of the
run which waits for it.

The queue of the parent may carry other messages as well, e.g. the events of
the experiments of `sbmlsim.experiment.progress`, which the thread hands to a
listener: one queue and one thread carry the progress of a run.
"""

from __future__ import annotations

import logging
import os
import queue as queue_module
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sbmlsim.progress.core import Activity, Reporter, _now, report, reporter

logger = logging.getLogger(__name__)

#: the seconds the thread of the parent waits for the next message
POLL = 0.05

#: the seconds the end of a run waits for the thread
JOIN = 1.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What an activity of a worker is at, the message of the queue.

    The snapshot holds what a reporter renders, not the activity itself: the
    activity of a worker lives in its process, and the parent builds one of
    its own from the snapshots, see `forwarding`.

    Attributes:
        pid: the process the activity runs in, which identifies it with `uid`.
        uid: the activity in its process.
        parent_uid: the activity it runs in, `None` for the outermost one of
            the worker.
        kind: what kind of operation it is.
        label: what it runs on.
        completed: the steps which are done.
        total: the steps in total, `None` while they are not known.
        workers: the steps which are processed at once.
        elapsed: the seconds since the activity started.
        state: `started`, `running` or `finished`.
        error: what the activity failed by, `None` if it did not.
    """

    pid: int
    uid: int
    parent_uid: int | None
    kind: str
    label: str
    completed: int
    total: int | None
    workers: int
    elapsed: float
    state: str
    error: str | None


def _snapshot(activity: Activity, state: str, error: str | None = None) -> Snapshot:
    """Snapshot an activity of a worker."""
    return Snapshot(
        pid=os.getpid(),
        uid=activity.uid,
        parent_uid=None if activity.parent is None else activity.parent.uid,
        kind=activity.kind,
        label=activity.label,
        completed=activity.completed,
        total=activity.total,
        workers=activity.workers,
        elapsed=activity.elapsed,
        state=state,
        error=error,
    )


class QueueReporter:
    """Put the activities of a worker on the queue of its parent."""

    def __init__(self, messages: Any) -> None:
        """Create the reporter of a worker.

        Args:
            messages: the queue to the parent.
        """
        self.messages = messages

    def started(self, activity: Activity) -> None:
        """Report that an activity started."""
        self.messages.put(_snapshot(activity, "started"))

    def advanced(self, activity: Activity) -> None:
        """Report a step of an activity."""
        self.messages.put(_snapshot(activity, "running"))

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Report that an activity ended, and what it ended by."""
        self.messages.put(
            _snapshot(activity, "finished", None if error is None else repr(error))
        )


@contextmanager
def worker_reporting(messages: Any | None) -> Generator[None]:
    """Report the activities of a worker to the queue of its parent.

    Args:
        messages: the queue to the parent, `None` for a worker without one,
            which reports nothing.
    """
    if messages is None:
        yield
        return
    with reporter(QueueReporter(messages)):
        yield


class _Replay:
    """The activities of the workers in the process of the parent.

    An activity of a worker is identified by its process and its number; the
    parent keeps one activity of its own per activity of a worker, so its
    reporters see a tree which holds the activities of every worker.
    """

    def __init__(self, target: Reporter, parent: Activity | None) -> None:
        """Create the replay.

        Args:
            target: what renders the activities.
            parent: the activity of the parent the activities of the workers
                nest under, `None` for a tree of its own.
        """
        self._target = target
        self._parent = parent
        self._activities: dict[tuple[int, int], Activity] = {}

    def __call__(self, snapshot: Snapshot) -> None:
        """Replay a snapshot of a worker on the reporter."""
        key = (snapshot.pid, snapshot.uid)
        if snapshot.state == "started":
            above = (
                self._activities.get((snapshot.pid, snapshot.parent_uid))
                if snapshot.parent_uid is not None
                else self._parent
            )
            current = Activity(
                kind=snapshot.kind,
                label=snapshot.label,
                total=snapshot.total,
                completed=snapshot.completed,
                workers=snapshot.workers,
                parent=above if above is not None else self._parent,
                started_at=_now() - snapshot.elapsed,
            )
            self._activities[key] = current
            report(self._target, "started", current)
            return
        current = self._activities.get(key)
        if current is None:
            # a worker whose start was lost, e.g. a queue which was closed
            return
        current.completed = snapshot.completed
        current.total = snapshot.total
        current.label = snapshot.label
        current.workers = snapshot.workers
        if snapshot.state != "finished":
            report(self._target, "advanced", current)
            return
        self._activities.pop(key, None)
        report(
            self._target,
            "finished",
            current,
            None if snapshot.error is None else RuntimeError(snapshot.error),
        )


@contextmanager
def forwarding(
    messages: Any | None,
    *,
    target: Reporter | None = None,
    listener: Callable[[Any], None] | None = None,
    parent: Activity | None = None,
    poll: float = POLL,
    close: bool = False,
) -> Generator[None]:
    """Forward the messages of the workers while the run lasts.

    A thread of the parent takes the messages from the queue: a `Snapshot` is
    replayed on the reporter, every other message is handed to the listener. A
    worker which is terminated while it writes, e.g. when Ctrl-C stops the
    pool, can leave a message half written, which the thread waits for in
    vain: it is a daemon and is not waited for longer than a second, so it
    never holds up the end of a run.

    Args:
        messages: the queue of the workers, `None` for a run without one.
        target: what renders the activities of the workers, `None` for a run
            which renders none.
        listener: what receives the other messages of the queue, `None` for a
            run which has none.
        parent: the activity the activities of the workers nest under.
        poll: the seconds the thread waits for the next message.
        close: close the queue when the run ends, for a queue of a pool which
            is not used again.
    """
    if messages is None or (target is None and listener is None):
        yield
        return
    replay = _Replay(target, parent) if target is not None else None
    stop = threading.Event()

    def forward() -> None:
        """Take the messages from the queue until the run ends."""
        while not stop.is_set():
            try:
                message = messages.get(timeout=poll)
            except queue_module.Empty:
                continue
            except EOFError, OSError:
                return
            if isinstance(message, Snapshot):
                if replay is not None:
                    replay(message)
            elif listener is not None:
                listener(message)

    thread = threading.Thread(target=forward, name="sbmlsim-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=JOIN)
        if close and not thread.is_alive():
            messages.close()
