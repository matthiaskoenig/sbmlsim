"""The activities of a long run and what reports them.

A long operation of sbmlsim (a run of simulation experiments, the points of a
scan, the runs of a fit, the scans of an identifiability, the figures of a
report) opens an `Activity`: what it does, how many steps it has and how many
are done. Activities nest, a fit over its jacobians, an identifiability over
its scans, so a run is a tree of what is running, and a `Reporter` renders it:
a bar on a terminal, a line in a log file, an event in `progress.jsonl`, see
`sbmlsim.progress.reporters`.

The activity of the context is a `ContextVar`, so an activity which is opened
inside another one nests under it without being handed its parent; an activity
of a worker process or of a thread which does not copy the context is opened
on its parent, see `Activity.child`.

The estimate of the time which is left is `remaining`, one for every kind of
activity: the time of a batch of the workers times the batches which are left.
A reporter which fails (a closed console, a full disk) never aborts the
computation: it is logged once and dropped for the rest of the run.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _now() -> float:
    """Get the monotonic time, the one clock of the activities."""
    return time.monotonic()


#: the number of the next activity, which identifies it in a renderer
_UIDS = count(1)

#: the reporters which failed and are dropped, by their id
_DROPPED: set[int] = set()


def remaining(
    completed: float, total: float, elapsed: float, workers: int = 1
) -> float | None:
    """Estimate the seconds which are left of an activity.

    The steps are handed to the workers in batches of `workers`, so the time
    of a batch is what the elapsed time measures: the estimate is the time per
    batch times the batches which are left. With a single worker this is the
    mean time per step times the steps which are left.

    Args:
        completed: the steps which are done.
        total: the steps in total.
        elapsed: the seconds since the activity started.
        workers: the steps which are processed at once.

    Returns:
        The seconds which are left, `0.0` for an activity which is done and
        `None` before the first step is done.
    """
    if completed <= 0 or total <= 0 or elapsed <= 0:
        return None
    workers = max(int(workers), 1)
    batches_done = math.ceil(completed / workers)
    batches_total = math.ceil(total / workers)
    return max(elapsed / batches_done * batches_total - elapsed, 0.0)


@runtime_checkable
class Reporter(Protocol):
    """What renders the activities of a run.

    Every method is called with the activity as it is, which changes while it
    runs: a reporter which keeps an activity keeps the object, not a copy.
    """

    def started(self, activity: Activity) -> None:
        """Report that the activity started."""
        ...

    def advanced(self, activity: Activity) -> None:
        """Report that the steps or the total of the activity changed."""
        ...

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Report that the activity ended, by an exception if there is one."""
        ...


@dataclass(eq=False, slots=True)
class Activity:
    """What a long operation does, and how far it got.

    Attributes:
        kind: what kind of operation it is, e.g. `"fit"`, `"scan"`; the
            renderers group and colour by it.
        label: what it runs on, e.g. the id of the experiment or the fit.
        total: the steps in total, `None` while they are not known.
        completed: the steps which are done.
        workers: the steps which are processed at once, for the estimate of
            the time which is left.
        parent: the activity this one runs in, `None` for the outermost one.
        started_at: the monotonic time the activity started.
        uid: the number which identifies the activity in a renderer.
    """

    kind: str
    label: str = ""
    total: int | None = None
    completed: int = 0
    workers: int = 1
    parent: Activity | None = None
    started_at: float = field(default_factory=lambda: _now())
    uid: int = field(default_factory=lambda: next(_UIDS))
    reporter: Reporter | None = field(default=None, repr=False)

    @property
    def depth(self) -> int:
        """Get the activities this one runs in, `0` for the outermost one."""
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def elapsed(self) -> float:
        """Get the seconds since the activity started."""
        return _now() - self.started_at

    def remaining(self) -> float | None:
        """Estimate the seconds which are left, see `remaining`."""
        if self.total is None:
            return None
        return remaining(self.completed, self.total, self.elapsed, self.workers)

    def advance(self, steps: int = 1) -> None:
        """Report that steps of the activity are done.

        Args:
            steps: the steps which are done, e.g. one point of a scan.
        """
        self.update(completed=self.completed + steps)

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        label: str | None = None,
        workers: int | None = None,
    ) -> None:
        """Report what the activity is at now.

        Args:
            completed: the steps which are done.
            total: the steps in total, e.g. once they are known.
            label: what it runs on.
            workers: the steps which are processed at once.
        """
        if completed is not None:
            self.completed = completed
        if total is not None:
            self.total = total
        if label is not None:
            self.label = label
        if workers is not None:
            self.workers = workers
        report(self.reporter, "advanced", self)

    def child(
        self,
        kind: str,
        label: str = "",
        *,
        total: int | None = None,
        workers: int = 1,
    ) -> AbstractContextManager[Activity]:
        """Open an activity inside this one, also outside of its context.

        A thread or a process which does not copy the context opens its
        activity on the one it belongs to, instead of relying on the activity
        of the context.

        Args:
            kind: what kind of operation it is.
            label: what it runs on.
            total: the steps in total, `None` while they are not known.
            workers: the steps which are processed at once.

        Returns:
            The context of the activity, which yields it.
        """
        return activity(kind, label, total=total, workers=workers, parent=self)


#: the activity of the context, `None` outside of one
_ACTIVITY: ContextVar[Activity | None] = ContextVar("sbmlsim_activity", default=None)

#: what reports the activities of the context, `None` outside of a reporting
_REPORTER: ContextVar[Reporter | None] = ContextVar("sbmlsim_reporter", default=None)


def report(
    target: Reporter | None,
    method: str,
    activity_: Activity,
    error: BaseException | None = None,
) -> None:
    """Call a reporter, which never aborts the computation.

    The first exception of a reporter is logged and the reporter is dropped
    for the rest of the process.

    Args:
        target: the reporter, `None` if nothing reports.
        method: `"started"`, `"advanced"` or `"finished"`.
        activity_: the activity which is reported.
        error: the exception an activity ended by, for `"finished"`.
    """
    if target is None or id(target) in _DROPPED:
        return
    try:
        if method == "finished":
            target.finished(activity_, error)
        elif method == "started":
            target.started(activity_)
        else:
            target.advanced(activity_)
    except Exception as err:
        _DROPPED.add(id(target))
        logger.warning(
            "The progress reporter %r failed and is dropped: %s", target, err
        )


@contextmanager
def reporter(target: Reporter | None) -> Generator[None]:
    """Report the activities which run in the context to a reporter.

    Args:
        target: what renders the activities, `None` for a run which reports
            nothing.
    """
    token = _REPORTER.set(target)
    try:
        yield
    finally:
        _REPORTER.reset(token)


def current_reporter() -> Reporter | None:
    """Get the reporter of the context, `None` outside of a reporting."""
    return _REPORTER.get()


def current_activity() -> Activity | None:
    """Get the activity of the context, `None` outside of one."""
    return _ACTIVITY.get()


@contextmanager
def activity(
    kind: str,
    label: str = "",
    *,
    total: int | None = None,
    workers: int = 1,
    parent: Activity | None = None,
) -> Generator[Activity]:
    """Report what a long operation does while it runs.

    The activity is reported to the reporter of the context, see `reporter`,
    and nests under the activity of the context or under `parent`. Outside of
    a reporting the activity is kept and nothing is rendered.

    Args:
        kind: what kind of operation it is, e.g. `"fit"`, `"scan"`.
        label: what it runs on, e.g. the id of the experiment.
        total: the steps in total, `None` while they are not known.
        workers: the steps which are processed at once, for the estimate of
            the time which is left.
        parent: the activity this one runs in, by default the one of the
            context.

    Yields:
        The activity, which the operation advances.
    """
    above = parent if parent is not None else _ACTIVITY.get()
    target = _REPORTER.get()
    if target is None and above is not None:
        target = above.reporter
    current = Activity(
        kind=kind,
        label=label,
        total=total,
        workers=workers,
        parent=above,
        reporter=target,
    )
    token = _ACTIVITY.set(current)
    report(target, "started", current)
    error: BaseException | None = None
    try:
        yield current
    except BaseException as err:
        error = err
        raise
    finally:
        _ACTIVITY.reset(token)
        report(target, "finished", current, error)
