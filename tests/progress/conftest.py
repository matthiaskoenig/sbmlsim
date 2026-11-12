"""Fixtures of the tests of the progress of a run."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sbmlsim.progress import Activity


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What an activity was when a reporter was called."""

    kind: str
    label: str
    completed: int
    total: int | None
    depth: int


def snapshot(activity: Activity) -> Snapshot:
    """Snapshot an activity, which changes while it runs."""
    return Snapshot(
        kind=activity.kind,
        label=activity.label,
        completed=activity.completed,
        total=activity.total,
        depth=activity.depth,
    )


class Recorder:
    """A reporter which keeps a snapshot of every call."""

    def __init__(self) -> None:
        self.events: list[Snapshot] = []
        self.starts: list[Snapshot] = []
        self.ends: list[tuple[str, BaseException | None]] = []

    def started(self, activity: Activity) -> None:
        self.events.append(snapshot(activity))
        self.starts.append(snapshot(activity))

    def advanced(self, activity: Activity) -> None:
        self.events.append(snapshot(activity))

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        self.events.append(snapshot(activity))
        self.ends.append((activity.kind, error))


@pytest.fixture
def recorder() -> Recorder:
    """A reporter which keeps a snapshot of every call."""
    return Recorder()


class Clock:
    """A monotonic clock which only moves when it is told to."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    """A monotonic clock which only moves when it is told to."""
    return Clock()
