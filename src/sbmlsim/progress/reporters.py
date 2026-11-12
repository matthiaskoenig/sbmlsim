"""What renders the activities of a run: a bar, a log line, an event.

The activities of a run, see `sbmlsim.progress.core`, are rendered by three
reporters, and `resolve` chooses between them:

- `BarReporter` draws a rich bar per activity on the console, nested by its
  depth, with the steps, the elapsed time and the estimate of the time which
  is left. It needs a terminal.
- `LogReporter` writes a line per activity every 30 seconds, so a run in a
  log file, in a pipe or in continuous integration says how far it is.
- `EventReporter` records the progress in the events of the run, see
  `sbmlsim.events`, so something outside the process (`simrun`, an agent)
  follows the run by reading `progress.jsonl`.

Every rendering says what of the machine the run uses, the cores, the CPU and
the memory of `sbmlsim.progress.resources`: the bar of the outermost activity,
the end of every line and the field `resources` of every event.

The rendering is chosen by `--progress auto|bar|log|none` of the run, by the
environment variable `SBMLSIM_PROGRESS` or by the argument of a function which
takes one, e.g. `show_progress` of a fit. `auto` draws bars on a terminal and
writes lines without one. The events are recorded whatever is chosen, they
cost a line in a file and nothing on the console.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Sequence
from typing import override

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

import sbmlsim.console
from sbmlsim import events
from sbmlsim.progress.core import Activity, Reporter, _now, report
from sbmlsim.progress.resources import SAMPLER, format_resources

#: what writes the lines of the progress of a run
logger = logging.getLogger("sbmlsim.progress")

#: the renderings of the progress, the values of `--progress`
NAMES: tuple[str, ...] = ("auto", "bar", "log", "none")

#: the environment variable which sets the rendering of the progress
ENV_NAME = "SBMLSIM_PROGRESS"

#: the seconds between two lines of `LogReporter`
LOG_INTERVAL = 30.0

#: the seconds between two events of an activity, see `EventReporter`
EVENT_THROTTLE = 1.0


def _clock(seconds: float | None) -> str:
    """Format seconds as `H:MM:SS`, `-:--:--` for a time which is not known."""
    if seconds is None:
        return "-:--:--"
    return str(datetime.timedelta(seconds=round(seconds)))


def _name(activity: Activity) -> str:
    """Get the kind and the label of an activity, e.g. `fit NMP`."""
    return f"{activity.kind} {activity.label}".strip()


def _line(activity: Activity, suffix: str = "") -> str:
    """Write what an activity is at, the line of `LogReporter`.

    Args:
        activity: the activity.
        suffix: what happened to it, e.g. `done`.

    Returns:
        The line, e.g. `identifiability scans 1/70 (1%) elapsed 0:01:40 eta
        1:56:40`.
    """
    parts = [_name(activity)]
    if activity.total:
        percent = 100.0 * activity.completed / activity.total
        parts.append(f"{activity.completed}/{activity.total} ({percent:.0f}%)")
    else:
        parts.append(f"{activity.completed} steps")
    parts.append(f"elapsed {_clock(activity.elapsed)}")
    parts.append(f"eta {_clock(activity.remaining())}")
    parts.append(format_resources(SAMPLER.sample(workers=activity.workers)))
    if suffix:
        parts.append(suffix)
    return " ".join(parts)


class ResourcesColumn(ProgressColumn):
    """What of the machine the run uses, on the outermost activity only.

    The resources are those of the process, not of the activity, so one line
    carries them and the bars of the activities inside stay readable, see
    `sbmlsim.progress.resources`.
    """

    @override
    def render(self, task: Task) -> Text:
        """Render the resources of the run on an outermost task."""
        activity = task.fields.get("activity")
        if not isinstance(activity, Activity) or activity.depth:
            return Text("")
        return Text(
            format_resources(SAMPLER.sample(workers=activity.workers)),
            style="progress.remaining",
        )


class RemainingColumn(ProgressColumn):
    """The estimate of the time which is left, `eta H:MM:SS`.

    The estimate is the one of the activity, see `Activity.remaining`, which
    is corrected for the steps its workers process at once. The rate of rich
    is not used, it is measured over a window of seconds and a step of a run
    takes minutes.
    """

    @override
    def render(self, task: Task) -> Text:
        """Render the estimate of a task."""
        activity = task.fields.get("activity")
        left = activity.remaining() if isinstance(activity, Activity) else None
        return Text(f"eta {_clock(left)}", style="progress.remaining")


class BarReporter:
    """Draw a bar per activity on the console of the package.

    An activity which runs in another one is indented by its depth and its bar
    is removed when it ends; the outermost activities stay until the run ends.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Create the reporter and its display, which is started on demand.

        Args:
            console: the console of the bars, by default the one of the
                package.
        """
        self.console = console if console is not None else sbmlsim.console.console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            RemainingColumn(),
            ResourcesColumn(),
            console=self.console,
            transient=False,
        )
        self._tasks: dict[int, TaskID] = {}
        self._open = 0

    def _description(self, activity: Activity) -> str:
        """Get the description of the bar of an activity, indented by depth."""
        return "  " * activity.depth + _name(activity)

    def started(self, activity: Activity) -> None:
        """Add the bar of an activity, and start the display with the first."""
        if self._open == 0:
            self.progress.start()
        self._open += 1
        self._tasks[activity.uid] = self.progress.add_task(
            self._description(activity), total=activity.total, activity=activity
        )

    def advanced(self, activity: Activity) -> None:
        """Update the bar of an activity."""
        task_id = self._tasks.get(activity.uid)
        if task_id is not None:
            self.progress.update(
                task_id,
                completed=activity.completed,
                total=activity.total,
                description=self._description(activity),
            )

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Complete the bar of an activity, and stop the display with the last."""
        task_id = self._tasks.pop(activity.uid, None)
        if task_id is not None:
            if activity.depth:
                self.progress.remove_task(task_id)
            else:
                self.progress.update(task_id, completed=activity.completed)
        self._open = max(self._open - 1, 0)
        if self._open == 0:
            self.progress.stop()


class LogReporter:
    """Write a line per activity, every interval and at its end.

    A run without a terminal (a log file, a pipe, continuous integration) says
    how far it is this way. The interval keeps the file short: the first line
    of an activity is written when it is older than the interval, so the
    activities which are over in a moment, e.g. the scans of a fit, write no
    line at all.
    """

    def __init__(
        self,
        interval: float | None = None,
        *,
        target: logging.Logger | None = None,
        now: object = _now,
    ) -> None:
        """Create the reporter.

        Args:
            interval: the seconds between two lines of an activity,
                `LOG_INTERVAL` by default.
            target: what writes the lines, by default the logger of the
                progress.
            now: the monotonic clock, for the tests.
        """
        self.interval = LOG_INTERVAL if interval is None else interval
        self.logger = target if target is not None else logger
        self._now = now
        self._last: dict[int, float] = {}
        self._written: set[int] = set()

    def started(self, activity: Activity) -> None:
        """Note the start of an activity, whose line comes after the interval."""
        self._last[activity.uid] = activity.started_at

    def advanced(self, activity: Activity) -> None:
        """Write the line of an activity, at most every interval."""
        now = float(self._now())  # ty: ignore[call-non-callable]
        last = self._last.get(activity.uid, activity.started_at)
        if now - last < self.interval:
            return
        self._last[activity.uid] = now
        self._written.add(activity.uid)
        self.logger.info("%s", _line(activity))

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Write the last line of an activity which wrote one, took a while or failed."""
        self._last.pop(activity.uid, None)
        written = activity.uid in self._written
        self._written.discard(activity.uid)
        if error is None and not written and activity.elapsed < self.interval:
            return
        self.logger.info("%s", _line(activity, "failed" if error else "done"))


class EventReporter:
    """Record the progress of the activities in the events of the run.

    Something outside the process follows the run by reading `progress.jsonl`:
    one `progress` event when an activity starts, one at most every second
    while it runs and one when it ends. Outside of a recording the events cost
    a lookup, see `sbmlsim.events.emit`.
    """

    def __init__(self, throttle: float = EVENT_THROTTLE, *, now: object = _now) -> None:
        """Create the reporter.

        Args:
            throttle: the seconds between two events of an activity.
            now: the monotonic clock, for the tests.
        """
        self.throttle = throttle
        self._now = now
        self._last: dict[int, float] = {}

    def _emit(self, activity: Activity, state: str, error: str | None = None) -> None:
        """Record the state of an activity."""
        events.emit(
            "progress",
            state=state,
            kind=activity.kind,
            label=activity.label,
            completed=activity.completed,
            total=activity.total,
            depth=activity.depth,
            elapsed=activity.elapsed,
            eta=activity.remaining(),
            uid=activity.uid,
            error=error,
            resources=SAMPLER.sample(workers=activity.workers).as_dict(),
        )

    def started(self, activity: Activity) -> None:
        """Record that an activity started."""
        self._last[activity.uid] = float("-inf")
        self._emit(activity, "started")

    def advanced(self, activity: Activity) -> None:
        """Record what an activity is at, at most every throttle."""
        now = float(self._now())  # ty: ignore[call-non-callable]
        last = self._last.get(activity.uid, float("-inf"))
        if now - last < self.throttle:
            return
        self._last[activity.uid] = now
        self._emit(activity, "running")

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Record that an activity ended, and what it ended by."""
        self._last.pop(activity.uid, None)
        self._emit(activity, "finished", None if error is None else repr(error))


class MultiReporter:
    """Report to several reporters, a failing one is dropped, see `report`.

    Attributes:
        reporters: what renders the activities, in the order they are called.
    """

    def __init__(self, reporters: Sequence[Reporter]) -> None:
        """Create the reporter of several reporters."""
        self.reporters: list[Reporter] = list(reporters)

    def started(self, activity: Activity) -> None:
        """Report the start of an activity to every reporter."""
        for target in self.reporters:
            report(target, "started", activity)

    def advanced(self, activity: Activity) -> None:
        """Report a step of an activity to every reporter."""
        for target in self.reporters:
            report(target, "advanced", activity)

    def finished(self, activity: Activity, error: BaseException | None) -> None:
        """Report the end of an activity to every reporter."""
        for target in self.reporters:
            report(target, "finished", activity, error)


def _setting_name(setting: str | bool | None) -> str:
    """Get the rendering of a setting, the environment or `auto`.

    Args:
        setting: the value of `--progress`, `True` for `auto`, `False` for
            `none`, `None` for the environment variable `SBMLSIM_PROGRESS` or
            `auto`.

    Returns:
        One of `NAMES`.

    Raises:
        ValueError: if the setting is no rendering.
    """
    if setting is True:
        return "auto"
    if setting is False:
        return "none"
    if setting is None:
        setting = os.environ.get(ENV_NAME) or "auto"
    name = str(setting).strip().lower()
    if name not in NAMES:
        raise ValueError(f"The progress '{setting}' is none of {NAMES}.")
    return name


def resolve(
    setting: str | bool | None = None, console: Console | None = None
) -> MultiReporter:
    """Get the reporter of a setting of the progress.

    The events are recorded whatever is chosen, see `EventReporter`; `none`
    means that nothing is rendered for a human.

    Args:
        setting: `auto`, `bar`, `log`, `none`, `True` for `auto`, `False` for
            `none` or `None` for the environment variable `SBMLSIM_PROGRESS`.
        console: the console of the bars, by default the one of the package.

    Returns:
        The reporter, which is installed with `sbmlsim.progress.reporter`.
    """
    name = _setting_name(setting)
    if name == "auto":
        target = console if console is not None else sbmlsim.console.console
        name = "bar" if target.is_terminal else "log"
    rendering: Reporter | None = None
    if name == "bar":
        rendering = BarReporter(console=console)
    elif name == "log":
        rendering = LogReporter()
    return MultiReporter(
        [target for target in (rendering, EventReporter()) if target is not None]
    )
