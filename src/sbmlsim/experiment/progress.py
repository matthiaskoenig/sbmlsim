"""The progress and the timings of a run of simulation experiments.

A run of `ExperimentRunner.run_experiments` shows its progress on the console of
the package, see `RunProgress`: a bar over the experiments and a row per
running experiment with what it does, e.g. its simulations or its figures, and
how far it got. The run ends with a table of the seconds every experiment took
in every phase, see `timings_table`.

The phases (`Phase`) are measured where they happen, by a `PhaseClock`: the
runner measures the loading of the models and the initialization of an
experiment, `SimulationExperiment.run` its simulations, data, figures and
outputs. The seconds of every phase are the `timings` of its
`ExperimentResult`, also of an experiment which ran in a worker process: the
worker measures them and sends them back with the result.

While a phase runs, its clock reports it as an `Event` to the listener of the
context, see `reporting`: the display of the run, or in a worker a queue which
the parent forwards to its display. Outside of a run of the runner, e.g. an
experiment run by `SimulationExperiment.run` directly, the phases are measured
and nothing is reported.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, override

from rich import box
from rich.console import Console, RenderableType
from rich.measure import Measurement
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

import sbmlsim.console

if TYPE_CHECKING:
    from sbmlsim.experiment.experiment import ExperimentResult

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    """A phase of the run of an experiment, the keys of its timings."""

    #: loading and compiling the models, which a runner does once per model
    MODELS = "models"
    #: creating the experiment and checking its definitions
    INITIALIZE = "initialize"
    #: running the tasks
    SIMULATIONS = "simulations"
    #: evaluating the data of the fit mappings
    DATA = "data"
    #: drawing the figures and writing their files
    FIGURES = "figures"
    #: writing the datasets, the results and the JSON of the experiment
    OUTPUTS = "outputs"
    #: the rest of the run of an experiment, e.g. what a `run` which a
    #: subclass overrides does around the run of its base class
    OTHER = "other"


#: the colour of every phase in the table of the timings
PHASE_STYLES: dict[Phase, str] = {
    Phase.MODELS: "cyan",
    Phase.INITIALIZE: "blue",
    Phase.SIMULATIONS: "green",
    Phase.DATA: "yellow",
    Phase.FIGURES: "magenta",
    Phase.OUTPUTS: "red",
    Phase.OTHER: "bright_black",
}

#: the characters of the longest bar of the table of the timings
BAR_WIDTH: int = 16

#: the characters of the name of an experiment in the progress, longer names
#: end in an ellipsis
NAME_WIDTH: int = 32

#: the characters a name keeps in the table of the timings before its bars
#: are left out, see `_bars_fit`
MIN_NAME_WIDTH: int = 16

#: the lines of the table of the timings: below the header, above the totals
_TABLE_BOX = box.Box(
    "    \n"  # top
    "    \n"  # header
    " ── \n"  # below the header
    "    \n"  # between the rows
    " ── \n"  # between the sections
    " ── \n"  # above the footer
    "    \n"  # footer
    "    \n"  # bottom
)


@dataclass(frozen=True, slots=True)
class Event:
    """What an experiment does, which its clock reports to the listener.

    Attributes:
        sid: the experiment.
        phase: the phase it is in.
        activity: what it does in the phase, e.g. `"writing figures"`.
        completed: the steps of the activity which are done.
        total: the steps of the activity, e.g. its figures, at least 1.
    """

    sid: str
    phase: Phase
    activity: str
    completed: int
    total: int


#: what receives the events of the experiments of a run
Listener = Callable[[Event], None]


@dataclass(frozen=True, slots=True)
class _Reporting:
    """The listener of a run of the runner, `None` if the run shows nothing."""

    listener: Listener | None


#: the reporting of the run of the runner, `None` outside of one
_REPORTING: ContextVar[_Reporting | None] = ContextVar(
    "sbmlsim_reporting", default=None
)

#: the clock of the phase which runs, see `advance`
_CLOCK: ContextVar[PhaseClock | None] = ContextVar("sbmlsim_clock", default=None)


@contextmanager
def reporting(listener: Listener | None) -> Generator[None]:
    """Report the phases of the experiments which run in the context.

    Args:
        listener: what receives the events, `None` for a run which shows no
            progress; the scans of its tasks then show none either, see
            `scan_progress`.
    """
    token = _REPORTING.set(_Reporting(listener))
    try:
        yield
    finally:
        _REPORTING.reset(token)


def scan_progress() -> bool | None:
    """Get the `progress` of `Simulator.run` for a task of an experiment.

    A run of the runner which shows no progress shows no progress bar of a
    scan either; otherwise the simulator decides, see `Simulator.run`, and in
    a run with progress a bar of a scan is shown below the rows of the run.

    Returns:
        `False` in a run of the runner without progress, `None` otherwise.
    """
    current = _REPORTING.get()
    return False if current is not None and current.listener is None else None


def advance(steps: int = 1) -> None:
    """Advance the activity of the phase which runs, nothing outside of one.

    Args:
        steps: the steps which are done, e.g. one figure.
    """
    clock = _CLOCK.get()
    if clock is not None:
        clock.advance(steps)


class PhaseClock:
    """Measure the phases of an experiment and report them to the listener.

    The listener is the one of the context the clock is created in, see
    `reporting`. Phases do not nest; a phase which runs twice, e.g. the
    figures which are drawn and then written, adds up.

    Attributes:
        sid: the experiment.
        timings: the seconds of every phase, by the value of the phase.
    """

    def __init__(self, sid: str) -> None:
        """Create the clock of an experiment, without any time."""
        self.sid = sid
        self.timings: dict[str, float] = {}
        current = _REPORTING.get()
        self._listener: Listener | None = (
            current.listener if current is not None else None
        )
        self._event: Event | None = None

    @contextmanager
    def phase(self, phase: Phase, activity: str, total: int = 1) -> Generator[None]:
        """Measure an activity of a phase, also when it raises.

        Args:
            phase: the phase whose time the activity adds to.
            activity: what the experiment does, shown in the progress.
            total: the steps of the activity, see `advance`.
        """
        token = _CLOCK.set(self)
        self._report(Event(self.sid, phase, activity, 0, max(total, 1)))
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - start)
            self._event = None
            _CLOCK.reset(token)

    def add(self, phase: Phase, seconds: float) -> None:
        """Add seconds to a phase, e.g. which were measured before the clock."""
        self.timings[phase.value] = self.timings.get(phase.value, 0.0) + seconds

    def advance(self, steps: int = 1) -> None:
        """Advance the activity which runs, see `advance`."""
        if self._event is not None:
            event = self._event
            self._report(replace(event, completed=event.completed + steps))

    def _report(self, event: Event) -> None:
        """Keep the event and report it; a failing listener never fails a run."""
        self._event = event
        if self._listener is None:
            return
        try:
            self._listener(event)
        except Exception:
            logger.debug(
                "The progress of '%s' is not reported", self.sid, exc_info=True
            )


def total_seconds(timings: Mapping[str, float]) -> float:
    """Get the seconds of all phases of an experiment."""
    return sum(timings.values())


def show_progress(progress: bool | None, console: Console | None = None) -> bool:
    """Decide whether a run shows its progress and timings.

    Args:
        progress: `True` or `False` as given; `None` shows them on an
            interactive terminal and in a notebook, and not when the output
            is redirected, e.g. into a file or a pipe, which would collect the
            frames of the bars, nor on a dumb terminal (`TERM=dumb`); rich
            reads `FORCE_COLOR`, `TTY_COMPATIBLE` and `TTY_INTERACTIVE` to
            override what it detects.
        console: the console of the output, the one of the package by default.

    Returns:
        Whether the progress and the timings are shown.
    """
    if progress is not None:
        return progress
    console = _console(console)
    return console.is_interactive or console.is_jupyter


class _StepsColumn(ProgressColumn):
    """The steps of a task, `3/7`; nothing for an activity of a single step."""

    @override
    def render(self, task: Task) -> Text:
        """Render the steps of a task."""
        if task.total is None or (task.total <= 1 and not task.fields.get("count")):
            return Text("")
        return Text(
            f"{int(task.completed)}/{int(task.total)}", style="progress.download"
        )


class RunProgress:
    """The progress of a run of experiments on the console.

    A bar over the experiments, which counts an experiment when it is done,
    and a row per running experiment with its activity and how far it got,
    which is removed when the experiment is done. An experiment in a worker
    reports its events through a queue, so the rows of a pool are the
    experiments the workers run.

    The display is safe to update from several threads: the events of the
    workers arrive in a thread of their own and the finished experiments of
    a pool in the thread of the pool.
    """

    def __init__(
        self, n_experiments: int, workers: int, console: Console | None = None
    ) -> None:
        """Create the display of a run, it starts with `start`.

        Args:
            n_experiments: the experiments of the run.
            workers: the processes which run them, for the overall row.
            console: the console of the output, the one of the package by
                default.
        """
        self._lock = threading.RLock()
        self._progress = Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn(
                "{task.description}",
                table_column=Column(no_wrap=True, max_width=NAME_WIDTH),
            ),
            TextColumn("{task.fields[activity]}", table_column=Column(no_wrap=True)),
            BarColumn(),
            _StepsColumn(),
            TimeElapsedColumn(),
            console=_console(console),
        )
        self._where = f"{workers} workers" if workers > 1 else ""
        self._overall = self._progress.add_task(
            "[bold]experiments", total=n_experiments, activity=self._where, count=True
        )
        self._rows: dict[str, TaskID] = {}
        self._done: set[str] = set()
        self._failed = 0

    def start(self) -> None:
        """Show the display."""
        self._progress.start()

    def stop(self) -> None:
        """End the display, its overall row stays on the console."""
        with self._lock:
            for task in self._rows.values():
                self._progress.remove_task(task)
            self._rows.clear()
        self._progress.stop()

    def __call__(self, event: Event) -> None:
        """Show what an experiment does, the listener of the run."""
        with self._lock:
            if event.sid in self._done:
                # a late event of a worker whose result arrived before it
                return
            task = self._rows.get(event.sid)
            if task is None:
                name = event.sid
                if len(name) > NAME_WIDTH - 2:
                    name = name[: NAME_WIDTH - 3] + "…"
                task = self._progress.add_task(
                    f"  {name}", total=event.total, activity=event.activity
                )
                self._rows[event.sid] = task
            self._progress.update(
                task,
                total=event.total,
                completed=event.completed,
                activity=f"[{PHASE_STYLES[event.phase]}]{event.activity}",
            )

    def note(self, sid: str, activity: str) -> None:
        """Show a note in the row of an experiment, e.g. that its worker died."""
        with self._lock:
            task = self._rows.get(sid)
            if task is not None:
                self._progress.update(task, activity=activity)

    def finished(self, sid: str, failed: bool = False) -> None:
        """Count an experiment as done, once.

        Args:
            sid: the experiment.
            failed: whether it or one of its figures failed.
        """
        with self._lock:
            if sid in self._done:
                return
            self._done.add(sid)
            task = self._rows.pop(sid, None)
            if task is not None:
                self._progress.remove_task(task)
            self._failed += failed
            activity = self._where
            if self._failed:
                failures = f"[red]{self._failed} failed"
                activity = f"{activity}, {failures}" if activity else failures
            self._progress.update(self._overall, advance=1, activity=activity)


def timings_table(
    results: Sequence[ExperimentResult],
    bars: bool = True,
) -> Table:
    """Get the table of the seconds every experiment took in every phase.

    A row per experiment with its seconds per phase, its total and a bar of
    its phases in their colours, the bars of all experiments at one scale;
    then the total of every phase over the experiments with the bar of the
    whole run, and the share of every phase. The headers of the phases have
    the colours of their parts of the bars. A phase which no experiment ran
    has no column, nor has `other` when it took no measurable time.

    In a run in a pool the experiments run at the same time, so the total
    over the experiments is more than the time of the run.

    Args:
        results: the results of the experiments, see `ExperimentResult.timings`.
        bars: show the bars, e.g. not on a console too narrow for them.

    Returns:
        The table.
    """
    phases = [
        phase
        for phase in Phase
        if any(phase.value in r.timings for r in results)
        and (
            phase != Phase.OTHER
            or round(sum(r.timings.get(phase.value, 0.0) for r in results), 2) > 0
        )
    ]
    table = Table(
        box=_TABLE_BOX,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
    )
    # the names give way on a narrow console, the numbers do not
    table.add_column("experiment", min_width=10, max_width=NAME_WIDTH)
    for phase in phases:
        table.add_column(
            phase.value,
            justify="right",
            no_wrap=True,
            min_width=len(phase.value),
            header_style=f"bold {PHASE_STYLES[phase]}",
        )
    table.add_column("total", justify="right", no_wrap=True, min_width=5)
    if bars:
        table.add_column("", no_wrap=True, width=BAR_WIDTH)

    totals = [total_seconds(r.timings) for r in results]
    longest = max(totals, default=0.0)
    for k, (result, total) in enumerate(zip(results, totals, strict=True)):
        # a name is cut and not wrapped, its mark stays in front of it
        name = Text.assemble(
            ("✗ ", "red") if result.failed else ("✓ ", "green"),
            (result.experiment.sid, "red" if result.failed else ""),
            no_wrap=True,
            overflow="ellipsis",
        )
        bar = _bar(result.timings, phases, total / longest if longest else 0.0)
        table.add_row(
            name,
            *(_seconds(result.timings.get(phase.value)) for phase in phases),
            Text(f"{total:.2f}", style="bold"),
            *([bar] if bars else []),
            end_section=k == len(results) - 1,
        )

    sums = {
        phase.value: sum(r.timings.get(phase.value, 0.0) for r in results)
        for phase in phases
    }
    overall = sum(totals)
    table.add_row(
        Text("total", style="bold"),
        *(Text(f"{sums[phase.value]:.2f}", style="bold") for phase in phases),
        Text(f"{overall:.2f}", style="bold"),
        *([_bar(sums, phases, 1.0 if overall else 0.0)] if bars else []),
    )
    table.add_row(
        Text("share", style="dim"),
        *(_share(sums[phase.value], overall) for phase in phases),
        _share(overall, overall),
        *([""] if bars else []),
    )
    return table


def show_timings(
    results: Sequence[ExperimentResult],
    initialized: float | None,
    elapsed: float,
    where: str,
    console: Console | None = None,
) -> None:
    """Print the timings of a run below a section, see `timings_table`.

    Args:
        results: the results of the experiments of the run.
        initialized: the seconds of the initialization of the experiments
            before the run, when the runner was created, `None` for none,
            e.g. in a second run of a runner.
        elapsed: the seconds of the run.
        where: where the experiments ran, e.g. `"on 4 workers"`.
        console: the console of the output, the one of the package by default.
    """
    if not results:
        return
    console = _console(console)
    n = len(results)
    noun = "experiment" if n == 1 else "experiments"
    before = "" if initialized is None else f", initialized in {initialized:.2f} s and"
    caption = f"{n} {noun}{before} run in {elapsed:.2f} s {where}"
    console.line()
    console.rule(
        ":stopwatch: [bold]Timings[/bold] in seconds", align="left", style="white"
    )
    console.print(timings_table(results, bars=_bars_fit(results, console)))
    console.print(caption, markup=False)


def _bars_fit(results: Sequence[ExperimentResult], console: Console) -> bool:
    """Check whether the bars of the timings fit on the console.

    The numbers are what matters: the names give way to the bars down to
    `MIN_NAME_WIDTH`, the bars give way on a console narrower than that.
    """
    table = timings_table(results, bars=False)
    unlimited = console.options.update(max_width=sys.maxsize)
    width = Measurement.get(console, unlimited, table).maximum
    names = max(
        (min(len(r.experiment.sid) + 2, NAME_WIDTH) for r in results), default=0
    )
    # the bar and the gap in front of it
    bars = BAR_WIDTH + 2
    return width - names + min(names, MIN_NAME_WIDTH) + bars <= console.width


def _console(console: Console | None) -> Console:
    """Get a console, the one of the package for `None`.

    The console of the package is looked up when it is used, so a script or
    a test which replaces it, e.g. by a console which records, gets the
    output.
    """
    return console if console is not None else sbmlsim.console.console


def _seconds(seconds: float | None) -> Text:
    """Get the seconds of a phase, a dim dash for a phase which did not run."""
    if seconds is None:
        return Text("-", style="dim")
    return Text(f"{seconds:.2f}")


def _share(seconds: float, total: float) -> Text:
    """Get the share of a phase in the total, in percent."""
    return Text(f"{100 * seconds / total:.0f}%" if total else "-", style="dim")


def _bar(
    timings: Mapping[str, float], phases: Iterable[Phase], fraction: float
) -> RenderableType:
    """Get a bar of the phases in their colours, `fraction` of `BAR_WIDTH` long.

    The boundaries of the parts are rounded from the cumulative seconds, so
    the parts add up to the length of the bar.
    """
    phases = list(phases)
    total = sum(timings.get(phase.value, 0.0) for phase in phases)
    length = round(BAR_WIDTH * fraction)
    bar = Text()
    if total <= 0 or length <= 0:
        return bar
    cumulative, drawn = 0.0, 0
    for phase in phases:
        cumulative += timings.get(phase.value, 0.0)
        end = round(length * cumulative / total)
        if end > drawn:
            bar.append("━" * (end - drawn), style=PHASE_STYLES[phase])
            drawn = end
    return bar
