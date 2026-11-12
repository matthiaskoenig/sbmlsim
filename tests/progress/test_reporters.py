"""The bar, the line and the event of the progress of a run."""

from __future__ import annotations

import io
import json
import logging

import pytest
from rich.console import Console

from sbmlsim import events, progress


@pytest.fixture
def ticking(clock, monkeypatch):
    """A clock which the activities and the reporters share."""
    monkeypatch.setattr(progress.core, "_now", clock)
    return clock


def test_resolve_picks_the_bar_on_a_terminal(monkeypatch):
    monkeypatch.delenv("SBMLSIM_PROGRESS", raising=False)
    console = Console(file=io.StringIO(), force_terminal=True)
    resolved = progress.resolve("auto", console=console)
    assert [type(r) for r in resolved.reporters] == [
        progress.BarReporter,
        progress.EventReporter,
    ]


def test_resolve_picks_the_log_reporter_without_a_terminal(monkeypatch):
    monkeypatch.delenv("SBMLSIM_PROGRESS", raising=False)
    console = Console(file=io.StringIO(), force_terminal=False)
    resolved = progress.resolve("auto", console=console)
    assert [type(r) for r in resolved.reporters] == [
        progress.LogReporter,
        progress.EventReporter,
    ]


def test_resolve_reads_the_environment(monkeypatch):
    monkeypatch.setenv("SBMLSIM_PROGRESS", "none")
    resolved = progress.resolve(None)
    assert [type(r) for r in resolved.reporters] == [progress.EventReporter]


def test_resolve_takes_the_booleans_of_the_old_arguments(monkeypatch):
    monkeypatch.delenv("SBMLSIM_PROGRESS", raising=False)
    console = Console(file=io.StringIO(), force_terminal=False)
    assert [type(r) for r in progress.resolve(False).reporters] == [
        progress.EventReporter
    ]
    assert [type(r) for r in progress.resolve(True, console=console).reporters] == [
        progress.LogReporter,
        progress.EventReporter,
    ]


def test_resolve_refuses_an_unknown_setting():
    with pytest.raises(ValueError, match="bars"):
        progress.resolve("bars")


def test_log_reporter_writes_the_first_line_at_once_then_every_interval(
    caplog, ticking
):
    caplog.set_level(logging.INFO, logger="sbmlsim.progress")
    reporter = progress.LogReporter(interval=30.0, now=ticking)
    with (
        progress.reporter(reporter),
        progress.activity("identifiability", "scans", total=70) as a,
    ):
        ticking.tick(100.0)
        a.advance()
        lines = [r.message for r in caplog.records]
        assert len(lines) == 1
        assert "identifiability scans 1/70" in lines[0]
        assert "eta" in lines[0]
        ticking.tick(10.0)
        a.advance()
        assert len(caplog.records) == 1
        ticking.tick(30.0)
        a.advance()
        assert len(caplog.records) == 2
    assert "identifiability scans 3/70" in caplog.records[-1].message


def test_log_reporter_writes_a_last_line_when_the_activity_finishes(caplog, ticking):
    caplog.set_level(logging.INFO, logger="sbmlsim.progress")
    reporter = progress.LogReporter(interval=30.0, now=ticking)
    with progress.reporter(reporter), progress.activity("fit", "NMP", total=2) as a:
        a.update(completed=2)
    assert "fit NMP 2/2" in caplog.records[-1].message
    assert "done" in caplog.records[-1].message


def test_log_reporter_names_an_activity_which_failed(caplog, ticking):
    caplog.set_level(logging.INFO, logger="sbmlsim.progress")
    with (
        pytest.raises(ValueError, match="boom"),
        progress.reporter(progress.LogReporter(now=ticking)),
        progress.activity("fit", "NMP", total=2),
    ):
        raise ValueError("boom")
    assert "failed" in caplog.records[-1].message


def test_event_reporter_throttles_to_one_event_per_second(tmp_path, ticking):
    path = tmp_path / "progress.jsonl"
    with (
        events.recording(path),
        progress.reporter(progress.EventReporter(throttle=1.0, now=ticking)),
        progress.activity("scan", "hctz", total=3) as a,
    ):
        a.advance()
        a.advance()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["progress"] * 3
    assert [r["state"] for r in records] == ["started", "running", "finished"]
    assert records[-1]["completed"] == 2
    assert records[-1]["kind"] == "scan"
    assert records[-1]["label"] == "hctz"


def test_bar_reporter_draws_an_activity_and_its_child():
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=120)
    with (
        progress.reporter(progress.BarReporter(console=console)),
        progress.activity("fit", "NMP", total=2) as run,
    ):
        run.advance()
        with run.child("jacobian", total=4) as jac:
            jac.advance(2)
    output = out.getvalue()
    assert "fit NMP" in output
    assert "jacobian" in output
    assert "1/2" in output


def test_a_reporter_which_fails_does_not_stop_the_others(caplog, tmp_path, ticking):
    class Exploding:
        def started(self, activity):
            raise RuntimeError("no console")

        def advanced(self, activity):
            raise RuntimeError("no console")

        def finished(self, activity, error):
            raise RuntimeError("no console")

    path = tmp_path / "progress.jsonl"
    multi = progress.MultiReporter(
        [Exploding(), progress.EventReporter(throttle=0.0, now=ticking)]
    )
    with (
        events.recording(path),
        progress.reporter(multi),
        progress.activity("scan", total=2) as a,
    ):
        a.advance()
    assert len(path.read_text().splitlines()) == 3
    assert "progress reporter" in caplog.text
