"""The activity tree of the progress of a long run."""

from __future__ import annotations

import pytest

from sbmlsim import progress


class Exploding:
    """A reporter which fails on every call."""

    def started(self, activity) -> None:
        raise RuntimeError("no console")

    def advanced(self, activity) -> None:
        raise RuntimeError("no console")

    def finished(self, activity, error) -> None:
        raise RuntimeError("no console")


def test_activity_advances_and_reports(recorder):
    with progress.reporter(recorder), progress.activity("fit", "NMP", total=3) as a:
        a.advance()
        a.advance(2)
    assert [(e.kind, e.completed, e.total) for e in recorder.events] == [
        ("fit", 0, 3),
        ("fit", 1, 3),
        ("fit", 3, 3),
        ("fit", 3, 3),
    ]
    assert recorder.ends == [("fit", None)]


def test_activity_updates_its_total(recorder):
    with progress.reporter(recorder), progress.activity("scan") as a:
        a.update(total=7, completed=2)
    assert (recorder.events[-1].completed, recorder.events[-1].total) == (2, 7)


def test_child_activity_nests_under_its_parent(recorder):
    with (
        progress.reporter(recorder),
        progress.activity("fit", total=2) as run,
        run.child("jacobian", total=36) as jac,
    ):
        assert jac.parent is run
        assert jac.depth == 1
        assert run.depth == 0
    assert [e.kind for e in recorder.starts] == ["fit", "jacobian"]


def test_an_activity_inside_another_nests_without_being_named(recorder):
    with (
        progress.reporter(recorder),
        progress.activity("fit"),
        progress.activity("jacobian") as jac,
    ):
        assert jac.depth == 1


def test_activity_closed_by_an_exception_reports_the_error(recorder):
    with (
        pytest.raises(ValueError, match="boom"),
        progress.reporter(recorder),
        progress.activity("scan", total=5) as a,
    ):
        a.advance()
        raise ValueError("boom")
    kind, error = recorder.ends[0]
    assert kind == "scan"
    assert isinstance(error, ValueError)


def test_a_reporter_which_raises_is_dropped_and_the_body_runs(caplog):
    done = False
    with progress.reporter(Exploding()), progress.activity("scan", total=2) as a:
        a.advance()
        done = True
    assert done
    assert "progress reporter" in caplog.text


def test_without_a_reporter_an_activity_is_a_no_op():
    with progress.activity("scan", total=2) as a:
        a.advance()
    assert a.completed == 1
    assert progress.current_reporter() is None


def test_remaining_estimates_the_seconds_which_are_left():
    assert progress.remaining(completed=10, total=20, elapsed=100.0) == pytest.approx(
        100.0
    )
    assert progress.remaining(
        completed=10, total=20, elapsed=100.0, workers=4
    ) == pytest.approx(66.6666, rel=1e-4)
    assert progress.remaining(completed=0, total=20, elapsed=5.0) is None
    assert progress.remaining(completed=20, total=20, elapsed=5.0) == 0.0


def test_an_activity_estimates_its_own_remaining_time(clock, monkeypatch):
    monkeypatch.setattr(progress.core, "_now", clock)
    with progress.activity("scan", total=4, workers=1) as a:
        clock.tick(10.0)
        a.advance()
        assert a.elapsed == pytest.approx(10.0)
        assert a.remaining() == pytest.approx(30.0)
