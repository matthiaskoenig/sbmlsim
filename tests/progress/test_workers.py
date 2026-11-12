"""The progress of the activities which run in a worker process."""

from __future__ import annotations

import queue
import threading
import time

from sbmlsim import progress
from sbmlsim.progress.workers import Snapshot, forwarding, worker_reporting


def _wait(condition, timeout: float = 5.0) -> bool:
    """Wait for a condition of another thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_a_worker_reports_through_the_queue(recorder):
    messages: queue.Queue = queue.Queue()
    with worker_reporting(messages), progress.activity("scan", "hctz", total=4) as a:
        a.advance(2)
    assert messages.qsize() == 3
    with forwarding(messages, target=recorder):
        assert _wait(lambda: len(recorder.ends) == 1)
    assert [(e.kind, e.label, e.completed, e.total) for e in recorder.events] == [
        ("scan", "hctz", 0, 4),
        ("scan", "hctz", 2, 4),
        ("scan", "hctz", 2, 4),
    ]


def test_a_nested_activity_of_a_worker_keeps_its_depth(recorder):
    messages: queue.Queue = queue.Queue()
    with (
        worker_reporting(messages),
        progress.activity("fit", total=2) as run,
        run.child("jacobian", total=4) as jacobian,
    ):
        jacobian.advance()
    with forwarding(messages, target=recorder):
        assert _wait(lambda: len(recorder.ends) == 2)
    assert {(e.kind, e.depth) for e in recorder.events} == {
        ("fit", 0),
        ("jacobian", 1),
    }


def test_the_activities_of_a_worker_nest_under_the_activity_of_the_parent(recorder):
    messages: queue.Queue = queue.Queue()
    with worker_reporting(messages), progress.activity("scan", total=1) as a:
        a.advance()
    with (
        progress.reporter(recorder),
        progress.activity("experiments", total=1) as run,
        forwarding(messages, target=recorder, parent=run),
    ):
        assert _wait(lambda: len(recorder.ends) == 1)
    assert {(e.kind, e.depth) for e in recorder.events} == {
        ("experiments", 0),
        ("scan", 1),
    }


def test_reporting_without_a_queue_is_a_no_op(recorder):
    with worker_reporting(None), progress.activity("scan", total=2) as a:
        a.advance()
    with forwarding(None, target=recorder):
        pass
    assert recorder.events == []


def test_the_forwarding_thread_stops_with_its_context(recorder):
    messages: queue.Queue = queue.Queue()
    names = {thread.name for thread in threading.enumerate()}
    with forwarding(messages, target=recorder):
        assert _wait(
            lambda: "sbmlsim-progress" in {t.name for t in threading.enumerate()}
        )
    assert _wait(
        lambda: {t.name for t in threading.enumerate()} <= names | {"MainThread"}
    )


def test_the_forwarding_thread_hands_other_messages_to_the_listener():
    messages: queue.Queue = queue.Queue()
    received: list[object] = []
    messages.put("an event of an experiment")
    with forwarding(messages, listener=received.append):
        assert _wait(lambda: received == ["an event of an experiment"])


def test_a_snapshot_of_an_activity_is_picklable():
    import pickle

    snapshot = Snapshot(
        pid=1,
        uid=2,
        parent_uid=None,
        kind="scan",
        label="hctz",
        completed=1,
        total=3,
        workers=1,
        elapsed=1.0,
        state="running",
        error=None,
    )
    assert pickle.loads(pickle.dumps(snapshot)) == snapshot


def test_a_run_of_experiments_reports_how_many_are_done(tmp_path, recorder):
    """The run of the experiments is an activity with the count and the eta."""
    from sbmlsim.experiment import ExperimentRunner
    from sbmlsim.simulator import Simulator
    from tests.experiment.test_experiment_run import FitMappingExperiment

    runner = ExperimentRunner(FitMappingExperiment, simulator=Simulator())
    with progress.reporter(recorder):
        runner.run_experiments(output_path=tmp_path, save_results=False, progress=False)
    run = [e for e in recorder.events if e.kind == "experiments"]
    assert (run[0].completed, run[0].total) == (0, 1)
    assert (run[-1].completed, run[-1].total) == (1, 1)
    assert recorder.ends == [("experiments", None)]


def test_a_run_of_experiments_logs_its_progress_without_a_terminal(tmp_path, caplog):
    """A run in a log file says how far it is, which a bar cannot."""
    import logging

    from sbmlsim.experiment import ExperimentRunner
    from sbmlsim.simulator import Simulator
    from tests.experiment.test_experiment_run import FitMappingExperiment

    runner = ExperimentRunner(FitMappingExperiment, simulator=Simulator())
    with caplog.at_level(logging.INFO, logger="sbmlsim.progress"):
        runner.run_experiments(output_path=tmp_path, save_results=False, progress=None)
    lines = [r.message for r in caplog.records if "experiments" in r.message]
    assert lines
    assert "experiments 1/1" in lines[-1]
    assert "cores" in lines[-1]
