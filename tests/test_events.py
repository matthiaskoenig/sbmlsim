"""Tests of the events of a run."""

import json
from pathlib import Path

from sbmlsim import events
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.scan import Dimension, Scan
from sbmlsim.simulator import Simulator


def test_emit_outside_a_recording_does_nothing(tmp_path: Path) -> None:
    events.emit("x", a=1)
    assert list(tmp_path.iterdir()) == []


def test_recording_writes_one_json_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    with events.recording(path):
        events.emit("a", n=1)
        events.emit("b", name="x")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["a", "b"]
    assert lines[0]["n"] == 1 and lines[1]["name"] == "x"
    assert all("time" in line for line in lines)


def test_recording_is_a_context(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    with events.recording(path):
        events.emit("a")
    events.emit("b")
    assert len(path.read_text().splitlines()) == 1


def test_a_scan_in_a_recording_emits_chunks(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    simulator = Simulator(n_workers=1)
    model = simulator.load(REPRESSILATOR_SBML)
    simulation = Simulation(start=0, end=1, time_unit="min")
    scan = Scan(
        simulation,
        [Dimension(values={"PX": list(range(300))}, id="d")],
    )
    with events.recording(path):
        simulator.run(model, scan, progress=False)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    chunks = [line for line in lines if line["event"] == "scan.chunk"]
    assert chunks and chunks[-1]["done"] == 300 and chunks[-1]["total"] == 300
