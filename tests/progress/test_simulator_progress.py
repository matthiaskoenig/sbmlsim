"""The progress of the points of a scan."""

from __future__ import annotations

import logging

import pytest

from sbmlsim import progress
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    """The model of the tests of the simulator."""
    return RoadrunnerSBMLModel(source=sbml())


@pytest.fixture
def simulator() -> Simulator:
    """A simulator which runs the points of a scan in this process."""
    return Simulator(n_workers=1)


def _scan(points: int) -> Scan:
    """A scan of a number of points."""
    return Scan(
        Simulation(end=1, steps=2),
        [Dimension("dim1", values={"k1": [0.1 * (k + 1) for k in range(points)]})],
    )


def test_a_scan_reports_its_points_as_one_activity(recorder, model, simulator):
    with progress.reporter(recorder):
        simulator.run(model, _scan(4))
    scans = [e for e in recorder.events if e.kind == "scan"]
    assert scans
    assert (scans[-1].completed, scans[-1].total) == (4, 4)
    assert recorder.ends == [("scan", None)]


def test_a_single_simulation_reports_no_scan(recorder, model, simulator):
    with progress.reporter(recorder):
        simulator.run(model, Simulation(end=1, steps=2))
    assert recorder.events == []


def test_a_scan_without_progress_reports_nothing(recorder, model, simulator, caplog):
    """`progress=False` of a run reports no activity, see `scan_progress`."""
    with (
        caplog.at_level(logging.INFO, logger="sbmlsim.progress"),
        progress.reporter(recorder),
    ):
        simulator.run(model, _scan(4), progress=False)
    assert recorder.events == []
    assert caplog.records == []
