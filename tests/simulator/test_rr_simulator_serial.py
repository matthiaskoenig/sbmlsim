"""Test SimulationWorkerRR."""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim.result import XResult
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator.rr_simulator_serial import SimulatorSerialRR


def test_from_sbml(repressilator_path: Path) -> None:
    """Test setting model."""
    simulator = SimulatorSerialRR.from_sbml(sbml_path=repressilator_path)
    assert simulator


def test_init() -> None:
    """Test init without arguments."""
    simulator = SimulatorSerialRR()
    assert simulator


def test_set_model(repressilator_model_state: str) -> None:
    """Test setting model."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    assert simulator


def test_set_default_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    simulator.set_timecourse_selections()
    assert simulator
    assert "time" in simulator.worker.r.timeCourseSelections


def test_set_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    simulator.set_timecourse_selections(["time"])
    assert simulator
    assert "time" in simulator.worker.r.timeCourseSelections
    assert len(simulator.worker.r.timeCourseSelections) == 1


def test_default_integrator_settings(repressilator_model_state: str) -> None:
    """Test default integrator settings."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    assert simulator.worker.get_integrator_setting("variable_step_size") is False
    assert simulator.worker.get_integrator_setting("stiff") is True
    assert pytest.approx(1e-8) == simulator.worker.get_integrator_setting(
        "relative_tolerance"
    )


def test_run_timecourse(repressilator_model_state: str) -> None:
    """Test timecourse simulation."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    simulation = TimecourseSim([Timecourse(start=0, end=5, steps=5)])
    xres: XResult = simulator.run_timecourse(simulation)
    assert xres


def test_run_scan(repressilator_model_state: str) -> None:
    """Test scan simulation."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    scan = ScanSim(
        simulation=TimecourseSim([Timecourse(start=0, end=5, steps=5)]),
        dimensions=[
            Dimension(
                "dim1",
                changes={
                    "n": np.linspace(start=2, stop=10, num=8),
                },
            )
        ],
    )
    xres: XResult = simulator.run_scan(scan)
    assert xres

    # assert len(df) == 6
    # assert "time" in df
