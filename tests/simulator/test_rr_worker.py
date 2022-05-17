"""Test SimulationWorkerRR."""
import pandas as pd
import pytest

from sbmlsim.simulation import TimecourseSim, Timecourse
from sbmlsim.simulator.rr_worker import SimulationWorkerRR


def test_init() -> None:
    """Test init without arguments."""
    worker = SimulationWorkerRR()
    assert worker


def test_set_model(repressilator_model_state: str) -> None:
    """Test setting model."""
    worker = SimulationWorkerRR()
    worker.set_model(repressilator_model_state)
    assert worker


def test_set_default_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections"""
    worker = SimulationWorkerRR()
    worker.set_model(repressilator_model_state)
    worker.set_timecourse_selections()
    assert worker
    assert "time" in worker.r.timeCourseSelections


def test_set_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections"""
    worker = SimulationWorkerRR()
    worker.set_model(repressilator_model_state)
    worker.set_timecourse_selections(["time"])
    assert worker
    assert "time" in worker.r.timeCourseSelections
    assert len(worker.r.timeCourseSelections) == 1


def test_default_integrator_settings(repressilator_model_state: str) -> None:
    """Test default integrator settings."""
    worker = SimulationWorkerRR()
    worker.set_model(repressilator_model_state)
    assert worker.get_integrator_setting("variable_step_size") is False
    assert worker.get_integrator_setting("stiff") is True
    assert pytest.approx(1E-8) == worker.get_integrator_setting("relative_tolerance")


def test_timecourse(repressilator_model_state: str) -> None:
    """Test timecourse simulation."""
    worker = SimulationWorkerRR()
    worker.set_model(repressilator_model_state)
    simulation = TimecourseSim([
        Timecourse(start=0, end=5, steps=5)
    ])
    df: pd.DataFrame = worker._timecourse(simulation)
    assert len(df) == 6
    assert "time" in df
