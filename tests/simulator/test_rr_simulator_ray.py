"""Test SimulatarRayRR."""
from pathlib import Path
from typing import List

import numpy as np
import pytest

from sbmlsim.result import XResult



import pytest

from sbmlsim.simulation import TimecourseSim, Timecourse, ScanSim, Dimension
from sbmlsim.simulator.rr_simulator_ray import SimulatorRayRR, SimulatorActor, ray


def test_from_sbml(repressilator_path: Path) -> None:
    """Test setting model."""
    simulator = SimulatorRayRR.from_sbml(sbml_path=repressilator_path, actor_count=1)
    assert simulator


def test_init() -> None:
    """Test init without arguments."""
    simulator = SimulatorRayRR(actor_count=1)
    assert simulator


def test_set_model(repressilator_model_state: str) -> None:
    """Test setting model."""
    simulator = SimulatorRayRR(actor_count=1)
    simulator.set_model(repressilator_model_state)
    assert simulator


def test_set_default_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections"""
    simulator = SimulatorRayRR(actor_count=1)
    simulator.set_model(repressilator_model_state)
    simulator.set_timecourse_selections()
    assert simulator
    for worker in simulator.workers:
        selections_ref = worker.get_timecourse_selections.remote()
        selections: List[str] = ray.get(selections_ref)
        assert "time" in selections


def test_set_timecourse_selections(repressilator_model_state: str) -> None:
    """Test setting timecourse selections"""
    simulator = SimulatorRayRR(actor_count=1)
    simulator.set_model(repressilator_model_state)
    simulator.set_timecourse_selections(["time"])
    assert simulator
    worker: SimulatorActor
    for worker in simulator.workers:
        selections_ref = worker.get_timecourse_selections.remote()
        selections: List[str] = ray.get(selections_ref)
        assert "time" in selections
        assert len(selections) == 1


def test_default_integrator_settings(repressilator_model_state: str) -> None:
    """Test default integrator settings."""
    simulator = SimulatorRayRR(actor_count=1)
    simulator.set_model(repressilator_model_state)
    for worker in simulator.workers:
        setting_ref = worker.get_integrator_setting.remote("variable_step_size")
        assert ray.get(setting_ref) is False

        setting_ref = worker.get_integrator_setting.remote("stiff")
        assert ray.get(setting_ref) is True

        setting_ref = worker.get_integrator_setting.remote("relative_tolerance")
        assert pytest.approx(1E-8) == ray.get(setting_ref)


def test_run_timecourse(repressilator_model_state: str) -> None:
    """Test timecourse simulation."""
    simulator = SimulatorRayRR(actor_count=1)
    simulator.set_model(repressilator_model_state)
    simulation = TimecourseSim([
        Timecourse(start=0, end=5, steps=5)
    ])
    xres: XResult = simulator.run_timecourse(simulation)
    assert xres


def test_run_scan(repressilator_model_state: str) -> None:
    """Test scan simulation."""
    simulator = SimulatorRayRR(actor_count=1)
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
        ]
    )
    xres: XResult = simulator.run_scan(scan)
    assert xres

    # assert len(df) == 6
    # assert "time" in df
