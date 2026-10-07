"""Test simulations of the repressilator."""

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator import SimulatorSerial


@pytest.fixture
def simulator() -> SimulatorSerial:
    """Get a simulator of the repressilator."""
    return SimulatorSerial(RoadrunnerSBMLModel(REPRESSILATOR_SBML))


def test_simulation(simulator: SimulatorSerial) -> None:
    """A simulation with a grid has its points, a change before the start is there."""
    xres = simulator.run_simulation(Simulation(end=100, steps=100))
    assert len(xres["time"]) == 101

    xres = simulator.run_simulation(
        Simulation(end=100, steps=100, preinit_changes={"PX": 10.0})
    )
    assert xres["time"].values[-1] == 100.0
    assert xres["[PX]"].values[0] == 10.0

    xres = simulator.run_simulation(
        Simulation(end=100, steps=100, preinit_changes={"[X]": 10.0})
    )
    assert xres["[X]"].values[0] == 10.0


def test_simulation_with_the_steps_of_the_integrator(
    simulator: SimulatorSerial,
) -> None:
    """Without times or steps the output are the steps of the integrator."""
    xres = simulator.run_simulation(Simulation(end=100))
    time = xres["time"].values
    assert time[0] == 0.0
    assert time[-1] == pytest.approx(100.0)
    assert np.all(np.diff(time) > 0)


def test_changes_at_times(simulator: SimulatorSerial) -> None:
    """A change at several times sets its value at each of them."""
    xres = simulator.run_simulation(
        Simulation(
            end=150,
            changes=[Change([0, 50, 100], {"X": 10})],
            times=[0, 50, 100, 150],
        )
    )
    assert xres["time"].values[-1] == 150.0
    assert xres["X"].values[:3].tolist() == [10.0, 10.0, 10.0]


def test_presimulation_to_steady_state(simulator: SimulatorSerial) -> None:
    """A steady state presimulation starts the simulation where nothing changes."""
    # the oscillation of the repressilator is damped for a small `n`
    xres = simulator.run_simulation(
        Simulation(
            end=100,
            preinit_changes={"n": 1.0},
            presimulation=SteadyState(),
            times=[0, 100],
        )
    )
    assert xres["[X]"].values[0] == pytest.approx(xres["[X]"].values[-1], rel=1e-4)
