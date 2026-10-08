"""Test simulations of the repressilator."""

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator import Simulator

SIMULATOR = Simulator(n_workers=1)


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    """Get the repressilator."""
    return RoadrunnerSBMLModel(REPRESSILATOR_SBML)


def test_simulation(model: RoadrunnerSBMLModel) -> None:
    """A simulation with a grid has its points, a change before the start is there."""
    res = SIMULATOR.run(model, Simulation(end=100, steps=100))
    assert len(res["time"]) == 101

    res = SIMULATOR.run(
        model, Simulation(end=100, steps=100, preinit_changes={"PX": 10.0})
    )
    assert res["time"].values[-1] == 100.0
    assert res["[PX]"].values[0] == 10.0

    res = SIMULATOR.run(
        model, Simulation(end=100, steps=100, preinit_changes={"[X]": 10.0})
    )
    assert res["[X]"].values[0] == 10.0


def test_simulation_with_the_steps_of_the_integrator(
    model: RoadrunnerSBMLModel,
) -> None:
    """Without times or steps the output are the steps of the integrator."""
    res = SIMULATOR.run(model, Simulation(end=100))
    assert res.ragged
    time = res["time"].values
    assert time[0] == 0.0
    assert time[-1] == pytest.approx(100.0)
    assert np.all(np.diff(time) > 0)


def test_changes_at_times(model: RoadrunnerSBMLModel) -> None:
    """A change at several times sets its value at each of them."""
    res = SIMULATOR.run(
        model,
        Simulation(
            end=150,
            changes=[Change([0, 50, 100], {"X": 10})],
            times=[0, 50, 100, 150],
        ),
    )
    assert res["time"].values[-1] == 150.0
    assert res["X"].values[:3].tolist() == [10.0, 10.0, 10.0]


def test_presimulation_to_steady_state(model: RoadrunnerSBMLModel) -> None:
    """A steady state presimulation starts the simulation where nothing changes."""
    # the oscillation of the repressilator is damped for a small `n`
    res = SIMULATOR.run(
        model,
        Simulation(
            end=100,
            preinit_changes={"n": 1.0},
            presimulation=SteadyState(),
            times=[0, 100],
        ),
    )
    assert res["[X]"].values[0] == pytest.approx(res["[X]"].values[-1], rel=1e-4)
