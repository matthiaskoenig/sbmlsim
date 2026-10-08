"""The serial simulator compiles simulations into plans and runs them."""

import numpy as np
import pytest

from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial
from tests.simulator.models import sbml


@pytest.fixture
def simulator(tmp_path) -> SimulatorSerial:
    """Get a simulator of the probe model."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    return SimulatorSerial(model=path)


def test_run_simulation(simulator: SimulatorSerial) -> None:
    """A simulation is a result of its selections without a scan dimension."""
    xres = simulator.run_simulation(
        Simulation(end=1, preinit_changes={"b0": 0.0}, steps=10)
    )
    assert xres["[B]"].dims == ("_point",)
    assert xres["[B]"].values[0] == pytest.approx(0.0)
    assert len(xres["time"]) == 11


def test_run_scan(simulator: SimulatorSerial) -> None:
    """A scan is a result with a dimension per dimension of the scan."""
    scan = ScanSim(
        Simulation(end=1, steps=10),
        [Dimension("d", changes={"b0": np.array([0.0, 2.0])})],
    )
    xres = simulator.run_scan(scan)
    assert xres["[B]"].dims == ("_point", "d")
    assert xres["[B]"].values[0].tolist() == pytest.approx([0.0, 2.0])


def test_simulate_with_selections(simulator: SimulatorSerial) -> None:
    """The selections of the simulator are the columns of the result."""
    simulator.set_timecourse_selections(["time", "[A]", "X"])
    result = simulator.simulate(
        Simulation(end=2, changes=[Change(1, {"X": 0.0})], times=[0, 1, 2])
    )
    assert isinstance(result, TimecourseResult)
    assert result.columns == ("time", "[A]", "X")
    np.testing.assert_allclose(result["X"], [12.0, 0.0, 0.0])


def test_the_roadrunner_instance_follows_a_derived_model(
    simulator: SimulatorSerial,
) -> None:
    """A model which is derived for a pre-initialization change is the one run."""
    result = simulator.simulate(
        Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0])
    )
    assert result["X"][0] == pytest.approx(21.0)
    assert simulator.r_loaded is simulator.model_loaded.r_loaded


def test_integrator_settings_are_passed_on_and_kept(tmp_path) -> None:
    """Every setting of the integrator reaches roadrunner, also for a later model."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    simulator = SimulatorSerial(model=path, initial_time_step=1e-9)
    simulator.set_integrator_settings(maximum_num_steps=1234)
    # a new model, as for every task of an experiment
    simulator.set_model(path)
    integrator = simulator.r_loaded.getIntegrator()
    assert integrator.getValue("initial_time_step") == pytest.approx(1e-9)
    assert integrator.getValue("maximum_num_steps") == 1234


def test_an_unknown_integrator_setting_is_an_error(simulator: SimulatorSerial) -> None:
    """A setting the integrator does not have is not dropped silently."""
    with pytest.raises(ValueError, match="has no settings"):
        simulator.set_integrator_settings(initial_step=1e-9)
