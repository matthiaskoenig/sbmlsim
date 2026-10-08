"""The serial simulator compiles simulations into plans and runs them."""

import numpy as np
import pytest

from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial
from tests.simulator.models import TOLERANCE_PROBE, sbml


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


def _vector(simulator: SimulatorSerial) -> list[float]:
    """Get the sorted absolute tolerances which CVODE uses."""
    integrator = simulator.r_loaded.getIntegrator()
    return sorted(float(v) for v in integrator.getAbsoluteToleranceVector())


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


def test_the_tolerance_of_every_state_reaches_cvode(tmp_path) -> None:
    """The tolerances per state are the vector of CVODE, after a new model too."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    simulator = SimulatorSerial(model=path, absolute_tolerance=tolerance)
    # A and S are concentration species in C = 2 and U = 1e-12 (raised to
    # 2e-6), X an amount species, D a parameter with a rate rule
    expected = sorted([1e-8 * 2, 1e-8 * 2e-6, 1e-9, 1e-7])
    assert _vector(simulator) == pytest.approx(expected)
    simulator.set_model(path)
    assert _vector(simulator) == pytest.approx(expected)
    table = simulator.model_loaded.tolerances()
    assert list(table["sid"]) == simulator.model_loaded.state_ids()
    assert set(table["kind"]) == {"amount", "concentration", "other"}


def test_a_float_tolerance_is_the_same_for_every_kind(tmp_path) -> None:
    """The scaling of roadrunner by the initial values is not used."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    simulator = SimulatorSerial(model=path, absolute_tolerance=1e-10)
    expected = sorted([1e-10 * 2, 1e-10 * 2e-6, 1e-10, 1e-10])
    assert _vector(simulator) == pytest.approx(expected)


def test_a_degenerate_compartment_is_logged(tmp_path, caplog) -> None:
    """A compartment whose volume is raised to the floor is logged once per model."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    with caplog.at_level("WARNING"):
        simulator = SimulatorSerial(model=path)
        simulator.set_integrator_settings(absolute_tolerance=1e-9)
    messages = [r.getMessage() for r in caplog.records if "'U'" in r.getMessage()]
    assert len(messages) == 1
