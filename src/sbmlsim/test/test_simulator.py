import pytest
import roadrunner

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.test import MODEL_REPRESSILATOR


def _tolerance_test(r: roadrunner.RoadRunner, abs_tol: float, rel_tol: float):
    """Check that tolerances are set correctly."""
    integrator = r.getIntegrator()  # type: roadrunner.Integrator

    assert pytest.approx(rel_tol, integrator.getSetting("relative_tolerance"))
    assert abs_tol <= integrator.getSetting("absolute_tolerance")


def test_tolerances_serial():
    abs_tol = 1e-14
    rel_tol = 1e-14
    simulator = SimulatorSerial(
        model=MODEL_REPRESSILATOR,
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
    )
    assert isinstance(simulator, SimulatorSerial)
    _tolerance_test(r=simulator.model.r, abs_tol=abs_tol, rel_tol=rel_tol)


def test_tolerances_serial2():
    abs_tol = 1e-14
    rel_tol = 1e-14
    simulator = SimulatorSerial(
        model=RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR),
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
    )
    assert isinstance(simulator, SimulatorSerial)
    _tolerance_test(r=simulator.model.r, abs_tol=abs_tol, rel_tol=rel_tol)


def test_tolerances_parallel():
    abs_tol = 1e-14
    rel_tol = 1e-14

    simulator = SimulatorParallel(
        model=MODEL_REPRESSILATOR,
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
    )
    assert isinstance(simulator, SimulatorParallel)
    _tolerance_test(r=simulator.model.r, abs_tol=abs_tol, rel_tol=rel_tol)


def test_tolerances_parallel2():
    abs_tol = 1e-14
    rel_tol = 1e-14

    simulator = SimulatorParallel(
        model=RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR),
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
    )
    assert isinstance(simulator, SimulatorParallel)
    _tolerance_test(r=simulator.model.r, abs_tol=abs_tol, rel_tol=rel_tol)
