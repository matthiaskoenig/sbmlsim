import pytest
import roadrunner
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_tolerances_serial():
    abs_tol = 1E-14
    rel_tol = 1E-14

    simulator = SimulatorSerial(
        model=MODEL_REPRESSILATOR,
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol
    )
    assert isinstance(simulator, SimulatorSerial)

    r = simulator.model.r  # type: roadrunner.RoadRunner
    integrator = r.getIntegrator()  # type: roadrunner.Integrator

    assert pytest.approx(rel_tol, integrator.getSetting("relative_tolerance"))
    assert abs_tol <= integrator.getSetting("absolute_tolerance")


def test_tolerances_parallel():
    abs_tol = 1E-14
    rel_tol = 1E-14

    simulator = SimulatorParallel(
        model=MODEL_REPRESSILATOR,
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol
    )
    assert isinstance(simulator, SimulatorParallel)

    r = simulator.model.r  # type: roadrunner.RoadRunner
    integrator = r.getIntegrator()  # type: roadrunner.Integrator

    assert pytest.approx(rel_tol, integrator.getSetting("relative_tolerance"))
    assert abs_tol <= integrator.getSetting("absolute_tolerance")
