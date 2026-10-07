"""The executor runs a plan with the initialization and change semantics of PEtab v2."""

import numpy as np
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import TimecourseResult
from sbmlsim.simulation import Change, Simulation, SteadyState
from sbmlsim.simulator.executor import SteadyStateError, execute
from sbmlsim.simulator.plan import compile_simulation
from tests.simulator.models import sbml

SEL = ["time", "[A]", "[B]", "X", "C", "k1"]

#: A + B at steady state, A = k2 / (k1 + k2) of the total
A_STEADY = 0.6 / 1.4


def run(sim: Simulation, model: RoadrunnerSBMLModel | None = None) -> TimecourseResult:
    """Run a simulation of the probe model."""
    model = model or RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(sim, model.symbols, model.uinfo)
    return execute(plan, model, SEL)


def test_preinit_reaches_initial_assignment() -> None:
    """A parameter before the initialization reaches the initial assignment of B."""
    # the rates are per volume of the compartment C = 2, the relaxation time
    # is 1 / 0.7, so the time 40 is at steady state
    res = run(Simulation(end=40, preinit_changes={"b0": 0.0}, times=[0, 40]))
    assert res["[B]"][0] == pytest.approx(0.0)
    assert res["[A]"][-1] == pytest.approx(A_STEADY, rel=1e-3)


def test_second_run_on_the_same_instance_uses_the_model_value() -> None:
    """A value of an earlier simulation does not leak into the next one."""
    model = RoadrunnerSBMLModel(source=sbml())
    run(
        Simulation(
            end=1, preinit_changes={"b0": 0.0}, changes=[Change(0.5, {"k1": 3.0})]
        ),
        model,
    )
    res = run(Simulation(end=1, times=[0, 1]), model)
    assert res["[B]"][0] == pytest.approx(1.0)
    assert res["k1"][0] == pytest.approx(0.8)
    assert res["C"][0] == pytest.approx(2.0)


def test_change_at_start_and_output_at_the_time_of_a_change() -> None:
    """The output at the time of a change is the state after the change."""
    res = run(
        Simulation(
            end=2,
            changes=[Change(0, {"[A]": 5.0}), Change(1, {"[A]": "[A] + 10"})],
            times=[0, 1, 2],
        )
    )
    assert res["[A]"][0] == pytest.approx(5.0)
    before = run(Simulation(end=1, changes=[Change(0, {"[A]": 5.0})], times=[0, 1]))
    assert res["[A]"][1] == pytest.approx(before["[A]"][1] + 10.0)


def test_simultaneous_formulas_use_the_old_values() -> None:
    """All values of a change are evaluated before any of them is assigned."""
    res = run(
        Simulation(
            end=1, changes=[Change(0.5, {"[A]": "[B]", "[B]": "[A]"})], times=[0.5]
        )
    )
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[B]"][0])
    assert res["[B]"][0] == pytest.approx(ref["[A]"][0])


def test_formula_reads_the_time() -> None:
    """`time` in a formula is the time of the change."""
    res = run(Simulation(end=2, changes=[Change(1.5, {"k1": "time * 2"})], times=[2]))
    assert res["k1"][0] == pytest.approx(3.0)


def test_compartment_change_keeps_concentrations_and_amounts() -> None:
    """A concentration species keeps its concentration, an amount species its amount."""
    res = run(Simulation(end=1, changes=[Change(0.5, {"C": 4.0})], times=[0.5]))
    ref = run(Simulation(end=0.5, times=[0.5]))
    assert res["[A]"][0] == pytest.approx(ref["[A]"][0])
    assert res["X"][0] == pytest.approx(ref["X"][0])
    assert res["C"][0] == pytest.approx(4.0)


def test_compartment_and_species_in_one_change() -> None:
    """A species set together with its compartment gets the value it is set to."""
    res = run(
        Simulation(end=1, changes=[Change(0.5, {"C": 4.0, "[A]": 7.0})], times=[0.5])
    )
    assert res["[A]"][0] == pytest.approx(7.0)


def test_negative_start_and_time_shift() -> None:
    """A simulation starts at a negative time, the time shift moves the result."""
    res = run(Simulation(start=-5, end=5, times=[-5, 0, 5], time_shift=5))
    np.testing.assert_allclose(res.time, [0.0, 5.0, 10.0])
    ref = run(Simulation(end=10, times=[0, 5, 10]))
    np.testing.assert_allclose(res["[A]"], ref["[A]"], rtol=1e-6)


def test_integrator_output_has_the_time_of_a_change_once() -> None:
    """The steps of the integrator are increasing and contain the change."""
    res = run(Simulation(end=2, changes=[Change(1, {"[A]": 5.0})]))
    times = res.time
    assert np.all(np.diff(times) > 0)
    k = int(np.flatnonzero(np.isclose(times, 1.0))[0])
    assert res["[A]"][k] == pytest.approx(5.0)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(2.0)


def test_multiple_dosing() -> None:
    """A change at a vector of times is applied at each of them."""
    res = run(
        Simulation(
            end=3,
            changes=[Change([0, 1, 2], {"X": "X + 1"})],
            times=[0, 1, 2, 3],
        )
    )
    np.testing.assert_allclose(res["X"], [13.0, 14.0, 15.0, 15.0])


def test_change_at_the_end() -> None:
    """A change at the end is applied and is the last output."""
    res = run(Simulation(end=1, changes=[Change(1, {"X": 0.0})], times=[0, 1]))
    assert res["X"][-1] == pytest.approx(0.0)


def test_steady_state_presimulation() -> None:
    """A presimulation starts the simulation in the steady state."""
    res = run(Simulation(end=1, presimulation=SteadyState(), times=[0]))
    assert res["[A]"][0] == pytest.approx(A_STEADY * 2.0, rel=1e-5)


def test_steady_state_with_its_preinit() -> None:
    """The pre-initialization of the steady state enters the initialization."""
    res = run(
        Simulation(
            end=1,
            presimulation=SteadyState(preinit_changes={"b0": 0.0}),
            times=[0],
        )
    )
    assert res["[A]"][0] == pytest.approx(A_STEADY, rel=1e-5)


def test_steady_state_not_reached_raises() -> None:
    """A model which does not reach a steady state fails with the time."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = RoadrunnerSBMLModel(source=growth)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(max_time=100)),
        model.symbols,
        model.uinfo,
    )
    with pytest.raises(SteadyStateError, match="100"):
        execute(plan, model, ["time", "x"])


def test_parameter_with_initial_assignment_preinit() -> None:
    """A parameter with an initial assignment is set before the initialization."""
    model = RoadrunnerSBMLModel(source=sbml())
    res = run(Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0]), model)
    assert res["X"][0] == pytest.approx(21.0)
    res = run(Simulation(end=1, times=[0]), model)
    assert res["X"][0] == pytest.approx(12.0)


def test_selections_without_time_get_it() -> None:
    """The result always has the time as its first column."""
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(
        Simulation(end=1, times=[0, 1]), model.symbols, model.uinfo
    )
    res = execute(plan, model, ["[A]"])
    assert res.columns == ("time", "[A]")


def test_integrator_setting_is_restored() -> None:
    """The output mode does not change the setting of the model."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.r_loaded.getIntegrator().setValue("variable_step_size", False)
    run(Simulation(end=1), model)
    assert model.r_loaded.getIntegrator().getValue("variable_step_size") is False


def test_change_at_the_end_keeps_the_earlier_output() -> None:
    """Only the output at the end is the state after a change at the end."""
    res = run(Simulation(end=1e6, changes=[Change(1e6, {"X": 0.0})], times=[0, 999995]))
    assert res.time.tolist() == [0.0, 999995.0]
    assert res["X"][1] == pytest.approx(12.0)


def test_steady_state_error_names_the_model() -> None:
    """The error of a steady state names the model even without an id."""
    growth = sbml("model g\n  x' = 1\n  x = 0\nend")
    model = RoadrunnerSBMLModel(source=growth)
    plan = compile_simulation(
        Simulation(end=1, presimulation=SteadyState(max_time=10)),
        model.symbols,
        model.uinfo,
    )
    with pytest.raises(SteadyStateError) as err:
        execute(plan, model, ["time", "x"])
    assert "'None'" not in str(err.value)
