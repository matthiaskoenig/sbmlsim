"""Simulations with uncertain values simulate their nominal values."""

from pathlib import Path

import numpy as np

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, Scan, Simulation, SteadyState
from sbmlsim.simulation.definition import Value
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import Uncertain
from tests.simulator.models import sbml_pk


def _simulation(dose: Value | Uncertain, ke: Value | Uncertain = 0.2) -> Simulation:
    return Simulation(
        end=24,
        steps=48,
        preinit_changes={"ke": ke},
        changes=[Change(0, {"PODOSE": dose})],
    )


def test_a_simulation_with_uncertain_values_simulates_the_nominal() -> None:
    simulator = Simulator(n_workers=1)
    uncertain = _simulation(Uncertain(Q(100, "mg"), cv=0.1), Uncertain(0.2, cv=0.2))
    nominal = _simulation(Q(100, "mg"))
    model = sbml_pk()
    a = simulator.simulate(model, uncertain)
    b = simulator.simulate(model, nominal)
    np.testing.assert_array_equal(a.values, b.values)
    scan = Dimension("ka_scan", values={"ka": np.array([0.5, 1.0])})
    ra = simulator.run(model, Scan(uncertain, [scan]))
    rb = simulator.run(model, Scan(nominal, [scan]))
    np.testing.assert_array_equal(ra["[C]"].values, rb["[C]"].values)
    # the plain values are the nominal values
    assert uncertain.changes[0].values == {"PODOSE": Q(100, "mg")}
    assert uncertain.preinit_changes == {"ke": 0.2}


def test_the_uncertain_values_of_a_simulation() -> None:
    dose = Uncertain(Q(100, "mg"), cv=0.1, name="dose")
    weight = Uncertain(Q(75, "kg"), sd=Q(10, "kg"))
    simulation = Simulation(
        end=48,
        preinit_changes={"BW": weight},
        changes=[Change([0, 24], {"PODOSE": dose, "ka": 1.0})],
        presimulation=SteadyState(preinit_changes={"V": Uncertain(Q(10, "l"), cv=0.2)}),
    )
    uses = simulation.uncertain_values()
    assert [(u.target, u.time, u.presimulation) for u in uses] == [
        ("BW", None, False),
        ("V", None, True),
        ("PODOSE", 0, False),
        ("PODOSE", 24, False),
    ]
    assert uses[2].value is dose and uses[3].value is dose
    assert simulation.nominal().uncertain_values() == []
    assert simulation.nominal().preinit_changes == {"BW": Q(75, "kg")}


def test_a_round_trip_of_a_simulation_with_uncertain_values(tmp_path: Path) -> None:
    simulation = _simulation(Uncertain(Q(100, "mg"), cv=0.1, name="dose"))
    again = Simulation.from_json(simulation.to_json())
    [use] = again.uncertain_values()
    assert use.value.name == "dose"
    assert use.value.to_dict() == simulation.uncertain_values()[0].value.to_dict()
    assert again.changes[0].values == {"PODOSE": Q(100, "mg")}
    # a simulation without uncertain values writes what it wrote before
    plain = _simulation(Q(100, "mg"))
    assert "uncertain" not in plain.to_json()


def test_a_replaced_value_is_no_longer_uncertain() -> None:
    simulation = _simulation(Uncertain(Q(100, "mg"), cv=0.1), Uncertain(0.2, cv=0.2))
    replaced = simulation.with_values({"PODOSE": Q(50, "mg")})
    assert [u.target for u in replaced.uncertain_values()] == ["ke"]
    assert replaced.changes[0].values == {"PODOSE": Q(50, "mg")}
    kept = simulation.with_preinit_defaults({"V": 10.0})
    assert [u.target for u in kept.uncertain_values()] == ["ke", "PODOSE"]
