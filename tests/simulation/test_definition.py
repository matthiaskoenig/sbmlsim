"""The definition of a simulation: Simulation, Change and SteadyState."""

import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Simulation, SteadyState


def test_change_scalar_and_vector_times() -> None:
    """A change is at one time or at a vector of times."""
    assert Change(10, {"k": 1.0}).times == (10,)
    assert Change([0, 24, 48], {"PODOSE": Q(10, "mg")}).times == (0, 24, 48)
    assert Change(Q([0, 1], "hr"), {"k": 1.0}).times == (Q(0, "hr"), Q(1, "hr"))


def test_change_requires_values() -> None:
    """A change without values is an error."""
    with pytest.raises(ValueError, match="no values"):
        Change(0, {})


def test_change_requires_times() -> None:
    """A change without times is an error."""
    with pytest.raises(ValueError, match="no times"):
        Change([], {"k": 1.0})


def test_simulation_validates_interval_and_output() -> None:
    """The interval, the output and the pre-initialization values are checked."""
    with pytest.raises(ValueError, match="start"):
        Simulation(start=10, end=5)
    with pytest.raises(ValueError, match=r"times.*steps"):
        Simulation(end=10, times=[0, 5], steps=10)
    with pytest.raises(ValueError, match="steps"):
        Simulation(end=10, steps=0)
    with pytest.raises(ValueError, match="formula"):
        Simulation(end=10, preinit_changes={"k": "2 * a"})  # ty: ignore[invalid-argument-type]


def test_simulation_interval_with_quantities() -> None:
    """Quantities of the interval are compared in their units."""
    with pytest.raises(ValueError, match="start"):
        Simulation(time_unit="hr", start=Q(120, "min"), end=1)
    sim = Simulation(time_unit="hr", start=Q(30, "min"), end=1)
    assert sim.end == 1


def test_simulation_rejects_conflicting_presimulation_targets() -> None:
    """A target of the simulation and of the presimulation is ambiguous."""
    with pytest.raises(ValueError, match="'k'"):
        Simulation(
            end=10,
            preinit_changes={"k": 1.0},
            presimulation=SteadyState(preinit_changes={"k": 2.0}),
        )


def test_steady_state_rejects_formulas() -> None:
    """The pre-initialization changes of a steady state are numbers."""
    with pytest.raises(ValueError, match="'k' is a formula"):
        SteadyState(preinit_changes={"k": "a + 1"})  # ty: ignore[invalid-argument-type]


def test_targets() -> None:
    """The targets are the ones of the pre-initialization and of the changes."""
    sim = Simulation(
        end=10,
        preinit_changes={"BW": Q(70, "kg")},
        changes=[Change(0, {"PODOSE": Q(1, "mg")}), Change(5, {"[A]": "[A] + 1"})],
    )
    assert sim.targets() == {"BW", "PODOSE", "[A]"}


def test_with_values_replaces_everywhere_or_adds_preinit() -> None:
    """A value replaces the target wherever it is set, else it is added."""
    sim = Simulation(
        end=72,
        preinit_changes={"BW": Q(70, "kg")},
        changes=[Change([0, 24, 48], {"PODOSE": Q(10, "mg")})],
    )
    scanned = sim.with_values({"PODOSE": Q(20, "mg"), "k": 2.0, "BW": Q(80, "kg")})
    assert scanned.changes[0].values["PODOSE"] == Q(20, "mg")
    assert scanned.preinit_changes == {"BW": Q(80, "kg"), "k": 2.0}
    assert sim.changes[0].values["PODOSE"] == Q(10, "mg")
    assert sim.preinit_changes == {"BW": Q(70, "kg")}


def test_json_round_trip() -> None:
    """A simulation is written as JSON and read back."""
    sim = Simulation(
        time_unit="hr",
        start=-72,
        end=48,
        preinit_changes={"BW": Q(70, "kg")},
        changes=[
            Change([-72, 0], {"PODOSE": Q(10, "mg")}),
            Change(Q(600, "min"), {"[glc]": "[glc] + 5"}),
        ],
        presimulation=SteadyState(preinit_changes={"ins": 0.0}, max_time=1e6),
        steps=100,
        time_shift=72,
    )
    again = Simulation.from_json(sim.to_json())
    assert again.to_dict() == sim.to_dict()
    assert again.changes[1].times == (Q(600, "min"),)
    assert again.presimulation is not None
    assert again.presimulation.max_time == 1e6


def test_json_file(tmp_path) -> None:
    """A path writes the JSON into a file."""
    sim = Simulation(end=10, times=[0, 5, 10])
    path = tmp_path / "sim.json"
    assert sim.to_json(path) is None
    assert Simulation.from_json(path).times == (0, 5, 10)


def test_with_values_replaces_a_target_of_the_presimulation() -> None:
    """A value of a target of the steady state replaces it there."""
    sim = Simulation(end=1, presimulation=SteadyState(preinit_changes={"k": 0.2}))
    changed = sim.with_values({"k": 5.0})
    assert changed.presimulation is not None
    assert changed.presimulation.preinit_changes == {"k": 5.0}
    assert changed.preinit_changes == {}


def test_with_preinit_defaults_only_fills_preinit() -> None:
    """Defaults are pre-initialization changes, the simulation's own win."""
    sim = Simulation(
        end=1,
        preinit_changes={"a": 1.0},
        changes=[Change(0.5, {"k1": 3.0})],
        presimulation=SteadyState(preinit_changes={"s": 1.0}),
    )
    filled = sim.with_preinit_defaults({"a": 9.0, "k1": 2.0, "s": 7.0, "b": 4.0})
    assert filled.preinit_changes == {"a": 1.0, "k1": 2.0, "b": 4.0}
    assert filled.changes[0].values == {"k1": 3.0}
    assert filled.presimulation is not None
    assert filled.presimulation.preinit_changes == {"s": 1.0}
