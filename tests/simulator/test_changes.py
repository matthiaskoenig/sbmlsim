"""The changes and the events a change fires, independent of the engine."""

import logging

import libsbml
import pytest

from sbmlsim.model.symbols import EventSymbols, ModelSymbols
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator.changes import apply_changes, fire_events, triggers
from sbmlsim.simulator.plan import compile_simulation
from sbmlsim.units import UnitsInformation
from tests.simulator.models import sbml


class FakeValues:
    """Values of a model in the selections of roadrunner, backed by a dict.

    A concentration `[S]` is the amount of `S` over the size of its
    compartment; setting a compartment keeps the amounts, as roadrunner does.
    """

    def __init__(
        self, values: dict[str, float], compartments: dict[str, str] | None = None
    ) -> None:
        self.values = dict(values)
        self.compartments = compartments or {}

    def __getitem__(self, selection: str) -> float:
        return self.getValue(selection)

    def getValue(self, selection: str) -> float:
        if selection.startswith("["):
            species = selection[1:-1]
            return self.values[species] / self.values[self.compartments[species]]
        return self.values[selection]

    def setValue(self, selection: str, value: float) -> None:
        if selection.startswith("["):
            species = selection[1:-1]
            self.values[species] = value * self.values[self.compartments[species]]
        else:
            self.values[selection] = value


class FakeModel:
    """A model for the messages of the events."""

    def __init__(self, sid: str | None = None) -> None:
        self.sid = sid
        self.reported_triggers: set[str] = set()

    @property
    def label(self) -> str:
        return self.sid or "model"


EVENT_MODEL = """
model events
  X = 10; n = 0
  E: at (X >= 8): X = 0, n = n + 1
end
"""


def _events() -> tuple[EventSymbols, ...]:
    return ModelSymbols.from_sbml(sbml(EVENT_MODEL)).events


def test_fire_events_fires_an_event_a_change_triggers() -> None:
    values = FakeValues({"X": 10.0, "n": 0.0})
    assert fire_events(_events(), [False], values, 3.0, FakeModel()) == [0]
    assert values["X"] == 0.0 and values["n"] == 1.0


def test_fire_events_does_not_fire_an_event_whose_trigger_was_true() -> None:
    values = FakeValues({"X": 10.0, "n": 0.0})
    assert fire_events(_events(), [True], values, 3.0, FakeModel()) == []
    assert values["X"] == 10.0 and values["n"] == 0.0


def test_triggers_evaluates_the_triggers_at_the_values() -> None:
    events = _events()
    assert triggers(events, FakeValues({"X": 10.0, "n": 0.0}), 0.0, FakeModel()) == [
        True
    ]
    assert triggers(events, FakeValues({"X": 1.0, "n": 0.0}), 0.0, FakeModel()) == [
        False
    ]


def test_triggers_reports_a_trigger_it_does_not_evaluate_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    math = libsbml.writeMathMLToString(libsbml.parseL3Formula("rateOf(X) > 0"))
    event = EventSymbols(eid="R", trigger=math, delayed=False, assignments=())
    model = FakeModel()
    values = FakeValues({"X": 1.0})
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.changes"):
        assert triggers([event], values, 0.0, model) == [None]
        assert triggers([event], values, 0.0, model) == [None]
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "of the model 'model'" in messages[0]
    assert model.reported_triggers == {"R"}


COMPARTMENT_MODEL = """
model volume
  compartment c = 1
  species S in c
  S = 3
  p = 1
end
"""


def test_apply_changes_keeps_the_concentration_of_a_compartment_change() -> None:
    source = sbml(COMPARTMENT_MODEL)
    symbols = ModelSymbols.from_sbml(source)
    uinfo = UnitsInformation.from_sbml(source)
    plan = compile_simulation(
        Simulation(end=2, changes=[Change(1, {"c": 2.0})]), symbols, uinfo
    )
    (event,) = plan.events
    values = FakeValues({"c": 1.0, "S": 3.0, "p": 1.0}, compartments={"S": "c"})
    apply_changes(event, values, plan)
    assert values["c"] == 2.0
    assert values["[S]"] == 3.0
    assert values["S"] == 6.0


def test_apply_changes_evaluates_every_value_before_it_sets_one() -> None:
    source = sbml(COMPARTMENT_MODEL)
    symbols = ModelSymbols.from_sbml(source)
    uinfo = UnitsInformation.from_sbml(source)
    plan = compile_simulation(
        Simulation(end=2, changes=[Change(1, {"p": "S", "S": "p + time"})]),
        symbols,
        uinfo,
    )
    (event,) = plan.events
    values = FakeValues({"c": 1.0, "S": 3.0, "p": 1.0}, compartments={"S": "c"})
    apply_changes(event, values, plan)
    assert values["p"] == 3.0
    assert values["S"] == 2.0
