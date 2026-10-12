"""External changes preserve the ordering and values of SBML events."""

import logging

import libsbml
import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.executor import MAX_EXECUTIONS_PER_EVENT
from tests.simulator.models import sbml


def _document(
    body: str, priorities: dict[str, str], functions: str = ""
) -> libsbml.SBMLDocument:
    doc = libsbml.readSBMLFromString(sbml(f"{functions}\nmodel order\n{body}\nend"))
    for name, priority in priorities.items():
        doc.getModel().getEvent(name).createPriority().setMath(
            libsbml.parseL3Formula(priority)
        )
    return doc


def _run(doc: libsbml.SBMLDocument, selection: str) -> list[float]:
    model = RoadrunnerSBMLModel(
        source=libsbml.writeSBMLToString(doc), selections=["time", selection]
    )
    result = Simulator(n_workers=1).simulate(
        model,
        Simulation(end=2, changes=[Change(1, {"q": 1})], times=[0, 1, 2]),
    )
    return result[selection].tolist()


@pytest.mark.parametrize("reverse", [False, True])
def test_event_priorities_override_document_order(reverse: bool) -> None:
    events = [
        "low: at (q > 0), t0=true: X=1;",
        "high: at (q > 0), t0=true: X=10;",
    ]
    if reverse:
        events.reverse()
    doc = _document("q=0; X=0;\n" + "\n".join(events), {"low": "1", "high": "10"})
    assert _run(doc, "X") == [0, 0, 1, 1]


@pytest.mark.parametrize("capture, expected", [(True, 0), (False, 10)])
def test_assignment_values_use_the_event_capture_setting(
    capture: bool, expected: float
) -> None:
    doc = _document(
        "q=0; X=0; Y=0;\n"
        "low: at (q > 0), t0=true: Y=X;\n"
        "high: at (q > 0), t0=true: X=10;",
        {"low": "1", "high": "10"},
    )
    doc.getModel().getEvent("low").setUseValuesFromTriggerTime(capture)
    assert _run(doc, "Y") == [0, 0, expected, expected]


@pytest.mark.parametrize("persistent, expected", [(True, 1), (False, 0)])
def test_a_pending_event_obeys_trigger_persistence(
    persistent: bool, expected: float
) -> None:
    doc = _document(
        "q=0; X=0;\nlow: at (q > 0), t0=true: X=1;\nhigh: at (q > 0), t0=true: q=0;",
        {"low": "1", "high": "10"},
    )
    doc.getModel().getEvent("low").getTrigger().setPersistent(persistent)
    assert _run(doc, "X") == [0, 0, expected, expected]


def test_priorities_are_evaluated_after_each_event() -> None:
    doc = _document(
        "q=0; X=0; p=1;\n"
        "first: at (q > 0), t0=true: p=20;\n"
        "third: at (q > 0), t0=true: X=3;\n"
        "second: at (q > 0), t0=true: X=2;",
        {"first": "10", "second": "p", "third": "5"},
    )
    assert _run(doc, "X") == [0, 0, 3, 3]


def test_events_of_the_same_priority_execute_in_document_order() -> None:
    # `later` is pending before `first`, which `trigger` makes pending
    doc = _document(
        "q=0; r=0; X=0;\n"
        "first: at (r > 0), t0=true: X=1;\n"
        "later: at (q > 0), t0=true: X=2;\n"
        "trigger: at (q > 0), t0=true: r=1;",
        {"trigger": "10"},
    )
    assert _run(doc, "X") == [0, 0, 2, 2]


def test_many_independent_events_do_not_exhaust_the_cycle_limit() -> None:
    events = "\n".join(f"e{k}: at (q > 0), t0=true: X=X+1;" for k in range(110))
    doc = _document("q=0; X=0;\n" + events, {})
    for event in doc.getModel().getListOfEvents():
        event.setUseValuesFromTriggerTime(False)
    assert _run(doc, "X") == [0, 0, 110, 110]


def test_events_which_trigger_each_other_stop_at_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    doc = _document(
        "q=0; X=0;\n"
        "on: at (q > 0 && X < 0.5), t0=true: X=1;\n"
        "off: at (X > 0.5), t0=true: X=0;",
        {},
    )
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.executor"):
        _run(doc, "X")
    cycles = [r.getMessage() for r in caplog.records if "cycle" in r.getMessage()]
    assert len(cycles) == 1
    assert f"more than {MAX_EXECUTIONS_PER_EVENT} times" in cycles[0]


@pytest.mark.parametrize("priority", ["floor(20.5)", "twice(10)"])
def test_a_priority_the_simulator_cannot_evaluate_counts_as_zero(
    priority: str, caplog: pytest.LogCaptureFixture
) -> None:
    doc = _document(
        "q=0; X=0;\n"
        "unknown: at (q > 0), t0=true: X=1;\n"
        "known: at (q > 0), t0=true: X=10;",
        {"unknown": priority, "known": "1"},
        functions="function twice(x)\n  2*x\nend",
    )
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.executor"):
        assert _run(doc, "X") == [0, 0, 1, 1]
    warnings = [r.getMessage() for r in caplog.records if "priority" in r.getMessage()]
    assert len(warnings) == 1
    assert "'unknown'" in warnings[0]


#: the event of necrosis of the model of the liver of pvl in pkdb_models: an
#: assignment rule which a change sets triggers it
NECROSIS = (
    "atp = 4; threshold = 1; alt = 1.5; alt_ext = 0;\n"
    "necrosis := piecewise(1, atp < threshold, 0);\n"
    "necrosis_event: at (necrosis > 0.5), t0=false: alt_ext = alt, alt = 0;"
)


def test_a_change_fires_an_event_with_numbers_with_units() -> None:
    """The numbers of the math of sbmlutils carry units (#293)."""
    doc = _document(NECROSIS, {})
    event: libsbml.Event = doc.getModel().getEvent("necrosis_event")
    event.getTrigger().setMath(libsbml.parseL3Formula("necrosis > 0.5 dimensionless"))
    event.getEventAssignment("alt").setMath(libsbml.parseL3Formula("0 mole"))
    model = RoadrunnerSBMLModel(
        source=libsbml.writeSBMLToString(doc),
        selections=["time", "necrosis", "alt", "alt_ext"],
    )
    result = Simulator(n_workers=1).simulate(
        model,
        Simulation(end=2, changes=[Change(1, {"threshold": 1000})], times=[0, 1, 2]),
    )
    assert result["necrosis"].tolist() == [0, 0, 1, 1]
    assert result["alt"].tolist() == [1.5, 1.5, 0, 0]
    assert result["alt_ext"].tolist() == [0, 0, 1.5, 1.5]


def test_a_trigger_the_simulator_cannot_evaluate_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A change cannot fire the event, the warning names it, once per model."""
    doc = _document(
        "q=0; X=0;\n"
        "unknown: at (twice(q) > 1), t0=true: X=1;\n"
        "known: at (q > 0), t0=true: X=10;",
        {},
        functions="function twice(x)\n  2*x\nend",
    )
    model = RoadrunnerSBMLModel(
        source=libsbml.writeSBMLToString(doc), selections=["time", "X"]
    )
    simulator = Simulator(n_workers=1)
    simulation = Simulation(
        end=3, changes=[Change(1, {"q": 1}), Change(2, {"q": 2})], times=[0, 1, 2, 3]
    )
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.executor"):
        simulator.simulate(model, simulation)
        simulator.simulate(model, simulation)
    warnings = [r.getMessage() for r in caplog.records if "trigger" in r.getMessage()]
    assert len(warnings) == 1
    assert "'unknown'" in warnings[0]
    assert "twice(q) > 1" in warnings[0]
