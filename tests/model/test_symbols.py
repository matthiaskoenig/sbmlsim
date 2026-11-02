"""The symbols of a model which a simulation can change."""

import libsbml
import pytest

from sbmlsim.model.symbols import ModelSymbols, TargetKind
from sbmlsim.sbml_math import compile_math, math_formula
from tests.simulator.models import sbml


@pytest.fixture(scope="module")
def symbols() -> ModelSymbols:
    """Get the symbols of the probe model."""
    return ModelSymbols.from_sbml(sbml())


def test_kinds_of_the_probe(symbols: ModelSymbols) -> None:
    """Every target is a parameter, a compartment, an amount or a concentration."""
    assert symbols.kind("k1") is TargetKind.PARAMETER
    assert symbols.kind("C") is TargetKind.COMPARTMENT
    assert symbols.kind("A") is TargetKind.SPECIES_AMOUNT
    assert symbols.kind("[A]") is TargetKind.SPECIES_CONCENTRATION
    assert "X" in symbols.only_substance
    assert "pinit" in symbols.initial_assignments
    assert symbols.species_compartment["A"] == "C"


def test_an_assignment_rule_is_no_target(symbols: ModelSymbols) -> None:
    """The target of an assignment rule cannot be changed."""
    with pytest.raises(ValueError, match=r"'kk'.*assignment rule"):
        symbols.kind("kk")


def test_an_unknown_id_is_no_target(symbols: ModelSymbols) -> None:
    """An id the model does not have is an error which names it."""
    with pytest.raises(ValueError, match="'nope'"):
        symbols.kind("nope")
    with pytest.raises(ValueError, match=r"'k1'.*species"):
        symbols.kind("[k1]")


def test_entity(symbols: ModelSymbols) -> None:
    """The entity of a concentration is its species."""
    assert symbols.entity("[A]") == "A"
    assert symbols.entity("k1") == "k1"


def test_symbols_from_a_file(tmp_path) -> None:
    """A model is read from a file as well."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    assert ModelSymbols.from_sbml(path).kind("k2") is TargetKind.PARAMETER


def test_dependencies_with_sciml_imported() -> None:
    """The dependencies are read with the SWIG modules of the extras loaded."""
    import sbmlsim.sciml  # noqa: F401

    symbols = ModelSymbols.from_sbml(sbml())
    assert symbols.initial_assignment_dependencies is not None
    assert symbols.initial_assignment_dependencies["B"] == frozenset({"b0"})


def test_the_math_of_an_event_has_no_units() -> None:
    """A number with a unit is the number in a formula of the simulator (#293).

    sbmlutils writes a unit on every number, libsbml writes it after the
    number, e.g. `0.5 dimensionless`; the MathML of the events has none.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(
        sbml(
            "model units\n  q = 0; X = 0\n"
            "  E: at (q > 1), t0=false, priority=1: X = 1\nend"
        )
    )
    event: libsbml.Event = doc.getModel().getEvent("E")
    event.getTrigger().setMath(libsbml.parseL3Formula("q > 0.5 dimensionless"))
    event.getPriority().setMath(libsbml.parseL3Formula("2 dimensionless"))
    event.getEventAssignment("X").setMath(libsbml.parseL3Formula("1e-3 mole + q"))
    assert "dimensionless" in libsbml.writeSBMLToString(doc)
    (symbols,) = ModelSymbols.from_sbml(libsbml.writeSBMLToString(doc)).events
    assert symbols.trigger is not None
    assert symbols.priority is not None
    assert "units" not in symbols.trigger + symbols.priority
    assert math_formula(symbols.trigger) == "q > 0.5"
    assert math_formula(symbols.priority) == "2"
    ((target, assignment),) = symbols.assignments
    assert (target, math_formula(assignment)) == ("X", "1e-3 + q")
    assert compile_math(assignment).evaluate([2.0]) == 1e-3 + 2.0


def test_a_concentration_depends_on_its_compartment_of_a_rule() -> None:
    """The initial value of a species follows its compartment of a rule (#344)."""
    symbols = ModelSymbols.from_sbml(
        sbml(
            "model c\n"
            "  BW = 75; f = 0.1; c0 = 50\n"
            "  compartment V; compartment W = BW * f\n"
            "  V := BW * f\n"
            "  species S in V; species T in V = 2\n"
            "  species A in W = 3; substanceOnly species X in V = 4\n"
            "  S = c0\n"
            "  R = 2 * S\n"
            "end"
        )
    )
    dependencies = symbols.initial_assignment_dependencies
    assert dependencies is not None
    volume = frozenset({"V", "BW", "f"})
    assert dependencies["S"] == volume | {"c0"}
    assert dependencies["T"] == volume
    # a compartment of an initial assignment keeps the concentrations itself,
    # a species with only substance units keeps its amount
    assert "A" not in dependencies
    assert "X" not in dependencies
    assert symbols.initial_concentration_values == {"T": 2.0}
    # an initial assignment which reads the concentration after it
    assert symbols.initial_assignment_order == ("S", "T", "W", "R")
