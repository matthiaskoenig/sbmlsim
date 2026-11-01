"""The symbols of a model which a simulation can change."""

import math

import libsbml
import pytest

from sbmlsim.model.symbols import ModelSymbols, TargetKind, _selections
from sbmlsim.simulator.formula import compile_formula
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
    number, e.g. `0.5 dimensionless`, which is no math of PEtab.
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
    assert symbols.trigger == "q > 0.5"
    assert symbols.priority == "2"
    assert symbols.assignments == (("X", "1e-3 + q"),)


def _petab(formula: str) -> str | None:
    """Get the formula of selections of math of SBML, with the functions of L3V2."""
    settings = libsbml.L3ParserSettings()
    settings.setParseL3v2Functions(True)
    return _selections(
        libsbml.parseL3FormulaWithSettings(formula, settings), frozenset({"S"})
    )


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("arcsin(b)", math.asin(0.5)),
        ("arccos(b)", math.acos(0.5)),
        ("arctan(a)", math.atan(2.5)),
        ("infinity", math.inf),
        ("-infinity", -math.inf),
        ("a < infinity", True),
        ("geq(a, b, b)", True),
        ("geq(b, a, b)", False),
        ("lt(b, 1, a, 3)", True),
        ("lt(b, a, a)", False),
        ("eq(a, a, a)", True),
        ("eq(a, a, b)", False),
        ("xor(a > b, a < b)", True),
        ("xor(a > b, a > 0)", False),
        ("xor(a > b, a < b, a < 0)", True),
        ("xor(a > b, a < b, a > 0)", False),
        ("implies(a > b, a > 0)", True),
        ("implies(a > b, a < 0)", False),
        ("implies(a < b, a < 0)", True),
    ],
)
def test_the_math_of_sbml_which_the_math_of_petab_expresses(
    formula: str, expected: float
) -> None:
    """A function of SBML is written in the math of PEtab with its value (#299)."""
    compiled = compile_formula(_petab(formula))
    values = {"a": 2.5, "b": 0.5}
    assert compiled.evaluate([values[s] for s in compiled.symbols]) == pytest.approx(
        float(expected)
    )


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("a >= b", "a >= b"),
        ("eq(a, b)", "a == b"),
        ("log(2, a)", "log(a, 2)"),
        ("pi", "3.141592653589793"),
        ("arcsec(a)", "arcsec(a)"),
        ("root(2, a)", "sqrt(a)"),
        ("a && b && a", "a && b && a"),
        ("piecewise(a, a > b, b)", "piecewise(a, a > b, b)"),
        ("S + 1", "[S] + 1"),
        # which the simulator does not evaluate
        ("floor(a)", "floor(a)"),
        ("ceiling(a)", "ceil(a)"),
        ("factorial(a)", "factorial(a)"),
        ("quotient(a, b)", "quotient(a, b)"),
        ("rem(a, b)", "rem(a, b)"),
        ("notanumber", "NaN"),
        ("rateOf(a)", "rateOf(a)"),
        ("delay(a, 1)", "delay(a, 1)"),
        ("root(3, a)", "root(3, a)"),
    ],
)
def test_the_math_which_the_math_of_petab_reads_is_written_as_before(
    formula: str, expected: str
) -> None:
    """The math which #299 does not rewrite is written as before."""
    assert _petab(formula) == expected


def test_the_function_definitions_of_an_event_are_expanded() -> None:
    """A function definition, also one which calls another, is expanded (#299)."""
    (event,) = ModelSymbols.from_sbml(
        sbml(
            "function g(x)\n  2 * x\nend\n"
            "function f(x, y)\n  g(x) + y\nend\n"
            "model c\n  q = 0; X = 0\n"
            "  E: at (f(q, 1) > 3), t0=false, priority=g(2): X = f(q, X)\nend"
        )
    ).events
    for formula, value in [
        (event.trigger, True),
        (event.priority, 4.0),
        (event.assignments[0][1], 5.0),
    ]:
        assert formula is not None
        compiled = compile_formula(formula)
        values = {"q": 2.0, "X": 1.0}
        assert compiled.evaluate([values[s] for s in compiled.symbols]) == value
