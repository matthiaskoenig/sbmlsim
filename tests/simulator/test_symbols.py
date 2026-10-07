"""The symbols of a model which a simulation can change."""

import pytest

from sbmlsim.simulator.symbols import ModelSymbols, TargetKind
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
