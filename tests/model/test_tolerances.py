"""The absolute tolerances of the integrator, one per state of a model."""

import math

import pytest

from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.model.tolerances import (
    VOLUME_FLOOR,
    AbsoluteTolerance,
    StateKind,
    state_kinds,
    state_tolerances,
)
from tests.simulator.models import TOLERANCE_PROBE, sbml

#: the states of the probe as roadrunner integrates them
STATES = ["A", "S", "X", "D"]
VOLUMES = {"C": 2.0, "U": 1e-12}


@pytest.fixture(scope="module")
def symbols() -> ModelSymbols:
    """Get the symbols of the tolerance probe."""
    return ModelSymbols.from_sbml(sbml(TOLERANCE_PROBE))


def test_kinds(symbols: ModelSymbols) -> None:
    """Amount species, concentration species and other states are told apart."""
    assert state_kinds(STATES, symbols) == {
        "A": StateKind.CONCENTRATION,
        "S": StateKind.CONCENTRATION,
        "X": StateKind.AMOUNT,
        "D": StateKind.OTHER,
    }


def test_tolerance_per_state(symbols: ModelSymbols) -> None:
    """A concentration species gets its tolerance times the reference volume."""
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    by_id = {t.sid: t for t in state_tolerances(STATES, symbols, VOLUMES, tolerance)}
    assert by_id["A"].absolute_tolerance == pytest.approx(1e-8 * 2.0)
    assert by_id["A"].compartment == "C"
    assert by_id["X"].absolute_tolerance == pytest.approx(1e-9)
    assert by_id["X"].volume is None
    assert by_id["D"].absolute_tolerance == pytest.approx(1e-7)


def test_degenerate_volume_is_raised(symbols: ModelSymbols) -> None:
    """A tiny compartment does not collapse the tolerance of its species."""
    by_id = {
        t.sid: t
        for t in state_tolerances(STATES, symbols, VOLUMES, AbsoluteTolerance())
    }
    assert by_id["S"].volume_raised
    assert by_id["S"].volume == pytest.approx(VOLUME_FLOOR * 2.0)
    assert by_id["S"].absolute_tolerance == pytest.approx(1e-10 * VOLUME_FLOOR * 2.0)
    assert not by_id["A"].volume_raised


def test_no_finite_volume(symbols: ModelSymbols) -> None:
    """Without a finite positive volume the reference volume is the floor of 1."""
    volumes = {"C": math.nan, "U": 0.0}
    by_id = {
        t.sid: t
        for t in state_tolerances(STATES, symbols, volumes, AbsoluteTolerance())
    }
    assert by_id["A"].volume == pytest.approx(VOLUME_FLOOR)
    assert by_id["A"].volume_raised


def test_override_by_id(symbols: ModelSymbols) -> None:
    """An override is the tolerance of the state, not multiplied by a volume."""
    tolerance = AbsoluteTolerance(ids={"A": 1e-14})
    by_id = {t.sid: t for t in state_tolerances(STATES, symbols, VOLUMES, tolerance)}
    assert by_id["A"].absolute_tolerance == pytest.approx(1e-14)


def test_override_of_no_state(symbols: ModelSymbols) -> None:
    """An override of an id which is not a state names the states."""
    with pytest.raises(ValueError, match=r"not a state.*'A'"):
        state_tolerances(STATES, symbols, VOLUMES, AbsoluteTolerance(ids={"Y": 1e-9}))


@pytest.mark.parametrize("value", [0.0, -1e-9, math.nan, math.inf])
def test_invalid_tolerance(value: float) -> None:
    """A tolerance is finite and positive."""
    with pytest.raises(ValueError, match="finite and positive"):
        AbsoluteTolerance(amount=value)
    with pytest.raises(ValueError, match="finite and positive"):
        AbsoluteTolerance(ids={"A": value})


def test_of_a_float() -> None:
    """A float is the same tolerance for every kind."""
    tolerance = AbsoluteTolerance.of(1e-6)
    assert tolerance == AbsoluteTolerance(amount=1e-6, concentration=1e-6, other=1e-6)
    assert AbsoluteTolerance.of(tolerance) is tolerance


def test_round_trip() -> None:
    """A tolerance is a dictionary and back, a float of stored settings reads."""
    tolerance = AbsoluteTolerance(amount=1e-9, ids={"b": 1e-12, "a": 1e-13})
    assert AbsoluteTolerance.from_dict(tolerance.to_dict()) == tolerance
    assert AbsoluteTolerance.from_dict(1e-6) == AbsoluteTolerance.of(1e-6)
    assert tolerance.overrides == {"a": 1e-13, "b": 1e-12}
    # the overrides are sorted, so equal tolerances are equal and hash alike
    assert hash(tolerance) == hash(
        AbsoluteTolerance(amount=1e-9, ids={"a": 1e-13, "b": 1e-12})
    )
    assert "2 by id" in str(tolerance)
