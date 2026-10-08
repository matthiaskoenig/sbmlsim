"""The absolute tolerances of the integrator, one per state of a model.

CVODE weighs the error of a state `x` with `1 / (rtol * |x| + atol)`. sbmlsim
sets `atol` of every state by its kind instead of handing roadrunner one value,
which roadrunner scales by the initial value or by the volume of a state:

| kind | states | absolute tolerance |
| --- | --- | --- |
| `AMOUNT` | species with `hasOnlySubstanceUnits=true` | `amount` |
| `CONCENTRATION` | other species, integrated as amounts | `concentration * V` |
| `OTHER` | every other state, i.e. the targets of rate rules | `other` |

The numbers are in the units of the model of the state, so they need no units.
`V` is the reference volume of the compartment: its initial volume, raised to
`VOLUME_FLOOR` times the largest finite positive initial volume of the model
when it is smaller, not finite or not positive, so that a degenerate
compartment does not collapse the tolerance of its species.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sbmlsim.model.symbols import ModelSymbols

#: the smallest reference volume relative to the largest initial volume
VOLUME_FLOOR = 1e-6


class StateKind(StrEnum):
    """The kind of a state, which decides its absolute tolerance."""

    AMOUNT = "amount"
    CONCENTRATION = "concentration"
    OTHER = "other"


def _checked(name: str, value: float) -> float:
    """Get a tolerance as a float.

    Raises:
        ValueError: if the tolerance is not finite or not positive.
    """
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"The absolute tolerance of '{name}' must be finite and positive, "
            f"but is {value}."
        )
    return value


@dataclass(frozen=True)
class AbsoluteTolerance:
    """The absolute tolerance per kind of state, with overrides per id.

    Attributes:
        amount: tolerance of a species with `hasOnlySubstanceUnits=true`.
        concentration: tolerance of the concentration of a species, which is
            multiplied by the reference volume of its compartment.
        other: tolerance of every other state, i.e. the target of a rate rule.
        ids: tolerances of single states by id, in the unit of the model of the
            state; an override of a concentration species is an amount. Any
            mapping is normalized to a sorted tuple, so that tolerances are
            hashable and compare by value.
    """

    amount: float = 1e-10
    concentration: float = 1e-10
    other: float = 1e-10
    ids: Mapping[str, float] | tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Check the tolerances and normalize the overrides."""
        for kind in StateKind:
            object.__setattr__(
                self, kind.value, _checked(kind.value, getattr(self, kind.value))
            )
        overrides = tuple(
            sorted(
                (str(sid), _checked(str(sid), v)) for sid, v in dict(self.ids).items()
            )
        )
        object.__setattr__(self, "ids", overrides)

    @classmethod
    def of(cls, value: float | AbsoluteTolerance) -> AbsoluteTolerance:
        """Get the tolerance of a setting, a float is the same for every kind."""
        if isinstance(value, AbsoluteTolerance):
            return value
        value = float(value)
        return cls(amount=value, concentration=value, other=value)

    @property
    def overrides(self) -> dict[str, float]:
        """Get the tolerances of single states by id."""
        return dict(self.ids)

    def of_kind(self, kind: StateKind) -> float:
        """Get the tolerance of a kind of state."""
        return float(getattr(self, kind.value))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "amount": self.amount,
            "concentration": self.concentration,
            "other": self.other,
            "ids": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: float | Mapping[str, Any]) -> AbsoluteTolerance:
        """Create a tolerance from `to_dict`, or from the float of stored settings."""
        if isinstance(d, Mapping):
            return cls(
                amount=d["amount"],
                concentration=d["concentration"],
                other=d["other"],
                ids=d.get("ids", {}),
            )
        return cls.of(d)

    def __str__(self) -> str:
        """Get the tolerances per kind and the number of overrides."""
        text = (
            f"amount {self.amount:.1e}, concentration {self.concentration:.1e}, "
            f"other {self.other:.1e}"
        )
        if self.ids:
            text += f", {len(self.ids)} by id"
        return text


@dataclass(frozen=True)
class StateTolerance:
    """The absolute tolerance of a state.

    Attributes:
        sid: id of the state.
        kind: kind of the state.
        compartment: compartment of a concentration species, else `None`.
        volume: reference volume of the compartment of a concentration
            species, else `None`.
        volume_raised: whether the initial volume was raised to the floor.
        absolute_tolerance: the tolerance in the unit of the model of the
            state, an amount for a species.
    """

    sid: str
    kind: StateKind
    compartment: str | None
    volume: float | None
    volume_raised: bool
    absolute_tolerance: float


def state_kinds(states: Iterable[str], symbols: ModelSymbols) -> dict[str, StateKind]:
    """Get the kind of every state.

    Args:
        states: ids of the states which the integrator integrates.
        symbols: symbols of the model.

    Returns:
        The kind by id.
    """
    kinds: dict[str, StateKind] = {}
    for sid in states:
        if sid not in symbols.species:
            kinds[sid] = StateKind.OTHER
        elif sid in symbols.only_substance:
            kinds[sid] = StateKind.AMOUNT
        else:
            kinds[sid] = StateKind.CONCENTRATION
    return kinds


def _reference_volumes(
    initial_volumes: Mapping[str, float],
) -> tuple[dict[str, float], set[str]]:
    """Get the reference volume of every compartment and the raised ones."""
    finite = [v for v in initial_volumes.values() if math.isfinite(v) and v > 0]
    floor = VOLUME_FLOOR * max(finite, default=1.0)
    volumes: dict[str, float] = {}
    raised: set[str] = set()
    for cid, volume in initial_volumes.items():
        if math.isfinite(volume) and volume >= floor:
            volumes[cid] = volume
        else:
            volumes[cid] = floor
            raised.add(cid)
    return volumes, raised


def state_tolerances(
    states: Sequence[str],
    symbols: ModelSymbols,
    initial_volumes: Mapping[str, float],
    tolerance: AbsoluteTolerance,
) -> list[StateTolerance]:
    """Get the absolute tolerance of every state, see the module.

    Args:
        states: ids of the states which the integrator integrates.
        symbols: symbols of the model.
        initial_volumes: initial volume of every compartment.
        tolerance: the tolerances per kind and by id.

    Returns:
        The tolerance of every state, in the order of `states`.

    Raises:
        ValueError: if an override is not a state.
    """
    overrides = tolerance.overrides
    unknown = sorted(set(overrides) - set(states))
    if unknown:
        raise ValueError(
            f"The absolute tolerances of {unknown} are overrides of ids which are "
            f"not a state, the states are {sorted(states)}."
        )
    volumes, raised = _reference_volumes(initial_volumes)
    result: list[StateTolerance] = []
    for sid, kind in state_kinds(states, symbols).items():
        compartment: str | None = None
        volume: float | None = None
        value = tolerance.of_kind(kind)
        if kind is StateKind.CONCENTRATION:
            compartment = symbols.species_compartment[sid]
            volume = volumes.get(compartment, VOLUME_FLOOR)
            value *= volume
        result.append(
            StateTolerance(
                sid=sid,
                kind=kind,
                compartment=compartment,
                volume=volume,
                volume_raised=compartment in raised,
                absolute_tolerance=overrides.get(sid, value),
            )
        )
    return result
