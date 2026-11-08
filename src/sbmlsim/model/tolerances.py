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

The kinds give a small state the same tolerance as a large one, so a state
whose magnitude is far below the tolerance over the relative tolerance, e.g.
a drug in nM next to glucose in mM, is integrated with the relative accuracy
`atol / |x|` instead of `rtol` (#435). A tolerance `relative` scales the
tolerance of every state by its typical magnitude instead,
`relative * scale`, which gives every state the relative accuracy
`relative`, best the relative tolerance of the integrator. The scale of a
state is its value in `scales`, e.g. its largest magnitude in test
simulations (`sbmlsim.simulator.scales.typical_scales`), else the magnitude
of its initial value in the model as loaded (the amount of a species); a
state without either, e.g. a drug which starts at 0, keeps the tolerance of
its kind. Rescaling the units of a model changes the integration only through
these tolerances, they are the same as rescaled states: CVODE weighs every
state by its tolerance.
`V` is the reference volume of the compartment: its initial volume in the
model as loaded, raised to `VOLUME_FLOOR` times the largest finite positive
initial volume of the model when it is smaller, not finite or not positive,
so that a degenerate compartment does not collapse the tolerance of its
species. A volume which the pre-initialization of a simulation changes does
not change the tolerances, so a simulation does not depend on the one before
it on the same model.
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
        relative: the tolerance of a state relative to its scale, see the
            module; `None` for the tolerances of the kinds alone.
        scales: the typical magnitudes of states by id, in the unit of the
            model of the state, an amount for a species; a state without one
            is scaled by its initial value. Normalized as `ids`.
    """

    amount: float = 1e-10
    concentration: float = 1e-10
    other: float = 1e-10
    ids: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    relative: float | None = None
    scales: Mapping[str, float] | tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Check the tolerances and normalize the overrides and the scales.

        Raises:
            ValueError: for a tolerance or a scale which is not finite and
                positive, or scales without a relative tolerance.
        """
        for kind in StateKind:
            object.__setattr__(
                self, kind.value, _checked(kind.value, getattr(self, kind.value))
            )
        for field in ("ids", "scales"):
            values = tuple(
                sorted(
                    (str(sid), _checked(str(sid), v))
                    for sid, v in dict(getattr(self, field)).items()
                )
            )
            object.__setattr__(self, field, values)
        if self.relative is not None:
            object.__setattr__(self, "relative", _checked("relative", self.relative))
        elif self.scales:
            raise ValueError(
                "The scales of the states need the tolerance relative to them, "
                "`relative`."
            )

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

    @property
    def state_scales(self) -> dict[str, float]:
        """Get the scales of single states by id."""
        return dict(self.scales)

    def of_kind(self, kind: StateKind) -> float:
        """Get the tolerance of a kind of state."""
        return float(getattr(self, kind.value))

    def distinct_values(self) -> set[float]:
        """Get the distinct tolerances of the kinds and of the single states."""
        return {self.amount, self.concentration, self.other} | set(
            self.overrides.values()
        )

    def smallest(self) -> float:
        """Get the smallest tolerance of the kinds and of the single states.

        The one absolute tolerance of an engine which has no tolerance per
        state, COPASI and AMICI, is derived from it; the scaled tolerances of
        `relative` are tolerances per state, which these engines do not
        have.
        """
        return min(self.distinct_values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "amount": self.amount,
            "concentration": self.concentration,
            "other": self.other,
            "ids": self.overrides,
            "relative": self.relative,
            "scales": self.state_scales,
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
                relative=d.get("relative"),
                scales=d.get("scales", {}),
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
        if self.relative is not None:
            text += f", {self.relative:.1e} of the scale of a state"
            if self.scales:
                text += f" ({len(self.scales)} scales)"
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
        scale: the scale of the state which the tolerance is relative to,
            `None` for the tolerance of its kind or of an override.
    """

    sid: str
    kind: StateKind
    compartment: str | None
    volume: float | None
    volume_raised: bool
    absolute_tolerance: float
    scale: float | None = None


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


def _scale(value: float | None) -> float | None:
    """Get the scale of an initial value, `None` for 0 or a value which is not finite."""
    if value is None or not math.isfinite(value) or value == 0:
        return None
    return abs(value)


def state_tolerances(
    states: Sequence[str],
    symbols: ModelSymbols,
    initial_volumes: Mapping[str, float],
    tolerance: AbsoluteTolerance,
    initial_values: Mapping[str, float] | None = None,
) -> list[StateTolerance]:
    """Get the absolute tolerance of every state, see the module.

    Args:
        states: ids of the states which the integrator integrates.
        symbols: symbols of the model.
        initial_volumes: initial volume of every compartment.
        tolerance: the tolerances per kind and by id.
        initial_values: the initial value of every state in the model as
            loaded, an amount for a species, which scales its tolerance with
            `relative` when it has no scale of its own.

    Returns:
        The tolerance of every state, in the order of `states`.

    Raises:
        ValueError: if an override or a scale is not a state.
    """
    overrides = tolerance.overrides
    scales = tolerance.state_scales
    for name, given in (("absolute tolerances", overrides), ("scales", scales)):
        unknown = sorted(set(given) - set(states))
        if unknown:
            raise ValueError(
                f"The {name} of {unknown} are given for ids which are not a "
                f"state, the states are {sorted(states)}."
            )
    initial_values = initial_values or {}
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
        scale: float | None = None
        if tolerance.relative is not None and sid not in overrides:
            scale = scales.get(sid, _scale(initial_values.get(sid)))
            if scale is not None:
                value = tolerance.relative * scale
        result.append(
            StateTolerance(
                sid=sid,
                kind=kind,
                compartment=compartment,
                volume=volume,
                volume_raised=compartment in raised,
                absolute_tolerance=overrides.get(sid, value),
                scale=scale,
            )
        )
    return result
