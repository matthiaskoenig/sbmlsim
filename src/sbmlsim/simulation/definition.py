"""The definition of a simulation: what is simulated, with units.

A `Simulation` is the interval of a simulation, the changes applied to the
model before it is initialized (`preinit_changes`), the changes at times
(`changes`, a list of `Change`), an optional pre-equilibration (`SteadyState`)
and the output. It is compiled against a model into a
`sbmlsim.simulator.plan.Plan`, which is what the simulator runs. The semantics
are the ones of PEtab v2:

1. The `preinit_changes` are applied to the model before its initial
   assignments are evaluated, so an initial assignment which depends on a
   changed parameter follows it.
2. The model is initialized at `start`.
3. The values of the changes at a time are evaluated with the state at that
   time and then assigned at once. A change at `start` is applied after the
   initialization.

A value is a `Quantity`, a number in the unit of its target in the model, or
a formula: a string of the math of PEtab over the symbols of the model, in the
units of the model. `S` is the amount of a species and `[S]` its
concentration, as in the selections of roadrunner.

A time is a number in the `time_unit` of the simulation, in the time unit of
the model if the simulation has none, or a `Quantity`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, overload

import numpy as np

from sbmlsim.units import Quantity, ureg

#: a time: a number in the time unit of the simulation or a quantity
Time = float | Quantity

#: a value of a change: a number in the unit of the target, a quantity or a
#: formula
Value = float | Quantity | str

#: one or several times
Times = float | Sequence[Time] | Quantity


def _times(times: Times) -> tuple[Time, ...]:
    """Get the times of a change as a tuple of scalars.

    Args:
        times: one time, a sequence of times or a quantity of one or several
            times.

    Returns:
        The times, a quantity of several times is split into scalars.
    """
    if isinstance(times, Quantity):
        magnitudes = np.atleast_1d(np.asarray(times.magnitude, dtype=float))
        return tuple(ureg.Quantity(float(m), times.units) for m in magnitudes)
    if isinstance(times, int | float | np.integer | np.floating):
        return (float(times) if isinstance(times, np.generic) else times,)
    sequence: Sequence[Time] = times
    return tuple(sequence)


@dataclass(frozen=True, init=False)
class Change:
    """Changes of the model at one or several times.

    A change at several times is the same change at each of them, e.g. the
    doses of a multiple dosing:

        Change([0, 24, 48], {"PODOSE_hctz": Q(10, "mg")})

    Attributes:
        times: the times, numbers in the `time_unit` of the simulation or
            quantities.
        values: target -> value, see the module.
    """

    times: tuple[Time, ...]
    values: dict[str, Value]

    def __init__(self, times: Times, values: Mapping[str, Value]) -> None:
        """Create a change, see the class.

        Raises:
            ValueError: if there are no values or no times.
        """
        if not values:
            raise ValueError(f"The change at {times} has no values.")
        resolved = _times(times)
        if not resolved:
            raise ValueError(f"The change of {sorted(values)} has no times.")
        object.__setattr__(self, "times", resolved)
        object.__setattr__(self, "values", dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "times": [_encode(t) for t in self.times],
            "values": {k: _encode(v) for k, v in self.values.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Change:
        """Create a change from its dictionary, see `to_dict`."""
        return cls(
            times=[_decode(t) for t in d["times"]],
            values={k: _decode(v) for k, v in d["values"].items()},
        )


@dataclass(frozen=True)
class SteadyState:
    """A pre-equilibration before the start of a simulation.

    The model is initialized once, with the `preinit_changes` of the
    simulation and of the steady state together, and integrated until the
    rates of change of every state are within the tolerances,
    `|dx/dt| <= absolute_tolerance + relative_tolerance * |x|`. The state at
    steady state is the state the simulation starts from at its `start`.

    Attributes:
        preinit_changes: changes before the initialization, e.g. the
            conditions of a pre-equilibration of PEtab.
        absolute_tolerance: absolute tolerance of the rates of change.
        relative_tolerance: relative tolerance of the rates of change.
        max_time: the longest pre-equilibration in the time unit of the model;
            a simulation which does not reach the steady state by then fails.
    """

    preinit_changes: dict[str, Value] = field(default_factory=dict)
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-6
    max_time: float = 1e8

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "preinit_changes": {k: _encode(v) for k, v in self.preinit_changes.items()},
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "max_time": self.max_time,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SteadyState:
        """Create a steady state from its dictionary, see `to_dict`."""
        return cls(
            preinit_changes={
                k: _decode(v) for k, v in d.get("preinit_changes", {}).items()
            },
            absolute_tolerance=d.get("absolute_tolerance", 1e-8),
            relative_tolerance=d.get("relative_tolerance", 1e-6),
            max_time=d.get("max_time", 1e8),
        )


class Simulation:
    """A simulation, see the module.

    Attributes:
        time_unit: unit of the times given as numbers, the time unit of the
            model if `None`.
        start: start of the simulation, negative times are allowed.
        end: end of the simulation.
        preinit_changes: changes before the initialization of the model.
        changes: changes at times in `[start, end]`.
        presimulation: a pre-equilibration before `start`.
        times: exact output times, `None` for the output of the integrator.
        steps: an equidistant output of `steps + 1` points, excludes `times`.
        time_shift: added to the time of the result. A correct `start` is the
            better way to place a simulation in time.
    """

    def __init__(
        self,
        *,
        end: Time,
        start: Time = 0.0,
        time_unit: str | None = None,
        preinit_changes: Mapping[str, Value] | None = None,
        changes: Sequence[Change] | None = None,
        presimulation: SteadyState | None = None,
        times: Sequence[Time] | Quantity | None = None,
        steps: int | None = None,
        time_shift: Time = 0.0,
    ) -> None:
        """Create a simulation, see the module and the attributes.

        Raises:
            ValueError: if `start >= end`, if both `times` and `steps` are
                given, if `steps < 1`, if a pre-initialization change is a
                formula, or if a target is a pre-initialization change of the
                simulation and of the presimulation.
        """
        self.time_unit: str | None = time_unit
        self.start: Time = start
        self.end: Time = end
        self.preinit_changes: dict[str, Value] = dict(preinit_changes or {})
        self.changes: list[Change] = list(changes or [])
        self.presimulation: SteadyState | None = presimulation
        self.times: tuple[Time, ...] | None = None if times is None else _times(times)
        self.steps: int | None = steps
        self.time_shift: Time = time_shift
        self._validate()

    def _validate(self) -> None:
        """Check the definition, see `__init__`."""
        if self._magnitude(self.start) >= self._magnitude(self.end):
            raise ValueError(
                f"The start '{self.start}' of the simulation is not before its "
                f"end '{self.end}'."
            )
        if self.times is not None and self.steps is not None:
            raise ValueError(
                "A simulation has exact output 'times' or an equidistant grid "
                "of 'steps', not both."
            )
        if self.steps is not None and self.steps < 1:
            raise ValueError(f"The 'steps' must be at least 1, not {self.steps}.")
        if self.presimulation is not None:
            shared = sorted(
                set(self.preinit_changes) & set(self.presimulation.preinit_changes)
            )
            if shared:
                raise ValueError(
                    f"The targets {', '.join(repr(t) for t in shared)} are "
                    f"pre-initialization changes of the simulation and of its "
                    f"presimulation. The model is initialized once; a value "
                    f"which differs after the steady state is a `Change` at "
                    f"the start of the simulation."
                )

    def _magnitude(self, time: Time) -> float:
        """Get a time as a number in the time unit of the simulation.

        A quantity of a simulation without a time unit is taken in the unit of
        the end of the simulation, or as it is, so that the interval can be
        checked without the model.
        """
        if not isinstance(time, Quantity):
            return float(time)
        if self.time_unit is not None:
            return float(time.to(self.time_unit).magnitude)
        if isinstance(self.end, Quantity):
            return float(time.to(self.end.units).magnitude)
        return float(time.magnitude)

    def targets(self) -> set[str]:
        """Get every target the simulation sets, before and after initialization."""
        targets = set(self.preinit_changes)
        for change in self.changes:
            targets.update(change.values)
        return targets

    def with_values(self, values: Mapping[str, float | Quantity]) -> Simulation:
        """Get a copy of the simulation with other values of targets.

        A value replaces the value of its target wherever the simulation sets
        it, i.e. in the pre-initialization changes, in those of the
        presimulation and in every change, and is added to the
        pre-initialization changes of a target which the simulation does not
        set. This defines the semantics which `Plan.with_values` applies to a
        compiled simulation, for the points of a scan and the parameters of a
        fit.

        Args:
            values: target -> value.

        Returns:
            The new simulation, this one is not changed.
        """
        preinit = dict(self.preinit_changes)
        changes: list[Change] = []
        set_elsewhere: set[str] = set()
        for change in self.changes:
            new_values = dict(change.values)
            for target in change.values:
                if target in values:
                    new_values[target] = values[target]
                    set_elsewhere.add(target)
            changes.append(Change(change.times, new_values))
        presimulation = self.presimulation
        if presimulation is not None:
            steady = dict(presimulation.preinit_changes)
            for target in steady:
                if target in values:
                    steady[target] = values[target]
                    set_elsewhere.add(target)
            presimulation = replace(presimulation, preinit_changes=steady)
        for target, value in values.items():
            if target in preinit or target not in set_elsewhere:
                preinit[target] = value
        return self._copy(preinit, changes, presimulation)

    def with_preinit_defaults(
        self, values: Mapping[str, float | Quantity]
    ) -> Simulation:
        """Get a copy of the simulation with default pre-initialization changes.

        A default applies unless the simulation sets its target before the
        initialization itself, in its own `preinit_changes` or in those of
        its presimulation; a `Change` of the target at a time does not replace
        it. This is how the changes of a model reach its simulations.

        Args:
            values: target -> value.

        Returns:
            The new simulation, this one is not changed.
        """
        own = set(self.preinit_changes)
        if self.presimulation is not None:
            own |= set(self.presimulation.preinit_changes)
        preinit: dict[str, Value] = {k: v for k, v in values.items() if k not in own}
        preinit.update(self.preinit_changes)
        return self._copy(preinit, list(self.changes), self.presimulation)

    def _copy(
        self,
        preinit: Mapping[str, Value],
        changes: Sequence[Change],
        presimulation: SteadyState | None,
    ) -> Simulation:
        """Get a copy with other changes, the output and the interval kept."""
        return Simulation(
            time_unit=self.time_unit,
            start=self.start,
            end=self.end,
            preinit_changes=preinit,
            changes=changes,
            presimulation=presimulation,
            times=None if self.times is None else list(self.times),
            steps=self.steps,
            time_shift=self.time_shift,
        )

    def __repr__(self) -> str:
        """Get the representation."""
        if self.times is not None:
            output = f"times={len(self.times)}"
        elif self.steps is not None:
            output = f"steps={self.steps}"
        else:
            output = "integrator"
        unit = f" {self.time_unit}" if self.time_unit else ""
        return (
            f"Simulation([{self.start}, {self.end}]{unit}, "
            f"preinit={len(self.preinit_changes)}, changes={len(self.changes)}, "
            f"{output})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "type": self.__class__.__name__,
            "time_unit": self.time_unit,
            "start": _encode(self.start),
            "end": _encode(self.end),
            "preinit_changes": {k: _encode(v) for k, v in self.preinit_changes.items()},
            "changes": [change.to_dict() for change in self.changes],
            "presimulation": None
            if self.presimulation is None
            else self.presimulation.to_dict(),
            "times": None if self.times is None else [_encode(t) for t in self.times],
            "steps": self.steps,
            "time_shift": _encode(self.time_shift),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Simulation:
        """Create a simulation from its dictionary, see `to_dict`."""
        presimulation = d.get("presimulation")
        times = d.get("times")
        return cls(
            time_unit=d.get("time_unit"),
            start=_decode(d.get("start", 0.0)),
            end=_decode(d["end"]),
            preinit_changes={
                k: _decode(v) for k, v in d.get("preinit_changes", {}).items()
            },
            changes=[Change.from_dict(c) for c in d.get("changes", [])],
            presimulation=None
            if presimulation is None
            else SteadyState.from_dict(presimulation),
            times=None if times is None else [_decode(t) for t in times],
            steps=d.get("steps"),
            time_shift=_decode(d.get("time_shift", 0.0)),
        )

    @overload
    def to_json(self, path: None = None) -> str: ...

    @overload
    def to_json(self, path: Path) -> None: ...

    def to_json(self, path: Path | None = None) -> str | None:
        """Convert the definition to JSON.

        Args:
            path: file to write the JSON to, the JSON is returned without one.

        Returns:
            The JSON without a path, `None` with a path.
        """
        if path is None:
            return json.dumps(self.to_dict(), indent=2)
        with open(path, "w", encoding="utf-8") as f_json:
            json.dump(self.to_dict(), fp=f_json, indent=2)
        return None

    @staticmethod
    def from_json(json_info: str | Path) -> Simulation:
        """Read a simulation from JSON, a path or the JSON itself."""
        if isinstance(json_info, Path):
            with open(json_info, encoding="utf-8") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        return Simulation.from_dict(d)


def _encode(value: Any) -> Any:
    """Encode a number, a quantity or a formula as a JSON type."""
    if isinstance(value, Quantity):
        magnitude = value.magnitude
        if isinstance(magnitude, np.ndarray):
            magnitude = magnitude.tolist()
        return {"value": magnitude, "unit": str(value.units)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any) -> Any:
    """Decode a value of `_encode`."""
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        return ureg.Quantity(value["value"], value["unit"])
    return value
