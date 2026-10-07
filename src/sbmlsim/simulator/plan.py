"""A simulation compiled against a model: the plan the executor runs.

`compile_simulation` does everything which does not depend on the values a
scan or a fit sets: it converts every time into the time unit of the model and
every quantity into the unit of its target, resolves the kind of every target,
merges the changes into one event per time, checks every formula and builds
the output times. A `Plan` holds numbers, strings and the symbols of the
model, no quantity and no libsbml object, so it is pickled for the workers of
a fit and an evaluation never touches pint.

A formula stays a string in the plan; the executor compiles it once per
process, see `sbmlsim.simulator.formula`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np
from pint.errors import DimensionalityError, UndefinedUnitError

from sbmlsim.model.symbols import ModelSymbols, TargetKind
from sbmlsim.simulation.definition import Simulation, SteadyState, Time
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.units import Quantity, UnitsInformation


@dataclass(frozen=True)
class Assignment:
    """The value of one target, a number in the unit of the model or a formula.

    Attributes:
        target: the target, `S`, `[S]` or the id of a parameter or compartment.
        kind: the kind of the target.
        value: the value, `None` for a formula.
        formula: the formula, `None` for a value.
    """

    target: str
    kind: TargetKind
    value: float | None = None
    formula: str | None = None


@dataclass(frozen=True)
class PlanEvent:
    """The assignments at one time.

    Attributes:
        time: the time in the time unit of the model.
        assignments: the assignments, evaluated together and then applied.
    """

    time: float
    assignments: tuple[Assignment, ...]


class OutputMode(StrEnum):
    """The output of a simulation."""

    #: the steps of the integrator
    INTEGRATOR = "integrator"
    #: exact output times
    TIMES = "times"


@dataclass(frozen=True)
class SteadyStatePlan:
    """A pre-equilibration of a plan, see `sbmlsim.simulation.SteadyState`.

    Attributes:
        preinit: the pre-initialization assignments of the steady state.
        absolute_tolerance: absolute tolerance of the rates of change.
        relative_tolerance: relative tolerance of the rates of change.
        max_time: the longest pre-equilibration, in the time unit of the
            model.
    """

    preinit: tuple[Assignment, ...]
    absolute_tolerance: float
    relative_tolerance: float
    max_time: float


@dataclass(frozen=True)
class Plan:
    """A simulation compiled against a model, see the module.

    Attributes:
        start: start of the simulation, in the time unit of the model.
        end: end of the simulation.
        preinit: the assignments before the initialization, values only.
        events: the assignments at times in `[start, end]`, sorted by time.
        steady_state: the pre-equilibration, `None` without one.
        output: the output of the simulation.
        times: the output times of `OutputMode.TIMES`, empty otherwise.
        time_shift: added to the time of the result.
        symbols: the symbols of the model, which resolve a target `with_values`
            adds.
    """

    start: float
    end: float
    preinit: tuple[Assignment, ...]
    events: tuple[PlanEvent, ...]
    steady_state: SteadyStatePlan | None
    output: OutputMode
    times: tuple[float, ...]
    time_shift: float
    symbols: ModelSymbols = field(repr=False, compare=False)

    def with_values(self, values: Mapping[str, float]) -> Plan:
        """Get the plan with other values of targets.

        A value replaces the assignment of its target wherever the plan has
        one, i.e. before the initialization, in the steady state and at every
        time, and is added
        to the assignments before the initialization of a target which the
        plan does not set. This is the rule of `Simulation.with_values` on
        numbers in the units of the model, which is what a fit sets.

        Args:
            values: target -> value in the unit of the target in the model.

        Returns:
            The new plan, this one is not changed.

        Raises:
            ValueError: if a target is not a target of the model.
        """
        if not values:
            return self

        def replaced(assignments: tuple[Assignment, ...]) -> tuple[Assignment, ...]:
            return tuple(
                Assignment(a.target, a.kind, value=float(values[a.target]))
                if a.target in values
                else a
                for a in assignments
            )

        events = tuple(
            PlanEvent(event.time, replaced(event.assignments)) for event in self.events
        )
        steady_state = self.steady_state
        if steady_state is not None:
            steady_state = replace(steady_state, preinit=replaced(steady_state.preinit))
        set_targets = {a.target for e in self.events for a in e.assignments}
        set_targets |= {a.target for a in self.preinit}
        if self.steady_state is not None:
            set_targets |= {a.target for a in self.steady_state.preinit}
        preinit = list(replaced(self.preinit))
        for target, value in values.items():
            if target in set_targets:
                continue
            preinit.append(
                Assignment(target, self.symbols.kind(target), value=float(value))
            )
        return replace(
            self, preinit=tuple(preinit), events=events, steady_state=steady_state
        )


class _Converter:
    """Convert the times and values of a simulation into the units of a model."""

    def __init__(
        self, simulation: Simulation, symbols: ModelSymbols, uinfo: UnitsInformation
    ) -> None:
        """Bind the converter to a simulation and a model.

        Raises:
            ValueError: if the time unit of the simulation is not a time of
                the model.
        """
        self.symbols = symbols
        self.uinfo = uinfo
        self.model_time_unit: str = uinfo.get("time", "") or "dimensionless"
        self.factor: float = 1.0
        if simulation.time_unit is not None:
            try:
                self.factor = float(
                    uinfo.ureg.Quantity(1.0, simulation.time_unit)
                    .to(self.model_time_unit)
                    .magnitude
                )
            except (DimensionalityError, UndefinedUnitError) as err:
                raise ValueError(
                    f"The time unit '{simulation.time_unit}' of the simulation "
                    f"cannot be converted into the time unit "
                    f"'{self.model_time_unit}' of the model: {err}"
                ) from err

    def time(self, time: Time) -> float:
        """Get a time in the time unit of the model.

        Raises:
            ValueError: if a quantity is not a time of the model.
        """
        if isinstance(time, Quantity):
            try:
                return float(time.to(self.model_time_unit).magnitude)
            except (DimensionalityError, UndefinedUnitError) as err:
                raise ValueError(
                    f"The time '{time}' cannot be converted into the time unit "
                    f"'{self.model_time_unit}' of the model: {err}"
                ) from err
        return float(time) * self.factor

    def assignment(self, target: str, value: Any) -> Assignment:
        """Get the assignment of a value to a target.

        Raises:
            ValueError: if the target is not a target of the model, if a
                quantity cannot be converted into the unit of the target, or
                if a formula is not valid or reads a symbol the model does not
                have.
        """
        kind = self.symbols.kind(target)
        if isinstance(value, str):
            self._check_formula(target, value)
            return Assignment(target, kind, formula=value)
        if isinstance(value, Quantity):
            unit = self.uinfo.get(target)
            if unit is None:
                raise ValueError(
                    f"'{target}' has no unit in the model, its value '{value}' "
                    f"cannot be converted: give a number in the unit the model "
                    f"means."
                )
            try:
                magnitude = value.to(unit or "dimensionless").magnitude
            except (DimensionalityError, UndefinedUnitError) as err:
                raise ValueError(
                    f"The value '{value}' of '{target}' cannot be converted into "
                    f"the unit '{unit}' of the model: {err}"
                ) from err
            return Assignment(target, kind, value=float(magnitude))
        return Assignment(target, kind, value=float(value))

    def _check_formula(self, target: str, formula: str) -> None:
        """Check that every symbol of a formula is the time or an entity."""
        compiled = compile_formula(formula)
        entities = (
            self.symbols.parameters | self.symbols.compartments | self.symbols.species
        )
        for symbol in compiled.symbols:
            if symbol == "time" or self.symbols.entity(symbol) in entities:
                continue
            raise ValueError(
                f"The formula '{formula}' of '{target}' reads '{symbol}', which "
                f"is neither the time nor an entity of the model."
            )

    def preinit(self, changes: Mapping[str, Any]) -> tuple[Assignment, ...]:
        """Get the assignments of pre-initialization changes."""
        return tuple(self.assignment(t, v) for t, v in changes.items())


def compile_simulation(
    simulation: Simulation, symbols: ModelSymbols, uinfo: UnitsInformation
) -> Plan:
    """Compile a simulation against a model.

    Args:
        simulation: the simulation.
        symbols: the symbols of the model.
        uinfo: the units of the model.

    Returns:
        The plan of the simulation, see the module.

    Raises:
        ValueError: if a time or a value cannot be converted, if a target is
            not a target of the model, if a change or an output time is
            outside of the interval of the simulation, or if two values of
            one target are set at one time.
    """
    convert = _Converter(simulation, symbols, uinfo)
    start = convert.time(simulation.start)
    end = convert.time(simulation.end)
    if start >= end:
        raise ValueError(
            f"The start '{simulation.start}' of the simulation is not before its "
            f"end '{simulation.end}'."
        )

    by_time: dict[float, dict[str, Assignment]] = {}
    for change in simulation.changes:
        for time in change.times:
            t = convert.time(time)
            if not start <= t <= end:
                raise ValueError(
                    f"The change of {sorted(change.values)} at '{time}' is "
                    f"outside of the simulation [{simulation.start}, "
                    f"{simulation.end}]."
                )
            assignments = by_time.setdefault(t, {})
            for target, value in change.values.items():
                if target in assignments:
                    raise ValueError(
                        f"Two values of '{target}' at the time '{time}' ({t} in "
                        f"the time unit of the model): a target is set once at a "
                        f"time."
                    )
                assignments[target] = convert.assignment(target, value)
    events = tuple(PlanEvent(t, tuple(by_time[t].values())) for t in sorted(by_time))

    output = OutputMode.INTEGRATOR
    times: tuple[float, ...] = ()
    if simulation.steps is not None:
        output = OutputMode.TIMES
        times = tuple(float(t) for t in np.linspace(start, end, simulation.steps + 1))
    elif simulation.times is not None:
        output = OutputMode.TIMES
        times = tuple(sorted({convert.time(t) for t in simulation.times}))
        outside = [t for t in times if not start <= t <= end]
        if outside:
            raise ValueError(
                f"The output times {outside} are outside of the simulation "
                f"[{start}, {end}] in the time unit of the model."
            )

    return Plan(
        start=start,
        end=end,
        preinit=convert.preinit(simulation.preinit_changes),
        events=events,
        steady_state=_steady_state(simulation.presimulation, convert),
        output=output,
        times=times,
        time_shift=convert.time(simulation.time_shift),
        symbols=symbols,
    )


def _steady_state(
    steady_state: SteadyState | None, convert: _Converter
) -> SteadyStatePlan | None:
    """Compile the presimulation of a simulation."""
    if steady_state is None:
        return None
    return SteadyStatePlan(
        preinit=convert.preinit(steady_state.preinit_changes),
        absolute_tolerance=steady_state.absolute_tolerance,
        relative_tolerance=steady_state.relative_tolerance,
        max_time=steady_state.max_time,
    )


def preinit_targets(plan: Plan) -> Sequence[Assignment]:
    """Get the pre-initialization assignments of a plan and of its steady state."""
    if plan.steady_state is None:
        return plan.preinit
    return plan.preinit + plan.steady_state.preinit
