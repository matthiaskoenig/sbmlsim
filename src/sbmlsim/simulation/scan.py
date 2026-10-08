"""A scan: a simulation and the dimensions of its changes.

A `Scan` runs its `Simulation` for every point of its dimensions; several
dimensions span their cartesian product, whose points are enumerated in C
order of the dimensions, the last dimension fastest. A `Dimension` varies one
of three things:

- `values`: targets to arrays of values. The arrays of a dimension have one
  length and are coupled, i.e. point `k` sets the `k`-th value of every
  target. A value replaces its target wherever the simulation sets it and is
  a pre-initialization change otherwise, see `Simulation.with_values`; with
  `at` the values are a `Change` at that time instead. A sampled design, e.g.
  a Latin hypercube or a virtual population, is one dimension with coupled
  values. A value is a number in the unit of its target in the model or a
  quantity; a formula per point is a dimension of simulations.
- `simulations`: labels to simulations. Every point simulates its own
  simulation, which replaces the simulation of the scan.
- `models`: labels to models (an `AbstractModel`, a `RoadrunnerSBMLModel` or
  a path). Every point simulates its own model, which replaces the model of
  the run.

A dimension and a scan are validated when they are created and never change:
the values are copied into read-only arrays, and running a scan never
changes the objects of the user. `sbmlsim.simulator.Simulator.run` runs a
scan.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from sbmlsim.result.scan import POINT, STATISTIC, STATUS, TIME
from sbmlsim.simulation.definition import Simulation, Time, _encode
from sbmlsim.units import Quantity, ureg

#: the names of the dimensions and variables of a result, which no dimension of
#: a scan takes
RESERVED: frozenset[str] = frozenset({TIME, POINT, STATISTIC, STATUS})


class DimensionKind(StrEnum):
    """What a dimension varies."""

    VALUES = "values"
    SIMULATIONS = "simulations"
    MODELS = "models"


def _array(target: str, values: Any) -> np.ndarray | Quantity:
    """Copy the values of a target into a read-only array.

    Raises:
        ValueError: if the values are a string, a scalar, empty or not one
            dimensional numbers.
    """
    if isinstance(values, str):
        raise ValueError(
            f"The values of '{target}' are the string '{values}': the values of a "
            f"dimension are numbers, a formula per point is a dimension of "
            f"simulations."
        )
    magnitude = values.magnitude if isinstance(values, Quantity) else values
    try:
        array = np.array(magnitude, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"The values of '{target}' are not numbers: {err}") from err
    if array.ndim != 1:
        raise ValueError(
            f"The values of '{target}' must be an array of one value per point, "
            f"not of the shape {array.shape}; a scalar is a change of the "
            f"simulation."
        )
    if array.size == 0:
        raise ValueError(f"The values of '{target}' are empty.")
    array.setflags(write=False)
    if isinstance(values, Quantity):
        return ureg.Quantity(array, values.units)
    return array


def _not_empty(mapping: object) -> bool:
    """Check whether an argument of a dimension is no empty mapping."""
    return not isinstance(mapping, Mapping) or len(mapping) > 0


def _model_text(model: Any) -> str:
    """Get the path of a model, or its id, for the provenance of a scan."""
    source = getattr(model, "source", None)
    if source is None:
        return str(model)
    if source.path is not None:
        return str(source.path)
    return str(getattr(model, "sid", None) or "<sbml>")


@dataclass(frozen=True, init=False, eq=False)
class Dimension:
    """A dimension of a scan, see the module.

    Attributes:
        id: the id of the dimension, its name in the result.
        kind: what the dimension varies.
        values: target -> a quantity or a read-only array of numbers in the
            unit of the target in the model (typed `Any`, so callers index it
            and take its `magnitude` without a check); empty unless `kind` is `VALUES`.
        simulations: label -> simulation; empty unless `kind` is
            `SIMULATIONS`.
        models: label -> model; empty unless `kind` is `MODELS`.
        at: the time of the values, `None` for the rule of
            `Simulation.with_values`.
        labels: the read-only coordinate of the dimension.
    """

    id: str
    # follows from the mapping which is given, see `__init__`
    kind: DimensionKind = field(init=False)
    values: Mapping[str, Any]
    simulations: Mapping[str, Simulation]
    models: Mapping[str, Any]
    at: Time | None
    labels: np.ndarray

    def __init__(
        self,
        id: str,
        *,
        values: Mapping[str, Any] | None = None,
        simulations: Mapping[str, Simulation] | None = None,
        models: Mapping[str, Any] | None = None,
        at: Time | None = None,
        labels: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        """Create a dimension, see the class.

        `dataclasses.replace` creates a dimension with other fields, e.g.
        another `at`; it passes the empty mappings of the other kinds, which
        count as not given next to a mapping which is not empty. Its labels
        are the given ones, `labels=None` takes the default ones.

        Raises:
            ValueError: if not exactly one of `values`, `simulations` and
                `models` is given, if it is empty, if the arrays of the values
                differ in their length or are no arrays of numbers, if a
                dimension which is no dimension of values has `at`, or if the
                labels do not fit.
        """
        mappings = {"values": values, "simulations": simulations, "models": models}
        given = [name for name, mapping in mappings.items() if mapping is not None]
        if len(given) > 1:
            given = [name for name in given if _not_empty(mappings[name])] or given
        if len(given) != 1:
            raise ValueError(
                f"The dimension '{id}' needs exactly one of 'values', "
                f"'simulations' and 'models', it has {given or 'none'}."
            )
        kind = DimensionKind(given[0])
        chosen = mappings[given[0]]
        if not isinstance(chosen, Mapping):
            raise ValueError(
                f"The {kind} of the dimension '{id}' must be a mapping, not {chosen!r}."
            )
        if isinstance(at, Quantity) and not at.check("[time]"):
            raise ValueError(
                f"The time 'at' of the dimension '{id}' must have a time "
                f"unit, not '{at.units}'."
            )
        arrays: dict[str, Any] = {}
        sims: dict[str, Simulation] = {}
        mods: dict[str, Any] = {}
        if kind is DimensionKind.VALUES:
            if not values:
                raise ValueError(f"The dimension '{id}' has no values.")
            arrays = {target: _array(target, v) for target, v in values.items()}
            lengths = {target: len(array) for target, array in arrays.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"The values of the dimension '{id}' have different lengths "
                    f"{lengths}: point k sets the k-th value of every target."
                )
            keys: list[Any] = list(range(next(iter(lengths.values()))))
        else:
            if at is not None:
                raise ValueError(
                    f"The dimension '{id}' of {kind} has a time 'at', which only "
                    f"a dimension of values has."
                )
            if kind is DimensionKind.SIMULATIONS:
                sims = dict(simulations or {})
                for label, simulation in sims.items():
                    if not isinstance(simulation, Simulation):
                        raise ValueError(
                            f"'{label}' of the dimension '{id}' is no Simulation: "
                            f"{simulation!r}."
                        )
            else:
                mods = dict(models or {})
            keys = list(sims or mods)
            if not keys:
                raise ValueError(f"The dimension '{id}' has no {kind}.")
        if labels is not None and (
            isinstance(labels, str) or not isinstance(labels, Iterable)
        ):
            raise ValueError(
                f"The labels {labels!r} of the dimension '{id}' must be a "
                f"sequence of {len(keys)} labels."
            )
        try:
            coordinate = np.array(keys if labels is None else list(labels))
        except TypeError as err:
            # e.g. an array of zero dimensions
            raise ValueError(
                f"The labels {labels!r} of the dimension '{id}' must be a "
                f"sequence of {len(keys)} labels."
            ) from err
        if coordinate.ndim != 1 or coordinate.size != len(keys):
            raise ValueError(
                f"The dimension '{id}' has {len(keys)} points, its labels "
                f"{coordinate.tolist()} do not fit."
            )
        if len(set(coordinate.tolist())) != coordinate.size:
            raise ValueError(
                f"The labels {coordinate.tolist()} of the dimension '{id}' are "
                f"not unique."
            )
        coordinate.setflags(write=False)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", MappingProxyType(arrays))
        object.__setattr__(self, "simulations", MappingProxyType(sims))
        object.__setattr__(self, "models", MappingProxyType(mods))
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "labels", coordinate)

    def __getstate__(self) -> dict[str, Any]:
        """Get the state, the read-only mappings as plain dictionaries."""
        state = dict(self.__dict__)
        for name in ("values", "simulations", "models"):
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state, the dictionaries become read-only mappings again."""
        for name, value in state.items():
            if name in ("values", "simulations", "models"):
                value = MappingProxyType(value)
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        """Get the number of points."""
        return int(self.labels.size)

    def __repr__(self) -> str:
        """Get the representation."""
        what = list(self.values or self.simulations or self.models)
        at = "" if self.at is None else f", at={self.at}"
        return f"Dimension({self.id}[{len(self)}], {self.kind}={what}{at})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "id": self.id,
            "kind": str(self.kind),
            "labels": self.labels.tolist(),
            "values": {
                target: _encode(values)
                if isinstance(values, Quantity)
                else values.tolist()
                for target, values in self.values.items()
            },
            "simulations": {
                label: simulation.to_dict()
                for label, simulation in self.simulations.items()
            },
            "models": {
                label: _model_text(model) for label, model in self.models.items()
            },
            "at": _encode(self.at),
        }


def _validate(simulation: Simulation, dimensions: tuple[Dimension, ...]) -> None:
    """Check the dimensions of a scan, see `Scan`.

    Raises:
        ValueError: see `Scan`.
    """
    ids = [dimension.id for dimension in dimensions]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ValueError(
            f"The dimensions {duplicates} appear more than once in the scan."
        )
    reserved = sorted(set(ids) & RESERVED)
    if reserved:
        raise ValueError(
            f"The dimension ids {reserved} are names of the result "
            f"({sorted(RESERVED)}), choose other ids."
        )
    targets = {target for dimension in dimensions for target in dimension.values}
    clash = sorted(set(ids) & targets)
    if clash:
        raise ValueError(
            f"The dimension ids {clash} are changed targets, which are "
            f"coordinates of the result, choose other ids."
        )
    for kind in (DimensionKind.SIMULATIONS, DimensionKind.MODELS):
        count = sum(dimension.kind is kind for dimension in dimensions)
        if count > 1:
            raise ValueError(
                f"A scan has at most one dimension of {kind}, this one has {count}."
            )
    simulations = next(
        (
            list(dimension.simulations.values())
            for dimension in dimensions
            if dimension.kind is DimensionKind.SIMULATIONS
        ),
        [simulation],
    )
    seen: dict[tuple[str, Any], str] = {}
    for dimension in dimensions:
        # the same time in another spelling, e.g. 60 min and 1 hr, is one time
        key = (
            None
            if dimension.at is None
            else tuple(round(sim._magnitude(dimension.at), 9) for sim in simulations)
        )
        for target in dimension.values:
            other = seen.get((target, key))
            if other is not None:
                when = (
                    "before the initialization"
                    if dimension.at is None
                    else f"at {dimension.at}"
                )
                raise ValueError(
                    f"The target '{target}' is set {when} by the dimensions "
                    f"'{other}' and '{dimension.id}': a point sets a target once."
                )
            seen[(target, key)] = dimension.id
    for dimension in dimensions:
        if dimension.at is None:
            continue
        for sim in simulations:
            at = sim._magnitude(dimension.at)
            if not sim._magnitude(sim.start) <= at <= sim._magnitude(sim.end):
                raise ValueError(
                    f"The time {dimension.at} of the dimension '{dimension.id}' is "
                    f"outside of the simulation [{sim.start}, {sim.end}]."
                )


@dataclass(frozen=True, init=False, eq=False)
class Scan:
    """A simulation over the points of its dimensions, see the module.

    Attributes:
        simulation: the simulation, which a dimension of simulations replaces.
        dimensions: the dimensions, the last one fastest.
    """

    simulation: Simulation
    dimensions: tuple[Dimension, ...]

    def __init__(
        self, simulation: Simulation, dimensions: Sequence[Dimension] = ()
    ) -> None:
        """Create a scan, see the class.

        Raises:
            ValueError: if two dimensions have one id; if an id is in
                `RESERVED` or a changed target; if a target is set by two
                dimensions without `at` or by two with the same `at`; if there
                is more than one dimension of simulations or of models; or if
                the `at` of a dimension is outside of a simulation it applies
                to.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError(f"A scan needs a Simulation, not {simulation!r}.")
        dims = tuple(dimensions)
        _validate(simulation, dims)
        object.__setattr__(self, "simulation", simulation)
        object.__setattr__(self, "dimensions", dims)

    @classmethod
    def of(cls, scan: Scan | Simulation) -> Scan:
        """Get a scan, a simulation is a scan without dimensions."""
        return scan if isinstance(scan, Scan) else cls(scan)

    def __repr__(self) -> str:
        """Get the representation."""
        return f"Scan({self.simulation!r}, {list(self.dimensions)})"

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the number of points of every dimension."""
        return tuple(len(dimension) for dimension in self.dimensions)

    @property
    def size(self) -> int:
        """Get the number of points of the scan."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        """Get the number of points of the scan."""
        return self.size

    @property
    def dims(self) -> tuple[str, ...]:
        """Get the ids of the dimensions."""
        return tuple(dimension.id for dimension in self.dimensions)

    def dimension(self, kind: DimensionKind) -> Dimension | None:
        """Get the dimension of simulations or of models, `None` without one."""
        return next((d for d in self.dimensions if d.kind is kind), None)

    def simulations(self) -> list[Simulation]:
        """Get the simulations of the points, those of a dimension of simulations."""
        dimension = self.dimension(DimensionKind.SIMULATIONS)
        if dimension is None:
            return [self.simulation]
        return list(dimension.simulations.values())

    def points(self) -> Iterator[tuple[int, ...]]:
        """Get the index of every point along every dimension, in C order."""
        return iter(np.ndindex(*self.shape))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types, the provenance of a result."""
        return {
            "type": self.__class__.__name__,
            "simulation": self.simulation.to_dict(),
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
        }
