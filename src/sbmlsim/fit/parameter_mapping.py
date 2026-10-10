"""The binding of the parameters of a fit to the simulations they apply to.

A `FitParameter` writes its value to an entity of a model. A parameter which
carries a selector writes it only for the fit mappings the selector passes, so
one entity is estimated separately for parts of the data, e.g. an absorption
rate once for the tablet arms and once for the solution arms of one fit.

`ParameterMapping` is that binding as an object: it resolves the selectors to
the simulation groups of an initialized `OptimizationProblem`, refuses a
binding which cannot be simulated, and answers which changes a group is run
with. It is the same knowledge a PEtab problem keeps in its condition table,
which is why `sbmlsim.fit.petab_v2` reads it instead of deriving it again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sbmlsim.fit.objects import FitParameter
from sbmlsim.units import (
    DimensionalityError,
    UnitRegistry,
    UnitsInformation,
)
from sbmlsim.units import ureg as package_ureg

logger = logging.getLogger(__name__)


def _same_unit(a: str, b: str, ureg: UnitRegistry, parameters: str) -> bool:
    """Check whether two units are one unit.

    Args:
        a: a unit of pint.
        b: another unit of pint.
        ureg: the registry of the units.
        parameters: the parameters of the units, for the message.

    Returns:
        Whether a value in `a` is the same value in `b`, e.g. for `1/min`
        and `min^-1`.

    Raises:
        Exception: the error of pint if a unit cannot be read, e.g. the
            `UndefinedUnitError` of a unit pint does not define, with a note
            which names the parameters.
    """
    try:
        quantity_a = ureg.Quantity(1.0, a)
        quantity_b = ureg.Quantity(1.0, b)
    # pint raises errors of many kinds for a unit it cannot read: its own, a
    # ValueError for a factor, a TypeError or the TokenError of its parser
    except Exception as err:
        err.add_note(
            f"The unit of {parameters} cannot be read, it is neither a unit of "
            f"pint nor a unit id of the model of its target."
        )
        raise
    try:
        factor = quantity_a.to(quantity_b.units).magnitude
    except DimensionalityError:
        return False
    return bool(np.isclose(factor, 1.0))


def has_renamed_targets(parameters: Sequence[FitParameter]) -> bool:
    """Check whether any parameter writes an entity of another name.

    This is not `FitParameter.is_versioned`, which asks whether a parameter
    carries a selector: a parameter can be versioned and still write its own
    id (several versions all named after the target with a selector each), or
    write a different id without a selector (a plain rename). What a fit's
    display, its report and the PEtab export need instead is this question,
    i.e. whether a target column is worth showing at all -- one parameter
    answers it with a one-element list, a whole parameter vector with all of
    them.

    Args:
        parameters: the parameters to check, e.g. an `OptimizationProblem`'s
            or a single `[parameter]`.

    Returns:
        `True` if some parameter's `target_id` differs from its `pid`.
    """
    return any(p.target_id != p.pid for p in parameters)


@dataclass(frozen=True)
class CoverageRow:
    """What one parameter of a fit reaches.

    Attributes:
        pid: id of the estimated parameter.
        target: entity of the model it writes.
        n_covered: number of simulations it applies to.
        n_groups: number of simulations of the problem.
        uncovered_groups: the simulations it does not apply to, by name. They
            keep the value the model has for the target.
    """

    pid: str
    target: str
    n_covered: int
    n_groups: int
    uncovered_groups: list[str] = field(default_factory=list)


class ParameterMapping:
    """Which parameter writes which entity in which simulation."""

    def __init__(
        self,
        parameters: Sequence[FitParameter],
        mapping_indices: dict[int, set[int]],
        groups: Sequence[Sequence[int]],
        mapping_keys: Sequence[str],
        group_names: Sequence[str] | None = None,
        units: Sequence[UnitsInformation] | None = None,
    ) -> None:
        """Resolve and validate the binding.

        Args:
            parameters: the parameters of the problem, in the order of the
                parameter vector.
            mapping_indices: for the index of every versioned parameter the
                indices of the fit mappings its selector passed. A parameter
                which is not versioned is absent and applies everywhere.
            groups: the simulation groups, each a list of mapping indices, see
                `OptimizationProblem.mapping_groups`.
            mapping_keys: id of every fit mapping, for the messages.
            group_names: name of every simulation group, for the coverage. The
                index of the group is used when they are not given.
            units: the units of the model of every simulation group, which
                resolve the unit ids of a model in the unit of a parameter,
                see `parameter_units`. Without them the unit of a parameter
                is a unit of pint.

        Raises:
            ValueError: if two parameters write one target in one simulation,
                if a selector splits a simulation, if the unit of a parameter
                means different units in the models of the simulations it
                reaches, or if the versions of a target disagree on their
                unit.
            UndefinedUnitError: if a unit of a parameter which is compared is
                neither a unit of pint nor a unit id of a model it reaches;
                pint raises errors of other kinds for a unit it cannot read.
        """
        self.parameters = list(parameters)
        self.groups = [list(group) for group in groups]
        self.group_names = (
            list(group_names)
            if group_names is not None
            else [str(k) for k in range(len(self.groups))]
        )

        #: index of the parameter which writes a target, per group
        self._by_group: list[dict[str, int]] = [
            self._resolve_group(k, group, mapping_indices, mapping_keys)
            for k, group in enumerate(self.groups)
        ]
        self._covered: dict[int, set[int]] = {}
        for k, targets in enumerate(self._by_group):
            for index in targets.values():
                self._covered.setdefault(index, set()).add(k)

        # the registry the units of the parameters are compared in
        ureg = units[0].ureg if units else package_ureg
        #: the unit of every parameter in the units of pint, a unit id of the
        #: model of its target replaced by its expression, see
        #: `UnitsInformation.resolve_unit`; `None` for a parameter in the unit
        #: of the model
        self.parameter_units: list[str | None] = [
            self._resolve_unit(index, units, ureg)
            for index in range(len(self.parameters))
        ]
        self._check_units(ureg)
        self._warn_about_uncovered_versions()

    def _warn_about_uncovered_versions(self) -> None:
        """Warn about a versioned parameter whose selector covers nothing.

        Such a parameter stays in the parameter vector and never changes the
        model, which makes the objective flat in it: a silent trap for a
        selector which was mistyped or a filter which is too strict.
        """
        for index, parameter in enumerate(self.parameters):
            if parameter.is_versioned and not self._covered.get(index):
                logger.warning(
                    "'%s' (writing '%s') covers no simulation: its selector "
                    "matched no fit mapping, so the parameter never changes "
                    "the model and the objective is flat in it.",
                    parameter.pid,
                    parameter.target_id,
                )

    def _resolve_unit(
        self,
        index: int,
        units: Sequence[UnitsInformation] | None,
        ureg: UnitRegistry,
    ) -> str | None:
        """Resolve the unit of a parameter in the models of the simulations it reaches.

        A parameter which reaches no simulation is resolved in the models of
        all simulations: it is in the parameter vector all the same.

        Args:
            index: index of the parameter.
            units: the units of the model of every group, see `__init__`.
            ureg: the registry the units are compared in.

        Returns:
            The unit in the units of pint, `None` for a parameter without a
            unit.

        Raises:
            ValueError: if the unit means different units in the models.
        """
        parameter = self.parameters[index]
        unit = parameter.unit
        if unit is None or not units:
            return unit
        # every resolved unit with the name of the first group it is resolved in
        resolved: dict[str, str] = {}
        for k in sorted(self._covered.get(index, set())) or range(len(self.groups)):
            resolved.setdefault(units[k].resolve_unit(unit), self.group_names[k])
        if not resolved:
            return unit
        first, *others = resolved
        for other in others:
            if not _same_unit(first, other, ureg, f"'{parameter.pid}'"):
                raise ValueError(
                    f"The unit '{unit}' of '{parameter.pid}' means '{first}' in "
                    f"'{resolved[first]}' and '{other}' in '{resolved[other]}': "
                    f"a unit id is resolved in the models of the simulations a "
                    f"parameter reaches, so it must mean one unit in all of them."
                )
        return first

    def _check_units(self, ureg: UnitRegistry) -> None:
        """Check that the versions of a target are given in one unit.

        The units are compared as units, e.g. `1/min` and `min^-1` or a unit
        id of the model and its expression are one unit.

        Args:
            ureg: the registry the units are compared in.

        Raises:
            ValueError: if two parameters of one target have different units.
        """
        first: dict[str, int] = {}
        for index, parameter in enumerate(self.parameters):
            target = parameter.target_id
            if target not in first:
                first[target] = index
                continue
            other = self.parameters[first[target]]
            a = self.parameter_units[first[target]]
            b = self.parameter_units[index]
            if a != b and (
                a is None
                or b is None
                or not _same_unit(a, b, ureg, f"'{other.pid}' or '{parameter.pid}'")
            ):
                raise ValueError(
                    f"The versions of '{target}' disagree on their unit: "
                    f"'{other.pid}' is '{other.unit}' and '{parameter.pid}' is "
                    f"'{parameter.unit}'. The unit is how the value reaches "
                    f"the model, so it is one unit for one entity."
                )

    def _resolve_group(
        self,
        group_index: int,
        group: Sequence[int],
        mapping_indices: dict[int, set[int]],
        mapping_keys: Sequence[str],
    ) -> dict[str, int]:
        """Get the parameter which writes every target of one simulation.

        Args:
            group_index: index of the simulation group.
            group: indices of the fit mappings of the group.
            mapping_indices: the mappings every versioned parameter selected.
            mapping_keys: id of every fit mapping, for the messages.

        Returns:
            The index of the parameter which writes a target, by target.

        Raises:
            ValueError: if two parameters write one target in the group.
        """
        # the parameter which writes a target, and the mapping it came from
        chosen: dict[str, tuple[int, int | None]] = {}
        for index, parameter in enumerate(self.parameters):
            target = parameter.target_id
            hits = (
                sorted(set(group) & mapping_indices.get(index, set()))
                if parameter.is_versioned
                else [None]
            )
            for hit in hits:
                if target in chosen:
                    other, other_hit = chosen[target]
                    if other == index:
                        # the same parameter selected another mapping of this
                        # group already: it covers the group, not a conflict
                        continue
                    raise self._conflict(
                        target, other, index, other_hit, hit, group_index, mapping_keys
                    )
                chosen[target] = (index, hit)

        return {target: index for target, (index, _) in chosen.items()}

    def _conflict(
        self,
        target: str,
        first: int,
        second: int,
        first_hit: int | None,
        second_hit: int | None,
        group_index: int,
        mapping_keys: Sequence[str],
    ) -> ValueError:
        """Build the error of two parameters writing one target."""
        a = self.parameters[first].pid
        b = self.parameters[second].pid
        if first_hit is not None and second_hit is not None and first_hit != second_hit:
            return ValueError(
                f"'{a}' and '{b}' both write '{target}' in one simulation: "
                f"'{a}' selects '{mapping_keys[first_hit]}' and '{b}' selects "
                f"'{mapping_keys[second_hit]}', which share the simulation "
                f"'{self.group_names[group_index]}'. A simulation has one "
                f"value for an entity, so a selector must not split one."
            )
        if first_hit == second_hit and first_hit is not None:
            return ValueError(
                f"'{a}' and '{b}' both write '{target}' for the fit mapping "
                f"'{mapping_keys[first_hit]}'. Two versions must not select "
                f"the same mapping."
            )
        return ValueError(
            f"'{a}' and '{b}' both write '{target}' in the simulation "
            f"'{self.group_names[group_index]}'. A parameter without a "
            f"selector applies everywhere, so it cannot be combined with a "
            f"version of its target: a target is estimated once for all of "
            f"the data or once per subset of it."
        )

    def indices_for(self, group: int) -> dict[str, int]:
        """Get the parameter which writes every target of a simulation.

        Args:
            group: index of the simulation group.

        Returns:
            The index into the parameter vector, by target. A target no
            parameter writes is absent, so the model keeps its value.
        """
        return self._by_group[group]

    def changes_for(self, group: int, quantities: Sequence[Any]) -> dict[str, Any]:
        """Get the changes a simulation is run with.

        Args:
            group: index of the simulation group.
            quantities: the quantity of every parameter, in the order of the
                parameter vector. They are built once per evaluation of the
                residuals and referenced here.

        Returns:
            The quantity by entity of the model. A target which is not an
            entity of the model, see `FitParameter.is_external`, is not a
            change: the derived changes of the problem read its value.
        """
        return {
            target: quantities[index]
            for target, index in self._by_group[group].items()
            if not self.parameters[index].is_external
        }

    def coverage(self) -> list[CoverageRow]:
        """Get what every parameter reaches, for the console and the report."""
        n_groups = len(self.groups)
        rows = []
        for index, parameter in enumerate(self.parameters):
            covered = self._covered.get(index, set())
            rows.append(
                CoverageRow(
                    pid=parameter.pid,
                    target=parameter.target_id,
                    n_covered=len(covered),
                    n_groups=n_groups,
                    uncovered_groups=[
                        name
                        for k, name in enumerate(self.group_names)
                        if k not in covered
                    ],
                )
            )
        return rows

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{len(self.parameters)} parameters, "
            f"{len(self.groups)} simulations>"
        )
