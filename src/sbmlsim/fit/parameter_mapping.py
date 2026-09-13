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

from sbmlsim.fit.objects import FitParameter

logger = logging.getLogger(__name__)


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

        Raises:
            ValueError: if two parameters write one target in one simulation,
                if a selector splits a simulation, or if the versions of a
                target disagree on their unit.
        """
        self.parameters = list(parameters)
        self.groups = [list(group) for group in groups]
        self.group_names = (
            list(group_names)
            if group_names is not None
            else [str(k) for k in range(len(self.groups))]
        )

        self._check_units()
        #: index of the parameter which writes a target, per group
        self._by_group: list[dict[str, int]] = [
            self._resolve_group(k, group, mapping_indices, mapping_keys)
            for k, group in enumerate(self.groups)
        ]
        self._covered: dict[int, set[int]] = {}
        for k, targets in enumerate(self._by_group):
            for index in targets.values():
                self._covered.setdefault(index, set()).add(k)

    def _check_units(self) -> None:
        """Check that the versions of a target are given in one unit.

        Raises:
            ValueError: if two parameters of one target have different units.
        """
        units: dict[str, tuple[str, str | None]] = {}
        for parameter in self.parameters:
            target = parameter.target_id
            if target in units:
                pid, unit = units[target]
                if unit != parameter.unit:
                    raise ValueError(
                        f"The versions of '{target}' disagree on their unit: "
                        f"'{pid}' is '{unit}' and '{parameter.pid}' is "
                        f"'{parameter.unit}'. The unit is how the value reaches "
                        f"the model, so it is one unit for one entity."
                    )
            else:
                units[target] = (parameter.pid, parameter.unit)

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

    @property
    def is_versioned(self) -> bool:
        """Check whether any parameter writes an entity of another name."""
        return any(p.target_id != p.pid for p in self.parameters)

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
            The quantity by entity of the model.
        """
        return {
            target: quantities[index] for target, index in self._by_group[group].items()
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
