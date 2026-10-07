"""The symbols of a model which a simulation can change.

A target of a change is named like a selection of roadrunner: `S` is the
amount of a species, `[S]` its concentration, a parameter or a compartment is
its id. `ModelSymbols` reads what the targets of a model are from its SBML once
and answers what kind a target is, which the compile step and the executor
need: a compartment change keeps the concentration of the species in it, a
parameter with an initial assignment cannot be set before the initialization
without deriving the model, and the target of an assignment rule cannot be set
at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import libsbml


class TargetKind(StrEnum):
    """The kind of a target of a change."""

    PARAMETER = "parameter"
    COMPARTMENT = "compartment"
    SPECIES_AMOUNT = "species amount"
    SPECIES_CONCENTRATION = "species concentration"


def _read(sbml: str | Path) -> libsbml.SBMLDocument:
    """Read the document of SBML, a path or the SBML itself.

    Raises:
        ValueError: if the SBML has no model.
    """
    text = str(sbml)
    doc: libsbml.SBMLDocument = (
        libsbml.readSBMLFromString(text)
        if text.lstrip().startswith("<")
        else libsbml.readSBMLFromFile(text)
    )
    if doc.getModel() is None:
        raise ValueError(f"The SBML has no model: '{text[:200]}'")
    return doc


@dataclass(frozen=True)
class ModelSymbols:
    """The entities of a model a simulation refers to.

    Attributes:
        parameters: ids of the global parameters.
        compartments: ids of the compartments.
        species: ids of the species.
        species_compartment: species -> its compartment.
        only_substance: species with `hasOnlySubstanceUnits=true`.
        initial_assignments: entities with an initial assignment.
        assignment_rules: entities which are the variable of an assignment
            rule.
        rate_rules: entities which are the variable of a rate rule.
        initial_assignment_order: the entities with an initial assignment, an
            assignment after the ones it reads.
        initial_assignment_dependencies: entity with an initial assignment ->
            the entities its math reads, through assignment rules.
        initial_concentration: species whose initial value is a
            concentration, which a change of their compartment before the
            initialization keeps.
    """

    parameters: frozenset[str]
    compartments: frozenset[str]
    species: frozenset[str]
    species_compartment: dict[str, str]
    only_substance: frozenset[str]
    initial_assignments: frozenset[str]
    assignment_rules: frozenset[str]
    rate_rules: frozenset[str]
    initial_assignment_order: tuple[str, ...] = ()
    initial_assignment_dependencies: dict[str, frozenset[str]] | None = None
    initial_concentration: frozenset[str] = frozenset()

    @classmethod
    def from_sbml(cls, sbml: str | Path) -> ModelSymbols:
        """Read the symbols of a model.

        Args:
            sbml: path of the SBML file or the SBML itself.

        Returns:
            The symbols of the model.
        """
        doc = _read(sbml)
        model: libsbml.Model = doc.getModel()
        species: list[libsbml.Species] = list(model.getListOfSpecies())
        rules: list[libsbml.Rule] = list(model.getListOfRules())
        rule_symbols = {
            r.getVariable(): _names(r.getMath()) for r in rules if r.isAssignment()
        }
        dependencies: dict[str, frozenset[str]] = {}
        for assignment in model.getListOfInitialAssignments():
            dependencies[assignment.getSymbol()] = frozenset(
                _expand(_names(assignment.getMath()), rule_symbols)
            )
        return cls(
            initial_assignment_order=_order(dependencies),
            initial_assignment_dependencies=dependencies,
            initial_concentration=frozenset(
                sp.getId()
                for sp in species
                if (
                    sp.isSetInitialConcentration()
                    or (
                        not sp.isSetInitialAmount()
                        and not sp.getHasOnlySubstanceUnits()
                    )
                )
                or (sp.getId() in dependencies and not sp.getHasOnlySubstanceUnits())
            ),
            parameters=frozenset(p.getId() for p in model.getListOfParameters()),
            compartments=frozenset(c.getId() for c in model.getListOfCompartments()),
            species=frozenset(s.getId() for s in species),
            species_compartment={s.getId(): s.getCompartment() for s in species},
            only_substance=frozenset(
                s.getId() for s in species if s.getHasOnlySubstanceUnits()
            ),
            initial_assignments=frozenset(
                a.getSymbol() for a in model.getListOfInitialAssignments()
            ),
            assignment_rules=frozenset(
                r.getVariable() for r in rules if r.isAssignment()
            ),
            rate_rules=frozenset(r.getVariable() for r in rules if r.isRate()),
        )

    @staticmethod
    def entity(target: str) -> str:
        """Get the entity of a target, the species of a concentration `[S]`."""
        if target.startswith("[") and target.endswith("]"):
            return target[1:-1]
        return target

    def kind(self, target: str) -> TargetKind:
        """Get the kind of a target.

        Args:
            target: `S` for the amount of a species, `[S]` for its
                concentration, or the id of a parameter or a compartment.

        Returns:
            The kind of the target.

        Raises:
            ValueError: if the model has no such entity, if a concentration is
                not one of a species, or if the entity is the variable of an
                assignment rule, which a change cannot set.
        """
        entity = self.entity(target)
        if entity in self.assignment_rules:
            raise ValueError(
                f"'{target}' is the variable of an assignment rule, its value "
                f"is the rule and a change cannot set it."
            )
        if entity != target:
            if entity not in self.species:
                raise ValueError(
                    f"'{target}' is a concentration, but '{entity}' is no "
                    f"species of the model."
                )
            return TargetKind.SPECIES_CONCENTRATION
        if entity in self.species:
            return TargetKind.SPECIES_AMOUNT
        if entity in self.compartments:
            return TargetKind.COMPARTMENT
        if entity in self.parameters:
            return TargetKind.PARAMETER
        raise ValueError(
            f"'{target}' is not a parameter, a compartment or a species of the model."
        )


def _names(math: libsbml.ASTNode | None) -> set[str]:
    """Get the identifiers a math reads, without the names of functions."""
    names: set[str] = set()
    if math is None:
        return names
    stack = [math]
    while stack:
        node = stack.pop()
        # `isName` and not the type: with a second SWIG module of libsbml
        # loaded, e.g. by `sbmlmath`, `getType` answers a pointer of an enum
        if node.isName() and node.getName():
            names.add(node.getName())
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return names


def _expand(names: set[str], rules: dict[str, set[str]]) -> set[str]:
    """Expand identifiers through the assignment rules which set them."""
    expanded: set[str] = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in expanded:
            continue
        expanded.add(name)
        stack.extend(rules.get(name, ()))
    return expanded


def _order(dependencies: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """Order the initial assignments, an assignment after the ones it reads.

    Raises:
        ValueError: if the initial assignments read each other in a cycle.
    """
    order: list[str] = []
    done: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = sorted(
            target
            for target, reads in remaining.items()
            if not (reads & set(remaining)) - {target}
        )
        if not ready:
            raise ValueError(
                f"The initial assignments of {sorted(remaining)} read each other "
                f"in a cycle."
            )
        for target in ready:
            order.append(target)
            done.add(target)
            del remaining[target]
    return tuple(order)
