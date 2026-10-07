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
    """

    parameters: frozenset[str]
    compartments: frozenset[str]
    species: frozenset[str]
    species_compartment: dict[str, str]
    only_substance: frozenset[str]
    initial_assignments: frozenset[str]
    assignment_rules: frozenset[str]
    rate_rules: frozenset[str]

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
        return cls(
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
