"""Translate the selections of roadrunner into the math of PEtab v2.

`sbmlsim` names an observable and the target of a change with a selection of
roadrunner, where `S1` is the amount of a species and `[S1]` its concentration.
PEtab has no selections: an observable is a math expression over the entities
of the model and the target of a condition is the identifier of an entity, and
what the identifier of a species means is decided by the model:

- `hasOnlySubstanceUnits=true`, i.e. an amount based species, is an amount,
- `hasOnlySubstanceUnits=false`, i.e. a concentration based species, is a
  concentration,

for the math of the observables (PEtab v2, observable table) and for the value
a condition assigns (PEtab v2, reinitialization semantics). Dropping the
brackets of a selection is therefore only right when the two agree, and this
module converts between them instead: the concentration of an amount based
species is `S1 / compartment` and the amount of a concentration based species
is `S1 * compartment`.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: a roadrunner selection of a concentration, i.e. `[S1]`
CONCENTRATION = re.compile(r"^\[(?P<sid>.+)\]$")


def split_selection(selection: str) -> tuple[str, bool]:
    """Split a selection into the entity and whether it is a concentration.

    Args:
        selection: selection of roadrunner, e.g. `[S1]` or `S1`.

    Returns:
        The identifier of the entity and whether the selection is the
        concentration of a species.
    """
    match = CONCENTRATION.match(selection)
    if match:
        return match.group("sid"), True
    return selection, False


def _species(sbml_model: Any, sid: str) -> Any | None:
    """Get the species of the model, `None` if the entity is not a species."""
    if sbml_model is None:
        return None
    return sbml_model.getSpecies(sid)


def observable_formula(selection: str, sbml_model: Any = None) -> str:
    """Get the PEtab math expression of a selection.

    Args:
        selection: selection of roadrunner, e.g. `[S1]` or `Aurine`.
        sbml_model: `libsbml.Model` of the problem. Without it the brackets are
            dropped, which is right when the selection and the model agree on
            amount and concentration.

    Returns:
        The expression which has the value of the selection, i.e. the
        identifier of the entity or the identifier scaled by its compartment.
    """
    sid, is_concentration = split_selection(selection)
    species = _species(sbml_model, sid)
    if species is None:
        # a parameter, a compartment or a reaction, which has one value
        return sid

    amount_based = bool(species.getHasOnlySubstanceUnits())
    compartment = species.getCompartment()
    if is_concentration and amount_based:
        # the model says amount, the fit wants the concentration
        return f"{sid} / {compartment}"
    if not is_concentration and not amount_based:
        # the model says concentration, the fit wants the amount
        return f"{sid} * {compartment}"
    return sid


def condition_target(selection: str, sbml_model: Any = None) -> str:
    """Get the target of a condition of a change of `sbmlsim`.

    The target of a condition is an identifier and not an expression, so a
    change whose selection means something else than the identifier does in the
    model cannot be written.

    Args:
        selection: selection of roadrunner which the change applies to.
        sbml_model: `libsbml.Model` of the problem.

    Returns:
        The identifier of the entity the condition changes.

    Raises:
        ValueError: if the change is the concentration of an amount based
            species or the amount of a concentration based species, which PEtab
            cannot assign.
    """
    sid, is_concentration = split_selection(selection)
    species = _species(sbml_model, sid)
    if species is None:
        return sid

    amount_based = bool(species.getHasOnlySubstanceUnits())
    if is_concentration and amount_based:
        raise ValueError(
            f"The change '{selection}' sets the concentration of the amount "
            f"based species '{sid}' (hasOnlySubstanceUnits=true). A condition of "
            f"PEtab assigns the amount of such a species, i.e. the change has no "
            f"PEtab representation; change the amount, or the species."
        )
    if not is_concentration and not amount_based:
        raise ValueError(
            f"The change '{selection}' sets the amount of the concentration "
            f"based species '{sid}' (hasOnlySubstanceUnits=false). A condition of "
            f"PEtab assigns the concentration of such a species, i.e. the change "
            f"has no PEtab representation; change the concentration, or the "
            f"species."
        )
    return sid


def selection_of_formula(formula: str, sbml_model: Any = None) -> str:
    """Get the selection of roadrunner of a PEtab math expression.

    This is the way back for a problem which does not carry the `sbmlsim`
    extension, i.e. a problem of another tool. Only an expression which is the
    identifier of an entity is a selection.

    Args:
        formula: the `observableFormula` of an observable.
        sbml_model: `libsbml.Model` of the problem.

    Returns:
        The selection of roadrunner which has the value of the expression.

    Raises:
        ValueError: if the expression is not the identifier of an entity, i.e.
            an observable which is a formula over several entities.
    """
    sid = formula.strip()
    if not sid.isidentifier():
        raise ValueError(
            f"The observable formula '{formula}' is not the identifier of an "
            f"entity of the model. `sbmlsim` observes the entities of a model, "
            f"an observable which is a formula is not supported."
        )
    species = _species(sbml_model, sid)
    if species is not None and not species.getHasOnlySubstanceUnits():
        # a concentration based species is its concentration in the math of
        # SBML, which roadrunner selects with brackets
        return f"[{sid}]"
    return sid
