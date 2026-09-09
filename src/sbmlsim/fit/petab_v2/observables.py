"""Observables of a PEtab problem which are formulas.

An observable of PEtab is a math expression over the entities of a model,
e.g. the fraction

    (100 * pApB + 200 * pApA * specC17) / (pApB + STAT5A * specC17 + ...)

of a phosphorylated protein, while `sbmlsim` observes what roadrunner selects,
i.e. an entity of the model. The formulas of a problem therefore become
entities: a copy of the model gets a parameter per formula observable and an
assignment rule which is the formula, and the fit selects the parameter.

The math of PEtab is the math of the model, i.e. the identifier of a species is
its amount or its concentration exactly as the model means it, so the formula
is the rule without translating it.
"""

import logging
from pathlib import Path
from typing import Any

import libsbml

logger = logging.getLogger(__name__)

#: prefix of the parameter of an observable which is a formula
OBSERVABLE_PREFIX = "observable_"

#: suffix of the model which carries the observables
MODEL_SUFFIX = "_observables"


def observable_id(petab_id: str) -> str:
    """Get the identifier of the model entity of an observable.

    Args:
        petab_id: id of the observable in the PEtab problem.

    Returns:
        The identifier of the parameter which the assignment rule sets.
    """
    return f"{OBSERVABLE_PREFIX}{petab_id}"


def add_observables(
    sbml_path: Path, formulas: dict[str, str], output_path: Path
) -> Path:
    """Write a copy of a model which has the observables of a problem.

    Args:
        sbml_path: path of the model of the PEtab problem.
        formulas: math expression per observable, i.e. `observableFormula`.
        output_path: path the model with the observables is written to.

    Returns:
        The path of the model which was written.

    Raises:
        ValueError: if the model cannot be read, if a formula is not valid math
            or if the model with the observables is not valid SBML.
    """
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model = document.getModel()
    if model is None:
        raise ValueError(f"No model in the SBML of the problem: '{sbml_path}'")

    for petab_id, formula in formulas.items():
        sid = observable_id(petab_id)
        if model.getElementBySId(sid) is not None:
            raise ValueError(
                f"The model '{sbml_path}' already has an entity '{sid}', which "
                f"is the entity of the observable '{petab_id}'."
            )
        math: libsbml.ASTNode = libsbml.parseL3FormulaWithModel(formula, model)
        if math is None:
            raise ValueError(
                f"The formula of the observable '{petab_id}' is not valid math: "
                f"'{formula}': {libsbml.getLastParseL3Error()}"
            )

        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setName(petab_id)
        parameter.setConstant(False)
        rule: libsbml.AssignmentRule = model.createAssignmentRule()
        rule.setVariable(sid)
        rule.setMath(math)

    document.checkConsistency()
    errors = [
        document.getError(k)
        for k in range(document.getNumErrors())
        if document.getError(k).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    ]
    if errors:
        messages = "; ".join(error.getMessage().strip() for error in errors)
        raise ValueError(
            f"The model with the observables of '{sbml_path}' is not valid "
            f"SBML: {messages}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(document, str(output_path))
    logger.info(
        "The observables %s are entities of '%s'",
        sorted(formulas),
        output_path.name,
    )
    return output_path


def is_entity(formula: str, sbml_model: Any) -> bool:
    """Check whether a formula is an entity of the model rather than math.

    Args:
        formula: the `observableFormula` of an observable.
        sbml_model: `libsbml.Model` of the problem.

    Returns:
        Whether the formula is the identifier of an entity, i.e. whether the
        fit can select it without adding it to the model.
    """
    sid = formula.strip()
    if not sid.isidentifier():
        return False
    if sbml_model is None:
        return True
    return sbml_model.getElementBySId(sid) is not None
