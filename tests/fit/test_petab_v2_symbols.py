"""Tests of the translation between roadrunner selections and PEtab math."""

import libsbml
import pytest

from sbmlsim.fit.petab_v2.symbols import (
    condition_target,
    observable_formula,
    selection_of_formula,
    split_selection,
)


@pytest.fixture(scope="module")
def sbml_document() -> libsbml.SBMLDocument:
    """Get a document with an amount based and a concentration based species.

    The document owns the model, so it has to stay alive as long as the model
    is used; libsbml is a SWIG wrapper and the model is a pointer into it.
    """
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel()
    compartment = model.createCompartment()
    compartment.setId("cyto")
    compartment.setConstant(True)
    compartment.setSize(1.0)

    for sid, only_substance in [("S_amount", True), ("S_conc", False)]:
        species = model.createSpecies()
        species.setId(sid)
        species.setCompartment("cyto")
        species.setHasOnlySubstanceUnits(only_substance)
        species.setBoundaryCondition(False)
        species.setConstant(False)
        species.setInitialAmount(1.0)

    parameter = model.createParameter()
    parameter.setId("k1")
    parameter.setConstant(True)
    parameter.setValue(1.0)
    return document


@pytest.fixture(scope="module")
def sbml_model(sbml_document: libsbml.SBMLDocument) -> libsbml.Model:
    """Get the model of the document, which keeps the document alive."""
    return sbml_document.getModel()


def test_split_selection() -> None:
    """The brackets of roadrunner mark a concentration."""
    assert split_selection("[S1]") == ("S1", True)
    assert split_selection("S1") == ("S1", False)


def test_formula_of_a_parameter(sbml_model: libsbml.Model) -> None:
    """A parameter has one value, its identifier is the formula."""
    assert observable_formula("k1", sbml_model) == "k1"


def test_formula_agrees_with_the_model(sbml_model: libsbml.Model) -> None:
    """A selection which means what the model means is the identifier.

    In the math of SBML a species with `hasOnlySubstanceUnits=true` is its
    amount and one with `false` is its concentration, which is what roadrunner
    selects as `S` and `[S]`.
    """
    assert observable_formula("S_amount", sbml_model) == "S_amount"
    assert observable_formula("[S_conc]", sbml_model) == "S_conc"


def test_formula_scales_with_the_compartment(sbml_model: libsbml.Model) -> None:
    """A selection which means the other thing is scaled by the compartment."""
    # the concentration of an amount based species
    assert observable_formula("[S_amount]", sbml_model) == "S_amount / cyto"
    # the amount of a concentration based species
    assert observable_formula("S_conc", sbml_model) == "S_conc * cyto"


def test_formula_without_a_model() -> None:
    """Without a model the brackets are dropped, which is the old behaviour."""
    assert observable_formula("[S1]") == "S1"
    assert observable_formula("S1") == "S1"


def test_condition_target_agrees_with_the_model(sbml_model: libsbml.Model) -> None:
    """A condition assigns what the model means by the identifier."""
    assert condition_target("k1", sbml_model) == "k1"
    assert condition_target("S_amount", sbml_model) == "S_amount"
    assert condition_target("[S_conc]", sbml_model) == "S_conc"


def test_condition_target_which_petab_cannot_assign(
    sbml_model: libsbml.Model,
) -> None:
    """A condition is an identifier, so it cannot scale by the compartment."""
    with pytest.raises(ValueError, match="concentration of the amount based"):
        condition_target("[S_amount]", sbml_model)
    with pytest.raises(ValueError, match="amount of the concentration based"):
        condition_target("S_conc", sbml_model)


def test_selection_of_formula(sbml_model: libsbml.Model) -> None:
    """The way back, for a problem which does not carry the extension."""
    assert selection_of_formula("S_amount", sbml_model) == "S_amount"
    assert selection_of_formula("S_conc", sbml_model) == "[S_conc]"
    assert selection_of_formula("k1", sbml_model) == "k1"


def test_selection_of_a_formula_which_is_not_an_entity(
    sbml_model: libsbml.Model,
) -> None:
    """An observable which is a formula is not a selection of roadrunner."""
    with pytest.raises(ValueError, match="not the identifier of an entity"):
        selection_of_formula("S_amount / cyto", sbml_model)


def test_the_round_trip_of_a_selection(sbml_model: libsbml.Model) -> None:
    """A selection which PEtab can express comes back as it was."""
    for selection in ["S_amount", "[S_conc]", "k1"]:
        formula = observable_formula(selection, sbml_model)
        assert selection_of_formula(formula, sbml_model) == selection
