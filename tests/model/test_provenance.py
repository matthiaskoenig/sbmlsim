"""Tests of the record of a derivation of a model."""

import copy
import dataclasses
import pickle
from collections.abc import Callable
from pathlib import Path

import libsbml
import pytest

from sbmlsim.fit.petab_v2.observables import add_observables
from sbmlsim.model.provenance import (
    NAMESPACE,
    Derivation,
    derivation_of,
    record_derivation,
    strip_derivation,
)
from sbmlsim.sciml import (
    Hybridization,
    Network,
    NetworkInput,
    NetworkPattern,
    compile_network,
)
from tests.sciml.hybrid import MODEL_PATH, feed_forward

#: an RDF annotation without a `metaid`, as the models of the PEtab SciML
#: test suite carry it
RDF = (
    '<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<rdf:Description rdf:about="#x"><dc:creator>who</dc:creator></rdf:Description>'
    "</rdf:RDF></annotation>"
)


def _read(path: Path) -> libsbml.Model:
    return libsbml.readSBMLFromFile(str(path)).getModel()


def _rhs(network: Network) -> Hybridization:
    return Hybridization(
        network=network,
        pattern=NetworkPattern.RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )


def test_a_model_without_annotation_gets_the_record(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    assert derivation_of(model) is None
    record_derivation(model, Path("lv.xml"), ["a", "b"], {"gamma": True})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    derivation = derivation_of(_read(path))
    assert derivation == Derivation("lv.xml", ("a", "b"), (("gamma", True),))


def test_the_record_keeps_an_rdf_annotation_without_metaid(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    assert model.setAnnotation(libsbml.XMLNode.convertStringToXMLNode(RDF)) == 0
    record_derivation(model, Path("lv.xml"), ["a"], {})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    model = _read(path)
    annotation = model.getAnnotation()
    names = {
        annotation.getChild(k).getName() for k in range(annotation.getNumChildren())
    }
    assert names == {"RDF", "derived"}
    assert derivation_of(model) == Derivation("lv.xml", ("a",))
    assert NAMESPACE in model.getAnnotationString()


def test_a_second_derivation_extends_the_record(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    record_derivation(model, Path("lv.xml"), ["a"], {"gamma": True})
    record_derivation(model, Path("lv_sciml.xml"), ["b"], {"a": False, "beta": True})
    derivation = derivation_of(model)
    # the source stays, a target which the first derivation created is no target
    assert derivation == Derivation(
        "lv.xml", ("a", "b"), (("gamma", True), ("beta", True))
    )
    annotation = model.getAnnotation()
    assert annotation.getNumChildren() == 1


def test_strip_gives_the_source_model_back(tmp_path: Path) -> None:
    network = feed_forward()
    compiled = compile_network(MODEL_PATH, [_rhs(network)], tmp_path / "lv_sciml.xml")
    derivation = derivation_of(_read(compiled))
    assert derivation is not None
    assert derivation.source == "lotka_volterra.xml"
    assert set(network.parameter_ids()) <= set(derivation.created)
    assert dict(derivation.targets) == {"gamma": True}

    document, stripped = strip_derivation(compiled)
    assert stripped == derivation
    model = document.getModel()
    source = _read(MODEL_PATH)
    assert model.getNumParameters() == source.getNumParameters()
    assert model.getNumRules() == source.getNumRules()
    assert (
        model.getParameter("gamma").getConstant()
        == source.getParameter("gamma").getConstant()
    )
    assert derivation_of(model) is None
    assert libsbml.writeSBMLToString(document) == libsbml.writeSBMLToString(
        libsbml.readSBMLFromFile(str(MODEL_PATH))
    )


def test_strip_of_observables_on_a_compiled_model(tmp_path: Path) -> None:
    compiled = compile_network(
        MODEL_PATH, [_rhs(feed_forward())], tmp_path / "lv_sciml.xml"
    )
    derived = add_observables(
        compiled, {"total": "prey + predator"}, tmp_path / "lv_sciml_observables.xml"
    )
    derivation = derivation_of(_read(derived))
    assert derivation is not None
    assert derivation.source == "lotka_volterra.xml"
    assert "observable_total" in derivation.created
    document, _ = strip_derivation(derived)
    assert libsbml.writeSBMLToString(document) == libsbml.writeSBMLToString(
        libsbml.readSBMLFromFile(str(MODEL_PATH))
    )


def test_strip_refuses_a_model_which_is_not_derived() -> None:
    with pytest.raises(ValueError, match="is not derived from a model"):
        strip_derivation(MODEL_PATH)


def test_strip_names_a_created_parameter_which_is_missing(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    record_derivation(document.getModel(), Path("lv.xml"), ["missing"], {})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    with pytest.raises(ValueError, match="has no parameter 'missing'"):
        strip_derivation(path)


def _compiled(tmp_path: Path) -> Path:
    return compile_network(
        MODEL_PATH, [_rhs(feed_forward())], tmp_path / "lv_sciml.xml"
    )


def _edited(tmp_path: Path, edit: Callable[[libsbml.Model], None]) -> Path:
    """Edit a compiled model by hand and write it."""
    path = _compiled(tmp_path)
    document = libsbml.readSBMLFromFile(str(path))
    edit(document.getModel())
    libsbml.writeSBMLToFile(document, str(path))
    return path


def _new_rule(model: libsbml.Model) -> None:
    parameter = model.createParameter()
    parameter.setId("x")
    parameter.setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("x")
    rule.setMath(libsbml.parseL3Formula("net1__output0__0 * 2"))


def _new_kinetic_law(model: libsbml.Model) -> None:
    law = model.getReaction("v1").getKineticLaw()
    law.setMath(libsbml.parseL3Formula("alpha * prey + net1__layer1__bias__0"))


def _new_event(model: libsbml.Model) -> None:
    event = model.createEvent()
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setInitialValue(False)
    trigger.setPersistent(True)
    trigger.setMath(libsbml.parseL3Formula("time > 5"))
    assignment = event.createEventAssignment()
    assignment.setVariable("alpha")
    assignment.setMath(libsbml.parseL3Formula("net1__output0__0"))


def _new_initial_assignment(model: libsbml.Model) -> None:
    assignment = model.createInitialAssignment()
    assignment.setSymbol("beta")
    assignment.setMath(libsbml.parseL3Formula("net1__layer1__bias__1"))


@pytest.mark.parametrize(
    ("edit", "element", "created"),
    [
        (_new_rule, "assignmentRule 'x'", "net1__output0__0"),
        (_new_kinetic_law, "kineticLaw of the reaction 'v1'", "net1__layer1__bias__0"),
        (_new_event, "eventAssignment of 'alpha'", "net1__output0__0"),
        (
            _new_initial_assignment,
            "initialAssignment of 'beta'",
            "net1__layer1__bias__1",
        ),
    ],
)
def test_strip_refuses_a_hand_edit_which_refers_to_a_created_parameter(
    tmp_path: Path, edit: Callable[[libsbml.Model], None], element: str, created: str
) -> None:
    path = _edited(tmp_path, edit)
    with pytest.raises(ValueError) as error:
        strip_derivation(path)
    message = str(error.value)
    assert path.name in message
    assert element in message
    assert f"'{created}'" in message


def _constant_rule(model: libsbml.Model) -> None:
    model.getRuleByVariable("gamma").setMath(libsbml.parseL3Formula("0.5"))


def _foreign_rule(model: libsbml.Model) -> None:
    model.getRuleByVariable("gamma").setMath(libsbml.parseL3Formula("alpha"))


def _removed_rule(model: libsbml.Model) -> None:
    model.removeRuleByVariable("gamma")


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_constant_rule, "does not refer to an output"),
        (_foreign_rule, "refers to 'alpha'"),
        (_removed_rule, "has no rule"),
    ],
)
def test_strip_refuses_a_target_whose_rule_is_not_the_one_of_the_network(
    tmp_path: Path, edit: Callable[[libsbml.Model], None], problem: str
) -> None:
    path = _edited(tmp_path, edit)
    with pytest.raises(ValueError) as error:
        strip_derivation(path)
    message = str(error.value)
    assert path.name in message
    assert "'gamma'" in message
    assert problem in message


@pytest.mark.parametrize("name", ["a&b.xml", 'q"b.xml', "a<b>'c.xml"])
def test_a_source_with_xml_characters_in_its_name(tmp_path: Path, name: str) -> None:
    source = tmp_path / name
    source.write_bytes(MODEL_PATH.read_bytes())
    compiled = compile_network(source, [_rhs(feed_forward())], tmp_path / "c.xml")
    derivation = derivation_of(_read(compiled))
    assert derivation is not None
    assert derivation.source == name
    document, derivation = strip_derivation(compiled)
    assert derivation.source == name
    assert libsbml.writeSBMLToString(document) == libsbml.writeSBMLToString(
        libsbml.readSBMLFromFile(str(source))
    )
    observables = add_observables(source, {"total": "prey"}, tmp_path / "o.xml")
    derivation = derivation_of(_read(observables))
    assert derivation is not None
    assert derivation.source == name


def test_a_derivation_is_hashable_and_immutable() -> None:
    derivation = Derivation("lv.xml", ("a",), (("gamma", True),))
    assert hash(derivation) == hash(Derivation("lv.xml", ("a",), (("gamma", True),)))
    assert hash(derivation) != hash(Derivation("lv.xml", ("a",), (("gamma", False),)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        derivation.targets = ()  # ty: ignore[invalid-assignment]


def test_a_derivation_is_a_value_which_pickles_and_copies() -> None:
    derivation = Derivation("lv.xml", ("a", "b"), (("gamma", True), ("beta", False)))
    assert pickle.loads(pickle.dumps(derivation)) == derivation
    assert copy.deepcopy(derivation) == derivation
    assert dataclasses.asdict(derivation)["targets"] == (
        ("gamma", True),
        ("beta", False),
    )


def test_strip_restores_a_target_which_was_not_constant(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    document.getModel().getParameter("gamma").setConstant(False)
    source = tmp_path / "lv_variable.xml"
    libsbml.writeSBMLToFile(document, str(source))
    compiled = compile_network(source, [_rhs(feed_forward())], tmp_path / "c.xml")
    derivation = derivation_of(_read(compiled))
    assert derivation is not None
    assert dict(derivation.targets) == {"gamma": False}
    stripped, _ = strip_derivation(compiled)
    assert stripped.getModel().getParameter("gamma").getConstant() is False
    assert libsbml.writeSBMLToString(stripped) == libsbml.writeSBMLToString(
        libsbml.readSBMLFromFile(str(source))
    )


def test_strip_names_a_target_which_is_missing(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    record_derivation(document.getModel(), Path("lv.xml"), [], {"missing": True})
    path = tmp_path / "derived.xml"
    libsbml.writeSBMLToFile(document, str(path))
    with pytest.raises(ValueError, match="has no parameter 'missing'"):
        strip_derivation(path)


def _annotated(model: libsbml.Model, record: str) -> None:
    node = libsbml.XMLNode.convertStringToXMLNode(f"<annotation>{record}</annotation>")
    assert model.setAnnotation(node) == 0


def test_a_record_without_a_source_is_refused() -> None:
    model = libsbml.readSBMLFromFile(str(MODEL_PATH)).getModel()
    _annotated(model, f'<derived xmlns="{NAMESPACE}"><created>a</created></derived>')
    with pytest.raises(ValueError, match="without the source"):
        derivation_of(model)


def test_a_record_with_a_bad_constant_is_refused() -> None:
    model = libsbml.readSBMLFromFile(str(MODEL_PATH)).getModel()
    _annotated(
        model,
        f'<derived xmlns="{NAMESPACE}" source="lv.xml">'
        '<target id="gamma" constant="yes"/></derived>',
    )
    with pytest.raises(ValueError, match="target 'gamma' and constant 'yes'"):
        derivation_of(model)
