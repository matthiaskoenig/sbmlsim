"""Tests of the record of a derivation of a model."""

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
    assert derivation == Derivation("lv.xml", ("a", "b"), {"gamma": True})


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
    assert derivation_of(model) == Derivation("lv.xml", ("a",), {})
    assert NAMESPACE in model.getAnnotationString()


def test_a_second_derivation_extends_the_record(tmp_path: Path) -> None:
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    model = document.getModel()
    record_derivation(model, Path("lv.xml"), ["a"], {"gamma": True})
    record_derivation(model, Path("lv_sciml.xml"), ["b"], {"a": False, "beta": True})
    derivation = derivation_of(model)
    # the source stays, a target which the first derivation created is no target
    assert derivation == Derivation("lv.xml", ("a", "b"), {"gamma": True, "beta": True})
    annotation = model.getAnnotation()
    assert annotation.getNumChildren() == 1


def test_strip_gives_the_source_model_back(tmp_path: Path) -> None:
    network = feed_forward()
    compiled = compile_network(MODEL_PATH, [_rhs(network)], tmp_path / "lv_sciml.xml")
    derivation = derivation_of(_read(compiled))
    assert derivation is not None
    assert derivation.source == "lotka_volterra.xml"
    assert set(network.parameter_ids()) <= set(derivation.created)
    assert derivation.targets == {"gamma": True}

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
    assert (
        document.getModel().getNumParameters() == _read(MODEL_PATH).getNumParameters()
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
