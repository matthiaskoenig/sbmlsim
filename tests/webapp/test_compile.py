"""The SBML of a model prepared and compiled with sbmlsim-core."""

import json

import libsbml
import pytest

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.webapp.compile import (
    WebAppError,
    compile_sbml,
    prepare_sbml,
    write_compiled,
)

pytest.importorskip("sbmlsim_core")

DISTRIB = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
  xmlns:distrib="http://www.sbml.org/sbml/level3/version1/distrib/version1"
  distrib:required="true">
  <model id="decay">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="10"
        hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true">
        <distrib:listOfUncertainties>
          <distrib:uncertainty>
            <distrib:listOfUncertParameters>
              <distrib:uncertParameter distrib:type="standardDeviation" distrib:value="0.1"/>
            </distrib:listOfUncertParameters>
          </distrib:uncertainty>
        </distrib:listOfUncertainties>
      </parameter>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><apply><minus/><ci>k</ci></apply><ci>S</ci></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>
"""

#: a math which draws from a distribution of distrib
DRAWN = DISTRIB.replace(
    "<apply><times/><apply><minus/><ci>k</ci></apply><ci>S</ci></apply>",
    '<apply><csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/distrib/normal">normal'
    "</csymbol><cn>0</cn><cn>1</cn></apply>",
)

COMP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
  xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
  comp:required="true">
  <model id="top">
    <comp:listOfSubmodels>
      <comp:submodel comp:id="sub" comp:modelRef="inner"/>
    </comp:listOfSubmodels>
  </model>
  <comp:listOfModelDefinitions>
    <comp:modelDefinition id="inner">
      <listOfParameters>
        <parameter id="p" value="2" constant="true"/>
      </listOfParameters>
    </comp:modelDefinition>
  </comp:listOfModelDefinitions>
</sbml>
"""


def _document(sbml: str) -> libsbml.SBMLDocument:
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    assert doc.getModel() is not None
    return doc


def test_the_uncertainties_of_distrib_are_stripped() -> None:
    prepared = prepare_sbml(DISTRIB)
    assert "distrib" not in prepared
    doc = _document(prepared)
    assert doc.getModel().getParameter("k").getValue() == 0.5


def test_a_hierarchical_model_is_flattened() -> None:
    doc = _document(prepare_sbml(COMP))
    assert not doc.isPackageEnabled("comp")
    assert doc.getModel().getNumParameters() == 1


def test_a_model_below_level_3_is_converted_to_level_3_version_2() -> None:
    doc = _document(prepare_sbml(DISTRIB))
    assert doc.setLevelAndVersion(2, 4, False)
    prepared = _document(prepare_sbml(libsbml.writeSBMLToString(doc)))
    assert (prepared.getLevel(), prepared.getVersion()) == (3, 2)


def test_a_distribution_function_in_a_math_is_refused_naming_the_element() -> None:
    with pytest.raises(WebAppError, match=r"rateRule of 'S' \(normal\)"):
        prepare_sbml(DRAWN)


def test_a_model_compiles_to_wasm_and_its_description(tmp_path) -> None:
    compiled = compile_sbml(REPRESSILATOR_SBML.read_text(encoding="utf-8"))
    assert compiled.wasm[:4] == b"\0asm"
    description = json.loads(compiled.description)
    assert description["contract"] == "model/2"
    assert compiled == compile_sbml(REPRESSILATOR_SBML.read_text(encoding="utf-8"))
    write_compiled(compiled, tmp_path / "model")
    assert (tmp_path / "model" / "model.wasm").read_bytes() == compiled.wasm
    assert (tmp_path / "model" / "model.json").read_text(
        encoding="utf-8"
    ) == compiled.description


def test_the_uncertainties_change_nothing_of_the_compiled_model() -> None:
    plain = DISTRIB.split("<distrib:listOfUncertainties>")[0] + (
        "</parameter>"
        + DISTRIB.split("</distrib:listOfUncertainties>")[1].split("</parameter>", 1)[1]
    )
    assert "uncertParameter" not in plain
    assert compile_sbml(DISTRIB).wasm == compile_sbml(plain).wasm


def test_a_model_the_compiler_refuses_raises_with_its_message() -> None:
    cyclic = DISTRIB.replace(
        '<rateRule variable="S">', '<assignmentRule variable="S">'
    ).replace("</rateRule>", "</assignmentRule>")
    with pytest.raises(WebAppError, match="sbmlsim-core cannot compile"):
        compile_sbml(cyclic)
