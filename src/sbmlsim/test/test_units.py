"""
Test units.
"""
from pathlib import Path
from typing import Tuple, List

import libsbml
import pytest

from sbmlsim.examples import example_units
from sbmlsim.test import MODEL_DEMO, MODEL_REPRESSILATOR
from sbmlsim.units import Units, UnitRegistry

sbml_paths = [MODEL_DEMO, MODEL_REPRESSILATOR]


def test_default_ureg() -> None:
    """Test creation of default unit registry."""
    ureg = Units.default_ureg()
    assert ureg
    assert isinstance(ureg, UnitRegistry)


@pytest.mark.parametrize("sbml_path", sbml_paths)
def test_units_from_sbml(sbml_path: Path) -> None:
    """Test reading units from SBML models."""
    udict, ureg = Units.units_from_sbml(sbml_path)
    assert udict
    assert isinstance(udict, dict)
    assert ureg
    assert isinstance(ureg, UnitRegistry)


@pytest.mark.parametrize("sbml_path", sbml_paths)
def test_ureg_from_doc(sbml_path: Path) -> None:
    """Test creation of unit registry from given model."""
    doc = libsbml.readSBMLFromFile(str(sbml_path))
    ureg = Units._ureg_from_doc(doc)
    assert ureg
    assert isinstance(ureg, UnitRegistry)


# TODO: specific test cases

def create_udef_examples() -> List[Tuple[libsbml.UnitDefinition, str]]:
    """Create example UnitDefinitions for testing."""
    # s
    u1 = libsbml.Unit(3, 1)
    u1.setId("u1")
    u1.setKind(libsbml.UNIT_KIND_SECOND)

    # 1/mmole
    u2 = libsbml.Unit(3, 1)
    u2.setId("u2")
    u2.setKind(libsbml.UNIT_KIND_MOLE)
    u2.setExponent(-3)

    udef1 = libsbml.UnitDefinition(3, 1)
    udef2 = libsbml.UnitDefinition(3, 1)
    udef2.addUnit(u1)
    udef3 = libsbml.UnitDefinition(3, 1)
    udef3.addUnit(u2)
    udef4 = libsbml.UnitDefinition(3, 1)
    udef4.addUnit(u1)
    udef4.addUnit(u2)
    return [
        (None, "None"),
        (udef1, ""),
        (udef2, "s"),
        (udef3, "1/mmole"),
        (udef4, "s/mmole"),
    ]


udef_examples = create_udef_examples()


@pytest.mark.parametrize("udef, s", udef_examples)
def test_udef_to_str(udef: libsbml.UnitDefinition, s: str) -> None:
    s2 = Units.udef_to_str(udef)
    assert s2 == s


# def test_normalize_changes() -> None:
#     Units.normalize_changes()


def test_example_units():
    example_units.run_demo_example()
