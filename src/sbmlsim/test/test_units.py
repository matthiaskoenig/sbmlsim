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


def create_udef_examples() -> List[Tuple[libsbml.UnitDefinition, str]]:
    """Create example UnitDefinitions for testing."""
    udef0 = libsbml.UnitDefinition(3, 1)

    # s
    udef1 = libsbml.UnitDefinition(3, 1)
    u1: libsbml.Unit = udef1.createUnit()
    u1.setId("u1")
    u1.setKind(libsbml.UNIT_KIND_SECOND)
    u1.setMultiplier(1.0)
    u1.setExponent(1)
    u1.setScale(0)

    # 1/mmole
    udef2 = libsbml.UnitDefinition(3, 1)
    u2: libsbml.Unit = udef2.createUnit()
    u2.setId("u2")
    u2.setKind(libsbml.UNIT_KIND_MOLE)
    u2.setMultiplier(1.0)
    u2.setExponent(-1)
    u2.setScale(-3)

    return [
        (None, "None"),
        (udef0, ""),
        (udef1, "s"),
        (udef2, "1/((10^-3)*mole)"),
    ]


udef_examples = create_udef_examples()


@pytest.mark.parametrize("udef, s", udef_examples)
def test_udef_to_str(udef: libsbml.UnitDefinition, s: str) -> None:
    stest = libsbml.UnitDefinition_printUnits(udef)
    print(stest)
    s2 = Units.udef_to_str(udef)
    assert s2 == s


# def test_normalize_changes() -> None:
#     Units.normalize_changes()


def test_example_units():
    example_units.run_demo_example()
