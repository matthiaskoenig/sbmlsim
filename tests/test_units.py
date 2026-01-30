"""Test units."""

from pathlib import Path
from typing import Tuple

import libsbml
import pytest

from sbmlsim.examples import example_units
from sbmlsim.resources import DEMO_SBML, MIDAZOLAM_SBML, REPRESSILATOR_SBML
from sbmlsim.units import UnitRegistry, Units, UnitsInformation


sbml_paths: list[Path] = [
    DEMO_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
]


@pytest.mark.parametrize("sbml_path", sbml_paths)
def test_units_from_sbml(sbml_path: Path) -> None:
    """Test reading units from SBML models."""
    uinfo = UnitsInformation.from_sbml(sbml_path)
    check_uinfo(uinfo)


@pytest.mark.parametrize("sbml_path", sbml_paths)
def test_units_from_doc(sbml_path: Path) -> None:
    """Test reading units from SBML models."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    uinfo = UnitsInformation.from_sbml_doc(doc)
    check_uinfo(uinfo)


def test_default_ureg() -> None:
    """Test creation of default unit registry."""
    ureg = UnitsInformation._default_ureg()
    assert ureg
    assert isinstance(ureg, UnitRegistry)


def check_uinfo(uinfo: UnitsInformation) -> None:
    """Check UnitsInformation."""
    assert uinfo
    assert isinstance(uinfo, UnitsInformation)
    assert uinfo.udict
    assert isinstance(uinfo.udict, dict)
    assert uinfo.ureg
    assert isinstance(uinfo.ureg, UnitRegistry)


def test_example_units() -> None:
    """Run demo examples."""
    example_units.run_demo_example()


def create_udef_examples() -> list[Tuple[libsbml.UnitDefinition, str]]:
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
        (udef2, "1/(10^-3*mole)"),
    ]


udef_examples = create_udef_examples()


@pytest.mark.parametrize("udef, s", udef_examples)
def test_udef_to_str(udef: libsbml.UnitDefinition, s: str) -> None:
    """Test UnitDefinition to string."""
    _ = libsbml.UnitDefinition.printUnits(udef)
    s2 = Units.udef_to_str(udef)
    assert s2 == s
