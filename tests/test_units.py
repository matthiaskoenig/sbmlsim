"""Test units."""

import logging
from pathlib import Path

import libsbml
import pytest

from examples import units as example_units
from sbmlsim.resources import DEMO_SBML, MIDAZOLAM_SBML, REPRESSILATOR_SBML
from sbmlsim.units import UnitRegistry, Units, UnitsInformation

LOTKA_VOLTERRA_SBML = Path(__file__).parent / "data" / "models" / "lotka_volterra.xml"

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


def test_a_document_without_a_model() -> None:
    """A document without a model has no units to read."""
    doc = libsbml.SBMLDocument(3, 2)
    with pytest.raises(ValueError, match="No model found in SBMLDocument"):
        UnitsInformation.from_sbml_doc(doc)


def test_package_registry() -> None:
    """The registry of the package knows the units sbmlsim defines."""
    from sbmlsim.units import ureg

    assert isinstance(ureg, UnitRegistry)
    assert ureg("percent").to("dimensionless").magnitude == 0.01


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


def create_udef_examples() -> list[tuple[libsbml.UnitDefinition | None, str]]:
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
        (udef2, "1/mmol"),
    ]


udef_examples = create_udef_examples()


@pytest.mark.parametrize("udef, s", udef_examples)
def test_udef_to_str(udef: libsbml.UnitDefinition, s: str) -> None:
    """Test UnitDefinition to string."""
    _ = libsbml.UnitDefinition.printUnits(udef)
    s2 = Units.udef_to_str(udef)
    assert s2 == s


def test_the_missing_units_of_a_model_are_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model without units is one line at INFO and one per entity at DEBUG."""
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.units"):
        uinfo = UnitsInformation.from_sbml(LOTKA_VOLTERRA_SBML)
    records = [r for r in caplog.records if r.name == "sbmlsim.units"]

    assert [r.levelno for r in records if r.levelno > logging.DEBUG] == [logging.INFO]
    info = next(r.getMessage() for r in records if r.levelno == logging.INFO)
    assert "'lv'" in info
    assert "7 entities" in info
    debug = "\n".join(r.getMessage() for r in records if r.levelno == logging.DEBUG)
    for entity in ["time", "[prey]", "[predator]", "v1", "v2", "v3", "v4"]:
        assert f"'{entity}'" in debug
    # the units are the ones read before
    assert uinfo["time"] == "second"
    assert uinfo["[prey]"] == ""


def _model_with_unit(
    uid: str, kind: int, scale: int = 0, multiplier: float = 1.0
) -> str:
    """Get the SBML of a model with one parameter in the unit `uid`."""
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(f"m_{uid}_{kind}")
    udef = model.createUnitDefinition()
    udef.setId(uid)
    unit = udef.createUnit()
    unit.setKind(kind)
    unit.setExponent(1)
    unit.setScale(scale)
    unit.setMultiplier(multiplier)
    p = model.createParameter()
    p.setId("p")
    p.setValue(1.0)
    p.setConstant(True)
    p.setUnits(uid)
    return libsbml.writeSBMLToString(doc)


def test_q_is_the_quantity_of_the_package_registry() -> None:
    """`sbmlsim.Q` creates quantities of `sbmlsim.units.ureg`."""
    from sbmlsim import Q
    from sbmlsim.units import ureg

    assert Q(1, "mg")._REGISTRY is ureg


def test_one_unit_id_in_two_models_converts_per_model() -> None:
    """Two models which define one unit id differently keep their own units."""
    from sbmlsim import Q

    u_gram = UnitsInformation.from_sbml(
        _model_with_unit("u1", libsbml.UNIT_KIND_GRAM, scale=-3)
    )
    u_mole = UnitsInformation.from_sbml(
        _model_with_unit("u1", libsbml.UNIT_KIND_MOLE, scale=-3)
    )
    assert Q(1, "g").to(u_gram["p"]).magnitude == pytest.approx(1000.0)
    assert Q(1, "mol").to(u_mole["p"]).magnitude == pytest.approx(1000.0)


def test_the_unit_ids_of_a_model_are_not_defined_in_the_registry() -> None:
    """The units of a model are expressions, the registry knows no model ids."""
    from pint.errors import UndefinedUnitError

    from sbmlsim.units import ureg

    UnitsInformation.from_sbml(
        _model_with_unit("u_not_in_registry", libsbml.UNIT_KIND_GRAM)
    )
    with pytest.raises(UndefinedUnitError):
        ureg("u_not_in_registry")


def _udef(*units: tuple[int, float, int, int]) -> libsbml.UnitDefinition:
    """Create a unit definition of `(kind, multiplier, scale, exponent)` units."""
    udef = libsbml.UnitDefinition(3, 1)
    for kind, multiplier, scale, exponent in units:
        u: libsbml.Unit = udef.createUnit()
        u.setKind(kind)
        u.setMultiplier(multiplier)
        u.setScale(scale)
        u.setExponent(exponent)
    return udef


@pytest.mark.parametrize(
    "udef, s",
    [
        (
            _udef(
                (libsbml.UNIT_KIND_MOLE, 1.0, -3, 1),
                (libsbml.UNIT_KIND_SECOND, 60.0, 0, -1),
            ),
            "mmol/min",
        ),
        (
            _udef(
                (libsbml.UNIT_KIND_MOLE, 1.0, -3, 1),
                (libsbml.UNIT_KIND_LITRE, 1.0, 0, -1),
                (libsbml.UNIT_KIND_SECOND, 3600.0, 0, -1),
            ),
            "mmol/(l * hr)",
        ),
        (_udef((libsbml.UNIT_KIND_METRE, 1.0, -3, 2)), "mm^2"),
        (_udef((libsbml.UNIT_KIND_GRAM, 1.0, 3, 1)), "kg"),
        (_udef((libsbml.UNIT_KIND_LITRE, 1.0, -6, 1)), "ul"),
        (_udef((libsbml.UNIT_KIND_MOLE, 2.0, -3, 2)), "(2.0*mmol)^2"),
    ],
)
def test_udef_to_str_is_readable(udef: libsbml.UnitDefinition, s: str) -> None:
    """A unit definition is written with the prefixes and names of pint."""
    assert Units.udef_to_str(udef) == s


@pytest.mark.parametrize(
    "units",
    [
        ((libsbml.UNIT_KIND_MOLE, 1.0, -3, 2),),
        ((libsbml.UNIT_KIND_METRE, 1.0, -2, -3),),
        ((libsbml.UNIT_KIND_GRAM, 1.0, -5, 2), (libsbml.UNIT_KIND_SECOND, 7.0, 1, -2)),
        ((libsbml.UNIT_KIND_ITEM, 1.0, 0, 1),),
    ],
)
def test_udef_to_str_parses(units: tuple[tuple[int, float, int, int], ...]) -> None:
    """Every unit definition is a unit pint parses, also with a scale and an exponent."""
    from sbmlsim.units import ureg

    ureg(Units.udef_to_str(_udef(*units)))


def test_udef_to_str_folds_a_decimal_multiplier_into_the_prefix() -> None:
    """A multiplier of 0.001 with the scale 0 is the prefix milli."""
    assert Units.udef_to_str(_udef((libsbml.UNIT_KIND_MOLE, 0.001, 0, 1))) == "mmol"
    assert Units.udef_to_str(_udef((libsbml.UNIT_KIND_GRAM, 1000.0, 0, 1))) == "kg"


def test_every_unit_of_a_model_is_a_unit_without_a_factor() -> None:
    """Every unit of a model parses as a unit, which a quantity needs."""
    from sbmlsim.units import ureg

    hctz = Path(__file__).parents[1] / "examples" / "hctz_fitting" / "models"
    uinfo = UnitsInformation.from_sbml(hctz / "hctz_body_flat.xml")
    for unit in set(uinfo.udict.values()):
        ureg.parse_units(unit)


def test_a_unit_with_a_factor_of_two_models_is_defined_per_model() -> None:
    """A unit pint cannot write without a factor is defined once per definition."""
    from sbmlsim import Q

    u_a = UnitsInformation.from_sbml(
        _model_with_unit("u_factor", libsbml.UNIT_KIND_GRAM, scale=0, multiplier=7.0)
    )
    u_b = UnitsInformation.from_sbml(
        _model_with_unit("u_factor", libsbml.UNIT_KIND_GRAM, scale=0, multiplier=3.0)
    )
    assert Q(21, "g").to(u_a["p"]).magnitude == pytest.approx(3.0)
    assert Q(21, "g").to(u_b["p"]).magnitude == pytest.approx(7.0)
