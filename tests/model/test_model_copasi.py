"""The model of an SBML model in COPASI."""

import contextlib
import gc
import logging
import math
from collections.abc import Callable
from pathlib import Path

import libsbml
import pytest

from sbmlsim.engine import import_copasi

with contextlib.suppress(ImportError):
    # COPASI imported with the locale of the process kept, see `import_copasi`
    import_copasi()
COPASI = pytest.importorskip("COPASI")

from sbmlsim.engine import Engine  # noqa: E402
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel, load_model  # noqa: E402
from sbmlsim.model.model_copasi import CopasiSBMLModel  # noqa: E402
from sbmlsim.model.symbols import TargetKind  # noqa: E402
from sbmlsim.model.tolerances import AbsoluteTolerance  # noqa: E402
from sbmlsim.resources import (  # noqa: E402
    DEMO_SBML,
    HCTZ_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
)
from sbmlsim.simulator.plan import Assignment  # noqa: E402
from tests.simulator.models import (  # noqa: E402
    PROBE,
    SPECIES_KINDS,
    VOLUME,
    VOLUME_OF_A_RULE,
    sbml,
)

#: `X` follows `x0` by an initial assignment, `y` follows `X`
OVERRIDE = """
model override
  compartment c = 1
  species X in c
  x0 = 1
  X = 2*x0
  y = X
  k = 0.1
  J: X -> ; k*X
end
"""

#: a species whose substance unit is not the substance unit of the model
OTHER_UNITS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="other_units" substanceUnits="mole" volumeUnits="litre"
      timeUnits="second" extentUnits="mole">
    <listOfUnitDefinitions>
      <unitDefinition id="mmol">
        <listOfUnits>
          <unit kind="mole" exponent="1" scale="-3" multiplier="1"/>
        </listOfUnits>
      </unitDefinition>
    </listOfUnitDefinitions>
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="AMOUNT" substanceUnits="mmol"
        hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="5"
        hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
  </model>
</sbml>
"""


def _a(
    target: str, value: float, kind: TargetKind = TargetKind.PARAMETER
) -> Assignment:
    return Assignment(target=target, kind=kind, value=value)


def _values(model: CopasiSBMLModel) -> dict[str, float]:
    """Get the value of every selection of a model which has one."""
    assert model.selections is not None
    return {s: model.value(s) for s in model.selections}


def _same(a: dict[str, float], b: dict[str, float]) -> bool:
    """Whether two values of the selections are equal bitwise, `NaN` included."""
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


@pytest.mark.parametrize(
    "source", [REPRESSILATOR_SBML, DEMO_SBML, MIDAZOLAM_SBML, HCTZ_SBML]
)
def test_loads_with_the_selections_of_roadrunner(source: Path) -> None:
    model = CopasiSBMLModel(source=source)
    assert model.engine is Engine.COPASI
    assert model.selections is not None
    assert set(model.selections) == set(
        RoadrunnerSBMLModel(source=source).selections or ()
    )
    assert not model.selections_given


@pytest.mark.parametrize(
    "source", [REPRESSILATOR_SBML, DEMO_SBML, MIDAZOLAM_SBML, HCTZ_SBML]
)
def test_values_are_the_ones_of_roadrunner_as_loaded(source: Path) -> None:
    c, r = CopasiSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    assert r.selections is not None
    for s in r.selections:
        assert c.value(s) == pytest.approx(r.value(s), rel=1e-12, abs=1e-300), s
        assert c.has_selection(s)


def test_preinit_overrides_an_initial_assignment_and_updates_dependents() -> None:
    """COPASI holds amounts as particle numbers, a concentration is exact in 1e-15."""
    model = CopasiSBMLModel(source=sbml(OVERRIDE))
    loaded = _values(model)

    def assert_x(x: float) -> None:
        assert model.value("[X]") == pytest.approx(x, rel=1e-15)
        assert model.value("y") == x

    assert_x(2.0)
    model.initialize([_a("x0", 3.0)])
    assert_x(6.0)
    model.simulated()
    model.initialize([])
    assert_x(2.0)
    assert _same(_values(model), loaded)
    # the target replaces its own initial assignment, its dependents follow
    model.initialize([_a("[X]", 10.0, TargetKind.SPECIES_CONCENTRATION)])
    assert_x(10.0)
    model.initialize([_a("[X]", 7.0, TargetKind.SPECIES_CONCENTRATION), _a("x0", 5)])
    assert_x(7.0)
    model.initialize([_a("x0", 4.0)])
    assert_x(8.0)
    model.initialize([])
    assert _same(_values(model), loaded)
    assert _same(_values(model), _values(CopasiSBMLModel(source=sbml(OVERRIDE))))


#: models and pre-initializations whose values are compared with roadrunner
PREINIT_CASES = [
    (PROBE, [_a("f", 5.0), _a("b0", 0.0)]),
    (PROBE, [_a("pinit", 7.0)]),
    (PROBE, [_a("[B]", 5.0, TargetKind.SPECIES_CONCENTRATION), _a("b0", 0.0)]),
    (PROBE, [_a("C", 4.0, TargetKind.COMPARTMENT)]),
    (
        PROBE,
        [
            _a("C", 4.0, TargetKind.COMPARTMENT),
            _a("[A]", 3.0, TargetKind.SPECIES_CONCENTRATION),
        ],
    ),
    (
        PROBE,
        [_a("C", 4.0, TargetKind.COMPARTMENT), _a("A", 3.0, TargetKind.SPECIES_AMOUNT)],
    ),
    (PROBE, [_a("X", 10.0, TargetKind.SPECIES_AMOUNT)]),
    (SPECIES_KINDS, [_a("BW", 80.0)]),
    (SPECIES_KINDS, [_a("V", 10.0, TargetKind.COMPARTMENT)]),
    (SPECIES_KINDS, [_a("BW", 80.0), _a("[F]", 3.0, TargetKind.SPECIES_CONCENTRATION)]),
    (SPECIES_KINDS, [_a("N", 1.0, TargetKind.SPECIES_AMOUNT), _a("k", 2.0)]),
    (VOLUME_OF_A_RULE, [_a("BW", 70.45995681845807)]),
    (VOLUME, [_a("f", 3.0)]),
    (OVERRIDE, [_a("[X]", 10.0, TargetKind.SPECIES_CONCENTRATION)]),
    (OVERRIDE, [_a("y", 4.0), _a("x0", 2.0)]),
    (
        OVERRIDE,
        [_a("c", 3.0, TargetKind.COMPARTMENT), _a("X", 3.0, TargetKind.SPECIES_AMOUNT)],
    ),
]


@pytest.mark.parametrize(("model", "assignments"), PREINIT_CASES)
def test_preinit_values_are_the_ones_of_roadrunner(
    model: str, assignments: list[Assignment]
) -> None:
    """A pre-initialization gives the values of roadrunner, a second one restores."""
    source = sbml(model)
    c, r = CopasiSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    loaded = _values(c)
    for _ in range(2):
        c.initialize(assignments)
        r.initialize(assignments)
        assert r.selections is not None
        for s in r.selections:
            assert c.value(s) == pytest.approx(r.value(s), rel=1e-12), s
        c.simulated()
        r.simulated()
    c.initialize([])
    assert _same(_values(c), loaded)


def test_preinit_formulas_read_the_parameters() -> None:
    model = CopasiSBMLModel(source=sbml(PROBE))
    model.initialize(
        [_a("f", 3.0), Assignment("k1", TargetKind.PARAMETER, formula="2 * f + a0")]
    )
    assert model.value("k1") == 7.0
    assert model.value("kk") == 9.0


def test_a_reset_leaves_the_model_as_loaded() -> None:
    model = CopasiSBMLModel(source=sbml(SPECIES_KINDS))
    loaded = _values(model)
    model.initialize([_a("BW", 80.0), _a("[F]", 3.0, TargetKind.SPECIES_CONCENTRATION)])
    model.values.setValue("k", 4.0)
    model.reset()
    assert _same(_values(model), loaded)


def test_the_values_as_loaded() -> None:
    model = CopasiSBMLModel(source=sbml(PROBE))
    model.initialize([_a("f", 5.0), _a("C", 4.0, TargetKind.COMPARTMENT)])
    assert model.loaded_value("f") == 2.0
    assert model.loaded_value("C") == 2.0
    assert model.loaded_value("[A]") == 1.0
    assert model.loaded_value("A") == pytest.approx(2.0, rel=1e-15)
    assert model.loaded_value("X") == pytest.approx(12.0, rel=1e-15)
    for selection in ["J1", "time", "[nope]", "nope"]:
        with pytest.raises(KeyError):
            model.loaded_value(selection)


def test_the_current_values_are_set_through_the_state() -> None:
    """A constant parameter, a compartment and a species are set between segments."""
    model = CopasiSBMLModel(source=sbml(PROBE))
    values = model.values
    assert values.getValue("time") == 0.0
    values.setValue("k1", 2.0)
    assert values.getValue("k1") == 2.0
    assert values.getValue("J1") == pytest.approx(2.0)  # k1 * [A]
    values.setValue("A", 4.0)
    assert values.getValue("A") == pytest.approx(4.0, rel=1e-15)
    assert values.getValue("[A]") == pytest.approx(2.0, rel=1e-15)
    values.setValue("[B]", 3.0)
    assert values.getValue("B") == pytest.approx(6.0, rel=1e-15)
    values.setValue("C", 4.0)
    assert values.getValue("A") == pytest.approx(4.0, rel=1e-15)
    assert values.getValue("[A]") == pytest.approx(1.0, rel=1e-15)
    values.setValue("f", 5.0)
    assert values.getValue("kk") == 15.0  # an assignment rule follows
    with pytest.raises(RuntimeError, match="assignment rule"):
        values.setValue("kk", 1.0)
    with pytest.raises(RuntimeError, match="no selection 'nope'"):
        values.getValue("nope")
    with pytest.raises(RuntimeError, match="cannot set 'J1'"):
        values.setValue("J1", 1.0)
    # the initialization restores the values as loaded
    model.initialize([])
    assert model.value("k1") == 0.8 and model.value("C") == 2.0
    assert not model.has_selection("[k1]")


def test_a_state_of_a_rate_rule_is_set() -> None:
    model = CopasiSBMLModel(source=sbml(SPECIES_KINDS))
    model.values.setValue("[RR]", 3.0)
    assert model.value("[RR]") == pytest.approx(3.0, rel=1e-15)
    assert model.value("RR") == pytest.approx(3.0 * 15.0, rel=1e-15)


def test_a_value_which_is_not_a_number_is_no_state() -> None:
    """An entity whose value is `NaN` is no state of COPASI, a change cannot set it.

    COPASI integrates no state which is not finite, as the compartments of
    the HCTZ model whose size is `NaN`; the value is set before the
    initialization.
    """
    source = (
        sbml(
            "model undefined_values\n  compartment V = 1\n  p = 1\n  X = 0\n  X' = 1\nend"
        )
        .replace('id="p" value="1"', 'id="p" value="NaN"')
        .replace(
            'id="V" spatialDimensions="3" size="1"',
            'id="V" spatialDimensions="3" size="NaN"',
        )
    )
    model = CopasiSBMLModel(source=source)
    assert math.isnan(model.value("p")) and math.isnan(model.value("V"))
    with pytest.raises(RuntimeError, match=r"'p'.*not a number"):
        model.values.setValue("p", 1.0)
    with pytest.raises(RuntimeError, match=r"'V'.*not a number"):
        model.values.setValue("V", 1.0)
    model.initialize([_a("p", 3.0)])
    assert model.value("p") == 3.0
    model.values.setValue("X", 2.0)
    assert model.value("X") == 2.0


def test_the_output_names_and_their_transform() -> None:
    model = CopasiSBMLModel(source=sbml(PROBE))
    selections = ["time", "A", "[A]", "k1", "C", "J1"]
    cns = model.output_cns(selections)
    assert len(cns) == len(selections)
    assert all(cn.startswith("CN=Root,Model=") for cn in cns)
    row = [0.0, 2.0 * model.mc.getQuantity2NumberFactor(), 1.0, 0.8, 2.0, 0.8]
    assert model.output_transform(selections)(row) == pytest.approx(
        [0.0, 2.0, 1.0, 0.8, 2.0, 0.8], rel=1e-15
    )


def test_species_in_units_other_than_the_model_keep_their_numbers() -> None:
    """COPASI reads a species in other units with the number of the SBML, as roadrunner."""
    source = OTHER_UNITS.replace("AMOUNT", "3")
    c, r = CopasiSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    for s in ["A", "[A]", "B", "[B]", "c"]:
        assert c.value(s) == pytest.approx(r.value(s), rel=1e-15)


def test_a_number_copasi_reads_otherwise_raises() -> None:
    """COPASI keeps amounts as particle numbers, `1e300` mole are infinite."""
    with pytest.raises(
        ValueError,
        match=r"COPASI reads 'A' of the model 'other_units' otherwise.*out of the range",
    ):
        CopasiSBMLModel(source=OTHER_UNITS.replace("AMOUNT", "1e300"))


def test_an_import_error_of_copasi_raises() -> None:
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    model: libsbml.Model = doc.getModel()
    model.createParameter().setId("z")
    model.getParameter("z").setConstant(False)
    rule: libsbml.AssignmentRule = model.createAssignmentRule()
    rule.setVariable("z")
    rule.setMath(libsbml.parseL3Formula("undefined_function(A)"))
    with pytest.raises(ValueError, match="undefined_function"):
        CopasiSBMLModel(source=libsbml.writeSBMLToString(doc))


def test_algebraic_rules_raise() -> None:
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    model: libsbml.Model = doc.getModel()
    model.getParameter("a0").setConstant(False)
    rule: libsbml.AlgebraicRule = model.createAlgebraicRule()
    rule.setMath(libsbml.parseL3Formula("a0 - 2"))
    with pytest.raises(ValueError, match="algebraic rule"):
        CopasiSBMLModel(source=libsbml.writeSBMLToString(doc))


def _with_a_constant_species_reference_id() -> str:
    """Get the probe model whose product `B`, a boundary species, has the id `B_sr`."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    model: libsbml.Model = doc.getModel()
    model.getSpecies("B").setBoundaryCondition(True)
    model.getReaction("J1").getProduct(0).setId("B_sr")
    return libsbml.writeSBMLToString(doc)


def test_a_species_reference_has_no_selection() -> None:
    """COPASI has no value of a species reference, the default selections raise."""
    source = _with_a_constant_species_reference_id()
    assert "B_sr" in (RoadrunnerSBMLModel(source=source).selections or ())
    with pytest.raises(ValueError, match="no selection 'B_sr'"):
        CopasiSBMLModel(source=source)
    model = CopasiSBMLModel(source=source, selections=["A"])
    assert not model.has_selection("B_sr")


def _probe_with(edit: Callable[[libsbml.Model], None]) -> str:
    """Get the probe model with an edit of its model."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    edit(doc.getModel())
    return libsbml.writeSBMLToString(doc)


def _rule_of_the_product(rule: str) -> Callable[[libsbml.Model], None]:
    """Get the edit which sets the stoichiometry of the product of J1 by a rule."""

    def edit(model: libsbml.Model) -> None:
        product: libsbml.SpeciesReference = model.getReaction("J1").getProduct(0)
        product.setId("B_sr")
        product.setConstant(rule == "initial")
        if rule == "assignment":
            assignment: libsbml.AssignmentRule = model.createAssignmentRule()
            assignment.setVariable("B_sr")
            assignment.setMath(libsbml.parseL3Formula("1 + time"))
        elif rule == "rate":
            rate: libsbml.RateRule = model.createRateRule()
            rate.setVariable("B_sr")
            rate.setMath(libsbml.parseL3Formula("1"))
        elif rule == "initial":
            initial: libsbml.InitialAssignment = model.createInitialAssignment()
            initial.setSymbol("B_sr")
            initial.setMath(libsbml.parseL3Formula("2 * f"))
        elif rule == "event":
            event: libsbml.Event = model.createEvent()
            event.setId("E")
            event.setUseValuesFromTriggerTime(True)
            trigger: libsbml.Trigger = event.createTrigger()
            trigger.setMath(libsbml.parseL3Formula("time > 1"))
            trigger.setInitialValue(False)
            trigger.setPersistent(True)
            assignment_e: libsbml.EventAssignment = event.createEventAssignment()
            assignment_e.setVariable("B_sr")
            assignment_e.setMath(libsbml.parseL3Formula("2"))

    return edit


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        ("non-constant", "non-constant species reference 'B_sr'"),
        ("assignment", "non-constant species reference 'B_sr'"),
        ("rate", "non-constant species reference 'B_sr'"),
        ("event", "non-constant species reference 'B_sr'"),
        ("initial", "initial assignment of the species reference 'B_sr'"),
    ],
)
def test_a_variable_stoichiometry_raises(rule: str, match: str) -> None:
    """COPASI makes the stoichiometry of a species reference a constant."""
    with pytest.raises(ValueError, match=match):
        CopasiSBMLModel(source=_probe_with(_rule_of_the_product(rule)))


def test_a_stoichiometry_math_raises() -> None:
    """COPASI makes the stoichiometry math of SBML level 2 a constant."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    doc.setLevelAndVersion(2, 4)
    product: libsbml.SpeciesReference = doc.getModel().getReaction("J1").getProduct(0)
    math: libsbml.StoichiometryMath = product.createStoichiometryMath()
    math.setMath(libsbml.parseL3Formula("2 * f"))
    with pytest.raises(ValueError, match="stoichiometry math of the species 'B'"):
        CopasiSBMLModel(source=libsbml.writeSBMLToString(doc))


def test_a_delay_raises() -> None:
    """COPASI does not simulate the delay csymbol as roadrunner."""

    def edit(model: libsbml.Model) -> None:
        model.getParameter("b0").setConstant(False)
        rule: libsbml.AssignmentRule = model.createAssignmentRule()
        rule.setVariable("b0")
        rule.setMath(libsbml.parseL3Formula("delay(A, 1)"))

    with pytest.raises(ValueError, match=r"delay.*assignment rule of 'b0'"):
        CopasiSBMLModel(source=_probe_with(edit))


@pytest.mark.parametrize("on", ["model", "species"])
def test_a_conversion_factor_raises(on: str) -> None:
    """A time course of COPASI ignores the conversion factors."""

    def edit(model: libsbml.Model) -> None:
        if on == "model":
            model.setConversionFactor("k1")
        else:
            model.getSpecies("A").setConversionFactor("k1")

    element = "model" if on == "model" else "species 'A'"
    with pytest.raises(ValueError, match=f"conversion factor 'k1' of the {element}"):
        CopasiSBMLModel(source=_probe_with(edit))


def test_a_libsbml_error_raises() -> None:
    """COPASI reports the errors of libsbml as messages of another type."""

    def edit(model: libsbml.Model) -> None:
        # a species reference of a species which does not exist
        model.getReaction("J1").getProduct(0).setSpecies("missing")

    with pytest.raises(ValueError, match="LIBSBML ERROR"):
        CopasiSBMLModel(source=_probe_with(edit))


def test_a_warning_of_copasi_which_is_not_known_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning of COPASI which may change the semantics is logged as a warning."""

    def edit(model: libsbml.Model) -> None:
        # COPASI sets a species without an initial value to 1
        model.getSpecies("A").unsetInitialAmount()
        model.getSpecies("A").unsetInitialConcentration()
        model.removeInitialAssignment("A")

    with caplog.at_level(logging.DEBUG, logger="sbmlsim.model.model_copasi"):
        CopasiSBMLModel(source=_probe_with(edit))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("SBML (41)" in w for w in warnings), warnings
    assert not any("SBML (91)" in w for w in warnings), warnings


def test_roadrunner_only_settings_warn_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sbmlsim.model.model_copasi"):
        model = CopasiSBMLModel(
            source=DEMO_SBML,
            settings={"variable_step_size": True, "initial_time_step": 0.1},
        )
        model.set_integrator_settings(variable_step_size=False, relative_tolerance=1e-8)
    assert caplog.text.count("variable_step_size") == 1
    assert caplog.text.count("initial_time_step") == 1


def test_unknown_setting_raises() -> None:
    with pytest.raises(ValueError, match="no_such_setting"):
        CopasiSBMLModel(source=DEMO_SBML).set_integrator_settings(no_such_setting=1)


def test_unknown_selection_raises() -> None:
    with pytest.raises(ValueError, match="no selection"):
        CopasiSBMLModel(source=DEMO_SBML, selections=["time", "c__A'"])


def test_the_settings_of_lsoda(caplog: pytest.LogCaptureFixture) -> None:
    model = CopasiSBMLModel(source=sbml(PROBE))
    model.set_integrator_settings(
        relative_tolerance=1e-8, absolute_tolerance=1e-11, maximum_num_steps=5000
    )
    name, settings = model.integrator_state()
    assert name == "lsoda"
    values = dict(settings)
    assert [k for k, _ in settings] == sorted(values)
    assert values["relative_tolerance"] == 1e-8
    assert values["absolute_tolerance"] == AbsoluteTolerance.of(1e-11)
    assert values["maximum_num_steps"] == 5000
    method = model.dm.getTask("Time-Course").getMethod()
    assert method.getParameter("Relative Tolerance").getDblValue() == 1e-8
    assert method.getParameter("Absolute Tolerance").getDblValue() == 1e-11
    assert method.getParameter("Max Internal Steps").getUIntValue() == 5000
    # a tolerance per kind is the smallest one, reported once
    with caplog.at_level(logging.WARNING, logger="sbmlsim.model.model_copasi"):
        tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-12, ids={"A": 1e-8})
        model.set_integrator_settings(absolute_tolerance=tolerance)
        model.set_integrator_settings(absolute_tolerance=tolerance)
    assert method.getParameter("Absolute Tolerance").getDblValue() == 1e-12
    assert dict(model.integrator_state()[1])["absolute_tolerance"] == tolerance
    assert caplog.text.count("smallest") == 1


def test_the_state_ids() -> None:
    model = CopasiSBMLModel(source=sbml(SPECIES_KINDS))
    assert model.state_ids() == ["F", "RR", "I", "N"]


def test_load_model_loads_a_model_in_copasi() -> None:
    roadrunner_model = RoadrunnerSBMLModel(source=sbml(PROBE), sid="p")
    model = load_model(roadrunner_model, Engine.COPASI)
    assert isinstance(model, CopasiSBMLModel)
    assert model.sid == "p"
    assert load_model(model, "copasi") is model
    assert isinstance(
        load_model(AbstractModel(source=DEMO_SBML), "copasi"), CopasiSBMLModel
    )


def test_the_data_model_is_removed_when_the_model_is_released() -> None:
    gc.collect()
    n = COPASI.CRootContainer.getDatamodelList().size()
    model = CopasiSBMLModel(source=sbml(PROBE))
    assert COPASI.CRootContainer.getDatamodelList().size() == n + 1
    model.close()
    model.close()
    assert COPASI.CRootContainer.getDatamodelList().size() == n
    CopasiSBMLModel(source=sbml(PROBE))
    gc.collect()
    assert COPASI.CRootContainer.getDatamodelList().size() == n


def test_loading_prints_nothing(capfd: pytest.CaptureFixture[str]) -> None:
    CopasiSBMLModel(source=HCTZ_SBML).initialize([_a("BW", 80.0)])
    out, err = capfd.readouterr()
    assert out == "" and err == ""
