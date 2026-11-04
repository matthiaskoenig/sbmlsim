"""The model of an SBML model in AMICI."""

import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import libsbml
import pytest

pytest.importorskip("amici")

from sbmlsim.engine import Engine
from sbmlsim.model import (
    AbstractModel,
    RoadrunnerSBMLModel,
    load_model,
    model_amici,
)
from sbmlsim.model.model_amici import (
    AMICI_ATOL_FACTOR,
    CACHE_ENV,
    COUNT_PREFIX,
    EVENT_TIME_PREFIX,
    TARGETS_EVENT,
    AmiciSBMLModel,
    amici_sbml,
    cache_key,
    compiled_dir,
)
from sbmlsim.model.symbols import TargetKind
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.resources import (
    DEMO_SBML,
    HCTZ_SBML,
    MIDAZOLAM_SBML,
    REPRESSILATOR_SBML,
)
from sbmlsim.simulator.plan import Assignment
from tests.simulator.models import (
    PROBE,
    SPECIES_KINDS,
    VOLUME,
    VOLUME_OF_A_RULE,
    VOLUME_OF_THE_REST,
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
      <species id="A" compartment="c" initialAmount="3" substanceUnits="mmol"
        hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="5"
        hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
  </model>
</sbml>
"""

#: the probe model with events: `E1` fires at the time 1, `E2` when `A` is
#: above 5, both with a trigger whose initial value is false, `E3` with one
#: whose initial value is true; a parameter `n` which only the events set
EVENTS = PROBE.replace(
    "end",
    """  n = 0
  E1: at (time > 1), t0=false: n = n + 1
  E2: at (A > 5), t0=false: n = n + 10
  E3: at (A > 7), t0=true: n = n + 100
end""",
)


def _a(
    target: str, value: float, kind: TargetKind = TargetKind.PARAMETER
) -> Assignment:
    return Assignment(target=target, kind=kind, value=value)


def _values(model: AmiciSBMLModel) -> dict[str, float]:
    """Get the value of every selection of a model which has one."""
    assert model.selections is not None
    return {s: model.value(s) for s in model.selections}


def _same(a: dict[str, float], b: dict[str, float]) -> bool:
    """Whether two values of the selections are equal bitwise, `NaN` included."""
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


def _probe_with(edit: Callable[[libsbml.Model], None], model: str = PROBE) -> str:
    """Get a model with an edit of its model."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(model))
    edit(doc.getModel())
    return libsbml.writeSBMLToString(doc)


@pytest.mark.parametrize(
    "source", [REPRESSILATOR_SBML, DEMO_SBML, MIDAZOLAM_SBML, HCTZ_SBML]
)
def test_loads_with_the_selections_of_roadrunner(source: Path) -> None:
    model = AmiciSBMLModel(source=source)
    assert model.engine is Engine.AMICI
    assert model.selections is not None
    assert set(model.selections) == set(
        RoadrunnerSBMLModel(source=source).selections or ()
    )
    assert model.selections == model.default_selections()
    assert not model.selections_given


@pytest.mark.parametrize(
    "source",
    [
        REPRESSILATOR_SBML,
        DEMO_SBML,
        MIDAZOLAM_SBML,
        HCTZ_SBML,
        sbml(PROBE),
        sbml(SPECIES_KINDS),
        sbml(VOLUME_OF_THE_REST),
        OTHER_UNITS,
    ],
)
def test_values_are_the_ones_of_roadrunner_as_loaded(source: Path | str) -> None:
    """AMICI keeps the numbers of the SBML, a species in other units included."""
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    assert r.selections is not None
    for s in r.selections:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-12, abs=1e-300), s
        assert a.has_selection(s)


#: a model which SBML level 1 and level 2 version 1 have, without initial
#: assignments and rules
PLAIN = """
model plain
  compartment C = 2
  species A in C = 1
  species B in C = 0.5
  k1 = 0.8
  J1: A -> B; k1*A
end
"""


@pytest.mark.parametrize(("level", "version"), [(2, 1), (1, 2)])
def test_a_model_of_an_older_level_is_converted(level: int, version: int) -> None:
    """AMICI compiles the model in level 3, its values are the ones of roadrunner."""
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PLAIN))
    assert doc.setLevelAndVersion(level, version, False)
    source = libsbml.writeSBMLToString(doc)
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    assignments = [_a("k1", 2.0), _a("C", 4.0, TargetKind.COMPARTMENT)]
    for model in (a, r):
        model.initialize(assignments)
    assert r.selections is not None
    assert set(a.selections or ()) == set(r.selections)
    for s in r.selections:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-12), s


def test_preinit_overrides_an_initial_assignment_and_updates_dependents() -> None:
    model = AmiciSBMLModel(source=sbml(OVERRIDE))
    loaded = _values(model)

    def assert_x(x: float) -> None:
        assert model.value("[X]") == x
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
    assert _same(_values(model), _values(AmiciSBMLModel(source=sbml(OVERRIDE))))


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
    (PROBE, [_a("[X]", 10.0, TargetKind.SPECIES_CONCENTRATION)]),
    (SPECIES_KINDS, [_a("BW", 80.0)]),
    (SPECIES_KINDS, [_a("V", 10.0, TargetKind.COMPARTMENT)]),
    (SPECIES_KINDS, [_a("BW", 80.0), _a("[F]", 3.0, TargetKind.SPECIES_CONCENTRATION)]),
    (SPECIES_KINDS, [_a("N", 1.0, TargetKind.SPECIES_AMOUNT), _a("k", 2.0)]),
    (SPECIES_KINDS, [_a("[RR]", 3.0, TargetKind.SPECIES_CONCENTRATION)]),
    (VOLUME_OF_A_RULE, [_a("BW", 70.45995681845807)]),
    (VOLUME_OF_THE_REST, [_a("BW", 80.0), _a("FVpa", 0.005)]),
    (VOLUME, [_a("f", 3.0)]),
    (OVERRIDE, [_a("[X]", 10.0, TargetKind.SPECIES_CONCENTRATION)]),
    (OVERRIDE, [_a("y", 4.0), _a("x0", 2.0)]),
    (
        OVERRIDE,
        [_a("c", 3.0, TargetKind.COMPARTMENT), _a("X", 3.0, TargetKind.SPECIES_AMOUNT)],
    ),
    (EVENTS, [_a("n", 5.0), _a("f", 3.0)]),
]


@pytest.mark.parametrize(("model", "assignments"), PREINIT_CASES)
def test_preinit_values_are_the_ones_of_roadrunner(
    model: str, assignments: list[Assignment]
) -> None:
    """A pre-initialization gives the values of roadrunner, a second one restores."""
    source = sbml(model)
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    loaded = _values(a)
    for _ in range(2):
        a.initialize(assignments)
        r.initialize(assignments)
        assert r.selections is not None
        for s in r.selections:
            assert a.value(s) == pytest.approx(r.value(s), rel=1e-12), s
        a.simulated()
        r.simulated()
    a.initialize([])
    assert _same(_values(a), loaded)


def test_preinit_of_the_hctz_model_is_the_one_of_roadrunner() -> None:
    """The body weight sets the volumes of the organs, whose species follow."""
    a, r = AmiciSBMLModel(source=HCTZ_SBML), RoadrunnerSBMLModel(source=HCTZ_SBML)
    loaded = _values(a)
    assignments = [_a("BW", 80.0), _a("PODOSE_hctz", 25.0)]
    a.initialize(assignments)
    r.initialize(assignments)
    assert r.selections is not None
    for s in r.selections:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-12, abs=1e-300), s
    a.initialize([])
    assert _same(_values(a), loaded)


def test_preinit_formulas_read_the_parameters() -> None:
    model = AmiciSBMLModel(source=sbml(PROBE))
    model.initialize(
        [_a("f", 3.0), Assignment("k1", TargetKind.PARAMETER, formula="2 * f + a0")]
    )
    assert model.value("k1") == 7.0
    assert model.value("kk") == 9.0
    # a formula reads an assignment rule on the values which are set
    model.initialize(
        [_a("f", 4.0), Assignment("k2", TargetKind.PARAMETER, formula="kk + pinit")]
    )
    assert model.value("k2") == 12.0 + 4.0


def test_a_reset_leaves_the_model_as_loaded() -> None:
    model = AmiciSBMLModel(source=sbml(SPECIES_KINDS))
    loaded = _values(model)
    model.initialize([_a("BW", 80.0), _a("[F]", 3.0, TargetKind.SPECIES_CONCENTRATION)])
    model.values.setValue("k", 4.0)
    model.set_time(3.0)
    model.reset()
    assert _same(_values(model), loaded)
    assert model.time == 0.0


def test_the_values_as_loaded() -> None:
    model = AmiciSBMLModel(source=sbml(PROBE))
    model.initialize([_a("f", 5.0), _a("C", 4.0, TargetKind.COMPARTMENT)])
    assert model.loaded_value("f") == 2.0
    assert model.loaded_value("C") == 2.0
    assert model.loaded_value("[A]") == 1.0
    assert model.loaded_value("A") == 2.0
    assert model.loaded_value("X") == 12.0
    for selection in ["J1", "time", "[nope]", "nope", "[f]"]:
        with pytest.raises(KeyError):
            model.loaded_value(selection)


def test_the_current_values_are_set_through_the_state() -> None:
    """A constant parameter, a compartment and a species are set between segments."""
    model = AmiciSBMLModel(source=sbml(PROBE))
    values = model.values
    assert values.getValue("time") == 0.0
    values.setValue("k1", 2.0)
    assert values.getValue("k1") == 2.0
    assert values.getValue("J1") == 2.0  # k1 * [A]
    values.setValue("A", 4.0)
    assert values.getValue("A") == 4.0
    assert values.getValue("[A]") == 2.0
    values.setValue("[B]", 3.0)
    assert values.getValue("B") == 6.0
    # a compartment keeps the amounts of its species, as in roadrunner
    values.setValue("C", 4.0)
    assert values.getValue("A") == 4.0
    assert values.getValue("[A]") == 1.0
    assert values.getValue("X") == 12.0
    values.setValue("f", 5.0)
    assert values.getValue("kk") == 15.0  # an assignment rule follows
    # a parameter of an initial assignment is a constant after the initialization
    values.setValue("pinit", 1.0)
    assert values.getValue("pinit") == 1.0
    assert values.getValue("X") == 12.0
    with pytest.raises(RuntimeError, match="assignment rule"):
        values.setValue("kk", 1.0)
    with pytest.raises(RuntimeError, match="no selection 'nope'"):
        values.getValue("nope")
    with pytest.raises(RuntimeError, match="cannot set 'J1'"):
        values.setValue("J1", 1.0)
    with pytest.raises(RuntimeError, match="cannot set 'time'"):
        values.setValue("time", 1.0)
    # the initialization restores the values as loaded
    model.initialize([])
    assert model.value("k1") == 0.8 and model.value("C") == 2.0
    assert not model.has_selection("[k1]")


def test_a_parameter_keeps_the_amounts_in_a_compartment_of_a_rule() -> None:
    """A compartment of an assignment rule keeps the amounts of its species.

    AMICI integrates the concentration of a species, roadrunner its amount,
    which a change of an entity the volume depends on keeps.
    """
    source = sbml(VOLUME_OF_THE_REST)
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    for model in (a.values, r.r_loaded):
        model.setValue("BW", 60.0)
    for s in ["S", "[S]", "V", "Vre"]:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-14), s


def test_a_state_of_a_rate_rule_is_set() -> None:
    model = AmiciSBMLModel(source=sbml(SPECIES_KINDS))
    model.values.setValue("[RR]", 3.0)
    assert model.value("[RR]") == 3.0
    assert model.value("RR") == pytest.approx(3.0 * 15.0, rel=1e-15)


def test_the_time_and_the_state() -> None:
    """The executor sets the time and the state of the model between segments."""
    model = AmiciSBMLModel(source=sbml(PROBE))
    states = model.amici_model.get_state_ids()
    state = model.state
    state[states.index("A")] = 3.0
    model.set_state(2.5, state)
    assert model.time == 2.5
    assert model.value("[A]") == 3.0 and model.value("J1") == pytest.approx(2.4)
    # the state of the model is a copy
    state[states.index("A")] = 4.0
    assert model.value("[A]") == 3.0


def test_a_value_which_is_not_a_number_is_no_state() -> None:
    """A compartment whose size is `NaN` is a number of AMICI, a change cannot set it.

    CVODE integrates no state which is not finite, as the compartments of
    the HCTZ model whose size is `NaN`; a parameter is a parameter of AMICI,
    which is set.
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
    model = AmiciSBMLModel(source=source)
    assert math.isnan(model.value("p")) and math.isnan(model.value("V"))
    with pytest.raises(RuntimeError, match=r"'V'.*not a number"):
        model.values.setValue("V", 1.0)
    with pytest.raises(NotImplementedError, match=r"'V'.*not a number"):
        model.initialize([_a("V", 3.0, TargetKind.COMPARTMENT)])
    # also with a formula, which reads the values which are set
    with pytest.raises(NotImplementedError, match=r"'V'.*not a number"):
        model.initialize(
            [
                _a("V", 3.0, TargetKind.COMPARTMENT),
                Assignment("p", TargetKind.PARAMETER, formula="2 * X"),
            ]
        )
    model.initialize([_a("p", 3.0)])
    assert model.value("p") == 3.0
    model.values.setValue("p", 2.0)
    assert model.value("p") == 2.0
    model.values.setValue("X", 2.0)
    assert model.value("X") == 2.0


def test_the_events_start_with_their_trigger_true() -> None:
    """AMICI fires no event at the start of an integration, see the module.

    The events whose trigger is false before the time 0 are named, the
    executor fires them at the start of a simulation.
    """
    model = AmiciSBMLModel(source=sbml(EVENTS))
    assert model.initial_value_false == ("E1", "E2")
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(
        amici_sbml(model.sbml, model.symbols, model.label).sbml
    )
    triggers = {
        e.getId(): e.getTrigger().getInitialValue()
        for e in doc.getModel().getListOfEvents()
    }
    assert triggers == {"E1": True, "E2": True, "E3": True, TARGETS_EVENT: True}
    # the values are evaluated without firing an event
    model.values.setValue("[A]", 10.0)
    model.set_time(3.0)
    assert model.value("n") == 0.0
    assert model.value("[A]") == 10.0


def test_species_in_units_other_than_the_model_keep_their_numbers() -> None:
    a, r = AmiciSBMLModel(source=OTHER_UNITS), RoadrunnerSBMLModel(source=OTHER_UNITS)
    for s in ["A", "[A]", "B", "[B]", "c"]:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-15)


def test_a_conversion_factor_is_a_parameter() -> None:
    """A conversion factor is a free parameter of AMICI, which is set as in roadrunner.

    Its effect on the rates is compared by the tests of the executor.
    """

    def edit(model: libsbml.Model) -> None:
        model.getSpecies("A").setConversionFactor("k2")

    source = _probe_with(edit)
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    assert "k2" in a.amici_model.get_free_parameter_ids()
    a.initialize([_a("k2", 0.3)])
    r.initialize([_a("k2", 0.3)])
    assert a.value("k2") == r.value("k2") == 0.3


def test_an_import_error_of_amici_raises() -> None:
    def edit(model: libsbml.Model) -> None:
        model.createParameter().setId("z")
        model.getParameter("z").setConstant(False)
        rule: libsbml.AssignmentRule = model.createAssignmentRule()
        rule.setVariable("z")
        rule.setMath(libsbml.parseL3Formula("undefined_function(A)"))

    with pytest.raises(ValueError, match="AMICI cannot import the model 'probe'"):
        AmiciSBMLModel(source=_probe_with(edit))


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

    return edit


def _event(edit: Callable[[libsbml.Event], None]) -> Callable[[libsbml.Model], None]:
    """Get the edit which adds the event `E` at the time 1 and edits it."""

    def add(model: libsbml.Model) -> None:
        event: libsbml.Event = model.createEvent()
        event.setId("E")
        event.setUseValuesFromTriggerTime(True)
        trigger: libsbml.Trigger = event.createTrigger()
        trigger.setMath(libsbml.parseL3Formula("time > 1"))
        trigger.setInitialValue(False)
        trigger.setPersistent(True)
        assignment: libsbml.EventAssignment = event.createEventAssignment()
        assignment.setVariable("A")
        assignment.setMath(libsbml.parseL3Formula("2"))
        edit(event)

    return add


def _algebraic(model: libsbml.Model) -> None:
    model.getParameter("a0").setConstant(False)
    rule: libsbml.AlgebraicRule = model.createAlgebraicRule()
    rule.setMath(libsbml.parseL3Formula("a0 - 2"))


def _delay(model: libsbml.Model) -> None:
    model.getParameter("b0").setConstant(False)
    rule: libsbml.AssignmentRule = model.createAssignmentRule()
    rule.setVariable("b0")
    rule.setMath(libsbml.parseL3Formula("delay(A, 1)"))


def _delayed(event: libsbml.Event) -> None:
    event.createDelay().setMath(libsbml.parseL3Formula("0.5"))


def _not_persistent(event: libsbml.Event) -> None:
    event.getTrigger().setPersistent(False)


def _priority(event: libsbml.Event) -> None:
    event.createPriority().setMath(libsbml.parseL3Formula("k1"))


def _initial_conversion_factor(model: libsbml.Model) -> None:
    model.getSpecies("A").setConversionFactor("k2")
    initial: libsbml.InitialAssignment = model.createInitialAssignment()
    initial.setSymbol("k2")
    initial.setMath(libsbml.parseL3Formula("2 * f"))


def _rate_of(model: libsbml.Model) -> None:
    model.getParameter("b0").setConstant(False)
    rule: libsbml.AssignmentRule = model.createAssignmentRule()
    rule.setVariable("b0")
    rule.setMath(libsbml.parseL3Formula("rateOf(A)"))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_algebraic, "an algebraic rule"),
        (_delay, r"the delay csymbol in the assignment rule of 'b0'"),
        (_rate_of, r"the csymbol rateOf in the assignment rule of 'b0'"),
        (_event(_delayed), "the delay of the event 'E'"),
        (
            _event(_not_persistent),
            "the trigger of the event 'E', which is not persistent",
        ),
        (_event(_priority), "the priority of the event 'E', which is no number"),
        (_rule_of_the_product("non-constant"), "non-constant species reference 'B_sr'"),
        (_rule_of_the_product("assignment"), "non-constant species reference 'B_sr'"),
        (_rule_of_the_product("rate"), "non-constant species reference 'B_sr'"),
        (
            _rule_of_the_product("initial"),
            "initial assignment of the species reference 'B_sr'",
        ),
        (
            _initial_conversion_factor,
            "initial assignment of the conversion factor 'k2'",
        ),
    ],
)
def test_an_unsupported_feature_raises(
    edit: Callable[[libsbml.Model], None], match: str
) -> None:
    """A feature AMICI does not simulate as roadrunner raises before AMICI compiles."""
    with pytest.raises(ValueError, match=match):
        AmiciSBMLModel(source=_probe_with(edit))


def test_a_stoichiometry_math_raises() -> None:
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    doc.setLevelAndVersion(2, 4)
    product: libsbml.SpeciesReference = doc.getModel().getReaction("J1").getProduct(0)
    math_: libsbml.StoichiometryMath = product.createStoichiometryMath()
    math_.setMath(libsbml.parseL3Formula("2 * f"))
    with pytest.raises(ValueError, match="stoichiometry math of the species 'B'"):
        AmiciSBMLModel(source=libsbml.writeSBMLToString(doc))


def test_a_fast_reaction_raises() -> None:
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(PROBE))
    doc.setLevelAndVersion(2, 4)
    doc.getModel().getReaction("J1").setFast(True)
    with pytest.raises(ValueError, match="the fast reaction 'J1'"):
        AmiciSBMLModel(source=libsbml.writeSBMLToString(doc))


def test_a_species_reference_is_its_stoichiometry() -> None:
    """A constant species reference with an id is its number, which AMICI folds in."""

    def edit(model: libsbml.Model) -> None:
        model.getSpecies("B").setBoundaryCondition(True)
        model.getReaction("J1").getProduct(0).setId("B_sr")
        model.getReaction("J1").getProduct(0).setStoichiometry(2.0)
        model.getReaction("J1").getReactant(0).setId("A_sr")
        # a reference of level 3 without a stoichiometry, 1 in roadrunner
        model.getReaction("J2").getProduct(0).setId("A2_sr")
        model.getReaction("J2").getProduct(0).unsetStoichiometry()

    source = _probe_with(edit)
    a, r = AmiciSBMLModel(source=source), RoadrunnerSBMLModel(source=source)
    assert r.selections is not None and "B_sr" in r.selections
    assert a.selections == a.default_selections()
    for s in [*r.selections, "A_sr", "A2_sr"]:
        assert a.value(s) == pytest.approx(r.value(s), rel=1e-12), s
    assert a.value("B_sr") == 2.0
    assert a.value("A2_sr") == 1.0
    with pytest.raises(RuntimeError, match="cannot set 'B_sr'"):
        a.values.setValue("B_sr", 3.0)


def test_roadrunner_only_settings_warn_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sbmlsim.model.model_amici"):
        model = AmiciSBMLModel(
            source=DEMO_SBML,
            settings={"variable_step_size": True, "initial_time_step": 0.1},
        )
        model.set_integrator_settings(variable_step_size=False, relative_tolerance=1e-8)
    assert caplog.text.count("variable_step_size") == 1
    assert caplog.text.count("initial_time_step") == 1


def test_unknown_setting_raises() -> None:
    with pytest.raises(ValueError, match="no_such_setting"):
        AmiciSBMLModel(source=DEMO_SBML).set_integrator_settings(no_such_setting=1)


def test_unknown_selection_raises() -> None:
    with pytest.raises(ValueError, match="no selection"):
        AmiciSBMLModel(source=DEMO_SBML, selections=["time", "c__A'"])
    with pytest.raises(RuntimeError, match="no selection"):
        AmiciSBMLModel(source=DEMO_SBML).value("c__A'")


def test_the_settings_of_cvodes(caplog: pytest.LogCaptureFixture) -> None:
    model = AmiciSBMLModel(source=sbml(PROBE))
    model.set_integrator_settings(
        relative_tolerance=1e-8,
        absolute_tolerance=1e-11,
        maximum_num_steps=5000,
        maximum_time_step=0.5,
    )
    name, settings = model.integrator_state()
    assert name == "cvodes"
    values = dict(settings)
    assert [k for k, _ in settings] == sorted(values)
    assert values == {
        "absolute_tolerance": AbsoluteTolerance.of(1e-11),
        "maximum_num_steps": 5000,
        "maximum_time_step": 0.5,
        "relative_tolerance": 1e-8,
    }
    solver = model.solver
    assert solver.get_relative_tolerance() == 1e-8
    assert solver.get_absolute_tolerance() == 1e-11 * AMICI_ATOL_FACTOR
    assert solver.get_max_steps() == 5000
    assert solver.get_max_step_size() == 0.5
    # a tolerance per kind is the smallest one, reported once
    with caplog.at_level(logging.WARNING, logger="sbmlsim.model.model_amici"):
        tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-12, ids={"A": 1e-8})
        model.set_integrator_settings(absolute_tolerance=tolerance)
        model.set_integrator_settings(absolute_tolerance=tolerance)
    assert solver.get_absolute_tolerance() == 1e-12 * AMICI_ATOL_FACTOR
    assert dict(model.integrator_state()[1])["absolute_tolerance"] == tolerance
    assert caplog.text.count("smallest") == 1
    # the state of the integrator is set on another model
    other = AmiciSBMLModel(source=sbml(PROBE))
    other.set_integrator_state(model.integrator_state())
    assert other.integrator_state() == model.integrator_state()
    with pytest.raises(ValueError, match="not with 'cvode'"):
        other.set_integrator_state(("cvode", ()))
    for setting in ({"maximum_num_steps": 0.5}, {"relative_tolerance": -1.0}):
        with pytest.raises(ValueError):
            other.set_integrator_settings(**setting)


def test_the_state_ids() -> None:
    model = AmiciSBMLModel(source=sbml(SPECIES_KINDS))
    assert model.state_ids() == ["F", "RR", "I", "N"]


def test_load_model_loads_a_model_in_amici() -> None:
    roadrunner_model = RoadrunnerSBMLModel(source=sbml(PROBE), sid="p")
    model = load_model(roadrunner_model, Engine.AMICI)
    assert isinstance(model, AmiciSBMLModel)
    assert model.sid == "p"
    assert load_model(model, "amici") is model
    assert isinstance(
        load_model(AbstractModel(source=DEMO_SBML), "amici"), AmiciSBMLModel
    )


def test_loading_a_compiled_model_prints_nothing(
    capfd: pytest.CaptureFixture[str],
) -> None:
    AmiciSBMLModel(source=HCTZ_SBML)
    capfd.readouterr()
    AmiciSBMLModel(source=HCTZ_SBML).initialize([_a("BW", 80.0)])
    out, err = capfd.readouterr()
    assert out == "" and err == ""


#: a small model of the tests of the cache, compiled into a cache of its own
CACHED = """
model cached
  compartment C = 1
  species S in C = 1
  k = {k}
  J: S -> ; k*S
end
"""


def test_the_cache_reuses_a_compiled_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    source = sbml(CACHED.format(k=0.25))
    model = AmiciSBMLModel(source=source)
    key = cache_key(amici_sbml(model.sbml, model.symbols, model.label).sbml)
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [key]
    assert compiled_dir(amici_sbml(model.sbml, model.symbols, model.label).sbml) == (
        tmp_path / key
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("compiled again")

    monkeypatch.setattr(model_amici, "_compile", fail)
    again = AmiciSBMLModel(source=source)
    assert again.value("k") == 0.25
    # another model is compiled
    with pytest.raises(AssertionError, match="compiled again"):
        AmiciSBMLModel(source=sbml(CACHED.format(k=0.5)))


def test_a_compilation_which_did_not_end_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory of a compilation of a process which was killed is removed."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    source = sbml(CACHED.format(k=0.125))
    model = AmiciSBMLModel(source=source, selections=["S"])
    compiled = amici_sbml(model.sbml, model.symbols, model.label).sbml
    key = cache_key(compiled)
    shutil.rmtree(tmp_path / key)
    left = tmp_path / f"{key}.left_over"
    (left / "build").mkdir(parents=True)
    assert compiled_dir(compiled) == tmp_path / key
    assert not left.exists()
    assert (tmp_path / f"{key}.lock").is_file()


def test_a_missing_build_tool_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model is not blamed for a build environment without ninja."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    real_which = shutil.which

    def which(cmd: str, *args: object, **kwargs: object) -> str | None:
        return None if cmd == "ninja" else real_which(cmd)

    monkeypatch.setattr(model_amici.shutil, "which", which)
    with pytest.raises(RuntimeError, match="builds a model with 'ninja'"):
        AmiciSBMLModel(source=sbml(CACHED.format(k=0.0625)))
    assert [p for p in tmp_path.iterdir() if p.is_dir()] == []


def _segment(model: AmiciSBMLModel, timepoints: list[float]) -> dict[str, list[float]]:
    """Integrate a model from its current state, see `AmiciSBMLModel.prepare`."""
    model.prepare(timepoints)
    rdata = model_amici.run_simulation()(model.amici_model, model.solver)
    assert rdata.status == 0
    states = list(model.amici_model.get_state_ids())
    x = [list(row) for row in rdata.x]
    model.set_state(timepoints[-1], x[-1])
    return {sid: [row[k] for row in x] for k, sid in enumerate(states)}


def test_an_integration_after_a_restart_fires_the_crossings_only() -> None:
    """AMICI fires no event at the start of a segment, and every crossing in it.

    A trigger which a change makes true (`A > 5`, `A > 7`) or which is true
    already (`time > 1`) does not fire at the start of the next segment; a
    trigger which becomes true in a segment fires, also after a restart.
    """
    model = AmiciSBMLModel(source=sbml(EVENTS))
    assert _segment(model, [0.0, 0.5, 1.5, 2.0])["n"] == [0.0, 0.0, 1.0, 1.0]
    assert _segment(model, [2.0, 3.0])["n"] == [1.0, 1.0]
    model.values.setValue("[A]", 10.0)
    assert _segment(model, [3.0, 3.5])["n"] == [1.0, 1.0]
    # A rises above 5 and 7 from B
    model.values.setValue("[A]", 4.0)
    model.values.setValue("[B]", 20.0)
    model.set_time(6.0)
    assert _segment(model, [6.0, 6.5, 8.0])["n"] == [1.0, 11.0, 111.0]
    # the time event in a segment which starts before its time
    model.initialize([])
    assert _segment(model, [0.0, 0.5])["n"] == [0.0, 0.0]
    assert _segment(model, [0.5, 1.5])["n"] == [0.0, 1.0]


#: the script of a process which loads a model in AMICI and reports whether
#: it compiled it
COMPILE_SCRIPT = """
import sys
from sbmlsim.model import model_amici

compiled = []
original = model_amici._compile

def counted(*args, **kwargs):
    compiled.append(True)
    return original(*args, **kwargs)

model_amici._compile = counted
model = model_amici.AmiciSBMLModel(source=sys.argv[1])
print(len(compiled), model.value("k"))
"""


def test_two_processes_compile_a_model_once(tmp_path: Path) -> None:
    path = tmp_path / "cached.xml"
    path.write_text(sbml(CACHED.format(k=0.75)), encoding="utf-8")
    script = tmp_path / "load.py"
    script.write_text(COMPILE_SCRIPT, encoding="utf-8")
    cache = tmp_path / "cache"
    env = {**os.environ, CACHE_ENV: str(cache)}
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [p.communicate(timeout=600) for p in processes]
    assert [p.returncode for p in processes] == [0, 0], outputs
    counts = sorted(out.split()[0] for out, _ in outputs)
    assert counts == ["0", "1"], outputs
    assert all(out.split()[1] == "0.75" for out, _ in outputs)
    entries = sorted(p.name for p in cache.iterdir() if p.is_dir())
    assert len(entries) == 1, entries


def test_the_states_of_the_events_are_no_selections() -> None:
    """The counters and times of the events sbmlsim adds are states of AMICI only."""
    model = AmiciSBMLModel(source=sbml(EVENTS))
    added = {
        f"{prefix}{k}" for prefix in (COUNT_PREFIX, EVENT_TIME_PREFIX) for k in range(3)
    }
    assert added <= {str(s) for s in model.amici_model.get_state_ids()}
    assert model.selections is not None
    for listed in (model.selections, model.default_selections(), model.state_ids()):
        assert not added & set(listed)
    for sid in added:
        assert not model.has_selection(sid)
    reference = RoadrunnerSBMLModel(source=sbml(EVENTS))
    assert model.default_selections() == reference.default_selections()
