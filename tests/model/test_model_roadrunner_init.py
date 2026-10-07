"""The model is initialized with the pre-initialization values of a plan."""

import time
from pathlib import Path

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.symbols import ModelSymbols, TargetKind
from sbmlsim.simulator.plan import Assignment
from tests.simulator.models import sbml


def _a(
    target: str, value: float, kind: TargetKind = TargetKind.PARAMETER
) -> Assignment:
    return Assignment(target=target, kind=kind, value=value)


def test_a_model_is_loaded_from_sbml() -> None:
    """The source of a model is a path or the SBML itself."""
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.r_loaded["[B]"] == pytest.approx(1.0)
    assert model.symbols.kind("[B]") is TargetKind.SPECIES_CONCENTRATION


def test_preinit_reaches_initial_assignments_and_is_restored() -> None:
    """A value before the initialization reaches the initial assignments."""
    model = RoadrunnerSBMLModel(source=sbml())
    r = model.r_loaded
    model.initialize([_a("f", 5.0), _a("b0", 0.0)])
    assert r["[B]"] == pytest.approx(0.0)
    assert r["pinit"] == pytest.approx(10.0)
    assert r["X"] == pytest.approx(30.0)
    assert r["kk"] == pytest.approx(15.0)
    model.initialize([])
    assert r["[B]"] == pytest.approx(1.0)
    assert r["pinit"] == pytest.approx(4.0)
    assert r["X"] == pytest.approx(12.0)
    assert r["f"] == pytest.approx(2.0)


def test_parameter_with_initial_assignment_can_be_set() -> None:
    """A parameter with an initial assignment is set before the initialization."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize([_a("pinit", 7.0)])
    assert model.r_loaded["pinit"] == pytest.approx(7.0)
    assert model.r_loaded["X"] == pytest.approx(21.0)
    model.initialize([_a("f", 3.0)])
    assert model.r_loaded["pinit"] == pytest.approx(6.0)
    assert model.r_loaded["X"] == pytest.approx(18.0)


def test_species_with_initial_assignment() -> None:
    """A species set before the initialization replaces its initial assignment."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize([_a("[B]", 5.0, TargetKind.SPECIES_CONCENTRATION), _a("b0", 0.0)])
    assert model.r_loaded["[B]"] == pytest.approx(5.0)
    model.initialize([_a("b0", 0.0)])
    assert model.r_loaded["[B]"] == pytest.approx(0.0)
    model.initialize([])
    assert model.r_loaded["[B]"] == pytest.approx(1.0)


def test_compartment_before_initialization() -> None:
    """A compartment before the initialization keeps the initial concentrations."""
    conc = sbml(
        "model c\n  compartment V = 1\n  species S in V = 3\n"
        "  substanceOnly species N in V\n  N = 2\nend"
    )
    model = RoadrunnerSBMLModel(source=conc)
    model.initialize([_a("V", 4.0, TargetKind.COMPARTMENT)])
    r = model.r_loaded
    assert r["V"] == pytest.approx(4.0)
    assert r["[S]"] == pytest.approx(3.0)
    assert r["S"] == pytest.approx(12.0)
    assert r["N"] == pytest.approx(2.0)


def test_initial_assignment_order() -> None:
    """The initial assignments are evaluated after the ones they read."""
    chain = sbml("model ch\n  a = 1\n  b = 2*a\n  c = b + 1\n  d = c * a\nend")
    symbols = ModelSymbols.from_sbml(chain)
    order = symbols.initial_assignment_order
    assert order.index("b") < order.index("c") < order.index("d")
    model = RoadrunnerSBMLModel(source=chain)
    model.initialize([_a("a", 3.0)])
    assert model.r_loaded["d"] == pytest.approx((2 * 3 + 1) * 3)


def test_helpers_are_not_selected() -> None:
    """The helper of an initial assignment is not in the default selections."""
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.selections is not None
    assert "pinit__initial" not in model.selections
    assert model.initial_helpers["pinit"] == "pinit__initial"


def test_initialize_is_fast() -> None:
    """An initialization sets values and does not regenerate the model."""
    model = RoadrunnerSBMLModel(source=sbml())
    start = time.perf_counter()
    for _ in range(100):
        model.initialize([_a("f", 5.0), _a("b0", 0.0)])
    assert (time.perf_counter() - start) / 100 < 0.005


T0_EVENT = """
model events_at_t0
  P1 = 3/2
  E0: at 2.5 after (P1 > 1), t0=false, fromTrigger=false: P1 = P1^2
  E1: at 1.3 after (P1 > 1), t0=false: P1 = 5
end
"""


def _p1(model: RoadrunnerSBMLModel) -> list[float]:
    import numpy as np

    r = model.r_loaded
    r.timeCourseSelections = ["time", "P1"]
    values = np.array(r.simulate(0, 5, 6))[:, 1].tolist()
    model.simulated()
    return values


def test_events_at_t0_fire_once() -> None:
    """A reset without a simulation after it does not fire the events at t0 again.

    roadrunner queues the events which fire at the time 0 with every reset, a
    loaded model counts as one (case 01757 of the SBML Test Suite).
    """
    expected = [1.5, 1.5, 5.0, 25.0, 25.0, 25.0]
    model = RoadrunnerSBMLModel(source=sbml(T0_EVENT))
    model.initialize([])
    assert _p1(model) == pytest.approx(expected)
    model.initialize([])
    model.initialize([])
    assert _p1(model) == pytest.approx(expected)


def test_values_of_an_initialization_without_simulation_are_restored() -> None:
    """Two initializations without a simulation do not leak into each other."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize(
        [_a("f", 5.0), _a("b0", 0.0), _a("C", 4.0, TargetKind.COMPARTMENT)]
    )
    model.initialize([_a("k1", 0.1)])
    r = model.r_loaded
    assert r["f"] == pytest.approx(2.0)
    assert r["X"] == pytest.approx(12.0)
    assert r["[B]"] == pytest.approx(1.0)
    assert r["C"] == pytest.approx(2.0)
    assert r["[A]"] == pytest.approx(1.0)
    assert r["k1"] == pytest.approx(0.1)


def test_preinit_concentration_with_a_compartment_of_an_initial_assignment() -> None:
    """A concentration set before the initialization keeps its value.

    The compartment follows a changed parameter through its initial
    assignment, the species set as a concentration keeps the concentration.
    """
    model = RoadrunnerSBMLModel(
        source=sbml(
            "model c\n  V = 1\n  compartment C = 2*V\n  species S in C = 1\nend"
        )
    )
    model.initialize([_a("V", 2.0), _a("[S]", 3.0, TargetKind.SPECIES_CONCENTRATION)])
    assert model.r_loaded["C"] == pytest.approx(4.0)
    assert model.r_loaded["[S]"] == pytest.approx(3.0)


def test_initial_assignment_of_a_concentration_follows_its_compartment() -> None:
    """An initial assignment reading a concentration follows a changed compartment."""
    import libsbml

    doc = libsbml.readSBMLFromString(
        sbml("model c\n  compartment C = 2\n  species T in C\n  T = 2\n  p = T\nend")
    )
    species = doc.getModel().getSpecies("T")
    # a concentration species defined by its initial amount: a larger
    # compartment dilutes it, and p reads its concentration
    species.unsetInitialConcentration()
    species.setInitialAmount(4.0)
    model = RoadrunnerSBMLModel(source=libsbml.writeSBMLToString(doc))
    model.initialize([_a("C", 4.0, TargetKind.COMPARTMENT)])
    assert model.r_loaded["[T]"] == pytest.approx(1.0)
    assert model.r_loaded["p"] == pytest.approx(1.0)


def test_events_and_a_start_other_than_zero_are_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """roadrunner evaluates the triggers of events at the time 0, which is reported.

    The state of the triggers is kept across a reset, so an event whose
    trigger depends on the time can fire at the start of a simulation which
    does not start at 0; nothing in the API of roadrunner sets the time of
    the initialization of the triggers.
    """
    import logging

    from sbmlsim.simulation import Simulation
    from sbmlsim.simulator.executor import execute
    from sbmlsim.simulator.plan import compile_simulation

    model = RoadrunnerSBMLModel(
        source=sbml("model ev\n  X = 0\n  E: at (time >= 0), t0=false: X = 5\nend")
    )
    plan = compile_simulation(Simulation(start=-10, end=10), model.symbols, model.uinfo)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.simulator.executor"):
        execute(plan, model, ["time", "X"])
        execute(plan, model, ["time", "X"])
    warnings = [r for r in caplog.records if "events" in r.getMessage()]
    assert len(warnings) == 1


def test_initialization_with_compartment_and_concentration_is_not_left_over() -> None:
    """An initialization without a simulation after it leaves nothing behind."""
    model = RoadrunnerSBMLModel(source=sbml())
    model.initialize(
        [
            _a("C", 4.0, TargetKind.COMPARTMENT),
            _a("[A]", 3.0, TargetKind.SPECIES_CONCENTRATION),
        ]
    )
    model.initialize([])
    assert model.r_loaded["C"] == pytest.approx(2.0)
    assert model.r_loaded["[A]"] == pytest.approx(1.0)


def test_parameter_df_has_no_helpers() -> None:
    """The helpers of the initial assignments are not parameters of the model."""
    model = RoadrunnerSBMLModel(source=sbml())
    df = RoadrunnerSBMLModel.parameter_df(model.r_loaded)
    assert "pinit__initial" not in set(df["sid"])
    assert "pinit" in set(df["sid"])


#: a model which enables the package comp without submodels, as the models of
#: sbmlutils do, with an initial assignment
COMP_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
      xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
      level="3" version="1" comp:required="true">
  <model id="comp_initial_assignment">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="2" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A1" compartment="c" initialAmount="1"
               hasOnlySubstanceUnits="true" boundaryCondition="false"
               constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="D" value="5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="A1">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci> D </ci><cn> 2 </cn></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <comp:listOfPorts>
      <comp:port comp:id="D_port" comp:idRef="D"/>
    </comp:listOfPorts>
  </model>
</sbml>
"""


def test_initial_assignments_of_a_model_with_the_package_comp() -> None:
    """A model which enables comp keeps its initial assignments."""
    model = RoadrunnerSBMLModel(source=COMP_SBML)
    assert model.symbols.initial_assignment_order == ("A1",)
    model.initialize([_a("D", 2.0)])
    assert model.r_loaded["A1"] == pytest.approx(4.0)
    model.initialize([])
    assert model.r_loaded["A1"] == pytest.approx(10.0)


#: a hierarchical model, the initial assignment is in its submodel
HIERARCHICAL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
      xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
      level="3" version="1" comp:required="true">
  <model id="top">
    <comp:listOfSubmodels>
      <comp:submodel comp:id="sub" comp:modelRef="inner"/>
    </comp:listOfSubmodels>
  </model>
  <comp:listOfModelDefinitions>
    <comp:modelDefinition id="inner">
      <listOfCompartments>
        <compartment id="c" spatialDimensions="3" size="2" constant="true"/>
      </listOfCompartments>
      <listOfSpecies>
        <species id="A1" compartment="c" initialAmount="1"
                 hasOnlySubstanceUnits="true" boundaryCondition="false"
                 constant="false"/>
      </listOfSpecies>
      <listOfParameters>
        <parameter id="k" value="3" constant="true"/>
      </listOfParameters>
      <listOfInitialAssignments>
        <initialAssignment symbol="A1">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><cn> 2 </cn></apply>
          </math>
        </initialAssignment>
      </listOfInitialAssignments>
    </comp:modelDefinition>
  </comp:listOfModelDefinitions>
</sbml>
"""


def test_initial_assignments_of_a_hierarchical_model(tmp_path: Path) -> None:
    """The flattened hierarchical model keeps the initial assignments."""
    path = tmp_path / "hierarchical.xml"
    path.write_text(HIERARCHICAL_SBML, encoding="utf-8")
    model = RoadrunnerSBMLModel(source=path)
    assert model.symbols.initial_assignment_order == ("sub__A1",)
    model.initialize([_a("sub__k", 10.0)])
    assert model.r_loaded["sub__A1"] == pytest.approx(20.0)
