"""Resolved model descriptions and compiled plans preserve their meaning."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.simulation import Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.plan import model_units
from sbmlsim.simulator.worker import ModelSpec, run_chunk, run_chunk_in_worker
from sbmlsim.units import UnitsInformation, ureg
from tests.simulator.models import PROBE, sbml, sbml_minutes


def test_loading_a_description_keeps_its_resolved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for directory, rate in [("first", 0.8), ("second", 0.1)]:
        path = tmp_path / directory
        path.mkdir()
        (path / "model.xml").write_text(sbml(PROBE.replace("k1 = 0.8", f"k1 = {rate}")))
    monkeypatch.chdir(tmp_path / "first")
    description = AbstractModel("model.xml", selections=["time", "k1"])
    monkeypatch.chdir(tmp_path / "second")
    model = Simulator(n_workers=1).load(description)
    assert model.source.path == description.source.path
    assert model.selections == description.selections
    result = Simulator(n_workers=1).simulate(description, Simulation(end=1, steps=1))
    np.testing.assert_array_equal(result["k1"], [0.8, 0.8])


def test_explicit_selections_override_model_description() -> None:
    description = AbstractModel(sbml(), selections=["time", "A"])
    model = RoadrunnerSBMLModel.from_abstract_model(
        description, selections=["time", "B"]
    )
    assert model.selections == ["time", "B"]


def test_a_plan_cannot_change_time_units_when_reused() -> None:
    simulator = Simulator(n_workers=1)
    seconds = simulator.load(sbml())
    minutes = simulator.load(sbml_minutes())
    definition = Simulation(end=ureg.Quantity(60, "second"), steps=1)
    plan = simulator.compile(seconds, definition)
    with pytest.raises(ValueError, match="different model units"):
        simulator.simulate(minutes, plan)
    np.testing.assert_array_equal(simulator.simulate(minutes, definition).time, [0, 1])


def test_a_plan_cannot_change_target_kinds_when_reused() -> None:
    simulator = Simulator(n_workers=1)
    amount = simulator.load(sbml())
    concentration = simulator.load(
        sbml(PROBE.replace("substanceOnly species X", "species X"))
    )
    plan = simulator.compile(amount, Simulation(end=1, steps=1))
    with pytest.raises(ValueError, match="different model symbols"):
        simulator.simulate(concentration, plan)


def test_compatible_plans_can_run_other_parameter_values_and_snapshots() -> None:
    simulator = Simulator(n_workers=1)
    first = simulator.load(sbml())
    other = simulator.load(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    definition = Simulation(end=1, steps=1)
    plan = simulator.compile(first, definition).with_values({"a0": 2})
    result = simulator.simulate(other, plan)
    assert result["k1"][0] == 0.1
    assert result["A"][0] == 4
    snapshot = ModelSpec.of(other).load()
    np.testing.assert_array_equal(
        simulator.simulate(snapshot, plan).values, result.values
    )


@pytest.mark.parametrize("worker", [False, True])
def test_chunks_reject_incompatible_model_units_before_executing(
    worker: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the worker runs in this process and keeps its model
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    simulator = Simulator(n_workers=1)
    seconds = simulator.load(sbml())
    minutes = simulator.load(sbml_minutes())
    compiled = simulator._compile(seconds, Scan(Simulation(end=1, steps=1)), None)
    chunk = compiled.chunks(1, "flag")[0]
    with pytest.raises(ValueError, match="different model units"):
        if worker:
            run_chunk_in_worker(ModelSpec.of(minutes), chunk)
        else:
            run_chunk(chunk, minutes)


#: a model in years, a year in seconds: its reaction has the unit
#: `item/(31557600.0*s)`, a unit with a factor (Rahman_MBS2016 of the PEtab
#: benchmark collection)
YEAR_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="years" substanceUnits="item" timeUnits="year" extentUnits="item">
    <listOfUnitDefinitions>
      <unitDefinition id="year">
        <listOfUnits>
          <unit kind="second" exponent="1" scale="0" multiplier="31557600"/>
        </listOfUnits>
      </unitDefinition>
      <unitDefinition id="per_year">
        <listOfUnits>
          <unit kind="second" exponent="-1" scale="0" multiplier="31557600"/>
        </listOfUnits>
      </unitDefinition>
    </listOfUnitDefinitions>
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" units="dimensionless"
                   constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="10" substanceUnits="item"
               hasOnlySubstanceUnits="true" boundaryCondition="false"
               constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" units="per_year" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="v" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> S </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_a_unit_with_a_factor_has_the_factor_in_its_scale() -> None:
    simulator = Simulator(n_workers=1)
    model = simulator.load(YEAR_SBML)
    assert model.uinfo["v"] == "item/(31557600.0*s)"
    plan = simulator.compile(model, Simulation(end=1, steps=1))
    assert plan.units is not None
    units = {key: (unit, scale) for key, unit, scale in plan.units}
    assert units["S"] == ("count", 1.0)
    assert units["v"] == ("count / second", pytest.approx(1 / 31557600))
    result = simulator.simulate(model, plan)
    assert result["S"][-1] == pytest.approx(10 * np.exp(-0.5), rel=1e-4)


def test_a_unit_pint_does_not_normalize_is_compared_as_it_is() -> None:
    uinfo = UnitsInformation({"x": "mole/(", "y": "mmol"}, ureg)
    assert model_units(uinfo) == (("x", "mole/(", None), ("y", "mole", 0.001))
