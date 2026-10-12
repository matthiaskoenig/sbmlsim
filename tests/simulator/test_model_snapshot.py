"""Worker models are snapshots of the model loaded in the parent."""

import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.model.test_model_roadrunner_init import EXTERNAL_SBML, HIERARCHICAL_SBML
from tests.simulator.models import PROBE, sbml

#: a rate rule without math (case 01236 of the SBML Test Suite), which the SBML
#: roadrunner writes back drops
NO_MATH_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="no_math">
    <listOfParameters>
      <parameter id="p" value="3" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="p"/>
    </listOfRules>
  </model>
</sbml>
"""


@pytest.fixture(autouse=True)
def _spawn_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Spawn also works in sandboxes that disallow the forkserver's Unix socket.
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **kwargs: "spawn")


@pytest.mark.parametrize("action", ["rewrite", "delete"])
def test_a_loaded_file_is_unchanged_in_workers(action: str, tmp_path: Path) -> None:
    path = tmp_path / "model.xml"
    path.write_text(sbml())
    model = RoadrunnerSBMLModel(
        source=path, parameters={"extra": 3}, selections=["k1", "pinit", "extra", "[A]"]
    )
    if action == "rewrite":
        path.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    else:
        path.unlink()
    for preinit in ({"f": 4}, {}):
        simulation = Simulation(end=1, steps=2, preinit_changes=preinit)
        serial = Simulator(n_workers=1).run(model, simulation)
        pooled = Simulator(n_workers=2).run(model, simulation, progress=False)
        xr.testing.assert_identical(pooled.ds, serial.ds)
        assert pooled["extra"].values.tolist() == [3, 3, 3]


def test_a_worker_snapshot_includes_external_models(tmp_path: Path) -> None:
    inner = HIERARCHICAL_SBML.split("<comp:listOfModelDefinitions>")[1]
    inner = inner.split("</comp:listOfModelDefinitions>")[0]
    inner = inner.replace("comp:modelDefinition", "model")
    dependency = tmp_path / "inner.xml"
    dependency.write_text(
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
        f'level="3" version="1">{inner}</sbml>'
    )
    path = tmp_path / "top.xml"
    path.write_text(EXTERNAL_SBML)
    model = RoadrunnerSBMLModel(source=path, selections=["sub__A1"])
    dependency.unlink()
    path.unlink()
    simulation = Simulation(end=1, steps=2, preinit_changes={"sub__k": 10})
    serial = Simulator(n_workers=1).run(model, simulation)
    pooled = Simulator(n_workers=2).run(model, simulation, progress=False)
    xr.testing.assert_identical(pooled.ds, serial.ds)
    assert pooled["sub__A1"].values.tolist() == [20, 20, 20]


def test_a_model_with_an_element_without_math_runs_in_workers() -> None:
    model = RoadrunnerSBMLModel(source=NO_MATH_SBML, selections=["time", "p"])
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("d", values={"p": np.linspace(1, 2, 4)})],
    )
    serial = Simulator(n_workers=1).run(model, scan)
    pooled = Simulator(n_workers=2).run(model, scan, progress=False)
    xr.testing.assert_identical(pooled.ds, serial.ds)
    assert pooled["p"].values[:, -1].tolist() == pytest.approx([1, 4 / 3, 5 / 3, 2])
