"""Flattened submodels retain units for simulation and results."""

from pathlib import Path

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.units import Q, _create_registry
from tests.model.test_model_roadrunner_init import HIERARCHICAL_SBML


@pytest.mark.parametrize("file_source", [False, True])
def test_submodel_units_support_quantities_and_result_metadata(
    file_source: bool, tmp_path: Path
) -> None:
    text = HIERARCHICAL_SBML.replace(
        'id="c" spatialDimensions', 'id="c" units="litre" spatialDimensions'
    ).replace('id="A1" compartment', 'id="A1" substanceUnits="mole" compartment')
    source: str | Path = text
    if file_source:
        source = tmp_path / "hierarchical.xml"
        source.write_text(text)
    registry = _create_registry()
    model = RoadrunnerSBMLModel(source=source, ureg=registry)
    assert model.uinfo.ureg is registry
    assert model.uinfo["sub__A1"] == "mole"
    assert model.uinfo["[sub__A1]"] == "mole/litre"
    assert not set(model.initial_helpers.values()) & set(model.uinfo)
    result = Simulator(n_workers=1).run(
        model,
        Simulation(end=1, steps=1, preinit_changes={"sub__A1": Q(12000, "millimole")}),
    )
    assert result["sub__A1"].values.tolist() == [12, 12]
    assert result.units["sub__A1"] == "mole"
    assert result.quantity("sub__A1").to("millimole").magnitude.tolist() == [
        12000,
        12000,
    ]
