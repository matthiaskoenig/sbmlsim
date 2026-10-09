"""The record of a design and the coordinates of a dimension."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulation.scan import Design
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml

DESIGN = Design(
    method="lhs",
    distributions={"k1": {"type": "Uniform", "lower": 0.5, "upper": 1.0}},
    options={"n": 3, "seed": 7},
    references={},
)


def test_a_design_is_json_and_compares_by_its_content() -> None:
    again = Design.from_dict(json.loads(json.dumps(DESIGN.to_dict())))
    assert again == DESIGN
    assert again is not DESIGN
    with pytest.raises(ValueError, match="JSON"):
        Design(method="lhs", options={"rng": np.random.default_rng(1)})


def test_a_dimension_carries_its_design_and_coordinates() -> None:
    dimension = Dimension(
        "d",
        values={"k1": [0.5, 0.75, 1.0]},
        design=DESIGN,
        coordinates={"BW": Q([60.0, 70.0, 80.0], "kg")},
    )
    assert dimension.design == DESIGN
    assert dimension.coordinates["BW"].magnitude.tolist() == [60.0, 70.0, 80.0]
    assert "lhs" in repr(dimension)
    data = dimension.to_dict()
    assert data["design"] == DESIGN.to_dict()
    assert data["coordinates"] == {
        "BW": {"value": [60.0, 70.0, 80.0], "unit": "kilogram"}
    }
    again = pickle.loads(pickle.dumps(dimension))
    assert again.design == DESIGN and set(again.coordinates) == {"BW"}


def test_a_design_and_coordinates_need_a_dimension_of_values() -> None:
    with pytest.raises(ValueError, match="values"):
        Dimension("d", simulations={"a": Simulation(end=1)}, design=DESIGN)
    with pytest.raises(ValueError, match="length"):
        Dimension("d", values={"k1": [1.0, 2.0]}, coordinates={"BW": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="target"):
        Dimension("d", values={"k1": [1.0, 2.0]}, coordinates={"k1": [1.0, 2.0]})
    with pytest.raises(TypeError):
        Dimension("d", values={"k1": [1.0]}, design={"method": "lhs"})  # ty: ignore[invalid-argument-type]


def test_the_result_carries_the_record_and_the_coordinates(tmp_path: Path) -> None:
    dimension = Dimension(
        "d",
        values={"k1": [0.5, 0.75, 1.0]},
        design=DESIGN,
        coordinates={"BW": Q([60.0, 70.0, 80.0], "kg")},
    )
    res = Simulator().run(sbml(), Scan(Simulation(end=1, steps=2), [dimension]))
    assert res.ds["BW"].dims == ("d",)
    assert res.units["BW"] == "kilogram"
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    again = ScanResult.from_netcdf(path)
    (stored,) = again.ds.attrs["scan"]["dimensions"]
    assert Design.from_dict(stored["design"]) == DESIGN
    np.testing.assert_array_equal(again.ds["BW"].values, [60.0, 70.0, 80.0])
