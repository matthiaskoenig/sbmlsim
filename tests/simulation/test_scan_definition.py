"""A scan and its dimensions are validated when they are created and immutable."""

import json
import pickle
from typing import Any

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulation.scan import DimensionKind

SIM = Simulation(end=10, steps=10)


def test_the_values_are_read_only_copies() -> None:
    values = np.array([1.0, 2.0])
    dimension = Dimension("d", values={"k1": values})
    values[0] = 5.0
    assert dimension.values["k1"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        dimension.values["k1"][0] = 3.0


def test_a_quantity_keeps_its_unit_and_is_read_only() -> None:
    dose = Dimension("dose", values={"PODOSE": Q([5, 10], "mg")}).values["PODOSE"]
    assert str(dose.units) == "milligram"
    assert dose.magnitude.tolist() == [5.0, 10.0]
    assert not dose.magnitude.flags.writeable


def test_a_list_is_an_array() -> None:
    assert len(Dimension("d", values={"k1": [1, 2, 3]})) == 3


@pytest.mark.parametrize(
    "values", [1.0, Q(5, "mg"), "k1*2", ["a", "b"], [[1.0, 2.0]], []]
)
def test_values_which_are_no_array_of_numbers_are_an_error(values: Any) -> None:
    with pytest.raises(ValueError, match="'k1'"):
        Dimension("d", values={"k1": values})


def test_the_values_of_a_dimension_have_one_length() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        Dimension("d", values={"k1": [1.0, 2.0], "k2": [1.0]})


def test_the_labels_of_values_are_their_positions() -> None:
    assert Dimension("d", values={"k1": [5.0, 6.0]}).labels.tolist() == [0, 1]


def test_the_labels_are_given() -> None:
    dimension = Dimension("d", values={"k1": [5.0, 6.0]}, labels=["low", "high"])
    assert dimension.labels.tolist() == ["low", "high"]
    assert not dimension.labels.flags.writeable


@pytest.mark.parametrize("labels", [["a"], ["a", "a"]])
def test_labels_which_do_not_fit_are_an_error(labels: list[str]) -> None:
    with pytest.raises(ValueError, match="labels"):
        Dimension("d", values={"k1": [5.0, 6.0]}, labels=labels)


def test_the_labels_of_simulations_and_models_are_their_keys() -> None:
    simulations = Dimension("regimen", simulations={"single": SIM, "multiple": SIM})
    assert simulations.kind is DimensionKind.SIMULATIONS
    assert simulations.labels.tolist() == ["single", "multiple"]
    models = Dimension("genotype", models={"wt": "wt.xml", "pm": "pm.xml"})
    assert models.kind is DimensionKind.MODELS
    assert len(models) == 2


@pytest.mark.parametrize(
    "kwargs", [{}, {"values": {"k1": [1.0]}, "simulations": {"a": SIM}}]
)
def test_a_dimension_varies_one_thing(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Dimension("d", **kwargs)


def test_only_values_have_a_time() -> None:
    with pytest.raises(ValueError, match="'at'"):
        Dimension("d", simulations={"a": SIM}, at=1.0)


def test_a_simulation_of_a_dimension_is_a_simulation() -> None:
    simulations: dict[str, Any] = {"a": "sim"}
    with pytest.raises(ValueError, match="Simulation"):
        Dimension("d", simulations=simulations)


def test_the_points_are_in_c_order() -> None:
    scan = Scan(
        SIM,
        [
            Dimension("a", values={"k1": [1.0, 2.0]}),
            Dimension("b", values={"k2": [1.0, 2.0, 3.0]}),
        ],
    )
    assert scan.shape == (2, 3)
    assert scan.size == len(scan) == 6
    assert scan.dims == ("a", "b")
    assert list(scan.points())[:4] == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_a_simulation_is_a_scan_of_one_point() -> None:
    scan = Scan.of(SIM)
    assert scan.shape == ()
    assert scan.size == 1
    assert list(scan.points()) == [()]
    assert Scan.of(scan) is scan
    assert scan.simulations() == [SIM]


@pytest.mark.parametrize("sid", ["time", "_point", "statistic", "status"])
def test_a_reserved_id_is_an_error(sid: str) -> None:
    with pytest.raises(ValueError, match="names of the result"):
        Scan(SIM, [Dimension(sid, values={"k1": [1.0]})])


def test_two_dimensions_of_one_id_are_an_error() -> None:
    with pytest.raises(ValueError, match="more than once"):
        Scan(
            SIM,
            [
                Dimension("d", values={"k1": [1.0]}),
                Dimension("d", values={"k2": [1.0]}),
            ],
        )


def test_a_dimension_named_as_a_target_is_an_error() -> None:
    with pytest.raises(ValueError, match="changed targets"):
        Scan(
            SIM,
            [
                Dimension("k1", values={"k2": [1.0]}),
                Dimension("d", values={"k1": [1.0]}),
            ],
        )


def test_a_target_is_set_once_before_the_initialization() -> None:
    with pytest.raises(ValueError, match="'k1'"):
        Scan(
            SIM,
            [
                Dimension("a", values={"k1": [1.0]}),
                Dimension("b", values={"k1": [2.0]}),
            ],
        )


def test_a_target_is_set_once_at_a_time() -> None:
    Scan(
        SIM,
        [
            Dimension("a", values={"k1": [1.0]}),
            Dimension("b", values={"k1": [2.0]}, at=5),
        ],
    )
    with pytest.raises(ValueError, match="'k1'"):
        Scan(
            SIM,
            [
                Dimension("a", values={"k1": [1.0]}, at=5),
                Dimension("b", values={"k1": [2.0]}, at=5),
            ],
        )


def test_one_dimension_of_simulations_and_one_of_models() -> None:
    with pytest.raises(ValueError, match="at most one"):
        Scan(
            SIM,
            [
                Dimension("a", simulations={"x": SIM}),
                Dimension("b", simulations={"y": SIM}),
            ],
        )
    with pytest.raises(ValueError, match="at most one"):
        Scan(
            SIM,
            [
                Dimension("a", models={"x": "x.xml"}),
                Dimension("b", models={"y": "y.xml"}),
            ],
        )


def test_a_time_outside_of_the_simulation_is_an_error() -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(SIM, [Dimension("d", values={"k1": [1.0]}, at=11)])


def test_a_time_outside_of_a_simulation_of_a_dimension_is_an_error() -> None:
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(
            SIM,
            [
                Dimension("sim", simulations={"short": Simulation(end=1), "long": SIM}),
                Dimension("d", values={"k1": [1.0]}, at=5),
            ],
        )


def test_a_time_with_a_unit() -> None:
    hours = Simulation(time_unit="hr", end=2)
    Scan(hours, [Dimension("d", values={"k1": [1.0]}, at=Q(90, "min"))])
    with pytest.raises(ValueError, match="outside of the simulation"):
        Scan(hours, [Dimension("d", values={"k1": [1.0]}, at=Q(3, "hr"))])


def test_the_scan_is_stored_as_json() -> None:
    scan = Scan(
        Simulation(end=10, changes=[Change(1, {"k1": 2.0})]),
        [
            Dimension("dose", values={"PODOSE": Q([5, 10], "mg")}, at=1),
            Dimension("regimen", simulations={"single": SIM}),
            Dimension("genotype", models={"wt": "wt.xml"}),
        ],
    )
    d = json.loads(json.dumps(scan.to_dict()))
    assert [dimension["id"] for dimension in d["dimensions"]] == [
        "dose",
        "regimen",
        "genotype",
    ]
    assert d["dimensions"][0]["values"]["PODOSE"] == {
        "value": [5.0, 10.0],
        "unit": "milligram",
    }
    assert d["dimensions"][2]["models"] == {"wt": "wt.xml"}


def test_a_scan_pickles() -> None:
    scan = Scan(SIM, [Dimension("d", values={"k1": [1.0, 2.0]})])
    again = pickle.loads(pickle.dumps(scan))
    assert again.dims == ("d",)
    assert again.dimensions[0].values["k1"].tolist() == [1.0, 2.0]
