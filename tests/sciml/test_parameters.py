"""Tests of the nominal values and the fit parameters of a network."""

from itertools import pairwise

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network, NetworkImportError
from sbmlsim.sciml.parameters import (
    covered_arrays,
    network_fit_parameters,
    nominal_parameters,
    resolve_entries,
)


def _network(with_values: bool = True) -> Network:
    """Build `layer2(layer1(x))`, `layer1` is a layer of a nested module."""
    layers = [
        Layer(
            layer_id="block.layer1",
            layer_type="Linear",
            args={"in_features": 2, "out_features": 2, "bias": True},
        ),
        Layer(
            layer_id="norm",
            layer_type="BatchNorm1d",
            args={"num_features": 2},
        ),
        Layer(
            layer_id="layer2",
            layer_type="Linear",
            args={"in_features": 2, "out_features": 1, "bias": True},
        ),
    ]
    names = ["net_input", "block.layer1", "norm", "layer2"]
    forward = [
        Node(name="net_input", op="placeholder", target="net_input", args=[], kwargs={})
    ]
    for previous, name in pairwise(names):
        forward.append(
            Node(name=name, op="call_module", target=name, args=[previous], kwargs={})
        )
    forward.append(
        Node(name="output", op="output", target="output", args=["layer2"], kwargs={})
    )
    model = NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=layers,
        forward=forward,
    )
    parameters = {
        "block.layer1": {
            "weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": np.array([5.0, 6.0]),
        },
        "norm": {
            "weight": np.array([1.5, 2.5]),
            "bias": np.array([0.5, 0.25]),
            "running_mean": np.array([0.1, 0.2]),
            "running_var": np.array([1.0, 2.0]),
        },
        "layer2": {"weight": np.array([[7.0, 8.0]]), "bias": np.array([9.0])},
    }
    return Network(
        sid="net1", model=model, parameters=parameters if with_values else {}
    )


def test_the_arrays_an_entry_covers() -> None:
    """An entry is the network, a layer or an array."""
    network = _network()
    assert covered_arrays(network, "net1") == (
        0,
        [
            ("block.layer1", "weight"),
            ("block.layer1", "bias"),
            ("norm", "weight"),
            ("norm", "bias"),
            ("layer2", "weight"),
            ("layer2", "bias"),
        ],
    )
    assert covered_arrays(network, "net1.layer2") == (
        1,
        [("layer2", "weight"), ("layer2", "bias")],
    )
    assert covered_arrays(network, "net1.layer2.bias") == (2, [("layer2", "bias")])
    # the id of a layer of a nested module holds the separator
    assert covered_arrays(network, "net1.block.layer1") == (
        1,
        [("block.layer1", "weight"), ("block.layer1", "bias")],
    )
    assert covered_arrays(network, "net1.block.layer1.weight") == (
        2,
        [("block.layer1", "weight")],
    )


@pytest.mark.parametrize(
    "key",
    ["net2", "net1.layer3", "net1.layer2.gain", "net1.", "", "net1.norm.running_mean"],
)
def test_an_entry_which_covers_nothing(key: str) -> None:
    """A key which is not part of the network is an error, not an empty entry."""
    with pytest.raises(NetworkImportError, match=r"is not the network, a layer or an"):
        covered_arrays(_network(), key)


def test_the_more_specific_entry_wins() -> None:
    """The order of the entries does not matter, the array beats the layer."""
    network = _network()
    entries = {"net1.layer2.bias": 3.0, "net1.layer2": 2.0, "net1": 1.0}
    resolved = resolve_entries(network, entries)
    assert resolved[("block.layer1", "weight")] == 1.0
    assert resolved[("norm", "bias")] == 1.0
    assert resolved[("layer2", "weight")] == 2.0
    assert resolved[("layer2", "bias")] == 3.0
    assert resolved == resolve_entries(network, dict(reversed(entries.items())))


def test_the_nominal_values_of_the_array_file_are_kept() -> None:
    """Without entries the nominal values are the ones of the network."""
    network = _network()
    parameters = nominal_parameters(network)
    np.testing.assert_array_equal(
        parameters["layer2"]["weight"], network.parameters["layer2"]["weight"]
    )
    assert parameters["layer2"]["weight"] is not network.parameters["layer2"]["weight"]
    np.testing.assert_array_equal(parameters["norm"]["running_var"], [1.0, 2.0])


def test_a_value_replaces_the_elements_it_covers() -> None:
    """A layer is set to zero and the other layers keep their values."""
    network = _network()
    parameters = nominal_parameters(network, {"net1.block.layer1": 0.0})
    np.testing.assert_array_equal(
        parameters["block.layer1"]["weight"], np.zeros((2, 2))
    )
    np.testing.assert_array_equal(parameters["block.layer1"]["bias"], np.zeros(2))
    np.testing.assert_array_equal(parameters["layer2"]["weight"], [[7.0, 8.0]])
    # the network keeps its values
    assert network.parameters["block.layer1"]["weight"][0, 0] == 1.0

    parameters = nominal_parameters(
        network, {"net1": 1.0, "net1.layer2": 2.0, "net1.layer2.bias": 3.0}
    )
    np.testing.assert_array_equal(parameters["block.layer1"]["bias"], [1.0, 1.0])
    np.testing.assert_array_equal(parameters["layer2"]["weight"], [[2.0, 2.0]])
    np.testing.assert_array_equal(parameters["layer2"]["bias"], [3.0])
    # the running statistics are not parameters, a value does not reach them
    np.testing.assert_array_equal(parameters["norm"]["running_mean"], [0.1, 0.2])


def test_values_for_a_network_without_an_array_file() -> None:
    """The entries of a problem may be all the values a network has."""
    parameters = nominal_parameters(_network(with_values=False), {"net1": 0.5})
    assert parameters["block.layer1"]["weight"].shape == (2, 2)
    assert np.all(parameters["layer2"]["weight"] == 0.5)
    assert "running_mean" not in parameters["norm"]


def test_an_array_without_values_is_an_error() -> None:
    """A network is not initialized with random values."""
    with pytest.raises(NetworkImportError, match=r"'layer2'.*has no values"):
        nominal_parameters(
            _network(with_values=False), {"net1.block.layer1": 0.0, "net1.norm": 1.0}
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_value_which_is_not_finite(value: float) -> None:
    """A nominal value is a number."""
    with pytest.raises(NetworkImportError, match=r"not finite"):
        nominal_parameters(_network(), {"net1.layer2": value})


def test_the_fit_parameters_of_the_estimated_elements() -> None:
    """One parameter per estimated element, the other elements are frozen."""
    fit_parameters = network_fit_parameters(
        _network(),
        estimate={"net1": True, "net1.block.layer1": False, "net1.norm": False},
        bounds={"net1": (-10.0, 10.0), "net1.layer2.bias": (0.0, 20.0)},
        values={"net1.layer2.weight": 0.5},
    )
    assert [p.pid for p in fit_parameters] == [
        "net1__layer2__weight__0_0",
        "net1__layer2__weight__0_1",
        "net1__layer2__bias__0",
    ]
    weight, _, bias = fit_parameters
    assert (weight.start_value, weight.lower_bound, weight.upper_bound) == (
        0.5,
        -10.0,
        10.0,
    )
    assert (bias.start_value, bias.lower_bound, bias.upper_bound) == (9.0, 0.0, 20.0)
    assert bias.unit == "dimensionless"
    assert bias.target_id == bias.pid


def test_an_element_is_frozen_and_unbounded_by_default() -> None:
    """No entry means not estimated, no bounds means not bounded."""
    assert network_fit_parameters(_network(), estimate={}, bounds={}) == []

    fit_parameters = network_fit_parameters(
        _network(), estimate={"net1.block.layer1.bias": True}, bounds={}
    )
    assert [p.pid for p in fit_parameters] == [
        "net1__block_layer1__bias__0",
        "net1__block_layer1__bias__1",
    ]
    assert fit_parameters[0].lower_bound == -np.inf
    assert fit_parameters[0].upper_bound == np.inf


def test_a_nominal_value_outside_of_its_bounds() -> None:
    """The start value of a fit parameter is inside its bounds."""
    with pytest.raises(ValueError, match=r"outside of the bounds"):
        network_fit_parameters(
            _network(), estimate={"net1.layer2": True}, bounds={"net1": (0.0, 1.0)}
        )


def test_the_ids_are_the_ids_of_the_network() -> None:
    """The fit parameters can be written back with `with_values`."""
    network = _network()
    fit_parameters = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    assert [p.pid for p in fit_parameters] == list(network.parameter_ids())
    values = {p.pid: 0.0 for p in fit_parameters}
    parameters = network.with_values(values)
    assert all(
        np.all(parameters[layer][name] == 0.0)
        for layer, name, _ in network.parameter_ids().values()
    )


def test_an_estimated_array_without_nominal_values() -> None:
    """An estimated element needs a start value.

    A layer which the forward pass does not call needs no values to be
    evaluated, but its elements are not estimated without them.
    """
    network = _network(with_values=False)
    model = network.model.model_copy(deep=True)
    model.layers.append(
        Layer(
            layer_id="unused",
            layer_type="Linear",
            args={"in_features": 2, "out_features": 1, "bias": False},
        )
    )
    network = Network(sid="net1", model=model)
    values = {"net1.block.layer1": 0.0, "net1.norm": 1.0, "net1.layer2": 0.5}
    assert "unused" not in nominal_parameters(network, values)
    with pytest.raises(
        NetworkImportError, match=r"'unused'.*'weight' is estimated and has no nominal"
    ):
        network_fit_parameters(
            network, estimate={"net1.unused": True}, bounds={}, values=values
        )


def test_a_layer_without_trainable_arrays() -> None:
    """A key of a layer without parameters is part of the network and covers nothing."""
    network = _network()
    model = network.model.model_copy(deep=True)
    model.layers.append(Layer(layer_id="drop", layer_type="Dropout", args={"p": 0.1}))
    network = Network(sid="net1", model=model, parameters=network.parameters)

    assert covered_arrays(network, "net1.drop") == (1, [])
    with_entry = nominal_parameters(network, {"net1.drop": 0.0})
    without_entry = nominal_parameters(network)
    assert list(with_entry) == list(without_entry)
    for layer, arrays in without_entry.items():
        for name, array in arrays.items():
            np.testing.assert_array_equal(with_entry[layer][name], array)
    fit_parameters = network_fit_parameters(
        network, estimate={"net1.drop": True}, bounds={"net1.drop": (0.0, 1.0)}
    )
    assert fit_parameters == []
