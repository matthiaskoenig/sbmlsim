"""Tests of a network: its files, its ids and its forward pass."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

from sbmlsim.sciml import (
    BackendKind,
    Network,
    NetworkImportError,
    UnsupportedLayerError,
)
from sbmlsim.sciml.network import element_id


def _node(name: str, op: str, target: str, args: list) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs={})


def _model(layer_type: str = "Linear") -> NNModel:
    """Build `layer2(tanh(layer1(x)))` with 2 inputs, 3 hidden units, 1 output."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3, "bias": True},
            ),
            Layer(
                layer_id="layer2",
                layer_type="Linear",
                args={"in_features": 3, "out_features": 1, "bias": False},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("tanh", "call_method", "tanh", ["layer1"]),
            _node("layer2", "call_module", "layer2", ["tanh"]),
            _node("output", "output", "output", ["layer2"]),
        ],
    )


def _parameters() -> dict[str, dict[str, np.ndarray]]:
    return {
        "layer1": {
            "weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "bias": np.array([0.1, 0.2, 0.3]),
        },
        "layer2": {"weight": np.array([[1.0, -1.0, 0.5]])},
    }


def _write(path: Path, parameters: dict, pytorch_format: bool = True) -> Path:
    """Write the YAML and the array file of the network into a directory."""
    NNModelStandard.save_data(data=_model(), filename=str(path / "net1.yaml"))
    with h5py.File(path / "net1_ps.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = pytorch_format
        for layer, arrays in parameters.items():
            for name, array in arrays.items():
                f[f"parameters/net1/{layer}/{name}"] = array
    return path


def test_the_forward_pass() -> None:
    """The network is evaluated by hand."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    x = np.array([0.5, -0.25])
    hidden = np.tanh(_parameters()["layer1"]["weight"] @ x + [0.1, 0.2, 0.3])
    (y,) = network.forward(x)
    np.testing.assert_allclose(y, [hidden[0] - hidden[1] + 0.5 * hidden[2]])
    assert y.dtype == float


def test_a_network_is_read_from_its_files(tmp_path: Path) -> None:
    """The YAML and the HDF5 file give the network."""
    _write(tmp_path, _parameters())
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")

    assert network.sid == "net1"
    assert list(network.parameters) == ["layer1", "layer2"]
    np.testing.assert_array_equal(
        network.parameters["layer1"]["weight"], _parameters()["layer1"]["weight"]
    )
    expected = Network(sid="net1", model=_model(), parameters=_parameters())
    x = np.array([[0.5, -0.25], [1.0, 2.0]])
    np.testing.assert_allclose(network.forward(x)[0], expected.forward(x)[0])


def test_the_id_of_a_problem_replaces_the_id_of_the_yaml(tmp_path: Path) -> None:
    """A problem names its networks, the arrays are read under that name."""
    NNModelStandard.save_data(data=_model(), filename=str(tmp_path / "net.yaml"))
    with h5py.File(tmp_path / "ps.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        f["parameters/net7/layer2/weight"] = np.ones((1, 3))
    network = Network.from_files(
        tmp_path / "net.yaml", tmp_path / "ps.hdf5", sid="net7"
    )
    assert network.sid == network.model.nn_model_id == "net7"
    assert next(iter(network.parameter_ids())) == "net7__layer1__weight__0_0"
    assert list(network.parameters) == ["layer2"]


def test_the_column_major_layout_is_permuted(tmp_path: Path) -> None:
    """Arrays which are not in the PyTorch layout have their axes reversed."""
    stored = {
        layer: {name: array.T for name, array in arrays.items()}
        for layer, arrays in _parameters().items()
    }
    _write(tmp_path, stored, pytorch_format=False)
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")
    np.testing.assert_array_equal(
        network.parameters["layer1"]["weight"], _parameters()["layer1"]["weight"]
    )
    assert network.parameters["layer2"]["weight"].shape == (1, 3)


def test_an_empty_array_is_an_array_without_values(tmp_path: Path) -> None:
    """The file of a problem leaves out the arrays the problem sets."""
    parameters = _parameters()
    parameters["layer1"]["weight"] = np.array([])
    _write(tmp_path, parameters)
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")
    assert list(network.parameters["layer1"]) == ["bias"]
    with pytest.raises(NetworkImportError, match=r"'layer1'.*'weight' has no values"):
        network.forward(np.zeros(2))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layer1": {"weight": np.ones((2, 3))}}, r"'weight' has the shape \(2, 3\)"),
        ({"layer1": {"gain": np.ones(3)}}, "'gain' is not an array of the layer"),
        ({"layer9": {"weight": np.ones(3)}}, "the layer 'layer9'"),
        ({"layer1": {"bias": np.array([0.0, np.nan, 0.0])}}, "not finite"),
        ({"layer1": {"bias": np.array([0.0, np.inf, 0.0])}}, "not finite"),
    ],
)
def test_arrays_which_do_not_fit(tmp_path: Path, change: dict, message: str) -> None:
    """An array of a file which does not fit the architecture is an error."""
    parameters = _parameters()
    for layer, arrays in change.items():
        parameters.setdefault(layer, {}).update(arrays)
    _write(tmp_path, parameters)
    with pytest.raises(NetworkImportError, match=message):
        Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")


def test_files_which_do_not_exist(tmp_path: Path) -> None:
    """A missing file is an error of the import which names the file."""
    with pytest.raises(NetworkImportError, match=r"missing\.yaml"):
        Network.from_files(tmp_path / "missing.yaml")
    _write(tmp_path, _parameters())
    with pytest.raises(NetworkImportError, match=r"missing\.hdf5"):
        Network.from_files(tmp_path / "net1.yaml", tmp_path / "missing.hdf5")


def test_an_array_file_of_another_network(tmp_path: Path) -> None:
    """The file must hold the arrays of the network."""
    _write(tmp_path, _parameters())
    with pytest.raises(NetworkImportError, match=r"no parameters of the network"):
        Network.from_files(
            tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5", sid="net2"
        )


def test_the_ids_of_the_elements() -> None:
    """Every element has an id with its PyTorch index."""
    network = Network(sid="net1", model=_model())
    ids = network.parameter_ids()

    assert len(ids) == 6 + 3 + 3
    assert list(ids)[:3] == [
        "net1__layer1__weight__0_0",
        "net1__layer1__weight__0_1",
        "net1__layer1__weight__1_0",
    ]
    assert ids["net1__layer1__weight__2_1"] == ("layer1", "weight", (2, 1))
    assert ids["net1__layer1__bias__2"] == ("layer1", "bias", (2,))
    assert ids["net1__layer2__weight__0_2"] == ("layer2", "weight", (0, 2))
    assert "net1__layer2__bias__0" not in ids


def test_an_id_is_an_sid() -> None:
    """The dot of a nested layer is not part of an id."""
    assert (
        element_id("net1", "block.0", "weight", (1, 2)) == "net1__block_0__weight__1_2"
    )


def test_two_elements_with_one_id() -> None:
    """Layers whose ids differ only in a replaced character are an error."""
    model = _model()
    model.layers[0].layer_id = "block.0"
    model.layers[1] = model.layers[0].model_copy(update={"layer_id": "block_0"})
    with pytest.raises(NetworkImportError, match=r"have the id 'net1__block_0__"):
        Network(sid="net1", model=model).parameter_ids()


def test_values_replace_elements() -> None:
    """`with_values` returns a copy, the network keeps its nominal values."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    parameters = network.with_values(
        {"net1__layer1__weight__2_1": 60.0, "net1__layer2__weight__0_0": -7.0}
    )

    assert parameters["layer1"]["weight"][2, 1] == 60.0
    assert parameters["layer2"]["weight"][0, 0] == -7.0
    assert network.parameters["layer1"]["weight"][2, 1] == 6.0
    assert parameters["layer1"]["bias"] is not network.parameters["layer1"]["bias"]

    x = np.array([0.5, -0.25])
    assert network.forward(x, parameters=parameters)[0] != network.forward(x)[0]
    np.testing.assert_array_equal(
        network.forward(x, parameters=network.with_values({}))[0], network.forward(x)[0]
    )


def test_a_value_of_an_unknown_element() -> None:
    """An id which is not an element of the network is an error."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(KeyError, match=r"net1__layer1__weight__3_0"):
        network.with_values({"net1__layer1__weight__3_0": 1.0})


def test_a_value_of_an_array_without_values() -> None:
    """An element of an array without nominal values cannot be set."""
    network = Network(sid="net1", model=_model())
    with pytest.raises(NetworkImportError, match=r"'weight' has no values"):
        network.with_values({"net1__layer1__weight__0_0": 1.0})


def test_a_layer_which_is_not_called_needs_no_arrays() -> None:
    """Only the layers of the forward pass are evaluated."""
    model = _model()
    model.forward = [
        _node("net_input", "placeholder", "net_input", []),
        _node("tanh", "call_method", "tanh", ["net_input"]),
        _node("output", "output", "output", ["tanh"]),
    ]
    network = Network(sid="net1", model=model)
    assert network.used_layers() == []
    np.testing.assert_allclose(network.forward(np.array([0.5]))[0], np.tanh([0.5]))


def test_the_backends_of_a_network() -> None:
    """`Linear` and `tanh` are evaluated by both backends."""
    assert Network(sid="net1", model=_model()).backends() == frozenset(BackendKind)


def test_a_layer_without_an_implementation() -> None:
    """An unknown layer names the network, the node and the type."""
    network = Network(sid="net1", model=_model(layer_type="LSTM"))
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'layer1'.*'LSTM'"):
        network.forward(np.zeros(2))
    with pytest.raises(UnsupportedLayerError):
        network.parameter_ids()


def test_the_number_of_inputs() -> None:
    """The inputs are the placeholders of the forward pass."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(ValueError, match=r"2 inputs were given for the 1 inputs"):
        network.forward(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match=r"0 inputs were given"):
        network.forward()


def test_an_input_of_the_wrong_size() -> None:
    """The error of numpy names the network and the node."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(ValueError, match=r"Network 'net1', node 'layer1'"):
        network.forward(np.zeros(3))


def test_several_outputs() -> None:
    """An output node with a list returns one array per entry."""
    model = _model()
    model.forward[-1] = _node("output", "output", "output", [["layer1", "layer2"]])
    network = Network(sid="net1", model=model, parameters=_parameters())
    hidden, y = network.forward(np.array([0.5, -0.25]))
    assert hidden.shape == (3,)
    assert y.shape == (1,)
