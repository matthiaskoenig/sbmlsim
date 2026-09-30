"""Tests of the interpreter of the forward pass."""

import inspect

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import UnsupportedLayerError
from sbmlsim.sciml.backend import NUMPY_ONLY, Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.interpreter import _array_parameters, evaluate
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS
from sbmlsim.sciml.layers.registry import FunctionType, function

WEIGHT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
BIAS = np.array([0.1, 0.2, 0.3])
ARRAYS = {"layer1": {"weight": WEIGHT, "bias": BIAS}}


def _node(name: str, op: str, target: str, args: list, **kwargs: object) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs=kwargs)


def _model(*nodes: Node, layer_type: str = "Linear") -> NNModel:
    """Build a network with the layer `layer1` and the nodes as forward pass."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3},
            )
        ],
        forward=list(nodes),
    )


X = _node("x", "placeholder", "x", [])
LAYER1 = _node("layer1", "call_module", "layer1", ["x"])


@pytest.fixture
def scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the function `scale(x, factor=2.0)` for a test."""

    def implementation(backend: Backend, x: np.ndarray, factor: float = 2.0):
        return factor * x

    monkeypatch.setitem(
        FUNCTIONS,
        "scale",
        FunctionType(
            name="scale",
            function=implementation,
            backends=frozenset({BackendKind.NUMPY}),
        ),
    )


@pytest.mark.usefixtures("scale")
@pytest.mark.parametrize("op", ["call_function", "call_method"])
def test_the_nodes_are_evaluated_in_their_order(op: str) -> None:
    """A layer and a function, with a positional and a keyword argument."""
    model = _model(
        X,
        LAYER1,
        _node("scale", op, "scale", ["layer1"]),
        _node("scale_1", op, "scale", ["scale", 5.0]),
        _node("scale_2", op, "scale", ["scale_1"], factor=0.1, inplace=False),
        _node("output", "output", "output", ["scale_2"]),
    )
    x = np.array([0.5, -0.25])
    (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
    np.testing.assert_allclose(y, WEIGHT @ x + BIAS)
    assert y.dtype == float


def test_an_input_is_converted_to_the_array_of_the_backend() -> None:
    """Single precision, integers and lists are inputs."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    expected = WEIGHT @ [1.0, 2.0] + BIAS
    for x in (np.array([1, 2]), np.array([1.0, 2.0], dtype="f4"), [1.0, 2.0]):
        (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
        np.testing.assert_allclose(y, expected)
        assert y.dtype == float


@pytest.mark.usefixtures("scale")
def test_several_outputs() -> None:
    """An output node with a list returns one array per entry."""
    model = _model(
        X,
        LAYER1,
        _node("scale", "call_function", "scale", ["layer1"]),
        _node("output", "output", "output", [["layer1", "scale"]]),
    )
    hidden, y = evaluate(model, ARRAYS, [np.array([0.5, -0.25])], NumpyBackend())
    np.testing.assert_allclose(y, 2.0 * hidden)


@pytest.mark.usefixtures("scale")
def test_the_hook_replaces_the_value_of_a_node() -> None:
    """The hook sees every layer and function, not the inputs and the output."""
    seen: list[str] = []

    def on_node(node: Node, value: np.ndarray) -> np.ndarray:
        seen.append(node.name)
        return np.ones_like(value) if node.name == "layer1" else value

    model = _model(
        X,
        LAYER1,
        _node("scale", "call_function", "scale", ["layer1"]),
        _node("output", "output", "output", ["scale"]),
    )
    (y,) = evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend(), on_node)
    assert seen == ["layer1", "scale"]
    np.testing.assert_array_equal(y, [2.0, 2.0, 2.0])


@pytest.mark.parametrize("n_inputs", [0, 2])
def test_the_number_of_inputs(n_inputs: int) -> None:
    """The inputs are the placeholders of the forward pass."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    with pytest.raises(ValueError, match=rf"{n_inputs} inputs were given for the 1"):
        evaluate(model, ARRAYS, [np.zeros(2)] * n_inputs, NumpyBackend())


def test_an_input_of_the_wrong_size() -> None:
    """The error of numpy names the network and the node."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    with pytest.raises(ValueError, match=r"Network 'net1', node 'layer1'"):
        evaluate(model, ARRAYS, [np.zeros(3)], NumpyBackend())


def test_a_forward_pass_without_an_output() -> None:
    """A forward pass which ends without an output node is an error."""
    with pytest.raises(ValueError, match=r"'net1'.*no output node"):
        evaluate(_model(X, LAYER1), ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_layer_which_the_network_does_not_have() -> None:
    """A node which calls an unknown layer names the layer."""
    model = _model(
        X,
        _node("layer2", "call_module", "layer2", ["x"]),
        _node("output", "output", "output", ["layer2"]),
    )
    with pytest.raises(ValueError, match=r"node 'layer2'.*no layer 'layer2'"):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_an_unknown_opcode() -> None:
    """`get_attr` of `torch.fx` is not part of the NN YAML."""
    model = _model(
        X,
        _node("w", "get_attr", "w", []),
        _node("output", "output", "output", ["w"]),
    )
    with pytest.raises(ValueError, match=r"node 'w'.*'get_attr' is not known"):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_layer_without_an_implementation() -> None:
    """An unknown layer type names the network, the node and the type."""
    model = _model(
        X,
        LAYER1,
        _node("output", "output", "output", ["layer1"]),
        layer_type="LSTM",
    )
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'layer1'.*'LSTM'") as e:
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())
    assert (e.value.network, e.value.node, e.value.target) == (
        "net1",
        "layer1",
        "LSTM",
    )


@pytest.mark.usefixtures("scale")
def test_a_node_which_the_backend_does_not_support(
    monkeypatch: pytest.MonkeyPatch, sympy_backend: Backend
) -> None:
    """A layer and a function declare the backends they are evaluated by."""
    function_model = _model(
        X,
        _node("scale", "call_function", "scale", ["x"]),
        _node("output", "output", "output", ["scale"]),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'scale'.*backend 'sympy'"):
        evaluate(function_model, {}, [np.zeros(2)], sympy_backend)

    layer_model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    evaluate(layer_model, ARRAYS, [np.zeros(2)], sympy_backend)
    monkeypatch.setitem(
        LAYERS,
        "Linear",
        LAYERS["Linear"].__class__(
            name="Linear",
            forward=LAYERS["Linear"].forward,
            arrays=LAYERS["Linear"].arrays,
            backends=NUMPY_ONLY,
        ),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'Linear'.*backend 'sympy'"):
        evaluate(layer_model, ARRAYS, [np.zeros(2)], sympy_backend)


def test_a_keyword_argument_of_a_layer_is_not_dropped() -> None:
    """A `call_module` node with a keyword argument the layer does not have."""
    model = _model(
        X,
        _node("layer1", "call_module", "layer1", ["x"], foo=1),
        _node("output", "output", "output", ["layer1"]),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'layer1'.*'foo'") as e:
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())
    assert (e.value.network, e.value.node, e.value.target) == (
        "net1",
        "layer1",
        "Linear",
    )


def test_the_ignored_keyword_arguments_of_a_layer() -> None:
    """`inplace` and a `dtype` of `None` change no value, as for a function."""
    model = _model(
        X,
        _node("layer1", "call_module", "layer1", ["x"], inplace=False, dtype=None),
        _node("output", "output", "output", ["layer1"]),
    )
    x = np.array([0.5, -0.25])
    (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
    np.testing.assert_allclose(y, WEIGHT @ x + BIAS)


def test_a_function_is_registered_under_every_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decorator registers the function and returns it unchanged."""
    monkeypatch.setattr("sbmlsim.sciml.layers.registry.FUNCTIONS", {})
    from sbmlsim.sciml.layers import registry

    @function("one", "uno", backends=NUMPY_ONLY)
    def one(backend: Backend, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    assert set(registry.FUNCTIONS) == {"one", "uno"}
    assert registry.FUNCTIONS["uno"].function is one
    assert registry.FUNCTIONS["uno"].backends == NUMPY_ONLY
    # the decorator returns the function which was decorated, not a wrapper
    assert one.__name__ == "one"
    np.testing.assert_array_equal(one(NumpyBackend(), np.zeros(2)), [1.0, 1.0])


@pytest.mark.parametrize(
    ("node", "output"),
    [
        (_node("output", "output", "output", ["nope"]), "output"),
        (_node("output", "output", "output", [["x", "nope"]]), "output"),
        (_node("output", "output", "output", [1.0]), "output"),
        (_node("layer1", "call_module", "layer1", ["nope"]), "layer1"),
        (_node("tanh", "call_function", "tanh", ["nope"]), "tanh"),
        (_node("cat", "call_function", "cat", [["x", "nope"]]), "cat"),
    ],
)
def test_a_reference_to_a_node_which_does_not_exist(node: Node, output: str) -> None:
    """An argument which names no evaluated node is an error, not a literal."""
    nodes = [X, node]
    if node.op != "output":
        nodes.append(_node("output", "output", "output", [output]))
    with pytest.raises(
        ValueError,
        match=rf"Network 'net1', node '{node.name}': the argument .* is "
        r"not a node which was evaluated",
    ):
        evaluate(_model(*nodes), ARRAYS, [np.zeros(2)], NumpyBackend())


def test_the_input_of_a_function_is_an_array() -> None:
    """The first argument of a function is the array it is applied to."""
    model = _model(
        X,
        _node("tanh", "call_function", "tanh", [1.0]),
        _node("output", "output", "output", ["tanh"]),
    )
    with pytest.raises(
        ValueError, match=r"Network 'net1', node 'tanh': the input 1.0 is not an array"
    ):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_type_error_of_a_function_names_the_node() -> None:
    """An argument of the wrong type is an error of the node."""
    model = _model(
        X,
        _node("softmax", "call_function", "softmax", ["x"], dim="a"),
        _node("output", "output", "output", ["softmax"]),
    )
    with pytest.raises(ValueError, match=r"Network 'net1', node 'softmax'") as e:
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())
    assert isinstance(e.value.__cause__, TypeError)


def test_an_output_node_without_an_argument() -> None:
    """The output node names the outputs."""
    model = _model(X, LAYER1, _node("output", "output", "output", []))
    with pytest.raises(
        ValueError, match=r"Network 'net1', node 'output': .*0 arguments, expected one"
    ):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_layer_with_the_wrong_number_of_inputs() -> None:
    """The inputs of a layer are bound to its implementation."""
    model = _model(
        X,
        _node("layer1", "call_module", "layer1", ["x", "x"]),
        _node("output", "output", "output", ["layer1"]),
    )
    with pytest.raises(
        UnsupportedLayerError, match=r"'net1', node 'layer1': 'Linear'.*inputs"
    ):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_literal_which_is_the_name_of_a_node() -> None:
    """Only the array arguments of a function are resolved to nodes."""
    model = _model(
        X,
        _node("tanh", "call_function", "tanh", ["x"]),
        _node("gelu", "call_function", "gelu", ["tanh", "tanh"]),
        _node("output", "output", "output", ["gelu"]),
    )
    x = np.array([0.5, -0.25])
    (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
    t = np.tanh(x)
    inner = np.sqrt(2.0 / np.pi) * (t + 0.044715 * t**3)
    np.testing.assert_allclose(y, 0.5 * t * (1.0 + np.tanh(inner)))


def test_the_value_of_a_node_is_an_array() -> None:
    """A function of an array without axes gives an array, not a scalar."""
    model = _model(
        X,
        _node("tanh", "call_function", "tanh", ["x"]),
        _node("output", "output", "output", ["tanh"]),
    )
    (y,) = evaluate(model, ARRAYS, [np.array(0.5)], NumpyBackend())
    assert type(y) is np.ndarray
    assert y.shape == ()


@pytest.mark.parametrize(
    "node",
    [
        _node("layer1", "call_module", "layer1", ["x"]),
        _node("flatten", "call_function", "flatten", ["x"]),
    ],
)
def test_an_output_does_not_share_memory_with_an_input(node: Node) -> None:
    """Writing into an output does not change the input."""
    model = _model(
        X, node, _node("output", "output", "output", [node.name]), layer_type="Dropout"
    )
    x = np.array([[0.5, -0.25]])
    (y,) = evaluate(model, {}, [x], NumpyBackend())
    assert not np.shares_memory(x, y)
    y[...] = 1.0
    np.testing.assert_array_equal(x, [[0.5, -0.25]])


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_the_input_of_every_function_takes_arrays(name: str) -> None:
    """The first parameter after the backend is resolved to nodes."""
    implementation = FUNCTIONS[name].function
    first = list(inspect.signature(implementation).parameters)[1]
    assert first in _array_parameters(implementation)
