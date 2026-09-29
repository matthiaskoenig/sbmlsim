"""Tests of the interpreter of the forward pass."""

from collections.abc import Callable

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import UnsupportedLayerError
from sbmlsim.sciml.backend import NUMPY_ONLY, Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.interpreter import evaluate
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


class ExpressionBackend(NumpyBackend):
    """A backend which says that it is the one on expressions."""

    kind = BackendKind.SYMPY


@pytest.mark.usefixtures("scale")
def test_a_node_which_the_backend_does_not_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layer and a function declare the backends they are evaluated by."""
    function_model = _model(
        X,
        _node("scale", "call_function", "scale", ["x"]),
        _node("output", "output", "output", ["scale"]),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'scale'.*backend 'sympy'"):
        evaluate(function_model, {}, [np.zeros(2)], ExpressionBackend())

    layer_model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    evaluate(layer_model, ARRAYS, [np.zeros(2)], ExpressionBackend())
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
        evaluate(layer_model, ARRAYS, [np.zeros(2)], ExpressionBackend())


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
    assert isinstance(one, Callable)
