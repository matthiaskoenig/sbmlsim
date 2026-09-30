"""The activation functions and tensor operations against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import Input, NNModel, Node

from sbmlsim.sciml import UnsupportedLayerError
from sbmlsim.sciml.backend import ALL_BACKENDS
from sbmlsim.sciml.layers import FUNCTIONS

#: values on both sides of every threshold of the piecewise functions, the
#: thresholds themselves and values which overflow a naive exponential
VALUES = np.array(
    [-800.0, -30.0, -6.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0, 20.0, 21.0, 800.0]
)

FUNCTION_CASES: list[tuple[str, dict[str, Any]]] = [
    ("tanh", {}),
    ("sigmoid", {}),
    ("relu", {}),
    ("relu", {"inplace": False}),
    ("relu6", {"inplace": False}),
    ("hardtanh", {}),
    ("hardtanh", {"min_val": -2.0, "max_val": 0.5, "inplace": False}),
    ("hardsigmoid", {"inplace": False}),
    ("hardswish", {"inplace": False}),
    ("leaky_relu", {}),
    ("leaky_relu", {"negative_slope": 0.2, "inplace": False}),
    ("elu", {}),
    ("elu", {"alpha": 2.0, "inplace": False}),
    ("celu", {}),
    ("celu", {"alpha": 2.0, "inplace": False}),
    ("selu", {"inplace": False}),
    ("gelu", {}),
    ("gelu", {"approximate": "none"}),
    ("gelu", {"approximate": "tanh"}),
    ("softplus", {}),
    ("softplus", {"beta": 2.0, "threshold": 5.0}),
    ("mish", {"inplace": False}),
    ("silu", {"inplace": False}),
    ("softsign", {}),
    ("tanhshrink", {}),
    ("softmax", {"dim": 0}),
    ("softmax", {"dim": -1, "_stacklevel": 3, "dtype": None}),
    ("log_softmax", {"dim": 0}),
    ("log_softmax", {"dim": -1, "_stacklevel": 3, "dtype": None}),
    ("flatten", {}),
    ("flatten", {"start_dim": 1, "end_dim": -1}),
]


@pytest.mark.parametrize(("target", "kwargs"), FUNCTION_CASES)
def test_function(
    compare_function: Callable[..., None], target: str, kwargs: dict[str, Any]
) -> None:
    """A function has the values of the function of PyTorch."""
    x = np.stack([VALUES, VALUES[::-1], 0.1 * VALUES])
    compare_function(target, kwargs, x)


def test_log_sigmoid(compare_function: Callable[..., None]) -> None:
    """`log_sigmoid` of the YAML is `logsigmoid` of PyTorch."""
    compare_function("log_sigmoid", {}, VALUES, torch_name="logsigmoid")
    compare_function("logsigmoid", {}, VALUES)


@pytest.mark.parametrize("target", ["tanh", "sigmoid", "relu", "flatten"])
def test_method(compare_function: Callable[..., None], target: str) -> None:
    """A method of a tensor is the function of the same name."""
    compare_function(target, {}, np.stack([VALUES, VALUES]), op="call_method")


def test_every_function_is_tested() -> None:
    """The functions which are registered are the ones which are compared."""
    tested = {target for target, _ in FUNCTION_CASES}
    tested |= {"log_sigmoid", "logsigmoid", "cat", "concat", "concatenate"}
    assert set(FUNCTIONS) == tested


def test_the_functions_support_both_backends() -> None:
    """Every function is written with the methods of the backend."""
    assert all(f.backends == ALL_BACKENDS for f in FUNCTIONS.values())


def test_the_values_are_finite(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """No function overflows on a large input."""
    for target in ("sigmoid", "softplus", "log_sigmoid", "mish", "silu", "elu"):
        (y,) = forward(function_model(target, {}), {}, VALUES)
        assert np.all(np.isfinite(y)), target


def _two_inputs(node: Node) -> NNModel:
    """Build a network of one node with the inputs `a` and `b`."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0"), Input(input_id="input1")],
        layers=[],
        forward=[
            Node(name="a", op="placeholder", target="a", args=[], kwargs={}),
            Node(name="b", op="placeholder", target="b", args=[], kwargs={}),
            node,
            Node(
                name="output", op="output", target="output", args=[node.name], kwargs={}
            ),
        ],
    )


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat(forward: Callable[..., tuple[np.ndarray, ...]], dim: int) -> None:
    """`cat` joins the arrays of a list along an axis."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=1)
    a, b = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    node = Node(
        name="cat",
        op="call_function",
        target="cat",
        args=[["a", "b"]],
        kwargs={"dim": dim},
    )
    (observed,) = forward(_two_inputs(node), {}, a, b)
    expected = torch.cat([torch.from_numpy(a), torch.from_numpy(b)], dim=dim).numpy()
    np.testing.assert_array_equal(observed, expected)


def test_cat_of_arrays_which_do_not_fit(
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """The error of numpy names the network and the node."""
    node = Node(
        name="cat", op="call_function", target="cat", args=[["a", "b"]], kwargs={}
    )
    with pytest.raises(ValueError, match=r"Network 'net1', node 'cat'"):
        forward(_two_inputs(node), {}, np.zeros((2, 3)), np.zeros((2, 4)))


def test_a_keyword_argument_is_a_literal(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """`approximate="tanh"` is not the node which is named `tanh`."""
    model = function_model("gelu", {"approximate": "tanh"})
    model.forward[0].name = "tanh"
    model.forward[1].args = ["tanh"]
    (y,) = forward(model, {}, VALUES)
    (expected,) = forward(function_model("gelu", {"approximate": "tanh"}), {}, VALUES)
    np.testing.assert_array_equal(y, expected)


def test_a_function_without_an_implementation(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """An unknown function names the network, the node and the function."""
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'f'.*'rrelu'") as info:
        forward(function_model("rrelu", {}), {}, VALUES)
    assert (info.value.network, info.value.node, info.value.target) == (
        "net1",
        "f",
        "rrelu",
    )


@pytest.mark.parametrize(
    ("target", "kwargs"),
    [
        ("relu", {"threshold": 1.0}),
        ("softmax", {}),
        ("gelu", {"approximate": "sigmoid"}),
    ],
)
def test_arguments_which_do_not_fit(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    target: str,
    kwargs: dict[str, Any],
) -> None:
    """An argument the function does not have is an error, it is not dropped."""
    with pytest.raises((UnsupportedLayerError, ValueError), match=r"node 'f'"):
        forward(function_model(target, kwargs), {}, VALUES)
