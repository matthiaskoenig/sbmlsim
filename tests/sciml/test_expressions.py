"""The layers and functions of both backends evaluated on expressions.

A layer or function which declares `BackendKind.SYMPY` is evaluated with the
backend on sympy expressions, on symbols for its inputs and its arrays. The
expressions, evaluated at random values, equal the forward pass with numpy.
This is the claim the compilation of a network into an SBML model rests on.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import sympy
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml.backend import Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.interpreter import evaluate
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS
from sbmlsim.sciml.network import Network

#: the relative and absolute tolerance of the expressions against numpy
TOLERANCE = 1e-12

#: arguments and input shapes of every layer of both backends
LAYER_CASES: dict[str, tuple[dict[str, Any], list[tuple[int, ...]]]] = {
    "Linear": ({"in_features": 3, "out_features": 2}, [(2, 3)]),
    "Bilinear": (
        {"in1_features": 2, "in2_features": 3, "out_features": 2},
        [(2, 2), (2, 3)],
    ),
    "Flatten": ({}, [(2, 3, 2)]),
    "Dropout": ({"p": 0.5}, [(2, 3)]),
    "Dropout1d": ({"p": 0.5}, [(2, 3)]),
    "Dropout2d": ({"p": 0.5}, [(2, 3, 2)]),
    "Dropout3d": ({"p": 0.5}, [(2, 3, 2, 2)]),
    "AlphaDropout": ({"p": 0.5}, [(2, 3)]),
    "FeatureAlphaDropout": ({"p": 0.5}, [(2, 3)]),
}

#: keyword arguments and input shapes of every function of both backends; a
#: function with two input shapes takes the list of the inputs
FUNCTION_CASES: dict[str, tuple[dict[str, Any], list[tuple[int, ...]]]] = {
    "tanh": ({}, [(4,)]),
    "sigmoid": ({}, [(4,)]),
    "relu": ({}, [(4,)]),
    "relu6": ({}, [(4,)]),
    "hardtanh": ({"min_val": -0.5, "max_val": 0.75}, [(4,)]),
    "hardsigmoid": ({}, [(4,)]),
    "hardswish": ({}, [(4,)]),
    "leaky_relu": ({"negative_slope": 0.2}, [(4,)]),
    "elu": ({"alpha": 0.5}, [(4,)]),
    "celu": ({"alpha": 0.5}, [(4,)]),
    "selu": ({}, [(4,)]),
    "gelu": ({"approximate": "tanh"}, [(4,)]),
    "softplus": ({"beta": 2.0}, [(4,)]),
    "log_sigmoid": ({}, [(4,)]),
    "logsigmoid": ({}, [(4,)]),
    "mish": ({}, [(4,)]),
    "silu": ({}, [(4,)]),
    "softsign": ({}, [(4,)]),
    "tanhshrink": ({}, [(4,)]),
    "softmax": ({"dim": 1}, [(2, 3)]),
    "log_softmax": ({"dim": 0}, [(2, 3)]),
    "flatten": ({"start_dim": 1}, [(2, 3, 2)]),
    "cat": ({"dim": 1}, [(2, 3), (2, 1)]),
    "concat": ({"dim": 0}, [(2, 3), (1, 3)]),
    "concatenate": ({}, [(2,), (3,)]),
}


def _of_both_backends(registry: dict[str, Any]) -> list[str]:
    return sorted(
        name for name, entry in registry.items() if BackendKind.SYMPY in entry.backends
    )


def _placeholders(n: int) -> list[Node]:
    return [
        Node(name=f"x{k}", op="placeholder", target=f"x{k}", args=[], kwargs={})
        for k in range(n)
    ]


def _output(name: str) -> Node:
    return Node(name="output", op="output", target="output", args=[name], kwargs={})


def _compare(
    model: NNModel,
    parameters: dict[str, dict[str, np.ndarray]],
    inputs: list[np.ndarray],
    sympy_backend: Backend,
    rng: np.random.Generator,
) -> None:
    """Evaluate on expressions and at random values, and compare with numpy."""
    symbols = [s for x in inputs for s in x.flat]
    symbols += [
        s for arrays in parameters.values() for a in arrays.values() for s in a.flat
    ]
    (expressions,) = evaluate(model, parameters, inputs, sympy_backend)
    assert expressions.dtype == object

    values = dict(zip(symbols, rng.normal(size=len(symbols)), strict=True))

    def numbers(array: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda s: values[s], otypes=[float])(array)

    numeric = {
        layer: {name: numbers(array) for name, array in arrays.items()}
        for layer, arrays in parameters.items()
    }
    (expected,) = evaluate(model, numeric, [numbers(x) for x in inputs], NumpyBackend())
    assert expressions.shape == expected.shape

    observed = np.empty(expressions.shape)
    for index in np.ndindex(expressions.shape):
        function = sympy.lambdify(symbols, sympy.sympify(expressions[index]), "math")
        observed[index] = function(*(values[s] for s in symbols))
    np.testing.assert_allclose(observed, expected, rtol=TOLERANCE, atol=TOLERANCE)


def test_every_entry_of_both_backends_has_a_case() -> None:
    """A layer or function which declares both backends is tested here."""
    assert _of_both_backends(LAYERS) == sorted(LAYER_CASES)
    assert _of_both_backends(FUNCTIONS) == sorted(FUNCTION_CASES)


@pytest.mark.parametrize("layer_type", _of_both_backends(LAYERS))
def test_a_layer_on_expressions(
    layer_type: str,
    sympy_backend: Backend,
    symbolic: Callable[[str, tuple[int, ...]], np.ndarray],
    rng: np.random.Generator,
) -> None:
    """The expressions of a layer equal its forward pass with numpy."""
    args, shapes = LAYER_CASES[layer_type]
    names = [f"x{k}" for k in range(len(shapes))]
    model = NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id=f"input{k}") for k in range(len(shapes))],
        layers=[Layer(layer_id="layer1", layer_type=layer_type, args=args)],
        forward=[
            *_placeholders(len(shapes)),
            Node(
                name="layer1", op="call_module", target="layer1", args=names, kwargs={}
            ),
            _output("layer1"),
        ],
    )
    parameters = {
        "layer1": {
            name: symbolic(name, spec.shape)
            for name, spec in LAYERS[layer_type].arrays(args).items()
        }
    }
    inputs = [symbolic(f"x{k}", shape) for k, shape in enumerate(shapes)]
    _compare(model, parameters, inputs, sympy_backend, rng)


@pytest.mark.parametrize("target", _of_both_backends(FUNCTIONS))
def test_a_function_on_expressions(
    target: str,
    sympy_backend: Backend,
    symbolic: Callable[[str, tuple[int, ...]], np.ndarray],
    rng: np.random.Generator,
) -> None:
    """The expressions of a function equal its forward pass with numpy."""
    kwargs, shapes = FUNCTION_CASES[target]
    names = [f"x{k}" for k in range(len(shapes))]
    model = NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id=f"input{k}") for k in range(len(shapes))],
        layers=[],
        forward=[
            *_placeholders(len(shapes)),
            Node(
                name="f",
                op="call_function",
                target=target,
                args=[names] if len(names) > 1 else names,
                kwargs=kwargs,
            ),
            _output("f"),
        ],
    )
    inputs = [symbolic(f"x{k}", shape) for k, shape in enumerate(shapes)]
    _compare(model, {}, inputs, sympy_backend, rng)


def test_a_network_is_compiled_node_by_node(
    sympy_backend: Backend,
    symbolic: Callable[[str, tuple[int, ...]], np.ndarray],
    rng: np.random.Generator,
) -> None:
    """`on_node` replaces the expressions of a node by one symbol per unit.

    The rules of `Linear`-`tanh`-`Linear`, one per unit and one layer deep,
    evaluated one after the other, give the forward pass of the network.
    """
    model = NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Linear",
                args={"in_features": 2, "out_features": 3},
            ),
            Layer(
                layer_id="layer2",
                layer_type="Linear",
                args={"in_features": 3, "out_features": 2},
            ),
        ],
        forward=[
            *_placeholders(1),
            Node(
                name="layer1", op="call_module", target="layer1", args=["x0"], kwargs={}
            ),
            Node(
                name="tanh",
                op="call_function",
                target="tanh",
                args=["layer1"],
                kwargs={},
            ),
            Node(
                name="layer2",
                op="call_module",
                target="layer2",
                args=["tanh"],
                kwargs={},
            ),
            _output("layer2"),
        ],
    )
    network = Network(
        sid="net1",
        model=model,
        parameters={
            "layer1": {"weight": rng.normal(size=(3, 2)), "bias": rng.normal(size=3)},
            "layer2": {"weight": rng.normal(size=(2, 3)), "bias": rng.normal(size=2)},
        },
    )
    parameters: dict[str, dict[str, np.ndarray]] = {}
    for sid, (layer, name, index) in network.parameter_ids().items():
        shape = network.array_specs()[layer][name].shape
        array = parameters.setdefault(layer, {}).setdefault(
            name, np.empty(shape, dtype=object)
        )
        array[index] = sympy.Symbol(sid)

    rules: dict[sympy.Symbol, sympy.Expr] = {}
    seen: list[str] = []

    def on_node(node: Node, value: np.ndarray) -> np.ndarray:
        seen.append(node.name)
        units = symbolic(f"net1__{node.name}", value.shape)
        for index in np.ndindex(value.shape):
            rules[units[index]] = value[index]
        return units

    x = symbolic("net1__x0", (2,))
    (outputs,) = evaluate(network.model, parameters, [x], sympy_backend, on_node)
    assert seen == ["layer1", "tanh", "layer2"]
    assert len(rules) == 3 + 3 + 2
    # one layer deep: a rule only names the units of the node before it
    assert rules[sympy.Symbol("net1__tanh_0")] == sympy.tanh(
        sympy.Symbol("net1__layer1_0")
    )

    point = rng.normal(size=2)
    values: dict[Any, Any] = {
        sympy.Symbol(sid): float(network.parameters[layer][name][index])
        for sid, (layer, name, index) in network.parameter_ids().items()
    }
    values.update({s: float(v) for s, v in zip(x, point, strict=True)})
    for unit, rule in rules.items():
        values[unit] = float(rule.subs(values))
    observed = [values[unit] for unit in outputs]
    np.testing.assert_allclose(
        observed, network.forward(point)[0], rtol=TOLERANCE, atol=TOLERANCE
    )
