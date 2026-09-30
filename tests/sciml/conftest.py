"""Fixtures of the tests of the neural networks.

The forward pass of `sbmlsim` is compared with PyTorch, for every layer and
function and with random arrays. `torch` is only in the `dev` extra, the
tests which need it are skipped without it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import sympy
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml.backend import NumpyBackend, SympyBackend
from sbmlsim.sciml.interpreter import evaluate

#: absolute and relative tolerance of the comparison with PyTorch, which runs
#: in double precision
TOLERANCE = 1e-10


def symbols(prefix: str, shape: tuple[int, ...]) -> np.ndarray:
    """Get an `object` array of symbols, named by the prefix and the index."""
    array = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        array[index] = sympy.Symbol("_".join([prefix, *map(str, index)]))
    return array


def build_layer_model(
    layer_type: str, args: dict[str, Any], n_inputs: int = 1
) -> NNModel:
    """Build the network `net1` which is the single layer `layer1`."""
    names = [f"net_input{k}" for k in range(n_inputs)]
    forward = [
        Node(name=name, op="placeholder", target=name, args=[], kwargs={})
        for name in names
    ]
    forward.append(
        Node(name="layer1", op="call_module", target="layer1", args=names, kwargs={})
    )
    forward.append(
        Node(name="output", op="output", target="output", args=["layer1"], kwargs={})
    )
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id=f"input{k}") for k in range(n_inputs)],
        layers=[Layer(layer_id="layer1", layer_type=layer_type, args=args)],
        forward=forward,
    )


def build_function_model(
    target: str, kwargs: dict[str, Any], op: str = "call_function"
) -> NNModel:
    """Build the network `net1` which is the single function `f` of its input."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[],
        forward=[
            Node(
                name="net_input",
                op="placeholder",
                target="net_input",
                args=[],
                kwargs={},
            ),
            Node(name="f", op=op, target=target, args=["net_input"], kwargs=kwargs),
            Node(name="output", op="output", target="output", args=["f"], kwargs={}),
        ],
    )


def run_forward(
    model: NNModel,
    parameters: dict[str, dict[str, np.ndarray]],
    *inputs: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Evaluate a network with numpy."""
    return evaluate(model, parameters, inputs, NumpyBackend())


@pytest.fixture
def layer_model() -> Callable[..., NNModel]:
    """Get the function which builds a network of a single layer."""
    return build_layer_model


@pytest.fixture
def function_model() -> Callable[..., NNModel]:
    """Get the function which builds a network of a single function."""
    return build_function_model


@pytest.fixture
def forward() -> Callable[..., tuple[np.ndarray, ...]]:
    """Get the function which evaluates a network with numpy."""
    return run_forward


@pytest.fixture
def rng() -> np.random.Generator:
    """Get a seeded random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def compare_layer(rng: np.random.Generator) -> Callable[..., None]:
    """Get the comparison of a layer with the layer of PyTorch.

    The arrays of the layer, the running statistics of a normalization layer
    and the inputs are random. PyTorch evaluates in evaluation mode.
    """
    torch = pytest.importorskip("torch")

    def compare(
        layer_type: str, args: dict[str, Any], *shapes: tuple[int, ...]
    ) -> None:
        module = getattr(torch.nn, layer_type)(**args).double()
        state = {}
        for name, tensor in module.state_dict().items():
            if name == "num_batches_tracked":
                state[name] = tensor
            elif name == "running_var":
                state[name] = torch.from_numpy(rng.uniform(0.5, 2.0, tensor.shape))
            else:
                state[name] = torch.from_numpy(rng.normal(size=tuple(tensor.shape)))
        module.load_state_dict(state)
        module.eval()

        parameters = {
            "layer1": {
                name: tensor.numpy().copy()
                for name, tensor in state.items()
                if name != "num_batches_tracked"
            }
        }
        inputs = [rng.normal(size=shape) for shape in shapes]
        with torch.no_grad():
            expected = module(*(torch.from_numpy(x) for x in inputs)).numpy()

        model = build_layer_model(layer_type, args, n_inputs=len(shapes))
        (observed,) = run_forward(model, parameters, *inputs)
        assert observed.shape == expected.shape
        np.testing.assert_allclose(observed, expected, rtol=TOLERANCE, atol=TOLERANCE)

    return compare


@pytest.fixture
def compare_function() -> Callable[..., None]:
    """Get the comparison of a function with the function of PyTorch."""
    torch = pytest.importorskip("torch")

    def compare(
        target: str,
        kwargs: dict[str, Any],
        x: np.ndarray,
        torch_name: str | None = None,
        op: str = "call_function",
    ) -> None:
        name = target if torch_name is None else torch_name
        reference = getattr(torch.nn.functional, name, None) or getattr(torch, name)
        expected = reference(torch.from_numpy(x), **kwargs).numpy()

        model = build_function_model(target, kwargs, op=op)
        (observed,) = run_forward(model, {}, x)
        assert observed.shape == expected.shape
        np.testing.assert_allclose(observed, expected, rtol=TOLERANCE, atol=TOLERANCE)

    return compare


@pytest.fixture
def sympy_backend() -> SympyBackend:
    """Get the backend on sympy expressions."""
    return SympyBackend()


@pytest.fixture
def symbolic() -> Callable[[str, tuple[int, ...]], np.ndarray]:
    """Get the function which builds an array of symbols."""
    return symbols
