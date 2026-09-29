"""The interpreter of the forward pass of a network.

The forward pass of the NN YAML is the list of the nodes of a `torch.fx`
graph: `placeholder` (an input), `call_module` (a layer), `call_function` and
`call_method` (a function), and `output`. `evaluate` walks the list once with
a backend, so the forward pass in numpy and the compilation into expressions
are the same interpreter and the same layers.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from petab_sciml import NNModel, Node

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.errors import UnsupportedLayerError
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS

#: the opcodes of the nodes
PLACEHOLDER = "placeholder"
CALL_MODULE = "call_module"
CALL_FUNCTION = "call_function"
CALL_METHOD = "call_method"
OUTPUT = "output"

#: keyword arguments of the functions of PyTorch which do not change a value
IGNORED_KWARGS: frozenset[str] = frozenset({"inplace", "_stacklevel"})

#: called with a node and its value, returns the value the next nodes see
NodeHook = Callable[[Node, np.ndarray], np.ndarray]


def _resolve(value: Any, state: Mapping[str, np.ndarray]) -> Any:
    """Replace the names of nodes in an argument by the values of the nodes.

    Args:
        value: an argument of a node: the name of a node, a list of arguments
            or a literal.
        state: the values of the nodes which were evaluated.

    Returns:
        The argument with the values of the nodes.
    """
    if isinstance(value, (list, tuple)):
        return [_resolve(v, state) for v in value]
    if isinstance(value, str) and value in state:
        return state[value]
    return value


def _call_function(
    network: str, node: Node, backend: Backend, args: list[Any], kwargs: dict[str, Any]
) -> np.ndarray:
    """Evaluate a `call_function` or `call_method` node.

    Args:
        network: id of the network.
        node: the node.
        backend: the backend.
        args: the resolved positional arguments.
        kwargs: the keyword arguments.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the function is not implemented, not
            available in the backend or called with an argument the
            implementation does not have.
    """
    function_type = FUNCTIONS.get(node.target)
    if function_type is None:
        raise UnsupportedLayerError(
            network, node.name, node.target, "the function is not implemented"
        )
    if backend.kind not in function_type.backends:
        raise UnsupportedLayerError(
            network,
            node.name,
            node.target,
            f"the function is not available in the backend '{backend.kind}'",
        )
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in IGNORED_KWARGS and not (key == "dtype" and value is None)
    }
    try:
        inspect.signature(function_type.function).bind(backend, *args, **kwargs)
    except TypeError as err:
        raise UnsupportedLayerError(
            network, node.name, node.target, f"the arguments do not fit: {err}"
        ) from err
    return function_type.function(backend, *args, **kwargs)


def evaluate(
    model: NNModel,
    parameters: Mapping[str, Mapping[str, np.ndarray]],
    inputs: Sequence[ArrayLike],
    backend: Backend,
    on_node: NodeHook | None = None,
) -> tuple[np.ndarray, ...]:
    """Evaluate the forward pass of a network.

    Args:
        model: the architecture of the network.
        parameters: the arrays of the layers, layer id -> array name -> array,
            in the PyTorch layout.
        inputs: the inputs, one per `placeholder` node in the order of the
            nodes.
        backend: the backend the layers and functions are evaluated with.
        on_node: called after every layer and function with the node and its
            value, the next nodes see what it returns. The compilation uses it
            to replace the expressions of a node by symbols.

    Returns:
        The outputs of the network.

    Raises:
        UnsupportedLayerError: if a layer or function is not implemented or
            not available in the backend.
        ValueError: if the number of inputs is not the number of placeholders,
            if a node has an unknown opcode or if a layer cannot be evaluated
            on its input; the message names the network and the node.
    """
    network = model.nn_model_id
    layers = {layer.layer_id: layer for layer in model.layers}
    placeholders = [node for node in model.forward if node.op == PLACEHOLDER]
    if len(placeholders) != len(inputs):
        raise ValueError(
            f"Network '{network}': {len(inputs)} inputs were given for the "
            f"{len(placeholders)} inputs {[node.name for node in placeholders]}"
        )

    state: dict[str, np.ndarray] = {}
    remaining = iter(inputs)
    for node in model.forward:
        args = _resolve(node.args or [], state)
        # the keyword arguments are literals: `petab_sciml` writes the nodes a
        # function is called with as positional arguments only
        kwargs = dict(node.kwargs or {})

        if node.op == PLACEHOLDER:
            state[node.name] = backend.asarray(next(remaining))
            continue
        if node.op == OUTPUT:
            output = args[0]
            return tuple(output) if isinstance(output, list) else (output,)

        try:
            if node.op == CALL_MODULE:
                value = _call_module(network, node, layers, parameters, backend, args)
            elif node.op in (CALL_FUNCTION, CALL_METHOD):
                value = _call_function(network, node, backend, args, kwargs)
            else:
                raise ValueError(f"the opcode '{node.op}' is not known")
        except UnsupportedLayerError:
            raise
        except ValueError as err:
            raise ValueError(f"Network '{network}', node '{node.name}': {err}") from err

        state[node.name] = value if on_node is None else on_node(node, value)

    raise ValueError(f"Network '{network}': the forward pass has no output node")


def _call_module(
    network: str,
    node: Node,
    layers: Mapping[str, Any],
    parameters: Mapping[str, Mapping[str, np.ndarray]],
    backend: Backend,
    args: list[Any],
) -> np.ndarray:
    """Evaluate a `call_module` node, i.e. a layer.

    Args:
        network: id of the network.
        node: the node.
        layers: the layers of the network by their id.
        parameters: the arrays of the layers.
        backend: the backend.
        args: the resolved positional arguments, i.e. the inputs of the layer.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the layer type is not implemented or not
            available in the backend.
        ValueError: if the network has no layer of the id.
    """
    if node.target not in layers:
        raise ValueError(f"the network has no layer '{node.target}'")
    layer = layers[node.target]
    layer_type = LAYERS.get(layer.layer_type)
    if layer_type is None:
        raise UnsupportedLayerError(
            network, node.name, layer.layer_type, "the layer is not implemented"
        )
    if backend.kind not in layer_type.backends:
        raise UnsupportedLayerError(
            network,
            node.name,
            layer.layer_type,
            f"the layer is not available in the backend '{backend.kind}'",
        )
    arrays = {
        name: backend.asarray(array)
        for name, array in parameters.get(layer.layer_id, {}).items()
    }
    return layer_type.forward(backend, layer.args or {}, arrays, *args)
