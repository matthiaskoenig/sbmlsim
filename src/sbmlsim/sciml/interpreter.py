"""The interpreter of the forward pass of a network.

The forward pass of the NN YAML is the list of the nodes of a `torch.fx`
graph: `placeholder` (an input), `call_module` (a layer), `call_function` and
`call_method` (a function), and `output`. `evaluate` walks the list once with
a backend, so the forward pass in numpy and the compilation into expressions
are the same interpreter and the same layers.

The NN YAML writes a reference to a node as the name of the node, so a
reference and a string literal look the same. The arguments of a layer and of
an output are references only. The arguments of a function are references
where the implementation takes arrays, i.e. where its parameter is annotated
with `numpy.ndarray` (or a type which holds it, e.g. a sequence of arrays),
and literals everywhere else, so `gelu(x, "tanh")` stays the literal `"tanh"`
in a network with a node named `tanh`.
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from petab_sciml import Layer, NNModel, Node

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


def _references(value: Any, state: Mapping[str, np.ndarray]) -> Any:
    """Replace the names of nodes by the values of the nodes.

    Args:
        value: the name of a node or a list of names.
        state: the values of the nodes which were evaluated.

    Returns:
        The value of the node, or the list of the values.

    Raises:
        ValueError: if a name is not a node which was evaluated, or the value
            is neither a name nor a list.
    """
    if isinstance(value, (list, tuple)):
        return [_references(v, state) for v in value]
    if isinstance(value, str) and value in state:
        return state[value]
    raise ValueError(f"the argument {value!r} is not a node which was evaluated")


def _array_references(value: Any, state: Mapping[str, np.ndarray]) -> Any:
    """Resolve the argument of a function which takes arrays.

    A string is the name of a node, a number is a literal.

    Args:
        value: the argument: the name of a node, a literal or a list of them.
        state: the values of the nodes which were evaluated.

    Returns:
        The argument with the values of the nodes.

    Raises:
        ValueError: if a string is not a node which was evaluated.
    """
    if isinstance(value, (list, tuple)):
        return [_array_references(v, state) for v in value]
    if isinstance(value, str):
        return _references(value, state)
    return value


def _is_array_type(hint: Any) -> bool:
    """Check whether a type is an array or holds one, e.g. `Sequence[ndarray]`."""
    return hint is np.ndarray or any(
        _is_array_type(arg) for arg in typing.get_args(hint)
    )


@functools.cache
def _signature(implementation: Callable[..., np.ndarray]) -> inspect.Signature:
    """Get the signature of the implementation of a layer or a function."""
    return inspect.signature(implementation)


@functools.cache
def _array_parameters(implementation: Callable[..., np.ndarray]) -> frozenset[str]:
    """Get the parameters of a function which take arrays.

    Args:
        implementation: the implementation of the function.

    Returns:
        The names of the parameters whose annotation holds `numpy.ndarray`.
    """
    hints = typing.get_type_hints(implementation)
    return frozenset(
        name
        for name, hint in hints.items()
        if name != "return" and _is_array_type(hint)
    )


def _is_array_input(value: Any) -> bool:
    """Check whether a value is an array or a list of arrays."""
    if isinstance(value, list):
        return bool(value) and all(isinstance(v, np.ndarray) for v in value)
    return isinstance(value, np.ndarray)


def _effective_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the keyword arguments of a node which do not change a value.

    Args:
        kwargs: the keyword arguments of the node.

    Returns:
        The keyword arguments without `inplace`, `_stacklevel` and a `dtype`
        of `None`.
    """
    return {
        key: value
        for key, value in kwargs.items()
        if key not in IGNORED_KWARGS and not (key == "dtype" and value is None)
    }


def _call_function(
    network: str,
    node: Node,
    backend: Backend,
    state: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a `call_function` or `call_method` node.

    Args:
        network: id of the network.
        node: the node.
        backend: the backend.
        state: the values of the nodes which were evaluated.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the function is not implemented, not
            available in the backend or called with an argument the
            implementation does not have.
        ValueError: if an argument which takes arrays names no node which was
            evaluated, or if the input of the function is not an array.
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
    implementation = function_type.function
    kwargs = _effective_kwargs(node.kwargs or {})
    try:
        bound = _signature(implementation).bind(backend, *(node.args or []), **kwargs)
    except TypeError as err:
        raise UnsupportedLayerError(
            network, node.name, node.target, f"the arguments do not fit: {err}"
        ) from err
    arrays = _array_parameters(implementation)
    for name, value in bound.arguments.items():
        if name in arrays:
            bound.arguments[name] = _array_references(value, state)
    # the first parameter is the backend, the second the input
    values = list(bound.arguments.values())
    if len(values) < 2 or not _is_array_input(values[1]):
        given = values[1] if len(values) > 1 else None
        raise ValueError(f"the input {given!r} is not an array or a list of arrays")
    return implementation(*bound.args, **bound.kwargs)


def _call_module(
    network: str,
    node: Node,
    layers: Mapping[str, Layer],
    parameters: Mapping[str, Mapping[str, np.ndarray]],
    backend: Backend,
    state: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a `call_module` node, i.e. a layer.

    Args:
        network: id of the network.
        node: the node.
        layers: the layers of the network by their id.
        parameters: the arrays of the layers.
        backend: the backend.
        state: the values of the nodes which were evaluated.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the layer type is not implemented, not
            available in the backend, the node has a keyword argument or the
            number of inputs does not fit the implementation.
        ValueError: if the network has no layer of the id or an input is not
            a node which was evaluated.
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
    if unknown := _effective_kwargs(node.kwargs or {}):
        raise UnsupportedLayerError(
            network,
            node.name,
            layer.layer_type,
            f"the layer does not take the keyword arguments {sorted(unknown)}",
        )
    inputs = [_references(arg, state) for arg in node.args or []]
    args = layer.args or {}
    arrays = {
        name: backend.asarray(array)
        for name, array in parameters.get(layer.layer_id, {}).items()
    }
    try:
        _signature(layer_type.forward).bind(backend, args, arrays, *inputs)
    except TypeError as err:
        raise UnsupportedLayerError(
            network,
            node.name,
            layer.layer_type,
            f"the inputs do not fit: {len(inputs)} inputs were given, {err}",
        ) from err
    return layer_type.forward(backend, args, arrays, *inputs)


def _outputs(
    node: Node, state: Mapping[str, np.ndarray], inputs: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Get the outputs of the network from the output node.

    Args:
        node: the output node.
        state: the values of the nodes which were evaluated.
        inputs: the inputs of the network.

    Returns:
        The outputs. An output which shares memory with an input is a copy,
        so writing into an output does not change an input.

    Raises:
        ValueError: if the node has not one argument, or if the argument does
            not name nodes which were evaluated.
    """
    args = node.args or []
    if len(args) != 1:
        raise ValueError(
            f"the output node has {len(args)} arguments, expected one argument "
            "with the outputs"
        )
    value = _references(args[0], state)
    outputs = tuple(value) if isinstance(value, list) else (value,)
    for output in outputs:
        if not isinstance(output, np.ndarray):
            raise ValueError(f"the output {output!r} is not an array")
    return tuple(
        output.copy() if any(np.may_share_memory(output, x) for x in inputs) else output
        for output in outputs
    )


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
        The outputs of the network, arrays which share no memory with the
        inputs.

    Raises:
        UnsupportedLayerError: if a layer or function is not implemented, not
            available in the backend or called with arguments its
            implementation does not take.
        ValueError: if the number of inputs is not the number of placeholders,
            if a node has an unknown opcode, if an argument does not name a
            node which was evaluated or if a layer or function cannot be
            evaluated on its input; the message names the network and the
            node.
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
    arrays: list[np.ndarray] = []
    remaining = iter(inputs)
    for node in model.forward:
        try:
            if node.op == PLACEHOLDER:
                state[node.name] = backend.asarray(next(remaining))
                arrays.append(state[node.name])
                continue
            if node.op == OUTPUT:
                return _outputs(node, state, arrays)
            if node.op == CALL_MODULE:
                value = _call_module(network, node, layers, parameters, backend, state)
            elif node.op in (CALL_FUNCTION, CALL_METHOD):
                value = _call_function(network, node, backend, state)
            else:
                raise ValueError(f"the opcode '{node.op}' is not known")
        except UnsupportedLayerError:
            raise
        except (TypeError, ValueError) as err:
            raise ValueError(f"Network '{network}', node '{node.name}': {err}") from err

        value = np.asarray(value)
        state[node.name] = value if on_node is None else on_node(node, value)

    raise ValueError(f"Network '{network}': the forward pass has no output node")
