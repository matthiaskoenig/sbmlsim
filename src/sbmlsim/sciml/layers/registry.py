"""The registry of the layers and functions of a forward pass.

A layer is a function `(backend, args, arrays, *inputs) -> output`, with the
arguments of the layer in the NN YAML (`args`, the keyword arguments of the
PyTorch class) and the arrays of the layer in the PyTorch layout. A function
is a function `(backend, *inputs, **kwargs) -> output` with the keyword
arguments of the PyTorch function. Both are registered under their PyTorch
name together with the backends they support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import ALL_BACKENDS, BackendKind


@dataclass(frozen=True)
class ArraySpec:
    """An array of a layer.

    Attributes:
        shape: the shape of the array in the PyTorch layout.
        required: whether the layer cannot be evaluated without the array.
        trainable: whether the elements are parameters of a fit. The running
            statistics of a normalization layer are arrays but not parameters.
    """

    shape: tuple[int, ...]
    required: bool = True
    trainable: bool = True


#: the arrays of a layer from its arguments
ArraysFunction = Callable[[Mapping[str, Any]], dict[str, ArraySpec]]


def no_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a layer without arrays.

    Args:
        args: the arguments of the layer.

    Returns:
        An empty dictionary.
    """
    return {}


@dataclass(frozen=True)
class LayerType:
    """A type of layer, e.g. `Linear`.

    Attributes:
        name: the name of the PyTorch class.
        forward: the implementation `(backend, args, arrays, *inputs)`.
        arrays: the arrays of a layer of this type from its arguments.
        backends: the backends the implementation supports.
    """

    name: str
    forward: Callable[..., np.ndarray]
    arrays: ArraysFunction
    backends: frozenset[BackendKind]


@dataclass(frozen=True)
class FunctionType:
    """A function or method of the forward pass, e.g. `relu`.

    Attributes:
        name: the name of the PyTorch function.
        function: the implementation `(backend, *inputs, **kwargs)`.
        backends: the backends the implementation supports.
    """

    name: str
    function: Callable[..., np.ndarray]
    backends: frozenset[BackendKind]


#: the layers by the name of their PyTorch class
LAYERS: dict[str, LayerType] = {}

#: the functions and methods by their PyTorch name
FUNCTIONS: dict[str, FunctionType] = {}


def layer(
    *names: str,
    arrays: ArraysFunction = no_arrays,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of one or more layer types.

    Args:
        *names: the names of the PyTorch classes the function implements.
        arrays: the arrays of a layer from its arguments.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(forward: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        for name in names:
            LAYERS[name] = LayerType(
                name=name, forward=forward, arrays=arrays, backends=backends
            )
        return forward

    return register


def layer_nd(
    template: str,
    arrays: Callable[[int, Mapping[str, Any]], dict[str, ArraySpec]] | None = None,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of a layer type for 1, 2 and 3 dimensions.

    The implementation is `(n, backend, args, arrays, *inputs)` with the
    number of spatial dimensions `n`, and is registered once per dimension
    with `n` bound.

    Args:
        template: the name of the PyTorch classes with `{n}` for the number of
            dimensions, e.g. `Conv{n}d`.
        arrays: the arrays of a layer from `n` and its arguments.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(forward: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        for n in (1, 2, 3):
            name = template.format(n=n)
            LAYERS[name] = LayerType(
                name=name,
                forward=partial(forward, n),
                arrays=no_arrays if arrays is None else partial(arrays, n),
                backends=backends,
            )
        return forward

    return register


def function(
    *names: str,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of one or more functions.

    Args:
        *names: the names of the PyTorch functions the function implements.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(
        implementation: Callable[..., np.ndarray],
    ) -> Callable[..., np.ndarray]:
        for name in names:
            FUNCTIONS[name] = FunctionType(
                name=name, function=implementation, backends=backends
            )
        return implementation

    return register


def as_tuple(value: Any, n: int) -> tuple[int, ...]:
    """Expand an argument of a layer to one value per spatial dimension.

    Args:
        value: an integer, which holds for every dimension, or a sequence of
            `n` integers.
        n: the number of spatial dimensions.

    Returns:
        The values of the dimensions.

    Raises:
        ValueError: if a sequence does not have `n` entries.
    """
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n
    values = tuple(int(v) for v in value)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"'{value}' does not have {n} entries")
    return values
