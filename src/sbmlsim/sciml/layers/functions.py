"""The activation functions and tensor operations, which both backends support.

Every function has the keyword arguments and the defaults of the function of
`torch.nn.functional` (or `torch` for the tensor operations) it is named
after. The functions with a condition are written with `Backend.select`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.layers.core import flatten_array
from sbmlsim.sciml.layers.registry import function

#: the constants of `selu`
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


def clip(backend: Backend, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Limit the values to the interval `[low, high]`.

    Args:
        backend: the backend.
        x: the values.
        low: the lower limit.
        high: the upper limit.

    Returns:
        `low` where `x <= low`, `high` where `x > high` and `x` between.
    """
    return backend.select(x, low, low, backend.select(x, high, x, high))


@function("tanh")
def tanh(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `tanh(x)`."""
    return backend.tanh(x)


@function("sigmoid")
def sigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `1 / (1 + exp(-x))`."""
    return 1.0 / (1.0 + backend.exp(-x))


@function("relu")
def relu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `max(x, 0)`."""
    return backend.select(x, 0.0, 0.0, x)


@function("relu6")
def relu6(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `min(max(x, 0), 6)`."""
    return clip(backend, x, 0.0, 6.0)


@function("hardtanh")
def hardtanh(
    backend: Backend, x: np.ndarray, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    """Evaluate `min(max(x, min_val), max_val)`."""
    return clip(backend, x, min_val, max_val)


@function("hardsigmoid")
def hardsigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `min(max(x / 6 + 1 / 2, 0), 1)`."""
    return clip(backend, x / 6.0 + 0.5, 0.0, 1.0)


@function("hardswish")
def hardswish(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * min(max(x + 3, 0), 6) / 6`."""
    return x * clip(backend, x + 3.0, 0.0, 6.0) / 6.0


@function("leaky_relu")
def leaky_relu(
    backend: Backend, x: np.ndarray, negative_slope: float = 0.01
) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `negative_slope * x` otherwise."""
    return backend.select(x, 0.0, negative_slope * x, x)


@function("elu")
def elu(backend: Backend, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `alpha * (exp(x) - 1)` otherwise."""
    return backend.select(x, 0.0, alpha * (backend.exp(x) - 1.0), x)


@function("celu")
def celu(backend: Backend, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `alpha * (exp(x / alpha) - 1)` otherwise."""
    return backend.select(x, 0.0, alpha * (backend.exp(x / alpha) - 1.0), x)


@function("selu")
def selu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `scale * elu(x, alpha)` with the constants of `selu`."""
    return SELU_SCALE * elu(backend, x, alpha=SELU_ALPHA)


@function("gelu")
def gelu(backend: Backend, x: np.ndarray, approximate: str = "none") -> np.ndarray:
    """Evaluate `x * Phi(x)` with the distribution function of the normal.

    Args:
        backend: the backend.
        x: the input.
        approximate: `none` for the error function, `tanh` for the
            approximation `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.

    Returns:
        The output.

    Raises:
        ValueError: if `approximate` is neither `none` nor `tanh`.
    """
    if approximate == "none":
        return 0.5 * x * (1.0 + backend.erf(x / math.sqrt(2.0)))
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + backend.tanh(inner))
    raise ValueError(f"gelu: approximate '{approximate}' is not 'none' or 'tanh'")


@function("softplus")
def softplus(
    backend: Backend, x: np.ndarray, beta: float = 1.0, threshold: float = 20.0
) -> np.ndarray:
    """Evaluate `log(1 + exp(beta * x)) / beta`, `x` for `beta * x > threshold`."""
    scaled = beta * x
    return backend.select(
        scaled, threshold, backend.log(1.0 + backend.exp(scaled)) / beta, x
    )


@function("log_sigmoid", "logsigmoid")
def log_sigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `log(1 / (1 + exp(-x)))`.

    The two branches are the same function, written so that the exponential of
    the branch which is chosen does not overflow.
    """
    return backend.select(
        x,
        0.0,
        x - backend.log(1.0 + backend.exp(x)),
        -backend.log(1.0 + backend.exp(-x)),
    )


@function("mish")
def mish(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * tanh(softplus(x))`."""
    return x * backend.tanh(softplus(backend, x))


@function("silu")
def silu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * sigmoid(x)`."""
    return x * sigmoid(backend, x)


@function("softsign")
def softsign(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x / (1 + |x|)`."""
    return x / (1.0 + backend.absolute(x))


@function("tanhshrink")
def tanhshrink(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x - tanh(x)`."""
    return x - backend.tanh(x)


@function("softmax")
def softmax(backend: Backend, x: np.ndarray, dim: int) -> np.ndarray:
    """Evaluate `exp(x) / sum(exp(x))` along the axis `dim`."""
    exponential = backend.exp(x - backend.stabilizer(x, dim))
    return exponential / exponential.sum(axis=dim, keepdims=True)


@function("log_softmax")
def log_softmax(backend: Backend, x: np.ndarray, dim: int) -> np.ndarray:
    """Evaluate `x - log(sum(exp(x)))` along the axis `dim`."""
    shifted = x - backend.stabilizer(x, dim)
    return shifted - backend.log(backend.exp(shifted).sum(axis=dim, keepdims=True))


@function("flatten")
def flatten(
    backend: Backend, x: np.ndarray, start_dim: int = 0, end_dim: int = -1
) -> np.ndarray:
    """Evaluate `torch.flatten`, in row major order."""
    return flatten_array(x, start_dim, end_dim)


@function("cat", "concat", "concatenate")
def cat(backend: Backend, tensors: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
    """Evaluate `torch.cat`, i.e. join the arrays along the axis `dim`."""
    return np.concatenate(list(tensors), axis=dim)
