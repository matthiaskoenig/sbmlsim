"""The layers which both backends support.

`Linear`, `Bilinear` and `Flatten` are array operations of numpy, which work
on arrays of numbers and of expressions. The dropout layers are the identity,
because a network is evaluated in evaluation mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.layers.registry import ArraySpec, layer


def flatten_array(x: np.ndarray, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
    """Flatten a range of axes of an array in row major order.

    Args:
        x: the array.
        start_dim: the first axis which is flattened.
        end_dim: the last axis which is flattened.

    Returns:
        The array with the axes `start_dim` to `end_dim` as one axis.

    Raises:
        ValueError: if an axis is out of range or `start_dim` is behind
            `end_dim`.
    """
    # PyTorch treats an array without axes as one with a single axis
    ndim = max(x.ndim, 1)
    for dim in (start_dim, end_dim):
        if not -ndim <= dim < ndim:
            raise ValueError(
                f"flatten: dimension '{dim}' is out of range for an array of "
                f"shape {x.shape}, expected the range [{-ndim}, {ndim - 1}]"
            )
    if x.ndim == 0:
        return x.reshape(1)
    start = start_dim % ndim
    end = end_dim % ndim
    if start > end:
        raise ValueError(
            f"flatten: start_dim '{start_dim}' is behind end_dim '{end_dim}'"
        )
    return x.reshape((*x.shape[:start], -1, *x.shape[end + 1 :]))


def linear_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `Linear` layer."""
    arrays = {"weight": ArraySpec((args["out_features"], args["in_features"]))}
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_features"],))
    return arrays


@layer("Linear", arrays=linear_arrays)
def linear(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Linear`, i.e. `y = x W^T + b` on the last axis.

    Args:
        backend: the backend.
        args: `in_features`, `out_features`, `bias` (default `True`).
        arrays: `weight` of shape `(out_features, in_features)` and `bias` of
            shape `(out_features,)`.
        x: input of shape `(*, in_features)`.

    Returns:
        The output of shape `(*, out_features)`.
    """
    y = x @ arrays["weight"].T
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y


def bilinear_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `Bilinear` layer."""
    arrays = {
        "weight": ArraySpec(
            (args["out_features"], args["in1_features"], args["in2_features"])
        )
    }
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_features"],))
    return arrays


@layer("Bilinear", arrays=bilinear_arrays)
def bilinear(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    """Evaluate `Bilinear`, i.e. `y_k = x1^T A_k x2 + b_k` on the last axis.

    Args:
        backend: the backend.
        args: `in1_features`, `in2_features`, `out_features`, `bias` (default
            `True`).
        arrays: `weight` of shape `(out_features, in1_features, in2_features)`
            and `bias` of shape `(out_features,)`.
        x1: first input of shape `(*, in1_features)`.
        x2: second input of shape `(*, in2_features)`.

    Returns:
        The output of shape `(*, out_features)`.
    """
    # (*, in1) . (out, in1, in2) over in1 -> (*, out, in2)
    left = np.tensordot(x1, arrays["weight"], axes=([-1], [1]))
    y = (left * x2[..., np.newaxis, :]).sum(axis=-1)
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y


@layer("Flatten")
def flatten_layer(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Flatten`.

    Args:
        backend: the backend.
        args: `start_dim` (default `1`) and `end_dim` (default `-1`).
        arrays: no arrays.
        x: the input.

    Returns:
        The input with the axes `start_dim` to `end_dim` as one axis, in row
        major order.
    """
    return flatten_array(x, args.get("start_dim", 1), args.get("end_dim", -1))


@layer(
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
)
def dropout(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate a dropout layer in evaluation mode, which is the identity.

    Args:
        backend: the backend.
        args: `p` and `inplace`, which are not used.
        arrays: no arrays.
        x: the input.

    Returns:
        The input.
    """
    return x
