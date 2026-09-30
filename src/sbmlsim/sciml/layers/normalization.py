"""The normalization layers, in numpy only and in evaluation mode.

A normalization layer is `y = (x - mean) / sqrt(var + eps) * weight + bias`.
`BatchNorm` and `InstanceNorm` use the stored statistics, i.e. the arrays
`running_mean` and `running_var` of the layer. A layer without stored
statistics calculates them from its input, which is what PyTorch does for a
layer with `track_running_stats=False` and for a layer whose statistics are
`None`: `BatchNorm` over the batch and the spatial axes, `InstanceNorm` over
the spatial axes of every sample. `LayerNorm` has no stored statistics. The
variance is the biased one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import ArraySpec, layer, layer_nd


def norm_arrays(args: Mapping[str, Any], affine: bool) -> dict[str, ArraySpec]:
    """Get the arrays of a `BatchNorm` or `InstanceNorm` layer.

    Args:
        args: the arguments of the layer.
        affine: the default of `affine`, which differs between the two.

    Returns:
        `weight` and `bias` for an affine layer and the running statistics,
        which are neither required nor parameters of a fit. The statistics
        include `num_batches_tracked`, the counter of the batches of the
        training, which the `state_dict` of PyTorch holds next to
        `running_mean` and `running_var` and which the evaluation does not
        use.
    """
    shape = (args["num_features"],)
    arrays: dict[str, ArraySpec] = {}
    if args.get("affine", affine):
        arrays["weight"] = ArraySpec(shape)
        if args.get("bias", True):
            arrays["bias"] = ArraySpec(shape)
    arrays["running_mean"] = ArraySpec(shape, required=False, trainable=False)
    arrays["running_var"] = ArraySpec(shape, required=False, trainable=False)
    arrays["num_batches_tracked"] = ArraySpec((), required=False, trainable=False)
    return arrays


def batch_norm_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `BatchNorm` layer, which is affine by default."""
    return norm_arrays(args, affine=True)


def instance_norm_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of an `InstanceNorm` layer, not affine by default."""
    return norm_arrays(args, affine=False)


def normalize(
    name: str,
    x: np.ndarray,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    axes: tuple[int, ...],
    channel_axis: int,
) -> np.ndarray:
    """Normalize an input per channel.

    Args:
        name: the name of the layer type, for the messages.
        x: the input.
        args: the arguments of the layer, `num_features` and `eps`.
        arrays: the arrays of the layer, all of them of shape `(C,)`.
        axes: the axes the statistics are calculated over when the layer
            stores none.
        channel_axis: the axis of the channels.

    Returns:
        The normalized input.

    Raises:
        ValueError: if the input has not `num_features` channels, if only one
            of the running statistics is stored, or if the statistics are
            calculated from a single value per channel.
    """
    if x.shape[channel_axis] != args["num_features"]:
        raise ValueError(
            f"{name}: the input has {x.shape[channel_axis]} channels on axis "
            f"{channel_axis}, expected num_features {args['num_features']}"
        )
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    stored = [key for key in ("running_mean", "running_var") if key in arrays]
    if len(stored) == 1:
        raise ValueError(
            f"{name}: the arrays hold {stored[0]} but not the other running "
            "statistic, running_mean and running_var are stored together"
        )
    if stored:
        mean = arrays["running_mean"].reshape(shape)
        var = arrays["running_var"].reshape(shape)
    else:
        size = int(np.prod([x.shape[axis] for axis in axes]))
        if size == 1:
            raise ValueError(
                f"{name}: without stored statistics the input needs more than "
                f"one value per channel, got input of shape {x.shape}"
            )
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
    y = (x - mean) / np.sqrt(var + args.get("eps", 1e-5))
    if "weight" in arrays:
        y = y * arrays["weight"].reshape(shape)
    if "bias" in arrays:
        y = y + arrays["bias"].reshape(shape)
    return y


@layer_nd("BatchNorm{n}d", arrays=batch_norm_arrays, backends=NUMPY_ONLY)
def batch_norm(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `BatchNorm1d`, `BatchNorm2d` and `BatchNorm3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `num_features`, `eps` (default `1e-5`), `affine` (default
            `True`), `bias` (default `True`); `momentum` and
            `track_running_stats` belong to the training and are not used.
        arrays: `weight` and `bias` of shape `(num_features,)` for an affine
            layer, `running_mean` and `running_var` when they are stored.
        x: input of shape `(N, C, *spatial)`, for one dimension also `(N, C)`.

    Returns:
        The output of the shape of the input.

    Raises:
        ValueError: if the input has no batch axis.
    """
    if x.ndim != n + 2 and not (n == 1 and x.ndim == 2):
        raise ValueError(
            f"BatchNorm{n}d: the input has {x.ndim} axes, expected "
            + (
                "2 or 3 axes (N, C and at most 1 spatial axis)"
                if n == 1
                else f"{n + 2} axes (N, C and {n} spatial axes)"
            )
        )
    axes = (0, *range(2, x.ndim))
    return normalize(f"BatchNorm{n}d", x, args, arrays, axes, channel_axis=1)


@layer_nd("InstanceNorm{n}d", arrays=instance_norm_arrays, backends=NUMPY_ONLY)
def instance_norm(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `InstanceNorm1d`, `InstanceNorm2d` and `InstanceNorm3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `num_features`, `eps` (default `1e-5`), `affine` (default
            `False`), `bias` (default `True`); `momentum` and
            `track_running_stats` belong to the training and are not used.
        arrays: `weight` and `bias` of shape `(num_features,)` for an affine
            layer, `running_mean` and `running_var` when they are stored.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of the shape of the input.

    Raises:
        ValueError: if the input has neither `n + 1` nor `n + 2` axes.
    """
    if x.ndim not in (n + 1, n + 2):
        raise ValueError(
            f"InstanceNorm{n}d: the input has {x.ndim} axes, expected {n + 1} "
            f"or {n + 2}"
        )
    axes = tuple(range(x.ndim - n, x.ndim))
    return normalize(
        f"InstanceNorm{n}d", x, args, arrays, axes, channel_axis=x.ndim - n - 1
    )


def normalized_shape(args: Mapping[str, Any]) -> tuple[int, ...]:
    """Get the `normalized_shape` of a `LayerNorm` layer as a tuple."""
    value = args["normalized_shape"]
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(s) for s in value)


def layer_norm_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `LayerNorm` layer."""
    shape = normalized_shape(args)
    arrays: dict[str, ArraySpec] = {}
    if args.get("elementwise_affine", True):
        arrays["weight"] = ArraySpec(shape)
        if args.get("bias", True):
            arrays["bias"] = ArraySpec(shape)
    return arrays


@layer("LayerNorm", arrays=layer_norm_arrays, backends=NUMPY_ONLY)
def layer_norm(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `LayerNorm`.

    Args:
        backend: the backend.
        args: `normalized_shape`, `eps` (default `1e-5`),
            `elementwise_affine` (default `True`), `bias` (default `True`).
        arrays: `weight` and `bias` of the shape `normalized_shape` for an
            affine layer.
        x: input of shape `(*, *normalized_shape)`.

    Returns:
        The output of the shape of the input, normalized over the last axes
        which `normalized_shape` covers.

    Raises:
        ValueError: if the last axes of the input are not `normalized_shape`.
    """
    shape = normalized_shape(args)
    if x.shape[x.ndim - len(shape) :] != shape:
        raise ValueError(
            f"LayerNorm: the input of shape {x.shape} does not end with the "
            f"normalized_shape {shape}"
        )
    axes = tuple(range(x.ndim - len(shape), x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    y = (x - mean) / np.sqrt(var + args.get("eps", 1e-5))
    if "weight" in arrays:
        y = y * arrays["weight"]
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y
