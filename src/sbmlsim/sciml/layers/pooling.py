"""The pooling layers, in numpy only.

A pooling layer reduces the windows of its kernel to one value: the maximum,
the mean or the p-norm. The adaptive layers choose the windows from the size
of the output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import as_tuple, layer_nd
from sbmlsim.sciml.layers.windows import add_batch, output_size, pad_spatial, windows


def pooled_windows(
    x: np.ndarray,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
    value: float,
) -> np.ndarray:
    """Get the windows of a pooling layer.

    Args:
        x: input of shape `(N, C, *spatial)`.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        padding: the padding on both sides per spatial axis.
        dilation: the step between two elements of a window per spatial axis.
        ceil_mode: whether the last window may reach over the end.
        value: the value of the padding.

    Returns:
        The windows of shape `(N, C, *output, *kernel_size)`.
    """
    n_outs = [
        output_size(size, k, s, p, d, ceil_mode)
        for size, k, s, p, d in zip(
            x.shape[2:], kernel_size, stride, padding, dilation, strict=True
        )
    ]
    after = []
    for size, n_out, k, s, p, d in zip(
        x.shape[2:], n_outs, kernel_size, stride, padding, dilation, strict=True
    ):
        needed = (n_out - 1) * s + d * (k - 1) + 1
        after.append(max(needed - size - p, p))
    padded = pad_spatial(x, padding, after, value=value)
    view = windows(padded, kernel_size, stride, dilation)
    index = (slice(None), slice(None), *(slice(0, n_out) for n_out in n_outs))
    return view[index]


def pool_arguments(
    args: Mapping[str, Any], n: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Get the kernel size, the stride and the padding of a pooling layer.

    Args:
        args: the arguments of the layer.
        n: the number of spatial dimensions.

    Returns:
        `kernel_size`, `stride` (the kernel size when it is not given) and
        `padding` (default `0`), each per spatial axis.
    """
    kernel_size = as_tuple(args["kernel_size"], n)
    stride = args.get("stride")
    padding = as_tuple(args.get("padding", 0), n)
    return kernel_size, kernel_size if stride is None else as_tuple(stride, n), padding


def average(
    x: np.ndarray,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> np.ndarray:
    """Calculate the mean of the windows of an average pooling.

    Args:
        x: input of shape `(N, C, *spatial)`.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        padding: the zero padding on both sides per spatial axis.
        ceil_mode: whether the last window may reach over the end.
        count_include_pad: whether the padding counts as elements of a window.
            The part of a window which reaches over the padding in `ceil_mode`
            never counts.
        divisor_override: the divisor of every window, the number of elements
            when `None`.

    Returns:
        The output of shape `(N, C, *output)`.
    """
    n = len(kernel_size)
    ones = (1,) * n
    kernel_axes = tuple(range(2 + n, 2 + 2 * n))
    total = pooled_windows(
        x, kernel_size, stride, padding, ones, ceil_mode, value=0.0
    ).sum(axis=kernel_axes)
    if divisor_override is not None:
        return total / divisor_override

    counted = np.ones((1, 1, *x.shape[2:]))
    if count_include_pad:
        counted = pad_spatial(counted, padding, padding, value=1.0)
        padding = (0,) * n
    count = pooled_windows(
        counted, kernel_size, stride, padding, ones, ceil_mode, value=0.0
    ).sum(axis=kernel_axes)
    return total / count


@layer_nd("MaxPool{n}d", backends=NUMPY_ONLY)
def max_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `MaxPool1d`, `MaxPool2d` and `MaxPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `kernel_size`, `stride` (default `kernel_size`), `padding`
            (default `0`, padded with `-inf`), `dilation` (default `1`),
            `return_indices` (only `False`), `ceil_mode` (default `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
        + 1`, rounded up in `ceil_mode`.

    Raises:
        ValueError: if `return_indices` is set.
    """
    if args.get("return_indices", False):
        raise ValueError("MaxPool: return_indices is not supported")
    x, unbatched = add_batch(x, n, "MaxPool")
    kernel_size, stride, padding = pool_arguments(args, n)
    dilation = as_tuple(args.get("dilation", 1), n)
    view = pooled_windows(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        args.get("ceil_mode", False),
        value=-np.inf,
    )
    y = view.max(axis=tuple(range(2 + n, 2 + 2 * n)))
    return y[0] if unbatched else y


@layer_nd("AvgPool{n}d", backends=NUMPY_ONLY)
def avg_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AvgPool1d`, `AvgPool2d` and `AvgPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `kernel_size`, `stride` (default `kernel_size`), `padding`
            (default `0`, padded with zeros), `ceil_mode` (default `False`),
            `count_include_pad` (default `True`), `divisor_override` (default
            `None`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial + 2 * padding - kernel_size) // stride + 1`, rounded up in
        `ceil_mode`.
    """
    x, unbatched = add_batch(x, n, "AvgPool")
    kernel_size, stride, padding = pool_arguments(args, n)
    y = average(
        x,
        kernel_size,
        stride,
        padding,
        args.get("ceil_mode", False),
        args.get("count_include_pad", True),
        args.get("divisor_override"),
    )
    return y[0] if unbatched else y


@layer_nd("LPPool{n}d", backends=NUMPY_ONLY)
def lp_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `LPPool1d`, `LPPool2d` and `LPPool3d`.

    PyTorch calculates the sum of a window as its mean times the size of the
    kernel, which is followed here: a window which reaches over the end in
    `ceil_mode` is scaled by the kernel and not by its elements.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `norm_type`, `kernel_size`, `stride` (default `kernel_size`),
            `ceil_mode` (default `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output `(sum(x ** norm_type)) ** (1 / norm_type)` over every
        window, of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial - kernel_size) // stride + 1`, rounded up in `ceil_mode`.
    """
    x, unbatched = add_batch(x, n, "LPPool")
    kernel_size, stride, _ = pool_arguments(args, n)
    norm_type = float(args["norm_type"])
    mean = average(
        x**norm_type,
        kernel_size,
        stride,
        (0,) * n,
        args.get("ceil_mode", False),
        count_include_pad=True,
        divisor_override=None,
    )
    y = (mean * float(np.prod(kernel_size))) ** (1.0 / norm_type)
    return y[0] if unbatched else y


def adaptive(
    n: int,
    x: np.ndarray,
    output: Any,
    reduce: Callable[..., np.ndarray],
    name: str,
) -> np.ndarray:
    """Reduce the windows of an adaptive pooling, one axis after the other.

    The window `i` of an axis of the size `L` with `O` outputs is
    `[floor(i * L / O), ceil((i + 1) * L / O))`. The windows of all axes are
    boxes, so the maximum and the mean of a box are the reduction along one
    axis after the other.

    Args:
        n: the number of spatial dimensions.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.
        output: `output_size` of the layer, an integer or one entry per axis,
            `None` keeps the size of the input.
        reduce: `numpy.max` or `numpy.mean`.
        name: the name of the layer, for the message of the error.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.
    """
    sizes = list(output) if isinstance(output, (list, tuple)) else [output] * n
    x, unbatched = add_batch(x, n, name)
    for k, n_out in enumerate(sizes):
        axis = 2 + k
        size = x.shape[axis]
        if n_out is None:
            continue
        slices = []
        for i in range(n_out):
            start = (i * size) // n_out
            end = -(-((i + 1) * size) // n_out)
            slices.append(reduce(x.take(range(start, end), axis=axis), axis=axis))
        x = np.stack(slices, axis=axis)
    return x[0] if unbatched else x


@layer_nd("AdaptiveMaxPool{n}d", backends=NUMPY_ONLY)
def adaptive_max_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `output_size`, `return_indices` (only `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.

    Raises:
        ValueError: if `return_indices` is set.
    """
    if args.get("return_indices", False):
        raise ValueError("AdaptiveMaxPool: return_indices is not supported")
    return adaptive(n, x, args["output_size"], np.max, "AdaptiveMaxPool")


@layer_nd("AdaptiveAvgPool{n}d", backends=NUMPY_ONLY)
def adaptive_avg_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `output_size`.
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.
    """
    return adaptive(n, x, args["output_size"], np.mean, "AdaptiveAvgPool")
