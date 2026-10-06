"""Sliding windows over the spatial axes of an array.

The convolution and the pooling layers are reductions over the windows of
their kernel. An input has the shape `(N, C, *spatial)` or, without a batch,
`(C, *spatial)`; the layers add the batch axis, work on the windows and remove
it again.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def add_batch(x: np.ndarray, n: int, name: str) -> tuple[np.ndarray, bool]:
    """Add the batch axis to an input without one.

    Args:
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.
        n: the number of spatial dimensions.
        name: the name of the layer, for the message of the error.

    Returns:
        The input of shape `(N, C, *spatial)` and whether the axis was added.

    Raises:
        ValueError: if the input has neither `n + 1` nor `n + 2` axes.
    """
    if x.ndim == n + 1:
        return x[np.newaxis], True
    if x.ndim == n + 2:
        return x, False
    raise ValueError(
        f"{name}: the input has {x.ndim} axes, expected {n + 1} (C and {n} "
        f"spatial axes) or {n + 2} (N, C and {n} spatial axes)"
    )


def pad_spatial(
    x: np.ndarray,
    before: Sequence[int],
    after: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """Pad the spatial axes of an input, a negative padding removes values.

    Args:
        x: input of shape `(N, C, *spatial)`.
        before: padding in front of every spatial axis.
        after: padding behind every spatial axis.
        mode: mode of `numpy.pad`.
        value: the value of the padding for the mode `constant`.

    Returns:
        The padded input.
    """
    crop = [slice(None), slice(None)]
    width = [(0, 0), (0, 0)]
    for size, b, a in zip(x.shape[2:], before, after, strict=True):
        crop.append(slice(max(-b, 0), size - max(-a, 0)))
        width.append((max(b, 0), max(a, 0)))
    x = x[tuple(crop)]
    if mode == "constant":
        return np.pad(x, width, mode="constant", constant_values=value)
    return np.pad(x, width, mode=mode)  # ty: ignore[no-matching-overload]


def windows(
    x: np.ndarray,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> np.ndarray:
    """Get the windows of a kernel over the spatial axes.

    Args:
        x: input of shape `(N, C, *spatial)`, already padded.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        dilation: the step between two elements of a window per spatial axis.

    Returns:
        A view of shape `(N, C, *output, *kernel_size)` with
        `output = (spatial - dilation * (kernel_size - 1) - 1) // stride + 1`.

    Raises:
        ValueError: if the kernel is larger than the input.
    """
    n = len(kernel_size)
    extent = tuple(d * (k - 1) + 1 for k, d in zip(kernel_size, dilation, strict=True))
    if any(e > s for e, s in zip(extent, x.shape[2:], strict=True)):
        raise ValueError(
            f"the kernel with the extent {extent} is larger than the input "
            f"with the spatial shape {x.shape[2:]}"
        )
    view = sliding_window_view(x, extent, axis=tuple(range(2, 2 + n)))
    index = (
        slice(None),
        slice(None),
        *(slice(None, None, s) for s in stride),
        *(slice(None, None, d) for d in dilation),
    )
    return view[index]


def output_size(
    size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> int:
    """Get the number of windows of a pooling layer along one axis.

    Args:
        size: the size of the input along the axis.
        kernel_size: the size of the kernel.
        stride: the step between two windows.
        padding: the padding on both sides.
        dilation: the step between two elements of a window.
        ceil_mode: whether the last window may reach over the end of the
            input. A window which starts behind the input and its left padding
            is not counted.

    Returns:
        The number of windows.
    """
    numerator = size + 2 * padding - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        n = -(-numerator // stride) + 1
        if (n - 1) * stride >= size + padding:
            n -= 1
        return n
    return numerator // stride + 1
