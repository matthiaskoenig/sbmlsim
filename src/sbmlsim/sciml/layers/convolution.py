"""The convolution and the transposed convolution layers, in numpy only.

A convolution is the tensor product of the windows of the kernel with the
weight. A transposed convolution is a convolution of the input with zeros
between its elements and the kernel flipped, which is what the gradient of a
convolution is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import ArraySpec, as_tuple, layer_nd
from sbmlsim.sciml.layers.windows import add_batch, pad_spatial, windows

#: the modes of `numpy.pad` for the `padding_mode` of a convolution
PADDING_MODES: dict[str, str] = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "edge",
    "circular": "wrap",
}


def check_groups(args: Mapping[str, Any]) -> int:
    """Get the number of groups of a layer and check the channels against it.

    Args:
        args: the arguments of the layer.

    Returns:
        The number of groups.

    Raises:
        ValueError: if the groups are not positive or the input or the output
            channels are not divisible by them.
    """
    groups = args.get("groups", 1)
    if groups < 1:
        raise ValueError(f"groups must be a positive integer, got {groups}")
    for name in ("in_channels", "out_channels"):
        if args[name] % groups != 0:
            raise ValueError(
                f"{name} ({args[name]}) must be divisible by groups ({groups})"
            )
    return groups


def layer_arrays(
    n: int, args: Mapping[str, Any], transposed: bool
) -> dict[str, ArraySpec]:
    """Get the arrays of a convolution with `n` spatial dimensions.

    Args:
        n: the number of spatial dimensions.
        args: the arguments of the layer.
        transposed: whether the layer is a transposed convolution, whose weight
            has the input channels first.

    Returns:
        The `weight` and, unless `bias` is `False`, the `bias`.
    """
    groups = check_groups(args)
    c_in, c_out = args["in_channels"], args["out_channels"]
    channels = (c_in, c_out // groups) if transposed else (c_out, c_in // groups)
    arrays = {"weight": ArraySpec((*channels, *as_tuple(args["kernel_size"], n)))}
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((c_out,))
    return arrays


def conv_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a convolution layer with `n` spatial dimensions."""
    return layer_arrays(n, args, transposed=False)


def conv_transpose_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a transposed convolution with `n` spatial dimensions."""
    return layer_arrays(n, args, transposed=True)


def check_channels(x: np.ndarray, args: Mapping[str, Any], name: str) -> None:
    """Check the channels of a batched input against the layer.

    Args:
        x: input of shape `(N, C_in, *spatial)`.
        args: the arguments of the layer.
        name: the name of the layer, for the message of the error.

    Raises:
        ValueError: if the input does not have `in_channels` channels.
    """
    if x.shape[1] != args["in_channels"]:
        raise ValueError(
            f"{name}: expected the input {x.shape} to have {args['in_channels']} "
            f"channels, but got {x.shape[1]}"
        )


def check_padding(padding: Sequence[int], name: str) -> None:
    """Check that a padding is not negative.

    Args:
        padding: the padding per spatial axis.
        name: the name of the argument, for the message of the error.

    Raises:
        ValueError: if a padding is negative.
    """
    if any(p < 0 for p in padding):
        raise ValueError(f"negative {name} is not supported, got {list(padding)}")


def check_padding_mode(
    shape: Sequence[int],
    before: Sequence[int],
    after: Sequence[int],
    padding_mode: str,
) -> None:
    """Check that the padding of a mode fits the input, as PyTorch does.

    Args:
        shape: the spatial shape of the input.
        before: padding in front of every spatial axis.
        after: padding behind every spatial axis.
        padding_mode: the `padding_mode` of the layer.

    Raises:
        ValueError: if a `reflect` padding is not smaller than the input or a
            `circular` padding is larger than the input, which would reflect
            or wrap more than once.
    """
    if padding_mode not in ("reflect", "circular"):
        return
    limit = 1 if padding_mode == "reflect" else 0
    for size, b, a in zip(shape, before, after, strict=True):
        if max(b, a) > size - limit:
            raise ValueError(
                f"Conv: the {padding_mode} padding {max(b, a)} must be "
                f"{'smaller than' if limit else 'at most'} the input size {size}"
            )


def add_bias(y: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Add the bias of a layer per output channel, if it has one."""
    if "bias" not in arrays:
        return y
    return y + arrays["bias"].reshape((1, -1) + (1,) * (y.ndim - 2))


def correlate(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> np.ndarray:
    """Calculate the cross correlation of a padded input with a kernel.

    Args:
        x: input of shape `(N, C_in, *spatial)`, already padded.
        weight: kernel of shape `(C_out, C_in / groups, *kernel_size)`.
        stride: the step between two windows per spatial axis.
        dilation: the step between two elements of the kernel per spatial axis.
        groups: the number of groups the channels are split into.

    Returns:
        The output of shape `(N, C_out, *output)`.
    """
    n = len(stride)
    c_out = weight.shape[0]
    c_in_group = weight.shape[1]
    c_out_group = c_out // groups
    view = windows(x, weight.shape[2:], stride, dilation)
    kernel_axes = list(range(2 + n, 2 + 2 * n))
    outputs = []
    for g in range(groups):
        view_g = view[:, g * c_in_group : (g + 1) * c_in_group]
        weight_g = weight[g * c_out_group : (g + 1) * c_out_group]
        # (N, C_in, *output, *kernel) . (C_out, C_in, *kernel) -> (N, *output, C_out)
        y = np.tensordot(
            view_g, weight_g, axes=([1, *kernel_axes], list(range(1, 2 + n)))
        )
        outputs.append(np.moveaxis(y, -1, 1))
    return np.concatenate(outputs, axis=1)


@layer_nd("Conv{n}d", arrays=conv_arrays, backends=NUMPY_ONLY)
def conv(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Conv1d`, `Conv2d` and `Conv3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `in_channels`, `out_channels`, `kernel_size`, `stride` (default
            `1`), `padding` (default `0`, an integer, one per axis, `valid` or
            `same`), `dilation` (default `1`), `groups` (default `1`), `bias`
            (default `True`), `padding_mode` (default `zeros`).
        arrays: `weight` of shape `(out_channels, in_channels / groups,
            *kernel_size)` and `bias` of shape `(out_channels,)`.
        x: input of shape `(N, C_in, *spatial)` or `(C_in, *spatial)`.

    Returns:
        The output of shape `(N, C_out, *output)` or `(C_out, *output)` with
        `output = (spatial + 2 * padding - dilation * (kernel_size - 1) - 1)
        // stride + 1`.

    Raises:
        ValueError: if the input does not have `in_channels` channels, the
            channels are not divisible by the groups, the padding is not known
            or, with the padding mode `zeros`, negative (the other modes crop),
            `same` is combined with a stride, the padding mode is not known or
            a `reflect` or `circular` padding does not fit the input.
    """
    weight = arrays["weight"]
    x, unbatched = add_batch(x, n, "Conv")
    check_channels(x, args, "Conv")
    groups = check_groups(args)
    stride = as_tuple(args.get("stride", 1), n)
    dilation = as_tuple(args.get("dilation", 1), n)
    padding = args.get("padding", 0)
    if padding == "valid":
        before = after = (0,) * n
    elif padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("Conv: padding 'same' requires a stride of 1")
        total = [d * (k - 1) for k, d in zip(weight.shape[2:], dilation, strict=True)]
        before = tuple(t // 2 for t in total)
        after = tuple(t - t // 2 for t in total)
    elif isinstance(padding, str):
        raise ValueError(
            f"Conv: padding '{padding}' is not known, use an integer, one per "
            "axis, 'valid' or 'same'"
        )
    else:
        before = after = as_tuple(padding, n)
    padding_mode = args.get("padding_mode", "zeros")
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"Conv: padding_mode '{padding_mode}' is not known")
    if padding_mode == "zeros":
        # the other modes crop a negative padding, as the transposed path does
        check_padding(before, "padding")
    check_padding_mode(x.shape[2:], before, after, padding_mode)
    x = pad_spatial(x, before, after, mode=PADDING_MODES[padding_mode])

    y = correlate(x, weight, stride, dilation, groups)
    y = add_bias(y, arrays)
    return y[0] if unbatched else y


@layer_nd("ConvTranspose{n}d", arrays=conv_transpose_arrays, backends=NUMPY_ONLY)
def conv_transpose(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `ConvTranspose1d`, `ConvTranspose2d` and `ConvTranspose3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `in_channels`, `out_channels`, `kernel_size`, `stride` (default
            `1`), `padding` (default `0`), `output_padding` (default `0`),
            `groups` (default `1`), `bias` (default `True`), `dilation`
            (default `1`), `padding_mode` (only `zeros`).
        arrays: `weight` of shape `(in_channels, out_channels / groups,
            *kernel_size)` and `bias` of shape `(out_channels,)`.
        x: input of shape `(N, C_in, *spatial)` or `(C_in, *spatial)`.

    Returns:
        The output of shape `(N, C_out, *output)` or `(C_out, *output)` with
        `output = (spatial - 1) * stride - 2 * padding + dilation *
        (kernel_size - 1) + output_padding + 1`.

    Raises:
        ValueError: if the input does not have `in_channels` channels, the
            channels are not divisible by the groups, the padding mode is not
            `zeros`, the padding is a string or negative, the output padding is
            negative or not smaller than either the stride or the dilation.
    """
    weight = arrays["weight"]
    x, unbatched = add_batch(x, n, "ConvTranspose")
    check_channels(x, args, "ConvTranspose")
    groups = check_groups(args)
    if args.get("padding_mode", "zeros") != "zeros":
        raise ValueError("ConvTranspose: only the padding_mode 'zeros' exists")
    stride = as_tuple(args.get("stride", 1), n)
    dilation = as_tuple(args.get("dilation", 1), n)
    if isinstance(args.get("padding", 0), str):
        raise ValueError(
            f"ConvTranspose: padding '{args['padding']}' is not known, use an "
            "integer or one per axis"
        )
    padding = as_tuple(args.get("padding", 0), n)
    output_padding = as_tuple(args.get("output_padding", 0), n)
    check_padding(padding, "padding")
    check_padding(output_padding, "output_padding")
    if any(
        o >= max(s, d) for o, s, d in zip(output_padding, stride, dilation, strict=True)
    ):
        raise ValueError(
            f"ConvTranspose: output_padding {list(output_padding)} must be smaller "
            f"than either stride {list(stride)} or dilation {list(dilation)}"
        )

    # the input with `stride - 1` zeros between its elements
    shape = (
        *x.shape[:2],
        *((s - 1) * st + 1 for s, st in zip(x.shape[2:], stride, strict=True)),
    )
    spread = np.zeros(shape, dtype=x.dtype)
    spread[(slice(None), slice(None), *(slice(None, None, st) for st in stride))] = x

    extent = [d * (k - 1) for k, d in zip(weight.shape[2:], dilation, strict=True)]
    before = [e - p for e, p in zip(extent, padding, strict=True)]
    after = [e - p + o for e, p, o in zip(extent, padding, output_padding, strict=True)]
    spread = pad_spatial(spread, before, after)

    # the kernel of the convolution: the channels of every group swapped and
    # the spatial axes flipped
    c_in_group = weight.shape[0] // groups
    flip = (slice(None), slice(None), *(slice(None, None, -1) for _ in range(n)))
    kernels = [
        np.swapaxes(weight[g * c_in_group : (g + 1) * c_in_group], 0, 1)[flip]
        for g in range(groups)
    ]
    kernel = np.concatenate(kernels, axis=0)

    y = correlate(spread, kernel, (1,) * n, dilation, groups)
    y = add_bias(y, arrays)
    return y[0] if unbatched else y
