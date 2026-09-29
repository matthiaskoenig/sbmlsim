"""The convolution and transposed convolution layers against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml import BackendKind, Network, NetworkImportError
from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

CONV_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    # the arguments as `petab_sciml` writes them, without a batch
    (
        "Conv1d",
        {
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [5],
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "groups": 1,
            "padding_mode": "zeros",
        },
        (1, 20),
    ),
    ("Conv1d", {"in_channels": 2, "out_channels": 3, "kernel_size": 3}, (4, 2, 11)),
    (
        "Conv1d",
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 2,
            "padding": 2,
            "dilation": 2,
            "groups": 2,
            "bias": False,
        },
        (3, 4, 17),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": [5, 2], "bias": True},
        (2, 9, 8),
    ),
    (
        "Conv2d",
        {
            "in_channels": 4,
            "out_channels": 4,
            "kernel_size": [3, 2],
            "stride": [2, 1],
            "padding": [1, 2],
            "dilation": [1, 2],
            "groups": 4,
        },
        (2, 4, 9, 8),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": "same"},
        (1, 2, 6, 7),
    ),
    (
        "Conv2d",
        {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": [4, 3],
            "padding": "same",
            "dilation": [1, 2],
        },
        (1, 2, 6, 7),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": "valid"},
        (1, 2, 6, 7),
    ),
    (
        "Conv3d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 4, 3]},
        (2, 7, 6, 5),
    ),
    (
        "Conv3d",
        {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "stride": [1, 2, 3],
            "padding": 1,
            "groups": 2,
        },
        (2, 2, 5, 6, 7),
    ),
]

PADDING_MODE_CASES = [
    (mode, padding)
    for mode in ("zeros", "reflect", "replicate", "circular")
    for padding in ([1, 2], 2)
]

CONV_TRANSPOSE_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "ConvTranspose1d",
        {
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [5],
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "groups": 1,
            "padding_mode": "zeros",
            "output_padding": [0],
        },
        (1, 20),
    ),
    (
        "ConvTranspose1d",
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 3,
            "padding": 2,
            "output_padding": 2,
            "dilation": 2,
            "groups": 2,
            "bias": False,
        },
        (3, 4, 7),
    ),
    (
        "ConvTranspose2d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 2]},
        (2, 6, 5),
    ),
    (
        "ConvTranspose2d",
        {
            "in_channels": 4,
            "out_channels": 2,
            "kernel_size": [3, 2],
            "stride": [2, 3],
            "padding": [1, 0],
            "output_padding": [1, 2],
            "dilation": [2, 1],
            "groups": 2,
        },
        (2, 4, 5, 4),
    ),
    (
        # the padding is larger than the extent of the kernel, the input is cropped
        "ConvTranspose2d",
        {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 2,
            "stride": 2,
            "padding": 3,
        },
        (1, 1, 6, 6),
    ),
    (
        "ConvTranspose3d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 4, 3]},
        (2, 4, 3, 2),
    ),
    (
        "ConvTranspose3d",
        {
            "in_channels": 2,
            "out_channels": 2,
            "kernel_size": 2,
            "stride": [1, 2, 3],
            "padding": [0, 1, 1],
            "output_padding": [0, 1, 2],
        },
        (2, 2, 3, 3, 3),
    ),
]


# PyTorch warns that `same` with an even kernel pads a copy of the input
@pytest.mark.filterwarnings("ignore:Using padding='same':UserWarning")
@pytest.mark.parametrize(("layer_type", "args", "shape"), CONV_CASES)
def test_conv(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A convolution has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("padding_mode", "padding"), PADDING_MODE_CASES)
def test_conv_padding_mode(
    compare_layer: Callable[..., None], padding_mode: str, padding: Any
) -> None:
    """The padding modes are the modes of `numpy.pad`."""
    args = {
        "in_channels": 2,
        "out_channels": 3,
        "kernel_size": 3,
        "padding": padding,
        "padding_mode": padding_mode,
    }
    compare_layer("Conv2d", args, (2, 2, 6, 7))


@pytest.mark.parametrize(("layer_type", "args", "shape"), CONV_TRANSPOSE_CASES)
def test_conv_transpose(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A transposed convolution has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


def test_the_shape_of_the_weight() -> None:
    """The weight of a transposed convolution has the input channels first."""
    args = {"in_channels": 4, "out_channels": 6, "kernel_size": [3, 2], "groups": 2}
    assert LAYERS["Conv2d"].arrays(args)["weight"].shape == (6, 2, 3, 2)
    assert LAYERS["ConvTranspose2d"].arrays(args)["weight"].shape == (4, 3, 3, 2)
    assert LAYERS["Conv2d"].arrays(args)["bias"].shape == (6,)
    assert "bias" not in LAYERS["Conv2d"].arrays({**args, "bias": False})


def test_the_convolutions_are_numpy_only() -> None:
    """A convolution is not part of a compiled network."""
    for n in (1, 2, 3):
        assert LAYERS[f"Conv{n}d"].backends == NUMPY_ONLY
        assert LAYERS[f"ConvTranspose{n}d"].backends == NUMPY_ONLY


def test_a_network_with_a_convolution_is_numpy_only(
    layer_model: Callable[..., NNModel],
) -> None:
    """The backends of a network are the backends of all its nodes."""
    model = layer_model(
        "Conv1d", {"in_channels": 1, "out_channels": 1, "kernel_size": 2}
    )
    assert Network(sid="net1", model=model).backends() == {BackendKind.NUMPY}


CONV2D = {"in_channels": 1, "out_channels": 1, "kernel_size": 3}
ARRAYS = {"layer1": {"weight": np.ones((1, 1, 3, 3)), "bias": np.ones(1)}}


def test_a_kernel_larger_than_the_input(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An input which is smaller than the kernel is an error, not an empty array."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("Conv2d", CONV2D), ARRAYS, np.ones((1, 1, 2, 5)))


def test_an_input_with_the_wrong_number_of_axes(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """The input of `Conv2d` has three or four axes."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*2 axes"):
        forward(layer_model("Conv2d", CONV2D), ARRAYS, np.ones((5, 5)))


def test_a_weight_of_the_wrong_shape(layer_model: Callable[..., NNModel]) -> None:
    """An array which does not fit the layer is an error of the import."""
    network = Network(sid="net1", model=layer_model("Conv2d", CONV2D))
    network.parameters = {"layer1": {"weight": np.ones((1, 1, 3, 2))}}
    with pytest.raises(NetworkImportError, match=r"'weight'.*\(1, 1, 3, 2\)"):
        network.forward(np.ones((1, 1, 5, 5)))


def test_an_unknown_padding_mode(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """A padding mode which does not exist is an error which names it."""
    model = layer_model("Conv2d", {**CONV2D, "padding_mode": "mirror"})
    with pytest.raises(ValueError, match=r"node 'layer1'.*'mirror'"):
        forward(model, ARRAYS, np.ones((1, 1, 5, 5)))


@pytest.mark.parametrize("layer_type", ["Conv2d", "ConvTranspose2d"])
@pytest.mark.parametrize("channels", [1, 3])
def test_the_channels_of_the_input(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    channels: int,
) -> None:
    """An input with other channels than `in_channels` is an error."""
    args = {"in_channels": 2, "out_channels": 1, "kernel_size": 3}
    specs = LAYERS[layer_type].arrays(args)
    arrays = {"layer1": {k: np.zeros(v.shape) for k, v in specs.items()}}
    with pytest.raises(ValueError, match=rf"node 'layer1'.*2 channels.*got {channels}"):
        forward(layer_model(layer_type, args), arrays, np.ones((1, channels, 5, 5)))
    forward(layer_model(layer_type, args), arrays, np.ones((1, 2, 5, 5)))


@pytest.mark.parametrize("layer_type", ["Conv1d", "Conv3d", "ConvTranspose2d"])
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "groups", "match"),
    [
        (5, 4, 2, "in_channels"),
        (4, 3, 2, "out_channels"),
        (4, 4, 0, "positive"),
    ],
)
def test_the_channels_must_be_divisible_by_groups(
    layer_type: str, in_channels: int, out_channels: int, groups: int, match: str
) -> None:
    """The channels are split into groups, torch refuses the layer otherwise."""
    args = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 2,
        "groups": groups,
    }
    with pytest.raises(ValueError, match=match):
        LAYERS[layer_type].arrays(args)


def test_the_limit_of_the_groups(compare_layer: Callable[..., None]) -> None:
    """One group per channel is the limit which torch accepts."""
    args = {"in_channels": 4, "out_channels": 4, "kernel_size": 2, "groups": 4}
    compare_layer("Conv1d", args, (4, 6))
    compare_layer("ConvTranspose1d", args, (4, 6))


@pytest.mark.parametrize("padding", [-1, [0, -1], [-1, 1]])
def test_a_negative_padding_of_a_convolution(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    padding: Any,
) -> None:
    """The padding mode `zeros` does not crop its input, as in torch."""
    args = {**CONV2D, "padding": padding}
    with pytest.raises(ValueError, match=r"node 'layer1'.*negative padding"):
        forward(layer_model("Conv2d", args), ARRAYS, np.ones((1, 1, 5, 5)))


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
@pytest.mark.parametrize("padding", [-1, [0, -1], [-1, 1], [1, -2]])
def test_a_negative_padding_crops_in_the_other_modes(
    compare_layer: Callable[..., None], padding_mode: str, padding: Any
) -> None:
    """The modes other than `zeros` crop a negative padding, as in torch."""
    args = {**CONV2D, "padding": padding, "padding_mode": padding_mode}
    compare_layer("Conv2d", args, (1, 6, 7))


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_a_negative_padding_which_empties_the_input(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    padding_mode: str,
) -> None:
    """A cropped input which is smaller than the kernel is an error."""
    args = {**CONV2D, "padding": -3, "padding_mode": padding_mode}
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("Conv2d", args), ARRAYS, np.ones((1, 1, 6, 6)))


@pytest.mark.parametrize(
    "args",
    [{"padding": -1}, {"output_padding": -1}, {"padding": [0, -1]}],
)
def test_a_negative_padding_of_a_transposed_convolution(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    args: dict[str, Any],
) -> None:
    """Torch rejects a negative padding and a negative output padding."""
    model = layer_model("ConvTranspose2d", {**CONV2D, **args})
    with pytest.raises(ValueError, match=r"node 'layer1'.*negative"):
        forward(model, ARRAYS, np.ones((1, 1, 5, 5)))


@pytest.mark.parametrize(
    ("stride", "dilation", "output_padding"),
    [(2, 1, 2), (2, 2, 2), (1, 1, 1), (3, 2, 3), ([2, 3], [1, 1], [1, 3])],
)
def test_the_output_padding_is_smaller_than_stride_or_dilation(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    stride: Any,
    dilation: Any,
    output_padding: Any,
) -> None:
    """The output padding must be smaller than the stride or the dilation."""
    args = {
        **CONV2D,
        "stride": stride,
        "dilation": dilation,
        "output_padding": output_padding,
    }
    with pytest.raises(ValueError, match=r"node 'layer1'.*output_padding"):
        forward(layer_model("ConvTranspose2d", args), ARRAYS, np.ones((1, 1, 5, 5)))


@pytest.mark.parametrize(
    ("stride", "dilation", "output_padding"),
    [(2, 1, 1), (1, 2, 1), (3, 2, 2), (2, 3, 2), ([2, 3], [1, 2], [1, 2])],
)
def test_the_limit_of_the_output_padding(
    compare_layer: Callable[..., None],
    stride: Any,
    dilation: Any,
    output_padding: Any,
) -> None:
    """The largest output padding which torch accepts."""
    args = {
        **CONV2D,
        "stride": stride,
        "dilation": dilation,
        "output_padding": output_padding,
    }
    compare_layer("ConvTranspose2d", args, (1, 4, 4))


@pytest.mark.parametrize(
    ("padding_mode", "padding", "size"),
    [
        ("reflect", 4, 4),
        ("reflect", 5, 4),
        ("reflect", [1, 4], 4),
        ("circular", 5, 4),
        ("circular", [1, 5], 4),
    ],
)
def test_a_padding_larger_than_the_input(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    padding_mode: str,
    padding: Any,
    size: int,
) -> None:
    """Reflect and circular padding wrap at most once, as in torch."""
    args = {**CONV2D, "padding": padding, "padding_mode": padding_mode}
    with pytest.raises(ValueError, match=rf"node 'layer1'.*{padding_mode}.*padding"):
        forward(layer_model("Conv2d", args), ARRAYS, np.ones((1, 1, size, size)))


@pytest.mark.parametrize(
    ("padding_mode", "padding"),
    [("reflect", 3), ("reflect", [3, 1]), ("circular", 4), ("replicate", 9)],
)
def test_the_limit_of_the_padding(
    compare_layer: Callable[..., None], padding_mode: str, padding: Any
) -> None:
    """The largest padding which torch accepts."""
    args = {**CONV2D, "padding": padding, "padding_mode": padding_mode}
    compare_layer("Conv2d", args, (1, 4, 4))


@pytest.mark.parametrize(
    ("layer_type", "padding"),
    [("Conv2d", "full"), ("ConvTranspose2d", "same"), ("ConvTranspose2d", "valid")],
)
def test_an_unknown_padding(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    padding: str,
) -> None:
    """A padding string which the layer does not have is an error which names it."""
    model = layer_model(layer_type, {**CONV2D, "padding": padding})
    with pytest.raises(ValueError, match=rf"node 'layer1'.*padding.*'{padding}'"):
        forward(model, ARRAYS, np.ones((1, 1, 5, 5)))
