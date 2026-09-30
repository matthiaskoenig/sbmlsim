"""The pooling layers against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

POOL_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    # the arguments as `petab_sciml` writes them, without a batch
    (
        "MaxPool3d",
        {
            "kernel_size": [3, 2, 1],
            "stride": [3, 2, 1],
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        },
        (2, 7, 6, 5),
    ),
    ("MaxPool1d", {"kernel_size": 3}, (2, 3, 11)),
    ("MaxPool1d", {"kernel_size": 3, "stride": 2, "padding": 1}, (3, 11)),
    (
        "MaxPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2, "ceil_mode": True},
        (2, 3, 12),
    ),
    ("MaxPool2d", {"kernel_size": [2, 2], "stride": [2, 2]}, (1, 6, 8, 8)),
    (
        "MaxPool2d",
        {"kernel_size": [3, 2], "stride": [2, 3], "padding": [1, 1], "ceil_mode": True},
        (2, 3, 10, 9),
    ),
    # the last window of the ceil mode starts in the padding and is dropped
    (
        "MaxPool2d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (1, 1, 5, 5),
    ),
    (
        "MaxPool3d",
        {"kernel_size": 2, "stride": [1, 2, 3], "padding": 1},
        (2, 2, 5, 6, 7),
    ),
    (
        "AvgPool3d",
        {
            "kernel_size": [3, 2, 1],
            "stride": [3, 2, 1],
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        (2, 7, 6, 5),
    ),
    ("AvgPool1d", {"kernel_size": 3}, (2, 3, 11)),
    ("AvgPool1d", {"kernel_size": 3, "stride": 2, "padding": 1}, (3, 11)),
    (
        "AvgPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": False},
        (2, 3, 11),
    ),
    (
        "AvgPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 3, 12),
    ),
    (
        "AvgPool2d",
        {
            "kernel_size": [3, 2],
            "stride": [2, 3],
            "padding": [1, 1],
            "ceil_mode": True,
            "count_include_pad": False,
        },
        (2, 3, 10, 9),
    ),
    (
        "AvgPool2d",
        {"kernel_size": [3, 2], "stride": [2, 3], "padding": [1, 1], "ceil_mode": True},
        (2, 3, 10, 9),
    ),
    (
        "AvgPool2d",
        {"kernel_size": 3, "padding": 1, "divisor_override": 4},
        (2, 3, 9, 9),
    ),
    (
        "AvgPool3d",
        {"kernel_size": 2, "stride": [1, 2, 3], "padding": 1},
        (2, 2, 5, 6, 7),
    ),
    # the drop rule of the ceil mode with a padding which counts
    (
        "AvgPool1d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 3, 5),
    ),
    (
        "AvgPool1d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 3, 1),
    ),
    (
        "AvgPool2d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (1, 1, 5, 5),
    ),
    (
        "AvgPool2d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 3, 1, 1),
    ),
    (
        "AvgPool3d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 2, 5, 3, 3),
    ),
    (
        "AdaptiveMaxPool3d",
        {"output_size": [3, 2, 1], "return_indices": False},
        (2, 7, 6, 5),
    ),
    ("AdaptiveMaxPool1d", {"output_size": 4}, (2, 3, 11)),
    ("AdaptiveMaxPool2d", {"output_size": [3, None]}, (2, 3, 10, 7)),
    ("AdaptiveMaxPool2d", {"output_size": 5}, (3, 10, 7)),
    ("AdaptiveAvgPool3d", {"output_size": [3, 2, 1]}, (2, 7, 6, 5)),
    ("AdaptiveAvgPool1d", {"output_size": 4}, (2, 3, 11)),
    ("AdaptiveAvgPool2d", {"output_size": [3, None]}, (2, 3, 10, 7)),
    ("AdaptiveAvgPool2d", {"output_size": 5}, (3, 10, 7)),
    # the padding at the limit of half the kernel size, the limit ignores the dilation
    ("MaxPool1d", {"kernel_size": 2, "padding": 1}, (2, 3, 5)),
    ("MaxPool1d", {"kernel_size": 3, "padding": 1, "dilation": 2}, (2, 3, 5)),
    ("AvgPool1d", {"kernel_size": 3, "padding": 1}, (2, 3, 5)),
    ("AvgPool2d", {"kernel_size": [3, 2], "padding": [1, 1]}, (2, 3, 5, 5)),
    ("MaxPool3d", {"kernel_size": 2, "padding": 1}, (1, 2, 4, 4, 4)),
    # more outputs than inputs, the windows overlap
    ("AdaptiveAvgPool1d", {"output_size": 7}, (2, 3, 4)),
]

LP_POOL_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "LPPool3d",
        {"norm_type": 2, "kernel_size": [3, 2, 1], "stride": None, "ceil_mode": False},
        (2, 7, 6, 5),
    ),
    ("LPPool1d", {"norm_type": 1, "kernel_size": 3, "stride": 2}, (2, 3, 11)),
    ("LPPool1d", {"norm_type": -2, "kernel_size": 3}, (2, 3, 11)),
    ("LPPool1d", {"norm_type": float("inf"), "kernel_size": 2}, (2, 3, 11)),
    ("LPPool1d", {"norm_type": -float("inf"), "kernel_size": 2}, (2, 3, 11)),
    (
        "LPPool1d",
        {"norm_type": float("inf"), "kernel_size": 3, "ceil_mode": True},
        (2, 3, 11),
    ),
    (
        "LPPool2d",
        {"norm_type": float("inf"), "kernel_size": [2, 3], "stride": [1, 2]},
        (2, 3, 6, 8),
    ),
    (
        "LPPool3d",
        {"norm_type": -float("inf"), "kernel_size": 2, "stride": 1},
        (1, 2, 4, 4, 4),
    ),
    ("LPPool1d", {"norm_type": 3, "kernel_size": 3, "ceil_mode": True}, (2, 3, 11)),
    ("LPPool2d", {"norm_type": 2, "kernel_size": [3, 2], "stride": [2, 1]}, (3, 9, 8)),
    (
        "LPPool2d",
        {"norm_type": 1.5, "kernel_size": 2, "stride": 3, "ceil_mode": True},
        (2, 3, 9, 8),
    ),
]


@pytest.mark.parametrize(("layer_type", "args", "shape"), POOL_CASES)
def test_pool(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A pooling layer has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("layer_type", "args", "shape"), LP_POOL_CASES)
def test_lp_pool(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """The p-norm of the windows has the values of PyTorch.

    The input is positive, the power of a negative number with a `norm_type`
    which is not an integer is not defined.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=3)
    x = rng.uniform(0.1, 2.0, size=shape)
    with torch.no_grad():
        expected = getattr(torch.nn, layer_type)(**args)(torch.from_numpy(x)).numpy()
    (observed,) = forward(layer_model(layer_type, args), {}, x)
    assert observed.shape == expected.shape
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_the_pooling_layers_are_numpy_only() -> None:
    """A pooling layer is not part of a compiled network."""
    names = ["MaxPool", "AvgPool", "LPPool", "AdaptiveMaxPool", "AdaptiveAvgPool"]
    for name in names:
        for n in (1, 2, 3):
            layer_type = LAYERS[f"{name}{n}d"]
            assert layer_type.backends == NUMPY_ONLY
            assert layer_type.arrays({"kernel_size": 2}) == {}


@pytest.mark.parametrize("layer_type", ["MaxPool2d", "AdaptiveMaxPool2d"])
def test_the_indices_are_not_returned(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
) -> None:
    """`return_indices` changes the output of a layer and is an error."""
    args = {"kernel_size": 2, "output_size": 2, "return_indices": True}
    with pytest.raises(ValueError, match=r"node 'layer1'.*return_indices"):
        forward(layer_model(layer_type, args), {}, np.ones((1, 4, 4)))


def test_a_window_larger_than_the_input(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An input which is smaller than the kernel is an error, not an empty array."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("MaxPool2d", {"kernel_size": 3}), {}, np.ones((1, 2, 5)))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("layer_type", ["MaxPool", "AvgPool"])
def test_a_padding_above_half_the_kernel_is_an_error(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    n: int,
) -> None:
    """PyTorch refuses a padding of more than half the kernel size."""
    args = {"kernel_size": 2, "padding": [1] * (n - 1) + [2]}
    with pytest.raises(ValueError, match=r"node 'layer1'.*padding.*kernel_size"):
        forward(layer_model(f"{layer_type}{n}d", args), {}, np.ones((1, *(6,) * n)))


def test_a_padding_without_a_kernel_is_an_error(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """A kernel of the size 1 has no padding, the padding is never a NaN."""
    args = {"kernel_size": 1, "padding": 1, "count_include_pad": False}
    with pytest.raises(ValueError, match=r"node 'layer1'.*padding"):
        forward(layer_model("AvgPool1d", args), {}, np.ones((1, 5)))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("layer_type", ["MaxPool", "AvgPool", "LPPool"])
@pytest.mark.parametrize("name", ["kernel_size", "stride"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_kernel_and_a_stride_are_positive(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    name: str,
    n: int,
    value: int,
) -> None:
    """PyTorch refuses a kernel size and a stride below 1."""
    args = {"norm_type": 2, "kernel_size": 2, name: [2] * (n - 1) + [value]}
    with pytest.raises(ValueError, match=rf"node 'layer1'.*{name}"):
        forward(layer_model(f"{layer_type}{n}d", args), {}, np.ones((1, *(5,) * n)))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("value", [0, -1])
def test_a_dilation_is_positive(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    n: int,
    value: int,
) -> None:
    """PyTorch refuses a dilation below 1."""
    args = {"kernel_size": 2, "dilation": [1] * (n - 1) + [value]}
    with pytest.raises(ValueError, match=r"node 'layer1': MaxPool: dilation must"):
        forward(layer_model(f"MaxPool{n}d", args), {}, np.ones((1, *(5,) * n)))


@pytest.mark.parametrize("n", [2, 3])
def test_a_divisor_of_zero_is_an_error(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    n: int,
) -> None:
    """`divisor_override` of zero is refused by PyTorch."""
    args = {"kernel_size": 2, "divisor_override": 0}
    with pytest.raises(ValueError, match=r"node 'layer1'.*divisor_override"):
        forward(layer_model(f"AvgPool{n}d", args), {}, np.ones((1, *(4,) * n)))


def test_avg_pool1d_has_no_divisor_override(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`AvgPool1d` of PyTorch has no `divisor_override`."""
    args = {"kernel_size": 2, "divisor_override": 2}
    with pytest.raises(ValueError, match=r"node 'layer1'.*divisor_override"):
        forward(layer_model("AvgPool1d", args), {}, np.ones((1, 4)))


def test_a_norm_type_of_zero_is_an_error(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """PyTorch refuses `norm_type` 0."""
    args = {"norm_type": 0, "kernel_size": 2}
    with pytest.raises(ValueError, match=r"node 'layer1'.*norm_type"):
        forward(layer_model("LPPool1d", args), {}, np.ones((1, 4)))


@pytest.mark.parametrize("layer_type", ["AdaptiveMaxPool2d", "AdaptiveAvgPool2d"])
@pytest.mark.parametrize("output_size", [[3], [2, 2, 2]])
def test_the_output_size_has_one_entry_per_axis(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    output_size: list[int],
) -> None:
    """PyTorch refuses an `output_size` of a length other than the dimensions."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*output_size"):
        forward(
            layer_model(layer_type, {"output_size": output_size}),
            {},
            np.ones((1, 4, 4)),
        )


@pytest.mark.parametrize("layer_type", ["AdaptiveMaxPool1d", "AdaptiveAvgPool1d"])
@pytest.mark.parametrize("output_size", [0, -1])
def test_the_output_size_is_positive(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    output_size: int,
) -> None:
    """An empty output is not supported, a negative size is refused by PyTorch."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*output_size"):
        forward(
            layer_model(layer_type, {"output_size": output_size}), {}, np.ones((1, 5))
        )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("name", ["AdaptiveMaxPool", "AdaptiveAvgPool"])
def test_an_output_size_of_none(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    name: str,
    n: int,
) -> None:
    """PyTorch refuses `None` as the output size, and as an entry for one axis."""
    x = np.ones((1, *(4,) * n))
    for output_size in (None, [None] * n):
        if n > 1 and output_size is not None:
            # an entry `None` keeps the size of the axis
            (y,) = forward(
                layer_model(f"{name}{n}d", {"output_size": output_size}), {}, x
            )
            assert y.shape == x.shape
            continue
        with pytest.raises(ValueError, match=r"node 'layer1'.*output_size"):
            forward(layer_model(f"{name}{n}d", {"output_size": output_size}), {}, x)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_an_avg_pool_3d_smaller_than_the_kernel(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    axis: int,
) -> None:
    """`AvgPool3d` refuses an input smaller than the kernel, the padding aside."""
    spatial = [6, 6, 6]
    spatial[axis] = 2
    x = np.ones((1, 2, *spatial))
    args = {"kernel_size": 3, "stride": 1, "padding": 1}
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match=r"smaller than kernel size"):
        torch.nn.AvgPool3d(**args)(torch.from_numpy(x))
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("AvgPool3d", args), {}, x)
    # the other pools of PyTorch compute it
    torch.nn.MaxPool3d(**args)(torch.from_numpy(x))
    forward(layer_model("MaxPool3d", args), {}, x)
    forward(layer_model("AvgPool2d", args), {}, x[0])


@pytest.mark.filterwarnings("error")
def test_an_lp_pool_of_a_negative_sum(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An odd norm of a window with a negative sum is `nan`, without a warning."""
    torch = pytest.importorskip("torch")
    x = np.array([[[-1.0, -2.0, 0.5, 3.0]]])
    args = {"norm_type": 3, "kernel_size": 2}
    with torch.no_grad():
        expected = torch.nn.LPPool1d(**args)(torch.from_numpy(x)).numpy()
    (observed,) = forward(layer_model("LPPool1d", args), {}, x)
    assert np.isnan(expected[0, 0, 0])
    np.testing.assert_allclose(observed, expected, rtol=1e-10, equal_nan=True)
