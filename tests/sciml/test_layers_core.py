"""The layers of both backends against PyTorch."""

from collections.abc import Callable

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml.backend import ALL_BACKENDS
from sbmlsim.sciml.layers import LAYERS


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 4, 3)])
def test_linear(
    compare_layer: Callable[..., None], bias: bool, shape: tuple[int, ...]
) -> None:
    """`Linear` works on the last axis, with and without a batch."""
    compare_layer("Linear", {"in_features": 3, "out_features": 5, "bias": bias}, shape)


def test_linear_has_a_bias_by_default(compare_layer: Callable[..., None]) -> None:
    """`bias` is `True` when the YAML does not give it."""
    compare_layer("Linear", {"in_features": 3, "out_features": 5}, (3,))


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch", [(), (4,)])
def test_bilinear(
    compare_layer: Callable[..., None], bias: bool, batch: tuple[int, ...]
) -> None:
    """`Bilinear` takes two inputs."""
    args = {"in1_features": 3, "in2_features": 4, "out_features": 2, "bias": bias}
    compare_layer("Bilinear", args, (*batch, 3), (*batch, 4))


@pytest.mark.parametrize(
    ("args", "shape"),
    [
        ({}, (2, 3, 4)),
        ({"start_dim": 1, "end_dim": -1}, (2, 3, 4, 5)),
        ({"start_dim": 0, "end_dim": -1}, (2, 3, 4)),
        ({"start_dim": 1, "end_dim": 2}, (2, 3, 4, 5)),
        ({"start_dim": -2, "end_dim": -1}, (2, 3, 4)),
    ],
)
def test_flatten(
    compare_layer: Callable[..., None], args: dict, shape: tuple[int, ...]
) -> None:
    """`Flatten` flattens in row major order, from the axis 1 by default."""
    compare_layer("Flatten", args, shape)


@pytest.mark.parametrize(
    ("layer_type", "shape"),
    [
        ("Dropout", (5,)),
        ("AlphaDropout", (5,)),
        ("FeatureAlphaDropout", (2, 3, 4)),
        ("Dropout1d", (2, 3, 4)),
        ("Dropout2d", (2, 3, 4, 5)),
        ("Dropout3d", (2, 3, 4, 5, 6)),
    ],
)
def test_dropout_is_the_identity(
    compare_layer: Callable[..., None], layer_type: str, shape: tuple[int, ...]
) -> None:
    """A dropout layer in evaluation mode returns its input."""
    compare_layer(layer_type, {"p": 0.5, "inplace": False}, shape)


def test_the_layers_of_both_backends() -> None:
    """The layers a compiled network may use declare both backends."""
    for name in ("Linear", "Bilinear", "Flatten", "Dropout", "AlphaDropout"):
        assert LAYERS[name].backends == ALL_BACKENDS


def test_flatten_with_the_start_behind_the_end(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """A range of axes which is empty is an error which names the node."""
    model = layer_model("Flatten", {"start_dim": 2, "end_dim": 1})
    with pytest.raises(ValueError, match=r"node 'layer1'.*start_dim"):
        forward(model, {}, np.zeros((2, 3, 4)))


def test_a_scalar_is_flattened_to_one_element(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An array without axes becomes an array with one element, as in PyTorch."""
    model = layer_model("Flatten", {"start_dim": 0, "end_dim": -1})
    (y,) = forward(model, {}, np.array(2.0))
    assert y.shape == (1,)
