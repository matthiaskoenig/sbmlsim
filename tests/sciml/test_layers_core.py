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


@pytest.mark.parametrize(
    ("args", "shape"),
    [
        ({"start_dim": 3, "end_dim": -1}, (2, 3, 4)),
        ({"start_dim": 0, "end_dim": 3}, (2, 3, 4)),
        ({"start_dim": -4, "end_dim": -1}, (2, 3, 4)),
        ({}, (3,)),
        ({"start_dim": 1, "end_dim": -1}, ()),
        ({"start_dim": 0, "end_dim": 1}, ()),
    ],
)
def test_flatten_with_an_axis_out_of_range(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    args: dict,
    shape: tuple[int, ...],
) -> None:
    """An axis which the input does not have is an error, as in PyTorch.

    The default `start_dim=1` on an input without the batch axis is the case
    which matters: it must not flatten from the axis 0 silently.
    """
    model = layer_model("Flatten", args)
    with pytest.raises(ValueError, match=r"node 'layer1'.*out of range"):
        forward(model, {}, np.zeros(shape))


@pytest.mark.parametrize(
    ("args", "shape", "expected"),
    [
        ({"start_dim": 0, "end_dim": 0}, (), (1,)),
        ({"start_dim": -1, "end_dim": -1}, (), (1,)),
        ({"start_dim": 0, "end_dim": -1}, (3,), (3,)),
        ({"start_dim": -1, "end_dim": -1}, (3,), (3,)),
    ],
)
def test_flatten_with_the_axes_of_torch_at_the_limit(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    args: dict,
    shape: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    """The axes at the limit of the range are valid, also for a scalar."""
    model = layer_model("Flatten", args)
    (y,) = forward(model, {}, np.zeros(shape))
    assert y.shape == expected


@pytest.mark.parametrize(
    ("layer_type", "args", "shapes", "message"),
    [
        (
            "Linear",
            {"in_features": 2, "out_features": 3},
            [(4, 3)],
            r"Linear: the input has 3 features on the last axis, expected "
            r"in_features 2",
        ),
        (
            "Linear",
            {"in_features": 1, "out_features": 3},
            [()],
            r"Linear: the input has no axes, expected in_features 1 on the last axis",
        ),
        (
            "Bilinear",
            {"in1_features": 2, "in2_features": 3, "out_features": 1},
            [(4, 3), (4, 3)],
            r"Bilinear: the input 1 has 3 features on the last axis, expected "
            r"in1_features 2",
        ),
        (
            "Bilinear",
            {"in1_features": 2, "in2_features": 3, "out_features": 1},
            [(4, 2), (4, 2)],
            r"Bilinear: the input 2 has 2 features on the last axis, expected "
            r"in2_features 3",
        ),
    ],
)
def test_an_input_with_the_wrong_features(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    args: dict,
    shapes: list[tuple[int, ...]],
    message: str,
) -> None:
    """The features of an input are checked against the layer, as in PyTorch."""
    arrays = {
        name: np.ones(spec.shape)
        for name, spec in LAYERS[layer_type].arrays(args).items()
    }
    model = layer_model(layer_type, args, n_inputs=len(shapes))
    with pytest.raises(ValueError, match=rf"node 'layer1': {message}"):
        forward(model, {"layer1": arrays}, *(np.ones(shape) for shape in shapes))
