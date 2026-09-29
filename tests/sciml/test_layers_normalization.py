"""The normalization layers against PyTorch, in evaluation mode."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml import Network
from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

#: the arguments as `petab_sciml` writes them
WRITTEN = {"momentum": 0.1, "eps": 1e-05, "bias": True}

STORED_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "BatchNorm1d",
        {"num_features": 3, "track_running_stats": True, "affine": True, **WRITTEN},
        (4, 3),
    ),
    ("BatchNorm1d", {"num_features": 3}, (4, 3, 7)),
    ("BatchNorm1d", {"num_features": 3, "affine": False, "eps": 0.1}, (4, 3, 7)),
    ("BatchNorm2d", {"num_features": 3}, (4, 3, 5, 6)),
    ("BatchNorm3d", {"num_features": 3}, (4, 3, 5, 6, 2)),
    ("InstanceNorm1d", {"num_features": 3, "track_running_stats": True}, (4, 3, 7)),
    (
        "InstanceNorm2d",
        {"num_features": 3, "track_running_stats": True, "affine": True},
        (4, 3, 5, 6),
    ),
    (
        "InstanceNorm3d",
        {"num_features": 3, "track_running_stats": True, "affine": True},
        (4, 3, 5, 6, 2),
    ),
]

CALCULATED_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    ("BatchNorm1d", {"num_features": 3, "track_running_stats": False}, (4, 3)),
    ("BatchNorm1d", {"num_features": 3, "track_running_stats": False}, (4, 3, 7)),
    (
        "BatchNorm2d",
        {"num_features": 3, "track_running_stats": False, "affine": False},
        (4, 3, 5, 6),
    ),
    (
        "BatchNorm3d",
        {"num_features": 3, "track_running_stats": False, "eps": 0.1},
        (4, 3, 5, 6, 2),
    ),
    (
        "InstanceNorm1d",
        {"num_features": 3, "track_running_stats": False, "affine": True, **WRITTEN},
        (4, 3, 7),
    ),
    ("InstanceNorm1d", {"num_features": 3}, (3, 7)),
    ("InstanceNorm2d", {"num_features": 3, "affine": True}, (4, 3, 5, 6)),
    ("InstanceNorm2d", {"num_features": 3}, (3, 5, 6)),
    ("InstanceNorm3d", {"num_features": 3, "eps": 0.1}, (4, 3, 5, 6, 2)),
    (
        "LayerNorm",
        {
            "normalized_shape": [4, 10, 11, 12],
            "eps": 1e-05,
            "elementwise_affine": True,
            "bias": True,
        },
        (2, 4, 10, 11, 12),
    ),
    ("LayerNorm", {"normalized_shape": [20]}, (3, 20)),
    ("LayerNorm", {"normalized_shape": 20}, (20,)),
    (
        "LayerNorm",
        {"normalized_shape": [5, 4], "elementwise_affine": False},
        (3, 2, 5, 4),
    ),
    ("LayerNorm", {"normalized_shape": [5, 4], "bias": False, "eps": 0.1}, (3, 5, 4)),
]


@pytest.mark.parametrize(("layer_type", "args", "shape"), STORED_CASES)
def test_the_stored_statistics_are_used(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A layer with running statistics normalizes with them."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("layer_type", "args", "shape"), CALCULATED_CASES)
def test_the_statistics_of_the_input_are_used(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A layer without running statistics calculates them from its input."""
    compare_layer(layer_type, args, shape)


def test_a_batch_norm_without_stored_statistics(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    rng: np.random.Generator,
) -> None:
    """The statistics of the batch are used when the arrays have none.

    This is the layer of the test suite: `track_running_stats` is true and
    the array file holds the weight and the bias only.
    """
    torch = pytest.importorskip("torch")
    args = {"num_features": 3, "track_running_stats": True, "affine": True, **WRITTEN}
    weight, bias = rng.normal(size=3), rng.normal(size=3)
    x = rng.normal(size=(4, 3, 5, 6))

    arrays = {"layer1": {"weight": weight, "bias": bias}}
    (observed,) = forward(layer_model("BatchNorm2d", args), arrays, x)

    module = torch.nn.BatchNorm2d(3, track_running_stats=False).double()
    module.load_state_dict(
        {"weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias)}
    )
    module.eval()
    with torch.no_grad():
        expected = module(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_the_arrays_of_a_normalization_layer(
    layer_model: Callable[..., NNModel],
) -> None:
    """The running statistics are arrays, but not required and not parameters."""
    arrays = LAYERS["BatchNorm2d"].arrays({"num_features": 3})
    assert list(arrays) == ["weight", "bias", "running_mean", "running_var"]
    assert arrays["weight"].required and arrays["weight"].trainable
    assert not arrays["running_mean"].required
    assert not arrays["running_var"].trainable

    assert list(LAYERS["InstanceNorm2d"].arrays({"num_features": 3})) == [
        "running_mean",
        "running_var",
    ]
    assert list(LAYERS["LayerNorm"].arrays({"normalized_shape": [2, 3]})) == [
        "weight",
        "bias",
    ]
    assert LAYERS["LayerNorm"].arrays({"normalized_shape": 4})["weight"].shape == (4,)

    network = Network(sid="net1", model=layer_model("BatchNorm2d", {"num_features": 2}))
    assert list(network.parameter_ids()) == [
        "net1__layer1__weight__0",
        "net1__layer1__weight__1",
        "net1__layer1__bias__0",
        "net1__layer1__bias__1",
    ]


def test_the_normalization_layers_are_numpy_only() -> None:
    """A normalization layer is not part of a compiled network."""
    for name in ("BatchNorm1d", "InstanceNorm2d", "LayerNorm"):
        assert LAYERS[name].backends == NUMPY_ONLY


def test_a_batch_norm_needs_a_batch(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`BatchNorm2d` has no input without a batch axis."""
    model = layer_model("BatchNorm2d", {"num_features": 3, "affine": False})
    with pytest.raises(ValueError, match=r"node 'layer1'.*3 axes"):
        forward(model, {}, np.ones((3, 4, 4)))


def test_a_layer_norm_on_the_wrong_shape(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """The last axes of the input are the normalized shape."""
    model = layer_model(
        "LayerNorm", {"normalized_shape": [4], "elementwise_affine": False}
    )
    with pytest.raises(ValueError, match=r"node 'layer1'.*normalized_shape"):
        forward(model, {}, np.ones((2, 5)))


def test_a_constant_input_is_normalized_to_zero(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`eps` keeps the division by a variance of zero finite."""
    model = layer_model("InstanceNorm1d", {"num_features": 2})
    (y,) = forward(model, {}, np.full((1, 2, 5), 3.0))
    np.testing.assert_array_equal(y, np.zeros((1, 2, 5)))
