"""The normalization layers against PyTorch, in evaluation mode."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from petab_sciml import NNModel, NNModelStandard

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
    assert list(arrays) == [
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]
    assert arrays["weight"].required and arrays["weight"].trainable
    assert not arrays["running_mean"].required
    assert not arrays["running_var"].trainable

    assert list(LAYERS["InstanceNorm2d"].arrays({"num_features": 3})) == [
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]
    assert arrays["num_batches_tracked"].shape == ()
    assert not arrays["num_batches_tracked"].required
    assert not arrays["num_batches_tracked"].trainable
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


def test_stored_statistics_accept_a_single_value_per_channel(
    compare_layer: Callable[..., None],
) -> None:
    """Stored statistics need no more than one value per channel, as in PyTorch."""
    compare_layer("BatchNorm1d", {"num_features": 3}, (1, 3))
    compare_layer("BatchNorm2d", {"num_features": 3}, (1, 3, 1, 1))
    compare_layer(
        "InstanceNorm1d", {"num_features": 3, "track_running_stats": True}, (2, 3, 1)
    )
    compare_layer(
        "InstanceNorm1d", {"num_features": 3, "track_running_stats": True}, (3, 1)
    )


@pytest.mark.parametrize(
    ("layer_type", "args", "shape"),
    [
        ("BatchNorm1d", {"num_features": 3}, (4, 1, 5)),
        ("BatchNorm1d", {"num_features": 3}, (4, 5, 5)),
        ("BatchNorm1d", {"num_features": 3, "affine": False}, (4, 1, 5)),
        ("BatchNorm2d", {"num_features": 3, "affine": True}, (4, 1, 5, 5)),
        ("InstanceNorm1d", {"num_features": 3, "affine": True}, (4, 1, 6)),
        ("InstanceNorm1d", {"num_features": 3}, (4, 5, 6)),
        ("InstanceNorm2d", {"num_features": 3}, (1, 5, 5)),
    ],
)
def test_a_wrong_number_of_channels_is_rejected(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    rng: np.random.Generator,
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """The channels of the input are `num_features`, whatever the arrays are."""
    arrays = {
        "weight": np.ones(3),
        "bias": np.zeros(3),
        "running_mean": np.zeros(3),
        "running_var": np.ones(3),
    }
    for stored in (arrays, {"weight": arrays["weight"], "bias": arrays["bias"]}, {}):
        parameters = {
            "layer1": {k: v for k, v in stored.items() if k in _spec(layer_type, args)}
        }
        with pytest.raises(ValueError, match=r"node 'layer1'.*num_features 3"):
            forward(layer_model(layer_type, args), parameters, rng.normal(size=shape))


def _spec(layer_type: str, args: dict[str, Any]) -> list[str]:
    """Get the names of the arrays of a layer."""
    return list(LAYERS[layer_type].arrays(args))


@pytest.mark.parametrize(
    ("layer_type", "args", "shape"),
    [
        ("BatchNorm1d", {"num_features": 3, "track_running_stats": False}, (1, 3)),
        ("BatchNorm1d", {"num_features": 3}, (1, 3, 1)),
        ("BatchNorm2d", {"num_features": 3}, (1, 3, 1, 1)),
        ("InstanceNorm1d", {"num_features": 3}, (2, 3, 1)),
        ("InstanceNorm1d", {"num_features": 3}, (3, 1)),
        ("InstanceNorm2d", {"num_features": 3}, (3, 1, 1)),
    ],
)
def test_statistics_of_a_single_value_are_rejected(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    rng: np.random.Generator,
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """Without stored statistics the input needs more than one value per channel."""
    parameters = (
        {"layer1": {"weight": np.ones(3), "bias": np.zeros(3)}}
        if layer_type.startswith("Batch")
        else {}
    )
    with pytest.raises(ValueError, match=r"node 'layer1'.*more than one value"):
        forward(layer_model(layer_type, args), parameters, rng.normal(size=shape))


@pytest.mark.parametrize("missing", ["running_mean", "running_var"])
def test_a_single_running_statistic_is_rejected(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    rng: np.random.Generator,
    missing: str,
) -> None:
    """The running statistics are both stored or both not."""
    stored = {"running_mean": np.zeros(3), "running_var": np.ones(3)}
    del stored[missing]
    model = layer_model("BatchNorm1d", {"num_features": 3, "affine": False})
    with pytest.raises(
        ValueError, match=r"node 'layer1'.*running_mean and running_var"
    ):
        forward(model, {"layer1": stored}, rng.normal(size=(4, 3, 5)))


def test_the_axes_of_a_batch_norm_1d_are_named(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`BatchNorm1d` takes 2 or 3 axes, and the message says so."""
    model = layer_model("BatchNorm1d", {"num_features": 3, "affine": False})
    with pytest.raises(ValueError, match=r"node 'layer1'.*2 or 3 axes"):
        forward(model, {}, np.ones(3))


@pytest.mark.parametrize(
    ("layer_type", "args", "shape"),
    [
        ("BatchNorm1d", {"num_features": 3}, (4, 3, 7)),
        (
            "InstanceNorm2d",
            {"num_features": 3, "track_running_stats": True},
            (4, 3, 5, 6),
        ),
    ],
)
def test_a_file_of_a_state_dict(
    tmp_path: Path,
    rng: np.random.Generator,
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """`num_batches_tracked` of a `state_dict` is an array, which is not used."""
    torch = pytest.importorskip("torch")
    module = getattr(torch.nn, layer_type)(**args).double()
    module.train()
    with torch.no_grad():
        for _ in range(3):
            module(torch.from_numpy(rng.normal(size=shape)))
    module.eval()
    NNModelStandard.save_data(
        data=NNModel.from_pytorch_module(
            torch.nn.Sequential(module), "net1", inputs=[]
        ),
        filename=str(tmp_path / "net1.yaml"),
    )
    with h5py.File(tmp_path / "net1_ps.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for key, tensor in module.state_dict().items():
            f[f"parameters/net1/0/{key}"] = tensor.numpy()

    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")
    tracked = module.state_dict()["num_batches_tracked"]
    assert network.parameters["0"]["num_batches_tracked"] == float(tracked)
    assert "net1__0__num_batches_tracked__" not in network.parameter_ids()

    x = rng.normal(size=shape)
    with torch.no_grad():
        expected = module(torch.from_numpy(x)).numpy()
    (observed,) = network.forward(x)
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
