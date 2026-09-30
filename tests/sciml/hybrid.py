"""A small hybrid problem for the tests: a model and its networks.

The model is the model of Lotka and Volterra of the PEtab SciML test suite:

    d prey / dt     = alpha * prey - beta * prey * predator
    d predator / dt = gamma * prey * predator - delta * predator

A network replaces `gamma` or sets it before the simulation.
"""

from pathlib import Path

import libsbml
import numpy as np
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network

#: the model, with the species `prey` and `predator` and the parameters
#: `alpha`, `beta`, `gamma` and `delta`
MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "lotka_volterra.xml"


def write_model(path: Path, level: int = 3, version: int = 1) -> Path:
    """Write a copy of the model of Lotka and Volterra.

    Args:
        path: the file the model is written to.
        level: the level of SBML.
        version: the version of SBML.

    Returns:
        The path.
    """
    document = libsbml.readSBMLFromFile(str(MODEL_PATH))
    if (level, version) != (document.getLevel(), document.getVersion()):
        assert document.setLevelAndVersion(level, version)
    libsbml.writeSBMLToFile(document, str(path))
    return path


def nominal_values(network: Network) -> dict[str, float]:
    """Get the nominal value of every element of the arrays of a network.

    Args:
        network: the network, with the values of its arrays.

    Returns:
        id of the element -> value, the values a fit starts from.
    """
    return {
        sid: float(network.parameters[layer][name][index])
        for sid, (layer, name, index) in network.parameter_ids().items()
    }


def _node(
    name: str, op: str, target: str, args: list, kwargs: dict | None = None
) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs=kwargs or {})


def feed_forward(
    sid: str = "net1",
    n_inputs: int = 2,
    n_hidden: int = 3,
    n_outputs: int = 1,
    activation: str = "tanh",
    kwargs: dict | None = None,
    seed: int = 1,
) -> Network:
    """Build `layer2(activation(layer1(x)))` with random arrays.

    Args:
        sid: id of the network.
        n_inputs: number of inputs.
        n_hidden: number of units of the hidden layer.
        n_outputs: number of outputs.
        activation: the function between the layers.
        kwargs: the keyword arguments of the function.
        seed: seed of the arrays.

    Returns:
        The network.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Linear",
                args={"in_features": n_inputs, "out_features": n_hidden, "bias": True},
            ),
            Layer(
                layer_id="layer2",
                layer_type="Linear",
                args={"in_features": n_hidden, "out_features": n_outputs, "bias": True},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("act", "call_function", activation, ["layer1"], kwargs),
            _node("layer2", "call_module", "layer2", ["act"]),
            _node("output", "output", "output", ["layer2"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            "layer1": {
                "weight": rng.normal(size=(n_hidden, n_inputs)),
                "bias": rng.normal(size=n_hidden),
            },
            "layer2": {
                "weight": rng.normal(size=(n_outputs, n_hidden)),
                "bias": rng.normal(size=n_outputs),
            },
        },
    )


def two_inputs(sid: str = "net6", seed: int = 2) -> Network:
    """Build `layer1(cat(x0, x1))` with an input of one and one of three elements.

    Args:
        sid: id of the network.
        seed: seed of the arrays.

    Returns:
        The network.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0"), Input(input_id="input1")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Linear",
                args={"in_features": 4, "out_features": 1, "bias": True},
            )
        ],
        forward=[
            _node("x0", "placeholder", "x0", []),
            _node("x1", "placeholder", "x1", []),
            _node("cat", "call_function", "cat", [["x0", "x1"]], {"dim": 0}),
            _node("layer1", "call_module", "layer1", ["cat"]),
            _node("output", "output", "output", ["layer1"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            # small weights, the model with the network stays a tame ODE
            "layer1": {
                "weight": 0.2 * rng.normal(size=(1, 4)),
                "bias": rng.normal(size=1),
            }
        },
    )


def convolution(sid: str = "net3", seed: int = 3) -> Network:
    """Build `layer2(flatten(layer1(x)))` with a convolution, numpy only.

    Args:
        sid: id of the network.
        seed: seed of the arrays.

    Returns:
        The network, which takes an input of the shape `(1, 4, 4)`.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type="Conv2d",
                args={"in_channels": 1, "out_channels": 1, "kernel_size": 3},
            ),
            Layer(layer_id="layer2", layer_type="Flatten", args={"start_dim": 0}),
            Layer(
                layer_id="layer3",
                layer_type="Linear",
                args={"in_features": 4, "out_features": 1, "bias": True},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("layer2", "call_module", "layer2", ["layer1"]),
            _node("layer3", "call_module", "layer3", ["layer2"]),
            _node("output", "output", "output", ["layer3"]),
        ],
    )
    return Network(
        sid=sid,
        model=model,
        parameters={
            "layer1": {
                "weight": rng.normal(size=(1, 1, 3, 3)),
                "bias": rng.normal(size=1),
            },
            "layer3": {"weight": rng.normal(size=(1, 4)), "bias": rng.normal(size=1)},
        },
    )
