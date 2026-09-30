"""The network of the neural ODE, defined without PyTorch.

The architecture is the one of the neural ODE how-to of PEtab SciML with
smaller layers: two hidden layers of `HIDDEN` units with `tanh`, from the two
species to their two rates. The arrays are drawn once from a seed; the small
weights of the last layer start the fit from a slow model.
"""

import numpy as np
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network

#: id of the network, the prefix of the ids of its elements
NETWORK_ID = "net1"

#: units of the hidden layers, 57 elements in all
HIDDEN = 5


def _linear(layer_id: str, n_in: int, n_out: int) -> Layer:
    return Layer(
        layer_id=layer_id,
        layer_type="Linear",
        args={"in_features": n_in, "out_features": n_out, "bias": True},
    )


def _node(name: str, op: str, target: str, args: list) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs={})


def build_network(seed: int = 1, hidden: int = HIDDEN) -> Network:
    """Build `layer3(tanh(layer2(tanh(layer1(x)))))` with random arrays.

    Args:
        seed: seed of the arrays.
        hidden: units of the two hidden layers.

    Returns:
        The network, with its nominal values.
    """
    rng = np.random.default_rng(seed)
    model = NNModel(
        nn_model_id=NETWORK_ID,
        inputs=[Input(input_id="input0")],
        layers=[
            _linear("layer1", 2, hidden),
            _linear("layer2", hidden, hidden),
            _linear("layer3", hidden, 2),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("tanh", "call_function", "tanh", ["layer1"]),
            _node("layer2", "call_module", "layer2", ["tanh"]),
            _node("tanh_1", "call_function", "tanh", ["layer2"]),
            _node("layer3", "call_module", "layer3", ["tanh_1"]),
            _node("output", "output", "output", ["layer3"]),
        ],
    )
    return Network(
        sid=NETWORK_ID,
        model=model,
        parameters={
            "layer1": {
                "weight": 0.5 * rng.normal(size=(hidden, 2)),
                "bias": 0.1 * rng.normal(size=hidden),
            },
            "layer2": {
                "weight": 0.5 * rng.normal(size=(hidden, hidden)),
                "bias": 0.1 * rng.normal(size=hidden),
            },
            "layer3": {
                "weight": 0.1 * rng.normal(size=(2, hidden)),
                "bias": 0.1 * rng.normal(size=2),
            },
        },
    )
