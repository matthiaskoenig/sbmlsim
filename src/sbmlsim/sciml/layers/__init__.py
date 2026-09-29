"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import (
    core,
    functions,
)
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "core",
    "functions",
]
