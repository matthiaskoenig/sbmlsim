"""Neural networks of hybrid problems.

A hybrid problem combines a mechanistic model in SBML with neural networks,
which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
describes. This package is the native half of the support: `Network` is the
architecture and the arrays of one network and `Network.forward` evaluates it
with numpy; `nominal_parameters` sets the arrays a problem describes;
`input_id`, `output_id`, `element_id` and `unit_id` build the ids of the
inputs, the outputs, the elements of the arrays and the units of a network in
a model. A `Hybridization` says where a network sits in a problem: before the
simulation, where a fit evaluates it, or in the right hand side or an
observable of the model, where `compile_network` writes it into the model;
`ALL_CONDITIONS` is the condition of an input which holds for every
condition. `Hybridization.fit_parameters` gives the parameters of a fit
defined in python and freezes the other elements; `network_fit_parameters` is
the primitive it and the reader share. The package knows nothing of PEtab,
the translation of a PEtab SciML problem is `sbmlsim.fit.petab_v2.sciml`.

The layers are implemented against the backends of `sbmlsim.sciml.backend`,
which are the extension point of the layers and not needed to use a network.

The architecture is read and written with `petab_sciml`, which is not a
dependency of `sbmlsim` but the `sciml` extra:

```bash
pip install sbmlsim[sciml]
```
"""

try:
    import petab_sciml  # noqa: F401
except ModuleNotFoundError as err:
    # a module which `petab_sciml` itself misses is not the missing extra
    if err.name != "petab_sciml":
        raise
    raise ImportError(
        "sbmlsim.sciml requires the package 'petab_sciml', which is installed "
        "with the extra 'sciml': pip install sbmlsim[sciml]"
    ) from err

from sbmlsim.sciml.compiler import compile_network, compiled_path
from sbmlsim.sciml.errors import (
    NetworkCompilationError,
    NetworkError,
    NetworkHybridizationError,
    NetworkImportError,
    UnsupportedLayerError,
)
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
)
from sbmlsim.sciml.network import (
    Network,
    NetworkParameters,
    element_id,
    input_id,
    output_id,
    parse_io_id,
    unit_id,
)
from sbmlsim.sciml.parameters import network_fit_parameters, nominal_parameters

__all__ = [
    "ALL_CONDITIONS",
    "Hybridization",
    "Network",
    "NetworkCompilationError",
    "NetworkError",
    "NetworkHybridizationError",
    "NetworkImportError",
    "NetworkInput",
    "NetworkParameters",
    "NetworkPattern",
    "UnsupportedLayerError",
    "compile_network",
    "compiled_path",
    "element_id",
    "input_id",
    "network_fit_parameters",
    "nominal_parameters",
    "output_id",
    "parse_io_id",
    "unit_id",
]
