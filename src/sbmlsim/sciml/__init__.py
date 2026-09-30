"""Neural networks of hybrid problems.

A hybrid problem combines a mechanistic model in SBML with neural networks,
which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
describes. This package is the native half of the support: `Network` is the
architecture and the arrays of one network and `Network.forward` evaluates it
with numpy; `nominal_parameters` and `network_fit_parameters` give the arrays
and the fit parameters a problem describes. The package knows nothing of
PEtab, the translation of a PEtab SciML problem will be part of
`sbmlsim.fit.petab_v2`.

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

from sbmlsim.sciml.errors import (
    NetworkError,
    NetworkImportError,
    UnsupportedLayerError,
)
from sbmlsim.sciml.network import Network, NetworkParameters
from sbmlsim.sciml.parameters import network_fit_parameters, nominal_parameters

__all__ = [
    "Network",
    "NetworkError",
    "NetworkImportError",
    "NetworkParameters",
    "UnsupportedLayerError",
    "network_fit_parameters",
    "nominal_parameters",
]
