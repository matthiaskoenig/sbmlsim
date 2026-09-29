"""Neural networks of hybrid problems.

A hybrid problem combines a mechanistic model in SBML with neural networks,
which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
describes. This package is the native half of the support: `Network` is the
architecture and the arrays of one network and `Network.forward` evaluates it
with numpy. The package knows nothing of PEtab, `sbmlsim.fit.petab_v2`
translates.

The architecture is read and written with `petab_sciml`, which is not a
dependency of `sbmlsim` but the `sciml` extra:

```bash
pip install sbmlsim[sciml]
```
"""

try:
    import petab_sciml  # noqa: F401
except ModuleNotFoundError as err:
    raise ImportError(
        "sbmlsim.sciml requires the package 'petab_sciml', which is installed "
        "with the extra 'sciml': pip install sbmlsim[sciml]"
    ) from err

from sbmlsim.sciml.backend import Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError

__all__ = [
    "Backend",
    "BackendKind",
    "NetworkImportError",
    "NumpyBackend",
    "UnsupportedLayerError",
]
